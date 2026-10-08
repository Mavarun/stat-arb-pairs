"""Causal and cost-aware tests for per-window universe re-screening."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statarb.rescreen import compare_rules, universe_formation_backtest
from statarb.synth import make_ou_pair, make_planted_universe, make_random_walk_universe


def _short_planted(seed: int = 1):
    # Smaller windows so unit tests stay fast; still enough for EG + BH.
    return make_planted_universe(6, 2, 120 + 30 * 3, half_life=6.0, seed=seed)


def test_window_count_and_no_trading_inside_first_formation():
    frame, _ = _short_planted(2)
    res = universe_formation_backtest(frame, rule="bh", formation=120, trading=30)
    assert res.n_windows == 3
    assert res.returns.index[0] == frame.index[120]
    assert len(res.returns) == 30 * 3


def test_pure_noise_bh_trades_far_less_than_naive():
    # BH should keep most formation windows flat on independent random walks.
    frame = make_random_walk_universe(10, 120 + 30 * 4, seed=11)
    bh = universe_formation_backtest(frame, rule="bh", formation=120, trading=30)
    naive = universe_formation_backtest(frame, rule="naive", formation=120, trading=30)
    assert bh.mean_discoveries < naive.mean_discoveries
    assert bh.frac_windows_traded <= naive.frac_windows_traded
    # Strict: BH should trade in a minority of windows on pure noise.
    assert bh.frac_windows_traded <= 0.5


def test_planted_pairs_are_discovered_and_traded_under_bh():
    # Longer formation + shorter half-life so EG has power; BH still controls FDR.
    frame, truth = make_planted_universe(8, 3, 200 + 40 * 3, half_life=4.0, seed=4)
    res = universe_formation_backtest(frame, rule="bh", formation=200, trading=40)
    naive = universe_formation_backtest(frame, rule="naive", formation=200, trading=40)
    assert res.mean_discoveries > 0.0
    assert res.frac_windows_traded > 0.0
    assert res.trades > 0
    # BH should not discover *more* than the uncorrected rule on the same data.
    assert res.mean_discoveries <= naive.mean_discoveries + 1e-12
    assert len(truth) == 3


def test_backtest_is_causal_future_break_does_not_change_earlier_blocks():
    frame, _ = make_planted_universe(6, 2, 120 + 30 * 4, half_life=6.0, seed=5)
    broken = frame.copy()
    # Break only inside the last trading block.
    cut_i = 120 + 30 * 3
    broken.iloc[cut_i:] *= 1.15
    for rule in ("bh", "naive"):
        a = universe_formation_backtest(frame, rule=rule, formation=120, trading=30)
        b = universe_formation_backtest(broken, rule=rule, formation=120, trading=30)
        cut = frame.index[cut_i - 1]
        pd.testing.assert_series_equal(a.returns.loc[:cut], b.returns.loc[:cut])
        # Chosen discovery counts for earlier windows must match too.
        pd.testing.assert_series_equal(
            a.windows.loc[:2, "discoveries"].reset_index(drop=True),
            b.windows.loc[:2, "discoveries"].reset_index(drop=True),
        )


def test_costs_monotone_and_zero_cost_matches_path_when_discoveries_exist():
    frame, _ = make_planted_universe(6, 2, 120 + 30 * 4, half_life=5.0, seed=6)
    free = universe_formation_backtest(
        frame, rule="bh", formation=120, trading=30, spread_bps=0, slippage_bps=0
    )
    paid = universe_formation_backtest(
        frame, rule="bh", formation=120, trading=30, spread_bps=5, slippage_bps=5
    )
    dear = universe_formation_backtest(
        frame, rule="bh", formation=120, trading=30, spread_bps=20, slippage_bps=20
    )
    assert free.cost_paid == 0.0
    assert free.trades > 0
    assert free.total_return >= paid.total_return >= dear.total_return - 1e-12


def test_flat_book_when_rule_finds_nothing():
    # One short window of independent walks: BH discoveries should be ~0.
    frame = make_random_walk_universe(6, 100 + 25, seed=99)
    res = universe_formation_backtest(frame, rule="bh", formation=100, trading=25)
    assert res.n_windows == 1
    if res.mean_discoveries == 0.0:
        assert (res.returns == 0.0).all()
        assert res.trades == 0
        assert res.cost_paid == 0.0


def test_compare_rules_returns_expected_index():
    frame, _ = _short_planted(7)
    table = compare_rules(frame, rules=("naive", "bh"), formation=120, trading=30)
    assert list(table.index) == ["naive", "bh"]
    assert "mean_discoveries" in table.columns


def test_input_validation():
    frame = make_random_walk_universe(4, 200, seed=1)
    with pytest.raises(ValueError):
        universe_formation_backtest(frame, rule="not_a_rule", formation=100, trading=30)
    with pytest.raises(ValueError):
        universe_formation_backtest(frame, hedge="lasso", formation=100, trading=30)
    with pytest.raises(ValueError):
        universe_formation_backtest(frame, formation=10, trading=30)
    with pytest.raises(ValueError):
        universe_formation_backtest(frame)  # too short for default 252+63
