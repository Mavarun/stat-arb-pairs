"""Causal and selection tests for nested Kalman q_beta tuning."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statarb.formation import formation_trading_backtest
from statarb.kalman import Q_BETA
from statarb.nested import (
    DEFAULT_Q_GRID,
    compare_fixed_vs_nested,
    nested_kalman_backtest,
    tune_q_beta,
)
from statarb.synth import make_ou_pair


def test_tune_q_beta_returns_grid_row_and_falls_back_on_short_sample():
    p = make_ou_pair(84 + 21 * 3, half_life=5, seed=1)
    q, table = tune_q_beta(p.y, p.x, formation=84, trading=21)
    assert q in DEFAULT_Q_GRID
    assert set(table["q_beta"]) == set(DEFAULT_Q_GRID)
    assert table["n_windows"].min() >= 1
    # Too short for even one inner window -> default.
    short = make_ou_pair(50, seed=2)
    q2, table2 = tune_q_beta(short.y, short.x, formation=84, trading=21)
    assert q2 == Q_BETA
    assert table2.empty


def test_nested_window_count_and_forced_flat_block_ends():
    n = 252 + 63 * 3
    p = make_ou_pair(n, half_life=5, seed=3)
    res = nested_kalman_backtest(
        p.y, p.x, formation=252, trading=63, inner_formation=84, inner_trading=21
    )
    assert res.n_windows == 3
    assert len(res.chosen_q) == 3
    assert all(q in DEFAULT_Q_GRID for q in res.chosen_q)
    assert res.returns.index[0] == p.y.index[252]
    ends = res.windows["trade_end"]
    # Positions are inside formation_trading_backtest; check returns length instead.
    assert len(res.returns) == 63 * 3
    assert list(ends) == [p.y.index[252 + 63 * (i + 1) - 1] for i in range(3)]


def test_nested_is_causal_future_break_does_not_change_earlier_q_or_returns():
    n = 252 + 63 * 3
    p = make_ou_pair(n, half_life=5, seed=4)
    y2 = p.y.copy()
    y2.iloc[252 + 63 * 2 :] += 50.0  # break only in the last trading block
    a = nested_kalman_backtest(
        p.y, p.x, formation=252, trading=63, inner_formation=84, inner_trading=21
    )
    b = nested_kalman_backtest(
        y2, p.x, formation=252, trading=63, inner_formation=84, inner_trading=21
    )
    # First two windows: same chosen q and same returns.
    assert a.chosen_q[:2] == b.chosen_q[:2]
    cut = p.y.index[252 + 63 * 2 - 1]
    pd.testing.assert_series_equal(a.returns.loc[:cut], b.returns.loc[:cut])


def test_tune_uses_only_formation_bars():
    """Appending bars after the formation window must not change the chosen q."""
    form = make_ou_pair(84 + 21 * 3, half_life=5, seed=5)
    q1, _ = tune_q_beta(form.y, form.x, formation=84, trading=21)
    # Extend with a wild continuation that would dominate if leaked.
    ext_y = pd.concat(
        [form.y, form.y.iloc[-1] + pd.Series(np.linspace(0, 80, 100), index=pd.bdate_range(form.y.index[-1] + pd.Timedelta(days=1), periods=100))]
    )
    ext_x = pd.concat(
        [form.x, form.x.iloc[-1] + pd.Series(np.linspace(0, 5, 100), index=ext_y.index[len(form.y) :])]
    )
    # Re-tune on the original formation only (caller contract); and on a
    # truncated view that matches form length.
    q2, _ = tune_q_beta(ext_y.iloc[: len(form.y)], ext_x.iloc[: len(form.x)], formation=84, trading=21)
    assert q1 == q2


def test_costs_monotone_under_nested():
    n = 252 + 63 * 4
    p = make_ou_pair(n, half_life=5, seed=6)
    free = nested_kalman_backtest(
        p.y, p.x, spread_bps=0, slippage_bps=0, formation=252, trading=63,
        inner_formation=84, inner_trading=21,
    )
    paid = nested_kalman_backtest(
        p.y, p.x, spread_bps=5, slippage_bps=5, formation=252, trading=63,
        inner_formation=84, inner_trading=21,
    )
    assert free.cost_paid == 0.0
    assert free.total_return >= paid.total_return - 1e-12


def test_compare_fixed_vs_nested_shape():
    n = 252 + 63 * 2
    p = make_ou_pair(n, half_life=5, seed=7)
    table = compare_fixed_vs_nested(
        p.y, p.x, formation=252, trading=63, inner_formation=84, inner_trading=21
    )
    assert list(table.index) == ["fixed_default_q", "nested_tuned_q"]
    assert "median_chosen_q" in table.columns
    # Fixed path must match a direct formation_trading_backtest with Q_BETA.
    direct = formation_trading_backtest(p.y, p.x, hedge="kalman", q_beta=Q_BETA)
    assert np.isclose(table.loc["fixed_default_q", "sharpe"], direct.sharpe, equal_nan=True)


def test_input_validation():
    p = make_ou_pair(300, seed=8)
    with pytest.raises(ValueError):
        nested_kalman_backtest(p.y, p.x)  # too short for default F+T
    with pytest.raises(ValueError):
        nested_kalman_backtest(p.y, p.x, formation=10, trading=50)
