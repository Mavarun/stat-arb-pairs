"""cusum_halt in the formation/trading walk-forward: legacy identity, halts, causality.

Breaks are planted at bar 567 = 252 + 5 * 63, the first bar of a trading block, so that block's
formation window is clean (a mid-window break contaminates the formation mean, std and rho)."""

from __future__ import annotations

import numpy as np
import pytest

from statarb.formation import formation_trading_backtest
from statarb.synth import make_broken_pair


@pytest.mark.parametrize("hedge", ["ols", "kalman"])
def test_off_is_the_legacy_book(hedge: str) -> None:
    p = make_broken_pair(900, break_at=500, kind="random_walk", seed=2)
    a = formation_trading_backtest(p.y, p.x, hedge=hedge)
    b = formation_trading_backtest(p.y, p.x, hedge=hedge, cusum_halt=False)
    assert a.returns.equals(b.returns) and b.windows["halt_bar"].isna().all()


def test_halted_block_is_flat_from_the_alarm_bar_on() -> None:
    p = make_broken_pair(1000, break_at=567, kind="level_shift", shift_sigmas=6.0, seed=3)
    r = formation_trading_backtest(p.y, p.x, cusum_halt=True)
    halted = r.windows[r.windows["halt_bar"].notna()]
    assert len(halted) > 0
    for _, w in halted.iterrows():
        block = r.positions.loc[w["trade_start"]:w["trade_end"]].to_numpy()
        assert (block[int(w["halt_bar"]):] == 0.0).all()


def test_halts_concentrate_after_a_planted_shift() -> None:
    pre = post = 0
    for seed in range(6):
        p = make_broken_pair(1000, break_at=567, kind="level_shift", shift_sigmas=6.0, seed=seed)
        w = formation_trading_backtest(p.y, p.x, cusum_halt=True).windows
        after = w["trade_start"] >= p.y.index[567]
        post += int(w.loc[after, "halt_bar"].notna().sum())
        pre += int(w.loc[~after & (w["trade_end"] < p.y.index[567]), "halt_bar"].notna().sum())
    assert post > pre


def test_halt_is_causal_under_a_late_break() -> None:
    clean = make_broken_pair(1000, break_at=900, kind="none", seed=4)
    late = make_broken_pair(1000, break_at=900, kind="random_walk", seed=4)
    a = formation_trading_backtest(clean.y, clean.x, cusum_halt=True)
    b = formation_trading_backtest(late.y, late.x, cusum_halt=True)
    cut = late.y.index[899]
    assert np.allclose(a.returns.loc[:cut].to_numpy(), b.returns.loc[:cut].to_numpy())
    assert a.positions.loc[:cut].equals(b.positions.loc[:cut])


def test_a_halt_never_adds_trades_within_its_block() -> None:
    p = make_broken_pair(1000, break_at=567, kind="level_shift", shift_sigmas=6.0, seed=5)
    off = formation_trading_backtest(p.y, p.x)
    on = formation_trading_backtest(p.y, p.x, cusum_halt=True)
    assert (on.windows["trades"].to_numpy() <= off.windows["trades"].to_numpy()).all()
