"""gate_p: Engle-Granger re-test of the formation window before each trading block."""

from __future__ import annotations

import numpy as np
import pytest

from statarb.formation import formation_trading_backtest
from statarb.synth import make_broken_pair, make_ou_pair


def test_off_is_the_legacy_book() -> None:
    p = make_broken_pair(900, break_at=500, kind="random_walk", seed=2)
    a = formation_trading_backtest(p.y, p.x, carry_over=False)
    b = formation_trading_backtest(p.y, p.x, gate_p=None, carry_over=False)
    assert a.returns.equals(b.returns)
    assert b.windows["eg_p"].isna().all() and not b.windows["gated"].any()


def test_gated_blocks_are_flat_and_cost_free() -> None:
    p = make_broken_pair(1000, break_at=400, kind="random_walk", seed=3)
    r = formation_trading_backtest(p.y, p.x, gate_p=0.01, carry_over=False)
    g = r.windows[r.windows["gated"]]
    assert len(g) > 0
    for _, w in g.iterrows():
        assert (r.positions.loc[w["trade_start"]:w["trade_end"]] == 0.0).all()
        assert (r.returns.loc[w["trade_start"]:w["trade_end"]] == 0.0).all()
        assert w["trades"] == 0 and w["eg_p"] > 0.01


def test_a_looser_gate_never_trades_less() -> None:
    p = make_ou_pair(1000, half_life=15.0, seed=1)
    tight = formation_trading_backtest(p.y, p.x, gate_p=0.001, carry_over=False).windows["gated"].sum()
    loose = formation_trading_backtest(p.y, p.x, gate_p=0.5, carry_over=False).windows["gated"].sum()
    assert tight >= loose


def test_gate_is_causal_under_a_late_break() -> None:
    clean = make_broken_pair(1000, break_at=900, kind="none", seed=4)
    late = make_broken_pair(1000, break_at=900, kind="random_walk", seed=4)
    a = formation_trading_backtest(clean.y, clean.x, gate_p=0.05, carry_over=False)
    b = formation_trading_backtest(late.y, late.x, gate_p=0.05, carry_over=False)
    cut = late.y.index[899]
    assert np.allclose(a.returns.loc[:cut].to_numpy(), b.returns.loc[:cut].to_numpy())


def test_gate_and_cusum_compose() -> None:
    p = make_broken_pair(1000, break_at=567, kind="random_walk", seed=5)
    r = formation_trading_backtest(p.y, p.x, gate_p=0.05, cusum_halt=True, carry_over=False)
    w = r.windows
    assert not (w["gated"] & w["halt_bar"].notna()).any()  # a gated block has nothing left to halt
