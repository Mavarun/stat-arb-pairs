"""gate_quantile: Engle-Granger formation gate at a half-life-calibrated level."""

from __future__ import annotations

import numpy as np
import pytest

from statarb.formation import formation_trading_backtest
from statarb.gatecal import calibrated_gate_level
from statarb.ou import fit_ou
from statarb.rank import formation_residual
from statarb.synth import make_broken_pair, make_ou_pair


def test_off_is_the_legacy_book() -> None:
    p = make_broken_pair(900, break_at=500, kind="random_walk", seed=2)
    a = formation_trading_backtest(p.y, p.x, carry_over=False)
    b = formation_trading_backtest(p.y, p.x, gate_quantile=None, carry_over=False)
    assert a.returns.equals(b.returns)
    assert b.windows["gate_level"].isna().all() and b.windows["gate_ref_half_life"].isna().all()


def test_fixed_gate_records_its_level_and_is_unchanged() -> None:
    p = make_broken_pair(1000, break_at=400, kind="random_walk", seed=3)
    r = formation_trading_backtest(p.y, p.x, gate_p=0.01, carry_over=False)
    assert (r.windows["gate_level"] == 0.01).all()
    assert (r.windows["gated"] == (r.windows["eg_p"] > 0.01)).all()


def test_reference_is_the_first_formation_half_life_and_frozen() -> None:
    p = make_ou_pair(1000, half_life=12.0, seed=11)
    r = formation_trading_backtest(p.y, p.x, gate_quantile=0.95, carry_over=False)
    hl0 = fit_ou(formation_residual(p.y.to_numpy()[:252], p.x.to_numpy()[:252])).half_life
    assert (r.windows["gate_ref_half_life"] == hl0).all()
    assert (r.windows["gate_level"] == calibrated_gate_level(hl0, 252, quantile=0.95)).all()
    assert (r.windows["gated"] == (r.windows["eg_p"] > r.windows["gate_level"])).all()


def test_explicit_reference_overrides_the_estimate() -> None:
    p = make_ou_pair(900, half_life=12.0, seed=11)
    r = formation_trading_backtest(p.y, p.x, gate_quantile=0.95, gate_ref_half_life=12.0, carry_over=False)
    assert (r.windows["gate_ref_half_life"] == 12.0).all()
    assert r.windows["gate_level"].iloc[0] == calibrated_gate_level(12.0, 252, quantile=0.95)


def test_gated_blocks_are_flat_and_cost_free() -> None:
    p = make_broken_pair(1260, break_at=567, kind="random_walk", half_life=10.0, seed=1)
    r = formation_trading_backtest(p.y, p.x, gate_quantile=0.9, carry_over=False)
    g = r.windows[r.windows["gated"]]
    assert len(g) > 0
    for _, w in g.iterrows():
        assert (r.positions.loc[w["trade_start"]:w["trade_end"]] == 0.0).all()
        assert (r.returns.loc[w["trade_start"]:w["trade_end"]] == 0.0).all()


def test_calibrated_gate_starves_a_slow_healthy_pair_less_than_a_fixed_one() -> None:
    fixed = calib = 0
    for s in range(4):
        p = make_ou_pair(1260, half_life=20.0, seed=s)
        fixed += formation_trading_backtest(p.y, p.x, gate_p=0.01, carry_over=False).windows["gated"].sum()
        calib += formation_trading_backtest(p.y, p.x, gate_quantile=0.95, carry_over=False).windows["gated"].sum()
    assert fixed >= 2 * calib and fixed > 20


def test_causal_under_a_late_break() -> None:
    clean = make_broken_pair(1000, break_at=900, kind="none", half_life=10.0, seed=4)
    late = make_broken_pair(1000, break_at=900, kind="random_walk", half_life=10.0, seed=4)
    a = formation_trading_backtest(clean.y, clean.x, gate_quantile=0.95, carry_over=False)
    b = formation_trading_backtest(late.y, late.x, gate_quantile=0.95, carry_over=False)
    cut = late.y.index[899]
    assert np.allclose(a.returns.loc[:cut].to_numpy(), b.returns.loc[:cut].to_numpy())
    assert a.windows["gate_level"].iloc[0] == b.windows["gate_level"].iloc[0]


def test_exclusive_with_a_fixed_gate() -> None:
    p = make_ou_pair(600, seed=0)
    with pytest.raises(ValueError):
        formation_trading_backtest(p.y, p.x, gate_p=0.05, gate_quantile=0.95, carry_over=False)
