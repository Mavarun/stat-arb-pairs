"""z_stop in the formation/trading walk-forward: legacy identity, causality, stop counts."""

from __future__ import annotations

import numpy as np
import pytest

from statarb.formation import formation_trading_backtest
from statarb.synth import make_broken_pair


@pytest.mark.parametrize("hedge", ["ols", "kalman"])
def test_off_is_the_legacy_book(hedge: str) -> None:
    p = make_broken_pair(900, break_at=500, kind="random_walk", seed=2)
    a = formation_trading_backtest(p.y, p.x, hedge=hedge)
    b = formation_trading_backtest(p.y, p.x, hedge=hedge, z_stop=None)
    assert a.returns.equals(b.returns) and (b.windows["z_stops"] == 0).all()


def test_stop_is_causal_under_a_late_break() -> None:
    clean = make_broken_pair(1000, break_at=900, kind="none", seed=4)
    late = make_broken_pair(1000, break_at=900, kind="random_walk", seed=4)
    a = formation_trading_backtest(clean.y, clean.x, z_stop=3.5, use_time_stop=True)
    b = formation_trading_backtest(late.y, late.x, z_stop=3.5, use_time_stop=True)
    cut = late.y.index[899]
    assert np.allclose(a.returns.loc[:cut].to_numpy(), b.returns.loc[:cut].to_numpy())
    assert a.positions.loc[:cut].equals(b.positions.loc[:cut])


def test_stops_fire_only_after_a_level_shift() -> None:
    pre = post = 0
    for seed in range(6):
        p = make_broken_pair(1000, break_at=600, kind="level_shift", shift_sigmas=6.0, seed=seed)
        on = formation_trading_backtest(p.y, p.x, z_stop=3.5)
        w = on.windows
        after = w["trade_start"] >= p.y.index[600]
        post += int(w.loc[after, "z_stops"].sum())
        pre += int(w.loc[~after & (w["trade_end"] < p.y.index[600]), "z_stops"].sum())
    # a 6-sigma shift drives z far past 3.5 against the book; a healthy OU rarely does
    assert post > pre


def test_invalid_z_stop_is_rejected() -> None:
    p = make_broken_pair(600, break_at=400, kind="none", seed=0)
    with pytest.raises(ValueError):
        formation_trading_backtest(p.y, p.x, z_stop=1.0)
