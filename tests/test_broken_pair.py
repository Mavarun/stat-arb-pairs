"""Planted cointegration breaks: shared pre-break path, the two break kinds, validation."""

from __future__ import annotations

import numpy as np
import pytest
from statsmodels.tsa.stattools import adfuller

from statarb.synth import make_broken_pair, make_ou_pair


def test_none_is_the_unbroken_pair() -> None:
    a = make_broken_pair(400, break_at=200, kind="none", seed=3)
    b = make_ou_pair(400, half_life=5.0, seed=3)
    assert np.array_equal(a.y.to_numpy(), b.y.to_numpy()) and np.array_equal(a.x.to_numpy(), b.x.to_numpy())


@pytest.mark.parametrize("kind", ["random_walk", "level_shift"])
def test_pre_break_path_and_x_leg_are_shared(kind: str) -> None:
    base = make_broken_pair(600, break_at=300, kind="none", seed=5)
    brk = make_broken_pair(600, break_at=300, kind=kind, seed=5)
    assert np.array_equal(brk.y.to_numpy()[:300], base.y.to_numpy()[:300])
    assert np.array_equal(brk.x.to_numpy(), base.x.to_numpy())
    assert not np.array_equal(brk.y.to_numpy()[300:], base.y.to_numpy()[300:])
    # y - intercept - beta x is exactly the planted spread
    resid = brk.y - brk.intercept - brk.beta * brk.x
    assert np.allclose(resid.to_numpy(), brk.spread.to_numpy())


def test_level_shift_moves_the_mean_by_the_planted_amount() -> None:
    base = make_broken_pair(600, break_at=300, kind="none", seed=1)
    brk = make_broken_pair(600, break_at=300, kind="level_shift", shift_sigmas=4.0, seed=1)
    d = (brk.spread - base.spread).to_numpy()
    assert np.allclose(d[:300], 0.0) and np.allclose(d[300:], 4.0)


def test_random_walk_break_kills_stationarity() -> None:
    # pre-break OU is stationary; a long post-break random walk is not (ADF, 10 seeds)
    rej_pre = rej_post = 0
    for seed in range(10):
        p = make_broken_pair(1600, break_at=400, kind="random_walk", seed=seed)
        rej_pre += adfuller(p.spread.to_numpy()[:400])[1] < 0.05
        rej_post += adfuller(p.spread.to_numpy()[400:])[1] < 0.05
    assert rej_pre == 10 and rej_post <= 3


def test_validation() -> None:
    with pytest.raises(ValueError):
        make_broken_pair(100, break_at=50, kind="regime")
    with pytest.raises(ValueError):
        make_broken_pair(100, break_at=100)
    with pytest.raises(ValueError):
        make_broken_pair(100, break_at=0)
