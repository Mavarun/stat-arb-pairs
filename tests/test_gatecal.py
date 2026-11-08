"""Half-life-calibrated gate level: healthy-pair p-value quantiles."""

from __future__ import annotations

import numpy as np
import pytest

from statarb.gatecal import _snap, calibrated_gate_level, healthy_eg_pvalues
from statarb.synth import make_ou_pair
from statsmodels.tsa.stattools import coint


def test_level_rises_with_the_half_life() -> None:
    lv = [calibrated_gate_level(hl, 252, n_sims=120) for hl in (3.0, 5.0, 10.0, 20.0)]
    assert all(a < b for a, b in zip(lv, lv[1:]))
    assert lv[0] < 0.05 and lv[-1] > 0.3


def test_level_is_reproducible_and_monotone_in_the_quantile() -> None:
    a = calibrated_gate_level(8.0, 252, quantile=0.9, n_sims=60)
    b = calibrated_gate_level(8.0, 252, quantile=0.9, n_sims=60)
    c = calibrated_gate_level(8.0, 252, quantile=0.99, n_sims=60)
    assert a == b and c >= a


def test_pvalues_are_the_engle_granger_pvalues_of_the_seeded_pairs() -> None:
    p = healthy_eg_pvalues(5.0, 100, n_sims=20, seed=7)
    q = make_ou_pair(100, half_life=_snap(5.0), seed=7 + 3)
    assert p[3] == pytest.approx(coint(q.y.to_numpy(), q.x.to_numpy(), trend="c", autolag="aic")[1])
    assert len(p) == 20 and ((p >= 0) & (p <= 1)).all()


def test_level_gates_about_one_minus_q_of_fresh_healthy_windows() -> None:
    lv = calibrated_gate_level(10.0, 252, quantile=0.9, n_sims=200)
    fresh = healthy_eg_pvalues(10.0, 252, n_sims=200, seed=90_000)
    assert 0.03 < float((fresh > lv).mean()) < 0.20


def test_snap_grid_and_undefined_reference() -> None:
    assert _snap(10.0) == _snap(10.1)
    assert _snap(0.2) == pytest.approx(1.0) and _snap(1e6) == pytest.approx(250.0, rel=0.03)
    assert calibrated_gate_level(float("inf"), 252) == 1.0
    assert calibrated_gate_level(float("nan"), 252) == 1.0


def test_validation() -> None:
    with pytest.raises(ValueError):
        calibrated_gate_level(5.0, 252, quantile=0.3)
    with pytest.raises(ValueError):
        healthy_eg_pvalues(5.0, 10)
    with pytest.raises(ValueError):
        healthy_eg_pvalues(-1.0, 252)
    with pytest.raises(ValueError):
        healthy_eg_pvalues(5.0, 252, n_sims=5)
