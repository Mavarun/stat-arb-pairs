"""CUSUM break watch: hand paths, scaling, causality, validation."""

from __future__ import annotations

import numpy as np
import pytest

from statarb.breakwatch import cusum_alarm, cusum_path, long_run_scale


def test_long_run_scale_is_one_for_white_noise_and_grows_with_rho() -> None:
    assert long_run_scale(0.0) == pytest.approx(1.0)
    assert long_run_scale(0.5) == pytest.approx(np.sqrt(3.0))
    assert long_run_scale(0.9) > long_run_scale(0.5) > 1.0
    assert np.isfinite(long_run_scale(1.5))  # clipped, not infinite


def test_cusum_path_matches_a_hand_computation() -> None:
    z = np.array([1.0, 1.0, -2.0, 0.0])
    # k = 0.5: S+ = 0.5, 1.0, 0, 0 ; S- = 0, 0, 1.5, 1.0 -> max = 0.5, 1.0, 1.5, 1.0
    assert np.allclose(cusum_path(z, rho=0.0, k=0.5), [0.5, 1.0, 1.5, 1.0])


def test_rho_scaling_divides_the_input() -> None:
    z = np.full(6, 2.0)
    s = long_run_scale(0.6)
    assert np.allclose(cusum_path(z, rho=0.6, k=0.0), cusum_path(z / s, rho=0.0, k=0.0))


def test_a_sustained_shift_alarms_and_noise_does_not() -> None:
    rng = np.random.default_rng(0)
    noise = rng.normal(0.0, 1.0, 400)
    assert cusum_alarm(noise, rho=0.0, k=0.5, h=8.0) is None
    shifted = noise.copy()
    shifted[100:] += 3.0
    i = cusum_alarm(shifted, rho=0.0, k=0.5, h=8.0)
    assert i is not None and 100 <= i < 110
    neg = noise.copy()
    neg[100:] -= 3.0
    assert cusum_alarm(neg, rho=0.0, k=0.5, h=8.0) is not None


def test_alarm_is_causal() -> None:
    rng = np.random.default_rng(1)
    z = rng.normal(0.0, 1.0, 200)
    z[80:] += 2.0
    i = cusum_alarm(z, k=0.5, h=6.0)
    changed = z.copy()
    changed[i + 1:] = 0.0  # rewrite the future
    assert cusum_alarm(changed, k=0.5, h=6.0) == i


def test_nan_carries_the_statistic_forward() -> None:
    z = np.array([1.0, np.nan, 1.0])
    assert np.allclose(cusum_path(z, k=0.0), [1.0, 1.0, 2.0])


def test_validation() -> None:
    with pytest.raises(ValueError):
        cusum_path(np.zeros(3), k=-0.1)
    with pytest.raises(ValueError):
        cusum_alarm(np.zeros(3), h=0.0)
