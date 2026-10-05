import numpy as np
import pandas as pd
import pytest

from statarb.cointegration import engle_granger
from statarb.ou import fit_ou
from statarb.signals import compute_spread, mean_reversion_positions
from statarb.synth import make_ou_pair


@pytest.mark.parametrize("half_life", [3.0, 10.0, 30.0])
def test_fit_ou_recovers_planted_half_life_on_true_spread(half_life):
    est = []
    for seed in range(20):
        p = make_ou_pair(3000, half_life=half_life, sigma_spread=1.5, seed=seed)
        est.append(fit_ou(p.spread).half_life)
    med = float(np.median(est))
    # Median across seeds within 15% of truth; small-sample bias is downward.
    assert abs(med - half_life) / half_life < 0.15


def test_fit_ou_recovers_mu_and_stationary_std():
    p = make_ou_pair(5000, half_life=8.0, sigma_spread=2.0, seed=4)
    fit = fit_ou(p.spread + 3.0)
    assert abs(fit.mu - 3.0) < 0.3
    assert abs(fit.sigma_eq - 2.0) / 2.0 < 0.1
    assert fit.is_mean_reverting


def test_half_life_on_estimated_engle_granger_residual():
    # End-to-end: hedge is estimated, not given. Still close to truth.
    est = []
    for seed in range(10):
        p = make_ou_pair(2000, beta=1.2, half_life=10.0, seed=100 + seed)
        eg = engle_granger(p.y, p.x)
        est.append(fit_ou(compute_spread(p.y, p.x, eg.hedge_ratio, eg.intercept)).half_life)
    assert abs(float(np.median(est)) - 10.0) / 10.0 < 0.2


def test_random_walk_spread_has_long_or_infinite_half_life():
    rng = np.random.default_rng(0)
    hls = [fit_ou(pd.Series(np.cumsum(rng.normal(size=1000)))).half_life for _ in range(10)]
    # A unit root shows up as b ~ 1: half-life >> any planted value above.
    assert min(hls) > 40


def test_thresholds_and_zscore_are_consistent():
    p = make_ou_pair(2000, half_life=5.0, seed=9)
    fit = fit_ou(p.spread)
    th = fit.thresholds(2.0, 0.5)
    assert th["long_entry"] < th["long_exit"] < fit.mu < th["short_exit"] < th["short_entry"]
    z = fit.zscore(pd.Series([th["short_entry"], th["long_exit"]]))
    np.testing.assert_allclose(z.to_numpy(), [2.0, -0.5])
    assert fit.max_hold(3.0) == int(np.ceil(3.0 * fit.half_life))


def test_fit_ou_rejects_tiny_sample():
    with pytest.raises(ValueError):
        fit_ou(pd.Series(np.arange(5.0)))


def test_time_stop_flattens_stuck_position_and_blocks_same_side_reentry():
    z = pd.Series([0.0, -2.5, -2.6, -2.7, -2.8, -2.9, -1.0, -2.5, -0.1])
    no_stop = mean_reversion_positions(z, z_enter=2.0, z_exit=0.5)
    np.testing.assert_array_equal(no_stop.to_numpy(), [0, 1, 1, 1, 1, 1, 1, 1, 0])
    stop = mean_reversion_positions(z, z_enter=2.0, z_exit=0.5, max_hold=3)
    # Held 3 bars, stopped out, blocked while |z| stays beyond 2, re-armed at z=-1.0,
    # re-enters at -2.5, then exits normally.
    np.testing.assert_array_equal(stop.to_numpy(), [0, 1, 1, 1, 0, 0, 0, 1, 0])


def test_time_stop_none_matches_legacy_behaviour():
    rng = np.random.default_rng(1)
    z = pd.Series(rng.normal(0, 1.5, 500))
    a = mean_reversion_positions(z, 2.0, 0.5)
    b = mean_reversion_positions(z, 2.0, 0.5, max_hold=None)
    pd.testing.assert_series_equal(a, b)
    with pytest.raises(ValueError):
        mean_reversion_positions(z, 2.0, 0.5, max_hold=0)
