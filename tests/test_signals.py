import numpy as np
import pandas as pd

from statarb.signals import compute_spread, mean_reversion_positions, rolling_zscore, zscore_from_stats


def test_compute_spread_matches_definition():
    idx = pd.RangeIndex(5)
    y = pd.Series([10.0, 12.0, 14.0, 16.0, 18.0], index=idx)
    x = pd.Series([4.0, 5.0, 6.0, 7.0, 8.0], index=idx)
    s = compute_spread(y, x, hedge_ratio=2.0, intercept=1.0)
    expected = y - 1.0 - 2.0 * x
    pd.testing.assert_series_equal(s, expected, check_names=False)


def test_rolling_zscore_mean_near_zero_on_stationary_spread():
    rng = np.random.default_rng(0)
    n = 4000
    # Stationary AR(1); z-score of a zero-mean series should itself be ~0 mean.
    e = np.zeros(n)
    shocks = rng.normal(0.0, 1.0, n)
    for t in range(1, n):
        e[t] = 0.3 * e[t - 1] + shocks[t]
    spread = pd.Series(e)
    z = rolling_zscore(spread, window=60).dropna()
    assert z.notna().all()
    assert abs(float(z.mean())) < 0.05
    # Unit-scale-ish: std of rolling z on a long stationary series ~ 1.
    assert 0.7 < float(z.std(ddof=1)) < 1.3


def test_rolling_zscore_no_lookahead_on_first_window():
    s = pd.Series(np.arange(20, dtype=float))
    z = rolling_zscore(s, window=10)
    assert z.iloc[:9].isna().all()
    assert np.isfinite(z.iloc[9])


def test_zscore_from_stats_uses_frozen_moments():
    s = pd.Series([1.0, 3.0, 5.0])
    z = zscore_from_stats(s, mean=3.0, std=2.0)
    np.testing.assert_allclose(z.to_numpy(), [-1.0, 0.0, 1.0])


def test_positions_enter_hold_exit_on_constructed_z():
    # Hand-built z path: enter long, hold, exit, enter short, exit.
    z = pd.Series(
        [
            0.0,
            -2.1,  # enter long
            -1.5,  # hold
            -0.6,  # hold: still beyond exit (|z| > 0.5)
            -0.2,  # exit long: z crossed -z_exit
            0.0,
            2.2,  # enter short
            0.8,  # hold
            0.4,  # exit short: z crossed +z_exit
            0.0,
        ]
    )
    pos = mean_reversion_positions(z, z_enter=2.0, z_exit=0.5)
    expected = [0, 1, 1, 1, 0, 0, -1, -1, 0, 0]
    np.testing.assert_array_equal(pos.to_numpy(), expected)


def test_nan_z_flattens_position():
    z = pd.Series([np.nan, np.nan, -3.0, -3.0, np.nan, -3.0])
    pos = mean_reversion_positions(z, z_enter=2.0, z_exit=0.5)
    np.testing.assert_array_equal(pos.to_numpy(), [0, 0, 1, 1, 0, 1])
