"""make_beta_trend_pair: a pair whose hedge ratio keeps trending, so every re-fit lags."""

import numpy as np
import pytest

from statarb.synth import make_beta_trend_pair, make_ou_pair


def test_zero_drift_is_the_plain_pair():
    a = make_beta_trend_pair(400, trend_from=100, drift_per_year=0.0, half_life=5.0, seed=7)
    b = make_ou_pair(400, half_life=5.0, seed=7)
    np.testing.assert_array_equal(a.y.to_numpy(), b.y.to_numpy())
    np.testing.assert_array_equal(a.x.to_numpy(), b.x.to_numpy())


def test_trend_is_linear_from_the_start_bar_and_spread_is_shared():
    a = make_beta_trend_pair(600, trend_from=252, drift_per_year=0.10, half_life=5.0, seed=7)
    b = make_beta_trend_pair(600, trend_from=252, drift_per_year=0.0, half_life=5.0, seed=7)
    np.testing.assert_array_equal(a.y.to_numpy()[:253], b.y.to_numpy()[:253])
    np.testing.assert_array_equal(a.spread.to_numpy(), b.spread.to_numpy())
    bp = a.beta_path.to_numpy()
    assert bp[252] == pytest.approx(1.5) and bp[252 + 252] == pytest.approx(1.65)
    assert np.allclose(np.diff(bp[252:]), 1.5 * 0.10 / 252)
    resid = a.y.to_numpy() - a.intercept - bp * a.x.to_numpy()
    np.testing.assert_allclose(resid, a.spread.to_numpy(), atol=1e-9)


def test_negative_drift_and_bars_per_year():
    a = make_beta_trend_pair(300, trend_from=0, drift_per_year=-0.2, bars_per_year=100, seed=1)
    assert a.beta_path.iloc[100] == pytest.approx(1.5 * 0.8)


def test_a_shorter_formation_window_lags_the_trend_less():
    # the trending ratio a 252-bar OLS fit recovers sits further behind the end-of-window
    # truth than a 126-bar fit's, on average over seeds
    def lag(seed, f):
        p = make_beta_trend_pair(756, trend_from=252, drift_per_year=0.10, half_life=5.0, seed=seed)
        w = slice(756 - f, 756)
        X = np.column_stack([np.ones(f), p.x.to_numpy()[w]])
        return p.beta_path.to_numpy()[-1] - np.linalg.lstsq(X, p.y.to_numpy()[w], rcond=None)[0][1]

    long_lag = np.mean([lag(s, 252) for s in range(12)])
    short_lag = np.mean([lag(s, 126) for s in range(12)])
    assert long_lag > short_lag > 0


@pytest.mark.parametrize("bad", [{"trend_from": -1}, {"trend_from": 50}, {"drift_per_year": float("inf")},
                                 {"bars_per_year": 0}, {"drift_per_year": -30.0}])
def test_rejects_bad_arguments(bad):
    kw = {"trend_from": 10, "drift_per_year": 0.1, **bad}
    with pytest.raises(ValueError):
        make_beta_trend_pair(50, **kw)
