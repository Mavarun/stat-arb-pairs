import numpy as np
import pandas as pd
import pytest

from statarb.cointegration import engle_granger
from statarb.kalman import kalman_hedge, static_hedge_path
from statarb.synth import make_ou_pair


def _rmse(est: pd.Series, truth: pd.Series) -> float:
    m = est.notna()
    return float(np.sqrt(np.mean((est[m] - truth[m]) ** 2)))


def test_warmup_bars_are_nan_and_outputs_after_are_finite():
    p = make_ou_pair(400, seed=1)
    kf = kalman_hedge(p.y, p.x, warmup=60)
    assert kf.iloc[:60].isna().all().all()
    assert np.isfinite(kf.iloc[60:].to_numpy()).all()
    with pytest.raises(ValueError):
        kalman_hedge(p.y.iloc[:50], p.x.iloc[:50], warmup=60)


def test_filter_is_causal_future_shock_does_not_move_past_estimates():
    p = make_ou_pair(500, seed=2)
    base = kalman_hedge(p.y, p.x)
    y2 = p.y.copy()
    y2.iloc[300:] += 50.0  # huge break after bar 300
    shocked = kalman_hedge(y2, p.x)
    pd.testing.assert_frame_equal(base.iloc[:300], shocked.iloc[:300])
    # The prior (used for the signal at bar 300) still has not seen bar 300.
    assert base["beta_prior"].iloc[300] == shocked["beta_prior"].iloc[300]
    assert base["innovation"].iloc[300] != shocked["innovation"].iloc[300]


def test_kalman_tracks_drifting_beta_where_static_ols_fails():
    rk, rs_causal, rs_full = [], [], []
    for seed in range(6):
        n = 1500
        truth_path = np.linspace(1.0, 2.0, n)
        p = make_ou_pair(n, beta_path=truth_path, half_life=5.0, seed=seed)
        kf = kalman_hedge(p.y, p.x)
        rk.append(_rmse(kf["beta"], p.beta_path))
        rs_causal.append(_rmse(static_hedge_path(p.y, p.x), p.beta_path))
        # Full-sample OLS even gets to look ahead, and is still wrong.
        full = pd.Series(engle_granger(p.y, p.x).hedge_ratio, index=p.y.index)
        rs_full.append(_rmse(full.where(kf["beta"].notna()), p.beta_path))
    assert np.median(rk) < 0.3
    assert np.median(rk) < np.median(rs_causal) / 2
    assert np.median(rk) < np.median(rs_full) / 2


def test_on_constant_beta_kalman_is_close_but_ols_is_better():
    # Honest trade-off: a random-walk state adds estimation noise when beta is fixed.
    rk, rs = [], []
    for seed in range(6):
        p = make_ou_pair(1500, beta=1.5, half_life=5.0, seed=10 + seed)
        rk.append(_rmse(kalman_hedge(p.y, p.x)["beta"], p.beta_path))
        rs.append(abs(engle_granger(p.y, p.x).hedge_ratio - 1.5))
    assert np.median(rk) < 0.1
    assert np.median(rs) < np.median(rk)


def test_zero_state_noise_reduces_to_recursive_ols():
    # q -> 0 makes the filter recursive least squares started from the warm-up OLS
    # (prior covariance = OLS covariance with the same r), so its final state is
    # the full-sample OLS fit. This pins the update algebra.
    p = make_ou_pair(1000, beta=1.5, seed=3)
    kf = kalman_hedge(p.y, p.x, q_beta=0.0, q_alpha=0.0)
    eg = engle_granger(p.y, p.x)
    assert abs(kf["beta"].iloc[-1] - eg.hedge_ratio) < 1e-6
    assert abs(kf["alpha"].iloc[-1] - eg.intercept) < 1e-4


def test_innovation_zscore_is_roughly_standardised_on_a_good_pair():
    zs = []
    for seed in range(5):
        p = make_ou_pair(1500, beta=1.2, half_life=5.0, seed=20 + seed)
        zs.append(float(kalman_hedge(p.y, p.x)["zscore"].std()))
    assert 0.5 < np.median(zs) < 1.5
