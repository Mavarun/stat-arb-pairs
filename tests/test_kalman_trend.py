import numpy as np
import pandas as pd
import pytest

from statarb.kalman import kalman_hedge, kalman_trend_hedge
from statarb.synth import make_beta_trend_pair, make_ou_pair


def test_trend_filter_is_causal_and_warmup_is_nan():
    p = make_ou_pair(500, seed=3)
    base = kalman_trend_hedge(p.y, p.x)
    assert base.iloc[:60].isna().all().all()
    assert np.isfinite(base.iloc[60:].to_numpy()).all()
    y2 = p.y.copy()
    y2.iloc[300:] += 50.0
    shocked = kalman_trend_hedge(y2, p.x)
    pd.testing.assert_frame_equal(base.iloc[:300], shocked.iloc[:300])
    assert base["beta_prior"].iloc[300] == shocked["beta_prior"].iloc[300]


def test_zero_drift_prior_reduces_to_random_walk_filter():
    p = make_ou_pair(400, seed=4)
    a = kalman_trend_hedge(p.y, p.x, q_beta=1e-6, q_drift=0.0, p_drift0=0.0)
    b = kalman_hedge(p.y, p.x, q_beta=1e-6)
    np.testing.assert_allclose(a[b.columns].to_numpy(), b.to_numpy(), rtol=1e-9, atol=1e-12, equal_nan=True)


def test_rejects_bad_inputs():
    p = make_ou_pair(100, seed=5)
    with pytest.raises(ValueError):
        kalman_trend_hedge(p.y.iloc[:50], p.x.iloc[:50])
    with pytest.raises(ValueError):
        kalman_trend_hedge(p.y, p.x, q_drift=-1.0)


def test_drift_state_learns_a_linear_trend_and_tracks_better_than_random_walk():
    err_llt, err_rw, drift_end = [], [], []
    for seed in range(5):
        p = make_beta_trend_pair(1260, trend_from=0, drift_per_year=0.05, half_life=5.0, seed=seed)
        llt = kalman_trend_hedge(p.y, p.x)
        rw = kalman_hedge(p.y, p.x, q_beta=1e-8)
        tail = slice(600, None)
        err_llt.append(np.mean(np.abs(llt["beta"].iloc[tail] - p.beta_path.iloc[tail])))
        err_rw.append(np.mean(np.abs(rw["beta"].iloc[tail] - p.beta_path.iloc[tail])))
        drift_end.append(llt["drift"].iloc[-1])
    true_g = 1.5 * 0.05 / 252
    assert np.median(err_llt) < np.median(err_rw)
    assert np.median(drift_end) > 0.3 * true_g
