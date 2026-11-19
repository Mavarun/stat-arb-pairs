"""Which Kalman drift moves the signal (dev seeds of the Nov 20 second slice, no decision)."""

import numpy as np

from statarb.formation import _window_signal
from statarb.kalman import kalman_hedge
from statarb.synth import make_beta_jump_pair


def _pair(seed=1, jump=0.02):
    return make_beta_jump_pair(1260, break_at=786, jump=jump, half_life=10, seed=seed)


def test_intercept_drift_barely_moves_z_and_z_is_not_white():
    p = _pair()
    a = kalman_hedge(p.y[:315], p.x[:315], warmup=252)
    b = kalman_hedge(p.y[:315], p.x[:315], warmup=252, q_alpha=0.0)
    assert np.nanmax(np.abs(a.zscore - b.zscore)) < 0.02
    z = a.zscore.dropna().to_numpy()
    assert np.corrcoef(z[:-1], z[1:])[0, 1] > 0.8


def _tracking(seed, hedge, q_beta, q_alpha=1e-4):
    p = _pair(seed, jump=0.0)
    y, x, sp = p.y.to_numpy(), p.x.to_numpy(), p.spread.to_numpy()
    cs, st = [], 0
    while st + 315 <= len(y):
        z, _, _ = _window_signal(y[st:st + 252], x[st:st + 252], y[st + 252:st + 315], x[st + 252:st + 315],
                                 hedge, q_beta, q_alpha)
        cs.append(np.corrcoef(z, sp[st + 252:st + 315])[0, 1])
        st += 63
    return float(np.mean(cs))


def test_slower_hedge_drift_tracks_the_planted_spread_better():
    for seed in (1, 2, 3):
        fast, slow, ols = _tracking(seed, "kalman", 1e-6), _tracking(seed, "kalman", 1e-8), _tracking(seed, "ols", 1e-6)
        assert fast < 0.9 < slow < ols
        assert abs(_tracking(seed, "kalman", 1e-6, q_alpha=0.0) - fast) < 0.01
