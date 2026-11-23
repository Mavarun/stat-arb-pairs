import numpy as np
import pytest

from statarb.formation import _window_signal, formation_trading_backtest
from statarb.synth import make_beta_trend_pair, make_ou_pair


def test_llt_hedge_runs_in_the_walk_forward_book():
    p = make_ou_pair(800, seed=11)
    r = formation_trading_backtest(p.y, p.x, hedge="kalman_llt", q_beta=1e-8)
    assert np.isfinite(r.sharpe)
    assert r.trades > 0


def test_llt_with_no_drift_prior_matches_kalman_book():
    p = make_ou_pair(800, seed=12)
    yv, xv = p.y.to_numpy(), p.x.to_numpy()
    za, ba, _ = _window_signal(yv[:252], xv[:252], yv[252:315], xv[252:315], "kalman", 1e-6, 1e-4)
    zb, bb, _ = _window_signal(yv[:252], xv[:252], yv[252:315], xv[252:315], "kalman_llt", 1e-6, 1e-4, 0.0)
    # q_drift 0 still has a non-zero drift prior, so they differ but stay close on a stable pair
    assert np.max(np.abs(bb - ba)) < 0.05
    assert np.corrcoef(za, zb)[0, 1] > 0.95


def test_unknown_hedge_rejected_and_ols_unchanged_by_q_drift():
    p = make_beta_trend_pair(800, trend_from=252, drift_per_year=0.05, seed=13)
    a = formation_trading_backtest(p.y, p.x)
    b = formation_trading_backtest(p.y, p.x, q_drift=1e-9)
    assert a.sharpe == b.sharpe
    with pytest.raises(ValueError):
        formation_trading_backtest(p.y, p.x, hedge="nope")
