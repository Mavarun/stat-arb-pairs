import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from statarb.cointegration import engle_granger
from statarb.synth import (
    half_life_to_rho,
    make_ou_pair,
    make_planted_universe,
    make_random_walk_universe,
    n_pairs,
)


def test_half_life_to_rho_is_exact():
    rho = half_life_to_rho(10.0)
    assert abs(rho**10 - 0.5) < 1e-12
    with pytest.raises(ValueError):
        half_life_to_rho(0.0)


def test_generator_is_seeded_and_reproducible():
    a = make_ou_pair(300, seed=11)
    b = make_ou_pair(300, seed=11)
    c = make_ou_pair(300, seed=12)
    pd.testing.assert_series_equal(a.y, b.y)
    assert not np.allclose(a.y.to_numpy(), c.y.to_numpy())


def test_identity_y_equals_intercept_plus_beta_x_plus_spread():
    p = make_ou_pair(400, beta=1.3, intercept=2.0, seed=3)
    recon = p.intercept + p.beta_path * p.x + p.spread
    np.testing.assert_allclose(p.y.to_numpy(), recon.to_numpy(), atol=1e-10)
    assert (p.x > 0).all()


def test_planted_spread_has_planted_ar1_and_stationary_std():
    # Long sample so sampling error is small; checks the generator, not an estimator.
    p = make_ou_pair(20000, half_life=12.0, sigma_spread=2.0, seed=5)
    s = p.spread.to_numpy()
    rho_hat = float(np.corrcoef(s[1:], s[:-1])[0, 1])
    assert abs(rho_hat - p.rho) < 0.01
    assert abs(float(s.std()) - 2.0) / 2.0 < 0.1


def test_engle_granger_recovers_planted_beta_across_seeds():
    errs, pvals = [], []
    for seed in range(10):
        p = make_ou_pair(1000, beta=1.5, intercept=5.0, half_life=10.0, seed=seed)
        eg = engle_granger(p.y, p.x)
        errs.append(eg.hedge_ratio - 1.5)
        pvals.append(eg.pvalue)
    # Superconsistent OLS: tiny error relative to beta for a fast-reverting spread.
    assert np.max(np.abs(errs)) < 0.05
    assert np.mean(np.array(pvals) < 0.05) == 1.0


def test_random_walk_universe_has_no_shared_driver():
    df = make_random_walk_universe(8, 2000, seed=1)
    rets = np.log(df).diff().dropna()
    off = rets.corr().to_numpy()[~np.eye(8, dtype=bool)]
    assert np.max(np.abs(off)) < 0.1


def test_planted_universe_truth_and_shape():
    df, truth = make_planted_universe(n_noise=6, n_pairs=3, n=500, seed=2)
    assert df.shape == (500, 12)
    assert len(truth) == 3
    for pair in truth:
        a, b = sorted(pair)
        # Each planted pair's levels regress tightly (spread std ~1 vs price ~100).
        ols = sm.OLS(df[a], sm.add_constant(df[b])).fit()
        assert ols.resid.std() < 2.0
    assert n_pairs(12) == 66
