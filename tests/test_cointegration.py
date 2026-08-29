import numpy as np
import pandas as pd

from statarb.cointegration import engle_granger, johansen
from tests._synth import make_cointegrated_pair, make_independent_walks


def test_engle_granger_recovers_known_hedge_ratio(coint_pair):
    y, x, beta, intercept = coint_pair
    res = engle_granger(y, x)
    assert abs(res.hedge_ratio - beta) < 0.05
    assert abs(res.intercept - intercept) < 1.5
    # Cointegrated construction should reject no-coint at 5%.
    assert res.pvalue < 0.05
    assert res.statistic < res.crit_values["5%"]
    assert res.nobs == len(y)


def test_engle_granger_independent_walks_not_forced_cointegrated():
    y, x = make_independent_walks(n=800, seed=123)
    res = engle_granger(y, x)
    # We do not require p > 0.05 (finite samples can false-reject), but the
    # recovered slope should not magically match a planted 1.6 from the other test.
    assert np.isfinite(res.pvalue)
    assert 0.0 <= res.pvalue <= 1.0
    assert abs(res.hedge_ratio - 1.6) > 0.1


def test_johansen_returns_real_statsmodels_shapes(coint_pair):
    y, x, _, _ = coint_pair
    data = pd.concat([y, x], axis=1)
    joh = johansen(data, det_order=0, k_ar_diff=1)
    assert joh.trace_stat.shape == (2,)
    assert joh.max_eig_stat.shape == (2,)
    assert joh.crit_trace.shape == (2, 3)
    assert joh.evec.shape == (2, 2)
    assert joh.eig.shape == (2,)
    assert np.all(np.isfinite(joh.trace_stat))
    assert np.all(np.isfinite(joh.crit_trace))
    # Eigenvalues from a real Johansen fit are non-negative.
    assert np.all(joh.eig >= -1e-12)
    frame = joh.as_frame()
    assert list(frame.columns) == ["trace_stat", "crit_90", "crit_95", "crit_99", "max_eig_stat"]
    # For this cointegrated construction, r=0 trace should beat 90% crit.
    # (Not used for trading; just a sanity check that we did not stub zeros.)
    assert joh.trace_stat[0] > joh.crit_trace[0, 0]


def test_johansen_independent_walks_not_stubbed_true():
    y, x = make_independent_walks(n=800, seed=3)
    joh = johansen(pd.concat([y, x], axis=1))
    # Must be real numbers, not a hardcoded "cointegrated" boolean.
    assert np.isfinite(joh.trace_stat[0])
    # Independent walks usually fail to reject r=0 at 99%; we only assert the
    # helper did not return a fake all-reject result.
    assert not bool(np.all(joh.trace_stat > joh.crit_trace[:, 2]))
