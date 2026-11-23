"""Nov 24 post-hoc drift-learning reading (decides nothing): committed csv recomputes on two seeds' worth of shape."""

import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("e24p", ROOT / "scripts" / "run_nov24_experiments.py")
E = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E)
P = pd.read_csv(ROOT / "results" / "nov24_drift_learning_posthoc.csv")


def test_committed_reading():
    assert list(P.half_life) == [5.0, 10.0, 20.0]
    # the drift state overshoots the planted per-bar drift about 2.2x in every row ...
    assert ((P.drift_frac > 2.0) & (P.drift_frac < 2.4)).all()
    # ... and the hedge error is only a few percent below OLS
    assert (P.abs_err_kalman_llt < P.abs_err_ols).all()
    assert ((P.abs_err_ols - P.abs_err_kalman_llt) / P.abs_err_ols < 0.06).all()


def test_recomputes():
    fresh = E.drift_learning()
    pd.testing.assert_frame_equal(fresh.round(4), P, check_exact=False, atol=1e-4)
