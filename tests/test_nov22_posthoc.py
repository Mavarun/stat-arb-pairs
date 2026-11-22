"""Nov 22 post-hoc readings (decide nothing): committed csvs reproduce and the quoted patterns hold."""

import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("e22", ROOT / "scripts" / "run_nov22_experiments.py")
E = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E)
TR = pd.read_csv(ROOT / "results" / "nov22_tracking_posthoc.csv").set_index(["half_life", "drift", "arm"])


def test_tracking_reproduces_at_reduced_scale():
    fresh = E.tracking(seeds=range(1800, 1804), half_lives=(5.0,), drifts=(0.05,)).set_index(["half_life", "drift", "arm"])
    full = TR.loc[(5.0, 0.05)]
    # 4 of 20 seeds: same order of magnitude, and the 126-bar OLS fit still has the smallest absolute error
    assert fresh.loc[(5.0, 0.05, "ols_f126"), "hedge_abs_err"] < fresh.loc[(5.0, 0.05, "ols"), "hedge_abs_err"]
    assert abs(fresh.loc[(5.0, 0.05, "ols"), "hedge_abs_err"] - full.loc["ols", "hedge_abs_err"]) < 0.06


def test_hedge_error_is_large_but_not_a_one_sided_lag():
    for hl in E.HALF_LIVES:
        r = TR.loc[(hl, 0.05)]
        assert (r.hedge_abs_err > 0.09).all() and (r.hedge_lag.abs() < 0.02).all()
        assert r.hedge_abs_err.idxmin() == "ols_f126"


def test_sharpe_order_does_not_follow_tracking_at_5pct():
    # the default Kalman z tracks the planted spread worst in every 5%/yr row, yet beats OLS on median Sharpe there
    df = pd.read_csv(ROOT / "results" / "nov22_beta_trend.csv")
    med = df.groupby(["half_life", "drift", "arm"]).sharpe.median()
    for hl in E.HALF_LIVES:
        r = TR.loc[(hl, 0.05)]
        assert r.corr_true.idxmin() == "kalman"
        assert med[(hl, 0.05, "kalman")] > med[(hl, 0.05, "ols")]
