"""Nov 22: committed results reproduce and the rules' verdicts recompute."""

import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("e22", ROOT / "scripts" / "run_nov22_experiments.py")
E = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E)
DF = pd.read_csv(ROOT / "results" / "nov22_beta_trend.csv")


def test_shape():
    assert len(DF) == 3 * 3 * 20 * 4 and sorted(DF.drift.unique()) == [0.0, 0.02, 0.05]
    assert set(DF.arm) == set(E.ARMS) and sorted(DF.seed.unique()) == list(E.SEEDS)


def test_one_cell_reproduces():
    fresh = E.run(half_lives=(5.0,), drifts=(0.05,))
    old = DF[(DF.half_life == 5.0) & (DF.drift == 0.05)].reset_index(drop=True)
    pd.testing.assert_frame_equal(fresh, old, check_exact=False, rtol=1e-9, atol=1e-12)


def test_verdicts():
    d = E.decide(DF)
    assert not d["H1"] and not d["H2"] and d["trend_only"] == {}
    t = d["table"].set_index(["arm", "half_life", "drift"])
    # the default Kalman hedge is ahead of OLS on median in every 5%/yr row but clears Holm only at half-life 5
    k5 = t.xs(("kalman", 0.05), level=("arm", "drift"))
    assert (k5.challenger > k5.ols).all() and list(k5.holm_p < 0.05) == [True, False, False]
    # ... and it loses to OLS in every zero-drift and 2%/yr row
    k0 = t[(t.index.get_level_values("arm") == "kalman") & (t.index.get_level_values("drift") < 0.05)]
    assert (k0.challenger < k0.ols).all() and (k0.losses > k0.wins).all()
    # every book loses money at 5%/yr in every half-life row
    assert (DF[DF.drift == 0.05].groupby(["half_life", "arm"]).sharpe.median() < 0).all()
