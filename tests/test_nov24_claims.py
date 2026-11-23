"""Nov 24: frozen design, committed results reproduce, and the rules' verdicts recompute."""

import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("e24", ROOT / "scripts" / "run_nov24_experiments.py")
E = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E)
DF = pd.read_csv(ROOT / "results" / "nov24_llt_hedge.csv")


def test_frozen_design_and_fresh_seeds():
    assert list(E.SEEDS) == list(range(1900, 1920))
    used = set()
    for f in (ROOT / "scripts").glob("run_*.py"):
        if f.name == "run_nov24_experiments.py":
            continue
        txt = f.read_text()
        for lo in range(1900, 1920):
            if f"range({lo}," in txt:
                used.add(lo)
    assert not used
    assert E.ARMS["kalman_llt"] == {"hedge": "kalman_llt", "q_beta": 1e-8, "q_drift": 1e-12}
    assert E.DECISION_DRIFT == 0.05 and E.DRIFTS == (0.0, 0.02, 0.05)


def test_shape():
    assert len(DF) == 3 * 3 * 20 * 3 and set(DF.arm) == set(E.ARMS)


def test_one_cell_reproduces():
    fresh = E.run(half_lives=(20.0,), drifts=(0.05,))
    old = DF[(DF.half_life == 20.0) & (DF.drift == 0.05)].reset_index(drop=True)
    pd.testing.assert_frame_equal(fresh, old, check_exact=False, rtol=1e-9, atol=1e-12)


def test_rules_on_hand_made_rows():
    rows = []
    for hl in E.HALF_LIVES:
        for s in range(20):
            for arm, v in (("ols", 0.0), ("kalman", 0.0), ("kalman_llt", 1.0)):
                rows.append({"half_life": hl, "drift": 0.05, "seed": s, "arm": arm, "sharpe": v})
            for arm, v in (("ols", 1.0), ("kalman", 0.0), ("kalman_llt", 0.5)):
                rows.append({"half_life": hl, "drift": 0.0, "seed": s, "arm": arm, "sharpe": v})
    d = E.decide(pd.DataFrame(rows))
    assert d["H1"] and d["H2"] and d["trend_only"]


def test_verdicts():
    d = E.decide(DF)
    assert not d["H1"] and not d["H2"] and not d["trend_only"]
    t = d["table"]
    o = t[t.vs == "ols"].set_index(["half_life", "drift"])
    # ahead of OLS on median at 5%/yr in every row, but no Holm pass; behind in every other row with Holm p < 0.01
    assert (o.xs(0.05, level="drift").challenger > o.xs(0.05, level="drift").base).all()
    assert (o.xs(0.05, level="drift").holm_p > 0.5).all()
    rest = o[o.index.get_level_values("drift") < 0.05]
    assert (rest.challenger < rest.base).all() and (rest.holm_p < 0.01).all()
    # never ahead of the default random-walk Kalman hedge on median
    k = t[t.vs == "kalman"]
    assert (k.challenger < k.base).all()
