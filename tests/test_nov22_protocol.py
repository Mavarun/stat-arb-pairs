"""Nov 22: frozen design, fresh seeds, and the rules on hand-made rows (no evaluation seed used)."""

import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("e22", ROOT / "scripts" / "run_nov22_experiments.py")
E = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E)


def test_design_is_frozen_and_seeds_are_new():
    assert list(E.SEEDS) == list(range(1800, 1820))
    assert E.DRIFTS == (0.0, 0.02, 0.05) and E.DECISION_DRIFT == 0.05 and E.TREND_FROM == 252
    assert E.ARMS["ols_f126"] == {"hedge": "ols", "formation": 126}
    assert E.ARMS["kalman_qb8"] == {"hedge": "kalman", "q_beta": 1e-8}
    used = set(range(1000, 1060)) | set(range(1100, 1160))
    for name in ("run_nov08_experiments", "run_nov11_experiments", "run_nov19_experiments", "run_nov20_experiments",
                 "run_nov20b_experiments", "run_nov20c_experiments"):
        s = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
        m = importlib.util.module_from_spec(s)
        s.loader.exec_module(m)
        used |= set(m.SEEDS)
    assert not used & set(E.SEEDS) and not set(E.DEV_SEEDS) & set(E.SEEDS)


def _toy(gain, zero_gain=-0.05):
    """gain[arm][hl] is the challenger's edge over ols at the decision drift; other drifts get zero_gain."""
    rows = []
    for hl in E.HALF_LIVES:
        for d in E.DRIFTS:
            for s in range(20):
                rows.append({"half_life": hl, "drift": d, "seed": s, "arm": "ols", "sharpe": 0.1 * s})
                for arm in E.CHALLENGERS:
                    g = gain.get(arm, {}).get(hl, -0.1) if d == E.DECISION_DRIFT else zero_gain
                    rows.append({"half_life": hl, "drift": d, "seed": s, "arm": arm, "sharpe": 0.1 * s + g})
    return pd.DataFrame(rows)


def test_h1_needs_two_rows():
    d = E.decide(_toy({"kalman": {5.0: 0.2, 10.0: 0.2, 20.0: -0.1}}))
    assert d["H1"] and d["H1_by_arm"] == {"kalman": True, "kalman_qb8": False} and not d["H2"]
    assert d["trend_only"] == {"kalman": True}
    assert not E.decide(_toy({"kalman": {5.0: 0.2, 10.0: -0.1, 20.0: -0.1}}))["H1"]


def test_h2_and_trend_only_flag():
    d = E.decide(_toy({"ols_f126": {5.0: 0.2, 10.0: 0.2, 20.0: 0.2}}, zero_gain=0.05))
    assert d["H2"] and not d["H1"] and d["trend_only"] == {"ols_f126": False}


def test_holm_covers_all_27_tests():
    t = E.table(_toy({}))
    assert len(t) == 27 and (t.holm_p >= t.sign_p - 1e-15).all()
