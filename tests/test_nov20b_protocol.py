"""Nov 20 second slice: the frozen design of the slow-hedge Kalman comparison."""

import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("e20b", ROOT / "scripts" / "run_nov20b_experiments.py")
E = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E)


def test_design_is_frozen():
    assert list(E.SEEDS) == list(range(1600, 1620)) and E.JUMPS == (0.0, 0.02) and E.HALF_LIVES == (5.0, 10.0, 20.0)
    assert E.ARMS["kalman_qb8"] == {"hedge": "kalman", "q_beta": 1e-8}
    assert E.ARMS["kalman"] == {"hedge": "kalman"}


def test_seeds_are_new():
    used = set()
    for name in ("run_nov08_experiments", "run_nov11_experiments", "run_nov19_experiments", "run_nov20_experiments"):
        s = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
        m = importlib.util.module_from_spec(s)
        s.loader.exec_module(m)
        used |= set(m.SEEDS)
    used |= set(range(1000, 1060)) | set(range(1100, 1160))  # calibration scripts
    assert not used & set(E.SEEDS)


def _toy(qb8_minus_k, qb8_minus_ols):
    rows = []
    for hl in E.HALF_LIVES:
        for j in E.JUMPS:
            for s in range(20):
                base = 0.1 * s
                rows += [{"half_life": hl, "jump": j, "seed": s, "arm": "kalman", "sharpe": base},
                         {"half_life": hl, "jump": j, "seed": s, "arm": "kalman_qb8", "sharpe": base + qb8_minus_k},
                         {"half_life": hl, "jump": j, "seed": s, "arm": "ols", "sharpe": base + qb8_minus_k - qb8_minus_ols}]
    return pd.DataFrame(rows)


def test_rules():
    d = E.decide(_toy(0.3, -0.1))
    assert d["H1"] and not d["H2"]
    d = E.decide(_toy(0.3, 0.1))
    assert d["H1"] and d["H2"]
    assert not E.decide(_toy(-0.1, 0.1))["H1"]
