"""Nov 20 third slice: frozen design and the rule on hand-made rows."""

import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("e20c", ROOT / "scripts" / "run_nov20c_experiments.py")
E = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E)


def test_design_is_frozen_and_seeds_are_new():
    assert list(E.SEEDS) == list(range(1700, 1720)) and E.JUMPS == (0.05, 0.10)
    used = set()
    for name in ("run_nov08_experiments", "run_nov11_experiments", "run_nov19_experiments", "run_nov20_experiments",
                 "run_nov20b_experiments"):
        s = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
        m = importlib.util.module_from_spec(s)
        s.loader.exec_module(m)
        used |= set(m.SEEDS)
    used |= set(range(1000, 1060)) | set(range(1100, 1160))
    assert not used & set(E.SEEDS)


def _toy(gain_by_hl):
    rows = []
    for hl in E.HALF_LIVES:
        for j in E.JUMPS:
            for s in range(20):
                g = gain_by_hl[hl] if j == 0.10 else -0.1
                rows += [{"half_life": hl, "jump": j, "seed": s, "arm": "ols", "sharpe": 0.1 * s},
                         {"half_life": hl, "jump": j, "seed": s, "arm": "kalman_qb8", "sharpe": 0.1 * s + g},
                         {"half_life": hl, "jump": j, "seed": s, "arm": "kalman", "sharpe": 0.0}]
    return pd.DataFrame(rows)


def test_rule():
    assert E.decide(_toy({5.0: -0.1, 10.0: 0.2, 20.0: 0.2}))["moving_hedge_pays"]
    assert not E.decide(_toy({5.0: -0.1, 10.0: -0.1, 20.0: 0.2}))["moving_hedge_pays"]
