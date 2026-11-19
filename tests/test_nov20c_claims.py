"""Nov 20 third slice: committed results reproduce and the rule's verdict recomputes."""

import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("e20c", ROOT / "scripts" / "run_nov20c_experiments.py")
E = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E)
DF = pd.read_csv(ROOT / "results" / "nov20c_big_jumps.csv")


def test_shape():
    assert len(DF) == 3 * 2 * 20 * 3 and sorted(DF.jump.unique()) == [0.05, 0.10]


def test_one_cell_reproduces():
    fresh = E.run(half_lives=(10.0,), jumps=(0.10,))
    old = DF[(DF.half_life == 10.0) & (DF.jump == 0.10)].reset_index(drop=True)
    pd.testing.assert_frame_equal(fresh, old, check_exact=False, rtol=1e-9, atol=1e-12)


def test_verdict():
    d = E.decide(DF)
    assert not d["moving_hedge_pays"] and d["rows_passing"] == 0
    t = d["table"]
    assert (t.losses > t.wins).all()  # the slow hedge loses more seeds than it wins in every row
