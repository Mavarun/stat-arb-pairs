"""The committed Nov 20b results reproduce, and the decisions are the pre-registered ones."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("e20b", ROOT / "scripts" / "run_nov20b_experiments.py")
E = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E)
DF = pd.read_csv(ROOT / "results" / "nov20b_kalman_qb8.csv")


def test_one_row_per_arm_seed_cell():
    assert len(DF) == 3 * 2 * 20 * 3
    assert set(DF.arm) == set(E.ARMS) and sorted(DF.seed.unique()) == list(E.SEEDS)


@pytest.mark.parametrize("hl,j", [(5.0, 0.02), (20.0, 0.0)])
def test_spot_cells_reproduce(hl, j):
    fresh = E.run(seeds=E.SEEDS, half_lives=(hl,), jumps=(j,))
    old = DF[(DF.half_life == hl) & (DF.jump == j)].reset_index(drop=True)
    pd.testing.assert_frame_equal(fresh, old, check_exact=False, rtol=1e-9, atol=1e-12)


def test_decisions():
    d = E.decide(DF)
    assert d["H1"] and not d["H2"]
    h1 = d["h1_table"]
    assert (h1.holm_p < 0.05).sum() == 5
    h2 = d["h2_table"]
    assert (h2.ols > h2.kalman_qb8).all()  # OLS is ahead in every row, not only the 2% ones


def test_tracking_posthoc_reproduces_one_cell_and_orders_the_arms():
    t = pd.read_csv(ROOT / "results" / "nov20b_tracking_posthoc.csv")
    fresh = E.tracking(seeds=E.SEEDS, half_lives=(10.0,), jumps=(0.02,)).round(4)
    old = t[(t.half_life == 10.0) & (t.jump == 0.02)].reset_index(drop=True)
    pd.testing.assert_frame_equal(fresh, old, check_exact=False, atol=1e-4)
    for (_, _), g in t.groupby(["half_life", "jump"]):
        c = g.set_index("arm").corr_true
        assert c["kalman"] < c["kalman_qb8"] < c["ols"]
