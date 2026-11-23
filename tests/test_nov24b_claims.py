"""Nov 24b: frozen design, committed results reproduce, and the verdicts recompute."""

import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("e24b", ROOT / "scripts" / "run_nov24b_experiments.py")
E = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E)
DF = pd.read_csv(ROOT / "results" / "nov24b_exec_delay.csv")


def test_frozen_design_and_fresh_seeds():
    assert list(E.SEEDS) == list(range(2000, 2020)) and E.DELAYS == (0, 1) and E.HAIRCUT == 0.25
    others = [f.read_text() for f in (ROOT / "scripts").glob("run_*.py") if f.name != "run_nov24b_experiments.py"]
    assert not any(f"range({lo}," in t for t in others for lo in range(2000, 2020))


def test_shape_and_one_cell_reproduces():
    assert len(DF) == 3 * 20 * 2
    fresh = E.run(half_lives=(5.0,))
    old = DF[DF.half_life == 5.0].reset_index(drop=True)
    pd.testing.assert_frame_equal(fresh, old, check_exact=False, rtol=1e-9, atol=1e-12)


def test_verdicts():
    d = E.decide(DF)
    t = d["table"].set_index("half_life")
    assert not d["H1"] and not d["H2"] and d["n_material"] == 1
    assert t.loc[20.0, "delay0"] < 0  # H1 fails because the slow row loses money even at the close
    assert (t.loc[5.0, "wins"], t.loc[5.0, "losses"]) == (1, 19) and t.loc[5.0, "holm_p"] < 0.001
    assert t.loc[10.0, "haircut"] > 0.25 and t.loc[10.0, "holm_p"] > 0.05


def test_delay_moves_gross_not_trades():
    p = DF.pivot_table(index=["half_life", "seed"], columns="arm", values=["trades", "cost", "gross"])
    dt = p["trades"]["delay1"] - p["trades"]["delay0"]
    assert (dt == 0).sum() == 57 and dt.min() == -1  # same trades on 57 of 60 pair-runs, one fewer on 3
    m = DF.groupby(["half_life", "arm"])["gross"].median()
    assert m.loc[(5.0, "delay1")] < m.loc[(5.0, "delay0")] and m.loc[(10.0, "delay1")] < m.loc[(10.0, "delay0")]
