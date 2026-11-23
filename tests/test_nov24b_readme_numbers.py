"""Numbers in the README's Nov 24 second slice match the committed results."""

import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text()
HEAD = "## Nov 24 slice (second):"
SEC = README[README.index(HEAD): README.index("## Nov 24 slice: a hedge filter")]
spec = importlib.util.spec_from_file_location("e24b", ROOT / "scripts" / "run_nov24b_experiments.py")
E = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E)
DF = pd.read_csv(ROOT / "results" / "nov24b_exec_delay.csv")


def test_order():
    assert README.index(HEAD) < README.index("## Nov 24 slice: a hedge filter") < README.index("## Nov 22 slice:")


def test_table_rows():
    t = E.decide(DF)["table"]
    for _, r in t.iterrows():
        p = "<0.001" if r.holm_p < 0.001 else f"{r.holm_p:.3f}"
        line = f"| {int(r.half_life)} | {r.delay0:.3f} | {r.delay1:.3f} | {int(r.wins)} / {int(r.losses)} | {p} |"
        assert line in SEC, line


def test_gross_and_cost_claims():
    m = DF.groupby(["half_life", "arm"])[["gross", "cost"]].median()
    assert f"median gross P&L ({m.loc[(20.0, 'delay0'), 'gross']:.3f})" in SEC
    assert f"median cost ({m.loc[(20.0, 'delay0'), 'cost']:.3f})" in SEC
    assert f"{m.loc[(5.0, 'delay0'), 'gross']:.3f} -> {m.loc[(5.0, 'delay1'), 'gross']:.3f}" in SEC


def test_assumption_line_points_here():
    t = E.decide(DF)["table"].set_index("half_life")
    assert f"from {t.loc[5.0, 'delay0']:.2f} to {t.loc[5.0, 'delay1']:.2f}" in README
