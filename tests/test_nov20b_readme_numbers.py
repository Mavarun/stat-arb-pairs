"""Every number in the README's Nov 20 second slice comes from the committed results."""

import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text()
spec = importlib.util.spec_from_file_location("e20b", ROOT / "scripts" / "run_nov20b_experiments.py")
E = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E)


def _section() -> str:
    i = README.index("## Nov 20 second slice:")
    return README[i: README.index("\n## ", i + 5)]


def _p(p: float) -> str:
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def test_order():
    heads = [h for h in README.splitlines() if h.startswith("## ")]
    i = next(k for k, h in enumerate(heads) if h.startswith("## Nov 20 second slice:"))
    assert heads[i + 1].startswith("## Nov 20 slice:")


def test_result_cells_and_decisions():
    d = E.decide(pd.read_csv(ROOT / "results" / "nov20b_kalman_qb8.csv"))
    sec = _section()
    h1, h2 = d["h1_table"].set_index(["half_life", "jump"]), d["h2_table"].set_index(["half_life", "jump"])
    for (hl, j), a in h1.iterrows():
        b = h2.loc[(hl, j)]
        cell = (f"{a.kalman:.2f} / {a.kalman_qb8:.2f} / {b.ols:.2f} ({int(a.wins)}/{int(a.losses)} [{_p(a.holm_p)}]; "
                f"{int(b.wins)}/{int(b.losses)} [{_p(b.holm_p)}])")
        assert cell in sec, cell
    assert d["H1"] and not d["H2"] and "**H1 passes.**" in sec and "**H2 fails" in sec
    assert f"Holm p < 0.05 in {(h1.holm_p < 0.05).sum()} of them" in sec
    a = h1.loc[(20.0, 0.0)]
    assert f"is {int(a.wins)}/{int(a.losses)})" in sec
    assert f"from {h1.loc[(5.0, 0.0)].kalman:.2f} to\n  {h1.loc[(5.0, 0.0)].kalman_qb8:.2f}" in sec


def test_tracking_cells():
    t = pd.read_csv(ROOT / "results" / "nov20b_tracking_posthoc.csv").set_index(["half_life", "jump", "arm"])
    sec = _section()
    for hl in (5.0, 10.0, 20.0):
        cells = [" / ".join(f"{t.loc[(hl, j, a)].corr_true:.3f}" for a in ("kalman", "kalman_qb8", "ols"))
                 for j in (0.0, 0.02)]
        assert f"| {hl:g} | " + " | ".join(cells) + " |" in sec
    assert (f"({t.loc[(5.0, 0.02, 'kalman_qb8')].corr_true:.3f} vs {t.loc[(5.0, 0.02, 'ols')].corr_true:.3f} "
            "at half-life 5)") in sec
