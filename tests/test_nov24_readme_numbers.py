"""Every number in the README's Nov 24 section comes from the committed results."""

import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text()
spec = importlib.util.spec_from_file_location("e24r", ROOT / "scripts" / "run_nov24_experiments.py")
E = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E)
DF = pd.read_csv(ROOT / "results" / "nov24_llt_hedge.csv")
P = pd.read_csv(ROOT / "results" / "nov24_drift_learning_posthoc.csv")
SEC = README[README.index("## Nov 24 slice:"): README.index("## Nov 22 slice:")]


def _p(p):
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def test_section_sits_above_nov22():
    assert README.index("## Nov 24 slice:") < README.index("## Nov 22 slice:")


def test_results_table():
    t = E.decide(DF)["table"]
    for hl in E.HALF_LIVES:
        cells = []
        for d in E.DRIFTS:
            o = t[(t.vs == "ols") & (t.half_life == hl) & (t.drift == d)].iloc[0]
            k = t[(t.vs == "kalman") & (t.half_life == hl) & (t.drift == d)].iloc[0]
            cells.append(f"{o.base:.2f} / {k.base:.2f} / {o.challenger:.2f} ({o.wins}/{o.losses} [{_p(o.holm_p)}]; "
                         f"{k.wins}/{k.losses} [{_p(k.holm_p)}])")
        assert f"| {int(hl)} | " + " | ".join(cells) + " |" in SEC


def test_posthoc_table_and_prose():
    for r in P.itertuples():
        assert f"| {int(r.half_life)} | {r.drift_frac:.2f} | {r.abs_err_ols:.3f} / {r.abs_err_kalman:.3f} / {r.abs_err_kalman_llt:.3f} |" in SEC
    assert "about 2.2x" in SEC and round(P.drift_frac.median(), 1) == 2.2
    gaps = (P.abs_err_ols - P.abs_err_kalman_llt) / P.abs_err_ols
    assert "2-5% below OLS" in SEC and gaps.min() >= 0.015 and gaps.max() < 0.055
    assert "12/8, 10/10 and 12/8" in SEC and "0.83 -> -0.31, 0/20" in SEC
    assert "20 new seeds (1900-1919)" in SEC and "H1 fails" in SEC and "H2 fails" in SEC
