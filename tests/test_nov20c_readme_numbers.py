"""Every number in the README's Nov 20 third slice comes from the committed results."""

import importlib.util
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text()
spec = importlib.util.spec_from_file_location("e20c", ROOT / "scripts" / "run_nov20c_experiments.py")
E = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E)
sys.path.insert(0, str(ROOT / "scripts"))
import run_nov20b_experiments as B  # noqa: E402

SEC = README[README.index("## Nov 20 third slice:"): README.index("## Nov 20 second slice:")]


def _p(p):
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def test_cells():
    df = pd.read_csv(ROOT / "results" / "nov20c_big_jumps.csv")
    t = E.decide(df)["table"].set_index(["half_life", "jump"])
    k = B.paired(df, "ols", "kalman").set_index(["half_life", "jump"])
    for (hl, j), a in t.iterrows():
        cell = (f"{k.loc[(hl, j)].kalman:.2f} / {a.kalman_qb8:.2f} / {a.ols:.2f} "
                f"({int(a.wins)}/{int(a.losses)} [{_p(a.holm_p)}])")
        assert cell in SEC, cell
        assert k.loc[(hl, j)].kalman < a.ols  # default Kalman loses to OLS in every row
    assert f"({t.loc[(5.0, 0.05)].kalman_qb8:.2f} vs {t.loc[(5.0, 0.05)].ols:.2f} at 5%)" in SEC
    assert f"wins only {int(t.loc[(5.0, 0.05)].wins)} and {int(t.loc[(5.0, 0.10)].wins)} of 20" in SEC
    assert (df[df.half_life == 20.0].groupby(["jump", "arm"]).sharpe.median() < 0).all()
