"""Every number in the README's Nov 22 section comes from the committed results."""

import importlib.util
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text()
spec = importlib.util.spec_from_file_location("e22", ROOT / "scripts" / "run_nov22_experiments.py")
E = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E)
DF = pd.read_csv(ROOT / "results" / "nov22_beta_trend.csv")
SEC = README[README.index("## Nov 22 slice:"): README.index("## Nov 20 third slice:")]


def _p(p):
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def test_section_sits_above_the_nov20_slices():
    assert README.index("## Nov 22 slice:") < README.index("## Nov 20 third slice:") < README.index("## Nov 20 second slice:")


def test_results_table():
    t = E.decide(DF)["table"].set_index(["arm", "half_life", "drift"])
    for hl in E.HALF_LIVES:
        cells = []
        for d in E.DRIFTS:
            parts = [f"{t.loc[('kalman', hl, d)].ols:.2f}"]
            for a in E.CHALLENGERS:
                r = t.loc[(a, hl, d)]
                parts.append(f"{r.challenger:.2f} ({int(r.wins)}/{int(r.losses)} [{_p(r.holm_p)}])")
            cells.append(" / ".join(parts))
        row = f"| {int(hl)} | " + " | ".join(cells) + " |"
        assert row in SEC, row
    k5 = t.xs(("kalman", 0.05), level=("arm", "drift"))
    assert f"wins {int(k5.loc[10.0].wins)}/{int(k5.loc[10.0].losses)} and {int(k5.loc[20.0].wins)}/{int(k5.loc[20.0].losses)}" in SEC
    assert f"({k5.loc[5.0].challenger:.2f} vs {k5.loc[5.0].ols:.2f} at half-life 5)" in SEC
    z = t.loc[("kalman", 5.0, 0.0)]
    assert f"{z.ols:.2f} -> {z.challenger:.2f}, {int(z.wins)}/{int(z.losses)}" in SEC
    f = t.xs("ols_f126", level="arm")
    assert not ((f.challenger > f.ols) & (f.holm_p < 0.05)).any() and int(((f.challenger < f.ols) & (f.holm_p < 0.05)).sum()) == 3


def test_cost_and_trade_numbers():
    m = DF.groupby(["half_life", "drift", "arm"])[["trades", "cost"]].median()
    a, b = m.loc[(5.0, 0.0, "ols")], m.loc[(5.0, 0.05, "ols")]
    assert f"median {a.trades:.0f} -> {b.trades:.0f} trades" in SEC and f"{a.cost:.3f} -> {b.cost:.3f}" in SEC
    assert f"median {m.loc[(5.0, 0.0, 'kalman')].trades:.1f} vs {a.trades:.0f} trades" in SEC
    lo, hi = m.trades.min(), m.trades.max()
    assert f"a median of {math.floor(lo)}-{math.ceil(hi)} trades" in SEC


def test_posthoc_numbers():
    tr = pd.read_csv(ROOT / "results" / "nov22_tracking_posthoc.csv").set_index(["half_life", "drift", "arm"])
    for hl in (5.0, 20.0):
        r = tr.loc[(hl, 0.05)]
        cells = " | ".join(f"{r.loc[a].hedge_abs_err:.3f} / {r.loc[a].hedge_lag:.3f} / {r.loc[a].corr_true:.2f}"
                           for a in ("ols", "ols_f126", "kalman", "kalman_qb8"))
        assert f"| {int(hl)}: mean abs error / signed error / corr(z, spread) | {cells} |" in SEC
    c = pd.read_csv(ROOT / "results" / "nov22_common_span_posthoc.csv").set_index(["half_life", "drift"])
    r = c.loc[(5.0, 0.05)]
    assert f"{r.ols_f126_common:.2f} vs {r.ols:.2f}, {int(r.wins)}/{int(r.losses)}" in SEC
    assert (c.losses > c.wins).all() and "in all 9 rows" in SEC
