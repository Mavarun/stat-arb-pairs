"""Every Nov 20 number in the README is recomputed from the committed results tables."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")
RES = ROOT / "results"


def _section() -> str:
    i = README.index("## Nov 20 slice:")
    return README[i: README.index("\n## ", i + 5)]


def _p(p: float) -> str:
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def test_it_comes_right_before_nov19():
    heads = [h for h in README.splitlines() if h.startswith("## ")]
    i = next(k for k, h in enumerate(heads) if h.startswith("## Nov 20 slice:"))
    assert heads[i + 1].startswith("## Nov 19 slice:")


def test_sharpe_cells():
    s, sec = pd.read_csv(RES / "nov20_beta_jump.csv"), _section()
    for r in s[s.hedge == "ols"].itertuples():
        assert f"{r.off:.2f} / **{r.carry:.2f}** ({r.wins}/{r.losses} [{_p(r.holm_p)}])" in sec, r
    k = s[s.hedge == "kalman"]
    assert f"lose money in {int((k.off < 0).sum() + (k.carry < 0).sum())} of 24 book-rows" in sec


def test_bar0_cells_and_total():
    b, sec = pd.read_csv(RES / "nov20_beta_jump_bar0.csv"), _section()
    ols = b[b.hedge == "ols"]
    for hl, g in ols.groupby("half_life"):
        cells = " | ".join(f"{r.carried_blocks} / {r.exit} / {r.flip}" for r in g.sort_values("jump").itertuples())
        assert f"| {hl:g} | {cells} |" in sec
    assert f"None of the {ols.carried_blocks.sum():,} carried OLS blocks flips" in sec
    assert f"carry at most {b[b.hedge == 'kalman'].carried_blocks.max()} blocks per row" in sec


def test_posthoc_cells():
    t, sec = pd.read_csv(RES / "nov20_beta_jump_posthoc.csv"), _section()
    t = t[t.jump == 0.02].set_index(["half_life", "window"])
    for hl in (5.0, 10.0, 20.0):
        cells = " | ".join(f"{100 * t.loc[(hl, w), 'hedge_err_median']:.1f}% / {t.loc[(hl, w), 'resid_sd_ratio_median']:.2f}"
                           for w in ("before jump", "straddles jump", "after jump"))
        assert f"| {hl:g} | {cells} |" in sec
