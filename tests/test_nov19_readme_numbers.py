"""Every Nov 19 number in the README is recomputed from the committed results tables."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")


def test_cost_sweep_cells():
    s = pd.read_csv(ROOT / "results" / "nov19_carry_costs.csv")
    for r in s.itertuples():
        assert f"{r.off:.2f} / **{r.carry:.2f}** ({r.wins}/{r.losses} [" in README, r


def test_bar0_rows_and_total():
    t = pd.read_csv(ROOT / "results" / "nov19_carry_bar0.csv")
    for r in t.itertuples():
        assert f"| {r.half_life:g}, {r.scenario} | {r.carried_blocks} of {r.all_blocks} | {r.hold} | {r.exit} (" in README
    assert f"None of the {t.carried_blocks.sum()} carried blocks flips" in README


def test_stop_rows():
    st = pd.read_csv(ROOT / "results" / "nov19_carry_stops.csv")
    for (hl, sc), g in st.groupby(["half_life", "scenario"], sort=False):
        g = g.set_index("book")
        h = g.loc["carry + CUSUM halt"]
        assert (f"| {hl:g}, {sc} | {g.loc['carry', 'median_sharpe']:.2f} |" in README
                and f"{h.median_sharpe:.2f} ({int(h.wins)}/{int(h.losses)}) |" in README)


def test_gain_ranges():
    s = pd.read_csv(ROOT / "results" / "nov19_carry_costs.csv")
    nb = s[s.scenario == "no break"].assign(gain=lambda d: d.carry - d.off)
    g25, g10 = nb[nb.cost_bps == 2.5].gain, nb[nb.cost_bps == 10].gain
    assert f"{g25.min():+.2f} to {g25.max():+.2f} Sharpe at 2.5 bps" in README
    assert f"{g10.min():+.2f} to {g10.max():+.2f} at 10 bps" in README
