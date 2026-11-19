"""Nov 19 slice: does carry's gain survive cheaper execution, and what happens at the re-fit?

Synthetic, seeded, offline. Run: ``python scripts/run_nov19_experiments.py`` (~2 min).
Static OLS book, 252-bar formation, 63-bar blocks, every traded bar out-of-sample.

Nov 11 shipped ``carry_over=True`` at one cost level (10 bps per leg) and listed two open
questions: carry's gain is partly a cost saving, so it should shrink with cheaper execution;
and a carried position crosses a re-fit, where the new block's z could flip it at once.

Fixed before the first run: books ``off`` (forced flat) and ``carry``; cost per leg in
{2.5, 5, 10, 20} bps (split evenly between spread and slippage); half-lives {5, 10, 20};
scenarios no break / random-walk break on bar 786; 20 new seeds (1400-1419). The re-fit
reading uses ``windows["carry_bar0"]`` (hold / exit / flip on a carried block's first bar).

Decision rule (also fixed): ``carry_over`` stays the default only if, at **every** cost level,
it beats ``off`` on median Sharpe in every half-life's no-break row. If it loses any such row
the default stays (the repo's cost level is 10 bps), but the README must state the cost level
below which carry stops paying. Holm-adjusted paired sign tests over every comparison are
reported.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from statarb.formation import formation_trading_backtest  # noqa: E402
from statarb.synth import make_broken_pair  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_nov11_experiments import holm  # noqa: E402

SEEDS = range(1400, 1420)
N = 1260
HALF_LIVES = (5.0, 10.0, 20.0)
COSTS = (2.5, 5.0, 10.0, 20.0)
SCENARIOS = (("no break", "none", 786), ("random walk (786)", "random_walk", 786))
BOOKS = {"off": {"carry_over": False}, "carry": {"carry_over": True}}


def run(seeds=SEEDS, half_lives=HALF_LIVES, costs=COSTS, scenarios=SCENARIOS,
        bar0_cost: float = 10.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, bar0 = [], []
    for hl in half_lives:
        for label, kind, break_at in scenarios:
            for s in seeds:
                p = make_broken_pair(N, break_at=break_at, kind=kind, half_life=hl, shift_sigmas=4.0, seed=s)
                for c in costs:
                    for book, kw in BOOKS.items():
                        r = formation_trading_backtest(p.y, p.x, hedge="ols", spread_bps=c / 2, slippage_bps=c / 2, **kw)
                        rows.append({"half_life": hl, "scenario": label, "seed": s, "cost_bps": c, "book": book,
                                     "sharpe": r.sharpe, "total_return": r.total_return, "cost": r.cost_paid,
                                     "trades": r.trades})
                        if book == "carry" and c == bar0_cost:
                            for a in r.windows.carry_bar0:
                                bar0.append({"half_life": hl, "scenario": label, "seed": s, "action": a or "none"})
    return pd.DataFrame(rows), pd.DataFrame(bar0)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (hl, sc, c), g in df.groupby(["half_life", "scenario", "cost_bps"], sort=False):
        piv = g.pivot(index="seed", columns="book", values="sharpe")
        d = piv["carry"] - piv["off"]
        w, lo = int((d > 1e-12).sum()), int((d < -1e-12).sum())
        out.append({"half_life": hl, "scenario": sc, "cost_bps": c, "off": piv["off"].median(),
                    "carry": piv["carry"].median(), "wins": w, "losses": lo,
                    "sign_p": binomtest(w, w + lo, 0.5).pvalue if w + lo else 1.0,
                    "cost_saved": float(g[g.book == "off"].cost.mean() - g[g.book == "carry"].cost.mean())})
    s = pd.DataFrame(out)
    s["holm_p"] = holm(s["sign_p"].to_numpy())
    return s


def decide(s: pd.DataFrame) -> dict:
    nb = s[s.scenario == "no break"]
    lost = nb[nb.carry <= nb.off]
    return {"carry_stays_without_caveat": bool(lost.empty),
            "rows_lost": [(float(r.half_life), float(r.cost_bps)) for r in lost.itertuples()]}


def bar0_shares(b: pd.DataFrame) -> pd.DataFrame:
    carried = b[b.action != "none"]
    t = carried.groupby(["half_life", "scenario"]).action.value_counts(normalize=False).unstack(fill_value=0)
    for col in ("hold", "exit", "flip"):
        if col not in t:
            t[col] = 0
    t["carried_blocks"] = t[["hold", "exit", "flip"]].sum(axis=1)
    t["all_blocks"] = b.groupby(["half_life", "scenario"]).size()
    return t[["all_blocks", "carried_blocks", "hold", "exit", "flip"]].reset_index()


def main(out: Path | None = None) -> dict:
    df, b0 = run()
    s, t = summarise(df), bar0_shares(b0)
    out = out or Path(__file__).resolve().parents[1] / "results"
    out.mkdir(exist_ok=True)
    s.to_csv(out / "nov19_carry_costs.csv", index=False, float_format="%.6f")
    t.to_csv(out / "nov19_carry_bar0.csv", index=False)
    pd.set_option("display.width", 200)
    print(s.round(3).to_string(index=False))
    print(t.to_string(index=False))
    d = decide(s)
    print("decision:", d)
    return d


if __name__ == "__main__":
    main()
