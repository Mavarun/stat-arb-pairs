"""Nov 20 slice: carried positions when the hedge ratio jumps.

Synthetic, seeded, offline. Run: ``python scripts/run_nov20_experiments.py`` (~1 min).
Static OLS book, 252-bar formation, 63-bar blocks, 10 bps per leg, every traded bar OOS.

Nov 19 found that no carried position ever flips at the re-fit (0 of 688), and said that was
a property of a simulator whose hedge ratio never moves. This slice makes it move: the pair
keeps the same mean-reverting spread but its hedge ratio jumps by ``jump`` (relative) at bar
786 (``make_beta_jump_pair``), so the formation windows that straddle the jump mis-fit, and
the re-fits after it move z.

Fixed before the first run: books ``off`` (forced flat) and ``carry``; jumps {0, 0.5%, 1%, 2%};
half-lives {5, 10, 20}; 20 new seeds (1500-1519); bar-0 hold / exit / flip counts for the
carry book. A Kalman-hedge pair of books is reported for context only (it is not decided on).

Decision rule (also fixed): ``carry_over`` stays the default with no caveat only if it beats
``off`` on median Sharpe in every (half-life, jump) row of the OLS book. If it loses a row,
the default stays (the jump is a stress, not the base case), but the README must name the
smallest jump at which carry stops paying. Holm-adjusted paired sign tests over all rows are
reported. Expectation written before the run: flips appear at 2%, and carry loses at 2% for
the slow pair.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from statarb.formation import formation_trading_backtest  # noqa: E402
from statarb.synth import make_beta_jump_pair  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_nov11_experiments import holm  # noqa: E402

SEEDS = range(1500, 1520)
N = 1260
BREAK_AT = 786
HALF_LIVES = (5.0, 10.0, 20.0)
JUMPS = (0.0, 0.005, 0.01, 0.02)
HEDGES = ("ols", "kalman")
BOOKS = {"off": {"carry_over": False}, "carry": {"carry_over": True}}


def run(seeds=SEEDS, half_lives=HALF_LIVES, jumps=JUMPS, hedges=HEDGES) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, bar0 = [], []
    for hl in half_lives:
        for j in jumps:
            for s in seeds:
                p = make_beta_jump_pair(N, break_at=BREAK_AT, jump=j, half_life=hl, seed=s)
                for hedge in hedges:
                    for book, kw in BOOKS.items():
                        r = formation_trading_backtest(p.y, p.x, hedge=hedge, **kw)
                        rows.append({"hedge": hedge, "half_life": hl, "jump": j, "seed": s, "book": book,
                                     "sharpe": r.sharpe, "cost": r.cost_paid, "trades": r.trades})
                        if book == "carry":
                            for a in r.windows.carry_bar0:
                                bar0.append({"hedge": hedge, "half_life": hl, "jump": j, "seed": s,
                                             "action": a or "none"})
    return pd.DataFrame(rows), pd.DataFrame(bar0)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (hedge, hl, j), g in df.groupby(["hedge", "half_life", "jump"], sort=False):
        piv = g.pivot(index="seed", columns="book", values="sharpe")
        d = piv["carry"] - piv["off"]
        w, lo = int((d > 1e-12).sum()), int((d < -1e-12).sum())
        out.append({"hedge": hedge, "half_life": hl, "jump": j, "off": piv["off"].median(),
                    "carry": piv["carry"].median(), "wins": w, "losses": lo,
                    "sign_p": binomtest(w, w + lo, 0.5).pvalue if w + lo else 1.0})
    s = pd.DataFrame(out)
    s["holm_p"] = holm(s["sign_p"].to_numpy())
    return s


def bar0_counts(b: pd.DataFrame) -> pd.DataFrame:
    carried = b[b.action != "none"]
    t = carried.groupby(["hedge", "half_life", "jump"]).action.value_counts().unstack(fill_value=0)
    for col in ("hold", "exit", "flip"):
        if col not in t:
            t[col] = 0
    t["carried_blocks"] = t[["hold", "exit", "flip"]].sum(axis=1)
    return t[["carried_blocks", "hold", "exit", "flip"]].reset_index()


def decide(s: pd.DataFrame) -> dict:
    ols = s[s.hedge == "ols"]
    lost = ols[ols.carry <= ols.off]
    return {"carry_stays_without_caveat": bool(lost.empty),
            "rows_lost": [(float(r.half_life), float(r.jump)) for r in lost.itertuples()],
            "smallest_losing_jump": float(lost.jump.min()) if not lost.empty else None}


def main(out: Path | None = None) -> dict:
    df, b0 = run()
    s, t = summarise(df), bar0_counts(b0)
    out = out or Path(__file__).resolve().parents[1] / "results"
    out.mkdir(exist_ok=True)
    s.to_csv(out / "nov20_beta_jump.csv", index=False, float_format="%.6f")
    t.to_csv(out / "nov20_beta_jump_bar0.csv", index=False)
    pd.set_option("display.width", 200)
    print(s.round(3).to_string(index=False))
    print(t.to_string(index=False))
    d = decide(s)
    print("decision:", d)
    return d


if __name__ == "__main__" and "--posthoc" not in sys.argv:
    main()


# --- Post-hoc (after the first run; decides nothing) -------------------------------------------
# The written expectation failed: 0 flips at every jump, and carry wins every row. This reads why,
# straight from the formation fits: for each 252-bar formation window, the OLS hedge error against
# the true hedge ratio on the block's first bar, and the residual std in units of the planted
# spread std. A flip needs the re-fit to move z by more than z_enter + z_exit = 2.5 sd; a residual
# std inflated by the jump shrinks every z, which works against a flip.
FORMATION, TRADING = 252, 63


def posthoc(seeds=SEEDS, half_lives=HALF_LIVES, jumps=JUMPS) -> pd.DataFrame:
    rows = []
    for hl in half_lives:
        for j in jumps:
            for s in seeds:
                p = make_beta_jump_pair(N, break_at=BREAK_AT, jump=j, half_life=hl, seed=s)
                y, x, bp = p.y.to_numpy(), p.x.to_numpy(), p.beta_path.to_numpy()
                for start in range(FORMATION, N - TRADING + 1, TRADING):
                    w = slice(start - FORMATION, start)
                    X = np.column_stack([np.ones(FORMATION), x[w]])
                    coef, *_ = np.linalg.lstsq(X, y[w], rcond=None)
                    resid = y[w] - X @ coef
                    rows.append({"half_life": hl, "jump": j, "seed": s, "block_start": start,
                                 "window": "straddles jump" if start - FORMATION < BREAK_AT < start else
                                           ("after jump" if start - FORMATION >= BREAK_AT else "before jump"),
                                 "hedge_err": abs(coef[1] - bp[start]) / bp[start],
                                 "resid_sd_ratio": resid.std(ddof=2) / p.sigma_spread})
    d = pd.DataFrame(rows)
    return (d.groupby(["half_life", "jump", "window"])
             .agg(windows=("hedge_err", "size"), hedge_err_median=("hedge_err", "median"),
                  resid_sd_ratio_median=("resid_sd_ratio", "median"))
             .reset_index())


def main_posthoc(out: Path | None = None) -> pd.DataFrame:
    t = posthoc()
    out = out or Path(__file__).resolve().parents[1] / "results"
    t.to_csv(out / "nov20_beta_jump_posthoc.csv", index=False, float_format="%.6f")
    print(t.round(4).to_string(index=False))
    return t


if __name__ == "__main__" and "--posthoc" in sys.argv:
    main_posthoc()
