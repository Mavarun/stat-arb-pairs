"""Nov 8 slice: does a half-life-calibrated gate make the break re-test usable on slow pairs?

Synthetic, seeded, offline. Run: ``python scripts/run_nov08_experiments.py`` (~6 min).
Static OLS book, 252/63 walk-forward, every traded bar out-of-sample, net of 10 bps per
leg. Nov 3 ran the break experiment at half-life 5 only and found the fixed p < 0.01
gate gates 75-92% of healthy half-life 10-20 blocks. Here the same three scenarios
(no break, random-walk break, 4-sigma level shift) run at half-lives 5, 10 and 20.

Fixed before the first run: books below, quantiles {0.95, 0.90}, 20 new seeds
(1200-1219), break on bar 756 (first bar of a block, clean formation) and bar 786
(30 bars into a block); the no-break path is the same for both, so it is run once. Paired sign tests vs ``off`` and vs the fixed 0.01 gate, with a
Holm adjustment over every comparison printed (a family of all rows in the table).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from statarb.formation import formation_trading_backtest
from statarb.synth import make_broken_pair

SEEDS = range(1200, 1220)
N = 1260
HALF_LIVES = (5.0, 10.0, 20.0)
BREAKS = (("aligned (756)", 756), ("mid-block (786)", 786))
SCENARIOS = (("no break", "none"), ("random-walk break", "random_walk"), ("4-sigma level shift", "level_shift"))
ORACLE = "oracle"
BOOKS = {
    "off": {},
    "EG gate 0.01": {"gate_p": 0.01},
    "EG gate 0.05": {"gate_p": 0.05},
    "calibrated q0.95": {"gate_quantile": 0.95},
    "calibrated q0.90": {"gate_quantile": 0.90},
    "calibrated q0.95 oracle": {"gate_quantile": 0.95, "gate_ref_half_life": ORACLE},
}
REFERENCES = ("off", "EG gate 0.01")


def run(seeds=SEEDS, half_lives=HALF_LIVES, breaks=BREAKS, scenarios=SCENARIOS, books=None) -> pd.DataFrame:
    books = BOOKS if books is None else books
    rows = []
    for hl in half_lives:
        for btag, break_at in breaks:
            for label, kind in scenarios:
                if kind == "none" and btag != breaks[0][0]:
                    continue  # the unbroken path does not depend on the break bar: run it once
                for s in seeds:
                    p = make_broken_pair(N, break_at=break_at, kind=kind, half_life=hl, shift_sigmas=4.0, seed=s)
                    cut = p.y.index[break_at]
                    for book, kw in books.items():
                        kw = {k: (hl if v == ORACLE else v) for k, v in kw.items()}
                        r = formation_trading_backtest(p.y, p.x, hedge="ols", **kw, carry_over=False)
                        w = r.windows
                        post_w = w["trade_end"] >= cut
                        post = r.returns.loc[cut:]
                        rows.append({
                            "half_life": hl, "break": btag, "scenario": label, "seed": s, "book": book,
                            "sharpe": r.sharpe,
                            "total_return": r.total_return,
                            "post_return": float((1 + post).prod() - 1),
                            "max_dd": r.max_drawdown,
                            "cost": r.cost_paid,
                            "pre_gated_frac": float(w["gated"][~post_w].mean()),
                            "post_gated_frac": float(w["gated"][post_w].mean()),
                            "gate_level": float(w["gate_level"].iloc[0]),
                        })
    return pd.DataFrame(rows)


def _holm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    q = p[ok]
    order = np.argsort(q)
    m = len(q)
    adj = np.maximum.accumulate((m - np.arange(m)) * q[order]).clip(max=1.0)
    res = np.empty(m)
    res[order] = adj
    out[ok] = res
    return out


def summarise(df: pd.DataFrame, references=REFERENCES) -> pd.DataFrame:
    keys = ["half_life", "break", "scenario"]
    out = []
    for (hl, br, sc), g in df.groupby(keys, sort=False):
        piv = g.pivot(index="seed", columns="book", values="sharpe")
        for book, gb in g.groupby("book", sort=False):
            row = {
                "half_life": hl, "break": br, "scenario": sc, "book": book,
                "median_sharpe": gb.sharpe.median(),
                "median_total_return": gb.total_return.median(),
                "median_post_break_return": gb.post_return.median(),
                "median_max_dd": gb.max_dd.median(),
                "mean_cost": gb.cost.mean(),
                "mean_pre_gated_frac": gb.pre_gated_frac.mean(),
                "mean_post_gated_frac": gb.post_gated_frac.mean(),
                "median_gate_level": gb.gate_level.median(),
            }
            for ref in references:
                tag = "off" if ref == "off" else "fixed"
                if book == ref or ref not in piv:
                    row[f"wins_vs_{tag}"] = row[f"losses_vs_{tag}"] = row[f"sign_p_vs_{tag}"] = np.nan
                    continue
                d = piv[book] - piv[ref]
                w, lo = int((d > 1e-12).sum()), int((d < -1e-12).sum())
                row[f"wins_vs_{tag}"], row[f"losses_vs_{tag}"] = w, lo
                row[f"sign_p_vs_{tag}"] = binomtest(w, w + lo, 0.5).pvalue if w + lo else 1.0
            out.append(row)
    s = pd.DataFrame(out)
    for tag in ("off", "fixed"):
        col = f"sign_p_vs_{tag}"
        if col in s:
            s[f"holm_p_vs_{tag}"] = _holm(s[col].to_numpy())
    return s


def main() -> None:
    s = summarise(run())
    pd.set_option("display.width", 320)
    pd.set_option("display.max_columns", 40)
    cols = ["book", "median_sharpe", "wins_vs_off", "losses_vs_off", "sign_p_vs_off", "holm_p_vs_off",
            "wins_vs_fixed", "losses_vs_fixed", "sign_p_vs_fixed", "holm_p_vs_fixed",
            "median_post_break_return", "median_max_dd", "mean_cost", "mean_pre_gated_frac",
            "mean_post_gated_frac", "median_gate_level"]
    for (hl, br, sc), g in s.groupby(["half_life", "break", "scenario"], sort=False):
        print(f"\n== half-life {hl:g} | break {br} | {sc}")
        print(g[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
