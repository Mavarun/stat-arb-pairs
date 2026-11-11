"""Nov 3 slice: re-testing a pair after it breaks (CUSUM halt, Engle-Granger gate).

Synthetic, seeded, offline. Run: ``python scripts/run_nov03_experiments.py``.
Static OLS book, 252/63 walk-forward, every traded bar out-of-sample, net of 10 bps
per leg. Oct 29 left "stop trading the dead pair" untried: a z stop reacts to one
move, these react to evidence. Settings fixed before the first run: CUSUM k = 0.5,
h = 5 (null-calibrated, ``run_nov03_calibration.py``), EG gate p in {0.01, 0.05},
z stop 4 (the Oct 29 best), and 20 new seeds (940-959). The break is planted either
on the first bar of a trading block (756, clean formation window) or 30 bars into
one (786, formation window contaminated for the next four blocks).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from statarb.formation import formation_trading_backtest
from statarb.synth import make_broken_pair

SEEDS = range(940, 960)
N = 1260
HALF_LIFE = 5.0
BREAKS = (("aligned (756)", 756), ("mid-block (786)", 786))
SCENARIOS = (("no break", "none"), ("random-walk break", "random_walk"), ("4-sigma level shift", "level_shift"))
BOOKS = {
    "off": {},
    "z stop 4": {"z_stop": 4.0},
    "CUSUM halt": {"cusum_halt": True},
    "EG gate 0.01": {"gate_p": 0.01},
    "EG gate 0.05": {"gate_p": 0.05},
    "CUSUM + gate 0.05": {"cusum_halt": True, "gate_p": 0.05},
    "CUSUM + z stop 4": {"cusum_halt": True, "z_stop": 4.0},
}


def run(seeds=SEEDS, breaks=BREAKS, scenarios=SCENARIOS, books=None) -> pd.DataFrame:
    books = BOOKS if books is None else books
    rows = []
    for btag, break_at in breaks:
        for label, kind in scenarios:
            for s in seeds:
                p = make_broken_pair(N, break_at=break_at, kind=kind, half_life=HALF_LIFE, shift_sigmas=4.0, seed=s)
                cut = p.y.index[break_at]
                for book, kw in books.items():
                    r = formation_trading_backtest(p.y, p.x, hedge="ols", **kw, carry_over=False)
                    w = r.windows
                    post_w = w["trade_end"] >= cut
                    pre_w = ~post_w
                    post = r.returns.loc[cut:]
                    halted = w["halt_bar"].notna()
                    first = w.loc[post_w & halted]
                    delay = np.nan
                    if len(first):
                        delay = float(
                            (p.y.index.get_loc(first["trade_start"].iloc[0]) + first["halt_bar"].iloc[0]) - break_at
                        )
                    rows.append({
                        "break": btag, "scenario": label, "seed": s, "book": book,
                        "sharpe": r.sharpe,
                        "post_return": float((1 + post).prod() - 1),
                        "worst_window": float(w["net_return"].min()),
                        "max_dd": r.max_drawdown,
                        "cost": r.cost_paid,
                        "trades": r.trades,
                        "pre_halt_frac": float(halted[pre_w].mean()) if pre_w.any() else np.nan,
                        "pre_gated_frac": float(w["gated"][pre_w].mean()) if pre_w.any() else np.nan,
                        "post_flat_frac": float((r.positions.loc[cut:] == 0).mean()),
                        "delay": delay,
                    })
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    off = df[df.book == "off"].set_index(["break", "scenario", "seed"])
    out = []
    for (br, sc, book), g in df.groupby(["break", "scenario", "book"], sort=False):
        g = g.set_index(["break", "scenario", "seed"])
        wins = int((g.sharpe > off.loc[g.index, "sharpe"] + 1e-12).sum()) if book != "off" else 0
        losses = int((g.sharpe < off.loc[g.index, "sharpe"] - 1e-12).sum()) if book != "off" else 0
        sign_p = binomtest(wins, wins + losses, 0.5).pvalue if wins + losses > 0 else np.nan
        out.append({
            "break": br, "scenario": sc, "book": book,
            "median_sharpe": g.sharpe.median(),
            "sharpe_wins_vs_off": wins if book != "off" else np.nan,
            "sharpe_losses_vs_off": losses if book != "off" else np.nan,
            "sign_p": sign_p if book != "off" else np.nan,
            "median_post_break_return": g.post_return.median(),
            "median_worst_window": g.worst_window.median(),
            "median_max_dd": g.max_dd.median(),
            "mean_cost": g.cost.mean(),
            "mean_pre_halt_frac": g.pre_halt_frac.mean(),
            "mean_pre_gated_frac": g.pre_gated_frac.mean(),
            "mean_post_flat_frac": g.post_flat_frac.mean(),
            "median_detect_delay": g.delay.median(),
            "detected": int(g.delay.notna().sum()),
        })
    return pd.DataFrame(out)


def main() -> None:
    s = summarise(run())
    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", 30)
    for br, g in s.groupby("break", sort=False):
        print(f"\n== break {br}")
        print(g.drop(columns="break").round(3).to_string(index=False))


if __name__ == "__main__":
    main()
