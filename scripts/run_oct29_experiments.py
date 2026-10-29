"""Reproduce the README numbers for the Oct 29 slice: stops on a pair that breaks.

Synthetic, seeded, offline. Run: ``python scripts/run_oct29_experiments.py``.
Static OLS book, 252-bar formation / 63-bar trading walk-forward, net of 10 bps
per leg per unit change (5 spread + 5 slippage). Every traded bar is
out-of-sample: hedge, z statistics, half-life and the auto time-stop are fitted
on the formation window only. Stop levels {3, 4} and 3 x HL were fixed before
the first run; nothing is tuned on these seeds.

The break is planted at bar 756 (after three years) of 1,260 bars (HL = 5), so
the first two trading years are clean and the last two are post-break.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from statarb.formation import formation_trading_backtest
from statarb.synth import make_broken_pair

SEEDS = range(900, 912)  # 12 seeds, new for this slice
N = 1260
BREAK_AT = 756
HALF_LIFE = 5.0
SCENARIOS = (("no break", "none"), ("random-walk break", "random_walk"), ("4-sigma level shift", "level_shift"))
BOOKS = {
    "off": {},
    "time stop 3xHL": {"use_time_stop": True, "time_stop_multiple": 3.0},
    "time stop auto": {"use_time_stop": True, "time_stop_multiple": "auto"},
    "z stop 3": {"z_stop": 3.0},
    "z stop 4": {"z_stop": 4.0},
    "z stop 4 + 3xHL": {"z_stop": 4.0, "use_time_stop": True, "time_stop_multiple": 3.0},
}


def run() -> pd.DataFrame:
    rows = []
    for label, kind in SCENARIOS:
        for s in SEEDS:
            p = make_broken_pair(N, break_at=BREAK_AT, kind=kind, half_life=HALF_LIFE, shift_sigmas=4.0, seed=s)
            cut = p.y.index[BREAK_AT]
            for book, kw in BOOKS.items():
                r = formation_trading_backtest(p.y, p.x, hedge="ols", **kw)
                post = r.returns.loc[cut:]
                w = r.windows
                rows.append({
                    "scenario": label, "seed": s, "book": book,
                    "sharpe": r.sharpe,
                    "post_return": float((1 + post).prod() - 1),
                    "worst_window": float(w["net_return"].min()),
                    "max_dd": r.max_drawdown,
                    "cost": r.cost_paid,
                    "trades": r.trades,
                    "z_stops": int(w["z_stops"].sum()),
                })
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    off = df[df.book == "off"].set_index(["scenario", "seed"])
    out = []
    for (sc, book), g in df.groupby(["scenario", "book"], sort=False):
        g = g.set_index(["scenario", "seed"])
        out.append({
            "scenario": sc, "book": book,
            "median_sharpe": g.sharpe.median(),
            "sharpe_wins_vs_off": int((g.sharpe > off.loc[g.index, "sharpe"] + 1e-12).sum()) if book != "off" else np.nan,
            "median_post_break_return": g.post_return.median(),
            "median_worst_window": g.worst_window.median(),
            "median_max_dd": g.max_dd.median(),
            "mean_trades": g.trades.mean(),
            "mean_cost": g.cost.mean(),
            "mean_z_stops": g.z_stops.mean(),
        })
    return pd.DataFrame(out)


def main() -> None:
    df = run()
    s = summarise(df)
    pd.set_option("display.width", 200)
    print(s.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
