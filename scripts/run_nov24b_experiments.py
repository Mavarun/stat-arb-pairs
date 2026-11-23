"""Nov 24 (second slice): how much of the default book's Sharpe is the same-close execution assumption?

Synthetic, seeded, offline. Run: ``python scripts/run_nov24b_experiments.py`` (~15 s).

Why: every slice since Oct 5 lists "trade at the close after computing z from that close" as an
optimistic assumption (README, Assumptions) and never measures it. ``exec_delay=1`` executes the
decided book one close later, with the same costs.

Book: default ``formation_trading_backtest`` (OLS hedge, carry on, 252-bar formation, 63-bar
blocks, 10 bps per leg), on ``make_beta_trend_pair`` with no drift (a stable pair), N 1260.

Dev check (seeds 1-2 only, never used for a decision): median Sharpe delay 0 -> 1 was
1.62 -> 1.41 and 1.09 -> 0.52 at half-life 5, 0.87 -> 0.70 and 0.84 -> 0.70 at 10, and
0.25 -> 0.15 and 0.39 -> 0.48 at 20.

Fixed before the first run on any seed below: half-lives {5, 10, 20}; delays {0, 1}; 20 new seeds
2000-2019.

Decision rules (Holm over the 3 paired sign tests, delay 1 vs delay 0):

* H1 (the edge survives): median net Sharpe with ``exec_delay=1`` is > 0 in all 3 rows.
* H2 (the haircut is material): in >= 2 of 3 rows the delayed median is more than 25% below the
  same-close median, with delay worse on a Holm-adjusted sign test p < 0.05.

No default changes either way (``exec_delay`` stays 0 so every earlier slice reproduces). If H2
passes, the README states the measured haircut next to the same-close assumption and calls the
headline Sharpe figures an upper bound. Expectation (dev seeds only): H1 holds; H2 passes, with
the largest haircut at half-life 5.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from statarb.formation import formation_trading_backtest  # noqa: E402
from statarb.synth import make_beta_trend_pair  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_nov20b_experiments as B  # noqa: E402

SEEDS = range(2000, 2020)
DEV_SEEDS = (1, 2)
N = 1260
HALF_LIVES = (5.0, 10.0, 20.0)
DELAYS = (0, 1)
HAIRCUT = 0.25
RESULTS = B.RESULTS


def run(seeds=SEEDS, half_lives=HALF_LIVES) -> pd.DataFrame:
    rows = []
    for hl in half_lives:
        for s in seeds:
            p = make_beta_trend_pair(N, trend_from=252, drift_per_year=0.0, half_life=hl, seed=s)
            for d in DELAYS:
                r = formation_trading_backtest(p.y, p.x, exec_delay=d)
                rows.append({"half_life": hl, "jump": 0.0, "seed": s, "arm": f"delay{d}", "sharpe": r.sharpe,
                             "cost": r.cost_paid, "trades": r.trades, "gross": r.gross_return_sum})
    return pd.DataFrame(rows)


def decide(df: pd.DataFrame) -> dict:
    t = B.paired(df, "delay0", "delay1").drop(columns="jump")
    t["haircut"] = 1.0 - t.delay1 / t.delay0
    h1 = bool((t.delay1 > 0).all())
    material = (t.haircut > HAIRCUT) & (t.losses > t.wins) & (t.holm_p < 0.05)
    return {"H1": h1, "H2": bool(int(material.sum()) >= 2), "n_material": int(material.sum()), "table": t}


def main() -> None:
    df = run()
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "nov24b_exec_delay.csv", index=False)
    d = decide(df)
    print(d["table"].round(3).to_string(index=False))
    print({k: v for k, v in d.items() if k != "table"})


if __name__ == "__main__":
    main()
