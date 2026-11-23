"""Nov 24: does a hedge filter that models the drift directly pay under a trending ratio?

Synthetic, seeded, offline. Run: ``python scripts/run_nov24_experiments.py`` (~20 s) and
``--posthoc``. Same pairs and book as Nov 22 (``make_beta_trend_pair``, N 1260, trend from bar
252, 252-bar formation, 63-bar blocks, carry on, 10 bps per leg, Sharpe net of costs on the
out-of-sample blocks).

Why: Nov 22 left "a trend-aware filter (a local-linear-trend state)" untried. ``kalman_llt``
adds a per-bar drift state to the hedge ratio, so a linear trend is a constant it can learn.

Dev check (seeds 1-2, half-lives 5 and 20, drifts 0 and 5%/yr; never used for a decision):
``q_drift`` in {0, 1e-12, 1e-11} gave the same Sharpe to two decimals in all eight cells,
because the filter is re-started from the formation OLS every block and 63 bars are too few to
move the drift state far from its prior. So one LLT arm is tested (``q_drift`` 1e-12).

Fixed before the first run on any seed below:

* arms ``ols`` (default book), ``kalman`` (defaults) and ``kalman_llt`` (q_beta 1e-8,
  q_drift 1e-12, drift prior sd = 10%/yr of a 1.5 ratio);
* drifts {0, 2%, 5%} per year; half-lives {5, 10, 20}; 20 new seeds 1900-1919.

Decision rules (Holm over all 18 paired sign tests: 2 comparisons x 9 rows):

* H1: ``kalman_llt`` beats ``ols`` on median Sharpe in >= 2 of the 3 half-life rows at 5%/yr,
  with Holm p < 0.05 in those rows.
* H2: ``kalman_llt`` beats the default ``kalman`` by the same test (the drift state adds
  something over a random walk).
* H3: if H1 passes and ``kalman_llt`` loses to ``ols`` on median in all 3 zero-drift rows, it
  is a trend-only option, not a default.

No default changes either way. Expectation (from the dev seeds only): H1 passes, H2 fails, H3
calls it trend-only.
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
from run_nov11_experiments import holm  # noqa: E402

SEEDS = range(1900, 1920)
DEV_SEEDS = (1, 2)
N = 1260
TREND_FROM = 252
HALF_LIVES = (5.0, 10.0, 20.0)
DRIFTS = (0.0, 0.02, 0.05)
DECISION_DRIFT = 0.05
ARMS = {
    "ols": {"hedge": "ols"},
    "kalman": {"hedge": "kalman"},
    "kalman_llt": {"hedge": "kalman_llt", "q_beta": 1e-8, "q_drift": 1e-12},
}
COMPARISONS = (("ols", "kalman_llt"), ("kalman", "kalman_llt"))
RESULTS = B.RESULTS


def run(seeds=SEEDS, half_lives=HALF_LIVES, drifts=DRIFTS) -> pd.DataFrame:
    rows = []
    for hl in half_lives:
        for d in drifts:
            for s in seeds:
                p = make_beta_trend_pair(N, trend_from=TREND_FROM, drift_per_year=d, half_life=hl, seed=s)
                for arm, kw in ARMS.items():
                    r = formation_trading_backtest(p.y, p.x, **kw)
                    rows.append({"half_life": hl, "drift": d, "seed": s, "arm": arm, "sharpe": r.sharpe,
                                 "cost": r.cost_paid, "trades": r.trades})
    return pd.DataFrame(rows)


def table(df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for base, ch in COMPARISONS:
        t = B.paired(df.rename(columns={"drift": "jump"}), base, ch).rename(
            columns={"jump": "drift", base: "base", ch: "challenger"})
        t.insert(0, "vs", base)
        parts.append(t.drop(columns="holm_p"))
    out = pd.concat(parts, ignore_index=True)
    out["holm_p"] = holm(out["sign_p"].to_numpy())
    return out


def _passes(t: pd.DataFrame, base: str) -> bool:
    r = t[(t.vs == base) & (t.drift == DECISION_DRIFT)]
    return int(((r.challenger > r.base) & (r.holm_p < 0.05)).sum()) >= 2


def decide(df: pd.DataFrame) -> dict:
    t = table(df)
    h1, h2 = _passes(t, "ols"), _passes(t, "kalman")
    z = t[(t.vs == "ols") & (t.drift == 0.0)]
    h3 = bool(h1 and (z.challenger < z.base).all())
    return {"H1": h1, "H2": h2, "trend_only": h3, "table": t}


def main() -> None:
    df = run()
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "nov24_llt_hedge.csv", index=False)
    d = decide(df)
    print(d["table"].round(3).to_string(index=False))
    print({k: v for k, v in d.items() if k != "table"})


if __name__ == "__main__":
    main()
