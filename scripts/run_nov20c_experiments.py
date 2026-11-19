"""Nov 20 (third slice): does a moving hedge pay once the hedge ratio jumps by 5% or 10%?

Synthetic, seeded, offline. Run: ``python scripts/run_nov20c_experiments.py`` (~20 s).
Same pairs, books and costs as the Nov 20 second slice (``run_nov20b_experiments``): arms
``ols``, ``kalman`` (defaults) and ``kalman_qb8`` (q_beta 1e-8), carry on, 252-bar formation,
63-bar blocks, 10 bps per leg.

The second slice found OLS ahead of the slow Kalman hedge in every row up to a 2% jump, and its
tracking reading said a moving hedge only pays when beta moves more inside one block than a
2% jump does. This slice asks that directly with larger jumps.

Fixed before the first run: jumps {5%, 10%}; half-lives {5, 10, 20}; 20 new seeds 1700-1719.

Decision rule (fixed): the README may say "a moving hedge pays for large hedge-ratio jumps"
only if ``kalman_qb8`` beats ``ols`` on median Sharpe in at least 2 of the 3 half-life rows at
10%, with a Holm-adjusted paired sign test p < 0.05 in those rows (Holm over all 6 rows).
Otherwise it says the static OLS book, re-fitted every block, is still the better hedge up to
a 10% jump. No default changes either way.

Expectation written before the run: the rule fails at 5% and passes at 10% for half-lives 10
and 20 (where trades are held longest across the jump).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_nov20b_experiments as B  # noqa: E402

SEEDS = range(1700, 1720)
JUMPS = (0.05, 0.10)
HALF_LIVES = B.HALF_LIVES
RESULTS = B.RESULTS


def run(seeds=SEEDS, half_lives=HALF_LIVES, jumps=JUMPS) -> pd.DataFrame:
    return B.run(seeds=seeds, half_lives=half_lives, jumps=jumps)


def decide(df: pd.DataFrame) -> dict:
    t = B.paired(df, "ols", "kalman_qb8")
    ten = t[t.jump == 0.10]
    passing = ten[(ten.kalman_qb8 > ten.ols) & (ten.holm_p < 0.05)]
    return {"moving_hedge_pays": bool(len(passing) >= 2), "rows_passing": int(len(passing)), "table": t}


def main() -> None:
    df = run()
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "nov20c_big_jumps.csv", index=False)
    d = decide(df)
    print(d["table"].round(3).to_string(index=False))
    print({k: v for k, v in d.items() if k != "table"})


if __name__ == "__main__":
    main()
