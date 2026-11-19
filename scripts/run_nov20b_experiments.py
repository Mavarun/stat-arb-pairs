"""Nov 20 (second slice): a Kalman hedge whose hedge ratio drifts 100x slower (q_beta = 1e-8).

Synthetic, seeded, offline. Run: ``python scripts/run_nov20b_experiments.py`` (~1 min) and
``--posthoc``. Same pairs as the Nov 20 slice (``make_beta_jump_pair``, jump at bar 786),
252-bar formation, 63-bar blocks, 10 bps per leg, carry_over on (the default).

Why: the Kalman books lose money in 23 of 24 Nov 20 book-rows. The first guess, written into a
draft of this script, was that the random-walk *intercept* (q_alpha = 1e-4) follows the spread so
the innovation is white noise. Dev seeds 1-3 (never used for a decision; no jump, half-life 10)
refuted it before any evaluation seed was run: fixing the intercept moves the innovation z by at
most 0.011 and z has lag-1 autocorrelation ~0.94. What does matter is the hedge ratio's drift:
the trading-block z's mean correlation with the planted spread is 0.72-0.89 under the default
q_beta 1e-6, 0.94-0.96 under q_beta 1e-8 and 0.98 for the OLS book, whatever q_alpha is. The
filter re-weights alpha and beta inside the block and soaks up part of the spread.

Fixed after that dev check and before the first run on any seed below: arms ``ols``,
``kalman`` (defaults: q_alpha 1e-4, q_beta 1e-6) and ``kalman_qb8`` (q_beta 1e-8); half-lives
{5, 10, 20}; jumps {0, 2%}; 20 new seeds 1600-1619 (earlier scripts used 1000-1159 and
1200-1519).

Decision rules (fixed):

* H1 (mechanism): ``kalman_qb8`` beats ``kalman`` on median Sharpe in all 6 rows, with a
  Holm-adjusted paired sign test p < 0.05 in at least 4 of them.
* H2 (does it earn a place): ``kalman_qb8`` beats ``ols`` on median Sharpe in all three 2%-jump
  rows. If H2 fails the Kalman book stays context only. Q_BETA's default is not changed by this
  slice either way (that would alter every earlier Kalman result).

Expectation written before the run: H1 passes; H2 fails, because a 100x slower hedge is close to
the static OLS hedge, which re-fits every block anyway.
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

SEEDS = range(1600, 1620)
DEV_SEEDS = (1, 2)
N = 1260
BREAK_AT = 786
HALF_LIVES = (5.0, 10.0, 20.0)
JUMPS = (0.0, 0.02)
ARMS = {"ols": {"hedge": "ols"}, "kalman": {"hedge": "kalman"}, "kalman_qb8": {"hedge": "kalman", "q_beta": 1e-8}}
RESULTS = Path(__file__).resolve().parents[1] / "results"


def run(seeds=SEEDS, half_lives=HALF_LIVES, jumps=JUMPS) -> pd.DataFrame:
    rows = []
    for hl in half_lives:
        for j in jumps:
            for s in seeds:
                p = make_beta_jump_pair(N, break_at=BREAK_AT, jump=j, half_life=hl, seed=s)
                for arm, kw in ARMS.items():
                    r = formation_trading_backtest(p.y, p.x, **kw)
                    rows.append({"half_life": hl, "jump": j, "seed": s, "arm": arm, "sharpe": r.sharpe,
                                 "cost": r.cost_paid, "trades": r.trades})
    return pd.DataFrame(rows)


def paired(df: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    """Per (half-life, jump): medians of arms a and b, wins/losses of b over a, sign test, Holm."""
    out = []
    for (hl, j), g in df.groupby(["half_life", "jump"], sort=False):
        piv = g.pivot(index="seed", columns="arm", values="sharpe")
        d = piv[b] - piv[a]
        w, lo = int((d > 1e-12).sum()), int((d < -1e-12).sum())
        out.append({"half_life": hl, "jump": j, a: piv[a].median(), b: piv[b].median(), "wins": w, "losses": lo,
                    "sign_p": binomtest(w, w + lo, 0.5).pvalue if w + lo else 1.0})
    s = pd.DataFrame(out)
    s["holm_p"] = holm(s["sign_p"].to_numpy())
    return s


def decide(df: pd.DataFrame) -> dict:
    h1t, h2t = paired(df, "kalman", "kalman_qb8"), paired(df, "ols", "kalman_qb8")
    h1 = bool((h1t.kalman_qb8 > h1t.kalman).all() and (h1t.holm_p < 0.05).sum() >= 4)
    j2 = h2t[h2t.jump == 0.02]
    h2 = bool((j2.kalman_qb8 > j2.ols).all())
    return {"H1": h1, "H2": h2, "h1_table": h1t, "h2_table": h2t}


def main() -> None:
    df = run()
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "nov20b_kalman_qb8.csv", index=False)
    d = decide(df)
    print(d["h1_table"].round(3).to_string(index=False))
    print(d["h2_table"].round(3).to_string(index=False))
    print({"H1": d["H1"], "H2": d["H2"]})


def tracking(seeds=SEEDS, half_lives=HALF_LIVES, jumps=JUMPS) -> pd.DataFrame:
    """POST-HOC (written after the decision; decides nothing): per arm, the mean over blocks of
    corr(trading-block z, planted spread) and the block z's std, the dev-seed mechanism re-read
    on the evaluation seeds."""
    from statarb.formation import _window_signal
    from statarb.kalman import Q_ALPHA, Q_BETA

    out = []
    for hl in half_lives:
        for j in jumps:
            for s in seeds:
                p = make_beta_jump_pair(N, break_at=BREAK_AT, jump=j, half_life=hl, seed=s)
                y, x, sp = p.y.to_numpy(), p.x.to_numpy(), p.spread.to_numpy()
                for arm, kw in ARMS.items():
                    cs, sds, st = [], [], 0
                    while st + 252 + 63 <= N:
                        f, t = slice(st, st + 252), slice(st + 252, st + 315)
                        z, _, _ = _window_signal(y[f], x[f], y[t], x[t], kw["hedge"], kw.get("q_beta", Q_BETA),
                                                 kw.get("q_alpha", Q_ALPHA))
                        cs.append(np.corrcoef(z, sp[t])[0, 1])
                        sds.append(z.std())
                        st += 63
                    out.append({"half_life": hl, "jump": j, "seed": s, "arm": arm,
                                "corr_true": float(np.mean(cs)), "z_sd": float(np.mean(sds))})
    df = pd.DataFrame(out)
    return df.groupby(["half_life", "jump", "arm"], sort=False)[["corr_true", "z_sd"]].median().reset_index()


if __name__ == "__main__":
    if "--posthoc" in sys.argv:
        t = tracking()
        t.round(4).to_csv(RESULTS / "nov20b_tracking_posthoc.csv", index=False)
        print(t.round(3).to_string(index=False))
    else:
        main()
