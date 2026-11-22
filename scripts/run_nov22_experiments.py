"""Nov 22: does a moving hedge, or a shorter look-back, pay when the hedge ratio keeps trending?

Synthetic, seeded, offline. Run: ``python scripts/run_nov22_experiments.py`` (~30 s) and
``--posthoc``. Pairs from ``make_beta_trend_pair`` (N 1260, ratio flat for the first 252 bars,
then trending linearly at ``drift`` per year, relative to 1.5). Every arm trades the walk-forward
book of the earlier slices: 63-bar blocks, carry on (the default), 10 bps per leg, Sharpe net
of costs on the out-of-sample trading blocks only.

Why: the Nov 20 slices found the static OLS book, re-fitted every block, ahead of every Kalman
hedge for one-off jumps of up to 10%, and both named a *trending* ratio, where every re-fit lags,
as the untested case where a moving hedge could pay.

Dev check (seeds 1-2, half-lives 5 and 20, never used for a decision): a 10%/yr drift drives
every arm's Sharpe below zero at both half-lives, so it cannot separate the arms; 5%/yr does
not. The drifts below were chosen from that check, before any evaluation seed was run.

Fixed before the first run on any seed below:

* arms ``ols`` (252-bar formation, the default book), ``ols_f126`` (126-bar formation: half the
  look-back, so roughly half the lag, and twice the fit noise), ``kalman`` (defaults
  q_alpha 1e-4, q_beta 1e-6) and ``kalman_qb8`` (q_beta 1e-8);
* drifts {0, 2%, 5%} per year; half-lives {5, 10, 20}; 20 new seeds 1800-1819 (earlier scripts
  used 1000-1159 and 1200-1719).

Decision rules (fixed; Holm over all 27 paired sign tests: 3 challengers x 9 rows):

* H1 (a moving hedge pays under a trend): ``kalman`` or ``kalman_qb8`` beats ``ols`` on median
  Sharpe in >= 2 of the 3 half-life rows at 5%/yr, with Holm p < 0.05 in those rows.
* H2 (a shorter look-back pays under a trend): ``ols_f126`` beats ``ols`` by the same test.
* H3 (what a passing arm costs when nothing moves): for any arm that passes H1 or H2, if it loses
  to ``ols`` on median Sharpe in all 3 zero-drift rows the README calls it a trend-only option,
  not a default.

No default changes in this slice either way (``FORMATION`` and ``Q_BETA`` stay, so every earlier
number reproduces); the outcome is what the README may claim and recommend.

Expectation written before the run (informed by the two dev seeds only): H1 passes for the
default Kalman hedge at 5%/yr; H2 fails (the two dev seeds were split); every passing arm loses
at zero drift.
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

SEEDS = range(1800, 1820)
DEV_SEEDS = (1, 2)
N = 1260
TREND_FROM = 252
HALF_LIVES = (5.0, 10.0, 20.0)
DRIFTS = (0.0, 0.02, 0.05)
DECISION_DRIFT = 0.05
ARMS = {
    "ols": {"hedge": "ols"},
    "ols_f126": {"hedge": "ols", "formation": 126},
    "kalman": {"hedge": "kalman"},
    "kalman_qb8": {"hedge": "kalman", "q_beta": 1e-8},
}
CHALLENGERS = ("ols_f126", "kalman", "kalman_qb8")
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
    """Every challenger vs ``ols`` per (half-life, drift): medians, wins/losses, sign p, Holm over all."""
    parts = []
    for arm in CHALLENGERS:
        t = B.paired(df.rename(columns={"drift": "jump"}), "ols", arm).rename(columns={"jump": "drift", arm: "challenger"})
        t.insert(0, "arm", arm)
        parts.append(t.drop(columns="holm_p"))
    out = pd.concat(parts, ignore_index=True)
    out["holm_p"] = holm(out["sign_p"].to_numpy())
    return out


def _passes(t: pd.DataFrame, arm: str) -> bool:
    r = t[(t.arm == arm) & (t.drift == DECISION_DRIFT)]
    return int(((r.challenger > r.ols) & (r.holm_p < 0.05)).sum()) >= 2


def decide(df: pd.DataFrame) -> dict:
    t = table(df)
    h1 = {a: _passes(t, a) for a in ("kalman", "kalman_qb8")}
    h2 = _passes(t, "ols_f126")
    passing = [a for a, ok in {**h1, "ols_f126": h2}.items() if ok]
    zero = t[t.drift == 0.0]
    trend_only = {a: bool((zero[zero.arm == a].challenger < zero[zero.arm == a].ols).all()) for a in passing}
    return {"H1": any(h1.values()), "H1_by_arm": h1, "H2": h2, "trend_only": trend_only, "table": t}


def main() -> None:
    df = run()
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "nov22_beta_trend.csv", index=False)
    d = decide(df)
    print(d["table"].round(3).to_string(index=False))
    print({k: v for k, v in d.items() if k != "table"})


def tracking(seeds=SEEDS, half_lives=HALF_LIVES, drifts=DRIFTS) -> pd.DataFrame:
    """POST-HOC (written after the decisions; decides nothing): per arm, the median over seeds of
    the block-mean relative hedge error (block-mean hedge ratio vs the planted ratio, signed,
    so a positive value is a hedge that lags an upward trend), its absolute value, and the
    block-mean corr(trading-block z, planted spread)."""
    import numpy as np

    from statarb.formation import _window_signal
    from statarb.kalman import Q_ALPHA, Q_BETA

    out = []
    for hl in half_lives:
        for d in drifts:
            for s in seeds:
                p = make_beta_trend_pair(N, trend_from=TREND_FROM, drift_per_year=d, half_life=hl, seed=s)
                y, x, sp, bp = p.y.to_numpy(), p.x.to_numpy(), p.spread.to_numpy(), p.beta_path.to_numpy()
                for arm, kw in ARMS.items():
                    f = kw.get("formation", 252)
                    lag, ae, cs, st = [], [], [], 252 - f  # every arm trades the same blocks as ols
                    while st + f + 63 <= N:
                        fw, tw = slice(st, st + f), slice(st + f, st + f + 63)
                        z, b, _ = _window_signal(y[fw], x[fw], y[tw], x[tw], kw["hedge"], kw.get("q_beta", Q_BETA),
                                                 kw.get("q_alpha", Q_ALPHA))
                        e = float(np.mean(bp[tw] - b) / 1.5)
                        lag.append(e)
                        ae.append(float(np.mean(np.abs(bp[tw] - b)) / 1.5))
                        cs.append(np.corrcoef(z, sp[tw])[0, 1])
                        st += 63
                    out.append({"half_life": hl, "drift": d, "seed": s, "arm": arm, "hedge_lag": float(np.mean(lag)),
                                "hedge_abs_err": float(np.mean(ae)), "corr_true": float(np.mean(cs))})
    df = pd.DataFrame(out)
    return df.groupby(["half_life", "drift", "arm"], sort=False)[["hedge_lag", "hedge_abs_err", "corr_true"]].median().reset_index()


def common_span(seeds=SEEDS, half_lives=HALF_LIVES, drifts=DRIFTS) -> pd.DataFrame:
    """POST-HOC (decides nothing): the frozen ``ols_f126`` arm starts trading at bar 126, so its
    Sharpe includes two blocks (bars 126-251) that ``ols`` never trades, both before the trend
    starts. This re-scores ``ols_f126`` on bars 252+ only (the span every other arm trades) and
    re-runs its paired sign test against ``ols``, Holm over these 9 rows only."""
    from statarb.formation import _sharpe

    rows = []
    for hl in half_lives:
        for d in drifts:
            for s in seeds:
                p = make_beta_trend_pair(N, trend_from=TREND_FROM, drift_per_year=d, half_life=hl, seed=s)
                r = formation_trading_backtest(p.y, p.x, **ARMS["ols_f126"])
                rets = r.returns.dropna()
                rows.append({"half_life": hl, "drift": d, "seed": s, "arm": "ols_f126_common",
                             "sharpe": _sharpe(rets[rets.index >= p.y.index[252]])})
    df = pd.DataFrame(rows)
    base = pd.read_csv(RESULTS / "nov22_beta_trend.csv")
    base = base[(base.arm == "ols") & base.seed.isin(list(seeds)) & base.half_life.isin(half_lives)
                & base.drift.isin(drifts)][["half_life", "drift", "seed", "arm", "sharpe"]]
    both = pd.concat([base, df], ignore_index=True).rename(columns={"drift": "jump"})
    t = B.paired(both, "ols", "ols_f126_common").rename(columns={"jump": "drift"})
    return t


if __name__ == "__main__":
    if "--posthoc" in sys.argv:
        t = tracking()
        t.round(4).to_csv(RESULTS / "nov22_tracking_posthoc.csv", index=False)
        print(t.round(3).to_string(index=False))
        c = common_span()
        c.round(4).to_csv(RESULTS / "nov22_common_span_posthoc.csv", index=False)
        print(c.round(3).to_string(index=False))
    else:
        main()
