"""Reproduce the README numbers for OU fit, Kalman hedge, screening and walk-forward.

Synthetic, seeded, offline. Run: ``python scripts/run_oct05_experiments.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from statarb.formation import compare_hedges
from statarb.kalman import kalman_hedge, static_hedge_path
from statarb.ou import fit_ou
from statarb.screening import screen_pairs, screen_summary
from statarb.synth import make_ou_pair, make_planted_universe, make_random_walk_universe

N_SEEDS = 20


def ou_recovery() -> pd.DataFrame:
    rows = []
    for hl in (5.0, 10.0, 20.0):
        est = [fit_ou(make_ou_pair(1500, half_life=hl, seed=s).spread).half_life for s in range(N_SEEDS)]
        rows.append({"planted_half_life": hl, "median_est": np.median(est),
                     "p10": np.percentile(est, 10), "p90": np.percentile(est, 90)})
    return pd.DataFrame(rows)


def hedge_tracking() -> pd.DataFrame:
    n = 1200
    rmse_k, rmse_s = [], []
    for s in range(N_SEEDS):
        p = make_ou_pair(n, beta_path=np.linspace(1.0, 2.5, n), seed=s)
        k = kalman_hedge(p.y, p.x)["beta"]
        st = static_hedge_path(p.y, p.x)
        m = k.notna() & st.notna()
        rmse_k.append(float(np.sqrt(((k[m] - p.beta_path[m]) ** 2).mean())))
        rmse_s.append(float(np.sqrt(((st[m] - p.beta_path[m]) ** 2).mean())))
    return pd.DataFrame({"hedge": ["kalman", "static_ols"],
                         "median_beta_rmse": [np.median(rmse_k), np.median(rmse_s)]})


def screening() -> pd.DataFrame:
    null = [screen_summary(screen_pairs(make_random_walk_universe(20, 750, seed=s))) for s in range(10)]
    planted = []
    for s in range(10):
        frame, truth = make_planted_universe(16, 4, 750, seed=s)
        planted.append(screen_summary(screen_pairs(frame), truth))
    out = pd.DataFrame({
        "null_mean_false_pos": sum(t["false_pos"] for t in null) / len(null),
        "planted_mean_power": sum(t["power"] for t in planted) / len(planted),
        "planted_mean_fdp": sum(t["fdp"] for t in planted) / len(planted),
    })
    return out


def walk_forward() -> pd.DataFrame:
    n = 252 + 63 * 12
    rows = []
    for name, path in (("constant_beta", None), ("drifting_beta_1.0_to_2.5", np.linspace(1.0, 2.5, n))):
        res = []
        for s in range(N_SEEDS):
            kw = {} if path is None else {"beta_path": path}
            p = make_ou_pair(n, half_life=5, seed=100 + s, **kw)
            res.append(compare_hedges(p.y, p.x))
        sr = np.array([[t.loc["ols", "sharpe"], t.loc["kalman", "sharpe"]] for t in res])
        tr = np.array([[t.loc["ols", "total_return"], t.loc["kalman", "total_return"]] for t in res])
        rows.append({"scenario": name, "median_sharpe_ols": np.median(sr[:, 0]),
                     "median_sharpe_kalman": np.median(sr[:, 1]),
                     "kalman_wins": f"{int((sr[:, 1] > sr[:, 0]).sum())}/{N_SEEDS}",
                     "median_return_ols": np.median(tr[:, 0]), "median_return_kalman": np.median(tr[:, 1]),
                     "median_rehedge_cost": np.median([t.loc["kalman", "rehedge_cost"] for t in res])})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 140)
    pd.set_option("display.float_format", "{:.3f}".format)
    for title, fn in (("OU half-life recovery (1500 bars)", ou_recovery),
                      ("Hedge tracking, drifting beta", hedge_tracking),
                      ("Pair screening (190 tests/universe, 10 seeds)", screening),
                      ("Walk-forward 252/63, net of 10 bps/leg", walk_forward)):
        print(f"\n== {title} ==")
        print(fn().to_string())
