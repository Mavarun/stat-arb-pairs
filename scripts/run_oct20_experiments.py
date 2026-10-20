"""Reproduce the README numbers for the Oct 20 slice: regularised nested tuning and borrow.

Synthetic, seeded, offline. Run: ``python scripts/run_oct20_experiments.py``.
All books are net of 10 bps per leg per unit change (5 spread + 5 slippage),
the Kalman re-hedge charge, and (third table only) a short-leg borrow fee.
Every number is out-of-sample: the outer 252/63 walk-forward only trades bars
after each formation window, and every tune runs inside the formation window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from statarb.formation import formation_trading_backtest
from statarb.kalman import Q_ALPHA, Q_BETA
from statarb.nested import nested_kalman_backtest
from statarb.rescreen import universe_formation_backtest
from statarb.synth import make_ou_pair, make_planted_universe

N_SEEDS_DRIFT = 12  # same seeds as Oct 14 (500 + s)
N_SEEDS_BORROW = 10
FORMATION = 252
TRADING = 63
N_BLOCKS = 8
INNER = dict(inner_formation=84, inner_trading=21)
Q_ALPHA_GRID = (1e-6, 1e-5, 1e-4, 1e-3)
BORROW_LEVELS = (0.0, 25.0, 100.0, 300.0)


def _scenarios(n: int):
    return (
        ("constant", None),
        ("slow_1.0_to_1.3", np.linspace(1.0, 1.3, n)),
        ("medium_1.0_to_1.5", np.linspace(1.0, 1.5, n)),
        ("fast_1.0_to_2.5", np.linspace(1.0, 2.5, n)),
    )


def selection_rules_by_drift(joint_q_alpha: bool = False) -> pd.DataFrame:
    """Fixed default q vs nested best / one-SE / sign rule across drift speeds."""
    n = FORMATION + TRADING * N_BLOCKS
    rules = ("best", "one_se", "sign")
    out = []
    for name, path in _scenarios(n):
        res = {k: [] for k in ("fixed", *rules)}
        for s in range(N_SEEDS_DRIFT):
            kw = {} if path is None else {"beta_path": path}
            p = make_ou_pair(n, half_life=5, seed=500 + s, **kw)
            fixed = formation_trading_backtest(p.y, p.x, hedge="kalman", q_beta=Q_BETA,
                                               formation=FORMATION, trading=TRADING)
            res["fixed"].append({"sharpe": fixed.sharpe, "total_return": fixed.total_return,
                                 "frac_default": 1.0, "q_alpha": Q_ALPHA})
            for rule in rules:
                r = nested_kalman_backtest(
                    p.y, p.x, formation=FORMATION, trading=TRADING, selection=rule,
                    q_alpha_grid=Q_ALPHA_GRID if joint_q_alpha else None, **INNER,
                )
                d = r.as_dict()
                res[rule].append({"sharpe": d["sharpe"], "total_return": d["total_return"],
                                  "frac_default": d["frac_windows_default_q"],
                                  "q_alpha": d["median_chosen_q_alpha"]})
        f_sr = np.array([v["sharpe"] for v in res["fixed"]])
        for k in ("fixed", *rules):
            sr = np.array([v["sharpe"] for v in res[k]])
            out.append({
                "scenario": name,
                "book": k if k == "fixed" else f"nested_{k}",
                "median_sharpe": float(np.median(sr)),
                "wins_vs_fixed": "-" if k == "fixed" else f"{int((sr > f_sr + 1e-12).sum())}/{N_SEEDS_DRIFT}",
                "median_return": float(np.median([v["total_return"] for v in res[k]])),
                "median_frac_windows_default_q": float(np.median([v["frac_default"] for v in res[k]])),
                "median_q_alpha": float(np.median([v["q_alpha"] for v in res[k]])),
            })
    return pd.DataFrame(out)


def borrow_sensitivity() -> pd.DataFrame:
    """BH re-screen universe book and a single known pair, at several borrow rates."""
    out = []
    n = FORMATION + TRADING * N_BLOCKS
    for bps in BORROW_LEVELS:
        uni, single = [], []
        for s in range(N_SEEDS_BORROW):
            frame, _ = make_planted_universe(10, 3, n, half_life=5.0, seed=100 + s)
            r = universe_formation_backtest(frame, rule="bh", hedge="ols", formation=FORMATION,
                                            trading=TRADING, borrow_bps=bps)
            uni.append(r.as_dict())
            p = make_ou_pair(n, half_life=5, seed=700 + s)
            f = formation_trading_backtest(p.y, p.x, hedge="ols", formation=FORMATION,
                                           trading=TRADING, borrow_bps=bps)
            single.append(f.as_dict())
        for label, rows in (("bh_rescreen_universe", uni), ("single_pair_ols", single)):
            df = pd.DataFrame(rows)
            out.append({
                "book": label,
                "borrow_bps_per_year": bps,
                "median_sharpe": float(df["sharpe"].median()),
                "median_total_return": float(df["total_return"].median()),
                "median_borrow_paid": float(df["borrow_cost"].median()),
                "median_trade_cost_paid": float((df["cost_paid"] - df["borrow_cost"]).median()),
            })
    return pd.DataFrame(out)


def auto_time_stop() -> pd.DataFrame:
    """Single-pair OLS book: no stop vs fixed 3 x HL vs formation-chosen k ("auto")."""
    n = FORMATION + TRADING * N_BLOCKS
    cases = (("constant_beta", None, 800), ("fast_drift_1.0_to_2.5", np.linspace(1.0, 2.5, n), 600))
    variants = (("off", False, 3.0), ("fixed_3x", True, 3.0), ("auto", True, "auto"))
    out = []
    for name, path, seed0 in cases:
        rows = {v[0]: [] for v in variants}
        ks = []
        for s in range(N_SEEDS_DRIFT):
            kw = {} if path is None else {"beta_path": path}
            p = make_ou_pair(n, half_life=5, seed=seed0 + s, **kw)
            for label, on, k in variants:
                r = formation_trading_backtest(p.y, p.x, hedge="ols", formation=FORMATION,
                                               trading=TRADING, use_time_stop=on, time_stop_multiple=k)
                rows[label].append(r.as_dict())
                if label == "auto":
                    ks.extend(r.windows["time_stop_multiple"].tolist())
        off_sr = np.array([d["sharpe"] for d in rows["off"]])
        for label, _, _ in variants:
            df = pd.DataFrame(rows[label])
            sr = df["sharpe"].to_numpy()
            out.append({
                "scenario": name,
                "time_stop": label,
                "median_sharpe": float(np.median(sr)),
                "wins_vs_off": "-" if label == "off" else f"{int((sr > off_sr + 1e-12).sum())}/{N_SEEDS_DRIFT}",
                "median_total_return": float(df["total_return"].median()),
                "median_trades": float(df["trades"].median()),
                "median_cost_paid": float(df["cost_paid"].median()),
                "auto_k_share_no_stop": (float(np.mean([not np.isfinite(k) for k in ks]))
                                         if label == "auto" else float("nan")),
            })
    return pd.DataFrame(out)


def main() -> None:
    pd.set_option("display.width", 160)
    print("=== Nested selection rules across drift speeds (q_alpha fixed) ===")
    print(selection_rules_by_drift().to_string(index=False))
    print()
    print("=== Same, with q_alpha tuned jointly on", Q_ALPHA_GRID, "===")
    print(selection_rules_by_drift(joint_q_alpha=True).to_string(index=False))
    print()
    print("=== Borrow-fee sensitivity (10 seeds, 8 blocks) ===")
    print(borrow_sensitivity().to_string(index=False))
    print()
    print("=== OU time-stop: off vs fixed 3x HL vs formation-chosen k (12 seeds) ===")
    print(auto_time_stop().to_string(index=False))


if __name__ == "__main__":
    main()
