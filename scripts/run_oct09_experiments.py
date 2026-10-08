"""Reproduce the README numbers for per-window re-screening and nested Kalman tune.

Synthetic, seeded, offline. Run: ``python scripts/run_oct09_experiments.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from statarb.formation import formation_trading_backtest
from statarb.kalman import Q_BETA
from statarb.nested import compare_fixed_vs_nested, nested_kalman_backtest
from statarb.rescreen import universe_formation_backtest
from statarb.synth import make_ou_pair, make_planted_universe, make_random_walk_universe

N_SEEDS_SCREEN = 10
N_SEEDS_NESTED = 12
FORMATION = 252
TRADING = 63
N_BLOCKS = 8  # shorter than Oct 5's 12 so the screen+nested grid stays tractable


def rescreen_noise() -> pd.DataFrame:
    """Pure random-walk universes: how often does each rule trade, and at what cost?"""
    rows = {r: [] for r in ("naive_single", "naive", "bh", "bh_johansen")}
    for s in range(N_SEEDS_SCREEN):
        frame = make_random_walk_universe(12, FORMATION + TRADING * N_BLOCKS, seed=s)
        for rule in rows:
            res = universe_formation_backtest(
                frame, rule=rule, hedge="ols", formation=FORMATION, trading=TRADING
            )
            rows[rule].append(res.as_dict())
    out = []
    for rule, dicts in rows.items():
        df = pd.DataFrame(dicts)
        out.append(
            {
                "rule": rule,
                "median_mean_discoveries": float(df["mean_discoveries"].median()),
                "median_frac_windows_traded": float(df["frac_windows_traded"].median()),
                "median_sharpe": float(df["sharpe"].median()),
                "median_total_return": float(df["total_return"].median()),
                "median_cost_paid": float(df["cost_paid"].median()),
            }
        )
    return pd.DataFrame(out)


def rescreen_planted() -> pd.DataFrame:
    """Universes with 3 planted pairs + noise: power vs costs after re-screening."""
    rows = {r: [] for r in ("naive", "bh", "bh_johansen")}
    for s in range(N_SEEDS_SCREEN):
        frame, _ = make_planted_universe(
            10, 3, FORMATION + TRADING * N_BLOCKS, half_life=5.0, seed=100 + s
        )
        for rule in rows:
            res = universe_formation_backtest(
                frame, rule=rule, hedge="ols", formation=FORMATION, trading=TRADING
            )
            rows[rule].append(res.as_dict())
    out = []
    for rule, dicts in rows.items():
        df = pd.DataFrame(dicts)
        out.append(
            {
                "rule": rule,
                "median_mean_discoveries": float(df["mean_discoveries"].median()),
                "median_frac_windows_traded": float(df["frac_windows_traded"].median()),
                "median_sharpe": float(df["sharpe"].median()),
                "median_total_return": float(df["total_return"].median()),
                "median_cost_paid": float(df["cost_paid"].median()),
            }
        )
    return pd.DataFrame(out)


def nested_vs_fixed() -> pd.DataFrame:
    """Fixed default q_beta vs nested-tuned q_beta, constant and drifting beta."""
    n = FORMATION + TRADING * N_BLOCKS
    rows = []
    for name, path in (
        ("constant_beta", None),
        ("drifting_beta_1.0_to_2.5", np.linspace(1.0, 2.5, n)),
    ):
        fixed_sr, nested_sr = [], []
        fixed_tr, nested_tr = [], []
        chosen = []
        nested_wins = 0
        for s in range(N_SEEDS_NESTED):
            kw = {} if path is None else {"beta_path": path}
            p = make_ou_pair(n, half_life=5, seed=200 + s, **kw)
            table = compare_fixed_vs_nested(
                p.y,
                p.x,
                formation=FORMATION,
                trading=TRADING,
                inner_formation=84,
                inner_trading=21,
            )
            fs = float(table.loc["fixed_default_q", "sharpe"])
            ns = float(table.loc["nested_tuned_q", "sharpe"])
            fixed_sr.append(fs)
            nested_sr.append(ns)
            fixed_tr.append(float(table.loc["fixed_default_q", "total_return"]))
            nested_tr.append(float(table.loc["nested_tuned_q", "total_return"]))
            chosen.append(float(table.loc["nested_tuned_q", "median_chosen_q"]))
            nested_wins += int(ns > fs)
        rows.append(
            {
                "scenario": name,
                "median_sharpe_fixed": float(np.median(fixed_sr)),
                "median_sharpe_nested": float(np.median(nested_sr)),
                "nested_wins": f"{nested_wins}/{N_SEEDS_NESTED}",
                "median_return_fixed": float(np.median(fixed_tr)),
                "median_return_nested": float(np.median(nested_tr)),
                "median_chosen_q": float(np.median(chosen)),
                "default_q": float(Q_BETA),
            }
        )
    return pd.DataFrame(rows)


def oracle_vs_rescreen() -> pd.DataFrame:
    """Upper bound: trade the known planted pair vs BH-rescreened universe book."""
    rows = []
    for s in range(N_SEEDS_SCREEN):
        frame, truth = make_planted_universe(
            10, 3, FORMATION + TRADING * N_BLOCKS, half_life=5.0, seed=300 + s
        )
        # Oracle: first planted pair, static OLS walk-forward on that pair alone.
        pair = next(iter(truth))
        a, b = sorted(pair)
        oracle = formation_trading_backtest(
            frame[a], frame[b], hedge="ols", formation=FORMATION, trading=TRADING
        )
        screened = universe_formation_backtest(
            frame, rule="bh", hedge="ols", formation=FORMATION, trading=TRADING
        )
        rows.append(
            {
                "oracle_sharpe": oracle.sharpe,
                "oracle_return": oracle.total_return,
                "bh_sharpe": screened.sharpe,
                "bh_return": screened.total_return,
                "bh_mean_discoveries": screened.mean_discoveries,
            }
        )
    df = pd.DataFrame(rows)
    return pd.DataFrame(
        [
            {
                "book": "oracle_known_pair",
                "median_sharpe": float(df["oracle_sharpe"].median()),
                "median_total_return": float(df["oracle_return"].median()),
                "median_mean_discoveries": 1.0,
            },
            {
                "book": "bh_rescreen_universe",
                "median_sharpe": float(df["bh_sharpe"].median()),
                "median_total_return": float(df["bh_return"].median()),
                "median_mean_discoveries": float(df["bh_mean_discoveries"].median()),
            },
        ]
    )


if __name__ == "__main__":
    pd.set_option("display.width", 160)

    def _fmt(x):
        if isinstance(x, float):
            ax = abs(x)
            if ax != 0 and (ax < 1e-3 or ax >= 1e3):
                return f"{x:.2e}"
            return f"{x:.4f}"
        return str(x)

    pd.set_option("display.float_format", _fmt)
    for title, fn in (
        ("Re-screen on pure noise (12 assets, 8 blocks, 10 seeds)", rescreen_noise),
        ("Re-screen with 3 planted pairs (10 seeds)", rescreen_planted),
        ("Oracle known pair vs BH re-screen (10 seeds)", oracle_vs_rescreen),
        ("Nested vs fixed Kalman q_beta (12 seeds, 8 blocks)", nested_vs_fixed),
    ):
        print(f"\n== {title} ==")
        print(fn().to_string(index=False))
