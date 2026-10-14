"""Reproduce the README numbers for ranked/capped re-screen and slower beta drift.

Synthetic, seeded, offline. Run: ``python scripts/run_oct14_experiments.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from statarb.formation import formation_trading_backtest
from statarb.nested import compare_fixed_vs_nested
from statarb.rescreen import universe_formation_backtest
from statarb.synth import make_ou_pair, make_planted_universe, make_random_walk_universe

N_SEEDS_RANK = 10
N_SEEDS_DRIFT = 12
FORMATION = 252
TRADING = 63
N_BLOCKS = 8


def ranked_vs_equal_weight(rule: str = "bh") -> pd.DataFrame:
    """Planted mix of fast/slow pairs: equal-weight vs capped score/eg_p/HL.

    Under BH, median discoveries are already ~1 so a top-1 cap is mostly a
    no-op. Under naive EG the screen over-discovers and the cap matters.
    """
    policies = (
        ("eg_p", None, "equal_weight_all"),
        ("eg_p", 1, "eg_p_top1"),
        ("score", 1, "score_top1"),
        ("half_life", 1, "hl_top1"),
        ("eg_p", 2, "eg_p_top2"),
    )
    rows = {label: [] for _, _, label in policies}
    for s in range(N_SEEDS_RANK):
        frame, _ = make_planted_universe(
            10,
            3,
            FORMATION + TRADING * N_BLOCKS,
            half_life=(4.0, 8.0, 28.0),
            seed=300 + s,
        )
        for rank_by, max_pairs, label in policies:
            res = universe_formation_backtest(
                frame,
                rule=rule,
                hedge="ols",
                formation=FORMATION,
                trading=TRADING,
                rank_by=rank_by,
                max_pairs=max_pairs,
            )
            rows[label].append(res.as_dict())
    out = []
    for _, _, label in policies:
        df = pd.DataFrame(rows[label])
        out.append(
            {
                "rule": rule,
                "policy": label,
                "median_mean_discoveries": float(df["mean_discoveries"].median()),
                "median_mean_selected": float(df["mean_selected"].median()),
                "median_frac_windows_traded": float(df["frac_windows_traded"].median()),
                "median_sharpe": float(df["sharpe"].median()),
                "median_total_return": float(df["total_return"].median()),
                "median_cost_paid": float(df["cost_paid"].median()),
            }
        )
    return pd.DataFrame(out)


def time_stop_on_planted() -> pd.DataFrame:
    """BH equal-weight book with and without formation OU time-stop."""
    rows = {"off": [], "on_3x": []}
    for s in range(N_SEEDS_RANK):
        frame, _ = make_planted_universe(
            10, 3, FORMATION + TRADING * N_BLOCKS, half_life=5.0, seed=400 + s
        )
        off = universe_formation_backtest(
            frame, rule="bh", formation=FORMATION, trading=TRADING, use_time_stop=False
        )
        on = universe_formation_backtest(
            frame,
            rule="bh",
            formation=FORMATION,
            trading=TRADING,
            use_time_stop=True,
            time_stop_multiple=3.0,
        )
        rows["off"].append(off.as_dict())
        rows["on_3x"].append(on.as_dict())
    out = []
    for label, dicts in rows.items():
        df = pd.DataFrame(dicts)
        out.append(
            {
                "time_stop": label,
                "median_sharpe": float(df["sharpe"].median()),
                "median_total_return": float(df["total_return"].median()),
                "median_trades": float(df["trades"].median()),
                "median_cost_paid": float(df["cost_paid"].median()),
            }
        )
    return pd.DataFrame(out)


def drift_speeds_nested() -> pd.DataFrame:
    """Fixed vs nested Kalman under slow / medium / fast linear beta drift."""
    n = FORMATION + TRADING * N_BLOCKS
    scenarios = (
        ("constant", None),
        ("slow_1.0_to_1.3", np.linspace(1.0, 1.3, n)),
        ("medium_1.0_to_1.5", np.linspace(1.0, 1.5, n)),
        ("fast_1.0_to_2.5", np.linspace(1.0, 2.5, n)),
    )
    out = []
    for name, path in scenarios:
        fixed_sr, nested_sr = [], []
        fixed_tr, nested_tr = [], []
        nested_wins = 0
        chosen = []
        for s in range(N_SEEDS_DRIFT):
            kw = {} if path is None else {"beta_path": path}
            p = make_ou_pair(n, half_life=5, seed=500 + s, **kw)
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
            ft = float(table.loc["fixed_default_q", "total_return"])
            nt = float(table.loc["nested_tuned_q", "total_return"])
            fixed_sr.append(fs)
            nested_sr.append(ns)
            fixed_tr.append(ft)
            nested_tr.append(nt)
            if ns > fs:
                nested_wins += 1
            chosen.append(float(table.loc["nested_tuned_q", "median_chosen_q"]))
        out.append(
            {
                "scenario": name,
                "median_sharpe_fixed": float(np.median(fixed_sr)),
                "median_sharpe_nested": float(np.median(nested_sr)),
                "nested_wins": f"{nested_wins}/{N_SEEDS_DRIFT}",
                "median_return_fixed": float(np.median(fixed_tr)),
                "median_return_nested": float(np.median(nested_tr)),
                "median_chosen_q": float(np.median(chosen)),
            }
        )
    return pd.DataFrame(out)


def time_stop_on_drift() -> pd.DataFrame:
    """Single-pair OLS book on fast drift: does a 3x HL time-stop help after costs?"""
    n = FORMATION + TRADING * N_BLOCKS
    path = np.linspace(1.0, 2.5, n)
    rows = {"off": [], "on_3x": []}
    for s in range(N_SEEDS_DRIFT):
        p = make_ou_pair(n, half_life=5, beta_path=path, seed=600 + s)
        off = formation_trading_backtest(
            p.y, p.x, hedge="ols", formation=FORMATION, trading=TRADING, use_time_stop=False
        )
        on = formation_trading_backtest(
            p.y,
            p.x,
            hedge="ols",
            formation=FORMATION,
            trading=TRADING,
            use_time_stop=True,
            time_stop_multiple=3.0,
        )
        rows["off"].append(off.as_dict())
        rows["on_3x"].append(on.as_dict())
    out = []
    for label, dicts in rows.items():
        df = pd.DataFrame(dicts)
        out.append(
            {
                "time_stop": label,
                "median_sharpe": float(df["sharpe"].median()),
                "median_total_return": float(df["total_return"].median()),
                "median_trades": float(df["trades"].median()),
                "median_cost_paid": float(df["cost_paid"].median()),
            }
        )
    return pd.DataFrame(out)


def noise_cap_stays_flat() -> pd.DataFrame:
    """Pure noise: BH + top1 score should stay near-flat like uncapped BH."""
    rows = {"bh_all": [], "bh_score_top1": []}
    for s in range(N_SEEDS_RANK):
        frame = make_random_walk_universe(12, FORMATION + TRADING * N_BLOCKS, seed=700 + s)
        a = universe_formation_backtest(
            frame, rule="bh", formation=FORMATION, trading=TRADING, rank_by="eg_p"
        )
        b = universe_formation_backtest(
            frame,
            rule="bh",
            formation=FORMATION,
            trading=TRADING,
            rank_by="score",
            max_pairs=1,
        )
        rows["bh_all"].append(a.as_dict())
        rows["bh_score_top1"].append(b.as_dict())
    out = []
    for label, dicts in rows.items():
        df = pd.DataFrame(dicts)
        out.append(
            {
                "policy": label,
                "median_mean_selected": float(df["mean_selected"].median()),
                "median_frac_windows_traded": float(df["frac_windows_traded"].median()),
                "median_sharpe": float(df["sharpe"].median()),
                "median_total_return": float(df["total_return"].median()),
                "median_cost_paid": float(df["cost_paid"].median()),
            }
        )
    return pd.DataFrame(out)


def main() -> None:
    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", 20)
    print("=== Ranked / capped re-screen (planted fast/slow mix, BH) ===")
    print(ranked_vs_equal_weight("bh").to_string(index=False))
    print()
    print("=== Ranked / capped re-screen (planted fast/slow mix, naive EG) ===")
    print(ranked_vs_equal_weight("naive").to_string(index=False))
    print()
    print("=== Pure noise: BH all vs BH score top-1 ===")
    print(noise_cap_stays_flat().to_string(index=False))
    print()
    print("=== Time-stop on BH re-screen (planted HL=5) ===")
    print(time_stop_on_planted().to_string(index=False))
    print()
    print("=== Nested vs fixed Kalman across drift speeds ===")
    print(drift_speeds_nested().to_string(index=False))
    print()
    print("=== Time-stop on single-pair OLS under fast drift ===")
    print(time_stop_on_drift().to_string(index=False))


if __name__ == "__main__":
    main()
