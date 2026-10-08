"""Nested walk-forward tuning of Kalman process noise.

Outer loop: the same 252 / 63 formation / trading blocks as
``formation.formation_trading_backtest``.

Inner loop (formation bars only): roll smaller F' / T' blocks, score each
candidate ``q_beta`` by median net Sharpe across those inner blocks, pick the
winner, then trade the outer block with that ``q_beta``. ``q_alpha`` stays at
the package default.

Causality: the grid search never sees an outer trading bar. A future shock
planted after the formation window must not change the chosen ``q_beta``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .backtest import _max_drawdown, _sharpe
from .formation import FORMATION, TRADING, formation_trading_backtest
from .kalman import Q_ALPHA, Q_BETA
from .params import N_LEGS, SLIPPAGE_BPS, SPREAD_BPS, Z_ENTER, Z_EXIT

# Log-spaced candidates around the package default (1e-6). Extremes are
# deliberately bad: tiny q collapses to static OLS, huge q chases noise.
DEFAULT_Q_GRID = (1e-8, 1e-7, 1e-6, 1e-5, 1e-4)

INNER_FORMATION = 84  # ~4 months of daily bars
INNER_TRADING = 21    # ~1 month


@dataclass
class NestedTuneResult:
    """One outer walk-forward with per-window nested q_beta selection."""

    sharpe: float
    total_return: float
    max_drawdown: float
    trades: int
    n_windows: int
    cost_paid: float
    rehedge_cost: float
    returns: pd.Series = field(repr=False)
    windows: pd.DataFrame = field(repr=False)
    chosen_q: list[float] = field(default_factory=list)

    def as_dict(self) -> dict[str, float]:
        return {
            "sharpe": self.sharpe,
            "total_return": self.total_return,
            "max_drawdown": self.max_drawdown,
            "trades": float(self.trades),
            "n_windows": float(self.n_windows),
            "cost_paid": self.cost_paid,
            "rehedge_cost": self.rehedge_cost,
            "median_chosen_q": float(np.median(self.chosen_q)) if self.chosen_q else float("nan"),
        }


def tune_q_beta(
    y: pd.Series,
    x: pd.Series,
    *,
    q_grid: tuple[float, ...] = DEFAULT_Q_GRID,
    formation: int = INNER_FORMATION,
    trading: int = INNER_TRADING,
    q_alpha: float = Q_ALPHA,
    z_enter: float = Z_ENTER,
    z_exit: float = Z_EXIT,
    spread_bps: float = SPREAD_BPS,
    slippage_bps: float = SLIPPAGE_BPS,
    n_legs: int = N_LEGS,
) -> tuple[float, pd.DataFrame]:
    """Pick ``q_beta`` by median net Sharpe on an inner formation/trading roll.

    Returns ``(best_q, score_table)``. If the sample is too short for even one
    inner window, falls back to the package default ``Q_BETA`` with an empty
    score table — the outer loop still runs, it just does not claim a tune.
    """
    both = pd.concat(
        [pd.Series(y, dtype=float).rename("y"), pd.Series(x, dtype=float).rename("x")],
        axis=1,
    ).dropna()
    if len(both) < formation + trading:
        return float(Q_BETA), pd.DataFrame(columns=["q_beta", "median_sharpe", "n_windows"])

    rows = []
    for q in q_grid:
        res = formation_trading_backtest(
            both["y"],
            both["x"],
            hedge="kalman",
            formation=formation,
            trading=trading,
            q_beta=float(q),
            q_alpha=q_alpha,
            z_enter=z_enter,
            z_exit=z_exit,
            spread_bps=spread_bps,
            slippage_bps=slippage_bps,
            n_legs=n_legs,
        )
        rows.append(
            {
                "q_beta": float(q),
                "median_sharpe": res.sharpe,  # one path; median over windows below
                "n_windows": res.n_windows,
                "window_sharpes": [
                    float(
                        _sharpe(res.returns.loc[w["trade_start"] : w["trade_end"]])
                    )
                    for _, w in res.windows.iterrows()
                ],
            }
        )
    # Prefer the candidate whose *per-window* Sharpes have the highest median;
    # fall back to the path Sharpe if a window is too short for a Sharpe.
    scored = []
    for r in rows:
        ws = [s for s in r["window_sharpes"] if np.isfinite(s)]
        med = float(np.median(ws)) if ws else float(r["median_sharpe"])
        scored.append({"q_beta": r["q_beta"], "median_sharpe": med, "n_windows": r["n_windows"]})
    table = pd.DataFrame(scored).sort_values("q_beta").reset_index(drop=True)
    # Tie-break toward the package default, then toward smaller q (less chasing).
    best_med = table["median_sharpe"].max()
    tied = table.loc[np.isclose(table["median_sharpe"], best_med)]
    if float(Q_BETA) in set(tied["q_beta"]):
        best_q = float(Q_BETA)
    else:
        best_q = float(tied["q_beta"].min())
    return best_q, table


def nested_kalman_backtest(
    y: pd.Series,
    x: pd.Series,
    *,
    q_grid: tuple[float, ...] = DEFAULT_Q_GRID,
    formation: int = FORMATION,
    trading: int = TRADING,
    inner_formation: int = INNER_FORMATION,
    inner_trading: int = INNER_TRADING,
    q_alpha: float = Q_ALPHA,
    z_enter: float = Z_ENTER,
    z_exit: float = Z_EXIT,
    spread_bps: float = SPREAD_BPS,
    slippage_bps: float = SLIPPAGE_BPS,
    n_legs: int = N_LEGS,
) -> NestedTuneResult:
    """Outer 252/63 walk-forward; tune ``q_beta`` inside each formation window."""
    both = pd.concat(
        [pd.Series(y, dtype=float).rename("y"), pd.Series(x, dtype=float).rename("x")],
        axis=1,
    ).dropna().sort_index()
    if formation < 30 or trading < 2:
        raise ValueError("formation must be >= 30 bars and trading >= 2 bars")
    if len(both) < formation + trading:
        raise ValueError("Sample shorter than one formation + trading window.")

    n = len(both)
    rets = pd.Series(np.nan, index=both.index, name="ret")
    cost_total = rehedge_total = 0.0
    trades = 0
    chosen: list[float] = []
    win_rows: list[dict] = []
    start = 0
    while start + formation + trading <= n:
        f_sl = slice(start, start + formation)
        t_end = start + formation + trading
        y_form = both["y"].iloc[f_sl]
        x_form = both["x"].iloc[f_sl]
        q_star, table = tune_q_beta(
            y_form,
            x_form,
            q_grid=q_grid,
            formation=inner_formation,
            trading=inner_trading,
            q_alpha=q_alpha,
            z_enter=z_enter,
            z_exit=z_exit,
            spread_bps=spread_bps,
            slippage_bps=slippage_bps,
            n_legs=n_legs,
        )
        chosen.append(q_star)
        # Re-run a single-block outer trade with the tuned q by calling the
        # shared formation backtest on just this F+T slice (one window).
        slice_yx = both.iloc[start:t_end]
        block = formation_trading_backtest(
            slice_yx["y"],
            slice_yx["x"],
            hedge="kalman",
            formation=formation,
            trading=trading,
            q_beta=q_star,
            q_alpha=q_alpha,
            z_enter=z_enter,
            z_exit=z_exit,
            spread_bps=spread_bps,
            slippage_bps=slippage_bps,
            n_legs=n_legs,
        )
        rets.loc[block.returns.index] = block.returns
        cost_total += block.cost_paid
        rehedge_total += block.rehedge_cost
        trades += block.trades
        win_rows.append(
            {
                "trade_start": block.windows.iloc[0]["trade_start"],
                "trade_end": block.windows.iloc[0]["trade_end"],
                "q_beta": q_star,
                "inner_best_sharpe": float(table["median_sharpe"].max()) if len(table) else float("nan"),
                "net_return": float(block.total_return),
                "trades": block.trades,
            }
        )
        start += trading

    traded = rets.dropna()
    return NestedTuneResult(
        sharpe=_sharpe(traded),
        total_return=float((1.0 + traded).prod() - 1.0) if len(traded) else 0.0,
        max_drawdown=_max_drawdown(traded),
        trades=trades,
        n_windows=len(win_rows),
        cost_paid=cost_total,
        rehedge_cost=rehedge_total,
        chosen_q=chosen,
        returns=traded,
        windows=pd.DataFrame(win_rows),
    )


def compare_fixed_vs_nested(
    y: pd.Series,
    x: pd.Series,
    *,
    q_grid: tuple[float, ...] = DEFAULT_Q_GRID,
    **kwargs,
) -> pd.DataFrame:
    """Same outer walk-forward: fixed default q_beta vs nested-tuned q_beta."""
    fixed_kw = {k: v for k, v in kwargs.items() if k not in ("inner_formation", "inner_trading", "q_grid")}
    fixed = formation_trading_backtest(y, x, hedge="kalman", q_beta=Q_BETA, **fixed_kw)
    nested = nested_kalman_backtest(y, x, q_grid=q_grid, **kwargs)
    return pd.DataFrame(
        {
            "fixed_default_q": {
                **fixed.as_dict(),
                "median_chosen_q": float(Q_BETA),
            },
            "nested_tuned_q": nested.as_dict(),
        }
    ).T
