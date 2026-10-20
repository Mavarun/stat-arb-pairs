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
from .params import BORROW_BPS, N_LEGS, SLIPPAGE_BPS, SPREAD_BPS, Z_ENTER, Z_EXIT

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
    borrow_cost: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "sharpe": self.sharpe,
            "total_return": self.total_return,
            "max_drawdown": self.max_drawdown,
            "trades": float(self.trades),
            "n_windows": float(self.n_windows),
            "cost_paid": self.cost_paid,
            "rehedge_cost": self.rehedge_cost,
            "borrow_cost": self.borrow_cost,
            "median_chosen_q": float(np.median(self.chosen_q)) if self.chosen_q else float("nan"),
        }


SELECTION_RULES = ("best", "one_se", "sign")
SIGN_ALPHA = 0.10


def _sign_test_p(wins: int, n: int) -> float:
    """One-sided sign-test p-value P(Binomial(n, 0.5) >= wins)."""
    from scipy.stats import binom

    if n == 0:
        return 1.0
    return float(binom.sf(wins - 1, n, 0.5))


def _select(
    cands: list[dict], default_key: tuple, selection: str, sign_alpha: float
) -> tuple[tuple, list[dict]]:
    """Pick a candidate key from per-window Sharpe lists (see ``tune_q_beta``).

    Every candidate dict has ``key`` (tuple of q values), ``ws`` (finite
    per-window net Sharpes, aligned across candidates) and ``median_sharpe``.
    Returns ``(chosen_key, annotated_candidates)``.
    """
    def dist(k: tuple) -> float:
        return float(sum(abs(np.log10(a) - np.log10(b)) for a, b in zip(k, default_key)))

    by_key = {c["key"]: c for c in cands}
    best_med = max(c["median_sharpe"] for c in cands)
    if selection == "best":
        tied = [c for c in cands if np.isclose(c["median_sharpe"], best_med)]
        keys = {c["key"] for c in tied}
        chosen = default_key if default_key in keys else min(keys)
        return chosen, cands
    if selection == "one_se":
        best = max(cands, key=lambda c: (c["median_sharpe"], -dist(c["key"])))
        n = len(best["ws"])
        # Standard error of a sample median ~ 1.2533 * sd / sqrt(n).
        se = 1.2533 * float(np.std(best["ws"], ddof=1)) / np.sqrt(n) if n > 1 else float("inf")
        for c in cands:
            c["within_one_se"] = bool(c["median_sharpe"] >= best_med - se)
        ok = [c for c in cands if c["within_one_se"]]
        chosen = min(ok, key=lambda c: (dist(c["key"]), c["key"]))["key"]
        return chosen, cands
    if selection == "sign":
        base = by_key.get(default_key)
        if base is None:
            raise ValueError("the sign rule needs the default q in the grid")
        for c in cands:
            d = np.asarray(c["ws"]) - np.asarray(base["ws"])
            nz = d[d != 0.0]
            wins = int((nz > 0).sum())
            c["sign_p"] = _sign_test_p(wins, len(nz)) if c["key"] != default_key else 1.0
            c["median_gain"] = float(np.median(d)) if len(d) else 0.0
        ok = [c for c in cands if c["key"] != default_key and c["sign_p"] < sign_alpha and c["median_gain"] > 0]
        if not ok:
            return default_key, cands
        chosen = max(ok, key=lambda c: (c["median_sharpe"], -dist(c["key"])))["key"]
        return chosen, cands
    raise ValueError(f"selection must be one of {SELECTION_RULES}, got {selection!r}")


def tune_q(
    y: pd.Series,
    x: pd.Series,
    *,
    q_grid: tuple[float, ...] = DEFAULT_Q_GRID,
    q_alpha_grid: tuple[float, ...] | None = None,
    formation: int = INNER_FORMATION,
    trading: int = INNER_TRADING,
    q_alpha: float = Q_ALPHA,
    z_enter: float = Z_ENTER,
    z_exit: float = Z_EXIT,
    spread_bps: float = SPREAD_BPS,
    slippage_bps: float = SLIPPAGE_BPS,
    n_legs: int = N_LEGS,
    borrow_bps: float = BORROW_BPS,
    selection: str = "best",
    sign_alpha: float = SIGN_ALPHA,
) -> tuple[float, float, pd.DataFrame]:
    """Pick ``(q_beta, q_alpha)`` on an inner formation/trading roll.

    Each candidate is scored by the median of its per-inner-window net
    Sharpes. ``selection``:

    - ``"best"`` (legacy): highest median; ties go to the default, then the
      smallest q.
    - ``"one_se"``: among candidates whose median is within one standard
      error of the best median (SE of a median ~ 1.2533 sd / sqrt(n) of the
      best candidate's window Sharpes), take the one closest to the default
      in log10 distance. Small, noisy edges therefore keep the default.
    - ``"sign"``: leave the default only for a candidate that beats it in
      enough paired inner windows (one-sided sign test p < ``sign_alpha``)
      with a positive median gain; among those, the highest median.

    ``q_alpha_grid=None`` keeps ``q_alpha`` fixed (only q_beta is tuned).
    Too-short samples fall back to the defaults with an empty table.
    """
    if selection not in SELECTION_RULES:
        raise ValueError(f"selection must be one of {SELECTION_RULES}, got {selection!r}")
    both = pd.concat(
        [pd.Series(y, dtype=float).rename("y"), pd.Series(x, dtype=float).rename("x")],
        axis=1,
    ).dropna()
    cols = ["q_beta", "q_alpha", "median_sharpe", "n_windows"]
    if len(both) < formation + trading:
        return float(Q_BETA), float(q_alpha), pd.DataFrame(columns=cols)

    alphas = tuple(q_alpha_grid) if q_alpha_grid is not None else (float(q_alpha),)
    cands = []
    for qa in alphas:
        for q in q_grid:
            res = formation_trading_backtest(
                both["y"],
                both["x"],
                hedge="kalman",
                formation=formation,
                trading=trading,
                q_beta=float(q),
                q_alpha=float(qa),
                z_enter=z_enter,
                z_exit=z_exit,
                spread_bps=spread_bps,
                slippage_bps=slippage_bps,
                n_legs=n_legs,
                borrow_bps=borrow_bps,
            )
            ws_all = [
                float(_sharpe(res.returns.loc[w["trade_start"] : w["trade_end"]]))
                for _, w in res.windows.iterrows()
            ]
            cands.append({"key": (float(q), float(qa)), "ws_all": ws_all,
                          "path_sharpe": res.sharpe, "n_windows": res.n_windows})
    # Align windows: keep only inner windows where every candidate has a finite Sharpe.
    n_w = min(len(c["ws_all"]) for c in cands)
    keep = [i for i in range(n_w) if all(np.isfinite(c["ws_all"][i]) for c in cands)]
    for c in cands:
        c["ws"] = [c["ws_all"][i] for i in keep]
        if selection == "best":
            # legacy scoring: each candidate's own finite window Sharpes
            own = [v for v in c["ws_all"] if np.isfinite(v)]
            c["median_sharpe"] = float(np.median(own)) if own else float(c["path_sharpe"])
        else:
            c["median_sharpe"] = float(np.median(c["ws"])) if c["ws"] else float(c["path_sharpe"])
    default_key = (float(Q_BETA), float(q_alpha))
    if default_key not in {c["key"] for c in cands} and selection != "best":
        raise ValueError("one_se / sign need the default (Q_BETA, q_alpha) in the grid")
    chosen, cands = _select(cands, default_key, selection, sign_alpha)
    rows = []
    for c in cands:
        row = {"q_beta": c["key"][0], "q_alpha": c["key"][1], "median_sharpe": c["median_sharpe"],
               "n_windows": c["n_windows"]}
        for k in ("within_one_se", "sign_p", "median_gain"):
            if k in c:
                row[k] = c[k]
        rows.append(row)
    table = pd.DataFrame(rows).sort_values(["q_alpha", "q_beta"]).reset_index(drop=True)
    return chosen[0], chosen[1], table


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
    borrow_bps: float = BORROW_BPS,
    selection: str = "best",
    sign_alpha: float = SIGN_ALPHA,
) -> tuple[float, pd.DataFrame]:
    """Pick ``q_beta`` (``q_alpha`` fixed); see ``tune_q`` for the rules.

    Returns ``(best_q, score_table)``. If the sample is too short for even one
    inner window, falls back to the package default ``Q_BETA`` with an empty
    score table — the outer loop still runs, it just does not claim a tune.
    """
    qb, _qa, table = tune_q(
        y, x, q_grid=q_grid, q_alpha_grid=None, formation=formation, trading=trading,
        q_alpha=q_alpha, z_enter=z_enter, z_exit=z_exit, spread_bps=spread_bps,
        slippage_bps=slippage_bps, n_legs=n_legs, borrow_bps=borrow_bps,
        selection=selection, sign_alpha=sign_alpha,
    )
    return qb, table.drop(columns=["q_alpha"], errors="ignore")


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
    borrow_bps: float = BORROW_BPS,
) -> NestedTuneResult:
    """Outer 252/63 walk-forward; tune ``q_beta`` inside each formation window.

    ``borrow_bps`` (annualised short-leg fee) is charged both in the inner
    scoring and in the outer book, so the tune optimises the same net PnL.
    """
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
    cost_total = rehedge_total = borrow_total = 0.0
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
            borrow_bps=borrow_bps,
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
            borrow_bps=borrow_bps,
        )
        rets.loc[block.returns.index] = block.returns
        cost_total += block.cost_paid
        rehedge_total += block.rehedge_cost
        borrow_total += block.borrow_cost
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
        borrow_cost=borrow_total,
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
