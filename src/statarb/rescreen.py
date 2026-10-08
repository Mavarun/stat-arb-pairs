"""Per-window universe re-screening walk-forward.

The Oct 5 walk-forward traded one *known* pair. This module screens the full
price universe inside each formation window, then trades only the discoveries
of a chosen rule (BH, BH+Johansen, or naive EG) in the following trading
block. Positions are equal-weighted across discoveries and forced flat at
block end so nothing leaks across windows.

Causality contract:

* Screening and hedge estimation use formation bars only.
* Trading bars never enter the screen, the OLS fit, or the Kalman warm-up.
* A break planted inside a later trading block must not change earlier returns.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .backtest import _max_drawdown, _sharpe
from .formation import FORMATION, TRADING, _block_pnl, _window_signal
from .params import N_LEGS, SLIPPAGE_BPS, SPREAD_BPS, Z_ENTER, Z_EXIT
from .screening import RULES, screen_pairs
from .signals import mean_reversion_positions
from .kalman import Q_ALPHA, Q_BETA

RULE_CHOICES = RULES  # re-export for callers


@dataclass
class RescreenResult:
    """Aggregated walk-forward book for one screening rule."""

    rule: str
    hedge: str
    sharpe: float
    total_return: float
    max_drawdown: float
    trades: int
    n_windows: int
    cost_paid: float
    rehedge_cost: float
    mean_discoveries: float
    frac_windows_traded: float
    returns: pd.Series = field(repr=False)
    windows: pd.DataFrame = field(repr=False)

    def as_dict(self) -> dict[str, float]:
        return {
            "sharpe": self.sharpe,
            "total_return": self.total_return,
            "max_drawdown": self.max_drawdown,
            "trades": float(self.trades),
            "n_windows": float(self.n_windows),
            "cost_paid": self.cost_paid,
            "rehedge_cost": self.rehedge_cost,
            "mean_discoveries": self.mean_discoveries,
            "frac_windows_traded": self.frac_windows_traded,
        }


def _pair_block_returns(
    y: np.ndarray,
    x: np.ndarray,
    *,
    hedge: str,
    z_enter: float,
    z_exit: float,
    rate: float,
    n_legs: int,
    q_beta: float,
    q_alpha: float,
    formation: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """One trading-block PnL for a single screened pair (formation already known)."""
    yf, xf = y[:formation], x[:formation]
    yt, xt = y[formation:], x[formation:]
    z, beta, _ = _window_signal(yf, xf, yt, xt, hedge, q_beta, q_alpha)
    pos = mean_reversion_positions(
        pd.Series(z), z_enter=z_enter, z_exit=z_exit
    ).to_numpy(dtype=float, copy=True)
    pos[-1] = 0.0
    g, c, rh = _block_pnl(yt, xt, pos, beta, rate, n_legs)
    prev = np.concatenate([[0.0], pos[:-1]])
    n_tr = int(((prev == 0) & (pos != 0)).sum() + ((prev != 0) & (pos != 0) & (prev != pos)).sum())
    return g - c - rh, c, rh, n_tr


def universe_formation_backtest(
    prices: pd.DataFrame,
    *,
    rule: str = "bh",
    hedge: str = "ols",
    formation: int = FORMATION,
    trading: int = TRADING,
    z_enter: float = Z_ENTER,
    z_exit: float = Z_EXIT,
    spread_bps: float = SPREAD_BPS,
    slippage_bps: float = SLIPPAGE_BPS,
    n_legs: int = N_LEGS,
    q_beta: float = Q_BETA,
    q_alpha: float = Q_ALPHA,
    alpha: float = 0.05,
    maxlag: int | None = 1,
) -> RescreenResult:
    """Roll formation/trading windows; screen on formation, trade discoveries.

    ``rule`` must be one of ``screening.RULES``. When a window yields zero
    discoveries the book stays flat (zero return, zero cost) for that block.
    When there are several, each pair is traded independently and the block
    return is the equal-weight average of their costed returns.
    """
    if rule not in RULES:
        raise ValueError(f"rule must be one of {RULES}, got {rule!r}")
    if hedge not in ("ols", "kalman"):
        raise ValueError("hedge must be 'ols' or 'kalman'")
    clean = prices.astype(float).dropna().sort_index()
    if formation < 30 or trading < 2:
        raise ValueError("formation must be >= 30 bars and trading >= 2 bars")
    if len(clean) < formation + trading:
        raise ValueError("Sample shorter than one formation + trading window.")
    if clean.shape[1] < 2:
        raise ValueError("Need at least two price columns.")

    rate = (float(spread_bps) + float(slippage_bps)) / 1e4
    n = len(clean)
    rets = pd.Series(np.nan, index=clean.index, name="ret")
    cost_total = rehedge_total = 0.0
    trades = 0
    win_rows: list[dict] = []
    start = 0
    while start + formation + trading <= n:
        f_sl = slice(start, start + formation)
        t_sl = slice(start + formation, start + formation + trading)
        form_px = clean.iloc[f_sl]
        screen = screen_pairs(form_px, alpha=alpha, maxlag=maxlag)
        selected = screen.loc[screen[rule]].reset_index(drop=True)
        n_disc = int(len(selected))
        block_idx = clean.index[t_sl]
        if n_disc == 0:
            rets.loc[block_idx] = 0.0
            win_rows.append(
                {
                    "trade_start": block_idx[0],
                    "trade_end": block_idx[-1],
                    "discoveries": 0,
                    "net_return": 0.0,
                    "trades": 0,
                }
            )
        else:
            pair_rets = []
            block_cost = block_rh = 0.0
            block_trades = 0
            for _, row in selected.iterrows():
                a, b = row["a"], row["b"]
                # Prefer the ordering with the smaller EG p (a on b vs b on a).
                if float(row["eg_p_ab"]) <= float(row["eg_p_ba"]):
                    y_col, x_col = a, b
                else:
                    y_col, x_col = b, a
                y = clean[y_col].to_numpy()[start : start + formation + trading]
                x = clean[x_col].to_numpy()[start : start + formation + trading]
                r, c, rh, n_tr = _pair_block_returns(
                    y,
                    x,
                    hedge=hedge,
                    z_enter=z_enter,
                    z_exit=z_exit,
                    rate=rate,
                    n_legs=n_legs,
                    q_beta=q_beta,
                    q_alpha=q_alpha,
                    formation=formation,
                )
                pair_rets.append(r)
                block_cost += float(c.sum())
                block_rh += float(rh.sum())
                block_trades += n_tr
            avg = np.mean(np.vstack(pair_rets), axis=0)
            rets.loc[block_idx] = avg
            # Costs are averages across equal-weighted pairs so they stay on the
            # same scale as the averaged returns.
            cost_total += block_cost / n_disc
            rehedge_total += block_rh / n_disc
            trades += block_trades
            win_rows.append(
                {
                    "trade_start": block_idx[0],
                    "trade_end": block_idx[-1],
                    "discoveries": n_disc,
                    "net_return": float(np.prod(1.0 + avg) - 1.0),
                    "trades": block_trades,
                }
            )
        start += trading

    traded = rets.dropna()
    windows = pd.DataFrame(win_rows)
    n_windows = len(windows)
    return RescreenResult(
        rule=rule,
        hedge=hedge,
        sharpe=_sharpe(traded),
        total_return=float((1.0 + traded).prod() - 1.0) if len(traded) else 0.0,
        max_drawdown=_max_drawdown(traded),
        trades=trades,
        n_windows=n_windows,
        cost_paid=cost_total,
        rehedge_cost=rehedge_total,
        mean_discoveries=float(windows["discoveries"].mean()) if n_windows else 0.0,
        frac_windows_traded=float((windows["discoveries"] > 0).mean()) if n_windows else 0.0,
        returns=traded,
        windows=windows,
    )


def compare_rules(
    prices: pd.DataFrame,
    rules: tuple[str, ...] = ("naive", "bh", "bh_johansen"),
    **kwargs,
) -> pd.DataFrame:
    """Run the same walk-forward under several screening rules."""
    rows = {r: universe_formation_backtest(prices, rule=r, **kwargs).as_dict() for r in rules}
    return pd.DataFrame(rows).T
