"""Rolling formation/trading walk-forward: static OLS hedge vs Kalman hedge, with costs.

Windows (bars, non-overlapping trading blocks)::

    | formation F | trading T |
                  | formation F | trading T |   (origin advances by T)

Static OLS ("ols"): Engle-Granger OLS on the formation window gives (alpha, beta);
the trading-window spread ``y - alpha - beta*x`` is z-scored with the
*formation* residual mean/std. beta is frozen for the whole trading block.

Kalman ("kalman"): ``kalman_hedge`` is run on formation+trading with
``warmup = F``, i.e. initialised by OLS on the same formation window, and then
filtered bar by bar through the trading block. The signal is the innovation
z-score (prior state, so it has not seen bar t); the hedge held over (t, t+1]
is the filtered ``beta[t]``.

Both books use the same ``mean_reversion_positions`` rule and thresholds, start
flat at the first trading bar and are forced flat on the last one, so no
position or state leaks between blocks except through the formation data.

Accounting (identical code path for both hedges)::

    gross_t  = |y_t| + |beta_t| * |x_t|
    ret_{t+1} = pos_t * ((y_{t+1} - y_t) - beta_t * (x_{t+1} - x_t)) / gross_t
    cost_t   = rate * [ n_legs * |pos_t - pos_{t-1}|
                        + |pos_t| * |beta_t - beta_{t-1}| * |x_t| / gross_t ]   (if pos unchanged)

with ``rate = (spread_bps + slippage_bps) / 1e4``. The first term is the
existing convention in ``backtest.apply_costs`` (each leg charged on gross
notional per unit change). The second is the re-hedge of the x leg when beta
moves while a position is held; it is zero for the static hedge. Cost is
charged on the bar the trade is made.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .backtest import _max_drawdown, _sharpe
from .kalman import Q_ALPHA, Q_BETA, kalman_hedge
from .params import N_LEGS, SLIPPAGE_BPS, SPREAD_BPS, Z_ENTER, Z_EXIT
from .signals import mean_reversion_positions

FORMATION = 252
TRADING = 63


@dataclass
class FormationResult:
    hedge: str
    sharpe: float
    total_return: float
    max_drawdown: float
    trades: int
    n_windows: int
    cost_paid: float
    rehedge_cost: float
    gross_return_sum: float
    returns: pd.Series = field(repr=False)
    positions: pd.Series = field(repr=False)
    beta: pd.Series = field(repr=False)
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
        }


def _window_signal(
    yf: np.ndarray, xf: np.ndarray, yt: np.ndarray, xt: np.ndarray, hedge: str,
    q_beta: float, q_alpha: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (z, beta) on the trading block and the formation beta."""
    ols = sm.OLS(yf, sm.add_constant(xf, has_constant="add")).fit()
    a, b = float(ols.params[0]), float(ols.params[1])
    if hedge == "ols":
        resid_f = yf - a - b * xf
        mu, sd = float(resid_f.mean()), float(resid_f.std(ddof=1))
        z = (yt - a - b * xt - mu) / sd
        return z, np.full(len(yt), b), b
    if hedge == "kalman":
        y = pd.Series(np.concatenate([yf, yt]))
        x = pd.Series(np.concatenate([xf, xt]))
        kf = kalman_hedge(y, x, q_beta=q_beta, q_alpha=q_alpha, warmup=len(yf))
        return kf["zscore"].to_numpy()[len(yf):], kf["beta"].to_numpy()[len(yf):], b
    raise ValueError("hedge must be 'ols' or 'kalman'")


def _block_pnl(
    y: np.ndarray, x: np.ndarray, pos: np.ndarray, beta: np.ndarray, rate: float, n_legs: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-bar (gross pnl return, trade cost, re-hedge cost) for one trading block.

    Bar i's return is earned by the book held from close i-1 to close i.
    """
    n = len(y)
    gross_ret = np.zeros(n)
    trade_cost = np.zeros(n)
    rehedge = np.zeros(n)
    notional = np.abs(y) + np.abs(beta) * np.abs(x)
    prev_pos, prev_beta = 0.0, beta[0]
    for i in range(n):
        if i > 0 and prev_pos != 0.0:
            gross_ret[i] = prev_pos * ((y[i] - y[i - 1]) - beta[i - 1] * (x[i] - x[i - 1])) / notional[i - 1]
        dpos = abs(pos[i] - prev_pos)
        trade_cost[i] = rate * n_legs * dpos
        if dpos == 0.0 and pos[i] != 0.0:
            rehedge[i] = rate * abs(pos[i]) * abs(beta[i] - prev_beta) * abs(x[i]) / notional[i]
        prev_pos, prev_beta = pos[i], beta[i]
    return gross_ret, trade_cost, rehedge


def formation_trading_backtest(
    y: pd.Series,
    x: pd.Series,
    *,
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
) -> FormationResult:
    """Roll formation/trading windows over the whole sample; see module docstring."""
    both = pd.concat(
        [pd.Series(y, dtype=float).rename("y"), pd.Series(x, dtype=float).rename("x")], axis=1
    ).dropna().sort_index()
    if formation < 30 or trading < 2:
        raise ValueError("formation must be >= 30 bars and trading >= 2 bars")
    if len(both) < formation + trading:
        raise ValueError("Sample shorter than one formation + trading window.")
    yv, xv = both["y"].to_numpy(), both["x"].to_numpy()
    rate = (float(spread_bps) + float(slippage_bps)) / 1e4

    n = len(both)
    rets = pd.Series(np.nan, index=both.index, name="ret")
    pos_s = pd.Series(np.nan, index=both.index, name="position")
    beta_s = pd.Series(np.nan, index=both.index, name="beta")
    cost_total = rehedge_total = gross_total = 0.0
    trades = 0
    win_rows = []
    start = 0
    while start + formation + trading <= n:
        f_sl = slice(start, start + formation)
        t_sl = slice(start + formation, start + formation + trading)
        z, beta, beta_f = _window_signal(
            yv[f_sl], xv[f_sl], yv[t_sl], xv[t_sl], hedge, q_beta, q_alpha
        )
        pos = mean_reversion_positions(
            pd.Series(z), z_enter=z_enter, z_exit=z_exit
        ).to_numpy(dtype=float, copy=True)
        pos[-1] = 0.0  # forced flat at block end
        g, c, rh = _block_pnl(yv[t_sl], xv[t_sl], pos, beta, rate, n_legs)
        r = g - c - rh
        idx = both.index[t_sl]
        rets.loc[idx] = r
        pos_s.loc[idx] = pos
        beta_s.loc[idx] = beta
        prev = np.concatenate([[0.0], pos[:-1]])
        n_tr = int(((prev == 0) & (pos != 0)).sum() + ((prev != 0) & (pos != 0) & (prev != pos)).sum())
        trades += n_tr
        cost_total += float(c.sum() + rh.sum())
        rehedge_total += float(rh.sum())
        gross_total += float(g.sum())
        win_rows.append(
            {
                "trade_start": idx[0],
                "trade_end": idx[-1],
                "beta_formation": beta_f,
                "beta_trade_end": float(beta[-1]),
                "net_return": float(np.prod(1.0 + r) - 1.0),
                "trades": n_tr,
            }
        )
        start += trading

    traded = rets.dropna()
    return FormationResult(
        hedge=hedge,
        sharpe=_sharpe(traded),
        total_return=float((1.0 + traded).prod() - 1.0),
        max_drawdown=_max_drawdown(traded),
        trades=trades,
        n_windows=len(win_rows),
        cost_paid=cost_total,
        rehedge_cost=rehedge_total,
        gross_return_sum=gross_total,
        returns=traded,
        positions=pos_s.dropna(),
        beta=beta_s.dropna(),
        windows=pd.DataFrame(win_rows),
    )


def compare_hedges(y: pd.Series, x: pd.Series, **kwargs) -> pd.DataFrame:
    """Run the same walk-forward with the static and the Kalman hedge."""
    rows = {h: formation_trading_backtest(y, x, hedge=h, **kwargs).as_dict() for h in ("ols", "kalman")}
    return pd.DataFrame(rows).T
