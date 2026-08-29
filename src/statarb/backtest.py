"""Long-short spread backtest with explicit costs and a date-based walk-forward.

Execution assumption (optimistic, documented): z-score at close t sets
``position[t]``; that position earns the close-to-close move from t to t+1.
We do not model next-open slippage beyond the parameterized bps, nor
borrow fees, nor market impact beyond ``slippage_bps``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from .cointegration import EngleGrangerResult, engle_granger
from .params import (
    ANNUALIZATION,
    MIN_TRAIN_OBS,
    N_LEGS,
    SLIPPAGE_BPS,
    SPREAD_BPS,
    TRAIN_FRAC,
    Z_ENTER,
    Z_EXIT,
    Z_WINDOW,
)
from .signals import (
    compute_spread,
    mean_reversion_positions,
    rolling_zscore,
    zscore_from_stats,
)


def _sharpe(returns: pd.Series, periods: int = ANNUALIZATION) -> float:
    r = returns.dropna().astype(float)
    if len(r) < 2:
        return float("nan")
    sd = float(r.std(ddof=1))
    if sd == 0.0:
        return 0.0
    return float(r.mean() / sd * np.sqrt(periods))


def _max_drawdown(returns: pd.Series) -> float:
    r = returns.fillna(0.0).astype(float)
    equity = (1.0 + r).cumprod()
    if equity.empty:
        return float("nan")
    dd = equity / equity.cummax() - 1.0
    return float(dd.min())


def _trade_count(position: pd.Series) -> int:
    """Number of times the book leaves zero or flips sign (entries + flips)."""
    p = position.fillna(0.0).astype(float)
    prev = p.shift(1).fillna(0.0)
    entered = (prev == 0.0) & (p != 0.0)
    flipped = (prev != 0.0) & (p != 0.0) & (np.sign(prev) != np.sign(p))
    return int((entered | flipped).sum())


def _turnover(position: pd.Series) -> float:
    """Mean daily |\u0394position| (1 = a full flip of a unit book that day)."""
    p = position.fillna(0.0).astype(float)
    return float(p.diff().abs().fillna(p.abs()).mean())


def hedge_portfolio_returns(
    y: pd.Series,
    x: pd.Series,
    hedge_ratio: float,
) -> pd.Series:
    """Close-to-close return of long 1 y vs short ``hedge_ratio`` x, / lagged gross notional."""
    y = pd.Series(y, dtype=float)
    x = pd.Series(x, dtype=float)
    dy = y.diff()
    dx = x.diff()
    pnl = dy - float(hedge_ratio) * dx
    gross = y.shift(1).abs() + abs(float(hedge_ratio)) * x.shift(1).abs()
    hp = pnl / gross
    hp.name = "hp_ret"
    return hp.replace([np.inf, -np.inf], np.nan)


def apply_costs(
    position: pd.Series,
    spread_bps: float,
    slippage_bps: float,
    n_legs: int = N_LEGS,
) -> pd.Series:
    """Cost of position changes, booked on the first bar the new book is held.

    ``pos_held[t] = position[t-1]`` (signal at t-1 earns the move into t).
    Cost = |\u0394 pos_held| * n_legs * (spread + slippage) bps.
    """
    held = position.shift(1).fillna(0.0).astype(float)
    traded = held.diff().abs().fillna(held.abs())
    rate = n_legs * (float(spread_bps) + float(slippage_bps)) / 1e4
    cost = traded * rate
    cost.name = "cost"
    return cost


@dataclass
class BacktestResult:
    total_return: float
    sharpe: float
    max_drawdown: float
    trade_count: int
    turnover: float
    n_obs: int
    hedge_ratio: float
    intercept: float
    returns: pd.Series = field(repr=False)
    equity: pd.Series = field(repr=False)
    positions: pd.Series = field(repr=False)
    spread: pd.Series = field(repr=False)
    zscore: pd.Series = field(repr=False)
    coint_pvalue: float | None = None

    def as_dict(self) -> dict[str, float]:
        return {
            "total_return": self.total_return,
            "sharpe": self.sharpe,
            "max_drawdown": self.max_drawdown,
            "trade_count": float(self.trade_count),
            "turnover": self.turnover,
            "n_obs": float(self.n_obs),
            "hedge_ratio": self.hedge_ratio,
            "intercept": self.intercept,
            "coint_pvalue": float("nan") if self.coint_pvalue is None else self.coint_pvalue,
        }


def backtest_spread(
    y: pd.Series,
    x: pd.Series,
    *,
    hedge_ratio: float,
    intercept: float = 0.0,
    zscore: pd.Series | None = None,
    z_window: int = Z_WINDOW,
    z_enter: float = Z_ENTER,
    z_exit: float = Z_EXIT,
    spread_bps: float = SPREAD_BPS,
    slippage_bps: float = SLIPPAGE_BPS,
    positions: pd.Series | None = None,
    coint_pvalue: float | None = None,
) -> BacktestResult:
    """Backtest a mean-reversion book on one hedge ratio.

    If ``positions`` is omitted, they are built from ``zscore`` (or a rolling
    z-score of the residual) with the default enter/exit rules.
    """
    y = pd.Series(y, dtype=float).rename("y")
    x = pd.Series(x, dtype=float).rename("x")
    frame = pd.concat([y, x], axis=1).dropna()
    y, x = frame["y"], frame["x"]
    spread = compute_spread(y, x, hedge_ratio, intercept)

    if positions is None:
        if zscore is None:
            zscore = rolling_zscore(spread, window=z_window)
        else:
            zscore = pd.Series(zscore, dtype=float).reindex(spread.index)
        positions = mean_reversion_positions(zscore, z_enter=z_enter, z_exit=z_exit)
    else:
        positions = pd.Series(positions, dtype=float).reindex(spread.index).fillna(0.0)
        if zscore is None:
            zscore = rolling_zscore(spread, window=z_window)
        else:
            zscore = pd.Series(zscore, dtype=float).reindex(spread.index)

    hp = hedge_portfolio_returns(y, x, hedge_ratio)
    held = positions.shift(1).fillna(0.0)
    cost = apply_costs(positions, spread_bps, slippage_bps)
    rets = (held * hp - cost).fillna(0.0)
    rets.name = "ret"
    equity = (1.0 + rets).cumprod()
    equity.name = "equity"
    return BacktestResult(
        total_return=float(equity.iloc[-1] - 1.0) if len(equity) else float("nan"),
        sharpe=_sharpe(rets),
        max_drawdown=_max_drawdown(rets),
        trade_count=_trade_count(positions),
        turnover=_turnover(positions),
        n_obs=int(len(rets)),
        hedge_ratio=float(hedge_ratio),
        intercept=float(intercept),
        returns=rets,
        equity=equity,
        positions=positions,
        spread=spread,
        zscore=zscore,
        coint_pvalue=coint_pvalue,
    )


@dataclass
class WalkForwardResult:
    """In-sample (train, overfit by construction) vs test, costed and uncosted.

    Test metrics are the ones that can possibly be believed; even those are
    one split on one pair, not a live track record.
    """

    train_end: pd.Timestamp
    z_mode: str
    engle_granger: EngleGrangerResult
    train_uncosted: BacktestResult
    train_costed: BacktestResult
    test_uncosted: BacktestResult
    test_costed: BacktestResult

    def metrics_table(self) -> pd.DataFrame:
        rows = {
            "train_uncosted": self.train_uncosted.as_dict(),
            "train_costed": self.train_costed.as_dict(),
            "test_uncosted": self.test_uncosted.as_dict(),
            "test_costed": self.test_costed.as_dict(),
        }
        return pd.DataFrame(rows).T


def _split_xy(
    y: pd.Series, x: pd.Series, train_end: pd.Timestamp
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    y = pd.Series(y, dtype=float)
    x = pd.Series(x, dtype=float)
    both = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna().sort_index()
    train = both.loc[:train_end]
    test = both.loc[both.index > train_end]
    if train.empty or test.empty:
        raise ValueError("Train or test window is empty; pick a train_end inside the sample.")
    return train["y"], train["x"], test["y"], test["x"]


def walk_forward(
    y: pd.Series,
    x: pd.Series,
    *,
    train_end: pd.Timestamp | str | None = None,
    train_frac: float = TRAIN_FRAC,
    z_enter: float = Z_ENTER,
    z_exit: float = Z_EXIT,
    z_window: int = Z_WINDOW,
    z_mode: str = "train_stats",
    spread_bps: float = SPREAD_BPS,
    slippage_bps: float = SLIPPAGE_BPS,
    min_train: int = MIN_TRAIN_OBS,
) -> WalkForwardResult:
    """Date-based walk-forward: fit hedge + z stats on train, trade test only.

    ``z_mode='train_stats'`` (default): freeze residual mean/std from train.
    ``z_mode='rolling'``: rolling z on the concatenated series (window uses
    only past points) but PnL is still scored only on the test dates.

    Hedge ratio is **never** re-estimated on test.
    """
    y = pd.Series(y, dtype=float)
    x = pd.Series(x, dtype=float)
    both = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna().sort_index()
    if train_end is None:
        cut = int(np.floor(len(both) * float(train_frac)))
        cut = max(cut, min_train)
        if cut >= len(both):
            raise ValueError("Not enough observations for a train/test split.")
        train_end = both.index[cut - 1]
    train_end = pd.Timestamp(train_end)

    y_tr, x_tr, y_te, x_te = _split_xy(both["y"], both["x"], train_end)
    if len(y_tr) < min_train:
        # Allow shorter synthetic tests; the default is a research warning, not a hard lock.
        if len(y_tr) < 30:
            raise ValueError(f"Train window too short: {len(y_tr)} obs")

    eg = engle_granger(y_tr, x_tr)
    spread_all = compute_spread(both["y"], both["x"], eg.hedge_ratio, eg.intercept)
    spread_tr = spread_all.loc[:train_end]
    spread_te = spread_all.loc[spread_all.index > train_end]

    if z_mode == "train_stats":
        mu = float(spread_tr.mean())
        sd = float(spread_tr.std(ddof=1))
        z_all = zscore_from_stats(spread_all, mu, sd)
    elif z_mode == "rolling":
        z_all = rolling_zscore(spread_all, window=z_window)
    else:
        raise ValueError("z_mode must be 'train_stats' or 'rolling'")

    pos_all = mean_reversion_positions(z_all, z_enter=z_enter, z_exit=z_exit)

    def _run(ys, xs, z, pos, cost_spread, cost_slip):
        return backtest_spread(
            ys,
            xs,
            hedge_ratio=eg.hedge_ratio,
            intercept=eg.intercept,
            zscore=z,
            positions=pos,
            spread_bps=cost_spread,
            slippage_bps=cost_slip,
            coint_pvalue=eg.pvalue,
        )

    z_tr, pos_tr = z_all.loc[:train_end], pos_all.loc[:train_end]
    z_te, pos_te = z_all.loc[z_all.index > train_end], pos_all.loc[pos_all.index > train_end]
    # Flatten the first test bar so we do not inherit a train position into OOS.
    if len(pos_te):
        pos_te = pos_te.copy()
        pos_te.iloc[0] = 0.0

    return WalkForwardResult(
        train_end=train_end,
        z_mode=z_mode,
        engle_granger=eg,
        train_uncosted=_run(y_tr, x_tr, z_tr, pos_tr, 0.0, 0.0),
        train_costed=_run(y_tr, x_tr, z_tr, pos_tr, spread_bps, slippage_bps),
        test_uncosted=_run(y_te, x_te, z_te, pos_te, 0.0, 0.0),
        test_costed=_run(y_te, x_te, z_te, pos_te, spread_bps, slippage_bps),
    )


def expanding_walk_forward(
    y: pd.Series,
    x: pd.Series,
    n_splits: int = 4,
    **kwargs,
) -> list[WalkForwardResult]:
    """Expanding-origin walk-forward using ``sklearn.model_selection.TimeSeriesSplit``.

    Each fold re-estimates the Engle-Granger hedge on that fold's train window
    and trades only the later test window. Folds are not independent.
    """
    y = pd.Series(y, dtype=float).rename("y")
    x = pd.Series(x, dtype=float).rename("x")
    both = pd.concat([y, x], axis=1).dropna().sort_index()
    tscv = TimeSeriesSplit(n_splits=n_splits)
    out: list[WalkForwardResult] = []
    for train_idx, _test_idx in tscv.split(both):
        train_end = both.index[train_idx[-1]]
        out.append(walk_forward(both["y"], both["x"], train_end=train_end, **kwargs))
    return out
