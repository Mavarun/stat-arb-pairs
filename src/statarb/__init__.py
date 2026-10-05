"""Engle-Granger pairs-trading research slice.

Trading signals use Engle-Granger (OLS hedge ratio + residual). Johansen is
exposed as a diagnostic helper only — this package does not turn Johansen
eigenvectors into positions.
"""

from .backtest import backtest_spread, expanding_walk_forward, walk_forward
from .cointegration import EngleGrangerResult, JohansenResult, engle_granger, johansen
from .data import download_pair, load_csv
from .kalman import kalman_hedge, static_hedge_path
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
from .signals import compute_spread, mean_reversion_positions, rolling_zscore

__version__ = "0.1.0"

__all__ = [
    "ANNUALIZATION",
    "MIN_TRAIN_OBS",
    "N_LEGS",
    "SLIPPAGE_BPS",
    "SPREAD_BPS",
    "TRAIN_FRAC",
    "Z_ENTER",
    "Z_EXIT",
    "Z_WINDOW",
    "EngleGrangerResult",
    "JohansenResult",
    "backtest_spread",
    "compute_spread",
    "download_pair",
    "engle_granger",
    "expanding_walk_forward",
    "johansen",
    "kalman_hedge",
    "load_csv",
    "mean_reversion_positions",
    "rolling_zscore",
    "static_hedge_path",
    "walk_forward",
]
