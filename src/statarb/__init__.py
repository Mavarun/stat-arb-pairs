"""Engle-Granger pairs-trading research slice.

Trading signals use Engle-Granger (OLS hedge ratio + residual) or a Kalman
hedge. Johansen is a diagnostic and a confirmation filter in the pair screen
(``statarb.screening``); this package does not turn Johansen eigenvectors into
positions. Per-window universe re-screening and nested Kalman ``q_beta`` tuning
live in ``statarb.rescreen`` and ``statarb.nested``.
"""

from .backtest import backtest_spread, expanding_walk_forward, walk_forward
from .cointegration import EngleGrangerResult, JohansenResult, engle_granger, johansen
from .data import download_pair, load_csv
from .kalman import kalman_hedge, static_hedge_path
from .screening import benjamini_hochberg, screen_pairs, screen_summary
from .rescreen import compare_rules, universe_formation_backtest
from .nested import compare_fixed_vs_nested, nested_kalman_backtest, tune_q_beta
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
    "benjamini_hochberg",
    "compare_fixed_vs_nested",
    "compare_rules",
    "compute_spread",
    "download_pair",
    "engle_granger",
    "expanding_walk_forward",
    "johansen",
    "kalman_hedge",
    "load_csv",
    "mean_reversion_positions",
    "nested_kalman_backtest",
    "rolling_zscore",
    "screen_pairs",
    "screen_summary",
    "static_hedge_path",
    "tune_q_beta",
    "universe_formation_backtest",
    "walk_forward",
]
