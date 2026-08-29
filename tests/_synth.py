"""Synthetic series used by unit tests. No market data."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_cointegrated_pair(
    n: int = 1500,
    beta: float = 1.6,
    intercept: float = 10.0,
    rho: float = 0.85,
    sigma_x: float = 1.0,
    sigma_e: float = 0.4,
    seed: int = 7,
    start: str = "2016-01-01",
) -> tuple[pd.Series, pd.Series, float, float]:
    """y = intercept + beta * x + AR(1) residual. x is a random walk."""
    rng = np.random.default_rng(seed)
    x = 50.0 + np.cumsum(rng.normal(0.0, sigma_x, n))
    e = np.zeros(n)
    shocks = rng.normal(0.0, sigma_e, n)
    for t in range(1, n):
        e[t] = rho * e[t - 1] + shocks[t]
    y = intercept + beta * x + e
    idx = pd.bdate_range(start, periods=n)
    return pd.Series(y, index=idx, name="y"), pd.Series(x, index=idx, name="x"), beta, intercept


def make_independent_walks(
    n: int = 1500, seed: int = 99, start: str = "2016-01-01"
) -> tuple[pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    y = pd.Series(80.0 + np.cumsum(rng.normal(0.0, 1.0, n)), index=idx, name="y")
    x = pd.Series(40.0 + np.cumsum(rng.normal(0.0, 1.0, n)), index=idx, name="x")
    return y, x
