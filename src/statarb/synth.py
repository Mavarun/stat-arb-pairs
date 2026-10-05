"""Synthetic price generators with *known* parameters, for recovery tests.

Everything here is seeded and offline. Nothing is market data.

Model for one cointegrated pair::

    x_t = x0 * exp(cumsum(N(0, vol_x)))           # geometric random walk, stays > 0
    s_t = rho * s_{t-1} + N(0, sigma_eps)         # AR(1) = discretised OU spread
    y_t = intercept + beta_t * x_t + s_t

with ``rho = 0.5 ** (1 / half_life)`` so the planted half-life is exact, and
``sigma_eps = sigma_spread * sqrt(1 - rho**2)`` so the *stationary* std of the
spread is ``sigma_spread``. ``s_0`` is drawn from the stationary distribution
(no burn-in transient). ``beta_t`` is constant unless a path is passed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd


def half_life_to_rho(half_life: float) -> float:
    """AR(1) coefficient whose shocks decay by half after ``half_life`` bars."""
    if not np.isfinite(half_life) or half_life <= 0:
        raise ValueError("half_life must be positive and finite")
    return float(0.5 ** (1.0 / float(half_life)))


@dataclass(frozen=True)
class SyntheticPair:
    """A generated pair plus the ground truth used to build it."""

    y: pd.Series
    x: pd.Series
    beta: float
    intercept: float
    half_life: float
    rho: float
    sigma_spread: float
    spread: pd.Series = field(repr=False)
    beta_path: pd.Series = field(repr=False)


def _geometric_walk(rng: np.random.Generator, n: int, x0: float, vol: float) -> np.ndarray:
    return float(x0) * np.exp(np.cumsum(rng.normal(0.0, float(vol), n)))


def make_ou_pair(
    n: int = 1000,
    *,
    beta: float = 1.5,
    intercept: float = 5.0,
    half_life: float = 10.0,
    sigma_spread: float = 1.0,
    x0: float = 100.0,
    vol_x: float = 0.01,
    beta_path: np.ndarray | None = None,
    seed: int = 0,
    start: str = "2015-01-01",
) -> SyntheticPair:
    """Cointegrated pair with planted hedge ratio, intercept and OU half-life.

    ``beta_path`` (length ``n``) overrides the constant ``beta`` to create a
    drifting hedge ratio; ``beta`` is then reported as the path mean.
    """
    if n < 10:
        raise ValueError("n must be >= 10")
    rng = np.random.default_rng(seed)
    rho = half_life_to_rho(half_life)
    x = _geometric_walk(rng, n, x0, vol_x)
    sigma_eps = float(sigma_spread) * np.sqrt(1.0 - rho**2)
    shocks = rng.normal(0.0, sigma_eps, n)
    s = np.empty(n)
    s[0] = rng.normal(0.0, float(sigma_spread))
    for t in range(1, n):
        s[t] = rho * s[t - 1] + shocks[t]
    if beta_path is None:
        b = np.full(n, float(beta))
    else:
        b = np.asarray(beta_path, dtype=float)
        if b.shape != (n,):
            raise ValueError("beta_path must have length n")
    y = float(intercept) + b * x + s
    idx = pd.bdate_range(start, periods=n)
    return SyntheticPair(
        y=pd.Series(y, index=idx, name="y"),
        x=pd.Series(x, index=idx, name="x"),
        beta=float(b.mean()),
        intercept=float(intercept),
        half_life=float(half_life),
        rho=rho,
        sigma_spread=float(sigma_spread),
        spread=pd.Series(s, index=idx, name="true_spread"),
        beta_path=pd.Series(b, index=idx, name="true_beta"),
    )


def make_random_walk_universe(
    n_assets: int = 20,
    n: int = 750,
    *,
    x0: float = 100.0,
    vol: float = 0.01,
    seed: int = 0,
    start: str = "2015-01-01",
) -> pd.DataFrame:
    """Independent geometric random walks: **no** pair is cointegrated.

    Every "discovery" a screen makes on this frame is a false positive.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    data = {f"N{i:02d}": _geometric_walk(rng, n, x0, vol) for i in range(n_assets)}
    return pd.DataFrame(data, index=idx)


def make_planted_universe(
    n_noise: int = 16,
    n_pairs: int = 4,
    n: int = 750,
    *,
    half_life: float = 8.0,
    sigma_spread: float = 1.0,
    seed: int = 0,
    start: str = "2015-01-01",
) -> tuple[pd.DataFrame, set[frozenset[str]]]:
    """Random-walk noise assets plus ``n_pairs`` planted cointegrated pairs.

    Returns the price frame and the set of true pairs (unordered names).
    Planted pairs use independent x legs, so no cross-pair cointegration.
    """
    rng = np.random.default_rng(seed)
    frame = make_random_walk_universe(n_noise, n, seed=int(rng.integers(1 << 31)), start=start)
    truth: set[frozenset[str]] = set()
    for k in range(n_pairs):
        beta = float(rng.uniform(0.5, 2.0))
        pair = make_ou_pair(
            n,
            beta=beta,
            intercept=float(rng.uniform(-5, 5)),
            half_life=half_life,
            sigma_spread=sigma_spread,
            seed=int(rng.integers(1 << 31)),
            start=start,
        )
        ya, xa = f"P{k}Y", f"P{k}X"
        frame[ya] = pair.y.to_numpy()
        frame[xa] = pair.x.to_numpy()
        truth.add(frozenset((ya, xa)))
    return frame, truth


def n_pairs(n_assets: int) -> int:
    """Number of unordered candidate pairs in a universe of ``n_assets``."""
    return len(list(combinations(range(n_assets), 2)))
