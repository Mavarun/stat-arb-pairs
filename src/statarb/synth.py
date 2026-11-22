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


BREAK_KINDS = ("none", "random_walk", "level_shift")


def make_beta_jump_pair(
    n: int = 1000,
    *,
    break_at: int,
    jump: float,
    beta: float = 1.5,
    seed: int = 0,
    **kwargs,
) -> SyntheticPair:
    """``make_ou_pair`` whose hedge ratio jumps by ``jump`` (relative) at bar ``break_at``.

    ``beta_t = beta`` before ``break_at`` and ``beta * (1 + jump)`` from it on. The
    spread stays the same mean-reverting OU, so the pair is still a pair; only the
    hedge ratio a formation fit must find has moved. With ``jump = 0`` the result is
    bit-identical to ``make_ou_pair`` for the same seed, and every ``jump`` shares the
    x leg and the spread path (common random numbers across jump sizes).

    A formation window that straddles the jump fits a blend of the two ratios, so the
    next window's re-fit can move z a long way: the case the Nov 19 slice could not
    produce (no carried position flipped there).
    """
    if not 1 <= int(break_at) < n:
        raise ValueError("break_at must be in [1, n)")
    if not np.isfinite(jump) or jump <= -1.0:
        raise ValueError("jump must be finite and > -1")
    path = np.full(n, float(beta))
    path[int(break_at):] = float(beta) * (1.0 + float(jump))
    return make_ou_pair(n, beta=beta, beta_path=path, seed=seed, **kwargs)


def make_beta_trend_pair(
    n: int = 1000,
    *,
    trend_from: int,
    drift_per_year: float,
    beta: float = 1.5,
    bars_per_year: int = 252,
    seed: int = 0,
    **kwargs,
) -> SyntheticPair:
    """``make_ou_pair`` whose hedge ratio trends linearly from bar ``trend_from`` on.

    ``beta_t = beta`` before ``trend_from`` and
    ``beta * (1 + drift_per_year * (t - trend_from) / bars_per_year)`` from it on, so the
    ratio moves by ``drift_per_year`` (relative to ``beta``) every ``bars_per_year`` bars
    and never settles. The spread stays the same mean-reverting OU. Unlike
    ``make_beta_jump_pair``, no formation window is ever clean after ``trend_from``: a
    fit on the last ``F`` bars estimates roughly the ratio of ``F / 2`` bars ago, so every
    re-fit lags. With ``drift_per_year = 0`` the result is bit-identical to
    ``make_ou_pair`` for the same seed, and every drift shares the x leg and the spread
    path (common random numbers across drifts).
    """
    if not 0 <= int(trend_from) < n:
        raise ValueError("trend_from must be in [0, n)")
    if not np.isfinite(drift_per_year):
        raise ValueError("drift_per_year must be finite")
    if int(bars_per_year) < 1:
        raise ValueError("bars_per_year must be >= 1")
    t = np.arange(n, dtype=float) - int(trend_from)
    path = float(beta) * (1.0 + float(drift_per_year) * np.clip(t, 0.0, None) / int(bars_per_year))
    if np.any(path <= 0):
        raise ValueError("drift_per_year drives the hedge ratio to <= 0 inside the sample")
    return make_ou_pair(n, beta=beta, beta_path=path, seed=seed, **kwargs)


def make_broken_pair(
    n: int = 1000,
    *,
    break_at: int,
    kind: str = "random_walk",
    shift_sigmas: float = 4.0,
    half_life: float = 5.0,
    sigma_spread: float = 1.0,
    seed: int = 0,
    **kwargs,
) -> SyntheticPair:
    """``make_ou_pair`` whose cointegration breaks at bar ``break_at``.

    ``kind``:

    * ``"random_walk"``: from ``break_at`` on, the spread is a driftless random
      walk with the OU innovation scale (``rho = 1``): the pair has stopped
      being a pair and nothing pulls the spread back.
    * ``"level_shift"``: the OU keeps mean-reverting but around a new mean
      ``shift_sigmas * sigma_spread`` away (a permanent re-rating of one leg).
    * ``"none"``: the unbroken pair (bit-identical to ``make_ou_pair``).

    The pre-break path, the x leg and the hedge ratio are identical across
    kinds for the same seed, so books can be compared on common random numbers.
    Post-break innovations come from a second generator seeded by ``seed``.
    """
    if kind not in BREAK_KINDS:
        raise ValueError(f"kind must be one of {BREAK_KINDS}")
    if not 1 <= int(break_at) < n:
        raise ValueError("break_at must be in [1, n)")
    base = make_ou_pair(n, half_life=half_life, sigma_spread=sigma_spread, seed=seed, **kwargs)
    if kind == "none":
        return base
    s = base.spread.to_numpy(copy=True)
    b = int(break_at)
    if kind == "level_shift":
        s[b:] += float(shift_sigmas) * float(sigma_spread)
    else:
        rng = np.random.default_rng([int(seed), 1])
        sigma_eps = float(sigma_spread) * np.sqrt(1.0 - base.rho**2)
        s[b:] = s[b - 1] + np.cumsum(rng.normal(0.0, sigma_eps, n - b))
    y = base.y.to_numpy() - base.spread.to_numpy() + s
    idx = base.y.index
    return SyntheticPair(
        y=pd.Series(y, index=idx, name="y"),
        x=base.x,
        beta=base.beta,
        intercept=base.intercept,
        half_life=base.half_life,
        rho=base.rho,
        sigma_spread=base.sigma_spread,
        spread=pd.Series(s, index=idx, name="true_spread"),
        beta_path=base.beta_path,
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
    half_life: float | list[float] | tuple[float, ...] = 8.0,
    sigma_spread: float = 1.0,
    seed: int = 0,
    start: str = "2015-01-01",
) -> tuple[pd.DataFrame, set[frozenset[str]]]:
    """Random-walk noise assets plus ``n_pairs`` planted cointegrated pairs.

    Returns the price frame and the set of true pairs (unordered names).
    Planted pairs use independent x legs, so no cross-pair cointegration.

    ``half_life`` may be a scalar (every planted pair shares it) or a sequence
    of length ``n_pairs`` so ranking tests can plant a mix of fast and slow
    mean-reverters in one universe.
    """
    if np.isscalar(half_life):
        half_lives = [float(half_life)] * int(n_pairs)
    else:
        half_lives = [float(h) for h in half_life]
        if len(half_lives) != int(n_pairs):
            raise ValueError("half_life sequence must have length n_pairs")
    rng = np.random.default_rng(seed)
    frame = make_random_walk_universe(n_noise, n, seed=int(rng.integers(1 << 31)), start=start)
    truth: set[frozenset[str]] = set()
    for k in range(n_pairs):
        beta = float(rng.uniform(0.5, 2.0))
        pair = make_ou_pair(
            n,
            beta=beta,
            intercept=float(rng.uniform(-5, 5)),
            half_life=half_lives[k],
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
