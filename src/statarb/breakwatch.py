"""Causal break watch for a traded spread: a Page CUSUM on the block z-score.

The stops of ``signals.mean_reversion_positions`` react to one bad move. A pair
that has stopped being a pair is better caught by an accumulating statistic: the
z-score of a healthy spread (formation mean/std) has mean zero, so its running
sum should wander, not drift. A one-sided Page CUSUM on each side flags a drift.

The z-score of an OU spread is serially correlated, so the sum of ``n`` bars has
variance ``n * LRV`` with the long-run variance ``LRV = (1 + rho) / (1 - rho)``
(AR(1), unit marginal variance), not ``n``. Dividing z by ``sqrt(LRV)`` makes the
sum a unit-variance random walk in the long run, which is what the fixed
``k`` (slack, in those units per bar) and ``h`` (alarm level) assume. ``rho``
comes from the *formation* window, never from the bars being watched.

    u_t  = z_t / sqrt(LRV)
    S+_t = max(0, S+_{t-1} + u_t - k)        S-_t = max(0, S-_{t-1} - u_t - k)
    alarm at the first t with max(S+_t, S-_t) > h

``cusum_alarm`` returns the alarm bar (or ``None``); the book is expected to stay
flat from that bar to the end of the block. Only bars up to and including t are
read, so an alarm can never depend on the future.
"""

from __future__ import annotations

import numpy as np

K_SLACK = 0.25
H_ALARM = 5.0
RHO_CLIP = (-0.5, 0.98)


def long_run_scale(rho: float) -> float:
    """``sqrt((1 + rho) / (1 - rho))`` with rho clipped to a safe AR(1) range."""
    r = float(np.clip(rho, *RHO_CLIP))
    return float(np.sqrt((1.0 + r) / (1.0 - r)))


def cusum_path(z: np.ndarray, rho: float = 0.0, k: float = K_SLACK) -> np.ndarray:
    """Running ``max(S+, S-)`` of the two-sided CUSUM on ``z / sqrt(LRV)``.

    A NaN z carries the statistics forward unchanged (no information).
    """
    if k < 0:
        raise ValueError("k must be >= 0")
    scale = long_run_scale(rho)
    zz = np.asarray(z, dtype=float)
    out = np.zeros(len(zz))
    s_pos = s_neg = 0.0
    for i, zi in enumerate(zz):
        if np.isfinite(zi):
            u = zi / scale
            s_pos = max(0.0, s_pos + u - k)
            s_neg = max(0.0, s_neg - u - k)
        out[i] = max(s_pos, s_neg)
    return out


def cusum_alarm(z: np.ndarray, rho: float = 0.0, k: float = K_SLACK, h: float = H_ALARM) -> int | None:
    """Index of the first bar whose CUSUM exceeds ``h``, or None."""
    if h <= 0:
        raise ValueError("h must be > 0")
    hit = np.flatnonzero(cusum_path(z, rho, k) > float(h))
    return int(hit[0]) if len(hit) else None
