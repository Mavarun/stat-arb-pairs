"""Spread, z-score, and mean-reversion positions.

Positions are in {-1, 0, +1} on the residual spread:
  +1 = long y, short hedge_ratio * x  (spread is cheap)
  -1 = short y, long hedge_ratio * x  (spread is rich)
   0 = flat

Enter when |z| exceeds ``z_enter``; flatten when |z| falls back through ``z_exit``.
No new trade is opened while already in a position unless we flattened first.

Optional time stop (``max_hold``): a position held for ``max_hold`` bars is
flattened even if z has not reverted, and the book must see |z| drop back
inside ``z_enter`` before re-entering the same side. The usual choice is a
small multiple of the OU half-life (see ``statarb.ou.OUFit.max_hold``): a
spread that has not reverted in ~3 half-lives is evidence the fit broke.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .params import Z_ENTER, Z_EXIT, Z_WINDOW


def compute_spread(
    y: pd.Series,
    x: pd.Series,
    hedge_ratio: float,
    intercept: float = 0.0,
) -> pd.Series:
    """OLS residual: y - intercept - hedge_ratio * x."""
    y = pd.Series(y, dtype=float)
    x = pd.Series(x, dtype=float)
    spread = y - float(intercept) - float(hedge_ratio) * x
    spread.name = "spread"
    return spread


def rolling_zscore(spread: pd.Series, window: int = Z_WINDOW) -> pd.Series:
    """Rolling z-score. The first ``window - 1`` points are NaN (no look-ahead fill)."""
    if window < 2:
        raise ValueError("z-score window must be >= 2")
    s = pd.Series(spread, dtype=float)
    mu = s.rolling(window, min_periods=window).mean()
    sd = s.rolling(window, min_periods=window).std(ddof=1)
    z = (s - mu) / sd
    # Zero-variance windows are undefined, not zero.
    z = z.mask(sd == 0.0, np.nan)
    z.name = "zscore"
    return z


def zscore_from_stats(spread: pd.Series, mean: float, std: float) -> pd.Series:
    """Z-score against frozen training mean/std (walk-forward, no test leakage)."""
    if std <= 0 or not np.isfinite(std):
        raise ValueError("Training spread std must be positive and finite.")
    z = (pd.Series(spread, dtype=float) - float(mean)) / float(std)
    z.name = "zscore"
    return z


def mean_reversion_positions(
    z: pd.Series,
    z_enter: float = Z_ENTER,
    z_exit: float = Z_EXIT,
    max_hold: int | None = None,
) -> pd.Series:
    """Stateful mean-reversion book from a z-score series.

    NaN z-scores force a flat book (we do not carry a position through undefined z).
    ``max_hold`` (bars, optional) is a time stop; see module docstring.
    """
    if max_hold is not None and int(max_hold) < 1:
        raise ValueError("max_hold must be >= 1 bar")
    if z_enter <= z_exit:
        raise ValueError("z_enter must be > z_exit")
    if z_enter <= 0 or z_exit < 0:
        raise ValueError("thresholds must be non-negative, with z_enter > 0")

    z_arr = pd.Series(z, dtype=float).to_numpy()
    pos = np.zeros(len(z_arr), dtype=float)
    current = 0.0
    held = 0
    blocked = 0.0  # side that hit the time stop; cannot re-enter until |z| < z_enter
    for i, zi in enumerate(z_arr):
        if not np.isfinite(zi):
            current = 0.0
            held = 0
            pos[i] = 0.0
            continue
        if blocked != 0.0 and abs(zi) < z_enter:
            blocked = 0.0
        if current != 0.0 and max_hold is not None and held >= int(max_hold):
            blocked = current
            current = 0.0
            held = 0
            pos[i] = 0.0
            continue
        prev = current
        if current == 0.0:
            if zi > z_enter and blocked != -1.0:
                current = -1.0
            elif zi < -z_enter and blocked != 1.0:
                current = 1.0
        elif current > 0.0:
            # Long spread: exit when z has reverted up through -z_exit.
            if zi > -z_exit:
                current = 0.0
                if zi > z_enter:
                    current = -1.0
        else:
            # Short spread: exit when z has reverted down through +z_exit.
            if zi < z_exit:
                current = 0.0
                if zi < -z_enter:
                    current = 1.0
        held = held + 1 if (current != 0.0 and current == prev) else (1 if current != 0.0 else 0)
        pos[i] = current
    return pd.Series(pos, index=pd.Series(z).index, name="position")
