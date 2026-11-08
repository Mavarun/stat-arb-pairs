"""Half-life-calibrated level for the Engle-Granger formation gate.

The Nov 3 gate (``formation_trading_backtest(gate_p=...)``) uses one fixed
p-value for every pair. With 252 formation bars the Engle-Granger test has
little power on a slow spread, so a fixed ``p < 0.01`` gate keeps a healthy
half-life-5 pair flat in ~20% of blocks and a healthy half-life-20 pair in
~92%: for anything slow it is a rule against trading, not a break detector.

This module turns the gate around. Instead of asking "is this window
cointegrated at 1%?", it asks "is this window's p-value worse than what a
*healthy* pair with this pair's own half-life would show?". For a reference
half-life ``hl`` and window length ``n`` it simulates healthy pairs
(``synth.make_ou_pair``: OU spread with exact half-life, geometric-walk x leg)
and returns the ``quantile`` of their Engle-Granger p-values. Gating at that
level keeps a healthy pair of half-life ``hl`` flat in about ``1 - quantile`` of
windows by construction, whatever ``hl`` is.

What it cannot do: a slow healthy pair has a wide p distribution, so the level
is high (about 0.7 at half-life 20, n = 252) and a dead pair's p must exceed it
to be gated. The power against a break therefore falls with the half-life; the
calibration only fixes the *premium*, it does not create information.

The simulation is seeded and cached on a log-spaced half-life grid
(``HL_GRID_STEP`` in log units), so the same reference always gets the same
level and repeated windows cost nothing.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from statsmodels.tsa.stattools import coint

from .synth import make_ou_pair

GATE_QUANTILE = 0.95
N_SIMS = 400
SIM_SEED = 20_000
HL_GRID_STEP = 0.05
HL_RANGE = (1.0, 250.0)


def _snap(half_life: float) -> float:
    """Clip to ``HL_RANGE`` and round to the log grid used for caching."""
    hl = float(np.clip(float(half_life), *HL_RANGE))
    return float(np.exp(np.round(np.log(hl) / HL_GRID_STEP) * HL_GRID_STEP))


@lru_cache(maxsize=512)
def _healthy_pvalues_cached(hl: float, n: int, n_sims: int, seed: int) -> tuple[float, ...]:
    out = []
    for i in range(n_sims):
        p = make_ou_pair(n, half_life=hl, seed=seed + i)
        out.append(float(coint(p.y.to_numpy(), p.x.to_numpy(), trend="c", autolag="aic")[1]))
    return tuple(out)


def healthy_eg_pvalues(
    half_life: float, n: int, *, n_sims: int = N_SIMS, seed: int = SIM_SEED
) -> np.ndarray:
    """Engle-Granger p-values (y on x) of ``n_sims`` simulated healthy pairs.

    ``half_life`` is snapped to the cache grid (see ``_snap``). Seeds
    ``seed .. seed + n_sims - 1`` are used, so the result is reproducible.
    """
    if not np.isfinite(half_life) or half_life <= 0:
        raise ValueError("half_life must be positive and finite")
    if int(n) < 30:
        raise ValueError("n must be >= 30")
    if int(n_sims) < 20:
        raise ValueError("n_sims must be >= 20")
    return np.asarray(_healthy_pvalues_cached(_snap(half_life), int(n), int(n_sims), int(seed)))


def calibrated_gate_level(
    half_life: float,
    n: int,
    *,
    quantile: float = GATE_QUANTILE,
    n_sims: int = N_SIMS,
    seed: int = SIM_SEED,
) -> float:
    """p-value level above which a window is worse than ``quantile`` of healthy ones.

    A non-finite or non-positive ``half_life`` (the reference fit found no mean
    reversion) returns 1.0, i.e. the gate never fires: there is no healthy
    reference to compare against, and the pair should not have been selected.
    """
    if not 0.5 <= float(quantile) < 1.0:
        raise ValueError("quantile must be in [0.5, 1)")
    if not np.isfinite(half_life) or half_life <= 0:
        return 1.0
    p = healthy_eg_pvalues(half_life, n, n_sims=n_sims, seed=seed)
    return float(np.quantile(p, float(quantile)))
