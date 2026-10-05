"""Ornstein-Uhlenbeck fit of a spread: half-life, equilibrium level, thresholds.

Discrete-time estimator (exact for an AR(1) sampled OU)::

    s_t = a + b * s_{t-1} + u_t,      u_t ~ iid(0, sigma_u^2)

    kappa      = -ln(b)                 (per bar)
    half_life  = ln(2) / kappa          (bars)
    mu         = a / (1 - b)            (equilibrium spread)
    sigma_eq   = sigma_u / sqrt(1 - b^2) (stationary std of the spread)

If ``b >= 1`` the spread is not mean-reverting in this window: half-life is
``inf`` and ``sigma_eq`` is ``nan``. If ``b <= 0`` the spread oscillates
faster than one bar; half-life is reported as ``nan`` (not a meaningful OU).

The OLS slope ``b`` is biased downward in small samples (Kendall bias), so the
half-life is biased *short*. We report it as estimated; we do not correct it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm


@dataclass(frozen=True)
class OUFit:
    """AR(1)/OU estimate of a spread. All times are in bars."""

    b: float
    b_se: float
    mu: float
    sigma_eq: float
    kappa: float
    half_life: float
    nobs: int

    @property
    def is_mean_reverting(self) -> bool:
        return bool(0.0 < self.b < 1.0)

    def thresholds(self, z_enter: float, z_exit: float) -> dict[str, float]:
        """Spread levels for entry / exit at ``mu +/- z * sigma_eq``."""
        if not np.isfinite(self.sigma_eq):
            raise ValueError("No stationary std: spread is not mean-reverting in this window.")
        return {
            "short_entry": self.mu + z_enter * self.sigma_eq,
            "short_exit": self.mu + z_exit * self.sigma_eq,
            "long_entry": self.mu - z_enter * self.sigma_eq,
            "long_exit": self.mu - z_exit * self.sigma_eq,
        }

    def zscore(self, spread: pd.Series) -> pd.Series:
        """Z-score of a (possibly out-of-sample) spread against the frozen OU moments."""
        if not np.isfinite(self.sigma_eq) or self.sigma_eq <= 0:
            raise ValueError("No stationary std: spread is not mean-reverting in this window.")
        z = (pd.Series(spread, dtype=float) - self.mu) / self.sigma_eq
        z.name = "zscore"
        return z

    def max_hold(self, multiple: float = 3.0) -> int | None:
        """Time stop: ``ceil(multiple * half_life)`` bars, or None if undefined."""
        if not np.isfinite(self.half_life) or self.half_life <= 0:
            return None
        return int(np.ceil(float(multiple) * self.half_life))


def fit_ou(spread: pd.Series) -> OUFit:
    """OLS AR(1) fit of a spread series (levels, not differences)."""
    s = pd.Series(spread, dtype=float).dropna()
    if len(s) < 20:
        raise ValueError("Need at least 20 observations to fit an OU half-life.")
    lag = s.shift(1).iloc[1:]
    cur = s.iloc[1:]
    ols = sm.OLS(cur.to_numpy(), sm.add_constant(lag.to_numpy())).fit()
    a, b = float(ols.params[0]), float(ols.params[1])
    b_se = float(ols.bse[1])
    sigma_u = float(np.sqrt(ols.scale))
    if 0.0 < b < 1.0:
        kappa = -np.log(b)
        half_life = float(np.log(2.0) / kappa)
        mu = a / (1.0 - b)
        sigma_eq = sigma_u / np.sqrt(1.0 - b * b)
    elif b >= 1.0:
        kappa, half_life = 0.0, float("inf")
        mu, sigma_eq = float("nan"), float("nan")
    else:
        kappa, half_life = float("nan"), float("nan")
        mu = a / (1.0 - b)
        sigma_eq = sigma_u / np.sqrt(max(1.0 - b * b, 1e-12))
    return OUFit(
        b=b,
        b_se=b_se,
        mu=float(mu),
        sigma_eq=float(sigma_eq),
        kappa=float(kappa),
        half_life=float(half_life),
        nobs=int(ols.nobs),
    )
