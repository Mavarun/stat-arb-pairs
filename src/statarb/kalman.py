"""Kalman-filter dynamic hedge ratio (random-walk alpha and beta; a local-linear-trend variant below).

State-space model::

    theta_t = [alpha_t, beta_t] = theta_{t-1} + w_t,   w_t ~ N(0, diag(q_alpha, q_beta))
    y_t     = alpha_t + beta_t * x_t + v_t,           v_t ~ N(0, r)

Initialisation is causal: OLS on the first ``warmup`` bars gives
``theta_0``, its covariance ``P_0`` and the observation variance ``r``. No
output is produced for those bars (NaN), so nothing downstream can trade on
an estimate that saw them "from the future" of the filter.

At each later bar t we report

* ``alpha_prior``, ``beta_prior``: theta_{t|t-1}, built from data up to t-1;
* ``innovation`` e_t = y_t - alpha_prior - beta_prior * x_t and its variance S_t;
* ``zscore`` = e_t / sqrt(S_t)  (the tradeable spread signal at close t);
* ``alpha``, ``beta``: theta_{t|t}, the filtered state after seeing bar t.

The book decided at close t holds ``beta[t]`` units of x per unit of y over
(t, t+1]. Re-hedging that leg as beta drifts costs money; the walk-forward
backtest charges it.

``q_beta`` / ``q_alpha`` are not estimated (no MLE). They are stated defaults;
too large and the filter chases noise (spread is absorbed into the hedge),
too small and it collapses to static OLS.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

Q_BETA = 1e-6
Q_ALPHA = 1e-4
WARMUP = 60


def kalman_hedge(
    y: pd.Series,
    x: pd.Series,
    *,
    q_beta: float = Q_BETA,
    q_alpha: float = Q_ALPHA,
    r: float | None = None,
    warmup: int = WARMUP,
) -> pd.DataFrame:
    """Run the filter; returns a frame indexed like the aligned inputs."""
    both = pd.concat(
        [pd.Series(y, dtype=float).rename("y"), pd.Series(x, dtype=float).rename("x")], axis=1
    ).dropna()
    n = len(both)
    if warmup < 10 or n <= warmup:
        raise ValueError("Need warmup >= 10 and more observations than warmup.")
    yv, xv = both["y"].to_numpy(), both["x"].to_numpy()

    ols = sm.OLS(yv[:warmup], sm.add_constant(xv[:warmup], has_constant="add")).fit()
    theta = np.asarray(ols.params, dtype=float).copy()
    P = np.asarray(ols.cov_params(), dtype=float).copy()
    r = float(ols.scale) if r is None else float(r)
    Q = np.diag([float(q_alpha), float(q_beta)])

    cols = ["alpha_prior", "beta_prior", "innovation", "innov_var", "zscore", "alpha", "beta"]
    out = np.full((n, len(cols)), np.nan)
    for t in range(warmup, n):
        P_prior = P + Q
        H = np.array([1.0, xv[t]])
        e = yv[t] - H @ theta
        S = float(H @ P_prior @ H + r)
        K = P_prior @ H / S
        out[t, 0:2] = theta
        out[t, 2] = e
        out[t, 3] = S
        out[t, 4] = e / np.sqrt(S)
        theta = theta + K * e
        P = P_prior - np.outer(K, H) @ P_prior
        P = 0.5 * (P + P.T)
        out[t, 5:7] = theta
    return pd.DataFrame(out, index=both.index, columns=cols)


def static_hedge_path(y: pd.Series, x: pd.Series, *, warmup: int = WARMUP) -> pd.Series:
    """Baseline for comparison: one OLS beta from the first ``warmup`` bars, held flat.

    This is what a formation-window Engle-Granger hedge does out of sample.
    """
    both = pd.concat(
        [pd.Series(y, dtype=float).rename("y"), pd.Series(x, dtype=float).rename("x")], axis=1
    ).dropna()
    ols = sm.OLS(both["y"].to_numpy()[:warmup], sm.add_constant(both["x"].to_numpy()[:warmup])).fit()
    out = pd.Series(float(ols.params[1]), index=both.index, name="beta")
    out.iloc[:warmup] = np.nan
    return out


Q_DRIFT = 1e-12
P_DRIFT0 = (1.5 * 0.10 / 252) ** 2


def kalman_trend_hedge(
    y: pd.Series,
    x: pd.Series,
    *,
    q_beta: float = 1e-8,
    q_alpha: float = Q_ALPHA,
    q_drift: float = Q_DRIFT,
    p_drift0: float = P_DRIFT0,
    r: float | None = None,
    warmup: int = WARMUP,
) -> pd.DataFrame:
    """Local-linear-trend hedge: the ratio carries its own per-bar drift.

    State ``[alpha_t, beta_t, g_t]`` with ``beta_t = beta_{t-1} + g_{t-1} + w``,
    ``g_t = g_{t-1} + u`` (``u ~ N(0, q_drift)``), so a ratio that trends linearly is a
    *constant* in the drift state, which the random-walk filter can only chase. Same
    causal OLS start as :func:`kalman_hedge`; ``g_0 = 0`` with prior variance ``p_drift0``
    (default: a 10%/yr move of a 1.5 ratio, one standard deviation). Returns the same
    columns as :func:`kalman_hedge` plus ``drift`` (filtered g_t, per bar).
    """
    both = pd.concat(
        [pd.Series(y, dtype=float).rename("y"), pd.Series(x, dtype=float).rename("x")], axis=1
    ).dropna()
    n = len(both)
    if warmup < 10 or n <= warmup:
        raise ValueError("Need warmup >= 10 and more observations than warmup.")
    if min(q_beta, q_alpha, q_drift, p_drift0) < 0:
        raise ValueError("variances must be >= 0")
    yv, xv = both["y"].to_numpy(), both["x"].to_numpy()
    ols = sm.OLS(yv[:warmup], sm.add_constant(xv[:warmup], has_constant="add")).fit()
    theta = np.array([float(ols.params[0]), float(ols.params[1]), 0.0])
    P = np.zeros((3, 3))
    P[:2, :2] = np.asarray(ols.cov_params(), dtype=float)
    P[2, 2] = float(p_drift0)
    r = float(ols.scale) if r is None else float(r)
    F = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]])
    Q = np.diag([float(q_alpha), float(q_beta), float(q_drift)])

    cols = ["alpha_prior", "beta_prior", "innovation", "innov_var", "zscore", "alpha", "beta", "drift"]
    out = np.full((n, len(cols)), np.nan)
    for t in range(warmup, n):
        theta_p = F @ theta
        P_prior = F @ P @ F.T + Q
        H = np.array([1.0, xv[t], 0.0])
        e = yv[t] - H @ theta_p
        S = float(H @ P_prior @ H + r)
        K = P_prior @ H / S
        out[t, 0:2] = theta_p[:2]
        out[t, 2] = e
        out[t, 3] = S
        out[t, 4] = e / np.sqrt(S)
        theta = theta_p + K * e
        P = P_prior - np.outer(K, H) @ P_prior
        P = 0.5 * (P + P.T)
        out[t, 5:7] = theta[:2]
        out[t, 7] = theta[2]
    return pd.DataFrame(out, index=both.index, columns=cols)
