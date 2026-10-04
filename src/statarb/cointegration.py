"""Cointegration helpers.

Trading in this slice uses Engle-Granger only: OLS of y on x, then a residual
ADF/coint test via ``statsmodels.tsa.stattools.coint``.

Johansen (``coint_johansen``) is a diagnostic, and a confirmation filter after
BH in ``statarb.screening``. We return the library's trace statistics and
critical values as-is. We do not invent p-values or feed Johansen vectors into
the book.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen


def _align(y: pd.Series, x: pd.Series) -> tuple[pd.Series, pd.Series]:
    y = pd.Series(y, dtype=float).rename("y")
    x = pd.Series(x, dtype=float).rename("x")
    both = pd.concat([y, x], axis=1).dropna()
    if len(both) < 10:
        raise ValueError("Need at least 10 aligned observations for cointegration.")
    return both["y"], both["x"]


@dataclass(frozen=True)
class EngleGrangerResult:
    """OLS hedge of y on x plus Engle-Granger cointegration test.

    ``hedge_ratio`` is the slope in ``y = intercept + hedge_ratio * x + e``.
    ``pvalue`` is the statsmodels Engle-Granger p-value for the null of
    *no* cointegration (MacKinnon). Small p-values reject that null.
    """

    hedge_ratio: float
    intercept: float
    statistic: float
    pvalue: float
    crit_values: dict[str, float]
    nobs: int
    residual_std: float

    def as_dict(self) -> dict[str, float]:
        out = {
            "hedge_ratio": self.hedge_ratio,
            "intercept": self.intercept,
            "eg_stat": self.statistic,
            "eg_pvalue": self.pvalue,
            "nobs": float(self.nobs),
            "residual_std": self.residual_std,
        }
        for k, v in self.crit_values.items():
            out[f"crit_{k}"] = v
        return out


def engle_granger(y: pd.Series, x: pd.Series, trend: str = "c") -> EngleGrangerResult:
    """Fit Engle-Granger: OLS hedge ratio of y on x, then coint test.

    Parameters
    ----------
    y, x:
        Price levels (adjusted closes). Not returns.
    trend:
        Passed to ``statsmodels.tsa.stattools.coint`` ('c' = constant).
    """
    y, x = _align(y, x)
    X = sm.add_constant(x, has_constant="add")
    ols = sm.OLS(y, X).fit()
    intercept = float(ols.params.iloc[0])
    hedge_ratio = float(ols.params.iloc[1])
    resid = y - intercept - hedge_ratio * x

    stat, pvalue, crit = coint(y, x, trend=trend, autolag="aic")
    # crit is [1%, 5%, 10%] critical values for the test statistic (more negative = stronger).
    crit_values = {"1%": float(crit[0]), "5%": float(crit[1]), "10%": float(crit[2])}
    return EngleGrangerResult(
        hedge_ratio=hedge_ratio,
        intercept=intercept,
        statistic=float(stat),
        pvalue=float(pvalue),
        crit_values=crit_values,
        nobs=int(ols.nobs),
        residual_std=float(resid.std(ddof=1)),
    )


@dataclass(frozen=True)
class JohansenResult:
    """Raw Johansen trace test output for a bivariate system.

    ``trace_stat[i]`` tests H0: cointegrating rank <= i.
    Critical values are 90 / 95 / 99 percent from statsmodels, not invented.
    """

    trace_stat: np.ndarray
    max_eig_stat: np.ndarray
    crit_trace: np.ndarray  # shape (n, 3) = 90%, 95%, 99%
    crit_max_eig: np.ndarray
    eig: np.ndarray
    evec: np.ndarray
    det_order: int
    k_ar_diff: int

    def trace_rejects_r0(self, level: str = "95%") -> bool:
        """Whether the r=0 trace statistic exceeds the given critical value.

        Diagnostic only. A True here is not a trading license.
        """
        col = {"90%": 0, "95%": 1, "99%": 2}[level]
        return bool(self.trace_stat[0] > self.crit_trace[0, col])

    def as_frame(self) -> pd.DataFrame:
        ranks = [f"r<={i}" for i in range(len(self.trace_stat))]
        return pd.DataFrame(
            {
                "trace_stat": self.trace_stat,
                "crit_90": self.crit_trace[:, 0],
                "crit_95": self.crit_trace[:, 1],
                "crit_99": self.crit_trace[:, 2],
                "max_eig_stat": self.max_eig_stat,
            },
            index=ranks,
        )


def johansen(
    data: pd.DataFrame,
    det_order: int = 0,
    k_ar_diff: int = 1,
) -> JohansenResult:
    """Run ``coint_johansen`` on a 2-column price frame.

    ``det_order=0`` is no deterministic trend (constant in the VECM).
    ``k_ar_diff`` is the lag of first differences. Both are defaults, not tuned.
    """
    if data.shape[1] != 2:
        raise ValueError("Johansen helper in this slice expects exactly two columns.")
    clean = data.astype(float).dropna()
    if len(clean) < 30:
        raise ValueError("Need at least 30 observations for Johansen.")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=np.exceptions.ComplexWarning)
        # NumPy 1.x used numpy.ComplexWarning; 2.x moved it.
        try:
            warnings.filterwarnings("ignore", category=np.ComplexWarning)  # type: ignore[attr-defined]
        except AttributeError:
            pass
        res = coint_johansen(clean.values, det_order=det_order, k_ar_diff=k_ar_diff)
    # statsmodels can hand back tiny imaginary parts; we keep the real numbers it computed.
    def _real(a):
        return np.real(np.asarray(a)).astype(float, copy=False)
    return JohansenResult(
        trace_stat=_real(res.lr1),
        max_eig_stat=_real(res.lr2),
        crit_trace=_real(res.cvt),
        crit_max_eig=_real(res.cvm),
        eig=_real(res.eig),
        evec=_real(res.evec),
        det_order=det_order,
        k_ar_diff=k_ar_diff,
    )
