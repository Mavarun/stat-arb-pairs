"""Multiple-testing-aware pair screening: Engle-Granger p-values, BH FDR, Johansen confirm.

Screening ``N`` assets means testing ``N * (N - 1) / 2`` pairs. Under the null
(no pair is cointegrated) a naive ``p < 0.05`` rule still "finds" about 5% of
them. On 20 independent random walks that is ~9-10 false pairs out of 190.

Procedure per unordered pair (a, b):

1. Engle-Granger both ways, ``p_ab`` (a on b) and ``p_ba`` (b on a).
   The pair p-value is ``max(p_ab, p_ba)``: we require *both* orderings to
   reject. ``P(max <= t) <= P(p_ab <= t) <= t`` under the null, so it is still a
   valid (conservative) p-value; taking the *min* would be another hidden
   multiple test.
2. Benjamini-Hochberg over all pair p-values at level ``alpha`` (FDR control).
   BH is proven for independent or PRDS p-values. Pair tests that share a leg
   are dependent and PRDS is not guaranteed here, so treat the FDR level as
   approximate; the random-walk tests check it empirically.
3. Johansen trace test (r = 0) at 95% as a *confirmation* filter. statsmodels
   gives critical values, not p-values, so Johansen cannot enter BH directly;
   it only removes BH discoveries, never adds any.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

from .cointegration import johansen


def benjamini_hochberg(pvalues, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg step-up procedure.

    Returns ``(reject, qvalues)`` in the input order. ``qvalues`` are the BH
    adjusted p-values (monotone, capped at 1); ``reject = qvalues <= alpha``.
    NaN p-values are never rejected and do not count towards ``m``.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    p = np.asarray(pvalues, dtype=float)
    q = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    m = int(ok.sum())
    if m == 0:
        return np.zeros(p.shape, dtype=bool), q
    if np.any((p[ok] < 0) | (p[ok] > 1)):
        raise ValueError("p-values must lie in [0, 1]")
    pv = p[ok]
    order = np.argsort(pv, kind="mergesort")
    ranked = pv[order] * m / np.arange(1, m + 1)
    # Step-up: q_(i) = min_{j >= i} p_(j) * m / j
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q_ok = np.empty(m)
    q_ok[order] = np.minimum(ranked, 1.0)
    q[ok] = q_ok
    reject = np.zeros(p.shape, dtype=bool)
    reject[ok] = q_ok <= alpha
    return reject, q


def _eg_pvalue(y: np.ndarray, x: np.ndarray, maxlag: int | None) -> float:
    autolag = "aic" if maxlag is None else None
    _, p, _ = coint(y, x, trend="c", maxlag=maxlag, autolag=autolag)
    return float(p)


def screen_pairs(
    prices: pd.DataFrame,
    *,
    alpha: float = 0.05,
    maxlag: int | None = 1,
    johansen_k_ar_diff: int = 1,
) -> pd.DataFrame:
    """Test every unordered pair of columns in ``prices`` (levels).

    ``maxlag=1`` with no autolag keeps the ADF regression fixed and the screen
    fast; pass ``None`` for statsmodels' AIC lag search.

    Columns of the result, one row per pair, sorted by ``eg_p``:

    ``a, b, eg_p_ab, eg_p_ba, eg_p`` (max of the two),
    ``naive_single`` (``eg_p_ab < alpha``: one ordering, the common shortcut),
    ``naive`` (``eg_p < alpha``: both orderings, still uncorrected),
    ``bh_q``, ``bh`` (BH discovery), ``johansen_trace``, ``johansen_crit95``,
    ``johansen_rejects`` / ``johansen_only`` (Johansen alone at 95%, uncorrected),
    ``bh_johansen`` (BH discovery confirmed by Johansen).
    """
    clean = prices.astype(float).dropna()
    if clean.shape[1] < 2:
        raise ValueError("Need at least two price columns to screen pairs.")
    if len(clean) < 30:
        raise ValueError("Need at least 30 aligned observations to screen pairs.")
    rows = []
    for a, b in combinations(clean.columns, 2):
        ya, yb = clean[a].to_numpy(), clean[b].to_numpy()
        p_ab = _eg_pvalue(ya, yb, maxlag)
        p_ba = _eg_pvalue(yb, ya, maxlag)
        joh = johansen(clean[[a, b]], det_order=0, k_ar_diff=johansen_k_ar_diff)
        rows.append(
            {
                "a": a,
                "b": b,
                "eg_p_ab": p_ab,
                "eg_p_ba": p_ba,
                "eg_p": max(p_ab, p_ba),
                "johansen_trace": float(joh.trace_stat[0]),
                "johansen_crit95": float(joh.crit_trace[0, 1]),
                "johansen_rejects": joh.trace_rejects_r0("95%"),
            }
        )
    out = pd.DataFrame(rows)
    out["naive_single"] = out["eg_p_ab"] < alpha
    out["naive"] = out["eg_p"] < alpha
    out["johansen_only"] = out["johansen_rejects"]
    out["bh"], out["bh_q"] = benjamini_hochberg(out["eg_p"].to_numpy(), alpha)
    out["bh_johansen"] = out["bh"] & out["johansen_rejects"]
    cols = [
        "a", "b", "eg_p_ab", "eg_p_ba", "eg_p", "naive_single", "naive", "bh_q", "bh",
        "johansen_trace", "johansen_crit95", "johansen_rejects", "johansen_only", "bh_johansen",
    ]
    return out[cols].sort_values("eg_p", kind="mergesort").reset_index(drop=True)


RULES = ("naive_single", "naive", "johansen_only", "bh", "bh_johansen")


def screen_summary(screen: pd.DataFrame, truth: set[frozenset[str]] | None = None) -> pd.DataFrame:
    """Discoveries per rule; with ``truth``, also true/false positives and FDP."""
    truth = truth or set()
    is_true = np.array([frozenset((a, b)) in truth for a, b in zip(screen["a"], screen["b"])])
    rows = {}
    for rule in RULES:
        sel = screen[rule].to_numpy(dtype=bool)
        tp = int((sel & is_true).sum())
        fp = int((sel & ~is_true).sum())
        n = tp + fp
        rows[rule] = {
            "discoveries": n,
            "true_pos": tp,
            "false_pos": fp,
            "fdp": fp / n if n else 0.0,
            "power": tp / len(truth) if truth else float("nan"),
        }
    out = pd.DataFrame(rows).T
    out.attrs["n_tests"] = len(screen)
    return out
