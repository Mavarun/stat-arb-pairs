"""Rank and cap screened pair discoveries.

Oct 9 equal-weighted every discovery of a screening rule. This module scores
each screened pair from *formation* residuals only (OU half-life of the
Engle-Granger residual plus the EG p-value), then keeps the top-k. Scoring
never sees a trading bar.

Rank keys
---------
``eg_p``
    Ascending Engle-Granger p (both-orderings max). Stronger rejection first.
``half_life``
    Ascending finite OU half-life. Non-mean-reverting rows sink to the bottom.
``score``
    Composite: ``(-log eg_p) / half_life`` when the half-life sits in a
    tradable band ``[hl_lo, hl_hi]``; otherwise ``-inf``. Prefers tight,
    fast mean-reverters that the EG screen also likes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .ou import fit_ou
from .screening import RULES

RANK_KEYS = ("eg_p", "half_life", "score")

# Default tradable half-life band (bars). Faster than ~2 bars is noise on
# daily closes; slower than ~40 bars barely mean-reverts inside a 63-bar block.
HL_LO = 2.0
HL_HI = 40.0


def formation_residual(y: np.ndarray | pd.Series, x: np.ndarray | pd.Series) -> pd.Series:
    """OLS residual of ``y`` on ``x`` (constant + slope) over the given window."""
    yv = np.asarray(y, dtype=float)
    xv = np.asarray(x, dtype=float)
    if yv.shape != xv.shape or yv.ndim != 1:
        raise ValueError("y and x must be 1-d arrays of equal length")
    if len(yv) < 20:
        raise ValueError("Need at least 20 bars to fit a formation residual")
    ols = sm.OLS(yv, sm.add_constant(xv, has_constant="add")).fit()
    resid = yv - ols.fittedvalues
    idx = getattr(y, "index", None)
    return pd.Series(resid, index=idx, name="residual")


def annotate_half_lives(screen: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Add formation OU half-life columns for each screened pair.

    Ordering follows the smaller of ``eg_p_ab`` / ``eg_p_ba`` (same preference
    the re-screen book already uses when it picks y vs x). Failures (too-short
    residual, singular OLS) yield NaN half-life and ``is_mean_reverting=False``.
    """
    if screen.empty:
        out = screen.copy()
        out["half_life"] = pd.Series(dtype=float)
        out["sigma_eq"] = pd.Series(dtype=float)
        out["is_mean_reverting"] = pd.Series(dtype=bool)
        return out
    clean = prices.astype(float).dropna()
    rows = []
    for _, row in screen.iterrows():
        a, b = row["a"], row["b"]
        if float(row["eg_p_ab"]) <= float(row["eg_p_ba"]):
            y_col, x_col = a, b
        else:
            y_col, x_col = b, a
        try:
            resid = formation_residual(clean[y_col], clean[x_col])
            fit = fit_ou(resid)
            rows.append(
                {
                    "half_life": float(fit.half_life),
                    "sigma_eq": float(fit.sigma_eq),
                    "is_mean_reverting": bool(fit.is_mean_reverting),
                }
            )
        except (ValueError, np.linalg.LinAlgError):
            rows.append(
                {
                    "half_life": float("nan"),
                    "sigma_eq": float("nan"),
                    "is_mean_reverting": False,
                }
            )
    ann = pd.DataFrame(rows, index=screen.index)
    return pd.concat([screen.reset_index(drop=True), ann.reset_index(drop=True)], axis=1)


def score_discoveries(
    annotated: pd.DataFrame,
    *,
    hl_lo: float = HL_LO,
    hl_hi: float = HL_HI,
) -> pd.DataFrame:
    """Attach a composite ``score`` column (higher is better).

    Rows outside the half-life band, with non-finite half-life, or with a
    non-positive / non-finite ``eg_p`` get ``score = -inf`` so they sort last
    under a descending score rank.
    """
    if hl_lo <= 0 or hl_hi <= hl_lo:
        raise ValueError("need 0 < hl_lo < hl_hi")
    out = annotated.copy()
    if "half_life" not in out.columns:
        raise ValueError("annotated frame needs a half_life column; call annotate_half_lives first")
    if "eg_p" not in out.columns:
        raise ValueError("annotated frame needs an eg_p column")
    hl = out["half_life"].to_numpy(dtype=float)
    p = out["eg_p"].to_numpy(dtype=float)
    mr = (
        out["is_mean_reverting"].to_numpy(dtype=bool)
        if "is_mean_reverting" in out.columns
        else (np.isfinite(hl) & (hl > 0))
    )
    scores = np.full(len(out), -np.inf, dtype=float)
    ok = (
        mr
        & np.isfinite(hl)
        & (hl >= hl_lo)
        & (hl <= hl_hi)
        & np.isfinite(p)
        & (p > 0)
    )
    # Clamp p away from 0 so -log stays finite on numerical underflow.
    scores[ok] = -np.log(np.maximum(p[ok], 1e-300)) / hl[ok]
    out["score"] = scores
    return out


def select_top(
    screen: pd.DataFrame,
    *,
    rule: str,
    rank_by: str = "score",
    max_pairs: int | None = None,
    prices: pd.DataFrame | None = None,
    hl_lo: float = HL_LO,
    hl_hi: float = HL_HI,
) -> pd.DataFrame:
    """Filter by ``rule``, rank, optionally cap at ``max_pairs``.

    When ``rank_by`` needs half-life or score and those columns are missing,
    ``prices`` must be supplied so ``annotate_half_lives`` can run. Cap of
    ``None`` or ``<= 0`` means keep every discovery after ranking.
    """
    if rule not in RULES:
        raise ValueError(f"rule must be one of {RULES}, got {rule!r}")
    if rank_by not in RANK_KEYS:
        raise ValueError(f"rank_by must be one of {RANK_KEYS}, got {rank_by!r}")
    selected = screen.loc[screen[rule]].copy()
    if selected.empty:
        return selected.reset_index(drop=True)

    if rank_by in ("half_life", "score") and "half_life" not in selected.columns:
        if prices is None:
            raise ValueError("prices required to annotate half-lives for this rank_by")
        selected = annotate_half_lives(selected, prices)
    if rank_by == "score" and "score" not in selected.columns:
        selected = score_discoveries(selected, hl_lo=hl_lo, hl_hi=hl_hi)

    if rank_by == "eg_p":
        ordered = selected.sort_values("eg_p", kind="mergesort", ascending=True)
    elif rank_by == "half_life":
        # Non-finite half-lives sort last under na_position='last'.
        ordered = selected.sort_values(
            "half_life", kind="mergesort", ascending=True, na_position="last"
        )
    else:  # score
        ordered = selected.sort_values("score", kind="mergesort", ascending=False)

    ordered = ordered.reset_index(drop=True)
    if max_pairs is not None and int(max_pairs) > 0:
        ordered = ordered.iloc[: int(max_pairs)].copy()
    return ordered.reset_index(drop=True)
