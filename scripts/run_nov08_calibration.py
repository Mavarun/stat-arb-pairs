"""Nov 8 slice: premium and power of a half-life-calibrated Engle-Granger gate.

Synthetic, seeded, offline. Run: ``python scripts/run_nov08_calibration.py`` (~5 min).

Premium: on unbroken OU pairs (calibration seeds 1000-1059, the Nov 3 healthy set), the
share of 252-bar formation windows (one per non-overlapping 63-bar block, 16 per path)
that each gate would keep flat. Power: on dead pairs (the spread is a random walk from
bar 1, seeds 1100-1159) the share of formation windows each gate would catch. Gates:
fixed p < 0.01 / 0.05 (Nov 3) and ``gate_quantile`` 0.95 / 0.90 with the reference
half-life either the planted one ("oracle") or estimated on the first formation window
("estimated", what the walk-forward does). The level is simulated on its own seeds
(``gatecal.SIM_SEED``), never on these.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from statarb.cointegration import engle_granger
from statarb.formation import FORMATION, TRADING
from statarb.gatecal import calibrated_gate_level
from statarb.ou import fit_ou
from statarb.rank import formation_residual
from statarb.synth import make_broken_pair, make_ou_pair

HEALTHY_SEEDS = range(1000, 1060)
DEAD_SEEDS = range(1100, 1160)
HALF_LIVES = (3.0, 5.0, 10.0, 20.0, 40.0)
QUANTILES = (0.95, 0.90)
FIXED = (0.01, 0.05)
N = 1260


def _window_pvalues(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.asarray([
        engle_granger(pd.Series(y[st:st + FORMATION]), pd.Series(x[st:st + FORMATION])).pvalue
        for st in range(0, N - FORMATION - TRADING + 1, TRADING)
    ])


def _first_hl(y: np.ndarray, x: np.ndarray) -> float:
    try:
        return fit_ou(formation_residual(y[:FORMATION], x[:FORMATION])).half_life
    except ValueError:
        return float("nan")


def gate_rates(
    half_life: float, seeds, dead: bool, quantiles=QUANTILES, fixed=FIXED, estimated: bool = True
) -> dict[str, float]:
    """Share of formation windows gated by each rule (pooled over seeds and blocks).

    For dead pairs the "estimated" reference is the healthy first-window estimate of the
    same seed's *unbroken* path (the pair was picked while healthy, then died).
    """
    hits: dict[str, list[float]] = {}
    ref_est = []
    for s in seeds:
        healthy = make_ou_pair(N, half_life=half_life, seed=s)
        hl_hat = _first_hl(healthy.y.to_numpy(), healthy.x.to_numpy())
        ref_est.append(hl_hat)
        p = make_broken_pair(N, break_at=1, kind="random_walk", half_life=half_life, seed=s) if dead else healthy
        pv = _window_pvalues(p.y.to_numpy(), p.x.to_numpy())
        for g in fixed:
            hits.setdefault(f"fixed {g:g}", []).extend(pv > g)
        for q in quantiles:
            hits.setdefault(f"q{q:g} oracle", []).extend(pv > calibrated_gate_level(half_life, FORMATION, quantile=q))
            if estimated:
                hits.setdefault(f"q{q:g} estimated", []).extend(
                    pv > calibrated_gate_level(hl_hat, FORMATION, quantile=q)
                )
    out = {k: float(np.mean(v)) for k, v in hits.items()}
    out["median_ref_hl_hat"] = float(np.nanmedian(ref_est))
    return out


def table(half_lives=HALF_LIVES, healthy_seeds=HEALTHY_SEEDS, dead_seeds=DEAD_SEEDS) -> pd.DataFrame:
    rows = []
    for hl in half_lives:
        for kind, seeds, dead in (("premium (healthy gated)", healthy_seeds, False), ("power (dead gated)", dead_seeds, True)):
            rows.append({"half_life": hl, "measure": kind, **gate_rates(hl, seeds, dead)})
    return pd.DataFrame(rows)


def levels(half_lives=HALF_LIVES, quantiles=QUANTILES) -> pd.DataFrame:
    return pd.DataFrame([
        {"half_life": hl, **{f"level q{q:g}": calibrated_gate_level(hl, FORMATION, quantile=q) for q in quantiles}}
        for hl in half_lives
    ])


def main() -> None:
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    print("Calibrated gate level (252 bars, 400 simulated healthy pairs)")
    print(levels().round(3).to_string(index=False))
    print("\nShare of formation windows gated")
    print(table().round(3).to_string(index=False))


if __name__ == "__main__":
    main()
