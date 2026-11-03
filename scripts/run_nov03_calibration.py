"""Null calibration of the CUSUM break watch (Nov 3 slice): false alarms on healthy pairs.

Synthetic, seeded, offline. Run: ``python scripts/run_nov03_calibration.py``.
For every 252-bar formation / 63-bar trading block of an unbroken OU pair, take the
static-OLS block z (formation mean/std) and the formation AR(1) ``rho``, and record
the largest CUSUM value reached in the block. The false-alarm rate is the share of
healthy blocks whose maximum exceeds ``h``. These are *calibration* seeds (1000+);
the break experiments use different seeds. No break outcome is used to pick k or h.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from statarb.breakwatch import cusum_path
from statarb.cointegration import engle_granger
from statarb.formation import FORMATION, TRADING, _window_signal
from statarb.kalman import Q_ALPHA, Q_BETA
from statarb.ou import fit_ou
from statarb.rank import formation_residual
from statarb.synth import make_ou_pair

SEEDS = range(1000, 1060)
HALF_LIVES = (3.0, 5.0, 10.0, 20.0)
KS = (0.25, 0.5, 1.0)
H_GRID = (3.0, 5.0, 8.0)
N = 1260


def block_maxima(half_life: float, seeds=SEEDS, ks=KS) -> pd.DataFrame:
    """One row per healthy block and k: the max of the CUSUM path over the block."""
    rows = []
    for seed in seeds:
        p = make_ou_pair(N, half_life=half_life, seed=seed)
        yv, xv = p.y.to_numpy(), p.x.to_numpy()
        for st in range(0, N - FORMATION - TRADING + 1, TRADING):
            f = slice(st, st + FORMATION)
            t = slice(st + FORMATION, st + FORMATION + TRADING)
            z, _, _ = _window_signal(yv[f], xv[f], yv[t], xv[t], "ols", Q_BETA, Q_ALPHA)
            rho = fit_ou(formation_residual(yv[f], xv[f])).b
            for k in ks:
                rows.append({"half_life": half_life, "seed": seed, "k": k, "max": float(cusum_path(z, rho, k).max())})
    return pd.DataFrame(rows)


def false_alarm_table(seeds=SEEDS, half_lives=HALF_LIVES, ks=KS, hs=H_GRID) -> pd.DataFrame:
    out = []
    for hl in half_lives:
        m = block_maxima(hl, seeds, ks)
        for (k,), g in m.groupby(["k"]):
            row = {"half_life": hl, "k": k, "blocks": len(g)}
            for h in hs:
                row[f"FA@h={h:g}"] = float((g["max"] > h).mean())
            out.append(row)
    return pd.DataFrame(out)


GATE_PS = (0.01, 0.05)
GATE_HALF_LIVES = (3.0, 5.0, 10.0, 20.0, 40.0)


def false_gating_table(seeds=SEEDS, half_lives=GATE_HALF_LIVES, gate_ps=GATE_PS) -> pd.DataFrame:
    """Share of healthy 252-bar formation windows whose Engle-Granger p exceeds the gate.

    A gated block is a block a healthy pair does not trade, so this is the premium of the
    ``gate_p`` insurance: the slower the spread, the less power the test has in 252 bars.
    One window per seed per non-overlapping block (the same windows the walk-forward uses).
    """
    out = []
    for hl in half_lives:
        pv = []
        for seed in seeds:
            p = make_ou_pair(N, half_life=hl, seed=seed)
            for st in range(0, N - FORMATION - TRADING + 1, TRADING):
                f = slice(st, st + FORMATION)
                pv.append(engle_granger(p.y.iloc[f].reset_index(drop=True), p.x.iloc[f].reset_index(drop=True)).pvalue)
        pv = np.asarray(pv)
        row = {"half_life": hl, "windows": len(pv), "median_p": float(np.median(pv))}
        for g in gate_ps:
            row[f"gated@{g:g}"] = float((pv > g).mean())
        out.append(row)
    return pd.DataFrame(out)


def main() -> None:
    pd.set_option("display.width", 200)
    print("CUSUM false-alarm rate per healthy block")
    print(false_alarm_table().round(3).to_string(index=False))
    print("\nEngle-Granger gate: share of healthy formation windows gated")
    print(false_gating_table().round(3).to_string(index=False))


if __name__ == "__main__":
    main()
