"""Write data/sample_ko_pep.csv — SYNTHETIC, not market data.

Run from the repo root:
    python data/make_sample.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20260829
N = 756
START = "2022-01-03"


def build_frame(n: int = N, seed: int = SEED, start: str = START) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    pep = 155.0 + np.cumsum(rng.normal(0.02, 1.15, n))
    pep = np.clip(pep, 80.0, None)
    e = np.zeros(n)
    shocks = rng.normal(0.0, 0.55, n)
    for t in range(1, n):
        e[t] = 0.88 * e[t - 1] + shocks[t]
    ko = np.clip(8.0 + 0.40 * pep + e, 20.0, None)
    return pd.DataFrame(
        {"date": idx.strftime("%Y-%m-%d"), "KO": np.round(ko, 4), "PEP": np.round(pep, 4)}
    )


def write_sample(path: str | Path | None = None) -> Path:
    path = Path(path) if path else Path(__file__).resolve().parent / "sample_ko_pep.csv"
    header = (
        "# SYNTHETIC SAMPLE — not live KO/PEP market data.\n"
        "# Generated for offline notebook/tests. Do not treat as a research tape.\n"
        "# Columns: date, KO, PEP (fake adjusted-close levels with a planted ~0.40 hedge).\n"
    )
    path.write_text(header + build_frame().to_csv(index=False), encoding="utf-8")
    return path


if __name__ == "__main__":
    out = write_sample()
    print(f"wrote {out}")
