"""Price loaders.

Yahoo Finance via yfinance is a convenience tape, not a research-grade
adjustments/corporate-action feed. ``load_csv`` is the offline path.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _extract_adjusted(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Prefer Adj Close; yfinance auto_adjust=True puts adjusted prices in Close."""
    if isinstance(raw.columns, pd.MultiIndex):
        level0 = set(raw.columns.get_level_values(0))
        if "Adj Close" in level0:
            px = raw["Adj Close"]
        elif "Close" in level0:
            px = raw["Close"]
        else:
            raise ValueError(f"No Close/Adj Close in yfinance columns: {level0}")
        missing = [t for t in tickers if t not in px.columns]
        if missing:
            raise ValueError(f"Missing tickers in download: {missing}")
        return px.loc[:, tickers].astype(float)
    # Single-ticker download
    col = "Adj Close" if "Adj Close" in raw.columns else "Close"
    if col not in raw.columns:
        raise ValueError(f"No Close/Adj Close in columns: {list(raw.columns)}")
    out = raw[[col]].astype(float)
    out.columns = [tickers[0]]
    return out


def download_pair(
    ticker_y: str = "KO",
    ticker_x: str = "PEP",
    start: str = "2015-01-01",
    end: str | None = None,
) -> pd.DataFrame:
    """Download two Yahoo tickers and return a two-column Adj Close frame.

    Raises on empty/failed downloads so callers can fall back to ``load_csv``.
    """
    import yfinance as yf

    tickers = [ticker_y, ticker_x]
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=False,
        group_by="column",
    )
    if raw is None or raw.empty:
        raise RuntimeError(f"yfinance returned no rows for {tickers}")
    px = _extract_adjusted(raw, tickers).dropna()
    if px.empty or px.shape[1] != 2:
        raise RuntimeError(f"yfinance produced an unusable frame: {px.shape}")
    px.columns = [ticker_y, ticker_x]
    return px.sort_index()


def load_csv(
    path: str | Path,
    y_col: str = "KO",
    x_col: str = "PEP",
    date_col: str = "date",
) -> pd.DataFrame:
    """Load a two-price CSV. Lines starting with ``#`` are comments."""
    path = Path(path)
    df = pd.read_csv(path, comment="#", parse_dates=[date_col])
    df = df.set_index(date_col).sort_index()
    missing = [c for c in (y_col, x_col) if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns {missing}; have {list(df.columns)}")
    out = df[[y_col, x_col]].astype(float).dropna()
    out.index.name = "date"
    return out
