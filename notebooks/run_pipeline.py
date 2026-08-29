"""KO/PEP Engle-Granger walk-forward. Called by pairs_trading.ipynb."""

from __future__ import annotations

import importlib.util
import warnings
from pathlib import Path

import pandas as pd

from statarb.backtest import expanding_walk_forward, walk_forward
from statarb.cointegration import engle_granger, johansen
from statarb.data import download_pair, load_csv
from statarb.params import SLIPPAGE_BPS, SPREAD_BPS, TRAIN_FRAC, Z_ENTER, Z_EXIT

warnings.filterwarnings("ignore", category=FutureWarning)


def repo_root() -> Path:
    here = Path.cwd()
    if (here / "data").exists():
        return here
    if (here.parent / "data").exists():
        return here.parent
    return here


def sample_path() -> Path:
    root = repo_root()
    path = root / "data" / "sample_ko_pep.csv"
    if path.exists():
        return path
    gen = root / "data" / "make_sample.py"
    spec = importlib.util.spec_from_file_location("make_sample", gen)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.write_sample(path)


def load_prices() -> tuple[pd.DataFrame, str, Exception | None]:
    err = None
    try:
        prices = download_pair("KO", "PEP", start="2016-01-01")
        return prices, "yfinance KO/PEP Adj Close (auto_adjust=True)", None
    except Exception as e:
        err = e
        prices = load_csv(sample_path(), y_col="KO", x_col="PEP")
        return prices, "bundled synthetic sample (NOT live market data)", err


def main() -> None:
    print(
        "defaults",
        dict(
            Z_ENTER=Z_ENTER,
            Z_EXIT=Z_EXIT,
            SPREAD_BPS=SPREAD_BPS,
            SLIPPAGE_BPS=SLIPPAGE_BPS,
            TRAIN_FRAC=TRAIN_FRAC,
        ),
    )
    prices, source, err = load_prices()
    y, x = prices["KO"], prices["PEP"]
    print("source:", source)
    if err is not None:
        print("yfinance error:", type(err).__name__, err)
    print(prices.tail())
    print("nobs", len(prices), "from", prices.index.min().date(), "to", prices.index.max().date())

    eg = engle_granger(y, x)
    print("\nEngle-Granger")
    print(pd.Series(eg.as_dict()).round(6).to_string())
    print("null of no-coint rejected at 5%?", eg.pvalue < 0.05)

    joh = johansen(prices[["KO", "PEP"]], det_order=0, k_ar_diff=1)
    print("\nJohansen (diagnostic, not traded)")
    print(joh.as_frame().round(3).to_string())
    print("trace rejects r=0 at 95%?", joh.trace_rejects_r0("95%"))

    wf = walk_forward(
        y,
        x,
        train_frac=TRAIN_FRAC,
        z_enter=Z_ENTER,
        z_exit=Z_EXIT,
        z_mode="train_stats",
        spread_bps=SPREAD_BPS,
        slippage_bps=SLIPPAGE_BPS,
    )
    print("\ntrain_end", wf.train_end.date())
    print("train EG pvalue", round(wf.engle_granger.pvalue, 6), "beta", round(wf.engle_granger.hedge_ratio, 4))
    table = wf.metrics_table()[["total_return", "sharpe", "max_drawdown", "trade_count", "turnover", "n_obs"]]
    print(table.round(4).to_string())
    print(
        "OOS costed Sharpe",
        round(wf.test_costed.sharpe, 4),
        "| OOS uncosted Sharpe",
        round(wf.test_uncosted.sharpe, 4),
    )

    folds = expanding_walk_forward(
        y,
        x,
        n_splits=4,
        z_enter=Z_ENTER,
        z_exit=Z_EXIT,
        z_mode="train_stats",
        spread_bps=SPREAD_BPS,
        slippage_bps=SLIPPAGE_BPS,
        min_train=252,
    )
    rows = []
    for i, f in enumerate(folds, 1):
        rows.append(
            {
                "fold": i,
                "train_end": f.train_end.date(),
                "test_sharpe_uncosted": f.test_uncosted.sharpe,
                "test_sharpe_costed": f.test_costed.sharpe,
                "test_total_costed": f.test_costed.total_return,
                "test_trades": f.test_costed.trade_count,
                "test_n": f.test_costed.n_obs,
                "beta": f.engle_granger.hedge_ratio,
                "eg_p": f.engle_granger.pvalue,
            }
        )
    fold_table = pd.DataFrame(rows).set_index("fold")
    print("\nExpanding walk-forward")
    print(fold_table.round(4).to_string())
    print("mean OOS costed Sharpe across folds", round(fold_table["test_sharpe_costed"].mean(), 4))


if __name__ == "__main__":
    main()
