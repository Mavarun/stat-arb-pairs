# stat-arb-pairs

Research slice for a **KO / PEP Engle-Granger pairs trade**. It estimates a hedge ratio by OLS, treats the residual as a spread, trades a z-score mean-reversion rule, subtracts explicit spread + slippage costs, and reports a **date-based walk-forward** so in-sample Sharpe is not mistaken for an out-of-sample result.

This is a method check, not a live book. **Do not read any number here as live PnL or alpha.**

## Today's hypothesis

1. KO and PEP daily adjusted closes are cointegrated, so the OLS residual (spread) is mean-reverting.
2. A z-score of that spread, using a hedge ratio from Engle-Granger, can be traded after explicit spread + slippage costs.
3. A date-based walk-forward split will show whether out-of-sample Sharpe is real or only in-sample fit.

The notebook runs that pipeline on Yahoo KO/PEP when the network cooperates, otherwise on `data/sample_ko_pep.csv`. The CSV is a **synthetic sample, not live market data**. If the file is missing, `python data/make_sample.py` (or the notebook itself) writes it.

## What is implemented

| Piece | What it does | What it does not do |
| --- | --- | --- |
| `engle_granger` | OLS of *y* on *x* (constant + slope) and `statsmodels.tsa.stattools.coint` | Does not pick the pair, does not use returns as the coint input |
| `johansen` | `coint_johansen` trace / max-eig stats and critical values | **Not used for trading.** No fake "Johansen says yes" flag |
| `signals` | Spread \\(y - a - \\beta x\\), rolling or frozen z-score, {-1,0,+1} book | No Kalman hedge, no half-life sizer, no volatility targeting |
| `backtest` | Close-to-close long-short residual, costs on \\|\u0394position\\| \u00d7 2 legs, Sharpe / DD / trades / turnover | No borrow, no dividends as cash (prices are treated as already adjusted), no impact model |
| `walk_forward` | Hedge + z stats on a train window; positions scored only after `train_end` | Not a full expanding grid search. Defaults are not optimized |

Trading always uses **Engle-Granger**. Johansen is printed as a diagnostic so we do not pretend the VECM was estimated and then ignore it, and so we do not invent VECM results.

## Default parameters (not optimized)

These are textbook starting points, frozen before looking at KO/PEP:

| Name | Default | Role |
| --- | --- | --- |
| `Z_ENTER` | 2.0 | Open when \\|z\\| exceeds this |
| `Z_EXIT` | 0.5 | Flatten when \\|z\\| mean-reverts through this |
| `Z_WINDOW` | 60 | Rolling z window (only if `z_mode='rolling'`) |
| `SPREAD_BPS` | 5 | Bid-ask, per leg, on \\|\u0394position\\| |
| `SLIPPAGE_BPS` | 5 | Extra slippage, per leg, on \\|\u0394position\\| |
| `N_LEGS` | 2 | Long one name, short the other |
| `TRAIN_FRAC` | 0.70 | Walk-forward train fraction when no `train_end` is passed |
| `MIN_TRAIN_OBS` | 252 | Intended minimum train length on real daily data |
| `ANNUALIZATION` | 252 | Sharpe scale |

Cost of a unit entry or exit is `N_LEGS * (SPREAD_BPS + SLIPPAGE_BPS) / 1e4` of lagged gross notional (`|y| + |\u03b2| |x|`). A full round-trip is two of those (enter + exit), about **40 bps** at the defaults. That is conservative relative to a 5 bps mid-touch on liquid US large caps, and **still not a live cost model**.

Walk-forward default `z_mode='train_stats'` freezes residual mean and std on the train window. That is stricter (and more brittle) than a rolling z-score; it is the right test of "did we just fit the sample."

## Metric definitions

- **Gross notional return** of the hedge portfolio: `(\u0394y - \u03b2 \u0394x) / (|y_{t-1}| + |\u03b2| |x_{t-1}|)`.
- **Strategy return** (uncosted): `position[t-1] * hp_ret[t]`. The book decided at close *t-1* earns the move into close *t* (no same-bar look-ahead).
- **Costed return**: uncosted minus cost of that bar's change in the *held* book.
- **Total return**: last equity / 1 \u2212 1, equity compounded from daily returns starting at 1.
- **Sharpe**: `mean(r) / std(r, ddof=1) * sqrt(252)` on daily returns. Undefined series with < 2 points \u2192 NaN; zero vol \u2192 0.
- **Max drawdown**: min of `equity / cummax(equity) - 1` (a non-positive number).
- **Trade count**: entries from flat plus sign flips (not fills \u00d7 shares).
- **Turnover**: mean daily `|\u0394position|`.

In-sample (train) metrics are shown because they are the usual way to fool yourself. **Only the test columns can possibly be believed, and even those are one split on one pair.**

## Assumptions

- Prices are daily adjusted closes. Cointegration is tested in **levels**, not log-levels and not returns.
- The residual of a static OLS hedge is the traded spread. \u03b2 does not walk inside a window except when `expanding_walk_forward` re-fits on later train origins.
- We can trade at the close after computing z from that close, and the next day's move is ours. That is optimistic versus next-open or VWAP.
- Dollar book is 1 unit of *y* vs \u03b2 units of *x*, scaled by lagged gross notional. Not share-count integer, not dollar-neutral unless \u03b2 \u2248 y/x.
- Costs scale linearly with \\|\u0394position\\| and hit both legs. No borrow fee on the short, no locate, no halt.

## Why this can fail

Cointegration is a sample property. A KO/PEP residual that looks stationary through one decade can stop mean-reverting on a sugar-tax scare, a spin-off, a rates regime, or a year where one name re-rates on multiple while the other does not. Costs that look small in 5+5 bps still dominate a low-Sharpe spread once you actually turn the book; our defaults are a guess, not an exchange tape. The close-to-close z-score rule still has a look-ahead flavour: we assume the close is tradable at the close. Yahoo adjusted closes are **not a research-grade tape** \u2014 corporate actions, dividends, and error fills are whoever Yahoo had that day; survivorship is not modelled (both names are alive, liquid mega-caps); liquidity, borrow, and overnight gaps are not modelled. A walk-forward Sharpe on one pair after one split is still compatible with luck. None of this is live PnL.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

`requirements.txt` lists the same dependencies as `pyproject.toml`: pandas, numpy, scipy, statsmodels, scikit-learn, yfinance, pytest, jupyter.

## Run tests

Tests use **synthetic** cointegrated series only (no network):

```bash
pytest
```

They fail if the OLS hedge does not recover a planted \u03b2, if costs do not strictly reduce Sharpe versus the same path with zero costs, or if a rolling z-score on a stationary spread is not mean-zero.

## Run the notebook

```bash
jupyter notebook notebooks/pairs_trading.ipynb
```

or `jupyter nbconvert --to notebook --execute notebooks/pairs_trading.ipynb`. Same pipeline without Jupyter: `python notebooks/run_pipeline.py`. It tries `yfinance` KO/PEP and falls back to `data/sample_ko_pep.csv`. If that file is missing, `python data/make_sample.py` writes a **synthetic sample** (not KO/PEP market data).

## Layout

```
src/statarb/     cointegration, signals, backtest, data
tests/           synthetic-data unit tests
notebooks/       walk-forward KO/PEP (or sample) pipeline
data/            make_sample.py + sample_ko_pep.csv (sample, not live)
```

## What is still out of scope

Kalman / rolling \u03b2 as the primary hedge, Johansen-traded eigenvectors, universe search, multiple-testing control, execution at the open, borrow, taxes, and any claim that a backtest Sharpe survives contact with a broker.
