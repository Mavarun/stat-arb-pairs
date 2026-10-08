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

## Oct 5 slice: OU half-life, Kalman hedge, multiple-testing screen, rolling walk-forward

All numbers below are **synthetic, seeded and offline** (`src/statarb/synth.py` plants the hedge ratio,
intercept and OU half-life, so every estimate can be checked against the truth). Reproduce with
`python scripts/run_oct05_experiments.py`. None of this is a claim about live markets.

### Method

| Module | What it does |
| --- | --- |
| `synth` | Cointegrated pairs `y = c + beta_t x + s`, with `s` an AR(1) spread whose half-life is exact (`rho = 0.5**(1/hl)`); random-walk universes where every pair is a false pair; universes with planted true pairs |
| `ou` | AR(1) fit of the spread: half-life, stationary std, entry/exit levels at `mu +/- z*sigma_eq`, and a `3 x half-life` time stop |
| `kalman` | Random-walk state `(alpha, beta)` filter, initialised by OLS on a warm-up window. Signal is the innovation z-score (prior state, never sees bar t) |
| `screening` | Engle-Granger both orderings (max p), Benjamini-Hochberg FDR at 5%, Johansen trace as a confirmation, and a summary of false positives / power / FDP per rule |
| `formation` | Non-overlapping 252-bar formation / 63-bar trading blocks. Static OLS beta frozen per block vs Kalman beta updated each bar. Forced flat at block end. Costs: 10 bps per leg per unit change, plus a re-hedge charge when the Kalman beta moves while a position is held |

### Results (20 seeds unless noted)

OU half-life recovery on 1,500 bars: planted 5 / 10 / 20 bars, median estimate 5.02 / 10.10 / 19.96
(10th to 90th percentile 4.6 to 5.6, 8.9 to 11.7, 16.6 to 22.4).

Hedge tracking when beta drifts linearly from 1.0 to 2.5: median beta RMSE 0.215 for Kalman vs 0.990
for a beta frozen at the warm-up OLS.

Pair screening, 190 candidate pairs per universe, 10 seeds:

| Rule | False pairs per pure-noise universe | Power (4 planted pairs) | FDP with planted pairs |
| --- | --- | --- | --- |
| EG one ordering, p < 0.05 | 8.5 | 1.000 | 0.748 |
| EG both orderings, p < 0.05 | 3.6 | 1.000 | 0.513 |
| Johansen trace alone, 95% | 18.8 | 1.000 | 0.859 |
| EG + Benjamini-Hochberg | 0.0 | 0.975 | 0.000 |
| BH + Johansen confirm | 0.0 | 0.975 | 0.000 |

Rolling walk-forward, net of costs (half-life 5, 12 trading blocks):

| Scenario | Median Sharpe, static OLS | Median Sharpe, Kalman | Kalman wins | Median total return, static / Kalman |
| --- | --- | --- | --- | --- |
| Constant beta | 0.50 | -0.08 | 1/20 | 1.8% / -0.2% |
| Beta drifts 1.0 to 2.5 | -1.29 | -1.09 | 15/20 | -21.0% / -16.4% |

Reading: Kalman tracks a moving hedge far better and loses less than a stale static hedge, but **both
books lose money after costs when beta drifts this fast**, and when beta is truly constant the filter's
extra flexibility only adds noise; it also trades about half as often (median 7 vs 15 entries). Better beta tracking is not the same as a
tradable edge.

### What the tests check

- OU fit recovers planted half-lives; thresholds and time stop are well defined only when `0 < b < 1`.
- Kalman is causal (a future shock does not move past estimates) and beats static OLS on drifting beta.
- On pure random-walk universes, BH keeps false discoveries near zero where naive p < 0.05 does not.
- Walk-forward: no trading inside the first formation window, flat at every block end, a break in a
  later block does not change earlier returns, costs are monotone, zero cost equals gross PnL,
  a hand-computed 3-bar PnL, and seed sweeps for both hedge scenarios (not one lucky seed).

### Weaknesses

- Synthetic only: no real-pair universe has been screened yet, so there is no out-of-sample market result.
- Kalman `q_beta` / `q_alpha` are fixed defaults, not tuned in a nested walk-forward.
- The walk-forward trades one known pair; it does not yet re-screen the universe in each formation window.
- Costs are bps on gross notional; no borrow, no market impact, closes only.


## Oct 9 slice: per-window universe re-screening and nested Kalman q tune

Oct 5 traded one known pair and used fixed Kalman noise. This slice closes those
two gaps on the same synthetic generators. Reproduce with
`python scripts/run_oct09_experiments.py`. Still synthetic, seeded and offline —
not a live-market claim.

### Method

| Module | What it does |
| --- | --- |
| `rescreen` | Each 252-bar formation window runs `screen_pairs` on the full universe; only discoveries of a chosen rule (`naive_single`, `naive`, `bh`, `bh_johansen`) are traded in the next 63-bar block. Equal-weight across discoveries; flat when none; forced flat at block end |
| `nested` | Outer 252/63 walk-forward. Inside each formation window, roll 84/21 blocks over a log-spaced `q_beta` grid `(1e-8 … 1e-4)`, pick the median-net-Sharpe winner, trade the outer block with that `q_beta`. `q_alpha` stays at the package default |

Causality: screening, OLS/Kalman warm-up and the inner q grid never see an outer
trading bar. A break planted in a later block does not change earlier returns or
earlier chosen `q_beta`.

### Results

Re-screen on pure noise (12 independent random walks, 66 candidate pairs, 8
blocks, 10 seeds), net of 10 bps/leg:

| Rule | Median discoveries / window | Median frac windows traded | Median Sharpe | Median total return | Median cost paid |
| --- | --- | --- | --- | --- | --- |
| EG one ordering, p < 0.05 | 3.06 | 0.81 | -0.27 | -4.0% | 0.031 |
| EG both orderings, p < 0.05 | 0.81 | 0.62 | -0.16 | -2.5% | 0.023 |
| EG + Benjamini-Hochberg | 0.00 | 0.00 | 0.00 | 0.0% | 0.000 |
| BH + Johansen confirm | 0.00 | 0.00 | 0.00 | 0.0% | 0.000 |

Reading: without FDR control the walk-forward *trades* the false pairs and
pays for them. BH keeps the book flat on pure noise (median), so the cost of
multiple testing shows up as turnover rather than as a screen-time footnote.

Re-screen with 3 planted pairs + 10 noise assets (10 seeds):

| Rule | Median discoveries / window | Median frac windows traded | Median Sharpe | Median total return |
| --- | --- | --- | --- | --- |
| EG both, p < 0.05 | 4.38 | 1.00 | 0.54 | 2.6% |
| EG + BH | 0.81 | 0.50 | 0.29 | 0.7% |
| BH + Johansen | 0.81 | 0.50 | 0.29 | 0.7% |

Oracle (trade one known planted pair, static OLS) vs BH-rescreened universe book
(10 seeds):

| Book | Median Sharpe | Median total return | Median discoveries / window |
| --- | --- | --- | --- |
| Oracle known pair | 0.33 | 0.9% | 1.00 |
| BH re-screen universe | 0.45 | 0.5% | 0.75 |

BH's higher median Sharpe with a *lower* total return is the flat-window effect:
roughly half the blocks do nothing, so the Sharpe denominator shrinks while the
compounded return stays smaller than the always-on oracle. Power is real but
incomplete — median discoveries stay below the three planted pairs.

Nested vs fixed Kalman `q_beta` (12 seeds, 8 blocks, half-life 5), net of costs:

| Scenario | Median Sharpe, fixed `1e-6` | Median Sharpe, nested | Nested wins | Median return, fixed / nested | Median chosen `q_beta` |
| --- | --- | --- | --- | --- | --- |
| Constant beta | 0.15 | 0.15 | 4/12 | 0.1% / 0.2% | `1e-6` (11/12 seeds) |
| Beta drifts 1.0 to 2.5 | -1.79 | -1.32 | 9/12 | -21.9% / -8.2% | `1e-5` (7) or `1e-4` (3) |

Reading: when beta is truly constant the inner search almost always keeps the
default, so nested and fixed are the same book. When beta drifts, the tune
moves `q_beta` up and loses less than the stale default — but **both books still
lose money after costs** on this drift speed. Nested tuning is not a free edge;
it is a less-bad hedge under misspecification.

### What the tests check

- Re-screen: no trading inside the first formation window, future breaks do not
  change earlier returns or discovery counts, costs are monotone, BH trades less
  than naive on pure noise, planted pairs produce trades, empty discoveries stay
  flat.
- Nested: grid coverage, short-sample fallback to `Q_BETA`, future breaks do not
  change earlier chosen `q` or returns, formation-only tuning, monotone costs,
  fixed path matches a direct `formation_trading_backtest` with the default q.

### Weaknesses

- Still synthetic only: no real-pair universe has been screened, and no cited
  public tape is bundled offline yet.
- Re-screen equal-weights every discovery; it does not rank by half-life, EG p,
  or estimated Sharpe, and it does not cap the number of concurrent pairs.
- Nested grid is coarse (5 points) and scores by Sharpe on short inner blocks;
  that score is noisy, and `q_alpha` is never tuned.
- Drift scenario still loses money after costs even with the tuned q; a slower
  drift or a different signal rule was not tried.
- Costs remain bps on gross notional; no borrow, no impact, closes only.

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
src/statarb/     cointegration, signals, backtest, data, synth, ou, kalman, screening, formation, rescreen, nested
scripts/         run_oct05_experiments.py, run_oct09_experiments.py
tests/           synthetic-data unit tests
notebooks/       walk-forward KO/PEP (or sample) pipeline
data/            make_sample.py + sample_ko_pep.csv (sample, not live)
```

## What is still out of scope

Johansen-traded eigenvectors, ranking or capping re-screened discoveries, joint tuning of `q_alpha` with `q_beta`, execution at the open, borrow, taxes, a bundled public tape, and any claim that a backtest Sharpe survives contact with a broker.
