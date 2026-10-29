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


## Oct 14 slice: ranked/capped re-screen, OU time-stop, slower drift

Oct 9 equal-weighted every discovery and only tried a fast beta drift
(1.0 → 2.5). This slice ranks and caps discoveries, wires the existing OU
time-stop into formation/re-screen books, and re-runs nested vs fixed Kalman
across slow / medium / fast drifts. Reproduce with
`python scripts/run_oct14_experiments.py`. Still synthetic, seeded and offline.

### Method

| Module | What it does |
| --- | --- |
| `rank` | Annotate each screened pair with the OU half-life of its formation EG residual; score `(-log eg_p) / half_life` inside a tradable band [2, 40] bars; `select_top` ranks by `eg_p` / `half_life` / `score` and optionally caps at `max_pairs` |
| `rescreen` | `universe_formation_backtest(..., rank_by=, max_pairs=, use_time_stop=)` equal-weights only the selected pairs; reports `mean_selected` vs `mean_discoveries` |
| `formation` | Optional `use_time_stop` fits OU on the formation residual and passes `ceil(k * HL)` into `mean_reversion_positions` |
| `synth` | `make_planted_universe` accepts a per-pair half-life sequence so ranking tests can plant a fast/slow mix |

Causality: half-life, score, cap and the time-stop fit use formation bars only.
A break planted in a later trading block does not change earlier returns or
earlier selection counts.

### Results

Ranked / capped re-screen on a planted fast/slow mix (half-lives 4 / 8 / 28,
10 noise + 3 pairs, 8 blocks, 10 seeds), net of 10 bps/leg. Under **BH**,
median discoveries are already ~0.6 so a top-1 cap is a no-op (all policies
median Sharpe ≈ 0.23, return ≈ 0.4%). Under **naive EG** the screen over-discovers
and the cap matters:

| Policy (naive EG) | Median discoveries | Median selected | Median Sharpe | Median total return | Median cost paid |
| --- | ---: | ---: | ---: | ---: | ---: |
| Equal-weight all | 4.19 | 4.19 | -0.09 | -0.7% | 0.042 |
| `eg_p` top-1 | 4.19 | 1.00 | 0.99 | +4.0% | 0.054 |
| `score` top-1 | 4.19 | 1.00 | 0.99 | +3.7% | 0.054 |
| `half_life` top-1 | 4.19 | 1.00 | 0.89 | +3.3% | 0.050 |
| `eg_p` top-2 | 4.19 | 2.00 | 0.01 | +0.1% | 0.045 |

Reading: equal-weighting every naive discovery trades the false pairs and
pays for them (negative median Sharpe). Keeping only the strongest EG p (or
the composite score) flips the book to a positive median Sharpe on this
synthetic mix; top-2 already dilutes most of the gain. Ranking is **not** a
substitute for FDR control — on pure noise, BH (with or without a score cap)
stays flat (median selected 0, cost 0), which is the right answer.

OU time-stop on the BH re-screen book (planted HL = 5, `k = 3`, 10 seeds):

| Time-stop | Median Sharpe | Median total return | Median trades | Median cost paid |
| --- | ---: | ---: | ---: | ---: |
| Off | 0.51 | +2.2% | 10.5 | 0.028 |
| On (3 × HL) | 0.31 | +0.9% | 12.5 | 0.035 |

Reading: the time-stop **hurts** here. Mean-reversion that would have paid
gets cut early, turnover rises, and net Sharpe falls. It is a risk control,
not an edge, on this generator.

Nested vs fixed Kalman `q_beta` across drift speeds (12 seeds, 8 blocks,
HL = 5), net of costs:

| Scenario | Median Sharpe, fixed `1e-6` | Median Sharpe, nested | Nested wins | Median return, fixed / nested |
| --- | ---: | ---: | ---: | ---: |
| Constant beta | -0.14 | -0.02 | 6/12 | -0.2% / -0.0% |
| Slow drift 1.0 → 1.3 | -0.10 | -0.38 | 1/12 | -0.5% / -1.2% |
| Medium drift 1.0 → 1.5 | -0.97 | -0.85 | 7/12 | -4.8% / -4.9% |
| Fast drift 1.0 → 2.5 | -1.59 | -1.07 | 11/12 | -13.7% / -8.4% |

Reading: nested tuning still helps most when drift is fast (same qualitative
story as Oct 9). On a *slow* drift it **overfits** the noisy inner Sharpe and
loses to the fixed default (1/12 wins). Medium drift is a wash after costs;
every drifting book still loses money. A 3 × HL time-stop on the single-pair
OLS book under fast drift does not rescue it (median Sharpe -1.39 → -1.42).

### What the tests check

- Rank: formation residual recovers planted HL; scores outside the HL band are
  `-inf`; `select_top` by `eg_p` / `score` orders and caps correctly.
- Synth: per-pair half-life sequences plant a recoverable fast/slow mix.
- Re-screen: `max_pairs` never exceeds discoveries; score-ranked causality
  (future breaks do not change earlier returns or selection counts); time-stop
  paths stay causal.
- Formation: time-stop off matches the legacy path; time-stop on is causal
  under a late break.

### Weaknesses

- Still synthetic only: no real-pair universe or bundled public tape.
- Under BH the top-1 cap is usually a no-op; ranking shines when the screen is
  permissive (naive), which is exactly when you should prefer BH instead.
- Composite score uses a fixed HL band and `-log p`; it is not a nested
  estimate of tradable Sharpe.
- OU time-stop reduced net Sharpe on the planted BH book; no adaptive multiple
  was tried.
- Slow beta drift makes nested `q_beta` tuning worse than the default — the
  inner score is too noisy for small misspecifications.
- Costs remain bps on gross notional; no borrow, no impact, closes only.
- `q_alpha` is still never tuned.

## Oct 20 slice: regularised nested tuning, joint q_alpha, borrow fee, formation-chosen time-stop

Oct 14 left four open items: nested `q_beta` tuning overfits on slow drift, `q_alpha` was never
tuned, costs had no borrow, and the fixed 3 × HL time-stop hurt with no adaptive multiple tried.
This slice adds one tool for each and measures it on the same generators and seeds. Reproduce with
`python scripts/run_oct20_experiments.py`. Still synthetic, seeded and offline. Every book is
out-of-sample (outer 252/63 walk-forward, every tune inside the formation window) and net of
10 bps per leg plus the Kalman re-hedge charge.

### Method

| Piece | What it does |
| --- | --- |
| `nested.tune_q` | Scores each candidate by the median of its inner-window net Sharpes, then selects with `best` (legacy, chosen q bit-identical to Oct 14), `one_se` (closest-to-default candidate within one standard error of the best median; SE of a median ≈ 1.2533·sd/√n) or `sign` (leave the default only if a candidate beats it in paired inner windows with one-sided sign-test p < 0.10 and a positive median gain). Optional joint `(q_beta, q_alpha)` grid |
| `nested_kalman_backtest(selection=, q_alpha_grid=)` | Uses the rule per outer window and records the chosen `q_alpha` and the fraction of windows that kept the default `q_beta` |
| `borrow_bps` (formation, re-screen, nested) | Short-leg stock-borrow fee, annualised, accrued per held bar on the short leg's notional (`|β|x` long spread, `y` short spread) as a fraction of lagged gross notional. Default 0 keeps every earlier table |
| `choose_time_stop_multiple` / `time_stop_multiple="auto"` | Per window, replays the formation residual's own z rule with k ∈ {2, 3, 5, none} net of costs and keeps the best net Sharpe (ties to the larger k); formation bars only |

### Results

Nested selection rules across drift speeds (12 seeds, same seeds as Oct 14, HL = 5, 8 blocks),
median Sharpe net of costs, wins vs the fixed default `q_beta = 1e-6`:

| Scenario | Fixed `1e-6` | Nested `best` (Oct 14 rule) | Nested `one_se` | Nested `sign` | Windows on default q (best / one_se / sign) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Constant beta | -0.14 | **-0.02** (6/12) | -0.14 (2/12) | -0.12 (2/12) | 81% / 100% / 100% |
| Slow drift 1.0 → 1.3 | **-0.10** | -0.38 (1/12) | -0.20 (2/12) | -0.26 (1/12) | 63% / 94% / 100% |
| Medium drift 1.0 → 1.5 | -0.97 | -0.85 (7/12) | **-0.77** (5/12) | -0.87 (4/12) | 50% / 81% / 88% |
| Fast drift 1.0 → 2.5 | -1.59 | **-1.07** (11/12) | -1.17 (10/12) | -1.12 (10/12) | 25% / 44% / 56% |

Reading: regularising the inner choice does what it is built for, and no more. On slow drift
`one_se` halves the overfit damage (-0.38 → -0.20) and is best on medium drift, but it still loses
to simply not tuning on slow drift (-0.20 vs -0.10). On constant beta it gives back the small gain
the unregularised search found, and on fast drift it gives back about 0.1 of Sharpe. The sign test
sits between the two. No rule wins every regime, and **every drifting book still loses money after
costs**. Tuning `q_alpha` jointly on {1e-6, 1e-5, 1e-4, 1e-3} changes almost nothing: the one-SE
book is identical in every regime, and `best` gets slightly worse on medium drift (-0.85 → -0.89).
`q_alpha` is not where the drift loss comes from.

Borrow-fee sensitivity (10 seeds; BH re-screen of 3 planted + 10 noise assets, and one known
pair with static OLS, HL = 5):

| Borrow (bps / yr) | BH re-screen median Sharpe | return | borrow paid | Single pair median Sharpe | return |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.29 | +0.65% | 0.0000 | 0.74 | +1.57% |
| 25 | 0.27 | +0.62% | 0.0004 | 0.72 | +1.52% |
| 100 | 0.24 | +0.52% | 0.0016 | 0.66 | +1.39% |
| 300 | 0.16 | +0.32% | 0.0048 | 0.49 | +1.02% |

Reading: easy-to-borrow rates (25 bps) cost little next to the 10 bps/leg trade costs (0.027 paid
on the BH book), but a hard-to-borrow 300 bps cuts the BH book's median Sharpe by about 45%. The
fee scales with time in the trade, so slower spreads pay more of it.

OU time-stop, single pair, static OLS (12 seeds):

| Scenario | Off | Fixed 3 × HL | Auto (formation-chosen k) | Auto chose "no stop" |
| --- | ---: | ---: | ---: | ---: |
| Constant beta | 0.43 | 0.02 | 0.43 (beats off on 0/12 seeds) | 95% of windows |
| Fast drift 1.0 → 2.5 | -1.39 | -1.42 | -1.39 (identical to off) | 100% of windows |

Reading: on these generators the formation replay almost always picks "no stop", so `auto`
avoids the fixed stop's damage (0.43 → 0.02 on constant beta) but adds nothing over leaving the
stop off. It is a guard against a bad fixed multiple, not an edge.

### What the tests check

- Borrow: hand-computed 4-bar fee with the short leg following the sign; zero borrow equals the
  legacy book in formation, re-screen and nested; monotone in the rate; positions and trade counts
  unchanged; fee only on held bars; negative rates rejected.
- Selection: sign-test p-values; `best` tie-break to the default; `one_se` keeps the default inside
  the noise band and leaves it for a large clean edge; `sign` needs consistent paired wins and the
  default in the grid; joint grid covers every pair; new rules stay causal under a late break; the
  default rule is the legacy book.
- Auto time-stop: chosen k is a candidate, deterministic, recorded per window, causal under a late
  break; fixed-k path unchanged.

### Weaknesses

- Still synthetic only: no real-pair universe or bundled public tape.
- No inner selection rule wins across regimes; a regime-aware choice (e.g. drift detection on the
  formation window) was not tried, and choosing among rules would need its own nested layer.
- The one-SE band uses the best candidate's window spread only; the sign test ignores magnitude.
- Borrow is a flat annual rate: no recall, no locate failure, no rate that rises with crowding.
- The auto time-stop is scored in-sample on the formation window; on these generators it nearly
  always picks no stop, so it was not tested where a stop should help (e.g. a planted break).
- Costs otherwise remain bps on gross notional; no impact, closes only.

## Oct 29 slice: stops on a pair that breaks

Oct 20 left the auto time-stop untested "where a stop should help (e.g. a planted break)". This
slice plants one and compares the time stops with a new adverse-z stop-loss. Reproduce with
`python scripts/run_oct29_experiments.py` (synthetic, seeded, offline, ~8 s). Static OLS book,
252/63 walk-forward, every traded bar out-of-sample, net of 10 bps per leg. The stop levels
{3, 4} and 3 × HL were fixed before the first run, and nothing is tuned on these 12 new seeds
(900–911).

### Method

| Piece | What it does |
| --- | --- |
| `synth.make_broken_pair` | `make_ou_pair` (HL = 5) whose cointegration breaks at a chosen bar: `random_walk` (from the break on, the spread is a driftless random walk with the OU innovation scale) or `level_shift` (the OU re-centres `shift_sigmas` σ away). The pre-break path, x leg and hedge are identical across kinds, so books are compared on common random numbers |
| `mean_reversion_positions(z_stop=)` | Flattens a held position when z moves against it beyond `z_stop` (long at z < −z_stop, short at z > z_stop) and blocks that side until \|z\| < z_enter, like the time stop. Off by default; must exceed z_enter |
| `formation_trading_backtest(z_stop=)` | Applies it on the trading-block z (formation mean/std) and records `z_stops` per window |

Break at bar 756 of 1,260, so trading years 1–2 are clean and years 3–4 are post-break.

### Results (12 seeds, medians; wins = seeds where the book's Sharpe beats `off`)

| Scenario | Book | Sharpe | wins | post-break return | worst 63-bar window | max DD | stops / path |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| No break | off | **0.58** | – | +1.5% | −0.3% | −1.0% | – |
| | time stop 3×HL | 0.25 | 0/12 | +0.5% | −0.5% | −1.5% | – |
| | time stop auto | 0.58 | 0/12 | +1.5% | −0.3% | −1.0% | – |
| | z stop 3 | 0.06 | 0/12 | +0.3% | −1.0% | −2.1% | 4.3 |
| | z stop 4 | 0.51 | 0/12 | +1.5% | −0.3% | −1.2% | 0.4 |
| Random-walk break | off | **−0.08** | – | −2.7% | −2.0% | −4.1% | – |
| | time stop 3×HL | −0.32 | 1/12 | −1.7% | −1.9% | −4.0% | – |
| | time stop auto | −0.08 | 0/12 | −2.7% | −2.0% | −4.1% | – |
| | z stop 3 | −0.44 | 1/12 | −2.6% | −1.3% | −4.0% | 6.9 |
| | z stop 4 | −0.08 | **7/12** | −1.7% | **−1.4%** | **−3.4%** | 2.7 |
| | z stop 4 + 3×HL | −0.27 | 5/12 | −1.6% | −1.4% | −3.6% | 1.8 |
| 4σ level shift | off | **0.45** | – | +1.3% | −0.7% | −1.8% | – |
| | time stop 3×HL | 0.17 | 1/12 | +0.3% | −0.9% | −2.5% | – |
| | time stop auto | 0.45 | 1/12 | +1.3% | −0.7% | −1.8% | – |
| | z stop 3 | −0.16 | 0/12 | −0.7% | −1.2% | −2.7% | 5.3 |
| | z stop 4 | 0.31 | 4/12 | +0.9% | −0.8% | −2.1% | 1.6 |

Reading:

- **When the pair really dies (random walk), z stop 4 is the only book that helps on most seeds.**
  It wins on 7 of 12 seeds, cuts the median post-break loss from −2.7% to −1.7% and the worst
  window from −2.0% to −1.4%, and the median Sharpe does not move (−0.08 → −0.08). It does not
  make the broken pair profitable. Every book still loses money after the break.
- **It costs Sharpe when nothing breaks.** With no break, z stop 4 loses on all 12 seeds
  (0.58 → 0.51), and z stop 3 gives up almost the whole edge (0.06), because a healthy OU often
  overshoots 3σ before it reverts. The stop is insurance, and this is its premium.
- **A level shift is mostly absorbed by re-formation.** Each 252-bar formation window re-fits the
  mean, so a year after the shift the book is trading the new level. `off` keeps a 0.45 Sharpe
  and every stop does worse on median.
- **The auto time-stop still adds nothing.** It replays the formation window, and before the break
  that window is clean, so it picks "no stop" and matches `off` almost exactly in all three
  scenarios. The fixed 3×HL stop hurts everywhere, as it did on Oct 20.

### What the tests check

- Generator: `none` equals `make_ou_pair`; the pre-break path and x leg are shared across kinds;
  the level shift is exactly the planted amount; ADF rejects the pre-break spread on 10/10 seeds
  and the post-break random walk on at most 3/10.
- Stop rule: hand paths for long and short stops, same-side blocking and re-entry, favourable
  moves never stop, composition with `max_hold`, legacy identity when off, validation.
- Walk-forward: off is bit-identical for OLS and Kalman; causal under a late break (returns and
  positions identical before it); stops concentrate after a planted shift.
- The reading above at reduced scale (4 seeds): `test_oct29_claims.py`.

### Weaknesses

- Still synthetic. The break is planted in the same OU family the book assumes, and only two
  shapes were tried.
- The stop level is fixed, not chosen. A formation-window choice (like the auto time-stop) would
  pick "no stop" for the same reason the auto time-stop does, because the formation window
  before a break is clean.
- The real remedy for a dead pair is to stop trading it. That means a post-break re-test
  (re-screen with a stricter gate, or a CUSUM on the trading-block z), which was not tried here.
- Static OLS only. The Kalman hedge adapts to drift and may hide or mimic a break.

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
src/statarb/     cointegration, signals, backtest, data, synth, ou, kalman, screening, formation, rescreen, nested, rank
scripts/         run_oct05_experiments.py, run_oct09_experiments.py, run_oct14_experiments.py, run_oct20_experiments.py, run_oct29_experiments.py
tests/           synthetic-data unit tests
notebooks/       walk-forward KO/PEP (or sample) pipeline
data/            make_sample.py + sample_ko_pep.csv (sample, not live)
```

## What is still out of scope

Johansen-traded eigenvectors, execution at the open, borrow recall / locate, taxes, a bundled public tape, and any claim that a backtest Sharpe survives contact with a broker. Ranking/capping of re-screened discoveries and an OU time-stop on formation/re-screen books landed in the Oct 14 slice; joint `q_alpha` tuning, a flat borrow fee and regularised nested selection landed on Oct 20; planted cointegration breaks and an adverse-z stop-loss landed on Oct 29.
