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

## Nov 3 slice: re-testing a pair after it breaks

Oct 29 ended with "the real remedy for a dead pair is to stop trading it", untried. This slice tries two
re-tests that read only past bars: a **CUSUM halt** inside the trading block and an **Engle-Granger gate**
on the formation window before it. Reproduce with `python scripts/run_nov03_experiments.py` (~50 s) and
`python scripts/run_nov03_calibration.py` (~2 min); synthetic, seeded, offline. Static OLS book, 252/63
walk-forward, every traded bar out-of-sample, net of 10 bps per leg. CUSUM `k = 0.5, h = 5`, gate levels
{0.01, 0.05}, z stop 4 and the 20 seeds (940-959) were fixed before the first run. `k`, `h` were chosen on
the **no-break** false-alarm rate (calibration seeds 1000-1059), not on any break outcome.

### Method

| Piece | What it does |
| --- | --- |
| `breakwatch.cusum_alarm` | Two-sided Page CUSUM on the block z-score divided by `sqrt((1+rho)/(1-rho))` (long-run std of an AR(1) sum, `rho` from the formation OU fit), so the sum is a unit-variance walk. Alarm at the first bar with `max(S+, S-) > h` |
| `formation_trading_backtest(cusum_halt=True)` | Flat from the alarm bar to the block end; the next block starts fresh. Kalman innovations use `rho = 0` |
| `formation_trading_backtest(gate_p=)` | Engle-Granger (y on x) on the formation window; the whole block stays flat when `p > gate_p`. Records `eg_p`, `gated` |

The break is planted on bar 756 (first bar of a trading block, clean formation window) or on bar 786
(30 bars into one). Books are compared on common random numbers.

### Calibration on healthy pairs (60 seeds x 16 blocks per half-life)

| Half-life (bars) | CUSUM k=0.25 | **CUSUM k=0.5** | CUSUM k=1.0 | EG gate p<0.01 gates | EG gate p<0.05 gates |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 31.5% | **5.9%** | 0.2% | 3.8% | 1.5% |
| 5 | 26.4% | **4.1%** | 0.3% | 20.1% | 5.1% |
| 10 | 21.9% | **4.5%** | 0.6% | 74.6% | 44.3% |
| 20 | 21.0% | **4.9%** | 0.7% | 91.9% | 77.0% |
| 40 | - | - | - | 96.8% | 87.0% |

(CUSUM columns: share of healthy blocks with an alarm at h = 5. Gate columns: share of healthy formation
windows that fail the gate.) The long-run-variance scaling makes the CUSUM false-alarm rate nearly
independent of the half-life. The Engle-Granger gate is not: with 252 bars it has little power on a slow
spread, so it starves a healthy slow pair.

### Results (20 seeds, half-life 5, break on the first bar of a block; Sharpe medians, wins/losses = paired seeds vs `off`)

| Scenario | Book | Sharpe | wins / losses (sign p) | post-break return | max DD | detected (median delay, bars) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| No break | off | **0.52** | - | +1.3% | -1.0% | - |
| | z stop 4 | 0.34 | 0 / 8 (0.008) | +1.0% | -1.2% | - |
| | CUSUM halt | 0.44 | 1 / 11 (0.006) | +1.2% | -1.1% | 8 false halts after bar 756 |
| | EG gate 0.01 | 0.50 | 5 / 9 (0.42) | +1.3% | -1.0% | 22% of blocks gated |
| | EG gate 0.05 | 0.52 | 2 / 3 (1.0) | +1.3% | -1.0% | 6% gated |
| Random-walk break | off | **-0.35** | - | -3.2% | -4.3% | - |
| | z stop 4 | -0.29 | 4 / 16 (0.012) | -3.2% | -4.9% | - |
| | CUSUM halt | -0.17 | 10 / 8 (0.82) | -2.0% | -4.5% | 19/20 (31) |
| | EG gate 0.01 | **0.06** | **15 / 5 (0.041)** | **-0.6%** | **-1.3%** | gated, not alarmed |
| | EG gate 0.05 | -0.03 | 14 / 6 (0.12) | -1.5% | -2.2% | - |
| | CUSUM + gate 0.05 | -0.01 | 14 / 6 (0.12) | -0.9% | -1.6% | 17/20 (26) |
| 4-sigma level shift | off | **0.28** | - | +0.5% | -1.8% | - |
| | z stop 4 | 0.09 | 5 / 15 (0.041) | -0.2% | -2.1% | - |
| | CUSUM halt | 0.17 | 9 / 11 (0.82) | +0.3% | -1.6% | 20/20 (6.5) |
| | EG gate 0.01 | 0.28 | 11 / 9 (0.82) | +0.2% | -1.3% | - |

With the break 30 bars into a block (formation window contaminated for four blocks) the gate's advantage
is smaller and no longer significant: random-walk break Sharpe -0.16 (off) vs 0.11 (gate 0.01), 12 wins /
8 losses (p = 0.50); level shift 0.16 vs 0.20, 14 / 6 (p = 0.12).

Reading:

- **The formation-window gate is the only re-test that pays on a dead pair, and only for a fast pair.**
  With the break on a block boundary it turns the median Sharpe from -0.35 to +0.06, cuts the post-break
  loss from -3.2% to -0.6% and the max drawdown from -4.3% to -1.3%, and wins 15 of 20 seeds (sign p = 0.041,
  one of many paired comparisons in this table with no multiplicity correction, so read it as suggestive). Cost paid halves because the book is flat
  91% of the post-break bars. A new generator seed set reproduces the direction of Oct 29 (a stop is not
  enough) but not its levels: `off` is -0.35 here vs -0.08 there, so medians over 12-20 seeds move a lot.
- **Its premium is large for anything slower.** It gates 20% of healthy half-life-5 blocks (the no-break
  Sharpe moves 0.52 to 0.50, 5 wins / 9 losses, not significant) but 75% at half-life 10 and 92% at half-life
  20: there it is a rule against trading slow pairs, not a break detector. p < 0.05 gates 6% at half-life 5
  and 44% at half-life 10 but helps less (-0.35 to -0.03).
- **The CUSUM is a fast alarm that does not pay.** It catches a 4-sigma level shift in a median of 6.5 bars
  (20/20 seeds) and a random-walk break in 31 bars (19/20), but its Sharpe is not significantly better on a
  dead pair (10 / 8), is worse on a level shift (0.28 to 0.17, which re-formation absorbs anyway), and its
  4% per-block false-alarm rate costs a healthy pair 0.08 Sharpe (loses 11 of 12 paired seeds, p = 0.006).
  Stacking it with the gate does not beat the gate alone (-0.01 vs 0.06).
- **The Oct 29 z stop is the worst book here.** On a healthy pair it loses on 8 of 8 paired seeds and it
  loses to `off` on both break types (4 / 16 and 5 / 15). Treat Oct 29's 7/12 random-walk win as a
  12-seed result that did not replicate on 20 new seeds.

### What the tests check

- CUSUM: hand-computed path, the `rho` scaling is exactly an input rescaling, alarms on a sustained shift
  (either sign) and not on noise, the alarm cannot depend on later bars, NaN carries state, validation.
- Walk-forward: off is bit-identical for OLS and Kalman; the book is flat from `halt_bar`; halts concentrate
  after a planted shift; causal under a late break; a halt never adds trades; a gated block is flat, earns
  nothing and pays nothing; a looser gate never gates more; gate and CUSUM compose.
- Calibration: the default false-alarm rate is < 12% at half-lives 3 and 20, `k` is what controls it, and
  the gate gates > 50% of healthy half-life-20 windows against < 15% at half-life 3.
- The reading above at reduced scale (8 seeds): `test_nov03_claims.py`.

### Weaknesses

- Still synthetic, one pair at a time, half-life 5 only in the break experiment, two break shapes, and 20
  seeds, so most paired comparisons are not significant.
- The gate needs a fast spread; the half-life-10+ premium makes it unusable unless the test is given more
  bars or the gate is conditioned on the pair's own half-life.
- A mid-window break contaminates the formation mean, std and `rho` (the CUSUM was silent on a break placed
  inside a formation window in the first test run), so both re-tests are weakest exactly when a break is
  most likely to be noticed late.
- The Kalman hedge was only checked for plumbing (`rho = 0`), not evaluated.
- The CUSUM drift allowance `k` and level `h` are one null-calibrated pair; a joint calibration over trade
  duration or an exit-by-block rule instead of "flat to block end" was not tried.

## Nov 8 slice: a gate calibrated to the pair's own half-life

Nov 3 left the Engle-Granger gate "unusable unless the test is given more bars or the gate is conditioned
on the pair's own half-life": with 252 formation bars a fixed `p < 0.01` keeps a healthy half-life-10 pair
flat in 75% of blocks. This slice conditions the gate on the half-life and runs the break experiment at
half-lives 5, 10 and 20 (Nov 3 tried 5 only). Reproduce with `python scripts/run_nov08_calibration.py`
(~6 min) and `python scripts/run_nov08_experiments.py` (~6 min); synthetic, seeded, offline. Static OLS
book, 252/63 walk-forward, every traded bar out-of-sample, net of 10 bps per leg. Books, quantiles
{0.95, 0.90}, the 20 seeds (1200-1219) and both break bars were fixed before the first run.

### Method

| Piece | What it does |
| --- | --- |
| `gatecal.calibrated_gate_level(hl, n, quantile)` | Simulates 400 healthy pairs (`make_ou_pair`, own seeds 20000+) with half-life `hl` on `n` bars and returns the `quantile` of their Engle-Granger p-values. A window above that level is worse than 95% (or 90%) of healthy windows for that half-life. Cached on a 5% log grid; an undefined half-life returns 1.0 (never gates) |
| `formation_trading_backtest(gate_quantile=, gate_ref_half_life=)` | Gates the block when the formation EG p exceeds that level. The reference half-life is given, or estimated on the **first** formation window (the half-life the pair was picked with) and frozen, so a later break cannot move it. Records `gate_level`, `gate_ref_half_life`. Off by default, exclusive with `gate_p` |
| `run_nov08_experiments.summarise` | Paired sign tests vs `off` and vs the fixed 0.01 gate, plus a Holm adjustment over every comparison in the table (Nov 3 had none) |

Level at 252 bars, q0.95: 0.001 (HL 3), 0.027 (HL 5), 0.29 (HL 10), 0.67 (HL 20), 0.86 (HL 40).

### Premium and power (healthy seeds 1000-1059, dead seeds 1100-1159; share of formation windows gated)

| Half-life | Fixed 0.01: healthy / dead | q0.95 planted HL: healthy / dead | q0.95 estimated HL: healthy / dead | Median estimated HL |
| ---: | ---: | ---: | ---: | ---: |
| 3 | 4% / 98% | 10% / 99% | 14% / 100% | 2.6 |
| 5 | 20% / 98% | 9% / 96% | 22% / 95% | 4.1 |
| 10 | 75% / 98% | 6% / 68% | 30% / 75% | 6.7 |
| 20 | 92% / 98% | 4% / 30% | 32% / 55% | 10.2 |
| 40 | 97% / 98% | 3% / 13% | 36% / 40% | 13.1 |

("Dead" = the spread is a random walk for the whole path.) The calibration does what it says on the
premium: with the true half-life the gate keeps 3-10% of healthy windows flat at every speed. But it pays
for it in power, and the slower the pair the more it pays: a dead window has a roughly flat p-value, so a
level of 0.67 catches only 30% of them. The level cannot invent information the 252-bar test does not have.
In practice the reference is estimated, and the AR(1) half-life of a 252-bar OLS residual is biased fast
(median 10.2 for a planted 20), so the estimated-reference gate sits between the two.

### Results (20 seeds, Sharpe medians net of costs; wins / losses = paired seeds; p = sign test, Holm-adjusted in brackets)

| Half-life, scenario | off | Fixed 0.01 | q0.95 estimated | q0.95 planted | Healthy blocks gated (fixed / q0.95 est.) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5, no break | **0.70** | 0.60 | 0.68 (vs fixed 7/6) | 0.68 (vs fixed 11/0, p 0.001 [0.055]) | 18% / 21% |
| 5, random walk (bar 756) | -0.31 | 0.07 | 0.12 (vs fixed 10/3, p 0.09) | **0.18** | |
| 5, random walk (bar 786) | -0.23 | 0.15 | **0.33** (vs fixed 10/2, p 0.039 [1.0]) | 0.21 | |
| 10, no break | **0.14** | -0.01 | 0.11 (vs fixed 12/7, p 0.36) | 0.09 | 73% / 21% |
| 10, random walk (bar 756) | -0.60 | **-0.07** | -0.30 (vs fixed 7/12) | -0.39 (vs fixed 3/17, p 0.003 [0.13]) | |
| 10, random walk (bar 786) | -0.44 | **-0.05** | -0.15 (vs fixed 8/11) | -0.30 | |
| 20, no break | -0.41 | **0.00** | -0.37 (vs fixed 3/16, p 0.004 [0.20]) | -0.47 | 94% / 27% |
| 20, random walk (bar 756) | -0.85 | **0.00** | -0.65 (vs fixed 2/17, p 0.001 [0.044]) | -0.79 | |
| 20, random walk (bar 786) | -0.84 | **0.00** | -0.45 (vs fixed 1/18, p < 0.001 [0.005]) | -0.75 | |

4-sigma level shifts follow the same pattern (HL 5: all books 0.33-0.49; HL 20: fixed 0.00 vs calibrated
-0.30 to -0.34, calibrated loses 17-19 of 20 paired seeds). Full table: `run_nov08_experiments.py`.

Reading:

- **The calibrated gate stops starving healthy slow pairs, and that does not pay on this book.** At half-life
  10 it trades 79% of healthy blocks instead of 27% and keeps more of the healthy Sharpe (0.11 vs -0.01 for
  the fixed gate, 12 / 7, not significant), but on a dead half-life-10 pair it loses to the fixed gate on
  median (-0.30 vs -0.07), because it misses a quarter of the dead windows.
- **At half-life 20 the "premium" was nearly free, because the healthy pair loses money anyway.** Ungated, a
  healthy half-life-20 pair has a median Sharpe of -0.41 after costs: with z 2 / 0.5 and blocks forced flat
  every 63 bars, a slow spread rarely reverts inside a block. The fixed gate's 94% gating is then the right
  answer for the wrong reason, and every calibrated book loses to it (Holm p from < 0.001 to 0.20 depending on
  scenario). This is the most important result of the day: the slow-pair problem is the trading rule and
  block length, not only the test's power.
- **At half-life 5 the calibrated gate is roughly a wash with the fixed one.** The planted-reference gate
  keeps the healthy Sharpe better than the fixed gate (11 / 0, Holm 0.055) and the estimated one cuts a
  mid-block dead pair's loss more (0.33 vs 0.15, 10 / 2, p 0.039 uncorrected, Holm 1.0). Neither survives
  the multiplicity correction. The estimated reference (median 4.1 for planted 5) sets a level of 0.005,
  stricter than the fixed 0.01.
- **Ships off.** `gate_quantile` is not a clear out-of-sample win anywhere after the Holm correction and is a
  clear loss on slow pairs. The fixed `gate_p` (also off by default) is unchanged.

### What the tests check

- Level: rises with the half-life, reproducible, monotone in the quantile, equal to the EG p of the seeded
  pairs, gates 3-20% of fresh healthy windows at q0.90, never gates on an undefined reference, validation.
- Walk-forward: off is the legacy book; the fixed gate records its level; the reference is the first
  formation window's half-life and frozen; an explicit reference overrides it; gated blocks are flat and
  cost-free; a slow healthy pair is gated far less than by the fixed gate; causal under a late break;
  exclusive with `gate_p`.
- Calibration at reduced scale (6 seeds, HL 20): premium < 20% vs > 70%, power < 60% vs > 90%, estimated
  reference biased fast: `test_nov08_calibration.py`.
- The reading above at reduced scale (8 seeds, HL 10 and 20, aligned break): `test_nov08_claims.py`.

### Weaknesses

- Still synthetic, one pair at a time, two break shapes, 20 seeds per cell; after the Holm correction only
  the large half-life-20 differences are significant.
- A healthy half-life-20 pair loses money on this book (z 2 / 0.5, 63-bar blocks, forced flat). Before any
  gate can help a slow pair, the trading rule needs a longer block or no forced block-end exit, and that
  was not tried.
- The first-window half-life estimate is biased fast (about half the planted value at 20); a
  median-unbiased reference was not tried, and the planted-reference results suggest it would gate less
  and catch fewer breaks.
- The gate is still a 252-bar test; a longer look-back for the test only (more power on slow spreads, slower
  to see a break) was not tried.
- The Kalman hedge and the CUSUM halt were not re-run at slower half-lives.

## Nov 11 slice: carry positions across blocks instead of forcing them flat

Nov 8 found that a healthy half-life-20 pair loses money ungated (median Sharpe -0.41 after costs). The guess
was that the trading rule causes it, not the test: with z 2 / 0.5 and every 63-bar block forced flat, a slow
spread rarely reverts inside a block. This slice tests that guess directly with longer blocks and with no
forced block-end exit. Reproduce with `python scripts/run_nov11_experiments.py` (~1 min). The data is
synthetic, seeded and offline. Static OLS book, 252-bar formation, net of 10 bps per leg. Every traded bar is
out-of-sample, and every book trades the same 1008 bars of a 1260-bar path. The books, half-lives {5, 10, 20},
20 new seeds (1300-1319), the three scenarios and the decision rule were all fixed before the first run.
After the first run one diagnostic was corrected: the forced-exit share had counted a carried position at a
block end as a forced exit. No Sharpe number changed.

### Method

| Piece | What it does |
| --- | --- |
| `mean_reversion_positions(initial_position=, initial_held=)` | Starts the book in a position carried in from an earlier block. The position exits on this series' z, and the time stop counts the bars already held. Flat is the legacy book |
| `formation_trading_backtest(carry_over=)` | Drops the forced flat at every block end except the last. Bar 0 of the next block earns the carried position's return over the boundary under the old hedge, then pays the re-hedge cost (or the exit cost) of moving to the new block's hedge. Borrow accrues over the gap. The position exits on the new block's z (new formation fit). A gated block or a CUSUM halt closes the position. `windows["carried_in"]` records it |
| `run_nov11_experiments` | Compares `off` (forced flat, 63-bar blocks; every earlier slice), 126- and 252-bar blocks, `carry`, and the fixed EG gate 0.01 with and without carry. It reports the share of exits the block end forced, and runs paired sign tests vs `off` with a Holm adjustment over all 45 comparisons |

Decision rule (fixed beforehand): `carry_over` becomes the default only if it beats `off` on median Sharpe
in every half-life's no-break row, with Holm p < 0.05 in at least one, and loses no break row on median
Sharpe. Block length is a parameter and is reported but not shipped.

### Results (20 seeds, Sharpe medians net of costs; wins / losses vs `off` on paired seeds; Holm-adjusted sign-test p in brackets)

| Half-life, scenario | off | 126-bar | 252-bar | carry | EG gate 0.01 | carry + gate | Exits forced by block end (off / carry) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5, no break | 0.63 | 0.55 (13/7) | 0.70 (15/5) | **0.82** (20/0 [< 0.001]) | 0.55 (3/10) | 0.70 (17/3 [0.046]) | 18% / 1% |
| 5, random walk (756) | -0.16 | 0.06 | 0.13 | -0.02 (18/2 [0.011]) | 0.14 | **0.23** (18/2 [0.011]) | 39% / 4% |
| 5, random walk (786) | -0.07 | -0.09 | 0.07 | 0.05 (19/1 [0.001]) | 0.28 | **0.44** (17/3 [0.046]) | 38% / 4% |
| 10, no break | 0.14 | 0.26 | **0.31** | **0.31** (19/0 [< 0.001]) | 0.02 | 0.06 | 27% / 2% |
| 10, random walk (756) | -0.35 | -0.18 | -0.09 | -0.14 (19/1 [0.001]) | -0.00 | **0.00** | 45% / 5% |
| 10, random walk (786) | -0.39 | -0.29 | -0.09 | -0.09 (19/1 [0.001]) | **-0.00** | -0.01 | 46% / 5% |
| 20, no break | -0.53 | -0.29 | -0.08 | -0.18 (20/0 [< 0.001]) | **-0.00** | -0.01 | 46% / 3% |
| 20, random walk (756) | -0.80 | -0.45 | -0.38 | -0.46 (20/0 [< 0.001]) | **0.00** | **0.00** | 60% / 6% |
| 20, random walk (786) | -0.79 | -0.59 | -0.44 | -0.37 (20/0 [< 0.001]) | **0.00** | **0.00** | 59% / 6% |

Reading:

- **The Nov 8 guess was right: the forced exit costs money.** The block end forces 46% of a healthy
  half-life-20 pair's exits (18% at half-life 5). Carrying the position across blocks raises the median
  Sharpe in every one of the 9 rows, on 18-20 of 20 seeds each. Even the fast healthy pair gains (0.63 to
  0.82, 20 / 0), because carrying saves the forced exit and re-entry costs and lets reversions finish.
- **It narrows the slow-pair loss but does not close it.** A healthy half-life-20 pair still has a median
  Sharpe of -0.18 with carry (252-bar blocks: -0.08). z 2 / 0.5 on a spread this slow still pays costs for
  little reversion inside 4 years. The fixed gate, which keeps a slow pair flat 94% of the time, still has
  the best median at half-life 20 (about 0.00). It wins by not trading.
- **Carry and the gate do different jobs, and together they are the best book at half-life 5 under a
  break** (0.23 and 0.44 vs -0.16 and -0.07 for `off`). At half-lives 10-20 the gate dominates, and carry
  adds little on top of it.
- **Ships on.** The pre-registered rule passes: carry beats `off` in all three no-break rows (Holm p <
  0.001 in each) and in all six break rows. `carry_over=True` is now the default of
  `formation_trading_backtest`. Every earlier experiment script, the nested Kalman tuner and the
  legacy-property tests pass `carry_over=False` explicitly, so the numbers published in earlier sections
  still reproduce. Longer blocks also help (252-bar is the best ungated book at half-lives 10 and 20). They
  stay a parameter, because the rule only covered carry.

### What the tests check

- Signals: flat start is the legacy book. A carried long or short is held until z reverts and flips on a
  cross. The time stop counts bars held before the block. A NaN z still forces flat. Validation.
- Walk-forward: carry is the default and `False` is the legacy book (flat at every block end). A slow pair
  carries positions and the final block ends flat. The boundary bar earns exactly the carried return minus
  the re-hedge or exit cost. Costs strictly reduce the carried book. A gated block closes a carried position.
  Carry is causal under a late break. The Kalman hedge also carries.
- The reading above at reduced scale (8 seeds, HL 5 and 20): `test_nov11_claims.py`.

### Weaknesses

- Still synthetic: one pair at a time, one break shape, 20 seeds per cell, and one cost level (10 bps per
  leg). Carry's gain is partly a cost saving, so it would shrink with cheaper execution and grow with
  dearer execution. That was not swept.
- A carried position crosses a re-fit. The new block's hedge and z can disagree with the old ones (a
  long carried into a block whose z says short flips at once and pays both legs). How often that happens
  was not measured.
- A healthy half-life-20 pair still loses after costs. The thresholds (z 2 / 0.5) were never re-chosen
  for slow spreads, and a formation-chosen exit or entry was not tried with carry.
- The nested Kalman tuner, the CUSUM halt and the time stop were not re-run with carry. The tuner is pinned
  to the legacy book.
- Longer blocks were reported but not shipped. Combining 252-bar blocks with carry was not tried.

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
src/statarb/     cointegration, signals, backtest, data, synth, ou, kalman, screening, formation, rescreen, nested, rank, breakwatch, gatecal
scripts/         run_oct05_experiments.py, run_oct09_experiments.py, run_oct14_experiments.py, run_oct20_experiments.py, run_oct29_experiments.py, run_nov03_experiments.py, run_nov03_calibration.py, run_nov08_experiments.py, run_nov08_calibration.py
tests/           synthetic-data unit tests
notebooks/       walk-forward KO/PEP (or sample) pipeline
data/            make_sample.py + sample_ko_pep.csv (sample, not live)
```

## What is still out of scope

Johansen-traded eigenvectors, execution at the open, borrow recall / locate, taxes, a bundled public tape, and any claim that a backtest Sharpe survives contact with a broker. Ranking/capping of re-screened discoveries and an OU time-stop on formation/re-screen books landed in the Oct 14 slice; joint `q_alpha` tuning, a flat borrow fee and regularised nested selection landed on Oct 20; planted cointegration breaks and an adverse-z stop-loss landed on Oct 29; a CUSUM halt and an Engle-Granger formation gate landed on Nov 3; a half-life-calibrated gate level (`gate_quantile`, off) landed on Nov 8.
