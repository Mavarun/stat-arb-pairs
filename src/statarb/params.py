"""Default research parameters.

These are conventional textbook starting points, **not optimized**.
Changing them after seeing results is a form of overfitting.
"""

# Mean-reversion z-score thresholds (standard deviation units of the spread).
Z_ENTER = 2.0
Z_EXIT = 0.5

# Rolling window (trading days) for z-score mean/std when not freezing train stats.
Z_WINDOW = 60

# Transaction costs, applied to each of the two legs on |\u0394position|.
# Round-trip of a 1-unit spread position \u2248 4 * (SPREAD_BPS + SLIPPAGE_BPS) bps of gross notional.
SPREAD_BPS = 5.0
SLIPPAGE_BPS = 5.0
N_LEGS = 2

# Walk-forward: estimate hedge ratio / z stats on this fraction, trade the remainder.
TRAIN_FRAC = 0.70
MIN_TRAIN_OBS = 252

# Daily-to-annual Sharpe scale. Assumes ~252 equity sessions; not calendar days.
ANNUALIZATION = 252
