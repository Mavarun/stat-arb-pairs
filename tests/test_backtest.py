import numpy as np
import pandas as pd

from statarb.backtest import (
    apply_costs,
    backtest_spread,
    expanding_walk_forward,
    walk_forward,
)
from statarb.cointegration import engle_granger
from statarb.signals import compute_spread, mean_reversion_positions, zscore_from_stats
from tests._synth import make_cointegrated_pair


def _strong_mr_pair(n=1200, seed=11):
    """Tighter residual so the mean-reversion book actually trades."""
    return make_cointegrated_pair(
        n=n, beta=1.25, intercept=5.0, rho=0.6, sigma_x=0.8, sigma_e=0.7, seed=seed
    )


def test_costs_strictly_reduce_sharpe_on_same_path():
    y, x, beta, intercept = _strong_mr_pair()
    spread = compute_spread(y, x, beta, intercept)
    z = zscore_from_stats(spread, float(spread.mean()), float(spread.std(ddof=1)))
    pos = mean_reversion_positions(z, z_enter=1.5, z_exit=0.25)
    assert int((pos != 0).sum()) > 20

    uncosted = backtest_spread(
        y, x, hedge_ratio=beta, intercept=intercept, zscore=z, positions=pos,
        spread_bps=0.0, slippage_bps=0.0,
    )
    costed = backtest_spread(
        y, x, hedge_ratio=beta, intercept=intercept, zscore=z, positions=pos,
        spread_bps=5.0, slippage_bps=5.0,
    )
    assert costed.turnover > 0
    assert costed.trade_count > 0
    assert costed.total_return < uncosted.total_return
    assert costed.sharpe < uncosted.sharpe
    # Costs are a non-negative drag; they cannot raise every daily return.
    assert (costed.returns <= uncosted.returns + 1e-15).all()
    assert (costed.returns < uncosted.returns).any()


def test_apply_costs_zero_when_flat():
    pos = pd.Series([0.0, 0.0, 0.0, 0.0])
    cost = apply_costs(pos, spread_bps=5.0, slippage_bps=5.0)
    assert (cost == 0.0).all()


def test_apply_costs_two_legs_on_unit_entry():
    # position set at t=1 (index 1). Held starting next bar (index 2) with |\u0394held|=1.
    pos = pd.Series([0.0, 1.0, 1.0, 1.0])
    cost = apply_costs(pos, spread_bps=5.0, slippage_bps=5.0, n_legs=2)
    # rate = 2 * 10 / 1e4 = 0.002
    assert abs(float(cost.iloc[2]) - 0.002) < 1e-12
    assert float(cost.iloc[0]) == 0.0
    assert float(cost.iloc[3]) == 0.0  # no additional trade


def test_max_drawdown_non_positive():
    y, x, beta, intercept = _strong_mr_pair(n=400, seed=2)
    res = backtest_spread(y, x, hedge_ratio=beta, intercept=intercept, spread_bps=0, slippage_bps=0)
    assert res.max_drawdown <= 0.0


def test_walk_forward_hedge_ratio_is_train_only():
    y, x, _, _ = _strong_mr_pair(n=900, seed=21)
    cut = y.index[600]
    wf = walk_forward(y, x, train_end=cut, min_train=100, z_enter=1.5, z_exit=0.3)
    train_eg = engle_granger(y.loc[:cut], x.loc[:cut])
    assert abs(wf.engle_granger.hedge_ratio - train_eg.hedge_ratio) < 1e-12
    assert abs(wf.engle_granger.intercept - train_eg.intercept) < 1e-12
    # Test window must not be empty and must start after train_end.
    assert wf.test_costed.n_obs > 50
    assert wf.test_costed.returns.index.min() > cut
    # Train EG p-value is attached to both test results.
    assert wf.test_costed.coint_pvalue == train_eg.pvalue


def test_walk_forward_costed_worse_than_uncosted_when_trades_exist():
    y, x, _, _ = _strong_mr_pair(n=1000, seed=4)
    wf = walk_forward(
        y, x, train_frac=0.7, min_train=200, z_enter=1.5, z_exit=0.3,
        spread_bps=8.0, slippage_bps=8.0, z_mode="train_stats",
    )
    # If the test book never traded, the cost channel is untested — fail.
    assert wf.test_uncosted.trade_count + wf.train_uncosted.trade_count > 0
    if wf.test_uncosted.turnover > 0:
        assert wf.test_costed.sharpe < wf.test_uncosted.sharpe
        assert wf.test_costed.total_return < wf.test_uncosted.total_return
    if wf.train_uncosted.turnover > 0:
        assert wf.train_costed.sharpe < wf.train_uncosted.sharpe


def test_walk_forward_metrics_table_has_four_rows():
    y, x, _, _ = make_cointegrated_pair(n=400, seed=5)
    wf = walk_forward(y, x, train_frac=0.6, min_train=50)
    table = wf.metrics_table()
    assert list(table.index) == [
        "train_uncosted",
        "train_costed",
        "test_uncosted",
        "test_costed",
    ]
    assert "sharpe" in table.columns


def test_expanding_walk_forward_uses_later_test_windows():
    y, x, _, _ = make_cointegrated_pair(n=500, seed=8)
    folds = expanding_walk_forward(y, x, n_splits=3, min_train=50, z_enter=1.8)
    assert len(folds) == 3
    ends = [f.train_end for f in folds]
    assert ends == sorted(ends)
    assert ends[0] < ends[-1]
    # Each fold's test starts after its own train_end.
    for f in folds:
        assert f.test_uncosted.returns.index.min() > f.train_end


def test_position_is_lagged_so_same_bar_z_does_not_earn_same_bar_move():
    """If we did not lag, a constructed one-bar spike would be captured same-close."""
    idx = pd.bdate_range("2020-01-01", periods=6)
    x = pd.Series([10.0, 10.0, 10.0, 10.0, 10.0, 10.0], index=idx)
    # y jumps on bar 3 then reverts on bar 4. z uses frozen stats.
    y = pd.Series([10.0, 10.0, 10.0, 16.0, 10.0, 10.0], index=idx)
    # hedge 0, intercept 10 → spread = y-10 = [0,0,0,6,0,0]
    z = pd.Series([0.0, 0.0, 0.0, 5.0, 0.0, 0.0], index=idx)
    pos = mean_reversion_positions(z, z_enter=2.0, z_exit=0.5)
    # Short the rich spread on the jump bar.
    assert pos.iloc[3] == -1.0
    res = backtest_spread(
        y, x, hedge_ratio=0.0, intercept=10.0, zscore=z, positions=pos,
        spread_bps=0.0, slippage_bps=0.0,
    )
    # Same-bar jump must NOT be captured: position starts earning the next bar (the revert).
    # hp_ret on bar 3 is the jump; held pos is still 0.
    assert abs(float(res.returns.iloc[3])) < 1e-12
    # Next bar reverts; short spread earns the drop.
    assert float(res.returns.iloc[4]) > 0.0
