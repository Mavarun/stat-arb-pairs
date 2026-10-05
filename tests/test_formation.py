import numpy as np
import pandas as pd
import pytest

from statarb.formation import _block_pnl, compare_hedges, formation_trading_backtest
from statarb.synth import make_ou_pair


def test_window_count_and_no_trading_inside_first_formation():
    p = make_ou_pair(252 + 63 * 4 + 10, seed=3)
    res = formation_trading_backtest(p.y, p.x, hedge="ols")
    assert res.n_windows == 4
    # nothing is traded before the first formation window ends
    assert res.returns.index[0] == p.y.index[252]
    assert len(res.returns) == 63 * 4


def test_positions_forced_flat_at_every_block_end():
    p = make_ou_pair(252 + 63 * 5, seed=4)
    for hedge in ("ols", "kalman"):
        res = formation_trading_backtest(p.y, p.x, hedge=hedge)
        ends = res.windows["trade_end"]
        assert (res.positions.loc[ends] == 0).all()


def test_backtest_is_causal_future_break_does_not_change_earlier_blocks():
    p = make_ou_pair(252 + 63 * 4, seed=5)
    y2 = p.y.copy()
    y2.iloc[252 + 63 * 3 :] += 40.0  # break inside the last trading block only
    for hedge in ("ols", "kalman"):
        a = formation_trading_backtest(p.y, p.x, hedge=hedge).returns
        b = formation_trading_backtest(y2, p.x, hedge=hedge).returns
        cut = p.y.index[252 + 63 * 3 - 1]
        pd.testing.assert_series_equal(a.loc[:cut], b.loc[:cut])


def test_costs_monotone_and_zero_cost_matches_gross():
    p = make_ou_pair(252 + 63 * 6, half_life=5, seed=6)
    free = formation_trading_backtest(p.y, p.x, spread_bps=0, slippage_bps=0)
    paid = formation_trading_backtest(p.y, p.x, spread_bps=5, slippage_bps=5)
    dear = formation_trading_backtest(p.y, p.x, spread_bps=20, slippage_bps=20)
    assert free.cost_paid == 0.0
    assert free.trades > 0
    assert np.isclose(free.returns.sum(), free.gross_return_sum)
    assert free.total_return > paid.total_return > dear.total_return


def test_static_hedge_never_pays_rehedge_cost_kalman_does():
    p = make_ou_pair(252 + 63 * 6, half_life=5, seed=7)
    ols = formation_trading_backtest(p.y, p.x, hedge="ols")
    kal = formation_trading_backtest(p.y, p.x, hedge="kalman")
    assert ols.rehedge_cost == 0.0
    assert kal.rehedge_cost > 0.0


def test_block_pnl_hand_computed():
    y = np.array([10.0, 11.0, 10.5])
    x = np.array([5.0, 5.0, 5.5])
    beta = np.array([2.0, 2.0, 2.0])
    pos = np.array([1.0, 1.0, 0.0])
    g, c, rh = _block_pnl(y, x, pos, beta, rate=0.001, n_legs=2)
    # bar1: long spread held from bar0: (1 - 2*0) / (10 + 10)
    assert np.isclose(g[1], 1.0 / 20.0)
    # bar2: (-0.5 - 2*0.5) / (11 + 10)
    assert np.isclose(g[2], -1.5 / 21.0)
    assert np.allclose(c, [0.002, 0.0, 0.002])
    assert np.allclose(rh, 0.0)


def _sharpe_wins(beta_path_fn, seeds):
    n = 252 + 63 * 12
    wins = 0
    for s in seeds:
        kw = {} if beta_path_fn is None else {"beta_path": beta_path_fn(n)}
        p = make_ou_pair(n, half_life=5, sigma_spread=1.0, seed=s, **kw)
        t = compare_hedges(p.y, p.x)
        wins += int(t.loc["kalman", "sharpe"] > t.loc["ols", "sharpe"])
    return wins


def test_kalman_beats_static_hedge_on_drifting_beta_net_of_costs():
    # beta drifts 1.0 -> 2.5: a frozen formation beta goes stale inside each block
    wins = _sharpe_wins(lambda n: np.linspace(1.0, 2.5, n), range(200, 210))
    assert wins >= 7


def test_static_hedge_is_not_worse_when_beta_is_truly_constant():
    # the Kalman filter's extra flexibility is pure noise + re-hedge cost here
    wins = _sharpe_wins(None, range(300, 310))
    assert wins <= 3


def test_input_validation():
    p = make_ou_pair(300, seed=9)
    with pytest.raises(ValueError):
        formation_trading_backtest(p.y, p.x)  # shorter than F + T
    with pytest.raises(ValueError):
        formation_trading_backtest(p.y, p.x, formation=10)
    with pytest.raises(ValueError):
        formation_trading_backtest(p.y, p.x, hedge="lasso", formation=100, trading=50)
