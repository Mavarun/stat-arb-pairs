import numpy as np
import pandas as pd
import pytest

from statarb.formation import _borrow_cost, formation_trading_backtest
from statarb.synth import make_ou_pair


def test_borrow_hand_computed_short_leg_follows_the_sign():
    y = np.array([10.0, 11.0, 10.5, 10.0])
    x = np.array([5.0, 5.0, 5.5, 5.0])
    beta = np.array([2.0, 2.0, 2.0, 2.0])
    pos = np.array([1.0, -1.0, 0.0, 0.0])
    bw = _borrow_cost(y, x, pos, beta, borrow_bps=252.0)  # 1 bp a day
    daily = 252.0 / 1e4 / 252
    # bar1 pays for the long spread held over (0, 1]: short x leg = 2*5 of 10+10
    assert np.isclose(bw[1], daily * 10.0 / 20.0)
    # bar2 pays for the short spread held over (1, 2]: short y leg = 11 of 11+10
    assert np.isclose(bw[2], daily * 11.0 / 21.0)
    # flat books and the first bar pay nothing
    assert bw[0] == 0.0 and bw[3] == 0.0


def test_zero_borrow_reproduces_the_legacy_book_exactly():
    p = make_ou_pair(252 + 63 * 5, half_life=5, seed=11)
    for hedge in ("ols", "kalman"):
        legacy = formation_trading_backtest(p.y, p.x, hedge=hedge)
        zero = formation_trading_backtest(p.y, p.x, hedge=hedge, borrow_bps=0.0)
        pd.testing.assert_series_equal(legacy.returns, zero.returns)
        assert zero.borrow_cost == 0.0


def test_borrow_is_monotone_and_only_changes_returns_while_holding():
    p = make_ou_pair(252 + 63 * 6, half_life=5, seed=12)
    free = formation_trading_backtest(p.y, p.x)
    cheap = formation_trading_backtest(p.y, p.x, borrow_bps=50.0)
    dear = formation_trading_backtest(p.y, p.x, borrow_bps=500.0)
    assert 0.0 < cheap.borrow_cost < dear.borrow_cost
    assert free.total_return > cheap.total_return > dear.total_return
    # same positions: borrow never changes the trading decisions
    pd.testing.assert_series_equal(free.positions, dear.positions)
    held = free.positions.shift(1).fillna(0.0) != 0.0
    diff = (free.returns - dear.returns).abs()
    assert (diff[~held.reindex(diff.index, fill_value=False)] == 0.0).all()
    # borrow is included in cost_paid
    assert np.isclose(dear.cost_paid - free.cost_paid, dear.borrow_cost)


def test_negative_borrow_rejected():
    p = make_ou_pair(252 + 63 * 2, seed=13)
    with pytest.raises(ValueError):
        formation_trading_backtest(p.y, p.x, borrow_bps=-1.0)


def test_rescreen_borrow_zero_matches_legacy_and_is_monotone():
    from statarb.rescreen import universe_formation_backtest
    from statarb.synth import make_planted_universe

    frame, _ = make_planted_universe(4, 2, 252 + 63 * 3, half_life=5.0, seed=21)
    kw = dict(rule="naive", hedge="ols", formation=252, trading=63)
    legacy = universe_formation_backtest(frame, **kw)
    zero = universe_formation_backtest(frame, borrow_bps=0.0, **kw)
    dear = universe_formation_backtest(frame, borrow_bps=300.0, **kw)
    pd.testing.assert_series_equal(legacy.returns, zero.returns)
    assert legacy.trades > 0
    assert zero.borrow_cost == 0.0 < dear.borrow_cost
    assert dear.total_return < zero.total_return
    assert dear.trades == zero.trades
    assert np.isclose(dear.cost_paid - zero.cost_paid, dear.borrow_cost)
    with pytest.raises(ValueError):
        universe_formation_backtest(frame, borrow_bps=-5.0, **kw)
