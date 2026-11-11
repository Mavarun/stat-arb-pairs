"""carry_over: no forced flat at block ends; positions carried into the next block."""

from __future__ import annotations

import numpy as np

from statarb.formation import formation_trading_backtest
from statarb.synth import make_broken_pair, make_ou_pair


def _slow(seed=11):
    return make_ou_pair(1260, half_life=20.0, seed=seed)


def test_carry_is_the_default_since_nov11_and_false_is_the_legacy_book() -> None:
    p = _slow()
    a = formation_trading_backtest(p.y, p.x)
    b = formation_trading_backtest(p.y, p.x, carry_over=True)
    assert a.returns.equals(b.returns)
    legacy = formation_trading_backtest(p.y, p.x, carry_over=False)
    assert (legacy.windows["carried_in"] == 0.0).all()
    ends = legacy.positions.loc[legacy.windows["trade_end"]]
    assert (ends == 0.0).all()


def test_slow_pair_carries_positions_and_the_last_block_ends_flat() -> None:
    p = _slow()
    r = formation_trading_backtest(p.y, p.x, carry_over=True)
    assert (r.windows["carried_in"] != 0.0).sum() >= 2
    assert r.positions.iloc[-1] == 0.0
    ends = r.positions.loc[r.windows["trade_end"].iloc[:-1]]
    assert (ends != 0.0).any()


def test_boundary_bar_earns_the_carried_return_net_of_costs() -> None:
    p = _slow()
    rate = 10.0 / 1e4
    r = formation_trading_backtest(p.y, p.x, carry_over=True, spread_bps=5.0, slippage_bps=5.0)
    w = r.windows
    checked = 0
    for k in range(1, len(w)):
        c_pos = w["carried_in"].iloc[k]
        if c_pos == 0.0:
            continue
        t0, tp = w["trade_start"].iloc[k], w["trade_end"].iloc[k - 1]
        b_prev, b0 = r.beta.loc[tp], r.beta.loc[t0]
        y0, x0, yp, xp = p.y.loc[t0], p.x.loc[t0], p.y.loc[tp], p.x.loc[tp]
        assert r.positions.loc[tp] == c_pos
        gross = c_pos * ((y0 - yp) - b_prev * (x0 - xp)) / (abs(yp) + abs(b_prev) * abs(xp))
        pos0 = r.positions.loc[t0]
        if pos0 == c_pos:
            cost = rate * abs(pos0) * abs(b0 - b_prev) * abs(x0) / (abs(y0) + abs(b0) * abs(x0))
        else:
            cost = rate * 2 * abs(pos0 - c_pos)
        assert np.isclose(r.returns.loc[t0], gross - cost, atol=1e-12)
        checked += 1
    assert checked >= 2


def test_costs_strictly_reduce_the_carried_book() -> None:
    p = _slow()
    free = formation_trading_backtest(p.y, p.x, carry_over=True, spread_bps=0.0, slippage_bps=0.0)
    paid = formation_trading_backtest(p.y, p.x, carry_over=True)
    assert paid.sharpe < free.sharpe and paid.cost_paid > 0.0


def test_on_a_slow_pair_carrying_pays_less_trade_cost_than_forced_exits() -> None:
    p = _slow()
    forced = formation_trading_backtest(p.y, p.x)
    carried = formation_trading_backtest(p.y, p.x, carry_over=True)
    assert carried.cost_paid - carried.rehedge_cost <= forced.cost_paid + 1e-12


def test_a_gated_block_closes_a_carried_position_on_its_first_bar() -> None:
    p = make_broken_pair(1260, break_at=756, kind="random_walk", half_life=20.0, seed=5)
    r = formation_trading_backtest(p.y, p.x, carry_over=True, gate_p=0.01)
    w = r.windows
    for k in range(len(w)):
        if w["gated"].iloc[k]:
            blk = r.positions.loc[w["trade_start"].iloc[k]:w["trade_end"].iloc[k]]
            assert (blk == 0.0).all()


def test_carry_is_causal_under_a_late_break() -> None:
    clean = make_broken_pair(1260, break_at=1100, kind="none", half_life=20.0, seed=4)
    late = make_broken_pair(1260, break_at=1100, kind="random_walk", half_life=20.0, seed=4)
    a = formation_trading_backtest(clean.y, clean.x, carry_over=True)
    b = formation_trading_backtest(late.y, late.x, carry_over=True)
    cut = clean.y.index[1099]
    assert np.allclose(a.returns.loc[:cut].to_numpy(), b.returns.loc[:cut].to_numpy())


def test_kalman_hedge_also_carries() -> None:
    p = _slow(seed=12)
    r = formation_trading_backtest(p.y, p.x, hedge="kalman", carry_over=True)
    assert r.positions.iloc[-1] == 0.0 and np.isfinite(r.sharpe)
