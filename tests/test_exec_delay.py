"""exec_delay shifts the executed book inside each block and is a no-op at 0."""

import numpy as np
import pytest

from statarb.formation import formation_trading_backtest
from statarb.synth import make_beta_trend_pair


def _pair(seed=3):
    return make_beta_trend_pair(1260, trend_from=252, drift_per_year=0.0, half_life=10.0, seed=seed)


def test_zero_delay_is_the_default_book():
    p = _pair()
    a, b = formation_trading_backtest(p.y, p.x), formation_trading_backtest(p.y, p.x, exec_delay=0)
    assert a.sharpe == b.sharpe and a.trades == b.trades
    assert np.array_equal(a.positions.to_numpy(), b.positions.to_numpy())


def test_one_bar_delay_shifts_positions_inside_blocks():
    p = _pair()
    a, b = formation_trading_backtest(p.y, p.x), formation_trading_backtest(p.y, p.x, exec_delay=1)
    pa, pb = a.positions.to_numpy(), b.positions.to_numpy()
    for k in range(a.n_windows - 1):  # all but the last block (forced flat on its last bar)
        blk = slice(k * 63, (k + 1) * 63)
        assert np.array_equal(pb[blk][1:], pa[blk][:-1])
        prev_end = pb[k * 63 - 1] if k else 0.0
        assert pb[blk][0] == prev_end


def test_delay_is_causal():
    p = _pair()
    base = formation_trading_backtest(p.y, p.x, exec_delay=1)
    y2 = p.y.copy()
    y2.iloc[-50:] *= 1.2
    late = formation_trading_backtest(y2, p.x, exec_delay=1)
    cut = len(base.positions) - 50
    assert np.array_equal(base.positions.to_numpy()[:cut], late.positions.to_numpy()[:cut])


@pytest.mark.parametrize("bad", [-1, 1.5])
def test_rejects_bad_delay(bad):
    p = _pair()
    with pytest.raises(ValueError):
        formation_trading_backtest(p.y, p.x, exec_delay=bad)
