import numpy as np
import pandas as pd

from statarb.formation import (
    AUTO_TIME_STOP_MULTIPLES,
    choose_time_stop_multiple,
    formation_trading_backtest,
)
from statarb.synth import make_ou_pair


def test_choice_is_a_candidate_and_uses_formation_bars_only():
    p = make_ou_pair(252, half_life=5, seed=61)
    yf, xf = p.y.to_numpy(), p.x.to_numpy()
    k, hold = choose_time_stop_multiple(yf, xf)
    assert k in AUTO_TIME_STOP_MULTIPLES
    assert (hold is None) == (not np.isfinite(k))
    # deterministic on the same bars
    assert choose_time_stop_multiple(yf.copy(), xf.copy()) == (k, hold)


def test_single_candidate_grid_returns_it():
    p = make_ou_pair(252, half_life=5, seed=62)
    k, hold = choose_time_stop_multiple(p.y.to_numpy(), p.x.to_numpy(), multiples=(3.0,))
    assert k == 3.0 and hold is not None and hold >= 1


def test_auto_records_the_choice_per_window_and_stays_causal():
    p = make_ou_pair(252 + 63 * 3, half_life=5, seed=63)
    res = formation_trading_backtest(p.y, p.x, use_time_stop=True, time_stop_multiple="auto")
    ks = res.windows["time_stop_multiple"]
    assert len(ks) == 3 and set(ks) <= set(AUTO_TIME_STOP_MULTIPLES)
    y2 = p.y.copy()
    y2.iloc[252 + 63 * 2 :] += 40.0
    res2 = formation_trading_backtest(y2, p.x, use_time_stop=True, time_stop_multiple="auto")
    cut = p.y.index[252 + 63 * 2 - 1]
    pd.testing.assert_series_equal(res.returns.loc[:cut], res2.returns.loc[:cut])
    assert list(res2.windows["time_stop_multiple"])[:2] == list(ks)[:2]


def test_fixed_multiple_path_is_unchanged_and_off_records_nan():
    p = make_ou_pair(252 + 63 * 2, half_life=5, seed=64)
    on = formation_trading_backtest(p.y, p.x, use_time_stop=True, time_stop_multiple=3.0)
    assert (on.windows["time_stop_multiple"] == 3.0).all()
    off = formation_trading_backtest(p.y, p.x)
    assert off.windows["time_stop_multiple"].isna().all()
