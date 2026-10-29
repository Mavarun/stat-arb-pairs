"""Adverse-z stop-loss in mean_reversion_positions: hand paths, blocking, legacy identity."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statarb.signals import mean_reversion_positions


def _pos(z, **kw):
    return mean_reversion_positions(pd.Series(z, dtype=float), z_enter=2.0, z_exit=0.5, **kw).tolist()


def test_long_spread_is_stopped_when_z_keeps_falling_and_side_is_blocked() -> None:
    z = [0.0, -2.5, -3.0, -4.2, -3.5, -2.5, -1.5, -2.5, 0.0]
    #          enter  hold  stop  blocked (|z|>=2)  unblock  re-enter  exit
    assert _pos(z, z_stop=4.0) == [0, 1, 1, 0, 0, 0, 0, 1, 0]
    assert _pos(z) == [0, 1, 1, 1, 1, 1, 1, 1, 0]


def test_short_spread_stop_is_symmetric() -> None:
    z = [0.0, 2.5, 4.5, 3.0, 1.0, 0.2]
    assert _pos(z, z_stop=4.0) == [0, -1, 0, 0, 0, 0]
    # the opposite side stays tradable after a stop
    z2 = [0.0, 2.5, 4.5, -2.5, -1.0, 0.0]
    assert _pos(z2, z_stop=4.0) == [0, -1, 0, 1, 1, 0]


def test_favourable_moves_never_trigger_the_stop() -> None:
    z = [0.0, -2.5, 3.0, 5.0, 0.0]
    # long at -2.5, z jumps to +3: exit and flip short (legacy rule); +5 is adverse for the short
    assert _pos(z, z_stop=4.0) == [0, 1, -1, 0, 0]
    assert _pos(z) == [0, 1, -1, -1, 0]


def test_off_is_legacy_and_validation() -> None:
    rng = np.random.default_rng(0)
    z = pd.Series(np.cumsum(rng.normal(0, 0.6, 500)))
    a = mean_reversion_positions(z, max_hold=10)
    b = mean_reversion_positions(z, max_hold=10, z_stop=None)
    assert a.equals(b)
    assert (mean_reversion_positions(z, z_stop=1e9) == mean_reversion_positions(z)).all()
    with pytest.raises(ValueError):
        mean_reversion_positions(z, z_enter=2.0, z_stop=2.0)


def test_stop_and_time_stop_compose() -> None:
    z = [0.0, -2.5, -2.6, -2.7, -4.5, -1.0]
    # time stop after 2 bars fires first; z_stop would have fired on bar 4
    assert _pos(z, z_stop=4.0, max_hold=2) == [0, 1, 1, 0, 0, 0]
