import numpy as np
import pandas as pd
import pytest

from statarb.signals import mean_reversion_positions


def test_default_is_the_legacy_book():
    z = pd.Series([0.0, 2.5, 1.0, 0.2, -2.4, -0.3])
    a = mean_reversion_positions(z)
    b = mean_reversion_positions(z, initial_position=0.0, initial_held=0)
    pd.testing.assert_series_equal(a, b)


def test_carried_long_is_held_until_z_reverts():
    z = pd.Series([-1.5, -1.0, -0.8, -0.2, 0.0])
    pos = mean_reversion_positions(z, z_enter=2.0, z_exit=0.5, initial_position=1.0, initial_held=5)
    assert pos.tolist() == [1.0, 1.0, 1.0, 0.0, 0.0]


def test_carried_short_flips_when_z_crosses_to_the_other_side():
    z = pd.Series([1.0, -2.5])
    pos = mean_reversion_positions(z, z_enter=2.0, z_exit=0.5, initial_position=-1.0, initial_held=1)
    assert pos.tolist() == [-1.0, 1.0]


def test_time_stop_counts_bars_held_before_the_block():
    z = pd.Series([-1.5, -1.5, -1.5])
    pos = mean_reversion_positions(z, z_enter=2.0, z_exit=0.5, max_hold=4, initial_position=1.0, initial_held=3)
    assert pos.tolist() == [1.0, 0.0, 0.0]


def test_nan_z_still_forces_flat():
    pos = mean_reversion_positions(pd.Series([np.nan, -1.0]), initial_position=1.0, initial_held=2)
    assert pos.tolist() == [0.0, 0.0]


def test_validation():
    with pytest.raises(ValueError):
        mean_reversion_positions(pd.Series([0.0]), initial_position=0.5)
    with pytest.raises(ValueError):
        mean_reversion_positions(pd.Series([0.0]), initial_position=0.0, initial_held=3)
    with pytest.raises(ValueError):
        mean_reversion_positions(pd.Series([0.0]), initial_position=1.0, initial_held=-1)
