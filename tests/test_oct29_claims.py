"""The Oct 29 README reading, re-checked at reduced scale (4 of the 12 seeds)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "oct29", Path(__file__).resolve().parents[1] / "scripts" / "run_oct29_experiments.py"
)
oct29 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oct29)


@pytest.fixture(scope="module")
def summary():
    oct29.SEEDS = range(900, 904)
    return oct29.summarise(oct29.run()).set_index(["scenario", "book"])


def test_z_stop_4_cuts_the_post_break_loss_on_a_dead_pair(summary) -> None:
    rw = summary.loc["random-walk break"]
    assert rw.loc["z stop 4", "median_post_break_return"] > rw.loc["off", "median_post_break_return"]
    assert rw.loc["z stop 4", "median_worst_window"] > rw.loc["off", "median_worst_window"]
    assert rw.loc["off", "median_post_break_return"] < 0  # nothing makes the broken pair pay


def test_stops_cost_sharpe_when_nothing_breaks(summary) -> None:
    nb = summary.loc["no break"]
    for book in ("z stop 3", "z stop 4", "time stop 3xHL"):
        assert nb.loc[book, "median_sharpe"] < nb.loc["off", "median_sharpe"]
    assert nb.loc["z stop 3", "median_sharpe"] < nb.loc["z stop 4", "median_sharpe"]


def test_level_shift_is_absorbed_and_auto_matches_off(summary) -> None:
    ls = summary.loc["4-sigma level shift"]
    assert ls.loc["off", "median_sharpe"] > 0
    assert ls.loc["z stop 4", "median_sharpe"] < ls.loc["off", "median_sharpe"]
    nb = summary.loc["no break"]
    assert nb.loc["time stop auto", "median_sharpe"] == pytest.approx(nb.loc["off", "median_sharpe"])
