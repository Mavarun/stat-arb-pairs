"""The Nov 11 README reading, re-checked at reduced scale (first 8 of the 20 seeds)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "nov11", Path(__file__).resolve().parents[1] / "scripts" / "run_nov11_experiments.py"
)
nov11 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nov11)

BOOKS = {k: nov11.BOOKS[k] for k in ("off", "252-bar blocks", "carry", "EG gate 0.01")}


@pytest.fixture(scope="module")
def summary():
    df = nov11.run(seeds=range(1300, 1308), half_lives=(5.0, 20.0), scenarios=nov11.SCENARIOS[:2], books=BOOKS)
    return nov11.summarise(df).set_index(["half_life", "scenario", "book"])


def test_forced_exits_are_most_of_the_slow_pair_exits(summary) -> None:
    off20 = summary.loc[(20.0, "no break", "off"), "mean_forced_exit_share"]
    off5 = summary.loc[(5.0, "no break", "off"), "mean_forced_exit_share"]
    assert off20 > 0.35 > off5
    assert summary.loc[(20.0, "no break", "carry"), "mean_forced_exit_share"] < 0.1


def test_carrying_beats_forced_flat_on_healthy_pairs_at_both_speeds(summary) -> None:
    for hl in (5.0, 20.0):
        nb = summary.loc[(hl, "no break")]
        assert nb.loc["carry", "median_sharpe"] > nb.loc["off", "median_sharpe"]
        assert nb.loc["carry", "wins"] >= 6


def test_a_healthy_hl20_pair_still_loses_after_costs_with_carry(summary) -> None:
    assert summary.loc[(20.0, "no break", "carry"), "median_sharpe"] < 0.0


def test_the_fixed_gate_still_wins_on_a_dead_slow_pair(summary) -> None:
    rw = summary.loc[(20.0, "random walk (756)")]
    assert rw.loc["EG gate 0.01", "median_sharpe"] > rw.loc["carry", "median_sharpe"] > rw.loc["off", "median_sharpe"]


def test_every_book_trades_the_same_out_of_sample_bars(summary) -> None:
    assert set(summary["n_bars"]) == {1008}
