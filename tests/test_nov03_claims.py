"""The Nov 3 README reading, re-checked at reduced scale (first 8 of the 20 seeds, aligned break)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "nov03", Path(__file__).resolve().parents[1] / "scripts" / "run_nov03_experiments.py"
)
nov03 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nov03)

BOOKS = {k: nov03.BOOKS[k] for k in ("off", "CUSUM halt", "EG gate 0.01")}


@pytest.fixture(scope="module")
def frame():
    return nov03.run(seeds=range(940, 948), breaks=nov03.BREAKS[:1], books=BOOKS)


@pytest.fixture(scope="module")
def summary(frame):
    return nov03.summarise(frame).set_index(["scenario", "book"])


def test_gate_cuts_the_loss_on_a_dead_pair(summary) -> None:
    rw = summary.loc["random-walk break"]
    assert rw.loc["EG gate 0.01", "median_post_break_return"] > rw.loc["off", "median_post_break_return"]
    assert rw.loc["EG gate 0.01", "median_max_dd"] > rw.loc["off", "median_max_dd"]
    assert rw.loc["EG gate 0.01", "sharpe_wins_vs_off"] > rw.loc["EG gate 0.01", "sharpe_losses_vs_off"]


def test_cusum_detects_a_dead_pair_quickly_but_pays_for_false_halts(summary) -> None:
    rw = summary.loc["random-walk break"].loc["CUSUM halt"]
    assert rw["detected"] >= 7 and rw["median_detect_delay"] < 63
    nb = summary.loc["no break"]
    assert nb.loc["CUSUM halt", "median_sharpe"] < nb.loc["off", "median_sharpe"]
    assert nb.loc["CUSUM halt", "sharpe_losses_vs_off"] > nb.loc["CUSUM halt", "sharpe_wins_vs_off"]


def test_a_level_shift_is_caught_in_days_and_still_does_not_pay(summary) -> None:
    ls = summary.loc["4-sigma level shift"]
    assert ls.loc["CUSUM halt", "detected"] == 8 and ls.loc["CUSUM halt", "median_detect_delay"] < 21
    assert ls.loc["CUSUM halt", "median_sharpe"] < ls.loc["off", "median_sharpe"]


def test_the_gate_costs_a_healthy_fast_pair_a_fifth_of_its_blocks(summary) -> None:
    nb = summary.loc["no break"]
    assert 0.05 < nb.loc["EG gate 0.01", "mean_pre_gated_frac"] < 0.40
    assert nb.loc["off", "mean_pre_gated_frac"] == 0.0
