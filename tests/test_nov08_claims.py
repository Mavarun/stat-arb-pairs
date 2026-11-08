"""The Nov 8 README reading, re-checked at reduced scale (first 8 of the 20 seeds, aligned break)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "nov08", Path(__file__).resolve().parents[1] / "scripts" / "run_nov08_experiments.py"
)
nov08 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nov08)

BOOKS = {k: nov08.BOOKS[k] for k in ("off", "EG gate 0.01", "calibrated q0.95")}


@pytest.fixture(scope="module")
def summary():
    df = nov08.run(seeds=range(1200, 1208), half_lives=(10.0, 20.0), breaks=nov08.BREAKS[:1],
                   scenarios=nov08.SCENARIOS[:2], books=BOOKS)
    return nov08.summarise(df).set_index(["half_life", "scenario", "book"])


def test_calibrated_gate_trades_a_healthy_hl10_pair_that_the_fixed_gate_starves(summary) -> None:
    nb = summary.loc[(10.0, "no break")]
    assert nb.loc["EG gate 0.01", "mean_pre_gated_frac"] > 0.5
    assert nb.loc["calibrated q0.95", "mean_pre_gated_frac"] < 0.4
    assert nb.loc["calibrated q0.95", "median_sharpe"] > nb.loc["EG gate 0.01", "median_sharpe"]


def test_a_healthy_hl20_pair_loses_after_costs_so_starving_it_is_cheap(summary) -> None:
    nb = summary.loc[(20.0, "no break")]
    assert nb.loc["off", "median_sharpe"] < 0.0
    assert nb.loc["EG gate 0.01", "mean_pre_gated_frac"] > 0.8
    assert nb.loc["EG gate 0.01", "median_sharpe"] >= nb.loc["calibrated q0.95", "median_sharpe"]


def test_the_fixed_gate_keeps_more_power_on_a_dead_slow_pair(summary) -> None:
    rw = summary.loc[(20.0, "random-walk break")]
    cal = rw.loc["calibrated q0.95"]
    assert cal["losses_vs_fixed"] > cal["wins_vs_fixed"]
    assert rw.loc["EG gate 0.01", "mean_post_gated_frac"] > cal["mean_post_gated_frac"]


def test_every_gate_cuts_the_dead_pair_loss(summary) -> None:
    for hl in (10.0, 20.0):
        rw = summary.loc[(hl, "random-walk break")]
        off = rw.loc["off", "median_post_break_return"]
        assert rw.loc["EG gate 0.01", "median_post_break_return"] > off
        assert rw.loc["calibrated q0.95", "median_post_break_return"] > off
