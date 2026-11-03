"""The null false-alarm rate of the default CUSUM is small and not driven by the half-life."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from statarb.breakwatch import H_ALARM, K_SLACK

spec = importlib.util.spec_from_file_location(
    "nov03cal", Path(__file__).resolve().parents[1] / "scripts" / "run_nov03_calibration.py"
)
cal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cal)


@pytest.fixture(scope="module")
def table():
    return cal.false_alarm_table(seeds=range(1000, 1015), half_lives=(3.0, 20.0), ks=(0.25, K_SLACK, 1.0), hs=(H_ALARM,))


def test_default_false_alarm_rate_is_small_at_both_half_lives(table) -> None:
    d = table[table.k == K_SLACK]
    assert (d[f"FA@h={H_ALARM:g}"] < 0.12).all()


def test_slack_is_what_controls_false_alarms(table) -> None:
    col = f"FA@h={H_ALARM:g}"
    for hl, g in table.groupby("half_life"):
        by_k = g.set_index("k")[col]
        assert by_k[0.25] > by_k[K_SLACK] >= by_k[1.0]
        assert by_k[0.25] > 0.10  # a small slack is not usable on a healthy pair


def test_blocks_counted(table) -> None:
    assert (table["blocks"] == 15 * 16).all()


def test_engle_granger_gate_starves_slow_pairs() -> None:
    t = cal.false_gating_table(seeds=range(1000, 1010), half_lives=(3.0, 20.0), gate_ps=(0.01, 0.05)).set_index("half_life")
    assert t.loc[3.0, "windows"] == 10 * 16
    assert t.loc[3.0, "gated@0.01"] < 0.15  # a fast healthy pair is almost always traded
    assert t.loc[20.0, "gated@0.01"] > 0.5  # a slow healthy pair mostly is not: 252 bars have little EG power
    assert (t["gated@0.05"] <= t["gated@0.01"]).all()
