"""Nov 8 calibration reading at reduced scale: the calibrated gate trades premium for power."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "nov08cal", Path(__file__).resolve().parents[1] / "scripts" / "run_nov08_calibration.py"
)
cal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cal)


@pytest.fixture(scope="module")
def hl20():
    kw = dict(quantiles=(0.95,), estimated=False)
    return (cal.gate_rates(20.0, range(1000, 1006), dead=False, **kw),
            cal.gate_rates(20.0, range(1100, 1106), dead=True, **kw))


def test_calibrated_gate_cuts_the_slow_pair_premium(hl20) -> None:
    healthy, _ = hl20
    assert healthy["fixed 0.01"] > 0.7
    assert healthy["q0.95 oracle"] < 0.2


def test_and_gives_up_most_of_the_power_on_a_dead_slow_pair(hl20) -> None:
    _, dead = hl20
    assert dead["fixed 0.01"] > 0.9
    assert dead["q0.95 oracle"] < 0.6


def test_estimated_reference_is_biased_fast_on_a_slow_pair(hl20) -> None:
    healthy, _ = hl20
    assert healthy["median_ref_hl_hat"] < 16.0  # planted 20: the 252-bar AR(1) fit is biased down
