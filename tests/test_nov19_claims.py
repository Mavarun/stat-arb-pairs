"""The Nov 19 reading, re-checked at reduced scale, and the committed tables against the rule."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("nov19", ROOT / "scripts" / "run_nov19_experiments.py")
nov19 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nov19)


@pytest.fixture(scope="module")
def small():
    df, b0 = nov19.run(seeds=range(1400, 1408), half_lives=(5.0, 20.0), costs=(2.5, 20.0), scenarios=nov19.SCENARIOS[:1],
                       bar0_cost=2.5)
    return nov19.summarise(df).set_index(["half_life", "cost_bps"]), b0


def test_carry_beats_off_at_dear_execution_and_wins_most_seeds_at_cheap(small) -> None:
    s, _ = small
    for hl in (5.0, 20.0):
        assert s.loc[(hl, 20.0), "carry"] > s.loc[(hl, 20.0), "off"]
        assert s.loc[(hl, 2.5), "wins"] > s.loc[(hl, 2.5), "losses"]
    # At 2.5 bps on 8 seeds the fast pair's median gain is within noise (2.105 vs 2.109); the full
    # 20-seed run has carry ahead (2.141 vs 2.047). The README says the cheap-cost gain is small.
    assert abs(s.loc[(5.0, 2.5), "carry"] - s.loc[(5.0, 2.5), "off"]) < 0.05


def test_the_gain_is_partly_a_cost_saving_so_it_grows_with_cost(small) -> None:
    s, _ = small
    for hl in (5.0, 20.0):
        assert s.loc[(hl, 20.0), "cost_saved"] > s.loc[(hl, 2.5), "cost_saved"] > 0


def test_no_carried_position_flips_at_the_refit(small) -> None:
    _, b0 = small
    assert (b0.action != "flip").all() and (b0.action == "hold").any() and (b0.action == "exit").any()


def test_committed_tables_follow_the_rule() -> None:
    s = pd.read_csv(ROOT / "results" / "nov19_carry_costs.csv")
    assert len(s) == 3 * 2 * 4 and nov19.decide(s) == {"carry_stays_without_caveat": True, "rows_lost": []}
    t = pd.read_csv(ROOT / "results" / "nov19_carry_bar0.csv")
    assert (t.flip == 0).all() and (t.all_blocks == 320).all()


@pytest.fixture(scope="module")
def stops():
    df = nov19.run_stops(seeds=range(1400, 1408), half_lives=(5.0, 20.0))
    return nov19.summarise_stops(df).set_index(["half_life", "scenario", "book"])


def test_cusum_halt_with_carry_loses_even_under_a_break(stops) -> None:
    for hl in (5.0, 20.0):
        for sc in ("no break", "random walk (786)"):
            assert stops.loc[(hl, sc, "carry + CUSUM halt"), "median_sharpe"] < stops.loc[(hl, sc, "carry"), "median_sharpe"]


def test_time_stop_with_carry_is_nearly_inert(stops) -> None:
    ts = stops.xs("carry + time stop", level="book")
    assert ((ts.wins + ts.losses) <= 3).all()


def test_committed_stop_table_follows_the_rule() -> None:
    s = pd.read_csv(ROOT / "results" / "nov19_carry_stops.csv")
    assert nov19.decide_stops(s) == {"carry + time stop": False, "carry + CUSUM halt": False}
