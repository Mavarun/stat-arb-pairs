"""The Nov 20 beta-jump reading, re-checked at reduced scale, and the committed tables against the rule."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("nov20", ROOT / "scripts" / "run_nov20_experiments.py")
nov20 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nov20)
RES = ROOT / "results"


@pytest.fixture(scope="module")
def small():
    df, b0 = nov20.run(seeds=range(1500, 1508), half_lives=(5.0, 20.0), jumps=(0.0, 0.02), hedges=("ols",))
    return nov20.summarise(df).set_index(["half_life", "jump"]), nov20.bar0_counts(b0)


def test_carry_still_beats_off_under_a_2pct_jump(small) -> None:
    s, _ = small
    for hl in (5.0, 20.0):
        for j in (0.0, 0.02):
            assert s.loc[(hl, j), "carry"] > s.loc[(hl, j), "off"]
            assert s.loc[(hl, j), "wins"] >= 6


def test_the_jump_costs_both_books(small) -> None:
    s, _ = small
    for book in ("off", "carry"):
        assert s.loc[(5.0, 0.02), book] < s.loc[(5.0, 0.0), book]


def test_no_flip_even_with_a_jump(small) -> None:
    _, t = small
    assert t["flip"].sum() == 0 and t["carried_blocks"].sum() > 0


def test_committed_table_against_the_rule() -> None:
    s = pd.read_csv(RES / "nov20_beta_jump.csv")
    assert len(s) == 24 and set(s.hedge) == {"ols", "kalman"}
    d = nov20.decide(s)
    assert d == {"carry_stays_without_caveat": True, "rows_lost": [], "smallest_losing_jump": None}
    ols = s[s.hedge == "ols"]
    assert (ols.holm_p < 0.01).all() and (ols.wins >= 19).all()
    b = pd.read_csv(RES / "nov20_beta_jump_bar0.csv")
    assert b["flip"].sum() == 0 and b[b.hedge == "ols"]["carried_blocks"].sum() == 1196


def test_kalman_books_carry_almost_nothing() -> None:
    b = pd.read_csv(RES / "nov20_beta_jump_bar0.csv")
    s = pd.read_csv(RES / "nov20_beta_jump.csv")
    k = s[s.hedge == "kalman"]
    assert b[b.hedge == "kalman"]["carried_blocks"].max() <= 8
    assert (k.carry < 0).sum() >= 10  # the Kalman book loses money in almost every row either way
