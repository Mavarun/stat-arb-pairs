"""Formation-only ranking and top-k selection for screened pairs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statarb.rank import (
    HL_HI,
    HL_LO,
    annotate_half_lives,
    formation_residual,
    score_discoveries,
    select_top,
)
from statarb.screening import screen_pairs
from statarb.synth import make_ou_pair, make_planted_universe, make_random_walk_universe
from statarb.ou import fit_ou


def test_formation_residual_recovers_planted_spread_half_life():
    p = make_ou_pair(800, beta=1.4, half_life=8.0, seed=3)
    resid = formation_residual(p.y, p.x)
    est = fit_ou(resid).half_life
    assert abs(est - 8.0) / 8.0 < 0.25


def test_annotate_half_lives_marks_planted_pairs_mean_reverting():
    frame, truth = make_planted_universe(6, 2, 400, half_life=5.0, seed=7)
    screen = screen_pairs(frame)
    ann = annotate_half_lives(screen, frame)
    assert "half_life" in ann.columns
    assert "is_mean_reverting" in ann.columns
    # At least the planted pairs should show up with finite HL somewhere in the frame.
    planted_mask = [
        frozenset((a, b)) in truth for a, b in zip(ann["a"], ann["b"])
    ]
    planted = ann.loc[planted_mask]
    assert len(planted) == 2
    assert planted["is_mean_reverting"].all()
    assert planted["half_life"].between(2.0, 20.0).all()


def test_score_penalises_half_lives_outside_band():
    rows = pd.DataFrame(
        {
            "a": ["A", "B", "C"],
            "b": ["X", "Y", "Z"],
            "eg_p": [1e-4, 1e-4, 1e-4],
            "half_life": [5.0, 80.0, 1.0],
            "is_mean_reverting": [True, True, True],
            "sigma_eq": [1.0, 1.0, 1.0],
        }
    )
    scored = score_discoveries(rows, hl_lo=2.0, hl_hi=40.0)
    assert np.isfinite(scored.loc[0, "score"])
    assert scored.loc[1, "score"] == -np.inf
    assert scored.loc[2, "score"] == -np.inf
    # Tighter HL with same p scores higher.
    tight = rows.iloc[[0]].copy()
    tight.loc[0, "half_life"] = 4.0
    loose = rows.iloc[[0]].copy()
    loose.loc[0, "half_life"] = 20.0
    assert score_discoveries(tight).iloc[0]["score"] > score_discoveries(loose).iloc[0]["score"]


def test_select_top_by_eg_p_and_cap():
    frame, _ = make_planted_universe(8, 3, 350, half_life=4.0, seed=11)
    screen = screen_pairs(frame)
    # Use naive so we have several discoveries to rank/cap.
    all_naive = select_top(screen, rule="naive", rank_by="eg_p", max_pairs=None)
    top2 = select_top(screen, rule="naive", rank_by="eg_p", max_pairs=2)
    assert len(top2) == min(2, len(all_naive))
    if len(all_naive) >= 2:
        assert list(top2["eg_p"]) == sorted(all_naive["eg_p"])[:2]


def test_select_top_by_score_prefers_short_half_life_when_p_similar():
    # Hand-built screen: same eg_p, different HL → score rank picks the short one.
    screen = pd.DataFrame(
        {
            "a": ["S", "L"],
            "b": ["X", "Y"],
            "eg_p_ab": [1e-3, 1e-3],
            "eg_p_ba": [1e-3, 1e-3],
            "eg_p": [1e-3, 1e-3],
            "naive": [True, True],
            "bh": [True, True],
            "bh_johansen": [False, False],
            "naive_single": [True, True],
            "johansen_only": [False, False],
            "half_life": [5.0, 30.0],
            "sigma_eq": [1.0, 1.0],
            "is_mean_reverting": [True, True],
        }
    )
    top = select_top(screen, rule="bh", rank_by="score", max_pairs=1)
    assert len(top) == 1
    assert top.iloc[0]["a"] == "S"
    assert top.iloc[0]["half_life"] == 5.0


def test_select_top_rejects_unknown_rank_key():
    frame = make_random_walk_universe(4, 80, seed=1)
    screen = screen_pairs(frame)
    with pytest.raises(ValueError, match="rank_by"):
        select_top(screen, rule="bh", rank_by="not_a_key", prices=frame)
