"""Regularised inner selection for nested Kalman tuning (one-SE and sign-test rules)."""

from __future__ import annotations

import numpy as np
import pytest

from statarb.kalman import Q_ALPHA, Q_BETA
from statarb.nested import DEFAULT_Q_GRID, _select, _sign_test_p, tune_q, tune_q_beta
from statarb.synth import make_ou_pair

D = (Q_BETA, Q_ALPHA)


def _c(q, ws):
    return {"key": (q, Q_ALPHA), "ws": list(ws), "median_sharpe": float(np.median(ws))}


def test_sign_test_p_values():
    assert _sign_test_p(0, 0) == 1.0
    assert np.isclose(_sign_test_p(5, 5), 1 / 32)
    assert np.isclose(_sign_test_p(4, 5), 6 / 32)
    assert np.isclose(_sign_test_p(0, 5), 1.0)


def test_best_rule_takes_the_highest_median_and_breaks_ties_to_default():
    cands = [_c(1e-7, [0.1, 0.2, 0.3]), _c(Q_BETA, [0.1, 0.2, 0.3]), _c(1e-5, [0.0, 0.1, 0.2])]
    assert _select(cands, D, "best", 0.1)[0] == D
    cands = [_c(1e-7, [0.5, 0.6, 0.7]), _c(Q_BETA, [0.1, 0.2, 0.3])]
    assert _select(cands, D, "best", 0.1)[0] == (1e-7, Q_ALPHA)


def test_one_se_keeps_the_default_when_the_edge_is_inside_the_noise():
    noisy = [-1.0, 2.0, 0.4, -0.5, 1.5, 0.2]
    cands = [_c(1e-5, [v + 0.1 for v in noisy]), _c(Q_BETA, noisy)]
    q, ann = _select(cands, D, "one_se", 0.1)
    assert q == D
    assert all(c["within_one_se"] for c in ann)
    # a large, clean edge leaves the one-SE band
    clean = [_c(1e-5, [2.0, 2.1, 1.9, 2.05]), _c(Q_BETA, [0.0, 0.1, -0.1, 0.05])]
    assert _select(clean, D, "one_se", 0.1)[0] == (1e-5, Q_ALPHA)


def test_one_se_prefers_the_closest_q_to_default_inside_the_band():
    cands = [_c(1e-4, [1.0, 1.1, 0.9, 1.2]), _c(1e-5, [0.95, 1.05, 0.9, 1.15]), _c(Q_BETA, [0.0, 0.1, -0.1, 0.0])]
    assert _select(cands, D, "one_se", 0.1)[0] == (1e-5, Q_ALPHA)


def test_sign_rule_needs_consistent_paired_wins():
    base = [0.0, 0.1, -0.2, 0.3, 0.05, -0.1]
    always = _c(1e-5, [v + 0.2 for v in base])            # wins 6/6, p = 1/64
    mixed = _c(1e-4, [v + d for v, d in zip(base, [1, -1, 1, -1, 1, -1])])
    q, ann = _select([always, mixed, _c(Q_BETA, base)], D, "sign", 0.1)
    assert q == (1e-5, Q_ALPHA)
    by = {c["key"][0]: c for c in ann}
    assert np.isclose(by[1e-5]["sign_p"], 1 / 64)
    assert by[1e-4]["sign_p"] > 0.1
    # nothing consistent -> default
    assert _select([mixed, _c(Q_BETA, base)], D, "sign", 0.1)[0] == D


def test_sign_rule_requires_the_default_in_the_grid():
    with pytest.raises(ValueError):
        _select([_c(1e-5, [1.0, 2.0])], D, "sign", 0.1)


def test_unknown_rule_rejected_and_short_sample_falls_back():
    p = make_ou_pair(84 + 21 * 3, half_life=5, seed=41)
    with pytest.raises(ValueError):
        tune_q_beta(p.y, p.x, selection="max")
    short = make_ou_pair(50, seed=42)
    qb, qa, table = tune_q(short.y, short.x, selection="sign")
    assert (qb, qa) == (Q_BETA, Q_ALPHA) and table.empty


def test_regularised_rules_return_grid_members_and_annotate_the_table():
    p = make_ou_pair(84 + 21 * 6, half_life=5, seed=43)
    for rule, col in (("one_se", "within_one_se"), ("sign", "sign_p")):
        q, table = tune_q_beta(p.y, p.x, formation=84, trading=21, selection=rule)
        assert q in DEFAULT_Q_GRID
        assert col in table.columns and len(table) == len(DEFAULT_Q_GRID)


def test_joint_grid_scores_every_pair():
    p = make_ou_pair(84 + 21 * 4, half_life=5, seed=44)
    qa_grid = (Q_ALPHA, Q_ALPHA * 100)
    qb, qa, table = tune_q(p.y, p.x, formation=84, trading=21, q_alpha_grid=qa_grid, selection="one_se")
    assert len(table) == len(DEFAULT_Q_GRID) * 2
    assert qb in DEFAULT_Q_GRID and qa in qa_grid
    assert set(zip(table["q_beta"], table["q_alpha"])) == {(b, a) for a in qa_grid for b in DEFAULT_Q_GRID}


def test_regularised_rules_are_deterministic():
    form = make_ou_pair(84 + 21 * 4, half_life=5, seed=45)
    for rule in ("one_se", "sign"):
        a = tune_q_beta(form.y, form.x, formation=84, trading=21, selection=rule)[0]
        b = tune_q_beta(form.y.copy(), form.x.copy(), formation=84, trading=21, selection=rule)[0]
        assert a == b
