"""Pair screening under multiple testing. Synthetic, seeded, offline."""

import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.multitest import multipletests

from statarb.screening import benjamini_hochberg, screen_pairs, screen_summary
from statarb.synth import make_planted_universe, make_random_walk_universe, n_pairs


def test_bh_matches_statsmodels_reference():
    rng = np.random.default_rng(0)
    p = np.concatenate([rng.uniform(size=80), rng.uniform(0, 0.002, size=20)])
    rng.shuffle(p)
    rej, q = benjamini_hochberg(p, alpha=0.1)
    ref_rej, ref_q, _, _ = multipletests(p, alpha=0.1, method="fdr_bh")
    np.testing.assert_array_equal(rej, ref_rej)
    np.testing.assert_allclose(q, ref_q, rtol=1e-12)


def test_bh_hand_example_and_nan_handling():
    # m = 4 finite p-values; thresholds 0.0125, 0.025, 0.0375, 0.05 at alpha=0.05.
    p = [0.01, np.nan, 0.02, 0.035, 0.20]
    rej, q = benjamini_hochberg(p, alpha=0.05)
    assert rej.tolist() == [True, False, True, True, False]
    assert np.isnan(q[1])
    np.testing.assert_allclose(q[[0, 2, 3, 4]], [0.04, 0.04, 0.035 * 4 / 3, 0.2])
    # 0.035 <= 3/4 * 0.05 but 0.04 would not be: step-up boundary.
    assert not benjamini_hochberg([0.01, 0.02, 0.04, 0.20])[0][2]
    with pytest.raises(ValueError):
        benjamini_hochberg([0.5, 1.5])
    with pytest.raises(ValueError):
        benjamini_hochberg([0.5], alpha=0.0)
    rej0, q0 = benjamini_hochberg([np.nan, np.nan])
    assert not rej0.any() and np.isnan(q0).all()


def test_naive_screen_picks_false_pairs_on_pure_random_walks_bh_does_not():
    totals = {"naive_single": 0, "naive": 0, "johansen_only": 0, "bh": 0, "bh_johansen": 0}
    n_tests = 0
    for seed in range(5):
        px = make_random_walk_universe(20, 500, seed=seed)
        screen = screen_pairs(px)
        assert len(screen) == n_pairs(20) == 190
        summ = screen_summary(screen)  # no truth: every discovery is false
        n_tests += len(screen)
        for rule in totals:
            assert summ.loc[rule, "true_pos"] == 0
            totals[rule] += int(summ.loc[rule, "false_pos"])
    # Uncorrected rules "discover" pairs that cannot exist.
    assert totals["naive_single"] >= 10
    assert 0.01 < totals["naive_single"] / n_tests < 0.10
    assert totals["naive"] >= 5
    # BH across 950 null tests: essentially nothing survives.
    assert totals["bh"] <= 1
    assert totals["bh_johansen"] <= totals["bh"]


def test_johansen_alone_over_rejects_on_random_walks():
    # Finite-sample size distortion of the trace test on geometric walks:
    # a 95% critical value rejects well above 5% here. That is why Johansen is
    # only a confirmation filter after BH, never a screen on its own.
    rates = []
    for seed in range(5):
        s = screen_pairs(make_random_walk_universe(20, 500, seed=seed))
        rates.append(s["johansen_only"].mean())
    assert np.mean(rates) > 0.06


def test_bh_recovers_planted_pairs_with_no_false_discoveries():
    tp = fp_bh = fp_naive = 0
    for seed in range(3):
        px, truth = make_planted_universe(16, 4, 750, seed=seed)
        summ = screen_summary(screen_pairs(px), truth)
        tp += int(summ.loc["bh_johansen", "true_pos"])
        fp_bh += int(summ.loc["bh_johansen", "false_pos"])
        fp_naive += int(summ.loc["naive_single", "false_pos"])
        assert summ.loc["naive", "true_pos"] == 4  # uncorrected finds them, plus junk
    assert tp >= 10  # of 12 planted
    assert fp_bh == 0
    assert fp_naive > 0


def test_bh_johansen_is_subset_of_bh_and_screen_is_sorted():
    px, _ = make_planted_universe(6, 2, 400, seed=5)
    s = screen_pairs(px)
    assert (s["bh_johansen"] <= s["bh"]).all()
    assert (s["bh"] <= s["naive"]).all()
    assert s["eg_p"].is_monotonic_increasing
    np.testing.assert_allclose(s["eg_p"], np.maximum(s["eg_p_ab"], s["eg_p_ba"]))


def test_screen_rejects_bad_inputs():
    idx = pd.bdate_range("2020-01-01", periods=100)
    with pytest.raises(ValueError):
        screen_pairs(pd.DataFrame({"a": np.arange(100.0)}, index=idx))
    with pytest.raises(ValueError):
        screen_pairs(pd.DataFrame({"a": np.arange(20.0), "b": np.arange(20.0)}))
