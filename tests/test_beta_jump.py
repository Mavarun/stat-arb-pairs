"""make_beta_jump_pair: a pair whose hedge ratio jumps but whose spread stays mean-reverting."""

import numpy as np
import pytest

from statarb.synth import make_beta_jump_pair, make_ou_pair


def test_zero_jump_is_the_plain_pair():
    a = make_beta_jump_pair(400, break_at=200, jump=0.0, half_life=5.0, seed=7)
    b = make_ou_pair(400, half_life=5.0, seed=7)
    np.testing.assert_array_equal(a.y.to_numpy(), b.y.to_numpy())
    np.testing.assert_array_equal(a.x.to_numpy(), b.x.to_numpy())


def test_jump_moves_only_the_post_break_hedge():
    a = make_beta_jump_pair(400, break_at=200, jump=0.02, half_life=5.0, seed=7)
    b = make_beta_jump_pair(400, break_at=200, jump=0.0, half_life=5.0, seed=7)
    np.testing.assert_array_equal(a.y.to_numpy()[:200], b.y.to_numpy()[:200])
    np.testing.assert_array_equal(a.spread.to_numpy(), b.spread.to_numpy())
    bp = a.beta_path.to_numpy()
    assert bp[199] == pytest.approx(1.5) and bp[200] == pytest.approx(1.53)
    # the residual against the true path is the planted spread
    resid = a.y.to_numpy() - a.intercept - bp * a.x.to_numpy()
    np.testing.assert_allclose(resid, a.spread.to_numpy(), atol=1e-9)


def test_straddling_ols_fit_is_worse_than_a_clean_one():
    # the jump at x ~ 100 is a 3-sigma step in y; a window that straddles it trades
    # slope against intercept, so its hedge error is larger than a clean window's,
    # and often lands outside [1.5, 1.53] altogether
    def err(seed, start, truth):
        p = make_beta_jump_pair(600, break_at=300, jump=0.02, half_life=5.0, seed=seed)
        w = slice(start, start + 252)
        X = np.column_stack([np.ones(252), p.x.to_numpy()[w]])
        return abs(np.linalg.lstsq(X, p.y.to_numpy()[w], rcond=None)[0][1] - truth)

    clean = np.mean([err(s, 0, 1.5) for s in range(10)])
    straddle = np.mean([err(s, 174, 1.515) for s in range(10)])
    assert straddle > 2 * clean


@pytest.mark.parametrize("bad", [{"jump": -1.0}, {"jump": float("nan")}, {"jump": 0.01, "break_at": 0}])
def test_rejects_bad_arguments(bad):
    kw = {"break_at": 10, **bad}
    with pytest.raises(ValueError):
        make_beta_jump_pair(50, **kw)
