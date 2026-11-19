"""windows["carry_bar0"]: what a block does on its first bar with a carried-in position."""

from statarb.formation import formation_trading_backtest
from statarb.synth import make_broken_pair


def _run(hl=20.0, seed=1300, **kw):
    p = make_broken_pair(1260, break_at=756, kind="none", half_life=hl, shift_sigmas=4.0, seed=seed)
    return formation_trading_backtest(p.y, p.x, hedge="ols", **kw)


def test_labels_match_carried_in_and_first_position():
    r = _run()
    w = r.windows
    assert set(w.carry_bar0) <= {"", "hold", "exit", "flip"}
    assert ((w.carried_in == 0.0) == (w.carry_bar0 == "")).all()
    for _, row in w[w.carried_in != 0.0].iterrows():
        first = r.positions.loc[row.trade_start]
        expect = "hold" if first == row.carried_in else ("exit" if first == 0.0 else "flip")
        assert row.carry_bar0 == expect


def test_slow_pair_carries_something_and_legacy_book_never_does():
    assert (_run().windows.carry_bar0 != "").any()
    assert (_run(carry_over=False).windows.carry_bar0 == "").all()


def test_gated_block_reports_an_exit_for_a_carried_position():
    found = False
    for s in range(1300, 1310):
        w = _run(seed=s, gate_p=0.01).windows
        g = w[(w.carried_in != 0.0) & w.gated]
        if len(g):
            assert (g.carry_bar0 == "exit").all()
            found = True
    assert found
