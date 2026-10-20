import statarb


def test_oct20_names_are_exported():
    for name in ("BORROW_BPS", "SELECTION_RULES", "tune_q", "tune_q_beta", "choose_time_stop_multiple",
                 "formation_trading_backtest", "nested_kalman_backtest"):
        assert name in statarb.__all__, name
        assert hasattr(statarb, name), name


def test_borrow_default_keeps_legacy_results():
    assert statarb.BORROW_BPS == 0.0
    assert statarb.SELECTION_RULES == ("best", "one_se", "sign")
