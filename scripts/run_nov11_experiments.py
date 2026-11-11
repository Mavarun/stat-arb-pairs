"""Nov 11 slice: is the slow-pair loss the trading rule (63-bar blocks, forced flat) rather than the test?

Synthetic, seeded, offline. Run: ``python scripts/run_nov11_experiments.py`` (~4 min).
Static OLS book, 252-bar formation, every traded bar out-of-sample (the same 1008 traded
bars for every block length on a 1260-bar path), net of 10 bps per leg. Nov 8 found a
healthy half-life-20 pair loses money ungated (median Sharpe -0.41) and guessed the cause:
with z 2 / 0.5 and every block forced flat after 63 bars, a slow spread rarely reverts
inside a block. This tests that guess directly.

Fixed before the first run: the books below, half-lives {5, 10, 20}, 20 new seeds
(1300-1319), scenarios no break / random-walk break on bar 756 (first bar of a 63-bar
block, also a 126- and 252-bar boundary) / random-walk break on bar 786 (mid-block).
Paired sign tests of every book vs ``off`` with a Holm adjustment over every comparison
in the table. Decision rule (also fixed): ``carry_over`` ships on as the default only if
it beats ``off`` on median Sharpe in every half-life's no-break row with Holm p < 0.05
in at least one of them and does not lose to ``off`` on median Sharpe in any break row;
longer blocks are a parameter (``trading``) and are reported, not shipped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from statarb.formation import formation_trading_backtest
from statarb.synth import make_broken_pair

SEEDS = range(1300, 1320)
N = 1260
HALF_LIVES = (5.0, 10.0, 20.0)
SCENARIOS = (("no break", "none", 756), ("random walk (756)", "random_walk", 756), ("random walk (786)", "random_walk", 786))
BOOKS = {
    "off": {},
    "126-bar blocks": {"trading": 126},
    "252-bar blocks": {"trading": 252},
    "carry": {"carry_over": True},
    "EG gate 0.01": {"gate_p": 0.01},
    "carry + EG gate 0.01": {"carry_over": True, "gate_p": 0.01},
}


def forced_exit_share(r, carry: bool = False) -> float:
    """Share of exits the block end forced (open on a block's second-to-last bar, flat on its last).

    A carried book is only forced flat on the final block, so only that block counts for it.
    """
    pos = r.positions
    ends = list(r.windows["trade_end"])
    forced = 0
    for k, e in enumerate(ends):
        if carry and k < len(ends) - 1:
            continue
        i = pos.index.get_loc(e)
        forced += i > 0 and pos.iloc[i - 1] != 0.0 and pos.iloc[i] == 0.0
    prev = pos.shift(1, fill_value=0.0)
    exits = int(((prev != 0.0) & (pos != prev)).sum())
    return forced / exits if exits else float("nan")


def run(seeds=SEEDS, half_lives=HALF_LIVES, scenarios=SCENARIOS, books=None) -> pd.DataFrame:
    books = BOOKS if books is None else books
    rows = []
    for hl in half_lives:
        for label, kind, break_at in scenarios:
            for s in seeds:
                p = make_broken_pair(N, break_at=break_at, kind=kind, half_life=hl, shift_sigmas=4.0, seed=s)
                cut = p.y.index[break_at]
                for book, kw in books.items():
                    r = formation_trading_backtest(p.y, p.x, hedge="ols", **kw)
                    post = r.returns.loc[cut:]
                    rows.append({
                        "half_life": hl, "scenario": label, "seed": s, "book": book,
                        "sharpe": r.sharpe, "total_return": r.total_return,
                        "post_return": float((1 + post).prod() - 1), "max_dd": r.max_drawdown,
                        "cost": r.cost_paid, "trades": r.trades, "n_bars": len(r.returns),
                        "forced_exit_share": forced_exit_share(r, carry=bool(kw.get("carry_over"))),
                    })
    return pd.DataFrame(rows)


def holm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    q = p[ok]
    order = np.argsort(q)
    m = len(q)
    adj = np.maximum.accumulate((m - np.arange(m)) * q[order]).clip(max=1.0)
    res = np.empty(m)
    res[order] = adj
    out[ok] = res
    return out


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (hl, sc), g in df.groupby(["half_life", "scenario"], sort=False):
        piv = g.pivot(index="seed", columns="book", values="sharpe")
        for book, gb in g.groupby("book", sort=False):
            row = {"half_life": hl, "scenario": sc, "book": book,
                   "median_sharpe": gb.sharpe.median(), "median_total_return": gb.total_return.median(),
                   "median_post_return": gb.post_return.median(), "median_max_dd": gb.max_dd.median(),
                   "mean_cost": gb.cost.mean(), "mean_trades": gb.trades.mean(),
                   "mean_forced_exit_share": gb.forced_exit_share.mean(), "n_bars": int(gb.n_bars.iloc[0])}
            if book == "off":
                row.update(wins=np.nan, losses=np.nan, sign_p=np.nan)
            else:
                d = piv[book] - piv["off"]
                w, lo = int((d > 1e-12).sum()), int((d < -1e-12).sum())
                row.update(wins=w, losses=lo, sign_p=binomtest(w, w + lo, 0.5).pvalue if w + lo else 1.0)
            out.append(row)
    s = pd.DataFrame(out)
    s["holm_p"] = holm(s["sign_p"].to_numpy())
    return s


def carry_ships(s: pd.DataFrame) -> bool:
    """The pre-registered decision rule for making ``carry_over`` the default."""
    c = s[s.book == "carry"].set_index(["half_life", "scenario"])
    o = s[s.book == "off"].set_index(["half_life", "scenario"])
    nb = [k for k in c.index if k[1] == "no break"]
    br = [k for k in c.index if k[1] != "no break"]
    beats = all(c.loc[k, "median_sharpe"] > o.loc[k, "median_sharpe"] for k in nb)
    sig = any(c.loc[k, "holm_p"] < 0.05 and c.loc[k, "wins"] > c.loc[k, "losses"] for k in nb)
    no_loss = all(c.loc[k, "median_sharpe"] >= o.loc[k, "median_sharpe"] for k in br)
    return bool(beats and sig and no_loss)


def main() -> None:
    s = summarise(run())
    pd.set_option("display.width", 320)
    pd.set_option("display.max_columns", 40)
    cols = ["book", "median_sharpe", "wins", "losses", "sign_p", "holm_p", "median_post_return",
            "median_max_dd", "mean_cost", "mean_trades", "mean_forced_exit_share", "n_bars"]
    for (hl, sc), g in s.groupby(["half_life", "scenario"], sort=False):
        print(f"\n== half-life {hl:g} | {sc}")
        print(g[cols].round(3).to_string(index=False))
    print("\ncarry_over ships on:", carry_ships(s))


if __name__ == "__main__":
    main()
