"""Backtest result summarization: per-rank win rate, avg return, avg alpha.

Bug 4 fixes:
  - A "SPY_BH" (SPY buy-and-hold) row is appended to the summary so callers
    can compare strategy performance against the passive benchmark over the
    same evaluation windows.
  - Coverage gap counts (n_no_price, n_delisted) are propagated from
    result.attrs into summary.attrs so callers can report how many
    recommendations were excluded or filled at last available price.
"""

from __future__ import annotations

import pandas as pd

_BT_SUMMARY_COLS = ["avg_raw_return_pct", "avg_leverage"]


def summarize_backtest(results: pd.DataFrame) -> pd.DataFrame:
    # Propagate coverage counts set by evaluate_backtest (Bug 4)
    n_no_price = results.attrs.get("n_no_price", 0)
    n_delisted = results.attrs.get("n_delisted", 0)

    valid = results.dropna(subset=["bt_return_pct"])
    if valid.empty:
        summary = pd.DataFrame()
        summary.attrs["n_no_price"] = n_no_price
        summary.attrs["n_delisted"] = n_delisted
        return summary

    by_rank = []
    for rank, grp in valid.groupby("rank"):
        by_rank.append(_summary_row(rank, grp))

    summary = pd.DataFrame(by_rank)

    overall = _summary_row("ALL", valid)

    # Bug 4: explicit SPY buy-and-hold baseline over the same evaluation windows
    spy_row = _spy_baseline_row(valid)

    summary = pd.concat(
        [summary, pd.DataFrame([overall, spy_row])],
        ignore_index=True,
    )

    summary.attrs["n_no_price"] = n_no_price
    summary.attrs["n_delisted"] = n_delisted
    return summary


def _summary_row(rank, grp: pd.DataFrame) -> dict:
    entry = {
        "rank": rank,
        "count": len(grp),
        "win_rate_pct": round((grp["bt_return_pct"] > 0).mean() * 100, 1),
        "avg_return_pct": round(grp["bt_return_pct"].mean(), 2),
        "avg_alpha_pct": (
            round(grp["bt_alpha_pct"].mean(), 2)
            if "bt_alpha_pct" in grp.columns
            else None
        ),
    }
    for col in _BT_SUMMARY_COLS:
        if col.replace("avg_", "bt_") in grp.columns:
            entry[col] = round(grp[col.replace("avg_", "bt_")].mean(), 2)
    return entry


def _spy_baseline_row(valid: pd.DataFrame) -> dict:
    """Compute SPY buy-and-hold statistics over the same evaluation windows.

    Uses bt_spy_return_pct already stored per-recommendation to avoid
    re-fetching prices.  avg_alpha_pct is 0 by definition (SPY vs itself).
    """
    if "bt_spy_return_pct" not in valid.columns:
        return {"rank": "SPY_BH", "count": 0, "win_rate_pct": None,
                "avg_return_pct": None, "avg_alpha_pct": 0.0}
    spy_rets = valid["bt_spy_return_pct"].dropna()
    if spy_rets.empty:
        return {"rank": "SPY_BH", "count": 0, "win_rate_pct": None,
                "avg_return_pct": None, "avg_alpha_pct": 0.0}
    return {
        "rank": "SPY_BH",
        "count": len(spy_rets),
        "win_rate_pct": round((spy_rets > 0).mean() * 100, 1),
        "avg_return_pct": round(spy_rets.mean(), 2),
        "avg_alpha_pct": 0.0,  # SPY alpha vs itself is zero by definition
    }
