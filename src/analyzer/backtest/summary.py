"""Backtest result summarization: per-rank win rate, avg return, avg alpha."""

from __future__ import annotations

import pandas as pd

_BT_SUMMARY_COLS = ["avg_raw_return_pct", "avg_leverage"]


def summarize_backtest(results: pd.DataFrame) -> pd.DataFrame:
    valid = results.dropna(subset=["bt_return_pct"])
    if valid.empty:
        return pd.DataFrame()

    by_rank = []
    for rank, grp in valid.groupby("rank"):
        by_rank.append(_summary_row(rank, grp))

    summary = pd.DataFrame(by_rank)

    overall = _summary_row("ALL", valid)
    summary = pd.concat([summary, pd.DataFrame([overall])], ignore_index=True)
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
