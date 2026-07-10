"""Top signals: highest-conviction purchase signals and per-member signals.

`get_top_signals` returns the global top N by composite score (alpha +
realized return). `get_member_signals` returns the top N for a single
member. Both apply the quality filter and use the conviction score
weights from `constants.py`.
"""

from __future__ import annotations

import pandas as pd

from analyzer.exceptions import AnalysisError
from analyzer.models import TransactionType

from analyzer.signals.constants import (
    CONVICTION_WEIGHT_ALPHA,
    CONVICTION_WEIGHT_REALIZED,
    MIN_ENTRY_PRICE,
)
from analyzer.signals.filters import _apply_quality_filter, _get_horizon_data


_TOP_COLS = [
    "member", "ticker", "disclosure_date", "spy_alpha_pct", "peak_potential_pct",
    "total_return_pct", "total_spy_alpha_pct", "signal_score",
]
_MEMBER_TOP_COLS = [
    "ticker", "disclosure_date", "spy_alpha_pct", "peak_potential_pct",
    "total_return_pct", "total_spy_alpha_pct", "signal_score",
]


def _get_top_signals(signals_df: pd.DataFrame, horizon: int = 90, top_n: int = 15) -> pd.DataFrame:
    top_data = _get_horizon_data(signals_df, horizon, TransactionType.PURCHASE.value)
    if top_data.empty:
        raise AnalysisError(f"No purchase signals found for horizon {horizon}")

    top_data = _apply_quality_filter(top_data)
    if top_data.empty:
        raise AnalysisError(f"No signals survived quality filter (min price ${MIN_ENTRY_PRICE})")

    top_data = top_data.copy()
    top_data["signal_score"] = _compute_conviction_score(top_data)
    top_data = top_data[top_data["signal_score"] > 0]
    return top_data.nlargest(top_n, "signal_score")[_TOP_COLS]


def _get_member_signals(
    signals_df: pd.DataFrame, member: str, horizon: int = 90, top_n: int = 5,
) -> pd.DataFrame:
    member_data = _get_horizon_data(signals_df, horizon)
    member_data = member_data[member_data["member"] == member]

    if member_data.empty:
        raise AnalysisError(f"No signals found for member {member} at horizon {horizon}")

    purchases = member_data[member_data["signal_type"] == TransactionType.PURCHASE.value]
    if purchases.empty:
        raise AnalysisError(f"No purchase signals for member {member} at horizon {horizon}")

    purchases = _apply_quality_filter(purchases)
    if purchases.empty:
        raise AnalysisError(f"No signals survived quality filter for {member}")

    purchases = purchases.copy()
    purchases["signal_score"] = _compute_conviction_score(purchases)
    return purchases.nlargest(top_n, "signal_score")[_MEMBER_TOP_COLS]


def get_top_signals(signal_df: pd.DataFrame, horizon: int = 90, top_n: int = 15) -> pd.DataFrame:
    if signal_df.empty:
        raise AnalysisError("Empty signals dataframe")
    return _get_top_signals(signal_df, horizon, top_n)


def get_member_signals(
    signal_df: pd.DataFrame, member: str, horizon: int = 90, top_n: int = 5,
) -> pd.DataFrame:
    if signal_df.empty:
        raise AnalysisError("Empty signals dataframe")
    return _get_member_signals(signal_df, member, horizon, top_n)


def _compute_conviction_score(df: pd.DataFrame) -> pd.Series:
    """Composite score = total_spy_alpha * ALPHA + total_return * REALIZED."""
    return (
        df["total_spy_alpha_pct"].fillna(0) * CONVICTION_WEIGHT_ALPHA
        + df["total_return_pct"].fillna(0) * CONVICTION_WEIGHT_REALIZED
    )
