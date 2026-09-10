"""Descriptive historical purchase outcomes."""

from __future__ import annotations

import pandas as pd

from analyzer.exceptions import AnalysisError
from analyzer.models import TransactionType
from analyzer.signals.filters import _collapse_to_episodes, _get_horizon_data


_TOP_COLS = [
    "member",
    "ticker",
    "disclosure_date",
    "total_return_pct",
    "total_spy_alpha_pct",
]
_MEMBER_TOP_COLS = [
    "ticker",
    "disclosure_date",
    "total_return_pct",
    "total_spy_alpha_pct",
]


def _completed_purchases(signals_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    purchases = _get_horizon_data(
        signals_df, horizon, TransactionType.PURCHASE.value
    ).copy()
    if purchases.empty:
        raise AnalysisError(f"No completed purchase signals found for horizon {horizon}")
    if "total_spy_alpha_pct" not in purchases.columns:
        raise AnalysisError("Historical purchase outcomes require endpoint SPY alpha")
    purchases = _collapse_to_episodes(purchases)
    return purchases[purchases["total_spy_alpha_pct"].notna()].copy()


def _get_top_signals(
    signals_df: pd.DataFrame, horizon: int = 90, top_n: int = 15
) -> pd.DataFrame:
    purchases = _completed_purchases(signals_df, horizon)
    if purchases.empty:
        raise AnalysisError("No complete endpoint purchase outcomes found")
    return purchases.nlargest(top_n, "total_spy_alpha_pct")[_TOP_COLS]


def _get_member_signals(
    signals_df: pd.DataFrame,
    member: str,
    horizon: int = 90,
    top_n: int = 5,
) -> pd.DataFrame:
    purchases = _completed_purchases(signals_df, horizon)
    purchases = purchases[purchases["member"] == member]
    if purchases.empty:
        raise AnalysisError(
            f"No complete purchase outcomes for member {member} at horizon {horizon}"
        )
    return purchases.nlargest(top_n, "total_spy_alpha_pct")[_MEMBER_TOP_COLS]


def get_top_signals(
    signal_df: pd.DataFrame, horizon: int = 90, top_n: int = 15
) -> pd.DataFrame:
    if signal_df.empty:
        raise AnalysisError("Empty signals dataframe")
    return _get_top_signals(signal_df, horizon, top_n)


def get_member_signals(
    signal_df: pd.DataFrame,
    member: str,
    horizon: int = 90,
    top_n: int = 5,
) -> pd.DataFrame:
    if signal_df.empty:
        raise AnalysisError("Empty signals dataframe")
    return _get_member_signals(signal_df, member, horizon, top_n)
