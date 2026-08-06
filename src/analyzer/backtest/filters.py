"""Training / recent / ticker-perf window filters for backtest.

Each filter returns a memoized copy of the source DataFrame so downstream
scoring helpers can rely on stable `id()` for further memoization. `copy=False`
is used because callers treat the result as read-only.
"""

from __future__ import annotations

import pandas as pd

from analyzer._memo import df_memoize
from analyzer.models import TransactionType


@df_memoize(copy=False)
def _filter_training(
    signals_df: pd.DataFrame,
    horizon: int,
    as_of_iso: str,
    training_lookback_iso: str | None,
) -> pd.DataFrame:
    """Filter signals to the training window. Result is shared (id-stable)."""
    as_of_date = pd.Timestamp(as_of_iso)
    cutoff = as_of_date - pd.Timedelta(days=horizon)
    training = signals_df[
        (signals_df["horizon_days"] == horizon)
        & (signals_df["disclosure_date"] <= cutoff)
    ].copy()
    # Defense in depth for price datasets that end before the requested
    # horizon.  Such rows are censored observations, not realized outcomes.
    if "total_spy_alpha_pct" in training.columns:
        training = training[training["total_spy_alpha_pct"].notna()]
    if training_lookback_iso is not None:
        training_start = pd.Timestamp(training_lookback_iso)
        training = training[training["disclosure_date"] >= training_start]
    return training


@df_memoize(copy=False)
def _filter_recent_trades(
    transactions_df: pd.DataFrame,
    lookback_days: int,
    as_of_iso: str,
) -> pd.DataFrame:
    """Filter transactions to the recent-trade window (Purchase only)."""
    as_of_date = pd.Timestamp(as_of_iso)
    lookback_start = as_of_date - pd.Timedelta(days=lookback_days)
    mask = (
        (transactions_df["disclosure_date"] >= lookback_start)
        & (transactions_df["disclosure_date"] < as_of_date)
        & (transactions_df["transaction_type"] == TransactionType.PURCHASE.value)
    )
    return transactions_df[mask].copy()


@df_memoize(copy=False)
def _filter_ticker_perf(
    signals_df: pd.DataFrame,
    horizon: int,
    as_of_iso: str,
) -> pd.DataFrame:
    """Filter signals to the ticker-performance window. Result is id-stable."""
    as_of_date = pd.Timestamp(as_of_iso)
    cutoff = as_of_date - pd.Timedelta(days=horizon)
    result = signals_df[
        (signals_df["horizon_days"] == horizon)
        & (signals_df["disclosure_date"] <= cutoff)
    ].copy()
    if "total_spy_alpha_pct" in result.columns:
        result = result[result["total_spy_alpha_pct"].notna()]
    return result
