"""Generate point-in-time recommendations with the production consensus rule."""

from __future__ import annotations

import pandas as pd

from analyzer.member_ranking.buyer_scoring import (
    CONSENSUS_LOOKBACK_DAYS,
    CONSENSUS_MIN_BUYERS,
    _get_consensus_candidate_tickers,
    score_ticker_by_buyers,
)
from analyzer.models import TransactionType


def backtest_recommendations(
    transactions_df: pd.DataFrame,
    as_of_date: pd.Timestamp,
    lookback_days: int = CONSENSUS_LOOKBACK_DAYS,
    min_buyers: int = CONSENSUS_MIN_BUYERS,
    top_n: int = 10,
) -> pd.DataFrame:
    """Replay the same distinct-buyer decision rule used by live analysis."""
    if transactions_df.empty:
        return pd.DataFrame()

    as_of = pd.Timestamp(as_of_date).normalize()
    lookback_start = as_of - pd.Timedelta(days=lookback_days)
    disclosure_dates = pd.to_datetime(
        transactions_df["disclosure_date"], errors="coerce"
    )
    recent_trades = transactions_df[
        disclosure_dates.notna()
        & (disclosure_dates >= lookback_start)
        & (disclosure_dates <= as_of)
        & (transactions_df["transaction_type"] == TransactionType.PURCHASE.value)
    ].copy()
    if recent_trades.empty:
        return pd.DataFrame()

    candidates = _get_consensus_candidate_tickers(
        recent_trades,
        min_buyers,
        as_of_date=as_of,
    )
    if not candidates:
        return pd.DataFrame()

    scores = [
        score_ticker_by_buyers(
            ticker,
            recent_trades,
            min_buyers=min_buyers,
            as_of_date=as_of,
        )
        for ticker in candidates
    ]
    result = pd.concat(scores, ignore_index=True)
    result = result[result["signal_score"] > 0]
    if result.empty:
        return pd.DataFrame()

    result = (
        result.sort_values(["signal_score", "ticker"], ascending=[False, True])
        .head(top_n)
        .reset_index(drop=True)
    )
    result.insert(0, "rank", range(1, len(result) + 1))
    result["instrument_type"] = "stock"
    return result
