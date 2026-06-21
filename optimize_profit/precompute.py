"""Per-as_of_date precomputation shared across scoring-function combinations.

Returns {as_of_iso: {training, member_rankings, recent_trades,
candidate_tickers_by_min_buyers, ticker_perf_signals, as_of_ts, horizon}}.

This is the expensive part of the sweep — `rank_members` and the signal
filters are memoized, so once a combo's params match a previous run we
hit cache instead of recomputing.
"""

import pandas as pd

from analyzer.backtest import (
    _filter_recent_trades,
    _filter_ticker_perf,
    _filter_training,
)
from analyzer.member_ranking import rank_members


def precompute_walk_forward_data(
    signals_df,
    transactions_df,
    prices_df,
    as_of_dates,
    horizon,
    lookback_days,
    training_lookback_days,
    min_buyers_list,
):
    """Precompute per-as_of_date data shared across all scoring combos.

    Returns {as_of_iso: {training, member_rankings, recent_trades, ...}}.
    """
    precomputed = {}

    for as_of in as_of_dates:
        as_of_ts = pd.Timestamp(as_of)
        as_of_iso = as_of_ts.isoformat()

        training = _filter_training_for_period(
            signals_df, horizon, as_of_iso, training_lookback_days,
        )
        if training.empty:
            continue

        member_rankings = _rank_members_for_period(training, horizon)
        if member_rankings is None or member_rankings.empty:
            continue

        recent_trades = _filter_recent_trades_for_period(
            transactions_df, lookback_days, as_of_iso,
        )
        if recent_trades.empty:
            continue

        candidate_tickers_by_mb = _candidate_tickers_by_min_buyers(
            recent_trades, min_buyers_list,
        )
        ticker_perf_signals = _filter_ticker_perf_for_period(
            signals_df, horizon, as_of_iso,
        )

        precomputed[as_of_iso] = {
            "training": training,
            "member_rankings": member_rankings,
            "recent_trades": recent_trades,
            "candidate_tickers": candidate_tickers_by_mb,
            "ticker_perf_signals": ticker_perf_signals,
            "as_of_ts": as_of_ts,
            "horizon": horizon,
        }

    return precomputed


def _filter_training_for_period(signals_df, horizon, as_of_iso, training_lookback_days):
    as_of_ts = pd.Timestamp(as_of_iso)
    training_lookback_iso = (
        (as_of_ts - pd.Timedelta(days=training_lookback_days)).isoformat()
    )
    return _filter_training(signals_df, horizon, as_of_iso, training_lookback_iso)


def _rank_members_for_period(training, horizon):
    try:
        return rank_members(training, horizon, 5.0)
    except Exception:
        return None


def _filter_recent_trades_for_period(transactions_df, lookback_days, as_of_iso):
    return _filter_recent_trades(transactions_df, lookback_days, as_of_iso)


def _candidate_tickers_by_min_buyers(recent_trades: pd.DataFrame, min_buyers_list) -> dict:
    """For each min_buyers threshold, return the list of tickers whose
    buyer-count meets the threshold. Pre-building all thresholds in one pass
    avoids repeating the buyer_counts groupby."""
    buyer_counts = recent_trades.groupby("ticker")["member"].nunique()
    return {
        mb: buyer_counts[buyer_counts >= mb].index.tolist()
        for mb in min_buyers_list
    }


def _filter_ticker_perf_for_period(signals_df, horizon, as_of_iso):
    return _filter_ticker_perf(signals_df, horizon, as_of_iso)
