"""Causal per-period inputs shared by optimizer configurations."""

from __future__ import annotations

import pandas as pd

from analyzer.backtest import (
    _filter_recent_trades,
    _filter_ticker_perf,
    _filter_training,
)
from analyzer.exceptions import AnalysisError
from analyzer.member_names import canonical_member_key
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
    bayes_prior_strength=None,
):
    """Return every requested period, including explicit rejection records.

    Expected data insufficiency is recorded in ``status``/``reason``. Unexpected
    failures propagate; they are never converted into apparently valid coverage.
    """
    del prices_df  # retained in the public signature for compatibility
    precomputed: dict[str, dict] = {}

    for as_of in as_of_dates:
        as_of_ts = pd.Timestamp(as_of)
        as_of_iso = as_of_ts.isoformat()
        base = {"as_of_ts": as_of_ts, "horizon": horizon}

        training = _filter_training_for_period(
            signals_df, horizon, as_of_iso, training_lookback_days
        )
        if training.empty:
            precomputed[as_of_iso] = {
                **base,
                "status": "rejected",
                "reason": "empty_training",
            }
            continue

        try:
            member_rankings = rank_members(
                training,
                horizon,
                5.0,
                _bayes_prior_strength=bayes_prior_strength,
            )
        except AnalysisError as exc:
            precomputed[as_of_iso] = {
                **base,
                "status": "rejected",
                "reason": "ranking_unavailable",
                "detail": str(exc),
            }
            continue
        if member_rankings.empty:
            precomputed[as_of_iso] = {
                **base,
                "status": "rejected",
                "reason": "empty_rankings",
            }
            continue

        recent_trades = _filter_recent_trades_for_period(
            transactions_df, lookback_days, as_of_iso
        )
        if recent_trades.empty:
            precomputed[as_of_iso] = {
                **base,
                "status": "rejected",
                "reason": "empty_recent_trades",
            }
            continue

        precomputed[as_of_iso] = {
            **base,
            "status": "ready",
            "training": training,
            "member_rankings": member_rankings,
            "recent_trades": recent_trades,
            "candidate_tickers": _candidate_tickers_by_min_buyers(
                recent_trades, min_buyers_list
            ),
            "ticker_perf_signals": _filter_ticker_perf_for_period(
                signals_df, horizon, as_of_iso
            ),
        }

    return precomputed


def _filter_training_for_period(signals_df, horizon, as_of_iso, training_lookback_days):
    as_of_ts = pd.Timestamp(as_of_iso)
    training_lookback_iso = (
        as_of_ts - pd.Timedelta(days=training_lookback_days)
    ).isoformat()
    return _filter_training(signals_df, horizon, as_of_iso, training_lookback_iso)


def _filter_recent_trades_for_period(transactions_df, lookback_days, as_of_iso):
    return _filter_recent_trades(transactions_df, lookback_days, as_of_iso)


def _candidate_tickers_by_min_buyers(
    recent_trades: pd.DataFrame, min_buyers_list
) -> dict:
    canonical = recent_trades.assign(
        _member_canonical=recent_trades["member"].map(canonical_member_key)
    )
    buyer_counts = canonical.groupby("ticker")["_member_canonical"].nunique()
    return {
        threshold: buyer_counts[buyer_counts >= threshold].index.tolist()
        for threshold in min_buyers_list
    }


# Compatibility helpers retained for callers that import private symbols.
def _rank_members_for_period(training, horizon, bayes_prior_strength=None):
    try:
        return rank_members(
            training, horizon, 5.0, _bayes_prior_strength=bayes_prior_strength
        )
    except AnalysisError:
        return None


def _filter_ticker_perf_for_period(signals_df, horizon, as_of_iso):
    return _filter_ticker_perf(signals_df, horizon, as_of_iso)
