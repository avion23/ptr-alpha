"""Generate backtest recommendations: score candidate tickers and rank them.

For each `as_of_date`, this:
  1. Filters signals/transactions into training + recent windows
  2. Builds member rankings only for explicit historical scoring modes
  3. Scores candidates with the same scorer used by live ticker analysis
  4. Returns the top-N by signal score
"""

from __future__ import annotations

import pandas as pd

from analyzer.backtest.filters import (
    _filter_recent_trades,
    _filter_ticker_perf,
    _filter_training,
)
from analyzer.exceptions import AnalysisError
from analyzer.member_ranking import (
    _build_ranking_dicts,
    rank_members,
    score_ticker_by_buyers,
)
from analyzer.member_ranking.buyer_scoring import (
    _filter_equity_rows,
    _get_consensus_candidate_tickers,
)


def backtest_recommendations(
    signals_df: pd.DataFrame,
    transactions_df: pd.DataFrame,
    as_of_date: pd.Timestamp,
    horizon: int = 90,
    lookback_days: int = 60,
    min_buyers: int = 2,
    top_n: int = 10,
    threshold: float = 5.0,
    training_lookback_days: int | None = None,
    scoring_mode: str = "consensus",
    bayes_prior_strength: float | None = None,
) -> pd.DataFrame:
    if bayes_prior_strength is None:
        from analyzer.signals import BAYES_PRIOR_STRENGTH

        bayes_prior_strength = BAYES_PRIOR_STRENGTH
    bayes = bayes_prior_strength

    as_of_iso = as_of_date.isoformat()
    training_lookback_iso = (
        (as_of_date - pd.Timedelta(days=training_lookback_days)).isoformat()
        if training_lookback_days is not None
        else None
    )

    is_consensus = scoring_mode == "consensus"
    if not is_consensus:
        signals_df = _filter_equity_rows(signals_df)
    if not is_consensus and signals_df.empty:
        return pd.DataFrame()
    training = (
        pd.DataFrame()
        if is_consensus
        else _filter_training(signals_df, horizon, as_of_iso, training_lookback_iso)
    )
    member_rankings = (
        pd.DataFrame()
        if is_consensus
        else _build_member_rankings(training, horizon, threshold, bayes)
    )
    if not is_consensus and (member_rankings is None or member_rankings.empty):
        return pd.DataFrame()

    recent_trades = _filter_recent_trades(transactions_df, lookback_days, as_of_iso)
    if recent_trades.empty:
        return pd.DataFrame()

    candidates = _get_consensus_candidate_tickers(
        recent_trades,
        min_buyers,
        as_of_date=as_of_date,
    )
    if not candidates:
        return pd.DataFrame()
    if not is_consensus:
        recent_trades = _filter_equity_rows(recent_trades)

    return _score_and_rank(
        signals_df,
        training,
        member_rankings,
        recent_trades,
        candidates,
        as_of_date,
        horizon,
        threshold,
        min_buyers,
        top_n,
        scoring_mode,
        bayes,
    )


def _build_member_rankings(training, horizon, threshold, bayes):
    if training.empty:
        return None
    try:
        return rank_members(training, horizon, threshold, _bayes_prior_strength=bayes)
    except AnalysisError:
        return None



def _score_and_rank(
    signals_df,
    training,
    member_rankings,
    recent_trades,
    candidate_tickers,
    as_of_date,
    horizon,
    threshold,
    min_buyers,
    top_n,
    scoring_mode,
    bayes,
) -> pd.DataFrame:
    _ranking_dicts = _build_ranking_dicts(member_rankings, scoring_mode=scoring_mode)
    metadata_maps = (
        {} if scoring_mode == "consensus" else _build_metadata_maps(recent_trades)
    )

    ticker_perf_signals = (
        pd.DataFrame()
        if scoring_mode == "consensus" or signals_df.empty
        else _filter_ticker_perf(signals_df, horizon, as_of_date.isoformat())
    )
    scores = []
    for ticker in candidate_tickers:
        row = _score_one_ticker(
            ticker=ticker,
            recent_trades=recent_trades,
            training=training,
            member_rankings=member_rankings,
            ticker_perf_signals=ticker_perf_signals,
            horizon=horizon,
            threshold=threshold,
            min_buyers=min_buyers,
            bayes=bayes,
            scoring_mode=scoring_mode,
            _ranking_dicts=_ranking_dicts,
            signals_df=signals_df,
            as_of_date=as_of_date,
        )
        if row is not None:
            scores.append(row)

    if not scores:
        return pd.DataFrame()

    result = pd.DataFrame(scores)
    # Drop rejected rows. score_ticker_by_buyers emits a zero signal_score
    # (with a `note`) for tickers that fail the min-buyers / solo-buyer skill
    # gate; those should not surface as recommendations.
    if "signal_score" in result.columns:
        result = result[result["signal_score"].fillna(0) > 0]
    if result.empty:
        return pd.DataFrame()
    result = (
        result.sort_values(
            ["signal_score", "ticker"], ascending=[False, True]
        )
        .head(top_n)
        .reset_index(drop=True)
    )
    result.insert(0, "rank", range(1, len(result) + 1))

    if scoring_mode == "consensus":
        # The consensus eligibility boundary already rejects explicit
        # non-equities. Keep replay pricing on that exact semantic contract
        # rather than trying to copy one arbitrary source row's metadata onto
        # a multi-buyer equity identity.
        result["instrument_type"] = "stock"
    else:
        for column, values in metadata_maps.items():
            result[column] = result["ticker"].map(values)

    return result


def _build_metadata_maps(recent_trades: pd.DataFrame) -> dict[str, dict]:
    """Build per-ticker maps only when all non-null values agree."""
    maps: dict[str, dict] = {}
    for column in (
        "instrument_type",
        "amount_midpoint",
        "strike_price",
        "expiry_date",
        "asset_description",
        "raw_asset_description",
        "raw_asset_class",
        "ticker_origin",
        "source",
        "source_record_id",
        "source_row_id",
        "available_date",
        "notification_date",
    ):
        if column not in recent_trades.columns:
            continue
        values: dict = {}
        for ticker, group in recent_trades.groupby("ticker"):
            non_null = group[column].dropna().unique()
            values[ticker] = non_null[0] if len(non_null) == 1 else None
        maps[column] = values
    return maps


def _score_one_ticker(
    *,
    ticker,
    recent_trades,
    training,
    member_rankings,
    ticker_perf_signals,
    horizon,
    threshold,
    min_buyers,
    bayes,
    scoring_mode,
    _ranking_dicts,
    signals_df,
    as_of_date,
) -> dict | None:
    score_df = score_ticker_by_buyers(
        ticker,
        recent_trades,
        training,
        horizon,
        threshold,
        member_rankings,
        min_buyers,
        ticker_perf_signals=ticker_perf_signals,
        _bayes_prior_strength=bayes,
        _ranking_dicts=_ranking_dicts,
        scoring_mode=scoring_mode,
        as_of_date=as_of_date,
    )
    if score_df.empty:
        return None

    return {c: score_df[c].iloc[0] for c in score_df.columns}
