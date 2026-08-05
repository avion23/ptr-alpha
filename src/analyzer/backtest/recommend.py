"""Generate backtest recommendations: score candidate tickers and rank them.

For each `as_of_date`, this:
  1. Filters signals/transactions into training + recent windows
  2. Builds member rankings on the training window
  3. Scores each candidate ticker with member-aware + signal-features
  4. Computes OU entry value V(0) + optimal horizon per ticker
  5. Applies disclosure-lag weight and crash-hazard penalty
  6. Returns the top-N by final signal score
"""

from __future__ import annotations

import pandas as pd

from analyzer.backtest.filters import (
    _filter_recent_trades,
    _filter_ticker_perf,
    _filter_training,
)
from analyzer.backtest.ou_params import _compute_ticker_ou_params
from analyzer.exceptions import AnalysisError
from analyzer.member_ranking import (
    _build_ranking_dicts,
    rank_members,
    score_ticker_by_buyers,
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
    prices_df: pd.DataFrame | None = None,
    training_lookback_days: int | None = None,
    solo_buyer_skill_threshold: float = 0.60,
    scoring_mode: str = "shrunk_alpha",
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

    training = _filter_training(signals_df, horizon, as_of_iso, training_lookback_iso)
    member_rankings = _build_member_rankings(training, horizon, threshold, bayes)
    if member_rankings is None or member_rankings.empty:
        return pd.DataFrame()

    recent_trades = _filter_recent_trades(transactions_df, lookback_days, as_of_iso)
    if recent_trades.empty:
        return pd.DataFrame()

    candidate_tickers = _candidate_tickers(recent_trades, min_buyers)
    if not candidate_tickers:
        return pd.DataFrame()

    return _score_and_rank(
        signals_df, prices_df, training, member_rankings, recent_trades,
        candidate_tickers, as_of_date, horizon, threshold, min_buyers,
        top_n, scoring_mode, bayes, solo_buyer_skill_threshold,
    )


def _build_member_rankings(training, horizon, threshold, bayes):
    if training.empty:
        return None
    try:
        return rank_members(training, horizon, threshold, _bayes_prior_strength=bayes)
    except AnalysisError:
        return None


def _candidate_tickers(recent_trades: pd.DataFrame, min_buyers: int) -> list:
    buyer_counts = recent_trades.groupby("ticker")["member"].nunique()
    return buyer_counts[buyer_counts >= min_buyers].index.tolist()


def _score_and_rank(
    signals_df, prices_df, training, member_rankings, recent_trades,
    candidate_tickers, as_of_date, horizon, threshold, min_buyers,
    top_n, scoring_mode, bayes, solo_buyer_skill_threshold,
) -> pd.DataFrame:
    as_of_for_features = (
        as_of_date.date() if hasattr(as_of_date, "date") else as_of_date
    )

    _ranking_dicts = _build_ranking_dicts(member_rankings, scoring_mode=scoring_mode)

    has_prices = prices_df is not None
    price_cols = set(prices_df.columns) if has_prices else set()

    inst_map, amt_map = _build_metadata_maps(recent_trades, has_prices)

    ticker_perf_signals = _filter_ticker_perf(signals_df, horizon, as_of_date.isoformat())
    recent_by_ticker = {t: grp for t, grp in recent_trades.groupby("ticker")}

    scores = []
    for ticker in candidate_tickers:
        row = _score_one_ticker(
            ticker=ticker,
            recent_trades=recent_trades, training=training,
            member_rankings=member_rankings, ticker_perf_signals=ticker_perf_signals,
            horizon=horizon, threshold=threshold, min_buyers=min_buyers,
            bayes=bayes, solo_buyer_skill_threshold=solo_buyer_skill_threshold,
            _ranking_dicts=_ranking_dicts,
            signals_df=signals_df, prices_df=prices_df, as_of_date=as_of_date,
            has_prices=has_prices, price_cols=price_cols,
            as_of_for_features=as_of_for_features,
            ticker_recent=recent_by_ticker.get(ticker),
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
    result = result.sort_values("signal_score", ascending=False).head(top_n).reset_index(drop=True)
    result.insert(0, "rank", range(1, len(result) + 1))

    # Propagate instrument_type and amount_midpoint from recent trades so
    # evaluate_backtest can apply options leverage. Maps were precomputed
    # above, outside the ticker loop.
    if inst_map:
        result["instrument_type"] = result["ticker"].map(inst_map).fillna("stock")
    if amt_map:
        result["amount_midpoint"] = result["ticker"].map(amt_map)

    return result


def _build_metadata_maps(recent_trades: pd.DataFrame, has_prices: bool) -> tuple[dict, dict]:
    """Precompute {ticker: instrument_type} and {ticker: amount_midpoint} maps
    so evaluate_backtest can apply options leverage without re-scanning."""
    inst_map: dict = {}
    amt_map: dict = {}
    if has_prices and "instrument_type" in recent_trades.columns:
        inst_map = (
            recent_trades.drop_duplicates("ticker")
            .set_index("ticker")["instrument_type"]
            .to_dict()
        )
    if has_prices and "amount_midpoint" in recent_trades.columns:
        amt_map = (
            recent_trades.drop_duplicates("ticker")
            .set_index("ticker")["amount_midpoint"]
            .to_dict()
        )
    return inst_map, amt_map


def _score_one_ticker(
    *,
    ticker, recent_trades, training, member_rankings, ticker_perf_signals,
    horizon, threshold, min_buyers, bayes, solo_buyer_skill_threshold,
    _ranking_dicts, signals_df, prices_df, as_of_date,
    has_prices, price_cols, as_of_for_features, ticker_recent,
) -> dict | None:
    score_df = score_ticker_by_buyers(
        ticker, recent_trades, training, horizon, threshold,
        member_rankings, min_buyers,
        ticker_perf_signals=ticker_perf_signals,
        _bayes_prior_strength=bayes,
        solo_buyer_skill_threshold=solo_buyer_skill_threshold,
        _ranking_dicts=_ranking_dicts,
    )
    if score_df.empty:
        return None

    row = {c: score_df[c].iloc[0] for c in score_df.columns}

    if has_prices and ticker in price_cols:
        v0, optimal_h = _compute_ticker_ou_params(
            ticker, signals_df, prices_df, as_of_date, horizon,
        )
        row["ou_entry_value"] = round(v0, 4) if v0 is not None else None
        row["optimal_horizon"] = optimal_h
    else:
        row["ou_entry_value"] = None
        row["optimal_horizon"] = horizon

    if ticker_recent is not None and has_prices:
        _apply_features_to_row(
            row, ticker, ticker_recent, prices_df, recent_trades,
            as_of_for_features,
        )

    return row


def _apply_features_to_row(row, ticker, ticker_recent, prices_df, recent_trades, as_of_for_features):
    """Compute signal features + crash hazard, apply lag/crash adjustments."""
    from analyzer.signal_features import (
        compute_signal_features,
        compute_disclosure_lag_weight,
        estimate_crash_hazard,
    )

    latest_idx = ticker_recent["disclosure_date"].idxmax()
    latest_tx = ticker_recent.loc[latest_idx]
    tx_date = latest_tx.get("transaction_date")
    if tx_date is not None:
        tx_date = pd.Timestamp(tx_date).date()
    disc_date = pd.Timestamp(latest_tx["disclosure_date"]).date()

    features = compute_signal_features(
        ticker=ticker,
        disclosure_date=disc_date,
        transaction_date=tx_date,
        prices_df=prices_df,
        all_tx=recent_trades,
        as_of_date=as_of_for_features,
    )
    crash = estimate_crash_hazard(features)

    lag_weight = compute_disclosure_lag_weight(features.lag_days)
    base_score = float(row.get("signal_score", 0))
    adjusted_score = base_score * lag_weight

    if 0.0 <= crash.crash_prob <= 1.0:
        adjusted_score *= (1 - crash.crash_prob)

    row["signal_score"] = round(adjusted_score, 2)
    row["lag_days"] = features.lag_days
    row["lag_weight"] = round(lag_weight, 4)
    row["crash_prob"] = crash.crash_prob
    row["crash_var_95"] = crash.var_95
    row["volatility_20d"] = round(features.volatility_20d, 4)
    row["drawdown_from_ath"] = round(features.drawdown_from_ath, 4)
