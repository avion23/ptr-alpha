"""Backtest evaluation and recommendation scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd

from analyzer.exceptions import AnalysisError
from analyzer.models import TransactionType
from analyzer.signals import _price_at_or_before, _price_on_or_before
from analyzer.member_ranking import rank_members, score_ticker_by_buyers


def _compute_ticker_entry_value(
    ticker: str,
    signals_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    as_of_date: pd.Timestamp,
    horizon: int,
    rho: float = 0.000137,
) -> float | None:
    """Compute OU entry value V(0) for a ticker from historical return curves.

    Falls back to global prior (average across all tickers) if ticker has
    no prior disclosure history.
    """
    from analyzer.return_process import compute_entry_value, fit_ou

    ticker_col = ticker in prices_df.columns if hasattr(prices_df, 'columns') else False

    # Get all prior fully-elapsed purchase signals for this ticker
    eligible = signals_df[
        (signals_df["ticker"] == ticker)
        & (signals_df["horizon_days"] == horizon)
        & (signals_df["disclosure_date"] <= as_of_date - pd.Timedelta(days=horizon))
        & (signals_df["signal_type"] == TransactionType.PURCHASE.value)
    ].copy()

    curves = []
    if ticker_col and not eligible.empty:
        ticker_prices = prices_df[ticker].dropna()
        for _, row in eligible.iterrows():
            disc_date = row["disclosure_date"]
            entry_price = row.get("entry_price")
            if not entry_price or entry_price <= 0:
                continue
            end_date = disc_date + pd.Timedelta(days=horizon)
            window = ticker_prices[
                (ticker_prices.index >= disc_date) & (ticker_prices.index <= end_date)
            ]
            if len(window) < 3:
                continue
            curve = (window.values / entry_price) - 1.0
            curves.append(curve)

    if curves:
        v0, _mu, _theta = compute_entry_value(curves, rho=rho)
        return v0

    # Fallback: global prior from historical curves (filtered by cutoff to avoid look-ahead)
    all_purchases = signals_df[
        (signals_df["horizon_days"] == horizon)
        & (signals_df["disclosure_date"] <= as_of_date - pd.Timedelta(days=horizon))
        & (signals_df["signal_type"] == TransactionType.PURCHASE.value)
    ].copy()

    if all_purchases.empty or not ticker_col:
        return None

    ticker_prices = prices_df[ticker].dropna()
    global_curves = []
    for _, row in all_purchases.iterrows():
        disc_date = row["disclosure_date"]
        entry_price = row.get("entry_price")
        tkr = row["ticker"]
        if not entry_price or entry_price <= 0 or tkr not in prices_df.columns:
            continue
        tkr_prices = prices_df[tkr].dropna()
        end_date = disc_date + pd.Timedelta(days=horizon)
        window = tkr_prices[
            (tkr_prices.index >= disc_date) & (tkr_prices.index <= end_date)
        ]
        if len(window) < 3:
            continue
        curve = (window.values / entry_price) - 1.0
        global_curves.append(curve)

    if not global_curves:
        return None

    v0, _mu, _theta = compute_entry_value(global_curves, rho=rho)
    return v0


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
) -> pd.DataFrame:
    elapsed_cutoff = as_of_date - pd.Timedelta(days=horizon)
    training = signals_df[
        (signals_df["horizon_days"] == horizon)
        & (signals_df["disclosure_date"] <= elapsed_cutoff)
    ].copy()

    if training_lookback_days is not None:
        training_start = as_of_date - pd.Timedelta(days=training_lookback_days)
        training = training[training["disclosure_date"] >= training_start]

    member_rankings = None
    if not training.empty:
        try:
            member_rankings = rank_members(training, horizon, threshold)
        except AnalysisError:
            member_rankings = None

    if member_rankings is None or member_rankings.empty:
        return pd.DataFrame()

    lookback_start = as_of_date - pd.Timedelta(days=lookback_days)
    recent_mask = (
        (transactions_df["disclosure_date"] >= lookback_start)
        & (transactions_df["disclosure_date"] <= as_of_date)
        & (transactions_df["transaction_type"] == TransactionType.PURCHASE.value)
    )
    recent_trades = transactions_df[recent_mask].copy()

    if recent_trades.empty:
        return pd.DataFrame()

    buyer_counts = recent_trades.groupby("ticker")["member"].nunique()
    candidate_tickers = buyer_counts[buyer_counts >= min_buyers].index.tolist()

    if not candidate_tickers:
        return pd.DataFrame()

    ticker_perf_signals = signals_df[
        (signals_df["horizon_days"] == horizon)
        & (signals_df["disclosure_date"] <= elapsed_cutoff)
    ].copy()

    from analyzer.signal_features import (
        compute_signal_features,
        compute_disclosure_lag_weight,
        estimate_crash_hazard,
    )

    scores = []
    for ticker in candidate_tickers:
        try:
            score = score_ticker_by_buyers(
                ticker, recent_trades, training, horizon, threshold,
                member_rankings, min_buyers,
                ticker_perf_signals=ticker_perf_signals,
            )

            # Compute OU entry value V(0) if prices available
            if prices_df is not None:
                v0 = _compute_ticker_entry_value(
                    ticker, signals_df, prices_df, as_of_date, horizon,
                )
                score["ou_entry_value"] = round(v0, 4) if v0 is not None else None

            # Compute signal features and crash hazard for this ticker
            # Use the most recent transaction date for this ticker
            ticker_recent = recent_trades[recent_trades["ticker"] == ticker]
            if not ticker_recent.empty and prices_df is not None:
                latest_tx = ticker_recent.sort_values("disclosure_date").iloc[-1]
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
                    as_of_date=as_of_date.date() if hasattr(as_of_date, 'date') else as_of_date,
                )
                crash = estimate_crash_hazard(features)

                # Apply lag weight
                lag_weight = compute_disclosure_lag_weight(features.lag_days)
                base_score = float(score["signal_score"].iloc[0])
                adjusted_score = base_score * lag_weight

                # Apply crash penalty
                adjusted_score *= (1 - crash.crash_prob)

                score["signal_score"] = round(adjusted_score, 2)
                score["lag_days"] = features.lag_days
                score["lag_weight"] = round(lag_weight, 4)
                score["crash_prob"] = crash.crash_prob
                score["crash_var_95"] = crash.var_95
                score["volatility_20d"] = round(features.volatility_20d, 4)
                score["drawdown_from_ath"] = round(features.drawdown_from_ath, 4)

            scores.append(score)
        except AnalysisError:
            continue

    if not scores:
        return pd.DataFrame()

    result = pd.concat(scores, ignore_index=True)
    result = result.sort_values("signal_score", ascending=False).head(top_n).reset_index(drop=True)
    result.insert(0, "rank", range(1, len(result) + 1))
    return result


def evaluate_backtest(
    recommendations: pd.DataFrame,
    prices_df: pd.DataFrame,
    as_of_date: pd.Timestamp,
    horizon: int,
    max_staleness_days: int | None = 30,
) -> pd.DataFrame:
    if recommendations.empty:
        return recommendations

    exit_date = as_of_date + pd.Timedelta(days=horizon)

    spy_start = _price_at_or_before(prices_df, "SPY", as_of_date, max_staleness_days)
    spy_end = _price_on_or_before(prices_df, "SPY", exit_date)
    if not spy_start or not spy_end:
        raise AnalysisError(
            f"SPY price not available for backtest period "
            f"(as_of={as_of_date.date()}, exit={exit_date.date()})"
        )
    spy_return_pct = (spy_end / spy_start - 1) * 100

    rows = []
    for _, rec in recommendations.iterrows():
        ticker = rec["ticker"]
        entry = _price_at_or_before(prices_df, ticker, as_of_date, max_staleness_days)
        exit_price = _price_on_or_before(prices_df, ticker, exit_date)
        if not entry or not exit_price:
            continue
        return_pct = (exit_price / entry - 1) * 100
        alpha_pct = return_pct - spy_return_pct
        rows.append({
            "ticker": ticker,
            "bt_entry_price": round(entry, 2),
            "bt_exit_price": round(exit_price, 2),
            "bt_return_pct": round(return_pct, 2),
            "bt_spy_return_pct": round(spy_return_pct, 2),
            "bt_alpha_pct": round(alpha_pct, 2),
        })

    eval_df = pd.DataFrame(rows)
    if eval_df.empty:
        recommendations = recommendations.copy()
        for col in ["bt_entry_price", "bt_exit_price", "bt_return_pct", "bt_spy_return_pct", "bt_alpha_pct"]:
            recommendations[col] = None
        return recommendations
    return recommendations.merge(eval_df, on="ticker", how="left")


def summarize_backtest(results: pd.DataFrame) -> pd.DataFrame:
    valid = results.dropna(subset=["bt_return_pct"])
    if valid.empty:
        return pd.DataFrame()

    by_rank = []
    for rank, grp in valid.groupby("rank"):
        by_rank.append({
            "rank": rank,
            "count": len(grp),
            "win_rate_pct": round((grp["bt_return_pct"] > 0).mean() * 100, 1),
            "avg_return_pct": round(grp["bt_return_pct"].mean(), 2),
            "avg_alpha_pct": round(grp["bt_alpha_pct"].mean(), 2) if "bt_alpha_pct" in grp.columns else None,
        })

    summary = pd.DataFrame(by_rank)

    overall = {
        "rank": "ALL",
        "count": len(valid),
        "win_rate_pct": round((valid["bt_return_pct"] > 0).mean() * 100, 1),
        "avg_return_pct": round(valid["bt_return_pct"].mean(), 2),
        "avg_alpha_pct": round(valid["bt_alpha_pct"].mean(), 2) if "bt_alpha_pct" in valid.columns else None,
    }
    summary = pd.concat([summary, pd.DataFrame([overall])], ignore_index=True)
    return summary
