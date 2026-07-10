"""Analysis facade — re-exports from focused modules for backward compatibility."""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

from analyzer.models import TransactionType  # noqa: F401 — re-exported for backward compat

from analyzer.signals import (  # noqa: F401
    DECAY_LAMBDA,
    POSITION_SIZE_BASELINE,
    MAX_DISCLOSURE_METADATA_ADJUSTMENT,
    BAYES_PRIOR_STRENGTH,
    BUYER_RECENCY_DECAY,
    TICKER_PERF_MIN_TRADES,
    MIN_ENTRY_PRICE,
    CONVICTION_WEIGHT_ALPHA,
    CONVICTION_WEIGHT_REALIZED,
    _price_at_or_before,
    _price_at_or_near,
    _price_on_or_before,
    _get_horizon_data,
    _apply_quality_filter,
    _compute_dynamic_prior,
    _assign_episode_ids,
    _collapse_to_episodes,
    _get_top_signals,
    _get_member_signals,
    calculate_signal_potential,
    compute_signal_potential_with_member_decay,
    get_top_signals,
    get_member_signals,
)

from analyzer.member_ranking import (  # noqa: F401
    bayesian_win_probability,
    bayes_factor_against_market,
    _size_score_factor,
    _owner_score_factor,
    _conviction_score,
    _compute_ticker_member_performance,
    _compute_member_stats,
    _build_ranking_dicts,
    rank_members,
    rank_sales,
    score_ticker_by_buyers,
    get_ticker_buyers_with_rankings,
    estimate_member_decay_lambda,
    get_member_decay_map,
)

from analyzer.backtest import (  # noqa: F401
    _compute_ticker_entry_value,
    _compute_ticker_optimal_horizon,
    backtest_recommendations,
    evaluate_backtest,
    summarize_backtest,
)

from analyzer.sector_data import load_sector_data
from analyzer.exceptions import AnalysisError


def analyze_by_sector(
    trades: pd.DataFrame, signals: pd.DataFrame, horizons: tuple[int, ...]
) -> pd.DataFrame | None:
    tickers = trades['ticker'].unique()
    sectors = load_sector_data(tickers.tolist())
    if sectors.empty:
        return None

    sig_with_sector = signals.merge(sectors, on="ticker", how="left")

    results = []
    for sector in sectors["sector"].unique():
        sector_purchases = sig_with_sector[
            (sig_with_sector["sector"] == sector) &
            (sig_with_sector["signal_type"] == TransactionType.PURCHASE.value)
        ]
        if len(sector_purchases) < 3:
            continue
        try:
            ranked = rank_members(sector_purchases, horizons[0])
            if not ranked.empty:
                results.append({
                    "sector": sector,
                    "top_member": ranked.iloc[0]["member"],
                    "top_member_alpha": ranked.iloc[0]["avg_spy_alpha_pct"],
                    "num_trades": len(sector_purchases),
                    "num_members": sector_purchases["member"].nunique(),
                })
        except AnalysisError as e:
            logger.debug("Skipping sector %s: %s", sector, e)
            continue

    if not results:
        return None
    return pd.DataFrame(results).sort_values("top_member_alpha", ascending=False)


def get_analysis_table(
    signals_df,
    member_filter,
    show_signals,
    horizon,
    top_n,
    threshold,
):
    if member_filter:
        return _get_member_signals(signals_df, member_filter, horizon, top_n or 5)
    if show_signals:
        return _get_top_signals(signals_df, horizon, top_n or 15)
    return rank_members(signals_df, horizon, threshold).head(top_n)
