"""Analysis entry points shared by the CLI and replay pipeline."""

from __future__ import annotations

import logging

import pandas as pd

from analyzer.backtest import (
    backtest_recommendations,
    evaluate_backtest,
    summarize_backtest,
)
from analyzer.exceptions import AnalysisError
from analyzer.member_ranking.buyer_scoring import score_ticker_by_buyers
from analyzer.member_ranking.ranking import rank_members
from analyzer.models import AnalysisMode, TransactionType
from analyzer.sector_data import load_sector_data
from analyzer.signals import (
    _get_member_signals,
    _get_top_signals,
    calculate_signal_potential,
    get_member_signals,
    get_top_signals,
)

logger = logging.getLogger(__name__)


def analyze_by_sector(
    trades: pd.DataFrame, signals: pd.DataFrame, horizons: tuple[int, ...]
) -> pd.DataFrame | None:
    tickers = trades["ticker"].unique()
    sectors = load_sector_data(tickers.tolist())
    if sectors.empty:
        return None

    sig_with_sector = signals.merge(sectors, on="ticker", how="left")
    results = []
    for sector in sectors["sector"].unique():
        purchases = sig_with_sector[
            (sig_with_sector["sector"] == sector)
            & (sig_with_sector["signal_type"] == TransactionType.PURCHASE.value)
        ]
        if len(purchases) < 3:
            continue
        try:
            ranked = rank_members(purchases, horizons[0])
        except AnalysisError as exc:
            logger.debug("Skipping sector %s: %s", sector, exc)
            continue
        if ranked.empty:
            continue
        results.append(
            {
                "sector": sector,
                "top_member": ranked.iloc[0]["member"],
                "top_member_alpha_pct": ranked.iloc[0]["avg_spy_alpha_pct"],
                "num_purchase_rows": len(purchases),
                "num_members": purchases["member"].nunique(),
            }
        )

    if not results:
        return None
    return pd.DataFrame(results).sort_values(
        ["top_member_alpha_pct", "sector"], ascending=[False, True]
    )


def get_analysis_table(
    signals_df: pd.DataFrame,
    mode: AnalysisMode,
    member_filter: str | None,
    horizon: int,
    top_n: int | None,
) -> pd.DataFrame:
    match mode:
        case AnalysisMode.MEMBER_SIGNALS:
            if member_filter is None:
                raise ValueError("member_filter is required for member signals")
            return _get_member_signals(signals_df, member_filter, horizon, top_n or 5)
        case AnalysisMode.TOP_SIGNALS:
            return _get_top_signals(signals_df, horizon, top_n or 15)
        case AnalysisMode.MEMBER_RANKINGS:
            return rank_members(signals_df, horizon).head(top_n)
        case _:
            raise ValueError(f"Unsupported analysis mode: {mode}")


__all__ = [
    "analyze_by_sector",
    "backtest_recommendations",
    "calculate_signal_potential",
    "evaluate_backtest",
    "get_analysis_table",
    "get_member_signals",
    "get_top_signals",
    "rank_members",
    "score_ticker_by_buyers",
    "summarize_backtest",
]
