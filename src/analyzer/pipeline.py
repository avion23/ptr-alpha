from __future__ import annotations

from datetime import timedelta
from dataclasses import dataclass
from functools import wraps
import logging
import os
from pathlib import Path

import pandas as pd

from analyzer.exceptions import AnalyzerError, DataSourceError, AnalysisError
from analyzer.models import TransactionType
from analyzer import analysis

logger = logging.getLogger(__name__)


def _load_sector_data(tickers: list[str]) -> pd.DataFrame:
    """Load sector info for tickers using yfinance."""
    try:
        import yfinance as yf
        from concurrent.futures import ThreadPoolExecutor, as_completed
    except ImportError:
        return pd.DataFrame(columns=["ticker", "sector"])

    filtered = [t for t in tickers if t not in ("SPY", "SP500")]
    if not filtered:
        return pd.DataFrame(columns=["ticker", "sector"])

    def fetch_sector(ticker):
        try:
            return ticker, yf.Ticker(ticker).info.get("sector", "Unknown")
        except (OSError, ValueError, KeyError):
            return ticker, "Unknown"

    records = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(fetch_sector, t): t for t in filtered}
        for future in as_completed(futures):
            ticker, sector = future.result()
            records.append({"ticker": ticker, "sector": sector})

    return pd.DataFrame(records)



@dataclass
class AnalysisParams:
    source: str
    year: int
    horizons: list[int]
    threshold: float
    member_filter: str | None = None
    top_n: int | None = None
    show_signals: bool = False


@dataclass
class TickerScoringParams:
    year: int
    horizons: list[int]
    threshold: float = 5.0
    days_back: int = 28
    min_buyers: int = 2
    top_n: int = 15


@dataclass
class TickerAnalysisParams:
    ticker: str
    year: int
    horizon: int = 90
    threshold: float = 5.0

def pipeline_step(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except AnalyzerError:
            return False
    return wrapper

def _prepare_analysis_data(
    transaction_source, price_source, year: int, horizons: list[int]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trades = transaction_source.get_transactions(year)
    logger.info(f"Loaded {len(trades)} transactions for {year}")

    if len(trades) == 0:
        raise DataSourceError("No trading data found")

    start_date = trades['disclosure_date'].min() - timedelta(days=30)
    end_date = trades['disclosure_date'].max() + timedelta(days=max(horizons) + 10)

    prices = price_source.get_prices(trades['ticker'].unique(), start_date, end_date)
    logger.info(f"Fetched price data for {len(prices.columns)} tickers")

    all_tickers = trades['ticker'].unique().tolist()
    entry_prices = transaction_source.db.get_entry_prices(all_tickers, start_date, end_date)
    logger.info(f"Computed entry prices for {len(entry_prices)} transactions")

    signals = analysis.calculate_signal_potential(entry_prices, prices, horizons)
    logger.info(f"Calculated {len(signals)} signals")

    return trades, prices, signals

@pipeline_step
def run_fetch_pipeline(transaction_source, year: int) -> bool:
    transaction_source.fetch_and_cache_pdfs(year)
    logger.info(f"Successfully fetched PDFs for {year}")
    return True

@pipeline_step
def run_parse_pipeline(transaction_source, year: int) -> bool:
    transaction_source.parse_cached_pdfs(year)
    logger.info(f"Successfully parsed PDFs for {year}")
    return True

@pipeline_step
def run_analysis_pipeline(
    params: AnalysisParams, transaction_source, price_source, data_dir: Path, output_format: str
) -> bool:
    trades, prices, signals = _prepare_analysis_data(transaction_source, price_source, params.year, params.horizons)

    table = analysis.get_analysis_table(signals, params.member_filter, params.show_signals, params.horizons[0], params.top_n, params.threshold)
    logger.info(f"Generated analysis table with {len(table)} rows")

    _save_results(table, output_format, params.member_filter, params.show_signals, data_dir)

    sector_results = _analyze_by_sector(trades, signals, params.horizons)
    if sector_results is not None and not params.member_filter and not params.show_signals:
        print("\n=== Sector Analysis ===")
        print(sector_results.to_string(index=False))
    return True

@pipeline_step
def run_sales_pipeline(
    year: int, horizons: list[int], top_n: int,
    transaction_source, price_source, data_dir: Path, output_format: str
) -> bool:
    trades, prices, signals = _prepare_analysis_data(transaction_source, price_source, year, horizons)
    result = analysis.rank_sales(signals, horizons[0])
    result = result.head(top_n)
    _save_results(result, output_format, member_filter=None, show_signals=False, data_dir=data_dir)
    return True


@pipeline_step
def run_ticker_analysis(
    params: TickerAnalysisParams, transaction_source, price_source
) -> bool:
    trades, prices, signals = _prepare_analysis_data(transaction_source, price_source, params.year, [params.horizon])

    buyers = analysis.get_ticker_buyers_with_rankings(params.ticker, trades, signals, params.horizon, params.threshold)
    score = analysis.score_ticker_by_buyers(params.ticker, trades, signals, params.horizon, params.threshold)

    print(f"\n=== Buyers of {params.ticker} ===")
    print(buyers.to_string(index=False))
    print("\n=== Signal Score ===")
    print(score.to_string(index=False))
    return True

@pipeline_step
def run_recent_ticker_scoring(
    transaction_source, price_source, params: TickerScoringParams
) -> bool:
    if params.days_back < 1:
        raise DataSourceError("days_back must be at least 1")

    trades, prices, signals = _prepare_analysis_data(transaction_source, price_source, params.year, params.horizons)

    cutoff_date = trades['disclosure_date'].max() - timedelta(days=params.days_back)
    recent_trades = trades[trades['disclosure_date'] > cutoff_date]
    logger.info(f"Analyzing {len(recent_trades)} transactions from last {params.days_back} days")

    recent_purchases = recent_trades[recent_trades['transaction_type'] == TransactionType.PURCHASE.value]
    ticker_buyer_counts = recent_purchases.groupby('ticker')['member'].nunique()
    multi_buyer_tickers = ticker_buyer_counts[ticker_buyer_counts >= params.min_buyers].index.tolist()

    logger.info(f"Found {len(multi_buyer_tickers)} tickers with {params.min_buyers}+ buyers")

    member_rankings = analysis.rank_members(signals, params.horizons[0], params.threshold)

    scores = [analysis.score_ticker_by_buyers(ticker, trades, signals, params.horizons[0], params.threshold, member_rankings) for ticker in multi_buyer_tickers]

    if not scores:
        logger.warning(f"No tickers found with {params.min_buyers}+ buyers in last {params.days_back} days")
        return True

    result = pd.concat(scores, ignore_index=True).sort_values('signal_score', ascending=False).head(params.top_n)
    print(f"\n=== Top {params.top_n} Recent Signals (Last {params.days_back} Days, {params.min_buyers}+ Buyers) ===")
    print(result.to_string(index=False))
    return True

def _save_results(
    table: pd.DataFrame, output_format: str,
    member_filter: str | None, show_signals: bool, data_dir: Path
) -> None:
    if show_signals:
        display_cols = [
            'member', 'ticker', 'disclosure_date', 'spy_alpha_pct', 'peak_potential_pct'
        ]
    elif 'avg_loss_avoided_pct' in table.columns:
        display_cols = [
            'member', 'avg_loss_avoided_pct', 'median_loss_avoided_pct',
            'sale_trades', 'sharpe_ratio', 'bayes_win_prob', 'bayes_factor',
            'avg_spy_alpha_pct',
        ]
    elif 'avg_spy_alpha_pct' in table.columns:
        display_cols = [
            'member', 'avg_spy_alpha_pct', 'bayes_win_prob', 'hit_rate_pct', 'sharpe_ratio', 'bayes_factor', 'purchase_trades'
        ]
    else:
        display_cols = list(table.columns)
    available_display = [c for c in display_cols if c in table.columns]
    display_table = table[available_display]

    if output_format == 'csv':
        if member_filter:
            filename = f"{member_filter.replace(' ', '_').lower()}_signals.csv"
        elif show_signals:
            filename = "top_signals.csv"
        else:
            filename = "member_rankings.csv"

        filepath = data_dir / filename
        os.makedirs(data_dir, exist_ok=True)
        display_table.to_csv(filepath, index=False)
        logger.info(f"Results saved to {filepath}")
    else:
        print(display_table.to_string(index=False))


def _analyze_by_sector(
    trades: pd.DataFrame, signals: pd.DataFrame, horizons: list[int]
) -> pd.DataFrame | None:
    tickers = trades['ticker'].unique()
    sectors = _load_sector_data(tickers.tolist())
    if sectors.empty:
        logger.info("No sector data available")
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
            ranked = analysis.rank_members(sector_purchases, horizons[0])
            if not ranked.empty:
                results.append({
                    "sector": sector,
                    "top_member": ranked.iloc[0]["member"],
                    "top_member_alpha": ranked.iloc[0]["avg_spy_alpha_pct"],
                    "num_trades": len(sector_purchases),
                    "num_members": sector_purchases["member"].nunique(),
                })
        except AnalysisError as e:
            logger.warning(f"Skipping sector '{sector}' analysis: {e}")
            continue

    if not results:
        return None
    return pd.DataFrame(results).sort_values("top_member_alpha", ascending=False)
