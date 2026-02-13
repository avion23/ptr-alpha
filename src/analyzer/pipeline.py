from datetime import timedelta
import logging
import pandas as pd
import os
from functools import wraps
from dataclasses import dataclass
from analyzer.exceptions import AnalyzerError, DataSourceError
from analyzer import analysis

logger = logging.getLogger(__name__)

@dataclass
class AnalysisParams:
    source: str
    year: int
    horizons: list
    threshold: float
    member_filter: str = None
    top_n: int = None
    show_signals: bool = False
    output_format: str = 'console'

def pipeline_step(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except AnalyzerError as e:
            logger.error(f"{func.__name__} failed: {e}")
            return False
        except Exception as e:
            logger.exception(f"Unexpected error in {func.__name__}: {e}")
            raise
    return wrapper

def _prepare_analysis_data(transaction_source, price_source, year, horizons):
    trades = transaction_source.get_transactions(year)
    logger.info(f"Loaded {len(trades)} transactions for {year}")

    if len(trades) == 0:
        raise DataSourceError("No trading data found")

    start_date = trades['disclosure_date'].min() - timedelta(days=30)
    end_date = trades['disclosure_date'].max() + timedelta(days=max(horizons) + 10)

    prices = price_source.get_prices(trades['ticker'].unique(), start_date, end_date)
    logger.info(f"Fetched price data for {len(prices.columns)} tickers")

    signals = analysis.calculate_signal_potential(trades, prices, horizons)
    logger.info(f"Calculated {len(signals)} signals")

    return trades, prices, signals

@pipeline_step
def run_fetch_pipeline(transaction_source, year):
    transaction_source.fetch_and_cache_pdfs(year)
    logger.info(f"Successfully fetched PDFs for {year}")
    return True

@pipeline_step
def run_parse_pipeline(transaction_source, year):
    transaction_source.parse_cached_pdfs(year)
    logger.info(f"Successfully parsed PDFs for {year}")
    return True

@pipeline_step
def run_analysis_pipeline(params, transaction_source, price_source, data_dir, output_format):
    trades, prices, signals = _prepare_analysis_data(transaction_source, price_source, params.year, params.horizons)

    table = analysis.get_analysis_table(signals, params.member_filter, params.show_signals, params.horizons[0], params.top_n, params.threshold)
    logger.info(f"Generated analysis table with {len(table)} rows")

    result = _save_results(table, output_format, params.member_filter, params.show_signals, data_dir)
    return bool(result)

@pipeline_step
def run_ticker_analysis(ticker, transaction_source, price_source, year, horizon, threshold):
    trades, prices, signals = _prepare_analysis_data(transaction_source, price_source, year, [horizon])

    buyers = analysis.get_ticker_buyers_with_rankings(ticker, trades, signals, horizon, threshold)
    score = analysis.score_ticker_by_buyers(ticker, trades, signals, horizon, threshold)

    print(f"\n=== Buyers of {ticker} ===")
    print(buyers.to_string(index=False))
    print(f"\n=== Signal Score ===")
    print(score.to_string(index=False))
    return True

@pipeline_step
def run_recent_ticker_scoring(transaction_source, price_source, year, horizons, threshold, days_back, min_buyers, top_n):
    if days_back < 1:
        raise DataSourceError("days_back must be at least 1")

    trades, prices, signals = _prepare_analysis_data(transaction_source, price_source, year, horizons)

    cutoff_date = trades['disclosure_date'].max() - timedelta(days=days_back)
    recent_trades = trades[trades['disclosure_date'] > cutoff_date]
    logger.info(f"Analyzing {len(recent_trades)} transactions from last {days_back} days")

    recent_purchases = recent_trades[recent_trades['transaction_type'] == 'Purchase']
    ticker_buyer_counts = recent_purchases.groupby('ticker')['member'].nunique()
    multi_buyer_tickers = ticker_buyer_counts[ticker_buyer_counts >= min_buyers].index.tolist()

    logger.info(f"Found {len(multi_buyer_tickers)} tickers with {min_buyers}+ buyers")

    member_rankings = analysis.rank_members(signals, horizons[0], threshold)

    scores = [analysis.score_ticker_by_buyers(ticker, trades, signals, horizons[0], threshold, member_rankings) for ticker in multi_buyer_tickers]

    if not scores:
        logger.warning(f"No tickers found with {min_buyers}+ buyers in last {days_back} days")
        return True

    result = pd.concat(scores, ignore_index=True).sort_values('signal_score', ascending=False).head(top_n)
    print(f"\n=== Top {top_n} Recent Signals (Last {days_back} Days, {min_buyers}+ Buyers) ===")
    print(result.to_string(index=False))
    return True

def _save_results(table, output_format, member_filter, show_signals, data_dir):
    if output_format == 'csv':
        if member_filter:
            filename = f"{member_filter.replace(' ', '_').lower()}_signals.csv"
        elif show_signals:
            filename = "top_signals.csv"
        else:
            filename = "member_rankings.csv"

        filepath = data_dir / filename
        os.makedirs(data_dir, exist_ok=True)
        table.to_csv(filepath, index=False)
        logger.info(f"Results saved to {filepath}")
        return True
    else:
        print(table.to_string(index=False))
        return True
