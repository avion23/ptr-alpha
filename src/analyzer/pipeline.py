from datetime import timedelta
import logging
from analyzer.exceptions import DataSourceError, AnalysisError, ParsingError
from analyzer import sources
from analyzer import analysis

logger = logging.getLogger(__name__)

def _prepare_analysis_data(source, year, horizons, config):
    trades = sources.load_data(source, year, config)
    logger.info(f"Loaded {len(trades)} transactions from {source} for {year}")

    if len(trades) == 0:
        raise DataSourceError("No trading data found")

    start_date = trades['disclosure_date'].min() - timedelta(days=30)
    end_date = trades['disclosure_date'].max() + timedelta(days=max(horizons) + 10)

    prices = sources.fetch_prices(trades['ticker'].unique(), start_date, end_date, config)
    logger.info(f"Fetched price data for {len(prices.columns)} tickers")

    signals = analysis.calculate_signal_potential(trades, prices, horizons)
    logger.info(f"Calculated {len(signals)} signals")

    return trades, prices, signals

def run_fetch_pipeline(year, config):
    try:
        sources.fetch_and_cache_pdfs(year, config)
        logger.info(f"Successfully fetched PDFs for {year}")
        return True
    except (DataSourceError, ParsingError) as e:
        logger.error(f"Fetch pipeline failed: {e}")
        return False
    except Exception as e:
        logger.exception(f"Unexpected error in fetch pipeline: {e}")
        raise

def run_parse_pipeline(year, config):
    try:
        sources.parse_cached_pdfs(year, config)
        logger.info(f"Successfully parsed PDFs for {year}")
        return True
    except (DataSourceError, ParsingError) as e:
        logger.error(f"Parse pipeline failed: {e}")
        return False
    except Exception as e:
        logger.exception(f"Unexpected error in parse pipeline: {e}")
        raise

def run_analysis_pipeline(source, year, horizons, threshold, member_filter, top_n, show_signals, output_format, config):
    try:
        trades, prices, signals = _prepare_analysis_data(source, year, horizons, config)

        table = analysis.get_analysis_table(signals, member_filter, show_signals, horizons[0], top_n, threshold)
        logger.info(f"Generated analysis table with {len(table)} rows")

        result = sources.save_results(table, output_format, member_filter, show_signals, config)
        return bool(result)

    except (DataSourceError, AnalysisError) as e:
        logger.error(f"Analysis pipeline failed: {e}")
        return False
    except Exception as e:
        logger.exception(f"Unexpected error in analysis pipeline: {e}")
        raise

def run_ticker_analysis(ticker, source, year, horizon, threshold, config):
    try:
        trades, prices, signals = _prepare_analysis_data(source, year, [horizon], config)

        buyers = analysis.get_ticker_buyers_with_rankings(ticker, trades, signals, horizon, threshold)
        score = analysis.score_ticker_by_buyers(ticker, trades, signals, horizon, threshold)

        print(f"\n=== Buyers of {ticker} ===")
        print(buyers.to_string(index=False))
        print(f"\n=== Signal Score ===")
        print(score.to_string(index=False))
        return True

    except (DataSourceError, AnalysisError) as e:
        logger.error(f"Ticker analysis failed: {e}")
        return False
    except Exception as e:
        logger.exception(f"Unexpected error in ticker analysis: {e}")
        raise

def run_recent_ticker_scoring(source, year, horizons, threshold, days_back, min_buyers, top_n, config):
    try:
        import pandas as pd
        if days_back < 1:
            raise DataSourceError("days_back must be at least 1")

        trades, prices, signals = _prepare_analysis_data(source, year, horizons, config)

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

    except (DataSourceError, AnalysisError) as e:
        logger.error(f"Recent ticker scoring failed: {e}")
        return False
    except Exception as e:
        logger.exception(f"Unexpected error in recent ticker scoring: {e}")
        raise