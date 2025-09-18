from datetime import timedelta
import logging
from exceptions import DataSourceError, AnalysisError
import sources
import analysis

logger = logging.getLogger(__name__)

def run_fetch_pipeline(year, config):
    try:
        sources.fetch_and_cache_pdfs(year, config)
        logger.info(f"Successfully fetched PDFs for {year}")
        return True
    except Exception as e:
        logger.error(f"Fetch pipeline failed: {e}")
        return False

def run_parse_pipeline(year, config):
    try:
        sources.parse_cached_pdfs(year, config)
        logger.info(f"Successfully parsed PDFs for {year}")
        return True
    except Exception as e:
        logger.error(f"Parse pipeline failed: {e}")
        return False

def run_analysis_pipeline(source, year, horizons, threshold, member_filter, top_n, show_signals, output_format, config):
    try:
        trades = sources.load_data(source, year, config)
        logger.info(f"Loaded {len(trades)} transactions from {source} for {year}")

        if trades.empty:
            raise DataSourceError("No trading data found")

        start_date = trades['disclosure_date'].min() - timedelta(days=30)
        end_date = trades['disclosure_date'].max() + timedelta(days=max(horizons) + 10)

        prices = sources.fetch_prices(trades['ticker'].unique(), start_date, end_date, config)
        logger.info(f"Fetched price data for {len(prices.columns)} tickers")

        signals = analysis.calculate_signal_potential(trades, prices, horizons)
        logger.info(f"Calculated {len(signals)} signals")

        table = analysis.get_analysis_table(signals, member_filter, show_signals, horizons[0], top_n, threshold)
        logger.info(f"Generated analysis table with {len(table)} rows")

        result = sources.save_results(table, output_format, member_filter, show_signals, config)
        return bool(result)

    except (DataSourceError, AnalysisError) as e:
        logger.error(f"Analysis pipeline failed: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error in analysis pipeline: {e}")
        return False