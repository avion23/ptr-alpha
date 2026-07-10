from __future__ import annotations

from datetime import date, timedelta
from dataclasses import dataclass
from functools import wraps
import logging
import re
from pathlib import Path

import pandas as pd

from analyzer.exceptions import AnalyzerError, DataSourceError, StepResult, DataResult
from analyzer.models import TransactionType
from analyzer.price_snapshot import create_snapshot, save_snapshot
from analyzer import analysis

logger = logging.getLogger(__name__)

_VALID_TICKER_RE = re.compile(r"^[A-Z]{1,5}([.-][A-Z]{1,2})?$")


@dataclass(frozen=True, slots=True)
class AnalysisParams:
    year: int
    horizons: tuple[int, ...]
    threshold: float
    source: str = "house"
    member_filter: str | None = None
    top_n: int | None = None
    show_signals: bool = False


@dataclass(frozen=True, slots=True)
class TickerScoringParams:
    year: int
    horizons: tuple[int, ...]
    threshold: float = 5.0
    days_back: int = 28
    min_buyers: int = 3
    top_n: int = 15


@dataclass(frozen=True, slots=True)
class TickerAnalysisParams:
    ticker: str
    year: int
    horizon: int = 90
    threshold: float = 5.0


@dataclass(frozen=True, slots=True)
class BacktestParams:
    start_date: date
    end_date: date
    # Optimal defaults from pdfplumber-era sweep (sharpe=1.41, alpha=+1.92%,
    # DD=-9.89%, win=65.6%). min_buyers=3 is the single biggest driver: crowd
    # consensus filters out idiosyncratic single-member picks.
    horizon: int = 60
    lookback_days: int = 60
    training_lookback_days: int = 365
    min_buyers: int = 3
    top_n: int = 5
    threshold: float = 5.0
    frequency_days: int = 30

def pipeline_step(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            if result is True:
                return StepResult(success=True)
            return result  # Already a StepResult or DataResult or similar
        except AnalyzerError as exc:
            logger.error("Pipeline step %s failed: %s", func.__name__, exc)
            return StepResult(success=False, error=exc)
    return wrapper

def prepare_analysis_data(
    transaction_source, price_source, year: int, horizons: tuple[int, ...]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trades = transaction_source.get_transactions(year)
    logger.info("Loaded %d transactions for %d", len(trades), year)

    if len(trades) == 0:
        raise DataSourceError("No trading data found")

    # Filter out transactions with NULL tickers (from parser failures)
    trades = trades[trades['ticker'].notna()].copy()
    logger.info("After filtering NULL tickers: %d transactions", len(trades))

    if trades.empty:
        raise DataSourceError("No valid tickers found in transaction data")

    start_date = trades['disclosure_date'].min() - timedelta(days=30)
    end_date = trades['disclosure_date'].max() + timedelta(days=max(horizons) + 10)

    prices = price_source.get_prices(trades['ticker'].unique(), start_date, end_date)
    logger.info("Fetched price data for %d tickers", len(prices.columns))

    all_tickers = trades['ticker'].unique().tolist()
    entry_prices = transaction_source.db.get_entry_prices(all_tickers, start_date, end_date)
    logger.info("Computed entry prices for %d transactions", len(entry_prices))

    signals = analysis.calculate_signal_potential(entry_prices, prices, horizons)
    logger.info("Calculated %d signals", len(signals))

    return trades, prices, signals

@pipeline_step
def run_fetch_pipeline(transaction_source, year: int) -> DataResult:
    transaction_source.fetch_and_cache_pdfs(year)
    logger.info("Successfully fetched PDFs for %d", year)
    return DataResult(success=True, data=None)

@pipeline_step
def run_parse_pipeline(transaction_source, year: int) -> DataResult:
    transaction_source.parse_cached_pdfs(year)
    logger.info("Successfully parsed PDFs for %d", year)
    return DataResult(success=True, data=None)

@pipeline_step
def run_analysis_pipeline(
    params: AnalysisParams, transaction_source, price_source, data_dir: Path, output_format: str
) -> DataResult:
    trades, prices, signals = prepare_analysis_data(transaction_source, price_source, params.year, params.horizons)

    table = analysis.get_analysis_table(signals, params.member_filter, params.show_signals, params.horizons[0], params.top_n, params.threshold)
    logger.info("Generated analysis table with %d rows", len(table))

    sector_results = analysis.analyze_by_sector(trades, signals, params.horizons)
    return DataResult(success=True, data={
        "table": table,
        "sector_results": sector_results,
        "member_filter": params.member_filter,
        "show_signals": params.show_signals,
    })

@pipeline_step
def run_sales_pipeline(
    year: int, horizons: tuple[int, ...], top_n: int,
    transaction_source, price_source, data_dir: Path, output_format: str
) -> DataResult:
    trades, prices, signals = prepare_analysis_data(transaction_source, price_source, year, horizons)
    result = analysis.rank_sales(signals, horizons[0])
    result = result.head(top_n)
    return DataResult(success=True, data={
        "table": result,
    })


@pipeline_step
def run_ticker_analysis(
    params: TickerAnalysisParams, transaction_source, price_source
) -> DataResult:
    trades, prices, signals = prepare_analysis_data(transaction_source, price_source, params.year, (params.horizon,))

    buyers = analysis.get_ticker_buyers_with_rankings(params.ticker, trades, signals, params.horizon, params.threshold)
    score = analysis.score_ticker_by_buyers(params.ticker, trades, signals, params.horizon, params.threshold)

    return DataResult(success=True, data={
        "buyers": buyers,
        "score": score,
        "ticker": params.ticker,
    })

@pipeline_step
def run_recent_ticker_scoring(
    transaction_source, price_source, params: TickerScoringParams
) -> DataResult:
    if params.days_back < 1:
        raise DataSourceError("days_back must be at least 1")

    trades, prices, signals = prepare_analysis_data(transaction_source, price_source, params.year, params.horizons)

    as_of_date = pd.Timestamp(date.today())
    cutoff_date = as_of_date - timedelta(days=params.days_back)
    disclosure_dates = pd.to_datetime(trades['disclosure_date'])
    recent_trades = trades[
        (disclosure_dates >= cutoff_date) & (disclosure_dates <= as_of_date)
    ]
    logger.info("Analyzing %d transactions from last %d days", len(recent_trades), params.days_back)

    recent_purchases = recent_trades[recent_trades['transaction_type'] == TransactionType.PURCHASE.value]
    ticker_buyer_counts = recent_purchases.groupby('ticker')['member'].nunique()
    multi_buyer_tickers = ticker_buyer_counts[ticker_buyer_counts >= params.min_buyers].index.tolist()

    logger.info("Found %d tickers with %d+ buyers", len(multi_buyer_tickers), params.min_buyers)

    member_rankings = analysis.rank_members(signals, params.horizons[0], params.threshold)

    scores = [analysis.score_ticker_by_buyers(ticker, recent_trades, signals, params.horizons[0], params.threshold, member_rankings, params.min_buyers) for ticker in multi_buyer_tickers]

    if not scores:
        logger.warning("No tickers found with %d+ buyers in last %d days", params.min_buyers, params.days_back)
        return DataResult(success=True, data={
            "result": pd.DataFrame(),
            "top_n": params.top_n,
            "days_back": params.days_back,
            "min_buyers": params.min_buyers,
            "as_of_date": as_of_date.date(),
        })

    result = pd.concat(scores, ignore_index=True)
# This interface presents buy candidates, so rejected/negative scores must
    # not leak into the displayed recommendations merely to fill top_n.
    if "signal_score" not in result.columns:
        result = result.iloc[0:0]
    else:
        result = result[
            pd.to_numeric(result["signal_score"], errors="coerce").fillna(0) > 0
        ]
    result = result.sort_values('signal_score', ascending=False).head(params.top_n)
    return DataResult(success=True, data={
        "result": result,
        "top_n": params.top_n,
        "days_back": params.days_back,
        "min_buyers": params.min_buyers,
        "as_of_date": as_of_date.date(),
    })


@pipeline_step
def run_backtest_pipeline(
    params: BacktestParams,
    transaction_source,
    price_source,
    data_dir: Path = Path("data"),
) -> DataResult:
    tx_start = params.start_date - timedelta(
        days=params.training_lookback_days + params.horizon + 30
    )
    tx_end = params.end_date

    all_transactions = transaction_source.db.get_transactions_by_date_range(tx_start, tx_end)
    if all_transactions.empty:
        raise DataSourceError(f"No transactions found between {tx_start} and {tx_end}")

    logger.info("Loaded %d transactions for backtest window", len(all_transactions))

    price_start = tx_start
    price_end = params.end_date + timedelta(days=params.horizon + 10)
    all_tickers = all_transactions["ticker"].unique().tolist()
    all_tickers = [t for t in all_tickers if t and str(t).strip() and str(t) != "nan"]
    # Filter out non-stock tickers (OCR garbage)
    all_tickers = [t for t in all_tickers if _VALID_TICKER_RE.match(str(t))]
    all_tickers = sorted(set(all_tickers) | {"SPY"})

    # Create price snapshot for reproducibility
    snapshot = create_snapshot(transaction_source.db, all_tickers, price_start, price_end)

    prices = price_source.get_prices(all_tickers, price_start, price_end)

    if prices.empty:
        raise DataSourceError("No price data available for backtest window")

    entry_prices = transaction_source.db.get_entry_prices(all_tickers, price_start, price_end)
    if entry_prices.empty:
        raise DataSourceError("No entry prices could be computed")

    signals = analysis.calculate_signal_potential(entry_prices, prices, [params.horizon])
    logger.info("Computed %d signals for backtest", len(signals))

    as_of_dates = pd.date_range(
        params.start_date, params.end_date, freq=f"{params.frequency_days}D"
    )

    all_results = []
    # Finding 1 fix: accumulate per-date attrs counts explicitly because
    # pd.concat of DataFrames with differing .attrs yields attrs={} in
    # pandas 3.x.  We sum here and set them on the combined frame.
    total_no_price = 0
    total_delisted = 0
    for as_of in as_of_dates:
        as_of_ts = pd.Timestamp(as_of)

        recs = analysis.backtest_recommendations(
            signals, all_transactions, as_of_ts,
            horizon=params.horizon,
            lookback_days=params.lookback_days,
            min_buyers=params.min_buyers,
            top_n=params.top_n,
            threshold=params.threshold,
            prices_df=prices,
            training_lookback_days=params.training_lookback_days,
        )

        if recs.empty:
            continue

        evaluated = analysis.evaluate_backtest(recs, prices, as_of_ts, params.horizon)
        total_no_price += evaluated.attrs.get("n_no_price", 0)
        total_delisted += evaluated.attrs.get("n_delisted", 0)
        evaluated = evaluated.dropna(subset=["bt_return_pct"])
        evaluated.insert(0, "as_of_date", as_of_ts.date())
        all_results.append(evaluated)

    if not all_results:
        return DataResult(success=True, data={
            "combined": pd.DataFrame(),
            "summary": pd.DataFrame(),
            "snapshot": snapshot,
            "evaluable_dates": 0,
            "total_as_of_dates": len(as_of_dates),
        })

    combined = pd.concat(all_results, ignore_index=True)
    # Set accumulated coverage counts on the combined frame so summarize_backtest
    # can propagate them (Finding 1: pd.concat drops attrs in pandas 3.x).
    combined.attrs["n_no_price"] = total_no_price
    combined.attrs["n_delisted"] = total_delisted

    summary = analysis.summarize_backtest(combined)

    valid_returns = combined.dropna(subset=["bt_return_pct"])
    evaluable_dates = valid_returns["as_of_date"].nunique() if not valid_returns.empty else 0
    total_as_of_dates = len(pd.date_range(params.start_date, params.end_date, freq=f"{params.frequency_days}D"))

    # Save snapshot alongside backtest results
    snapshot_path = data_dir / "price_snapshot.json"
    save_snapshot(snapshot, snapshot_path)
    logger.info("Price snapshot saved to %s", snapshot_path)

    return DataResult(success=True, data={
        "combined": combined,
        "summary": summary,
        "snapshot": snapshot,
        "evaluable_dates": evaluable_dates,
        "total_as_of_dates": total_as_of_dates,
    })
