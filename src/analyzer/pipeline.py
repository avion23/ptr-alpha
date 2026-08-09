from __future__ import annotations

from datetime import date, timedelta
from dataclasses import dataclass
from functools import wraps
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from analyzer.exceptions import AnalyzerError, DataSourceError, StepResult, DataResult
from analyzer.member_names import canonical_member_key
from analyzer.models import AnalysisMode, TransactionType
from analyzer.price_repository import next_nyse_session
from analyzer.price_snapshot import create_snapshot, save_snapshot
from analyzer.ticker_resolver import TickerResolver
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
    mode: AnalysisMode = AnalysisMode.MEMBER_RANKINGS
    include_sector_analysis: bool = False


@dataclass(frozen=True, slots=True)
class TickerScoringParams:
    year: int
    horizons: tuple[int, ...]
    threshold: float = 5.0
    days_back: int = 28
    min_buyers: int = 3
    top_n: int = 15
    training_lookback_days: int = 1095
    as_of_date: date | None = None


@dataclass(frozen=True, slots=True)
class TickerAnalysisParams:
    ticker: str
    year: int
    horizon: int = 90
    threshold: float = 5.0
    as_of_date: date | None = None


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
    trades = trades[trades["ticker"].notna()].copy()
    logger.info("After filtering NULL tickers: %d transactions", len(trades))

    if trades.empty:
        raise DataSourceError("No valid tickers found in transaction data")

    start_date = trades["disclosure_date"].min() - timedelta(days=30)
    end_date = trades["disclosure_date"].max() + timedelta(days=max(horizons) + 10)

    prices = price_source.get_prices(trades["ticker"].unique(), start_date, end_date)
    logger.info("Fetched price data for %d tickers", len(prices.columns))

    all_tickers = trades["ticker"].unique().tolist()
    entry_prices = transaction_source.db.get_entry_prices(
        all_tickers, start_date, end_date
    )
    logger.info("Computed entry prices for %d transactions", len(entry_prices))

    signals = analysis.calculate_signal_potential(entry_prices, prices, horizons)
    logger.info("Calculated %d signals", len(signals))

    return trades, prices, signals


def prepare_live_analysis_data(
    transaction_source,
    price_source,
    horizons: tuple[int, ...],
    as_of_date: pd.Timestamp,
    training_lookback_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build live features from historical data available by ``as_of_date``."""
    history_start = as_of_date - timedelta(days=training_lookback_days + max(horizons))
    trades = transaction_source.db.get_transactions_by_date_range(
        history_start,
        as_of_date,
    )
    if trades.empty:
        raise DataSourceError("No trading data found through as-of date")

    disclosure_dates = pd.to_datetime(trades["disclosure_date"], errors="coerce")
    trades = trades[
        trades["ticker"].notna()
        & disclosure_dates.notna()
        & (disclosure_dates <= as_of_date)
    ].copy()
    if trades.empty:
        raise DataSourceError("No valid tickers found through as-of date")

    price_start = history_start - timedelta(days=30)
    prices = price_source.get_prices(
        trades["ticker"].unique(),
        price_start,
        as_of_date,
    )
    entry_prices = transaction_source.db.get_entry_prices(
        trades["ticker"].unique().tolist(),
        price_start,
        as_of_date,
    )
    signals = analysis.calculate_signal_potential(entry_prices, prices, horizons)

    # Outcomes before the training boundary are needed only to cover their
    # forward horizon; they must not influence member ranking themselves.
    training_start = as_of_date - timedelta(days=training_lookback_days)
    signal_dates = pd.to_datetime(signals["disclosure_date"], errors="coerce")
    signals = signals[
        (signal_dates >= training_start) & (signal_dates <= as_of_date)
    ].copy()
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
    params: AnalysisParams, transaction_source, price_source
) -> DataResult:
    trades, prices, signals = prepare_analysis_data(
        transaction_source, price_source, params.year, params.horizons
    )

    table = analysis.get_analysis_table(
        signals,
        params.mode,
        params.member_filter,
        params.horizons[0],
        params.top_n,
        params.threshold,
    )
    logger.info("Generated analysis table with %d rows", len(table))

    sector_results = (
        analysis.analyze_by_sector(trades, signals, params.horizons)
        if params.include_sector_analysis
        else None
    )
    return DataResult(
        success=True,
        data={
            "table": table,
            "sector_results": sector_results,
            "member_filter": params.member_filter,
            "mode": params.mode,
        },
    )


@pipeline_step
def run_sales_pipeline(
    year: int, horizons: tuple[int, ...], top_n: int, transaction_source, price_source
) -> DataResult:
    trades, prices, signals = prepare_analysis_data(
        transaction_source, price_source, year, horizons
    )
    result = analysis.rank_sales(signals, horizons[0])
    result = result.head(top_n)
    return DataResult(
        success=True,
        data={
            "table": result,
        },
    )


def _consensus_buyers_table(ticker: str, trades: pd.DataFrame) -> pd.DataFrame:
    """Display known buyers without joining identity-based performance history."""
    purchases = trades[
        (trades["ticker"] == ticker)
        & (trades["transaction_type"] == TransactionType.PURCHASE.value)
        & trades["member"].notna()
    ].copy()
    if purchases.empty:
        return pd.DataFrame(
            columns=[
                "member",
                "num_purchases",
                "transaction_date",
                "disclosure_date",
            ]
        )
    return (
        purchases.groupby("member", sort=True)
        .agg(
            num_purchases=("ticker", "size"),
            transaction_date=("transaction_date", list),
            disclosure_date=("disclosure_date", list),
        )
        .reset_index()
    )


@pipeline_step
def run_ticker_analysis(
    params: TickerAnalysisParams, transaction_source, price_source
) -> DataResult:
    analysis_as_of = pd.Timestamp(
        params.as_of_date or min(date.today(), date(params.year, 12, 31))
    ).normalize()
    if analysis_as_of.year != params.year:
        raise DataSourceError("year must match the ticker analysis as-of date year")

    trades, prices, signals = prepare_analysis_data(
        transaction_source, price_source, params.year, (params.horizon,)
    )
    disclosure_dates = pd.to_datetime(trades["disclosure_date"], errors="coerce")
    known_trades = trades[
        disclosure_dates.notna() & (disclosure_dates <= analysis_as_of)
    ].copy()

    buyers = _consensus_buyers_table(params.ticker, known_trades)
    score = analysis.score_ticker_by_buyers(
        params.ticker,
        known_trades,
        signals,
        horizon=params.horizon,
        threshold=params.threshold,
        member_rankings=None,
        scoring_mode="consensus",
        as_of_date=analysis_as_of,
    )

    return DataResult(
        success=True,
        data={
            "buyers": buyers,
            "score": score,
            "ticker": params.ticker,
        },
    )


@pipeline_step
def run_recent_ticker_scoring(
    transaction_source, price_source, params: TickerScoringParams
) -> DataResult:
    if params.days_back < 1:
        raise DataSourceError("days_back must be at least 1")
    if not params.horizons or any(horizon < 1 for horizon in params.horizons):
        raise DataSourceError("horizons must contain positive days")

    if params.training_lookback_days < 1:
        raise DataSourceError("training_lookback_days must be at least 1")

    as_of_date = pd.Timestamp(params.as_of_date or date.today()).normalize()
    if as_of_date.year != params.year:
        raise DataSourceError("year must match the as-of date year")
    trades, prices, signals = prepare_live_analysis_data(
        transaction_source,
        price_source,
        params.horizons,
        as_of_date,
        params.training_lookback_days,
    )
    cutoff_date = as_of_date - timedelta(days=params.days_back)
    disclosure_dates = pd.to_datetime(trades["disclosure_date"])
    recent_trades = trades[
        (disclosure_dates >= cutoff_date) & (disclosure_dates <= as_of_date)
    ]
    logger.info(
        "Analyzing %d transactions from last %d days",
        len(recent_trades),
        params.days_back,
    )

    recent_purchases = recent_trades[
        recent_trades["transaction_type"] == TransactionType.PURCHASE.value
    ].copy()
    recent_purchases["_member_canonical"] = recent_purchases["member"].map(
        canonical_member_key
    )
    ticker_buyer_counts = recent_purchases.groupby("ticker")[
        "_member_canonical"
    ].nunique()
    multi_buyer_tickers = ticker_buyer_counts[
        ticker_buyer_counts >= params.min_buyers
    ].index.tolist()

    logger.info(
        "Found %d tickers with %d+ buyers", len(multi_buyer_tickers), params.min_buyers
    )

    scores = [
        analysis.score_ticker_by_buyers(
            ticker,
            recent_trades,
            signals,
            horizon=params.horizons[0],
            threshold=params.threshold,
            member_rankings=None,
            min_buyers=params.min_buyers,
            scoring_mode="consensus",
            as_of_date=as_of_date,
        )
        for ticker in multi_buyer_tickers
    ]

    if not scores:
        logger.warning(
            "No tickers found with %d+ buyers in last %d days",
            params.min_buyers,
            params.days_back,
        )
        return DataResult(
            success=True,
            data={
                "result": pd.DataFrame(),
                "top_n": params.top_n,
                "days_back": params.days_back,
                "min_buyers": params.min_buyers,
                "as_of_date": as_of_date.date(),
            },
        )

    result = pd.concat(scores, ignore_index=True)
    # This interface presents buy candidates, so rejected/negative scores must
    # not leak into the displayed recommendations merely to fill top_n.
    if "signal_score_raw" not in result.columns:
        result = result.iloc[0:0]
    else:
        result = result[
            pd.to_numeric(result["signal_score_raw"], errors="coerce").fillna(0) > 0
        ]
    result = result.sort_values("signal_score", ascending=False).head(params.top_n)
    return DataResult(
        success=True,
        data={
            "result": result,
            "top_n": params.top_n,
            "days_back": params.days_back,
            "min_buyers": params.min_buyers,
            "as_of_date": as_of_date.date(),
        },
    )


def _entry_prices_from_matrix(
    transactions: pd.DataFrame, prices: pd.DataFrame
) -> pd.DataFrame:
    """Build entry rows from the exact matrix used to calculate labels."""
    if transactions.empty or prices.empty:
        return pd.DataFrame()

    index = pd.DatetimeIndex(pd.to_datetime(prices.index))
    if index.tz is not None:
        index = index.tz_localize(None)
    index = index.normalize()
    if index.has_duplicates:
        raise DataSourceError("Price matrix contains duplicate calendar dates")
    matrix = prices.copy()
    matrix.index = index
    matrix = matrix.sort_index()

    eligible = transactions[
        transactions["ticker"].notna()
        & transactions["disclosure_date"].notna()
        & (
            transactions["transaction_date"].isna()
            | (
                pd.to_datetime(transactions["transaction_date"])
                <= pd.to_datetime(transactions["disclosure_date"])
            )
        )
    ].copy()
    if eligible.empty:
        return pd.DataFrame()

    resolver = TickerResolver()
    price_columns = set(matrix.columns)
    rows = []
    for _, transaction in eligible.iterrows():
        raw_ticker = str(transaction["ticker"])
        price_ticker = raw_ticker
        if price_ticker not in price_columns:
            resolved = resolver.resolve(raw_ticker).price_symbol
            if resolved not in price_columns:
                continue
            price_ticker = resolved

        disclosure = pd.Timestamp(transaction["disclosure_date"])
        if disclosure.tz is not None:
            disclosure = disclosure.tz_localize(None)
        entry_date = next_nyse_session(disclosure)
        if entry_date not in matrix.index:
            continue
        entry_price = matrix.at[entry_date, price_ticker]
        if pd.isna(entry_price) or not np.isfinite(entry_price) or entry_price <= 0:
            continue

        row = transaction.to_dict()
        row["entry_price"] = float(entry_price)
        row["entry_price_date"] = entry_date
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


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

    all_transactions = transaction_source.db.get_transactions_by_date_range(
        tx_start, tx_end
    )
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

    prices = price_source.get_prices(all_tickers, price_start, price_end)

    if prices.empty:
        raise DataSourceError("No price data available for backtest window")

    # Snapshot the exact acquired in-memory values. In read-only mode the
    # price source may have merged fresh observations without writing the DB.
    snapshot = create_snapshot(
        transaction_source.db,
        all_tickers,
        price_start,
        price_end,
        prices=prices,
    )

    entry_prices = _entry_prices_from_matrix(all_transactions, prices)
    if entry_prices.empty:
        raise DataSourceError("No entry prices could be computed")

    signals = analysis.calculate_signal_potential(
        entry_prices, prices, [params.horizon]
    )
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
    total_unavailable = 0
    for as_of in as_of_dates:
        as_of_ts = pd.Timestamp(as_of)

        recs = analysis.backtest_recommendations(
            signals,
            all_transactions,
            as_of_ts,
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
        total_unavailable += evaluated.attrs.get("n_unavailable", 0)
        evaluated = evaluated.dropna(subset=["bt_return_pct"])
        evaluated.insert(0, "as_of_date", as_of_ts.date())
        all_results.append(evaluated)

    if not all_results:
        return DataResult(
            success=True,
            data={
                "combined": pd.DataFrame(),
                "summary": pd.DataFrame(),
                "snapshot": snapshot,
                "evaluable_dates": 0,
                "total_as_of_dates": len(as_of_dates),
            },
        )

    combined = pd.concat(all_results, ignore_index=True)
    # Set accumulated coverage counts on the combined frame so summarize_backtest
    # can propagate them (Finding 1: pd.concat drops attrs in pandas 3.x).
    combined.attrs["n_no_price"] = total_no_price
    combined.attrs["n_delisted"] = total_delisted
    combined.attrs["n_unavailable"] = total_unavailable

    summary = analysis.summarize_backtest(combined)

    valid_returns = combined.dropna(subset=["bt_return_pct"])
    evaluable_dates = (
        valid_returns["as_of_date"].nunique() if not valid_returns.empty else 0
    )
    total_as_of_dates = len(
        pd.date_range(
            params.start_date, params.end_date, freq=f"{params.frequency_days}D"
        )
    )

    # Save snapshot alongside backtest results
    snapshot_path = data_dir / "price_snapshot.json"
    save_snapshot(snapshot, snapshot_path)
    logger.info("Price snapshot saved to %s", snapshot_path)

    return DataResult(
        success=True,
        data={
            "combined": combined,
            "summary": summary,
            "snapshot": snapshot,
            "evaluable_dates": evaluable_dates,
            "total_as_of_dates": total_as_of_dates,
        },
    )
