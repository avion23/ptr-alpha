from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from dataclasses import dataclass
from functools import wraps
import logging

import numpy as np
import pandas as pd

from analyzer._price_index import _normalize_price_index
from analyzer.exceptions import AnalyzerError, DataSourceError, StepResult, DataResult
from analyzer.models import AnalysisMode
from analyzer.price_repository import next_nyse_session, previous_nyse_session
from analyzer.price_snapshot import create_snapshot, save_snapshot
from analyzer.ticker_resolver import TickerResolver
from analyzer import analysis
from analyzer.member_ranking.buyer_scoring import (
    CONSENSUS_LOOKBACK_DAYS,
    CONSENSUS_MIN_BUYERS,
    _get_consensus_candidate_tickers,
    _get_consensus_price_tickers,
    _get_consensus_ticker_purchases,
    _resolve_consensus_ticker,
)

logger = logging.getLogger(__name__)

_OFFICIAL_TRANSACTION_SOURCES = frozenset({"house_pdf", "gemini_ocr", "senate_efd"})


def _analysis_transactions(transaction_source, year: int) -> pd.DataFrame:
    """Read canonical official/legacy rows without chamber-specific filtering."""
    trades = transaction_source.db.get_transactions(year)
    if trades.empty or "source" not in trades.columns:
        return trades
    source = trades["source"]
    return trades[source.isna() | source.isin(_OFFICIAL_TRANSACTION_SOURCES)].copy()


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
    horizons: tuple[int, ...] = (90,)
    threshold: float = 5.0
    days_back: int = CONSENSUS_LOOKBACK_DAYS
    min_buyers: int = CONSENSUS_MIN_BUYERS
    top_n: int = 15
    training_lookback_days: int = 1095
    as_of_date: date | None = None


@dataclass(frozen=True, slots=True)
class TickerAnalysisParams:
    ticker: str
    year: int
    days_back: int = CONSENSUS_LOOKBACK_DAYS
    min_buyers: int = CONSENSUS_MIN_BUYERS
    as_of_date: date | None = None


@dataclass(frozen=True, slots=True)
class BacktestParams:
    start_date: date
    end_date: date
    horizon: int = 60
    # Match the live ticker candidate window. The evaluation horizon is separate.
    lookback_days: int = CONSENSUS_LOOKBACK_DAYS
    # Research validation uses these only for explicit historical scoring modes.
    # The production consensus replay does not consume member-training knobs.
    training_lookback_days: int = 365
    min_buyers: int = CONSENSUS_MIN_BUYERS
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


def _execution_price_window(
    first_decision_date,
    last_decision_date,
    horizon: int,
) -> tuple[date, date]:
    """Return the exact price dates needed by the execution convention."""
    first_entry = next_nyse_session(first_decision_date)
    last_entry = next_nyse_session(last_decision_date)
    last_exit = previous_nyse_session(last_entry + timedelta(days=horizon))
    return first_entry.date(), last_exit.date()


def prepare_analysis_data(
    transaction_source, price_source, year: int, horizons: tuple[int, ...]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trades = _analysis_transactions(transaction_source, year)
    logger.info("Loaded %d canonical official transactions for %d", len(trades), year)

    if len(trades) == 0:
        raise DataSourceError("No trading data found")

    # Filter out transactions with NULL tickers (from parser failures)
    trades = trades[trades["ticker"].notna()].copy()
    logger.info("After filtering NULL tickers: %d transactions", len(trades))

    if trades.empty:
        raise DataSourceError("No valid tickers found in transaction data")

    first_disclosure = pd.Timestamp(trades["disclosure_date"].min()).normalize()
    last_disclosure = pd.Timestamp(trades["disclosure_date"].max()).normalize()
    price_start, price_end = _execution_price_window(
        first_disclosure,
        last_disclosure,
        max(horizons),
    )

    prices = price_source.get_prices(
        trades["ticker"].unique(), price_start, price_end
    )
    logger.info("Fetched price data for %d tickers", len(prices.columns))

    # Use the exact acquired matrix. In read-only analysis the price source may
    # fetch observations that are intentionally not written back to DuckDB.
    entry_prices = _entry_prices_from_matrix(trades, prices)
    logger.info("Computed entry prices for %d transactions", len(entry_prices))

    signals = analysis.calculate_signal_potential(entry_prices, prices, horizons)
    logger.info("Calculated %d signals", len(signals))

    return trades, prices, signals


@pipeline_step
def run_fetch_pipeline(transaction_source, year: int) -> DataResult:
    transaction_source.fetch_and_cache_pdfs(year)
    logger.info("Successfully fetched PDFs for %d", year)
    return DataResult(success=True, data=None)


def prepare_live_consensus_data(
    transaction_source,
    as_of_date: pd.Timestamp,
    days_back: int,
) -> pd.DataFrame:
    """Load only the public transactions needed for live consensus.

    Consensus scoring is a transaction-count decision and does not use forward
    price labels. Keeping this path separate from historical outcome analysis
    prevents a live recommendation from acquiring data that it never consumes.
    The date filtering is repeated after the repository query so mocked or
    alternate transaction sources cannot make future disclosures visible.
    """
    as_of = pd.Timestamp(as_of_date).normalize()
    history_start = as_of - timedelta(days=days_back)
    trades = transaction_source.db.get_transactions_by_date_range(history_start, as_of)
    if trades.empty:
        raise DataSourceError("No trading data found through as-of date")

    disclosure_dates = pd.to_datetime(trades["disclosure_date"], errors="coerce")
    trades = trades[
        trades["ticker"].notna()
        & disclosure_dates.notna()
        & (disclosure_dates >= history_start)
        & (disclosure_dates <= as_of)
    ].copy()
    if trades.empty:
        raise DataSourceError("No valid tickers found through as-of date")
    return trades


@pipeline_step
def run_parse_pipeline(transaction_source, year: int) -> DataResult:
    transaction_source.parse_cached_pdfs(year)
    generation_id = transaction_source.db.get_latest_house_generation(year)
    if generation_id is None:
        raise DataSourceError(
            f"No acquired House generation exists for archive {year}"
        )
    unresolved = transaction_source.db.get_unresolved_house_doc_ids(
        year, generation_id
    )
    if unresolved:
        raise DataSourceError(
            f"House archive {year} generation {generation_id} has "
            f"{len(unresolved)} unresolved artifacts"
        )
    transaction_source.db.mark_house_generation_parse_complete(year, generation_id)
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
    """Display the same eligible buyers consumed by consensus scoring."""
    purchases = _get_consensus_ticker_purchases(ticker, trades)
    if "member" in purchases.columns:
        purchases = purchases[purchases["member"].notna()].copy()
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
    params: TickerAnalysisParams, transaction_source
) -> DataResult:
    if params.days_back < 1 or params.min_buyers < 1:
        raise DataSourceError("days_back and min_buyers must be positive")
    analysis_as_of = pd.Timestamp(
        params.as_of_date or min(date.today(), date(params.year, 12, 31))
    ).normalize()
    if analysis_as_of.year != params.year:
        raise DataSourceError("year must match the ticker analysis as-of date year")

    trades = _analysis_transactions(transaction_source, params.year)
    disclosure_dates = pd.to_datetime(trades["disclosure_date"], errors="coerce")
    cutoff = analysis_as_of - timedelta(days=params.days_back)
    known_trades = trades[
        disclosure_dates.notna()
        & (disclosure_dates >= cutoff)
        & (disclosure_dates <= analysis_as_of)
    ].copy()

    try:
        resolved_ticker = _resolve_consensus_ticker(params.ticker, analysis_as_of)
    except AnalyzerError:
        resolution = TickerResolver().resolve(params.ticker, analysis_as_of.date())
        if resolution.status != "pre_listing":
            raise
        # An explicit query before a symbol's listing is a valid zero-signal
        # question, not a pipeline error. Candidate discovery still excludes
        # the symbol because no eligible purchase row resolves at this cutoff.
        resolved_ticker = str(params.ticker).strip().upper()

    buyers = _consensus_buyers_table(resolved_ticker, known_trades)
    score = analysis.score_ticker_by_buyers(
        resolved_ticker,
        known_trades,
        member_rankings=None,
        min_buyers=params.min_buyers,
        scoring_mode="consensus",
        as_of_date=analysis_as_of,
    )

    return DataResult(
        success=True,
        data={
            "buyers": buyers,
            "score": score,
            "ticker": resolved_ticker,
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

    trades = prepare_live_consensus_data(
        transaction_source,
        as_of_date,
        params.days_back,
        history_lookback_days=params.training_lookback_days + max(params.horizons),
    )
    # Consensus is deliberately transaction-only: no prices, no forward labels.
    cutoff_date = as_of_date - timedelta(days=params.days_back)
    disclosure_dates = pd.to_datetime(trades["disclosure_date"])
    recent_trades = trades[
        (disclosure_dates >= cutoff_date) & (disclosure_dates <= as_of_date)
    ]
    logger.info(
        "Loaded %d disclosures through %s; %d in the last %d days",
        len(trades),
        as_of_date.date(),
        len(recent_trades),
        params.days_back,
    )

    tickers = _get_consensus_candidate_tickers(
        recent_trades,
        params.min_buyers,
        as_of_date=as_of_date,
    )
    logger.info(
        "Found %d tickers with %d+ distinct buyers", len(tickers), params.min_buyers
    )

    scores = [
        analysis.score_ticker_by_buyers(
            ticker,
            recent_trades,
            member_rankings=None,
            min_buyers=params.min_buyers,
            scoring_mode="consensus",
            as_of_date=as_of_date,
        )
        for ticker in tickers
    ]

    if not scores:
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
    if "signal_score_raw" not in result.columns:
        result = result.iloc[0:0]
    else:
        result = result[
            pd.to_numeric(result["signal_score_raw"], errors="coerce").fillna(0) > 0
        ]
    result = result.sort_values(
        ["signal_score", "ticker"], ascending=[False, True]
    ).head(params.top_n)
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

    matrix = _normalize_price_index(
        prices,
        duplicate_error=DataSourceError,
        duplicate_message="Price matrix contains duplicate calendar dates",
    )

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
        raw_ticker = str(transaction["ticker"]).strip().upper()
        disclosure = pd.Timestamp(transaction["disclosure_date"])
        transaction_date = transaction.get("transaction_date")
        if transaction_date is not None and not pd.isna(transaction_date):
            transaction_date = pd.Timestamp(transaction_date).date()
        else:
            transaction_date = None

        if raw_ticker in resolver.LISTING_START_MAP:
            listing_resolution = resolver.resolve(raw_ticker, disclosure.date())
            if listing_resolution.status == "pre_listing":
                continue

        if raw_ticker in resolver.RENAME_MAP:
            if transaction_date is None:
                continue
            price_ticker = resolver.resolve(raw_ticker, transaction_date).price_symbol
            if price_ticker not in price_columns:
                continue
        else:
            price_ticker = raw_ticker
            if price_ticker not in price_columns:
                resolved = resolver.resolve(raw_ticker, transaction_date).price_symbol
                if resolved not in price_columns:
                    continue
                price_ticker = resolved

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


def _benchmark_return(
    prices: pd.DataFrame, as_of_date: pd.Timestamp, horizon: int
) -> float | None:
    """Return the executable SPY return for one scheduled backtest date.

    Route benchmark calculations through the production evaluator so entry,
    exit, session alignment, and slippage remain identical to recommendation
    evaluation.  The direct import is only a compatibility fallback for
    callers that replace the analysis facade in tests or integrations and
    return a frame without the benchmark column.
    """
    recommendation = pd.DataFrame(
        [
            {
                "rank": 1,
                "ticker": "SPY",
                "signal_score": 1.0,
                "instrument_type": "stock",
            }
        ]
    )
    try:
        evaluated = analysis.evaluate_backtest(
            recommendation, prices, as_of_date, horizon
        )
    except (AnalyzerError, KeyError):
        return None

    value = _benchmark_value(evaluated)
    if value is not None:
        return value

    # ``analysis.evaluate_backtest`` is a facade re-export.  Bypass a facade
    # replacement only when it did not return the shared benchmark field; the
    # normal production path above remains the single source of arithmetic.
    try:
        from analyzer.backtest.evaluate import evaluate_backtest

        evaluated = evaluate_backtest(recommendation, prices, as_of_date, horizon)
    except (AnalyzerError, KeyError, TypeError, ValueError):
        return None
    return _benchmark_value(evaluated)


def _benchmark_value(evaluated: pd.DataFrame) -> float | None:
    if evaluated.empty or "bt_spy_return_pct" not in evaluated.columns:
        return None
    value = evaluated["bt_spy_return_pct"].iloc[0]
    return float(value) if pd.notna(value) else None


def _cash_observation(
    as_of_date: pd.Timestamp,
    benchmark_return: float,
    horizon: int,
    *,
    recommendation_count: int = 0,
    reason: str = "no_recommendations",
) -> dict:
    """Build a zero-return observation for supported dates without a trade."""
    entry_date = next_nyse_session(as_of_date)
    exit_date = previous_nyse_session(entry_date + timedelta(days=horizon))
    return {
        "as_of_date": as_of_date.date(),
        "rank": None,
        "ticker": None,
        "num_buyers": 0,
        "signal_score": 0.0,
        "recommendation_count": recommendation_count,
        "evaluable_recommendation_count": 0,
        "status": "cash",
        "reason": reason,
        "benchmark_status": "available",
        "benchmark_supported": True,
        "cash_observation": True,
        "traded": False,
        "strategy_return_pct": 0.0,
        "portfolio_return_pct": 0.0,
        "spy_return_pct": benchmark_return,
        "net_alpha_pct": -benchmark_return,
        "bt_entry_date": entry_date.date(),
        "bt_exit_date": exit_date.date(),
        "bt_entry_price": np.nan,
        "bt_exit_price": np.nan,
        "bt_raw_return_pct": 0.0,
        "bt_return_pct": 0.0,
        "bt_leverage": 1.0,
        "bt_spy_return_pct": benchmark_return,
        "bt_alpha_pct": -benchmark_return,
        "bt_horizon_days": horizon,
        "bt_entry_delay": (entry_date - as_of_date).days,
        "bt_delisted": False,
        "bt_coverage": "cash",
        "bt_unavailable_reason": None,
        "bt_stale_exit": False,
    }


def _date_observation(
    as_of_date: pd.Timestamp,
    benchmark_return: float,
    strategy_return: float,
    recommendation_count: int,
    evaluable_recommendation_count: int,
    *,
    status: str,
    reason: str | None = None,
) -> dict:
    """Build the one-row-per-supported-date backtest accounting record."""
    return {
        "as_of_date": as_of_date.date(),
        "strategy_return_pct": strategy_return,
        "portfolio_return_pct": strategy_return,
        "spy_return_pct": benchmark_return,
        "net_alpha_pct": strategy_return - benchmark_return,
        "recommendation_count": recommendation_count,
        "evaluable_recommendation_count": evaluable_recommendation_count,
        "status": status,
        "reason": reason,
        "benchmark_status": "available",
        "benchmark_supported": True,
        "traded": status == "invested",
    }


@pipeline_step
def run_backtest_pipeline(
    params: BacktestParams,
    transaction_source,
    price_source,
    data_dir: Path | None = None,
) -> DataResult:
    tx_start = params.start_date - timedelta(days=params.lookback_days)
    tx_end = params.end_date

    all_transactions = transaction_source.db.get_transactions_by_date_range(
        tx_start, tx_end
    )
    if all_transactions.empty:
        raise DataSourceError(f"No transactions found between {tx_start} and {tx_end}")

    logger.info("Loaded %d transactions for backtest window", len(all_transactions))

    as_of_dates = pd.date_range(
        params.start_date, params.end_date, freq=f"{params.frequency_days}D"
    )
    price_start, price_end = _execution_price_window(
        as_of_dates[0],
        as_of_dates[-1],
        params.horizon,
    )
    all_tickers = sorted(set(_get_consensus_price_tickers(all_transactions)) | {"SPY"})

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

    # Consensus replay is a public-disclosure rule. Historical outcome labels
    # are evaluation data, not decision inputs, so do not build them here.
    signals = pd.DataFrame()

    all_results = []
    date_observations = []
    # Finding 1 fix: accumulate per-date attrs counts explicitly because
    # pd.concat of DataFrames with differing .attrs yields attrs={} in
    # pandas 3.x.  We sum here and set them on the combined frame.
    total_no_price = 0
    total_delisted = 0
    total_unavailable = 0
    for as_of in as_of_dates:
        as_of_ts = pd.Timestamp(as_of)

        # A scheduled date belongs to the backtest support only when the same
        # executable SPY window used by recommendation evaluation exists.
        # Unsupported benchmark dates remain excluded; supported no-trade dates
        # are represented explicitly as cash below.
        benchmark_return = _benchmark_return(prices, as_of_ts, params.horizon)
        if benchmark_return is None:
            continue

        recs = analysis.backtest_recommendations(
            signals,
            all_transactions,
            as_of_ts,
            horizon=params.horizon,
            lookback_days=params.lookback_days,
            min_buyers=params.min_buyers,
            top_n=params.top_n,
        )

        if recs.empty:
            date_observations.append(
                _date_observation(
                    as_of_ts,
                    benchmark_return,
                    0.0,
                    0,
                    0,
                    status="cash",
                    reason="no_recommendations",
                )
            )
            all_results.append(
                pd.DataFrame(
                    [_cash_observation(as_of_ts, benchmark_return, params.horizon)]
                )
            )
            continue

        evaluated = analysis.evaluate_backtest(recs, prices, as_of_ts, params.horizon)
        total_no_price += evaluated.attrs.get("n_no_price", 0)
        total_delisted += evaluated.attrs.get("n_delisted", 0)
        total_unavailable += evaluated.attrs.get("n_unavailable", 0)
        valid_evaluated = evaluated.dropna(subset=["bt_return_pct"]).copy()
        if valid_evaluated.empty:
            date_observations.append(
                _date_observation(
                    as_of_ts,
                    benchmark_return,
                    0.0,
                    len(recs),
                    0,
                    status="cash",
                    reason="no_evaluable_recommendations",
                )
            )
            all_results.append(
                pd.DataFrame(
                    [
                        _cash_observation(
                            as_of_ts,
                            benchmark_return,
                            params.horizon,
                            recommendation_count=len(recs),
                            reason="no_evaluable_recommendations",
                        )
                    ]
                )
            )
            continue
        if len(valid_evaluated) != len(recs):
            # Fail-closed: an incomplete funded basket cannot reallocate
            # ex-post capital to measurable outcomes. Hold cash; report
            # survivors nowhere for this date.
            date_observations.append(
                _date_observation(
                    as_of_ts,
                    benchmark_return,
                    0.0,
                    len(recs),
                    len(valid_evaluated),
                    status="cash",
                    reason="incomplete_basket",
                )
            )
            all_results.append(
                pd.DataFrame(
                    [
                        _cash_observation(
                            as_of_ts,
                            benchmark_return,
                            params.horizon,
                            recommendation_count=len(recs),
                            reason="incomplete_basket",
                        )
                    ]
                )
            )
            continue

        strategy_return = float(
            pd.to_numeric(valid_evaluated["bt_return_pct"], errors="coerce").mean()
        )
        date_observations.append(
            _date_observation(
                as_of_ts,
                benchmark_return,
                strategy_return,
                len(recs),
                len(valid_evaluated),
                status="invested",
            )
        )
        valid_evaluated.insert(0, "as_of_date", as_of_ts.date())
        valid_evaluated["recommendation_count"] = len(recs)
        valid_evaluated["status"] = "invested"
        valid_evaluated["reason"] = None
        valid_evaluated["benchmark_status"] = "available"
        valid_evaluated["benchmark_supported"] = True
        valid_evaluated["cash_observation"] = False
        valid_evaluated["traded"] = True
        valid_evaluated["strategy_return_pct"] = strategy_return
        valid_evaluated["portfolio_return_pct"] = strategy_return
        valid_evaluated["spy_return_pct"] = benchmark_return
        valid_evaluated["net_alpha_pct"] = strategy_return - benchmark_return
        all_results.append(valid_evaluated)

    if not all_results:
        return DataResult(
            success=True,
            data={
                "combined": pd.DataFrame(),
                "summary": pd.DataFrame(),
                "snapshot": snapshot,
                "evaluable_dates": 0,
                "total_as_of_dates": len(as_of_dates),
                "date_observations": pd.DataFrame(),
            },
        )

    combined = pd.concat(all_results, ignore_index=True)
    # Set accumulated coverage counts on the combined frame so summarize_backtest
    # can propagate them (Finding 1: pd.concat drops attrs in pandas 3.x).
    combined.attrs["n_no_price"] = total_no_price
    combined.attrs["n_delisted"] = total_delisted
    combined.attrs["n_unavailable"] = total_unavailable

    spy_prices = prices["SPY"] if "SPY" in prices.columns else None
    summary = analysis.summarize_backtest(combined, spy_prices)

    observations = pd.DataFrame(date_observations)
    if not observations.empty:
        summary.attrs["benchmark_supported_dates"] = len(observations)
        summary.attrs["no_recommendation_dates"] = int(
            (observations["status"] == "cash").sum()
        )
        summary.attrs["mean_net_alpha_pct"] = round(
            float(observations["net_alpha_pct"].mean()), 2
        )
        portfolio_rows = summary[summary["rank"] == "PORTFOLIO"]
        if not portfolio_rows.empty:
            portfolio_index = portfolio_rows.index[0]
            summary.loc[portfolio_index, "count"] = len(observations)
            summary.loc[portfolio_index, "recommendation_count"] = int(
                observations["evaluable_recommendation_count"].sum()
            )
    evaluable_dates = len(observations)
    total_as_of_dates = len(as_of_dates)

    # Save snapshot alongside backtest results when a destination is given.
    if data_dir is not None:
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
            "date_observations": observations,
        },
    )
