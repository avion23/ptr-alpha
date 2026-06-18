#!/usr/bin/env python3

import sys
import logging
from datetime import date
from pathlib import Path
from dataclasses import dataclass
import typer

from analyzer.pipeline import (
    run_fetch_pipeline,
    run_parse_pipeline,
    run_analysis_pipeline,
    run_sales_pipeline,
    run_ticker_analysis,
    run_recent_ticker_scoring,
    run_backtest_pipeline,
    AnalysisParams,
    TickerAnalysisParams,
    TickerScoringParams,
    BacktestParams,
)
from analyzer.price_snapshot import create_snapshot, save_snapshot
from analyzer.exceptions import AnalyzerError
from analyzer.settings import Settings
from analyzer.datasources import HouseTransactionSource, YFinancePriceSource

app = typer.Typer(help="Congressional PTR disclosure analyzer", no_args_is_help=True)


@dataclass
class AppContext:
    settings: Settings
    transaction_source: HouseTransactionSource
    price_source: YFinancePriceSource


def setup_logging(verbose):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def get_context(ctx, data_dir=None, read_only=False):
    if ctx.obj is None:
        settings = Settings()
        if data_dir and data_dir != "data":
            settings.data.data_dir = data_dir
        ctx.obj = AppContext(
            settings=settings,
            transaction_source=HouseTransactionSource(settings, read_only=read_only),
            price_source=YFinancePriceSource(settings, read_only=read_only),
        )
    return ctx.obj


@app.callback()
def main_callback(
    ctx: typer.Context,
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
):
    setup_logging(verbose)


@app.command()
def analyze(
    ctx: typer.Context,
    year: int = typer.Option(2025, help="Year to process"),
    source: str = typer.Option("house", help="Data source"),
    mode: str = typer.Option(
        "ranks",
        help="Output mode: ranks | signals | member | sales | tickers",
    ),
    member: str | None = typer.Option(None, help="Filter to specific member"),
    ticker: str | None = typer.Option(None, help="Analyze specific ticker"),
    horizons: list[int] = typer.Option([90], help="Time horizons in days"),
    threshold: float = typer.Option(5.0, help="Hit rate threshold percentage"),
    days_back: int = typer.Option(28, help="Days back for ticker scoring"),
    min_buyers: int = typer.Option(3, help="Minimum buyers for ticker scoring"),
    top_n: int = typer.Option(20, help="Number of results to show"),
    output: str = typer.Option("console", help="Output format: console or csv"),
    data_dir: str = typer.Option("data", help="Data directory"),
):
    """
    Unified analysis command. Use --mode to select output type:

      ranks    - Rank members by trading performance (default)
      signals  - Show top trading signals
      member   - Show signals for specific member (use --member)
      sales    - Rank members by loss avoidance (sale performance)
      tickers  - Score multi-buyer tickers from recent period
    """
    app_ctx = get_context(ctx, data_dir, read_only=False)

    if ticker:
        params = TickerAnalysisParams(ticker=ticker, year=year, horizon=horizons[0], threshold=threshold)
        success = run_ticker_analysis(
            params,
            app_ctx.transaction_source,
            app_ctx.price_source,
        )
        raise typer.Exit(0 if success else 1)

    if mode == "tickers":
        params = TickerScoringParams(
            year=year,
            horizons=horizons,
            threshold=threshold,
            days_back=days_back,
            min_buyers=min_buyers,
            top_n=top_n,
        )
        success = run_recent_ticker_scoring(
            app_ctx.transaction_source,
            app_ctx.price_source,
            params,
        )
        raise typer.Exit(0 if success else 1)

    if mode == "sales":
        data_path = Path(app_ctx.settings.data.data_dir)
        success = run_sales_pipeline(
            year, horizons, top_n,
            app_ctx.transaction_source, app_ctx.price_source, data_path, output
        )
        raise typer.Exit(0 if success else 1)

    show_signals = mode == "signals"
    params = AnalysisParams(source, year, horizons, threshold, member, top_n, show_signals)
    data_path = Path(app_ctx.settings.data.data_dir)
    success = run_analysis_pipeline(
        params, app_ctx.transaction_source, app_ctx.price_source, data_path, output
    )
    raise typer.Exit(0 if success else 1)



@app.command()
def fetch(
    ctx: typer.Context,
    year: int = typer.Option(2025, help="Year to process"),
    data_dir: str = typer.Option("data", help="Data directory"),
    refresh_metadata: bool = typer.Option(False, "--refresh-metadata", help="Force refresh of metadata from House Clerk"),
):
    """Download House PDFs for a year"""
    app_ctx = get_context(ctx, data_dir, read_only=False)
    if refresh_metadata:
        app_ctx.transaction_source.fetch_metadata(year, refresh=True)
    success = run_fetch_pipeline(app_ctx.transaction_source, year)
    raise typer.Exit(0 if success else 1)


@app.command()
def parse(
    ctx: typer.Context,
    year: int = typer.Option(2025, help="Year to process"),
    data_dir: str = typer.Option("data", help="Data directory"),
):
    """Parse cached PDFs to database"""
    app_ctx = get_context(ctx, data_dir, read_only=False)
    success = run_parse_pipeline(app_ctx.transaction_source, year)
    raise typer.Exit(0 if success else 1)


@app.command()
def backtest(
    ctx: typer.Context,
    start: str = typer.Option(..., help="Backtest start date (YYYY-MM-DD)"),
    end: str = typer.Option(..., help="Backtest end date (YYYY-MM-DD)"),
    horizon: int = typer.Option(120, help="Forward return horizon in days"),
    lookback_days: int = typer.Option(60, help="Candidate purchase lookback window in days"),
    training_lookback_days: int = typer.Option(365, help="Training data lookback window in days"),
    min_buyers: int = typer.Option(2, help="Minimum buyers for a candidate ticker"),
    top_n: int = typer.Option(5, help="Top N recommendations per backtest date"),
    threshold: float = typer.Option(5.0, help="Hit rate threshold percentage"),
    frequency_days: int = typer.Option(30, help="Days between rolling backtest dates"),
    data_dir: str = typer.Option("data", help="Data directory"),
):
    """
    Run a rolling backtest of the recommendation algorithm.

    Simulates what the algorithm would have recommended at each date between
    --start and --end (stepped by --frequency-days), then evaluates the
    forward returns of those picks over --horizon days.

    Uses only data that would have been available at each as-of date
    (no lookahead). Member rankings are built from fully-elapsed signal
    windows only.
    """
    try:
        start_date = date.fromisoformat(start)
        end_date = date.fromisoformat(end)
    except ValueError:
        print("Error: dates must be in YYYY-MM-DD format", file=sys.stderr)
        raise typer.Exit(1)

    if end_date < start_date:
        print("Error: --end must be on or after --start", file=sys.stderr)
        raise typer.Exit(1)

    app_ctx = get_context(ctx, data_dir, read_only=True)
    params = BacktestParams(
        start_date=start_date,
        end_date=end_date,
        horizon=horizon,
        lookback_days=lookback_days,
        training_lookback_days=training_lookback_days,
        min_buyers=min_buyers,
        top_n=top_n,
        threshold=threshold,
        frequency_days=frequency_days,
    )
    success = run_backtest_pipeline(params, app_ctx.transaction_source, app_ctx.price_source)
    raise typer.Exit(0 if success else 1)


@app.command()
def snapshot(
    ctx: typer.Context,
    data_dir: str = typer.Option("data", help="Data directory"),
    output: str = typer.Option("data/price_snapshot.json", help="Output path for snapshot JSON"),
):
    """Create a frozen price snapshot manifest for reproducible backtests."""
    app_ctx = get_context(ctx, data_dir, read_only=True)

    from analyzer.settings import Settings
    settings = Settings()
    if data_dir and data_dir != "data":
        settings.data.data_dir = data_dir

    db = app_ctx.transaction_source.db
    tickers_result = db.conn.execute("SELECT DISTINCT ticker FROM prices").fetchall()
    all_tickers = sorted({row[0] for row in tickers_result})

    if not all_tickers:
        print("No price data found in database")
        raise typer.Exit(1)

    date_range = db.conn.execute(
        "SELECT MIN(date), MAX(date) FROM prices"
    ).fetchone()
    start_date = date_range[0]
    end_date = date_range[1]

    snap = create_snapshot(db, all_tickers, start_date, end_date)
    save_snapshot(snap, output)

    print(f"Snapshot created: {snap.snapshot_id}")
    print(f"  Created at:     {snap.created_at}")
    print(f"  Git SHA:        {snap.git_sha[:12]}")
    print(f"  yfinance:       {snap.yfinance_version}")
    print(f"  Python:         {snap.python_version}")
    print(f"  Tickers:        {snap.resolved_tickers}/{snap.requested_tickers} resolved")
    if snap.unresolved_tickers:
        print(f"  Unresolved:     {', '.join(snap.unresolved_tickers[:10])}")
    print(f"  Price rows:     {snap.price_rows}")
    print(f"  Date range:     {snap.first_date} to {snap.last_date}")
    print(f"  Saved to:       {output}")
    raise typer.Exit(0)


def main():
    try:
        app()
    except AnalyzerError as e:
        print(f"Error: {e}", file=sys.stderr)
        raise typer.Exit(1)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user", file=sys.stderr)
        raise typer.Exit(130)
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.exception(f"Unexpected error: {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    main()
