#!/usr/bin/env python3

import sys
import logging
from pathlib import Path
from dataclasses import dataclass
import typer
from analyzer.pipeline import (
    run_fetch_pipeline,
    run_parse_pipeline,
    run_analysis_pipeline,
    run_ticker_analysis,
    run_recent_ticker_scoring,
    AnalysisParams,
)
from analyzer.exceptions import AnalyzerError
from analyzer.settings import Settings
from analyzer.datasources import HouseTransactionSource, YFinancePriceSource

app = typer.Typer(help="Congressional insider trading analyzer", no_args_is_help=True)


@dataclass
class AppContext:
    settings: object
    transaction_source: HouseTransactionSource
    price_source: YFinancePriceSource


def setup_logging(verbose):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def get_context(ctx, data_dir=None):
    if ctx.obj is None:
        settings = Settings()
        if data_dir and data_dir != "data":
            settings.data.data_dir = data_dir
        ctx.obj = AppContext(
            settings=settings,
            transaction_source=HouseTransactionSource(settings),
            price_source=YFinancePriceSource(settings),
        )
    return ctx.obj


def _get_analysis_context(
    ctx, data_dir, year, horizons, threshold, member, top_n, show_signals, output
):
    app_ctx = get_context(ctx, data_dir)
    params = AnalysisParams(
        "house", year, horizons, threshold, member, top_n, show_signals, output
    )
    data_path = Path(app_ctx.settings.data.data_dir)
    return app_ctx, params, data_path


@app.callback()
def main_callback(
    ctx: typer.Context,
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
):
    setup_logging(verbose)


@app.command()
def fetch(
    ctx: typer.Context,
    year: int = typer.Option(2025, help="Year to process"),
    data_dir: str = typer.Option("data", help="Data directory"),
    refresh_metadata: bool = typer.Option(
        False, "--refresh-metadata", help="Force refresh of metadata from House Clerk"
    ),
):
    """Download House PDFs for a year"""
    app_ctx = get_context(ctx, data_dir)
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
    """Parse cached PDFs to Parquet"""
    app_ctx = get_context(ctx, data_dir)
    success = run_parse_pipeline(app_ctx.transaction_source, year)
    raise typer.Exit(0 if success else 1)


@app.command()
def rank_members(
    ctx: typer.Context,
    year: int = typer.Option(2025, help="Year to process"),
    source: str = typer.Option("house", help="Data source"),
    horizons: list[int] = typer.Option([90], help="Time horizons in days"),
    threshold: float = typer.Option(5.0, help="Hit rate threshold percentage"),
    top_n: int = typer.Option(20, help="Number of results to show"),
    output: str = typer.Option("console", help="Output format: console or csv"),
    data_dir: str = typer.Option("data", help="Data directory"),
):
    """Rank congressional members by trading performance"""
    app_ctx, params, data_path = _get_analysis_context(
        ctx, data_dir, year, horizons, threshold, None, top_n, False, output
    )
    success = run_analysis_pipeline(
        params, app_ctx.transaction_source, app_ctx.price_source, data_path, output
    )
    raise typer.Exit(0 if success else 1)


@app.command()
def show_signals(
    ctx: typer.Context,
    year: int = typer.Option(2025, help="Year to process"),
    source: str = typer.Option("house", help="Data source"),
    horizons: list[int] = typer.Option([90], help="Time horizons in days"),
    threshold: float = typer.Option(5.0, help="Hit rate threshold percentage"),
    top_n: int = typer.Option(15, help="Number of results to show"),
    output: str = typer.Option("console", help="Output format: console or csv"),
    data_dir: str = typer.Option("data", help="Data directory"),
):
    """Show top trading signals"""
    app_ctx, params, data_path = _get_analysis_context(
        ctx, data_dir, year, horizons, threshold, None, top_n, True, output
    )
    success = run_analysis_pipeline(
        params, app_ctx.transaction_source, app_ctx.price_source, data_path, output
    )
    raise typer.Exit(0 if success else 1)


@app.command()
def show_member_signals(
    ctx: typer.Context,
    member: str = typer.Argument(..., help="Member name to analyze"),
    year: int = typer.Option(2025, help="Year to process"),
    source: str = typer.Option("house", help="Data source"),
    horizons: list[int] = typer.Option([90], help="Time horizons in days"),
    threshold: float = typer.Option(5.0, help="Hit rate threshold percentage"),
    top_n: int = typer.Option(10, help="Number of results to show"),
    output: str = typer.Option("console", help="Output format: console or csv"),
    data_dir: str = typer.Option("data", help="Data directory"),
):
    """Show signals for a specific member"""
    app_ctx, params, data_path = _get_analysis_context(
        ctx, data_dir, year, horizons, threshold, member, top_n, False, output
    )
    success = run_analysis_pipeline(
        params, app_ctx.transaction_source, app_ctx.price_source, data_path, output
    )
    raise typer.Exit(0 if success else 1)


@app.command()
def analyze_ticker(
    ctx: typer.Context,
    ticker: str = typer.Argument(..., help="Ticker symbol to analyze"),
    year: int = typer.Option(2025, help="Year to process"),
    source: str = typer.Option("house", help="Data source"),
    horizon: int = typer.Option(90, help="Time horizon in days"),
    threshold: float = typer.Option(5.0, help="Hit rate threshold percentage"),
    data_dir: str = typer.Option("data", help="Data directory"),
):
    """Show all buyers of a ticker with rankings and signal score"""
    app_ctx = get_context(ctx, data_dir)
    success = run_ticker_analysis(
        ticker,
        app_ctx.transaction_source,
        app_ctx.price_source,
        year,
        horizon,
        threshold,
    )
    raise typer.Exit(0 if success else 1)


@app.command()
def score_recent_tickers(
    ctx: typer.Context,
    year: int = typer.Option(2025, help="Year to process"),
    source: str = typer.Option("house", help="Data source"),
    horizons: list[int] = typer.Option([90], help="Time horizons in days"),
    threshold: float = typer.Option(5.0, help="Hit rate threshold percentage"),
    days_back: int = typer.Option(28, help="How many days back to analyze"),
    min_buyers: int = typer.Option(2, help="Minimum number of buyers required"),
    top_n: int = typer.Option(15, help="Number of top signals to show"),
    data_dir: str = typer.Option("data", help="Data directory"),
):
    """Score multi-buyer tickers from recent period"""
    app_ctx = get_context(ctx, data_dir)
    success = run_recent_ticker_scoring(
        app_ctx.transaction_source,
        app_ctx.price_source,
        year,
        horizons,
        threshold,
        days_back,
        min_buyers,
        top_n,
    )
    raise typer.Exit(0 if success else 1)


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
        import traceback

        print(f"Unexpected error: {e}", file=sys.stderr)
        traceback.print_exc()
        raise typer.Exit(1)


if __name__ == "__main__":
    main()
