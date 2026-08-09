#!/usr/bin/env python3

import json
import sys
import logging
from datetime import date
from pathlib import Path
from dataclasses import dataclass
import pandas as pd
import typer

from analyzer.pipeline import (
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
from analyzer.database import Database
from analyzer.datasources import HouseTransactionSource, YFinancePriceSource
from analyzer.models import AnalysisMode

app = typer.Typer(help="Congressional PTR disclosure analyzer", no_args_is_help=True)
logger = logging.getLogger(__name__)
_CURRENT_YEAR = date.today().year
_HOUSE_PTR_FIRST_ARCHIVE_YEAR = 2015
_HOUSE_LEGACY_FIRST_ARCHIVE_YEAR = 2008
_BACKTEST_DEFAULTS = {
    name: field.default for name, field in BacktestParams.__dataclass_fields__.items()
}


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
        # Share a single DuckDB connection across both data sources to avoid
        # two independent Database instances pointing at the same file.
        shared_db = Database(
            Path(settings.data.data_dir) / "congress.duckdb", read_only=read_only
        )
        ctx.obj = AppContext(
            settings=settings,
            transaction_source=HouseTransactionSource(
                settings, read_only=read_only, db=shared_db
            ),
            price_source=YFinancePriceSource(
                settings, read_only=read_only, db=shared_db
            ),
        )
    return ctx.obj


def _save_results(
    table: pd.DataFrame,
    output_format: str,
    mode: AnalysisMode,
    member_filter: str | None,
    data_dir: Path,
) -> None:
    match mode:
        case AnalysisMode.MEMBER_SIGNALS | AnalysisMode.TOP_SIGNALS:
            display_cols = [
                "member",
                "ticker",
                "disclosure_date",
                "spy_alpha_pct",
                "peak_potential_pct",
                "total_return_pct",
                "total_spy_alpha_pct",
                "signal_score",
            ]
        case AnalysisMode.SALE_RANKINGS:
            display_cols = [
                "member",
                "avg_loss_avoided_pct",
                "median_loss_avoided_pct",
                "sale_trades",
                "sharpe_ratio",
                "bayes_win_prob",
                "posterior_lift",
                "avg_spy_alpha_pct",
            ]
        case AnalysisMode.MEMBER_RANKINGS:
            display_cols = [
                "member",
                "avg_total_spy_alpha_pct",
                "avg_spy_alpha_pct",
                "bayes_win_prob",
                "posterior_lift",
                "peak_hit_rate_pct",
                "sharpe_ratio",
                "conviction_score",
                "purchase_trades",
            ]
        case _:
            display_cols = list(table.columns)
    available_display = [c for c in display_cols if c in table.columns]
    display_table = table[available_display]

    if output_format == "csv":
        match mode:
            case AnalysisMode.MEMBER_SIGNALS:
                if member_filter is None:
                    raise ValueError("member_filter is required for member signals")
                filename = f"{member_filter.replace(' ', '_').lower()}_signals.csv"
            case AnalysisMode.TOP_SIGNALS:
                filename = "top_signals.csv"
            case AnalysisMode.SALE_RANKINGS:
                filename = "sale_rankings.csv"
            case AnalysisMode.MEMBER_RANKINGS:
                filename = "member_rankings.csv"

        filepath = data_dir / filename
        data_dir.mkdir(parents=True, exist_ok=True)
        display_table.to_csv(filepath, index=False)
        logger.info(f"Results saved to {filepath}")
    else:
        print(display_table.to_string(index=False))


@app.callback()
def main_callback(
    ctx: typer.Context,
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
):
    setup_logging(verbose)


def _validate_mode(mode: str, member: str | None, ticker: str | None) -> None:
    """Validate mode/member/ticker combinations. Exits on error."""
    valid_modes = {"ranks", "signals", "member", "sales", "tickers"}
    if mode not in valid_modes:
        print(f"Error: --mode must be one of {sorted(valid_modes)}", file=sys.stderr)
        raise typer.Exit(1)
    if mode == "member" and member is None and ticker is None:
        print("Error: --mode member requires --member NAME", file=sys.stderr)
        raise typer.Exit(1)
    if mode == "sales" and member is not None:
        print(
            "WARNING: --member flag is ignored for --mode sales (sales rankings are aggregate).",
            file=sys.stderr,
        )


def _validate_positive_options(**options: int | float) -> None:
    """Reject nonsensical numeric CLI inputs before opening the database."""
    for name, value in options.items():
        if value <= 0:
            option = name.replace("_", "-")
            print(f"Error: --{option} must be greater than zero", file=sys.stderr)
            raise typer.Exit(1)


def _validate_output(output: str) -> None:
    if output not in {"console", "csv"}:
        print("Error: --output must be one of ['console', 'csv']", file=sys.stderr)
        raise typer.Exit(1)


def _check_data_freshness(app_ctx: AppContext) -> None:
    """Warn if transaction data looks stale."""
    try:
        _max_date = app_ctx.transaction_source.db.conn.execute(
            "SELECT MAX(disclosure_date) FROM canonical_transactions"
        ).fetchone()[0]
        if _max_date:
            _age = (date.today() - _max_date).days
            if _age > 30:
                print(
                    f"WARNING: Data is {_age} days old (latest: {_max_date}). Run 'ptr-alpha refresh' first.",
                    file=sys.stderr,
                )
    except Exception:
        logger.debug("Freshness check failed", exc_info=True)


def _run_ticker_mode(
    app_ctx: AppContext,
    mode: str,
    ticker: str,
    year: int,
    horizons: list[int],
    threshold: float,
    as_of_date: date | None,
    output: str,
) -> None:
    """Handle --ticker analysis mode."""
    if mode != "ranks":
        print(
            f"WARNING: --mode {mode} is ignored when --ticker is provided; running ticker analysis.",
            file=sys.stderr,
        )
    if output == "csv":
        print(
            "WARNING: CSV output is not supported for --ticker analysis; using console output.",
            file=sys.stderr,
        )
    params = TickerAnalysisParams(
        ticker=ticker,
        year=year,
        horizon=horizons[0],
        threshold=threshold,
        as_of_date=as_of_date,
    )
    result = run_ticker_analysis(
        params,
        app_ctx.transaction_source,
        app_ctx.price_source,
    )
    if result.success and hasattr(result, "data") and result.data:
        print(f"\n=== Buyers of {result.data['ticker']} ===")
        print(result.data["buyers"].to_string(index=False))
        print("\n=== Signal Score ===")
        print(result.data["score"].to_string(index=False))
        score = result.data["score"]["signal_score"].iloc[0]
        verdict = "BUY CANDIDATE" if score > 0 else "NO BUY"
        print(f"\nRecommendation: {verdict} (score {score:.2f})")
    raise typer.Exit(0 if result.success else 1)


def _run_tickers_mode(
    app_ctx: AppContext,
    year: int,
    horizons: list[int],
    threshold: float,
    days_back: int,
    min_buyers: int,
    top_n: int,
    output: str,
    training_lookback_days: int,
    as_of_date: date | None,
) -> None:
    """Handle --mode tickers."""
    if output == "csv":
        print(
            "WARNING: CSV output is not supported for --mode tickers; using console output.",
            file=sys.stderr,
        )
    params = TickerScoringParams(
        year=year,
        horizons=tuple(horizons),
        threshold=threshold,
        days_back=days_back,
        min_buyers=min_buyers,
        top_n=top_n,
        training_lookback_days=training_lookback_days,
        as_of_date=as_of_date,
    )
    result = run_recent_ticker_scoring(
        app_ctx.transaction_source,
        app_ctx.price_source,
        params,
    )
    if result.success and hasattr(result, "data") and result.data:
        data = result.data
        if not data["result"].empty:
            print(
                f"\n=== Current Buy Candidates as of {data['as_of_date']} (Last {data['days_back']} Days, {data['min_buyers']}+ Buyers) ==="
            )
            print(data["result"].to_string(index=False))
        else:
            print(f"\nNo positive buy candidates as of {data['as_of_date']}.")
    raise typer.Exit(0 if result.success else 1)


def _run_sales_mode(
    app_ctx: AppContext,
    year: int,
    horizons: list[int],
    top_n: int,
    output: str,
) -> None:
    """Handle --mode sales."""
    data_path = Path(app_ctx.settings.data.data_dir)
    result = run_sales_pipeline(
        year, tuple(horizons), top_n, app_ctx.transaction_source, app_ctx.price_source
    )
    if result.success and hasattr(result, "data") and result.data:
        _save_results(
            result.data["table"], output, AnalysisMode.SALE_RANKINGS, None, data_path
        )
    raise typer.Exit(0 if result.success else 1)


def _run_analysis_mode(
    app_ctx: AppContext,
    year: int,
    horizons: list[int],
    threshold: float,
    member: str | None,
    top_n: int,
    mode: str,
    output: str,
    sectors: bool,
) -> None:
    """Handle ranks/signals/member modes via run_analysis_pipeline."""
    if member is not None:
        analysis_mode = AnalysisMode.MEMBER_SIGNALS
    elif mode == "signals":
        analysis_mode = AnalysisMode.TOP_SIGNALS
    else:
        analysis_mode = AnalysisMode.MEMBER_RANKINGS
    params = AnalysisParams(
        year=year,
        horizons=tuple(horizons),
        threshold=threshold,
        member_filter=member,
        top_n=top_n,
        mode=analysis_mode,
        include_sector_analysis=sectors,
    )
    data_path = Path(app_ctx.settings.data.data_dir)
    result = run_analysis_pipeline(
        params, app_ctx.transaction_source, app_ctx.price_source
    )
    if result.success and hasattr(result, "data") and result.data:
        _save_results(
            result.data["table"],
            output,
            result.data["mode"],
            result.data["member_filter"],
            data_path,
        )
        if (
            result.data["sector_results"] is not None
            and result.data["mode"] == AnalysisMode.MEMBER_RANKINGS
        ):
            print("\n=== Sector Analysis ===")
            print(result.data["sector_results"].to_string(index=False))
    raise typer.Exit(0 if result.success else 1)


@app.command()
def analyze(
    ctx: typer.Context,
    year: int = typer.Option(2025, help="Year to process"),
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
    training_lookback_days: int = typer.Option(
        1095,
        help="Historical days used to train live ticker rankings",
    ),
    as_of: str | None = typer.Option(
        None,
        help="Analysis cutoff date (YYYY-MM-DD; defaults to today)",
    ),
    sectors: bool = typer.Option(
        False,
        "--sectors",
        help="Fetch optional sector metadata for rank output",
    ),
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
    _validate_mode(mode, member, ticker)
    _validate_positive_options(
        year=year,
        days_back=days_back,
        min_buyers=min_buyers,
        top_n=top_n,
        training_lookback_days=training_lookback_days,
    )
    if not horizons or any(horizon <= 0 for horizon in horizons):
        print("Error: --horizons values must be greater than zero", file=sys.stderr)
        raise typer.Exit(1)
    _validate_output(output)
    try:
        as_of_date = date.fromisoformat(as_of) if as_of else None
    except ValueError:
        print("Error: --as-of must use YYYY-MM-DD", file=sys.stderr)
        raise typer.Exit(1)
    app_ctx = get_context(ctx, data_dir, read_only=False)
    _check_data_freshness(app_ctx)

    if ticker:
        _run_ticker_mode(
            app_ctx, mode, ticker, year, horizons, threshold, as_of_date, output
        )
    elif mode == "tickers":
        _run_tickers_mode(
            app_ctx,
            year,
            horizons,
            threshold,
            days_back,
            min_buyers,
            top_n,
            output,
            training_lookback_days,
            as_of_date,
        )
    elif mode == "sales":
        _run_sales_mode(app_ctx, year, horizons, top_n, output)
    else:
        _run_analysis_mode(
            app_ctx,
            year,
            horizons,
            threshold,
            member,
            top_n,
            mode,
            output,
            sectors,
        )


def _print_house_fetch_summary(summary) -> None:
    print(
        f"  House {summary.archive_year}: metadata={summary.metadata_count}, "
        f"PTR={summary.ptr_count}, valid PDFs={summary.valid_pdf_count}, "
        f"downloaded={summary.downloaded_count}, skipped={summary.skipped_count}, "
        f"orphan PDFs={summary.orphan_pdf_count}, "
        f"removed docs={summary.removed_doc_count}, "
        f"quarantined PDFs={summary.quarantined_pdf_count}, "
        f"generation={summary.generation_id} ({summary.generation_status})"
    )


@app.command()
def fetch(
    ctx: typer.Context,
    year: int = typer.Option(_CURRENT_YEAR, help="House archive year to process"),
    data_dir: str = typer.Option("data", help="Data directory"),
    refresh_metadata: bool = typer.Option(
        False, "--refresh-metadata", help="Force refresh of metadata from House Clerk"
    ),
):
    """Download and reconcile House PDFs for one official archive."""
    app_ctx = get_context(ctx, data_dir, read_only=False)
    try:
        summary = app_ctx.transaction_source.fetch_and_cache_pdfs(
            year,
            refresh_metadata=refresh_metadata or year == date.today().year,
        )
    except Exception as exc:
        logger.error("House archive %d fetch failed: %s", year, exc)
        print(f"House fetch incomplete: {exc}", file=sys.stderr)
        raise typer.Exit(1) from None
    _print_house_fetch_summary(summary)
    raise typer.Exit(0)


@app.command()
def parse(
    ctx: typer.Context,
    year: int = typer.Option(_CURRENT_YEAR, help="House archive year to process"),
    data_dir: str = typer.Option("data", help="Data directory"),
    use_gemini_ocr: bool = typer.Option(
        False,
        "--gemini-ocr",
        help="Use Gemini LLM OCR for zero-row PDFs (slower, costs API quota)",
    ),
    force_full_reparse: bool = typer.Option(
        False,
        "--force-full-reparse",
        help="Ignore deterministic parse fingerprints and reparse every cached PDF",
    ),
):
    """Parse cached PDFs to database."""
    app_ctx = get_context(ctx, data_dir, read_only=False)
    parse_success = False
    try:
        if force_full_reparse:
            app_ctx.transaction_source.parse_cached_pdfs(year, force=True)
            parse_success = True
        else:
            result = run_parse_pipeline(app_ctx.transaction_source, year)
            parse_success = result.success
    except Exception:
        logger.exception("Parse pipeline failed")
    ocr_inserted = 0
    if use_gemini_ocr:
        from scripts.ocr_zero_rows import run_gemini_ocr_for_year

        ocr_inserted = run_gemini_ocr_for_year(
            year, data_dir=app_ctx.settings.data.data_dir
        )
    if not parse_success and use_gemini_ocr and ocr_inserted > 0:
        logger.warning(
            "Parse pipeline failed but Gemini OCR inserted %s rows", ocr_inserted
        )
    raise typer.Exit(0 if parse_success else 1)


@app.command()
def backtest(
    ctx: typer.Context,
    start: str = typer.Option(..., help="Backtest start date (YYYY-MM-DD)"),
    end: str = typer.Option(..., help="Backtest end date (YYYY-MM-DD)"),
    horizon: int = typer.Option(
        _BACKTEST_DEFAULTS["horizon"], help="Forward return horizon in days"
    ),
    lookback_days: int = typer.Option(
        _BACKTEST_DEFAULTS["lookback_days"],
        help="Candidate purchase lookback window in days",
    ),
    training_lookback_days: int = typer.Option(
        _BACKTEST_DEFAULTS["training_lookback_days"],
        help="Training data lookback window in days",
    ),
    min_buyers: int = typer.Option(
        _BACKTEST_DEFAULTS["min_buyers"], help="Minimum buyers for a candidate ticker"
    ),
    top_n: int = typer.Option(
        _BACKTEST_DEFAULTS["top_n"], help="Top N recommendations per backtest date"
    ),
    threshold: float = typer.Option(
        _BACKTEST_DEFAULTS["threshold"], help="Hit rate threshold percentage"
    ),
    frequency_days: int = typer.Option(
        _BACKTEST_DEFAULTS["frequency_days"], help="Days between rolling backtest dates"
    ),
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

    _validate_positive_options(
        horizon=horizon,
        lookback_days=lookback_days,
        training_lookback_days=training_lookback_days,
        min_buyers=min_buyers,
        top_n=top_n,
        frequency_days=frequency_days,
    )

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
    resolved_data_dir = Path(app_ctx.settings.data.data_dir)
    result = run_backtest_pipeline(
        params, app_ctx.transaction_source, app_ctx.price_source, resolved_data_dir
    )
    if result.success and hasattr(result, "data") and result.data:
        data = result.data
        snapshot = data.get("snapshot")
        if snapshot:
            print("\n=== Price Snapshot ===")
            print(f"  Snapshot ID:  {snapshot.snapshot_id}")
            print(f"  Created:      {snapshot.created_at}")
            print(f"  Git SHA:      {snapshot.git_sha[:12]}")
            print(f"  yfinance:     {snapshot.yfinance_version}")
            print(
                f"  Tickers:      {snapshot.resolved_tickers}/{snapshot.requested_tickers} resolved"
            )
            if snapshot.unresolved_tickers:
                print(f"  Unresolved:   {', '.join(snapshot.unresolved_tickers[:10])}")
            print(f"  Price rows:   {snapshot.price_rows}")
            print(f"  Date range:   {snapshot.first_date} to {snapshot.last_date}")

        combined = data.get("combined", pd.DataFrame())
        if not combined.empty:
            display_cols = [
                "as_of_date",
                "rank",
                "ticker",
                "num_buyers",
                "signal_score",
                "ou_entry_value",
                "bt_entry_price",
                "bt_exit_price",
                "bt_return_pct",
                "bt_spy_return_pct",
                "bt_alpha_pct",
            ]
            available = [c for c in display_cols if c in combined.columns]
            for as_of_date, group in combined.groupby("as_of_date"):
                print(f"\n=== Backtest as of {as_of_date} ===")
                print(group[available].to_string(index=False))

            print(f"\n{'=' * 60}")
            print("=== Backtest Summary (by rank) ===")
            print(f"{'=' * 60}")
            summary = data.get("summary", pd.DataFrame())
            if not summary.empty:
                print(summary.to_string(index=False))

            valid_returns = combined.dropna(subset=["bt_return_pct"])
            print(
                f"\nDates evaluated: {data.get('evaluable_dates', 0)}/{data.get('total_as_of_dates', 0)}"
            )
            print(
                f"Total recommendations: {len(combined)}, with measurable returns: {len(valid_returns)}"
            )
        else:
            print("\n=== No backtest results produced ===")
    raise typer.Exit(0 if result.success else 1)


@app.command()
def portfolio(
    ctx: typer.Context,
    start: str = typer.Option(..., help="Simulation start date (YYYY-MM-DD)"),
    end: str = typer.Option(..., help="Simulation end date (YYYY-MM-DD)"),
    horizon: int = typer.Option(
        _BACKTEST_DEFAULTS["horizon"], help="Forward return horizon in days"
    ),
    lookback_days: int = typer.Option(
        _BACKTEST_DEFAULTS["lookback_days"],
        help="Candidate purchase lookback window in days",
    ),
    training_lookback_days: int = typer.Option(
        _BACKTEST_DEFAULTS["training_lookback_days"],
        help="Training data lookback window in days",
    ),
    min_buyers: int = typer.Option(
        _BACKTEST_DEFAULTS["min_buyers"], help="Minimum buyers for a candidate ticker"
    ),
    top_n: int = typer.Option(
        _BACKTEST_DEFAULTS["top_n"], help="Top N recommendations per backtest date"
    ),
    threshold: float = typer.Option(
        _BACKTEST_DEFAULTS["threshold"], help="Hit rate threshold percentage"
    ),
    # Intentionally bi-weekly (not the backtest's 30d step): rebalance cadence
    # for the portfolio sim, independent of the sweep-calibrated backtest.
    frequency_days: int = typer.Option(14, help="Days between rolling backtest dates"),
    initial_capital: float = typer.Option(20000, help="Initial portfolio capital"),
    max_positions: int = typer.Option(5, help="Maximum concurrent positions"),
    hold_days: int = typer.Option(120, help="Hold period in days before forced exit"),
    sector_map: str | None = typer.Option(
        None,
        "--sector-map",
        help="Deterministic JSON mapping or CSV with ticker,sector columns",
    ),
    data_dir: str = typer.Option("data", help="Data directory"),
):
    """
    Run portfolio-level simulation with overlapping positions and constraints.

    Unlike the backtest command which evaluates each recommendation independently,
    this simulates a real portfolio with position sizing, sector limits, and
    cash management across overlapping holding periods.
    """
    start_date, end_date = _parse_sim_dates(start, end)

    _validate_positive_options(
        horizon=horizon,
        lookback_days=lookback_days,
        training_lookback_days=training_lookback_days,
        min_buyers=min_buyers,
        top_n=top_n,
        frequency_days=frequency_days,
        initial_capital=initial_capital,
        max_positions=max_positions,
        hold_days=hold_days,
    )

    try:
        sector_by_ticker = _load_sector_map(sector_map)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise typer.Exit(1)

    app_ctx = get_context(ctx, data_dir, read_only=True)

    from analyzer.portfolio_sim import PortfolioSimulator, PortfolioConfig
    from datetime import timedelta

    tx_start = start_date - timedelta(days=training_lookback_days + horizon + 30)
    all_transactions = app_ctx.transaction_source.db.get_transactions_by_date_range(
        tx_start, end_date
    )
    if all_transactions.empty:
        print("Error: no transactions found for portfolio simulation", file=sys.stderr)
        raise typer.Exit(1)

    prices, entry_prices, signals, recommendations = _load_portfolio_inputs(
        app_ctx,
        all_transactions,
        tx_start,
        end_date,
        horizon,
        lookback_days,
        training_lookback_days,
        min_buyers,
        top_n,
        threshold,
        frequency_days,
        start_date,
    )

    missing_sectors = sorted(set(recommendations["ticker"]) - set(sector_by_ticker))
    if missing_sectors:
        sample = ", ".join(missing_sectors[:10])
        print(
            f"Error: sector map is missing {len(missing_sectors)} recommended ticker(s): {sample}",
            file=sys.stderr,
        )
        raise typer.Exit(1)

    config = PortfolioConfig(
        initial_capital=initial_capital,
        max_positions=max_positions,
        hold_period_days=hold_days,
        rebalance_freq_days=frequency_days,
        sector_by_ticker=sector_by_ticker,
    )

    sim = PortfolioSimulator(config)
    results_df = sim.run(recommendations, prices, start_date, end_date)
    metrics = sim.compute_metrics(prices)
    if metrics.get("valuation_status") != "unavailable":
        _print_portfolio_results(
            results_df, config, start_date, end_date, hold_days, max_positions
        )
    if metrics:
        _print_portfolio_metrics(metrics)
    if sim.closed_positions:
        _print_closed_positions(sim.closed_positions)


def _load_sector_map(path_value: str | None) -> dict[str, str]:
    """Load deterministic sector data without any live network fallback."""
    if not path_value:
        raise ValueError("--sector-map is required for portfolio simulation")
    path = Path(path_value)
    if not path.is_file():
        raise ValueError(f"sector map file not found: {path}")

    if path.suffix.lower() == ".json":
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid sector map JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(
                "sector map JSON must be an object of ticker: sector pairs"
            )
        pairs = payload.items()
    elif path.suffix.lower() == ".csv":
        try:
            frame = pd.read_csv(path)
        except (OSError, pd.errors.ParserError) as exc:
            raise ValueError(f"invalid sector map CSV: {exc}") from exc
        required = {"ticker", "sector"}
        if not required.issubset(frame.columns):
            raise ValueError("sector map CSV must contain ticker and sector columns")
        pairs = frame[["ticker", "sector"]].itertuples(index=False, name=None)
    else:
        raise ValueError("sector map must be a .json or .csv file")

    sectors: dict[str, str] = {}
    for raw_ticker, raw_sector in pairs:
        ticker = str(raw_ticker).strip()
        sector = str(raw_sector).strip()
        if not ticker or not sector or sector.lower() == "nan":
            raise ValueError("sector map contains a blank ticker or sector")
        if ticker in sectors and sectors[ticker] != sector:
            raise ValueError(f"sector map contains conflicting sectors for {ticker}")
        sectors[ticker] = sector
    if not sectors:
        raise ValueError("sector map is empty")
    return sectors


def _parse_sim_dates(start: str, end: str) -> tuple[date, date]:
    """Parse YYYY-MM-DD CLI date inputs and validate end >= start."""
    try:
        start_date = date.fromisoformat(start)
        end_date = date.fromisoformat(end)
    except ValueError:
        print("Error: dates must be in YYYY-MM-DD format", file=sys.stderr)
        raise typer.Exit(1)

    if end_date < start_date:
        print("Error: --end must be on or after --start", file=sys.stderr)
        raise typer.Exit(1)

    return start_date, end_date


def _load_portfolio_inputs(
    app_ctx,
    all_transactions,
    tx_start,
    end_date,
    horizon,
    lookback_days,
    training_lookback_days,
    min_buyers,
    top_n,
    threshold,
    frequency_days,
    start_date,
):
    """Load prices + entry_prices + signals + walk-forward recommendations.

    Returns (prices_df, entry_prices_df, signals_df, recommendations_df).
    Errors with `typer.Exit(1)` if any input is missing.
    """
    from datetime import timedelta
    from analyzer import analysis

    price_end_sim = end_date + timedelta(days=horizon + 10)
    raw_tickers = all_transactions["ticker"].dropna().unique().tolist()
    all_tickers = sorted(
        {t for t in raw_tickers if isinstance(t, str) and t.strip()} | {"SPY"}
    )
    prices = app_ctx.transaction_source.db.get_prices(
        all_tickers, tx_start, price_end_sim
    )
    if prices.empty:
        print("Error: no price data available", file=sys.stderr)
        raise typer.Exit(1)

    entry_prices = app_ctx.transaction_source.db.get_entry_prices(
        all_tickers, tx_start, price_end_sim
    )
    if entry_prices.empty:
        print("Error: no entry prices computed", file=sys.stderr)
        raise typer.Exit(1)

    signals = analysis.calculate_signal_potential(entry_prices, prices, [horizon])

    as_of_dates = pd.date_range(start_date, end_date, freq=f"{frequency_days}D")
    all_recs = []
    for as_of in as_of_dates:
        recs = analysis.backtest_recommendations(
            signals,
            all_transactions,
            pd.Timestamp(as_of),
            horizon=horizon,
            lookback_days=lookback_days,
            min_buyers=min_buyers,
            top_n=top_n,
            threshold=threshold,
            prices_df=prices,
            training_lookback_days=training_lookback_days,
        )
        if not recs.empty:
            recs = recs.copy()
            recs["as_of_date"] = as_of
            all_recs.append(recs)

    if not all_recs:
        print("No recommendations produced for any backtest date", file=sys.stderr)
        raise typer.Exit(1)

    recommendations = pd.concat(all_recs, ignore_index=True)
    print(
        f"Collected {len(recommendations)} recommendations across {len(as_of_dates)} dates"
    )

    return prices, entry_prices, signals, recommendations


def _print_portfolio_results(
    results_df: pd.DataFrame,
    config,
    start_date,
    end_date,
    hold_days: int,
    max_positions: int,
) -> None:
    """Print the simulation result summary block."""
    print(f"\n{'=' * 60}")
    print("=== Portfolio Simulation Results ===")
    print(f"{'=' * 60}")
    if not results_df.empty:
        print(f"  Period:             {start_date} to {end_date}")
        print(f"  Initial capital:    ${config.initial_capital:,.2f}")
        print(f"  Final value:        ${results_df.iloc[-1]['total_value']:,.2f}")
        print(f"  Cash remaining:     ${results_df.iloc[-1]['cash']:,.2f}")
        print(f"  Max positions:      {max_positions}")
        print(f"  Hold period:        {hold_days} days")


def _print_portfolio_metrics(metrics: dict) -> None:
    """Print performance metrics (Sharpe, drawdown, win rate, etc.)."""
    print("\n=== Performance Metrics ===")
    if metrics.get("valuation_status") == "unavailable":
        print("  Valuation:          UNAVAILABLE")
        print(f"  Reason:             {metrics.get('valuation_reason', 'unknown')}")
        print(f"  Open positions:     {metrics.get('open_position_count', 0)}")
        return
    print(f"  Total return:       {metrics['total_return_pct']:.2f}%")
    print(f"  Return status:      {metrics.get('return_status', 'unknown')}")
    print(f"  Annualized return:  {metrics['annualized_return_pct']:.2f}%")
    if metrics.get("daily_risk_status") != "available":
        print("  Daily risk metrics: UNAVAILABLE (nonconsecutive valuations)")
    else:
        print(f"  Sharpe ratio:       {metrics['sharpe_ratio']:.3f}")
        print(f"  Max drawdown:       {metrics['max_drawdown_pct']:.2f}%")
        print(f"  Volatility:         {metrics['volatility_pct']:.2f}%")
    if metrics["total_closed_trades"]:
        print(f"  Win rate:           {metrics['win_rate_pct']:.1f}%")
        print(f"  Avg holding days:   {metrics['avg_holding_days']:.1f}")
        print(f"  Turnover rate:      {metrics['turnover_rate']:.3f}")
    else:
        print("  Win rate:           N/A (no closed trades)")
        print("  Avg holding days:   N/A (no closed trades)")
        print("  Turnover rate:      N/A (no closed trades)")
    print(f"  Max concurrent:     {metrics['max_concurrent_positions']}")
    print(f"  Total closed:       {metrics['total_closed_trades']}")
    if metrics.get("spy_return_pct") is not None:
        print(f"  SPY buy-and-hold:   {metrics['spy_return_pct']:.2f}%")
    if metrics.get("sector_concentration"):
        print("\n=== Sector Concentration ===")
        for sector, pct in sorted(
            metrics["sector_concentration"].items(), key=lambda x: -x[1]
        ):
            print(f"  {sector}: {pct:.1f}%")


def _print_closed_positions(closed_positions: list[dict]) -> None:
    """Print per-position close details (ticker, return, holding days)."""
    print(f"\n=== Closed Positions ({len(closed_positions)}) ===")
    closed_df = pd.DataFrame(closed_positions)
    display_cols = [
        "ticker",
        "entry_date",
        "exit_date",
        "return_pct",
        "holding_days",
        "sector",
    ]
    available = [c for c in display_cols if c in closed_df.columns]
    print(closed_df[available].to_string(index=False))


@app.command()
def snapshot(
    ctx: typer.Context,
    data_dir: str = typer.Option("data", help="Data directory"),
    output: str = typer.Option(
        "data/price_snapshot.json", help="Output path for snapshot JSON"
    ),
):
    """Create a frozen price snapshot manifest for reproducible backtests."""
    app_ctx = get_context(ctx, data_dir, read_only=True)

    db = app_ctx.transaction_source.db
    tickers_result = db.conn.execute("SELECT DISTINCT ticker FROM prices").fetchall()
    all_tickers = sorted(
        {row[0] for row in tickers_result if row[0] and isinstance(row[0], str)}
    )

    if not all_tickers:
        print("No price data found in database")
        raise typer.Exit(1)

    date_range = db.conn.execute("SELECT MIN(date), MAX(date) FROM prices").fetchone()
    start_date = date_range[0]
    end_date = date_range[1]

    snap = create_snapshot(db, all_tickers, start_date, end_date)
    save_snapshot(snap, output)

    print(f"Snapshot created: {snap.snapshot_id}")
    print(f"  Created at:     {snap.created_at}")
    print(f"  Git SHA:        {snap.git_sha[:12]}")
    print(f"  yfinance:       {snap.yfinance_version}")
    print(f"  Python:         {snap.python_version}")
    print(
        f"  Tickers:        {snap.resolved_tickers}/{snap.requested_tickers} resolved"
    )
    if snap.unresolved_tickers:
        print(f"  Unresolved:     {', '.join(snap.unresolved_tickers[:10])}")
    print(f"  Price rows:     {snap.price_rows}")
    print(f"  Date range:     {snap.first_date} to {snap.last_date}")
    print(f"  Saved to:       {output}")
    raise typer.Exit(0)


@app.command()
def refresh(
    ctx: typer.Context,
    year: int = typer.Option(_CURRENT_YEAR, help="House archive year to refresh"),
    data_dir: str = typer.Option("data", help="Data directory"),
    use_gemini_ocr: bool = typer.Option(
        False, "--gemini-ocr", help="Use Gemini LLM OCR for zero-row PDFs"
    ),
    skip_capitol: bool = typer.Option(
        False,
        "--skip-capitol",
        help="Explicitly skip third-party reconciliation notice",
    ),
    refresh_metadata: bool = typer.Option(
        False, "--refresh-metadata", help="Force refresh House Clerk metadata"
    ),
    all_years: bool = typer.Option(
        False,
        "--all-years",
        "--full-history",
        help=(
            "Refresh the currently downloadable official House PTR scope "
            "(2015 through today); legacy archives are inventoried as excluded"
        ),
    ),
    force_full_reparse: bool = typer.Option(
        False,
        "--force-full-reparse",
        help="Reparse every cached PDF after all requested archives reconcile",
    ),
):
    """
    Official House refresh: fetch PDFs, parse, and optionally run Gemini OCR.

    Capitol Trades is third-party reconciliation data and is explicitly excluded
    from canonical refresh. Use `fetch-capitol` with an output manifest separately.
    """

    app_ctx = get_context(ctx, data_dir, read_only=False)
    archive_years = (
        list(range(_HOUSE_PTR_FIRST_ARCHIVE_YEAR, date.today().year + 1))
        if all_years
        else [year]
    )

    if all_years:
        excluded_years = list(
            range(_HOUSE_LEGACY_FIRST_ARCHIVE_YEAR, _HOUSE_PTR_FIRST_ARCHIVE_YEAR)
        )
        print(
            "Official downloadable scope starts in 2015; "
            f"excluded legacy archive count={len(excluded_years)} "
            f"({excluded_years[0]}-{excluded_years[-1]}), PDF inventory unavailable"
        )

    canonical_count_before = app_ctx.transaction_source.db.conn.execute(
        "SELECT COUNT(*) FROM canonical_transactions"
    ).fetchone()[0]
    raw_count_before = app_ctx.transaction_source.db.conn.execute(
        "SELECT COUNT(*) FROM transactions"
    ).fetchone()[0]
    failed_steps: list[str] = []

    # Fetch every requested archive before parsing or invoking a backup source.
    # A partial PDF set is not a usable refresh generation.
    label = (
        f"{archive_years[0]}-{archive_years[-1]}"
        if len(archive_years) > 1
        else str(archive_years[0])
    )
    print(f"[1/4] Fetching and reconciling House PDF archives {label}...")
    summaries = []
    try:
        for archive_year in archive_years:
            summary = app_ctx.transaction_source.fetch_and_cache_pdfs(
                archive_year,
                refresh_metadata=(
                    all_years
                    or refresh_metadata
                    or archive_year == date.today().year
                ),
            )
            summaries.append(summary)
            _print_house_fetch_summary(summary)
    except Exception as exc:
        logger.warning("House PDF fetch failed: %s", exc)
        print(f"House fetch incomplete: {exc}")
        print("FAILED steps: fetch")
        raise typer.Exit(1) from None

    print(
        "  House totals: "
        f"archives={len(summaries)}, "
        f"metadata={sum(item.metadata_count for item in summaries)}, "
        f"PTR={sum(item.ptr_count for item in summaries)}, "
        f"valid PDFs={sum(item.valid_pdf_count for item in summaries)}, "
        f"orphan PDFs={sum(item.orphan_pdf_count for item in summaries)}, "
        f"removed docs={sum(item.removed_doc_count for item in summaries)}, "
        f"quarantined PDFs={sum(item.quarantined_pdf_count for item in summaries)}, "
        "generation status=incomplete pending artifact-bound parse/OCR"
    )

    print(f"[2/4] Parsing cached House PDF archives {label}...")
    for archive_year in archive_years:
        try:
            if force_full_reparse:
                app_ctx.transaction_source.parse_cached_pdfs(
                    archive_year, force=True
                )
            else:
                parse_result = run_parse_pipeline(
                    app_ctx.transaction_source, archive_year
                )
                if not parse_result.success:
                    failed_steps.append(f"parse:{archive_year}")
        except Exception as exc:
            failed_steps.append(f"parse:{archive_year}")
            logger.warning("PDF parse failed for %d: %s", archive_year, exc)

    # Step 3: Third-party reconciliation is never part of official refresh.
    if skip_capitol:
        print("[3/4] Skipping Capitol Trades reconciliation (--skip-capitol)")

    else:
        print(
            "[3/4] Excluding Capitol Trades from official refresh "
            "(use fetch-capitol --output ... --generation ... for reconciliation)"
        )

    if use_gemini_ocr:
        print("[4/4] Running Gemini OCR on zero-row PDFs...")
        from scripts.ocr_zero_rows import run_gemini_ocr_for_year

        for archive_year in archive_years:
            try:
                ocr_inserted = run_gemini_ocr_for_year(
                    archive_year,
                    data_dir=app_ctx.settings.data.data_dir,
                )
                print(
                    f"  Gemini OCR {archive_year}: "
                    f"{ocr_inserted} transactions inserted"
                )
            except Exception as exc:
                failed_steps.append(f"gemini_ocr:{archive_year}")
                logger.warning("Gemini OCR failed for %d: %s", archive_year, exc)
    else:
        print("[4/4] Skipping Gemini OCR (use --gemini-ocr to enable)")

    for archive_year in archive_years:
        generation_id = app_ctx.transaction_source.db.get_latest_house_generation(
            archive_year
        )
        if generation_id is None:
            failed_steps.append(f"missing_house_generation:{archive_year}")
            continue
        unresolved = app_ctx.transaction_source.db.get_unresolved_house_doc_ids(
            archive_year, generation_id
        )
        if not unresolved:
            app_ctx.transaction_source.db.mark_house_generation_parse_complete(
                archive_year, generation_id
            )
            print(
                f"  House {archive_year} generation={generation_id} "
                "status=complete (activated)"
            )
            continue
        failed_steps.append(f"unresolved_house:{archive_year}")
        preview = ", ".join(unresolved[:10])
        print(
            f"  House {archive_year} generation incomplete: "
            f"{len(unresolved)} unresolved PDFs ({preview})"
        )

    canonical_count_after = app_ctx.transaction_source.db.conn.execute(
        "SELECT COUNT(*) FROM canonical_transactions"
    ).fetchone()[0]
    raw_count_after = app_ctx.transaction_source.db.conn.execute(
        "SELECT COUNT(*) FROM transactions"
    ).fetchone()[0]
    summary_year = archive_years[-1]
    max_date, max_disclosure_date, implausible_date_count = (
        app_ctx.transaction_source.db.conn.execute(
            """
            SELECT
                MAX(transaction_date) FILTER (
                    WHERE transaction_date IS NULL OR transaction_date <= disclosure_date
                ),
                MAX(disclosure_date),
                COUNT(*) FILTER (WHERE transaction_date > disclosure_date)
            FROM canonical_transactions
            WHERE EXTRACT(YEAR FROM disclosure_date) = ?
            """,
            [summary_year],
        ).fetchone()
    )

    canonical_added = canonical_count_after - canonical_count_before
    raw_added = raw_count_after - raw_count_before
    outcome = "Incomplete." if failed_steps else "Done."
    print(
        f"\n{outcome} canonical {canonical_count_before} -> "
        f"{canonical_count_after} transactions "
        f"({'+' if canonical_added >= 0 else ''}{canonical_added}); "
        f"raw {raw_count_before} -> {raw_count_after} "
        f"({'+' if raw_added >= 0 else ''}{raw_added})"
    )
    print(f"Latest transaction date: {max_date} (eligible: not after disclosure)")
    print(f"Latest disclosure date: {max_disclosure_date}")
    if implausible_date_count:
        print(
            f"Excluded from analyses: {implausible_date_count} transaction(s) "
            "dated after disclosure"
        )
    if failed_steps:
        print(f"FAILED steps: {', '.join(failed_steps)}")
        raise typer.Exit(1)
    raise typer.Exit(0)


@app.command()
def fetch_capitol(
    ctx: typer.Context,
    politician: str | None = typer.Option(
        None, help="Fetch reconciliation records for one politician"
    ),
    all: bool = typer.Option(False, "--all", help="Fetch all reconciliation records"),
    chamber: str | None = typer.Option(None, help="Filter by chamber (house/senate)"),
    start: str | None = typer.Option(None, help="Start date filter (YYYY-MM-DD)"),
    end: str | None = typer.Option(None, help="End date filter (YYYY-MM-DD)"),
    output: Path = typer.Option(..., "--output", help="New reconciliation manifest"),
    generation: str = typer.Option(
        ..., "--generation", help="Non-empty ingestion run generation"
    ),
    data_dir: str = typer.Option("data", help="Data directory (never written)"),
):
    """Fetch a Capitol Trades reconciliation artifact; never save canonical rows."""
    from analyzer.capitol_trades import CapitolTradesError, CapitolTradesSource

    if bool(politician) == all:
        print(
            "Error: specify exactly one of --politician NAME or --all", file=sys.stderr
        )
        raise typer.Exit(1)

    try:
        start_date = date.fromisoformat(start) if start else None
        end_date = date.fromisoformat(end) if end else None
    except ValueError:
        print("Error: dates must be in YYYY-MM-DD format", file=sys.stderr)
        raise typer.Exit(1)
    if start_date is not None and end_date is not None and end_date < start_date:
        print("Error: --end must be on or after --start", file=sys.stderr)
        raise typer.Exit(1)

    try:
        capitol = CapitolTradesSource(
            data_dir=data_dir, read_only=True, generation=generation
        )
        try:
            if politician:
                df = capitol.fetch_trades(politician, start_date, end_date)
            else:
                df = capitol.fetch_all_trades(start_date, end_date, chamber)
            capitol.write_reconciliation_artifact(output)
        finally:
            capitol.close()
    except CapitolTradesError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise typer.Exit(1)

    print(f"Wrote {len(df)} reconciliation records to {output}")
    print("No canonical transactions were saved.")
    raise typer.Exit(0)


@app.command()
def fetch_senate_efd(
    ctx: typer.Context,
    start: str | None = typer.Option(
        None, help="Start date (YYYY-MM-DD). Defaults to one year before end."
    ),
    end: str | None = typer.Option(
        None, help="End date (YYYY-MM-DD). Defaults to today."
    ),
    lookback: int | None = typer.Option(
        None, help="If set, look back N days from end (overrides --start)"
    ),
    data_dir: str = typer.Option(
        "data/senate", help="Data directory (default: isolated senate DB)"
    ),
):
    """Fetch Senate PTR trades from efdsearch.senate.gov (official source).

    Loads into an isolated data directory so chamber separation is exact.
    Then run: ptr-alpha analyze --year <YYYY> --data-dir data/senate
    """
    from datetime import timedelta
    from analyzer.senate_efd import SenateEFDSource

    try:
        end_date = date.fromisoformat(end) if end else date.today()
        if lookback is not None:
            if lookback <= 0:
                raise ValueError("lookback must be positive")
            start_date = end_date - timedelta(days=lookback)
        else:
            start_date = (
                date.fromisoformat(start)
                if start
                else end_date.replace(year=end_date.year - 1)
            )
    except ValueError:
        print("Error: dates must be YYYY-MM-DD and lookback > 0", file=sys.stderr)
        raise typer.Exit(1)

    if start_date > end_date:
        print("Error: --start must be on or before --end", file=sys.stderr)
        raise typer.Exit(1)

    src = SenateEFDSource(data_dir=data_dir, read_only=False)
    try:
        count = src.fetch_and_save_all(start_date, end_date)
        print(
            f"Saved {count} new Senate eFD trades ({start_date} to {end_date}) into {data_dir}/"
        )
    finally:
        src.close()

    raise typer.Exit(0)


@app.command()
def validate(
    ctx: typer.Context,
    train_start: str = typer.Option(
        "2022-01-01", help="Training window start (YYYY-MM-DD)"
    ),
    train_end: str = typer.Option(
        "2023-12-31", help="Training window end (YYYY-MM-DD)"
    ),
    test_start: str = typer.Option("2024-01-01", help="Test window start (YYYY-MM-DD)"),
    test_end: str = typer.Option("2025-06-30", help="Test window end (YYYY-MM-DD)"),
    full_grid: bool = typer.Option(
        False, "--full-grid", help="Use full 1296-combo grid (slow)"
    ),
    data_dir: str = typer.Option("data", help="Data directory"),
    null_samples: int = typer.Option(
        999,
        "--null-samples",
        help="Centered moving-block bootstrap samples (release minimum: 999)",
    ),
    member_null_samples: int = typer.Option(
        999,
        "--member-null-samples",
        help="Full-family member-identity permutations (release minimum: 999)",
    ),
):
    """
    Purged retrospective validation with dependence-safe corrections.

    Sweeps configurations on the purged training phase, then requires both
    Bonferroni and centered moving-block max-stat bootstrap survival plus a
    full-family member-identity negative control. Fewer than 999 null samples
    fail closed. The 2024-2025 test phase is previously used retrospective data,
    not fresh out-of-sample evidence. The post-2025 final phase stays locked.

    Results are written to <data-dir>/validation_results.json and any frozen
    evaluation is atomically consumed in the evaluation ledger.
    """
    from analyzer.validation import run_validation

    try:
        ts = date.fromisoformat(train_start)
        te = date.fromisoformat(train_end)
        vs = date.fromisoformat(test_start)
        ve = date.fromisoformat(test_end)
    except ValueError:
        print("Error: dates must be in YYYY-MM-DD format", file=sys.stderr)
        raise typer.Exit(1)

    if te < ts:
        print("Error: --train-end must be on or after --train-start", file=sys.stderr)
        raise typer.Exit(1)
    if ve < vs:
        print("Error: --test-end must be on or after --test-start", file=sys.stderr)
        raise typer.Exit(1)
    if vs <= te:
        print(
            "Error: --test-start must be after --train-end (no overlap)",
            file=sys.stderr,
        )
        raise typer.Exit(1)

    if full_grid:
        grid = {
            "horizon": [60, 90, 120],
            "frequency_days": [30, 90],
            "training_lookback_days": [180, 365],
            "min_buyers": [2, 3, 5],
            "top_n": [3, 5],
            "decay_lambda": [0.001, 0.005, 0.02],
            "bayes_prior_strength": [5, 20, 50],
            "scoring_mode": ["shrunk_alpha", "consistency"],
        }
    else:
        # Default ~36-combo grid: 3*3*2*2 combinations
        grid = {
            "horizon": [60, 90, 120],
            "frequency_days": [30],
            "training_lookback_days": [365],
            "min_buyers": [2, 3, 5],
            "top_n": [3, 5],
            "decay_lambda": [0.005],
            "bayes_prior_strength": [20],
            "scoring_mode": ["shrunk_alpha", "consistency"],
        }

    if null_samples < 1 or member_null_samples < 1:
        print("Error: null sample counts must be positive", file=sys.stderr)
        raise typer.Exit(1)

    n_trials = 1
    for v in grid.values():
        n_trials *= len(v)
    print(
        f"Running validation with {n_trials} configs (trials for snooping correction)"
    )

    settings = Settings()
    if data_dir and data_dir != "data":
        settings.data.data_dir = data_dir
    resolved_data_dir = Path(settings.data.data_dir)
    db_path = resolved_data_dir / "congress.duckdb"
    out_path = resolved_data_dir / "validation_results.json"
    try:
        run_validation(
            db_path=db_path,
            train_start=ts,
            train_end=te,
            test_start=vs,
            test_end=ve,
            grid=grid,
            out_path=out_path,
            n_permutations=null_samples,
            member_permutations=member_null_samples,
        )
    except Exception:
        logger.exception("Validation failed")
        raise typer.Exit(1)
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
