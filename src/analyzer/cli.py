#!/usr/bin/env python3

import sys
import logging
from datetime import date
from pathlib import Path
from dataclasses import dataclass
import pandas as pd
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
from analyzer.database import Database
from analyzer.datasources import HouseTransactionSource, YFinancePriceSource

app = typer.Typer(help="Congressional PTR disclosure analyzer", no_args_is_help=True)
logger = logging.getLogger(__name__)
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
        shared_db = Database(Path(settings.data.data_dir) / "congress.duckdb", read_only=read_only)
        ctx.obj = AppContext(
            settings=settings,
            transaction_source=HouseTransactionSource(settings, read_only=read_only, db=shared_db),
            price_source=YFinancePriceSource(settings, read_only=read_only, db=shared_db),
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
    valid_modes = {"ranks", "signals", "member", "sales", "tickers"}
    if mode not in valid_modes:
        print(f"Error: --mode must be one of {sorted(valid_modes)}", file=sys.stderr)
        raise typer.Exit(1)
    if mode == "member" and member is None and ticker is None:
        print("Error: --mode member requires --member NAME", file=sys.stderr)
        raise typer.Exit(1)
    if mode == "sales" and member is not None:
        print("WARNING: --member flag is ignored for --mode sales (sales rankings are aggregate).", file=sys.stderr)
    app_ctx = get_context(ctx, data_dir, read_only=False)

    # Freshness check — warn if data looks stale
    try:
        _max_date = app_ctx.transaction_source.db.conn.execute(
            "SELECT MAX(disclosure_date) FROM transactions"
        ).fetchone()[0]
        if _max_date:
            _age = (date.today() - _max_date).days
            if _age > 30:
                print(f"WARNING: Data is {_age} days old (latest: {_max_date}). Run 'ptr-alpha refresh' first.", file=sys.stderr)
    except Exception:
        logger.debug("Freshness check failed", exc_info=True)

    if ticker:
        if mode != "ranks":
            print(f"WARNING: --mode {mode} is ignored when --ticker is provided; running ticker analysis.", file=sys.stderr)
        if output == "csv":
            print("WARNING: CSV output is not supported for --ticker analysis; using console output.", file=sys.stderr)
        params = TickerAnalysisParams(ticker=ticker, year=year, horizon=horizons[0], threshold=threshold)
        success = run_ticker_analysis(
            params,
            app_ctx.transaction_source,
            app_ctx.price_source,
        )
        raise typer.Exit(0 if success else 1)

    if mode == "tickers":
        if output == "csv":
            print("WARNING: CSV output is not supported for --mode tickers; using console output.", file=sys.stderr)
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
    params = AnalysisParams(
        year=year,
        horizons=horizons,
        threshold=threshold,
        member_filter=member,
        top_n=top_n,
        show_signals=show_signals,
    )
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
    use_gemini_ocr: bool = typer.Option(
        False, "--gemini-ocr", help="Use Gemini LLM OCR for zero-row PDFs (slower, costs API quota)"
    ),
):
    """Parse cached PDFs to database"""
    app_ctx = get_context(ctx, data_dir, read_only=False)
    try:
        success = run_parse_pipeline(app_ctx.transaction_source, year)
    except Exception:
        logger.exception("Parse pipeline failed")
        success = False
    ocr_inserted = 0
    if use_gemini_ocr:
        from scripts.ocr_zero_rows import run_gemini_ocr_for_year
        ocr_inserted = run_gemini_ocr_for_year(year, data_dir=app_ctx.settings.data.data_dir)
    if not success and use_gemini_ocr and ocr_inserted > 0:
        logger.warning("Parse pipeline failed but Gemini OCR inserted %s rows", ocr_inserted)
    raise typer.Exit(0 if success else 1)


@app.command()
def backtest(
    ctx: typer.Context,
    start: str = typer.Option(..., help="Backtest start date (YYYY-MM-DD)"),
    end: str = typer.Option(..., help="Backtest end date (YYYY-MM-DD)"),
    horizon: int = typer.Option(_BACKTEST_DEFAULTS["horizon"], help="Forward return horizon in days"),
    lookback_days: int = typer.Option(_BACKTEST_DEFAULTS["lookback_days"], help="Candidate purchase lookback window in days"),
    training_lookback_days: int = typer.Option(_BACKTEST_DEFAULTS["training_lookback_days"], help="Training data lookback window in days"),
    min_buyers: int = typer.Option(_BACKTEST_DEFAULTS["min_buyers"], help="Minimum buyers for a candidate ticker"),
    top_n: int = typer.Option(_BACKTEST_DEFAULTS["top_n"], help="Top N recommendations per backtest date"),
    threshold: float = typer.Option(_BACKTEST_DEFAULTS["threshold"], help="Hit rate threshold percentage"),
    frequency_days: int = typer.Option(_BACKTEST_DEFAULTS["frequency_days"], help="Days between rolling backtest dates"),
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
    resolved_data_dir = Path(app_ctx.settings.data.data_dir)
    success = run_backtest_pipeline(params, app_ctx.transaction_source, app_ctx.price_source, resolved_data_dir)
    raise typer.Exit(0 if success else 1)


@app.command()
def portfolio(
    ctx: typer.Context,
    start: str = typer.Option(..., help="Simulation start date (YYYY-MM-DD)"),
    end: str = typer.Option(..., help="Simulation end date (YYYY-MM-DD)"),
    horizon: int = typer.Option(_BACKTEST_DEFAULTS["horizon"], help="Forward return horizon in days"),
    lookback_days: int = typer.Option(_BACKTEST_DEFAULTS["lookback_days"], help="Candidate purchase lookback window in days"),
    training_lookback_days: int = typer.Option(_BACKTEST_DEFAULTS["training_lookback_days"], help="Training data lookback window in days"),
    min_buyers: int = typer.Option(_BACKTEST_DEFAULTS["min_buyers"], help="Minimum buyers for a candidate ticker"),
    top_n: int = typer.Option(_BACKTEST_DEFAULTS["top_n"], help="Top N recommendations per backtest date"),
    threshold: float = typer.Option(_BACKTEST_DEFAULTS["threshold"], help="Hit rate threshold percentage"),
    # Intentionally bi-weekly (not the backtest's 30d step): rebalance cadence
    # for the portfolio sim, independent of the sweep-calibrated backtest.
    frequency_days: int = typer.Option(14, help="Days between rolling backtest dates"),
    initial_capital: float = typer.Option(20000, help="Initial portfolio capital"),
    max_positions: int = typer.Option(5, help="Maximum concurrent positions"),
    hold_days: int = typer.Option(120, help="Hold period in days before forced exit"),
    data_dir: str = typer.Option("data", help="Data directory"),
):
    """
    Run portfolio-level simulation with overlapping positions and constraints.

    Unlike the backtest command which evaluates each recommendation independently,
    this simulates a real portfolio with position sizing, sector limits, and
    cash management across overlapping holding periods.
    """
    start_date, end_date = _parse_sim_dates(start, end)

    app_ctx = get_context(ctx, data_dir, read_only=True)

    from analyzer.portfolio_sim import PortfolioSimulator, PortfolioConfig
    from datetime import timedelta

    tx_start = start_date - timedelta(days=training_lookback_days + horizon + 30)
    all_transactions = app_ctx.transaction_source.db.get_transactions_by_date_range(tx_start, end_date)
    if all_transactions.empty:
        print("Error: no transactions found for portfolio simulation", file=sys.stderr)
        raise typer.Exit(1)

    prices, entry_prices, signals, recommendations = _load_portfolio_inputs(
        app_ctx, all_transactions, tx_start, end_date, horizon,
        lookback_days, training_lookback_days, min_buyers, top_n,
        threshold, frequency_days, start_date,
    )

    config = PortfolioConfig(
        initial_capital=initial_capital,
        max_positions=max_positions,
        hold_period_days=hold_days,
        rebalance_freq_days=frequency_days,
    )

    sim = PortfolioSimulator(config)
    results_df = sim.run(recommendations, prices, start_date, end_date)
    _print_portfolio_results(results_df, config, start_date, end_date, hold_days, max_positions)

    metrics = sim.compute_metrics(prices)
    if metrics:
        _print_portfolio_metrics(metrics)
    if sim.closed_positions:
        _print_closed_positions(sim.closed_positions)


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
    app_ctx, all_transactions, tx_start, end_date, horizon,
    lookback_days, training_lookback_days, min_buyers, top_n,
    threshold, frequency_days, start_date,
):
    """Load prices + entry_prices + signals + walk-forward recommendations.

    Returns (prices_df, entry_prices_df, signals_df, recommendations_df).
    Errors with `typer.Exit(1)` if any input is missing.
    """
    from datetime import timedelta
    from analyzer import analysis

    price_end_sim = end_date + timedelta(days=horizon + 10)
    raw_tickers = all_transactions["ticker"].dropna().unique().tolist()
    all_tickers = sorted({t for t in raw_tickers if isinstance(t, str) and t.strip()} | {"SPY"})
    prices = app_ctx.transaction_source.db.get_prices(all_tickers, tx_start, price_end_sim)
    if prices.empty:
        print("Error: no price data available", file=sys.stderr)
        raise typer.Exit(1)

    entry_prices = app_ctx.transaction_source.db.get_entry_prices(all_tickers, tx_start, price_end_sim)
    if entry_prices.empty:
        print("Error: no entry prices computed", file=sys.stderr)
        raise typer.Exit(1)

    signals = analysis.calculate_signal_potential(entry_prices, prices, [horizon])

    as_of_dates = pd.date_range(start_date, end_date, freq=f"{frequency_days}D")
    all_recs = []
    for as_of in as_of_dates:
        recs = analysis.backtest_recommendations(
            signals, all_transactions, pd.Timestamp(as_of),
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
    print(f"Collected {len(recommendations)} recommendations across {len(as_of_dates)} dates")

    return prices, entry_prices, signals, recommendations


def _print_portfolio_results(
    results_df: pd.DataFrame, config, start_date, end_date,
    hold_days: int, max_positions: int,
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
    print(f"  Total return:       {metrics['total_return_pct']:.2f}%")
    print(f"  Annualized return:  {metrics['annualized_return_pct']:.2f}%")
    print(f"  Sharpe ratio:       {metrics['sharpe_ratio']:.3f}")
    print(f"  Max drawdown:       {metrics['max_drawdown_pct']:.2f}%")
    print(f"  Win rate:           {metrics['win_rate_pct']:.1f}%")
    print(f"  Avg holding days:   {metrics['avg_holding_days']:.1f}")
    print(f"  Turnover rate:      {metrics['turnover_rate']:.3f}")
    print(f"  Max concurrent:     {metrics['max_concurrent_positions']}")
    print(f"  Total closed:       {metrics['total_closed_trades']}")
    if metrics.get("spy_return_pct") is not None:
        print(f"  SPY buy-and-hold:   {metrics['spy_return_pct']:.2f}%")
    if metrics.get("sector_concentration"):
        print("\n=== Sector Concentration ===")
        for sector, pct in sorted(metrics["sector_concentration"].items(), key=lambda x: -x[1]):
            print(f"  {sector}: {pct:.1f}%")


def _print_closed_positions(closed_positions: list[dict]) -> None:
    """Print per-position close details (ticker, return, holding days)."""
    print(f"\n=== Closed Positions ({len(closed_positions)}) ===")
    closed_df = pd.DataFrame(closed_positions)
    display_cols = ["ticker", "entry_date", "exit_date", "return_pct", "holding_days", "sector"]
    available = [c for c in display_cols if c in closed_df.columns]
    print(closed_df[available].to_string(index=False))


@app.command()
def snapshot(
    ctx: typer.Context,
    data_dir: str = typer.Option("data", help="Data directory"),
    output: str = typer.Option("data/price_snapshot.json", help="Output path for snapshot JSON"),
):
    """Create a frozen price snapshot manifest for reproducible backtests."""
    app_ctx = get_context(ctx, data_dir, read_only=True)

    db = app_ctx.transaction_source.db
    tickers_result = db.conn.execute("SELECT DISTINCT ticker FROM prices").fetchall()
    all_tickers = sorted({row[0] for row in tickers_result if row[0] and isinstance(row[0], str)})

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


@app.command()
def refresh(
    ctx: typer.Context,
    year: int = typer.Option(2025, help="Year to refresh"),
    data_dir: str = typer.Option("data", help="Data directory"),
    use_gemini_ocr: bool = typer.Option(
        False, "--gemini-ocr", help="Use Gemini LLM OCR for zero-row PDFs"
    ),
    skip_capitol: bool = typer.Option(
        False, "--skip-capitol", help="Skip Capitol Trades API fetch"
    ),
    refresh_metadata: bool = typer.Option(
        False, "--refresh-metadata", help="Force refresh House Clerk metadata"
    ),
):
    """
    Full pipeline refresh: fetch House PDFs + parse + Capitol Trades API.

    This is the single command to keep the database up to date. It runs:
      1. Fetch House Clerk metadata and download new PTR PDFs
      2. Parse all cached PDFs into transactions
      3. Fetch Capitol Trades API (backup source for any missed filings)
      4. Optionally run Gemini OCR on zero-row PDFs
    """
    app_ctx = get_context(ctx, data_dir, read_only=False)

    # Count before
    count_before = app_ctx.transaction_source.db.conn.execute(
        "SELECT COUNT(*) FROM transactions"
    ).fetchone()[0]
    failed_steps: list[str] = []

    # Step 1: Fetch House PDFs
    print(f"[1/4] Fetching House PDFs for {year}...")
    if refresh_metadata:
        app_ctx.transaction_source.fetch_metadata(year, refresh=True)
    try:
        if not run_fetch_pipeline(app_ctx.transaction_source, year):
            failed_steps.append("fetch")
    except Exception as e:
        failed_steps.append("fetch")
        logger.warning(f"House PDF fetch failed: {e}")

    # Step 2: Parse cached PDFs
    print(f"[2/4] Parsing cached PDFs for {year}...")
    try:
        if not run_parse_pipeline(app_ctx.transaction_source, year):
            failed_steps.append("parse")
    except Exception as e:
        failed_steps.append("parse")
        logger.warning(f"PDF parse failed: {e}")

    # Step 3: Capitol Trades API
    if not skip_capitol:
        print("[3/4] Fetching Capitol Trades API...")
        from analyzer.capitol_trades import CapitolTradesSource
        capitol = CapitolTradesSource(
            data_dir=app_ctx.settings.data.data_dir,
            read_only=False,
            db=app_ctx.transaction_source.db,
        )
        try:
            capitol_count = capitol.fetch_and_save_all()
            print(f"  Capitol Trades: {capitol_count} transactions upserted")
        except Exception as e:
            failed_steps.append("capitol")
            logger.warning(f"Capitol Trades fetch failed: {e}")
        finally:
            capitol.close()
    else:
        print("[3/4] Skipping Capitol Trades API (--skip-capitol)")

    # Step 4: Gemini OCR (optional)
    if use_gemini_ocr:
        print("[4/4] Running Gemini OCR on zero-row PDFs...")
        try:
            from scripts.ocr_zero_rows import run_gemini_ocr_for_year
            ocr_inserted = run_gemini_ocr_for_year(year, data_dir=app_ctx.settings.data.data_dir)
            print(f"  Gemini OCR: {ocr_inserted} transactions inserted")
        except Exception as e:
            failed_steps.append("gemini_ocr")
            logger.warning(f"Gemini OCR failed: {e}")
    else:
        print("[4/4] Skipping Gemini OCR (use --gemini-ocr to enable)")

    # Count after
    count_after = app_ctx.transaction_source.db.conn.execute(
        "SELECT COUNT(*) FROM transactions"
    ).fetchone()[0]
    max_date = app_ctx.transaction_source.db.conn.execute(
        "SELECT MAX(transaction_date) FROM transactions"
    ).fetchone()[0]

    added = count_after - count_before
    print(f"\nDone. {count_before} -> {count_after} transactions ({'+' if added >= 0 else ''}{added} new)")
    print(f"Latest transaction date: {max_date}")
    if failed_steps:
        print(f"FAILED steps: {', '.join(failed_steps)}")
        raise typer.Exit(1)
    raise typer.Exit(0)


@app.command()
def fetch_capitol(
    ctx: typer.Context,
    politician: str | None = typer.Option(None, help="Fetch trades for a specific politician"),
    all: bool = typer.Option(False, "--all", help="Fetch all recent trades"),
    chamber: str | None = typer.Option(None, help="Filter by chamber (house/senate)"),
    start: str | None = typer.Option(None, help="Start date filter (YYYY-MM-DD)"),
    end: str | None = typer.Option(None, help="End date filter (YYYY-MM-DD)"),
    data_dir: str = typer.Option("data", help="Data directory"),
):
    """Fetch congressional trades from Capitol Trades API (backup data source)."""
    from analyzer.capitol_trades import CapitolTradesSource

    if not politician and not all:
        print("Error: specify --politician NAME or --all", file=sys.stderr)
        raise typer.Exit(1)

    try:
        start_date = date.fromisoformat(start) if start else None
        end_date = date.fromisoformat(end) if end else None
    except ValueError:
        print("Error: dates must be in YYYY-MM-DD format", file=sys.stderr)
        raise typer.Exit(1)

    capitol = CapitolTradesSource(data_dir=data_dir, read_only=False)
    try:
        if politician:
            count = capitol.fetch_and_save_politician(politician, start_date, end_date)
            print(f"Saved {count} trades for {politician}")
        else:
            count = capitol.fetch_and_save_all(start_date, end_date, chamber)
            print(f"Saved {count} trades from Capitol Trades API")
    finally:
        capitol.close()

    raise typer.Exit(0)


@app.command()
def validate(
    ctx: typer.Context,
    train_start: str = typer.Option("2022-01-01", help="Training window start (YYYY-MM-DD)"),
    train_end: str = typer.Option("2023-12-31", help="Training window end (YYYY-MM-DD)"),
    test_start: str = typer.Option("2024-01-01", help="Test window start (YYYY-MM-DD)"),
    test_end: str = typer.Option("2025-06-30", help="Test window end (YYYY-MM-DD)"),
    full_grid: bool = typer.Option(False, "--full-grid", help="Use full 1296-combo grid (slow)"),
    data_dir: str = typer.Option("data", help="Data directory"),
):
    """
    Honest time-split validation with snooping corrections.

    Sweeps parameter configurations on the TRAINING window only, applies
    Benjamini-Hochberg / Bonferroni corrections for multiple comparisons,
    then evaluates the selected config exactly once on the TEST window.

    Uses Newey-West HAC t-stats to correct for overlapping return windows.
    Results are written to <data-dir>/validation_results.json.
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
        print("Error: --test-start must be after --train-end (no overlap)", file=sys.stderr)
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

    n_trials = 1
    for v in grid.values():
        n_trials *= len(v)
    print(f"Running validation with {n_trials} configs (trials for snooping correction)")

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
