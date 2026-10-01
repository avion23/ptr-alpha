#!/usr/bin/env python3

import logging
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import cast
from uuid import uuid4

import pandas as pd
import typer

from analyzer.member_ranking.buyer_scoring import (
    CONSENSUS_LOOKBACK_DAYS,
    CONSENSUS_MIN_BUYERS,
    _get_consensus_price_tickers,
)
from analyzer.database import Database
from analyzer.download import HouseTransactionSource
from analyzer.exceptions import AnalyzerError, DataSourceError
from analyzer.models import AnalysisMode
from analyzer.pipeline import (
    AnalysisParams,
    BacktestParams,
    TickerAnalysisParams,
    TickerScoringParams,
    run_analysis_pipeline,
    run_backtest_pipeline,
    run_parse_pipeline,
    run_recent_ticker_scoring,
    run_ticker_analysis,
    source_freshness_metrics,
)
from analyzer.price_snapshot import create_snapshot, save_snapshot
from analyzer.price_source import YFinancePriceSource
from analyzer.settings import Settings

app = typer.Typer(help="Congressional PTR disclosure analyzer", no_args_is_help=True)
logger = logging.getLogger(__name__)
# Bulk CLI parsing never shells out to Docling (multi-GB model workers per
# zero-row PDF); scripts that want it opt in by unsetting this.
os.environ.setdefault("PTR_SKIP_DOCLING", "1")
_CURRENT_YEAR = date.today().year
_HOUSE_PTR_FIRST_ARCHIVE_YEAR = 2015
_HOUSE_LEGACY_FIRST_ARCHIVE_YEAR = 2008
_BACKTEST_DEFAULTS = {
    name: field.default for name, field in BacktestParams.__dataclass_fields__.items()
}


def _cli_exit(exit_code=0, message=None):
    if message is not None:
        print(message, file=sys.stderr)
    raise typer.Exit(exit_code) from None


def _parse_cli_date(value, error_message, default=None):
    if value is None:
        return default
    try:
        return date.fromisoformat(value)
    except ValueError:
        _cli_exit(1, error_message)


def _open_database(data_dir, read_only):
    return Database(Path(data_dir) / "congress.duckdb", read_only=read_only)


def _new_ingestion_generation():
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    return f"{timestamp}-{uuid4().hex[:12]}"


@dataclass
class AppContext:
    settings: Settings
    transaction_source: HouseTransactionSource
    price_source: YFinancePriceSource


def setup_logging(verbose):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logging.getLogger("analyzer").setLevel(level)
    logging.getLogger("scripts").setLevel(level)
    # yfinance prints one ERROR per bad symbol; price_source already emits one
    # bounded failure summary with the affected ticker count/sample.
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)


def get_context(ctx, data_dir=None, read_only=False):
    if ctx.obj is None:
        settings = Settings()
        if data_dir and data_dir != "data":
            settings.data.data_dir = data_dir
        # Share a single DuckDB connection across both data sources to avoid
        # two independent Database instances pointing at the same file.
        shared_db = _open_database(settings.data.data_dir, read_only)
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
                "total_return_pct",
                "total_spy_alpha_pct",
            ]
        case AnalysisMode.MEMBER_RANKINGS:
            display_cols = [
                "member",
                "shrunk_alpha_pct",
                "shrunk_alpha_std_pct",
                "alpha_shrinkage",
                "avg_spy_alpha_pct",
                "avg_return_pct",
                "avg_spy_return_pct",
                "positive_alpha_rate",
                "positive_return_rate",
                "purchase_episodes",
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
    valid_modes = {"ranks", "signals", "member", "tickers"}
    if mode not in valid_modes:
        _cli_exit(1, f"Error: --mode must be one of {sorted(valid_modes)}")
    if mode == "member" and member is None and ticker is None:
        _cli_exit(1, "Error: --mode member requires --member NAME")


def _validate_positive_options(**options: int | float) -> None:
    """Reject nonsensical numeric CLI inputs before opening the database."""
    for name, value in options.items():
        if value <= 0:
            option = name.replace("_", "-")
            _cli_exit(1, f"Error: --{option} must be greater than zero")


def _validate_output(output: str) -> None:
    if output not in {"console", "csv"}:
        _cli_exit(1, "Error: --output must be one of ['console', 'csv']")


def _warn_live_ticker_coverage(app_ctx: AppContext, days_back: int) -> None:
    """Report candidate-window coverage metrics without gating BUY scoring."""
    as_of = date.today()
    try:
        metrics = source_freshness_metrics(
            app_ctx.transaction_source,
            pd.Timestamp(as_of),
            days_back,
        )
    except Exception as exc:
        print(
            f"WARNING: source freshness metric unavailable: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return

    for row in metrics.itertuples(index=False):
        if row.coverage_state == "in_window":
            continue
        if row.coverage_state == "missing":
            print(
                f"WARNING: {row.chamber} has no canonical disclosures in the unified "
                "database. Refresh before treating an empty result as current.",
                file=sys.stderr,
            )
            continue
        print(
            f"WARNING: {row.chamber} has no stored disclosure in the {days_back}-day "
            f"candidate window ending {as_of}; latest is {row.latest_disclosure_date} "
            f"({row.latest_disclosure_age_days} days ago). This is a coverage metric, "
            "not a trading-score input.",
            file=sys.stderr,
        )


@app.command()
def health(
    ctx: typer.Context,
    as_of: str | None = typer.Option(
        None, help="Coverage cutoff date (YYYY-MM-DD; defaults to today)"
    ),
    window_days: int = typer.Option(
        CONSENSUS_LOOKBACK_DAYS,
        "--window-days",
        help="Candidate-window length used by the coverage metric",
    ),
    data_dir: str = typer.Option("data", help="Data directory"),
):
    """Report House/Senate candidate-window coverage for monitoring.

    This command does not change or authorize BUY signals. A nonzero exit means
    at least one chamber is missing or has no disclosure inside the requested
    window, which should trigger a refresh/investigation in operations.
    """
    if window_days < 1:
        _cli_exit(1, "Error: --window-days must be greater than zero")
    cutoff = _parse_cli_date(
        as_of or None, "Error: --as-of must use YYYY-MM-DD", date.today()
    )

    app_ctx = get_context(ctx, data_dir, read_only=True)
    try:
        metrics = source_freshness_metrics(
            app_ctx.transaction_source,
            pd.Timestamp(cutoff),
            window_days,
        )
    except Exception as exc:
        _cli_exit(
            1,
            f"Error: source freshness metric failed: {type(exc).__name__}: {exc}",
        )

    print(metrics.to_string(index=False))
    healthy = bool((metrics["coverage_state"] == "in_window").all())
    _cli_exit(0 if healthy else 1)


def _consensus_score_display(score: pd.DataFrame) -> pd.DataFrame:
    columns = [
        column
        for column in (
            "ticker",
            "num_buyers",
            "buyers",
            "signal_score",
            "max_trade_to_disclosure_days",
        )
        if column in score.columns
    ]
    return score[columns]


def _run_ticker_mode(
    app_ctx: AppContext,
    mode: str,
    ticker: str,
    year: int,
    days_back: int,
    min_buyers: int,
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
        _cli_exit(
            1,
            "Error: --output csv is not supported for --ticker analysis; "
            "use --output console.",
        )
    params = TickerAnalysisParams(
        ticker=ticker,
        year=year,
        days_back=days_back,
        min_buyers=min_buyers,
        as_of_date=as_of_date,
    )
    result = run_ticker_analysis(params, app_ctx.transaction_source)
    if result.success and hasattr(result, "data") and result.data:
        print(f"\n=== Buyers of {result.data['ticker']} ===")
        print(result.data["buyers"].to_string(index=False))
        print("\n=== Signal Score ===")
        print(_consensus_score_display(result.data["score"]).to_string(index=False))
        score = result.data["score"]["signal_score"].iloc[0]
        verdict = "BUY CANDIDATE" if score > 0 else "NO BUY"
        print(f"\nRecommendation: {verdict} (score {score:.2f})")
    _cli_exit(0 if result.success else 1)


def _run_tickers_mode(
    app_ctx: AppContext,
    year: int,
    days_back: int,
    min_buyers: int,
    top_n: int,
    output: str,
    as_of_date: date | None,
) -> None:
    """Handle --mode tickers."""
    if output == "csv":
        _cli_exit(
            1,
            "Error: --output csv is not supported for --mode tickers; "
            "use --output console.",
        )
    params = TickerScoringParams(
        year=year,
        days_back=days_back,
        min_buyers=min_buyers,
        top_n=top_n,
        as_of_date=as_of_date,
    )
    result = run_recent_ticker_scoring(app_ctx.transaction_source, params)
    if result.success and hasattr(result, "data") and result.data:
        data = result.data
        if not data["result"].empty:
            print(
                f"\n=== Current Buy Candidates as of {data['as_of_date']} (Last {data['days_back']} Days, {data['min_buyers']}+ Buyers) ==="
            )
            print(_consensus_score_display(data["result"]).to_string(index=False))
        else:
            print(f"\nNo positive buy candidates as of {data['as_of_date']}.")
    _cli_exit(0 if result.success else 1)


def _run_analysis_mode(
    app_ctx: AppContext,
    year: int,
    horizons: list[int],
    member: str | None,
    top_n: int,
    mode: str,
    output: str,
    sectors: bool,
    as_of_date: date | None,
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
        member_filter=member,
        top_n=top_n,
        mode=analysis_mode,
        include_sector_analysis=sectors,
        as_of_date=as_of_date,
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
    _cli_exit(0 if result.success else 1)


@app.command()
def analyze(
    ctx: typer.Context,
    year: int = typer.Option(_CURRENT_YEAR, help="Year to process"),
    mode: str = typer.Option(
        "ranks",
        help="Output mode: ranks | signals | member | tickers",
    ),
    member: str | None = typer.Option(None, help="Filter to specific member"),
    ticker: str | None = typer.Option(None, help="Analyze specific ticker"),
    horizons: list[int] = typer.Option(
        [90],
        help=(
            "Single time horizon in days: ranks/signals/member rank on this "
            "horizon only. --mode tickers and --ticker ignore horizons and use "
            "a fixed 28-day consensus lookback."
        ),
    ),
    top_n: int = typer.Option(20, help="Number of results to show"),
    as_of: str | None = typer.Option(
        None,
        help="Analysis cutoff date (YYYY-MM-DD; defaults to today)",
    ),
    sectors: bool = typer.Option(
        False,
        "--sectors",
        help="Fetch optional sector metadata for rank output",
    ),
    output: str = typer.Option(
        "console",
        help="Output format: console or csv (csv is rejected for --mode tickers and --ticker)",
    ),
    data_dir: str = typer.Option("data", help="Data directory"),
):
    """
    Unified analysis command. Use --mode to select output type:

      ranks    - Descriptive member purchase outcomes (default)
      signals  - Show top historical purchase outcomes
      member   - Show historical purchase outcomes for one member
      tickers  - Score current multi-buyer equity candidates
    """
    _validate_mode(mode, member, ticker)
    _validate_positive_options(year=year, top_n=top_n)
    if not horizons or any(horizon <= 0 for horizon in horizons):
        _cli_exit(1, "Error: --horizons values must be greater than zero")
    if len(horizons) > 1:
        _cli_exit(
            1,
            "Error: --horizons accepts a single value; ranks/signals/member "
            f"rank on one horizon only (got {len(horizons)}: {sorted(horizons)}). "
            "Run once per horizon.",
        )
    _validate_output(output)
    if output == "csv" and (ticker is not None or mode == "tickers"):
        _cli_exit(
            1,
            "Error: --output csv is not supported for --ticker/--mode tickers; "
            "use --output console",
        )
    as_of_date = _parse_cli_date(
        as_of or None, "Error: --as-of must use YYYY-MM-DD"
    )
    app_ctx = get_context(ctx, data_dir, read_only=True)
    if (
        as_of_date is None
        and year == date.today().year
        and (ticker is not None or mode == "tickers")
    ):
        _warn_live_ticker_coverage(app_ctx, CONSENSUS_LOOKBACK_DAYS)

    if ticker:
        _run_ticker_mode(
            app_ctx,
            mode,
            ticker,
            year,
            CONSENSUS_LOOKBACK_DAYS,
            CONSENSUS_MIN_BUYERS,
            as_of_date,
            output,
        )
    elif mode == "tickers":
        _run_tickers_mode(
            app_ctx,
            year,
            CONSENSUS_LOOKBACK_DAYS,
            CONSENSUS_MIN_BUYERS,
            top_n,
            output,
            as_of_date,
        )
    else:
        _run_analysis_mode(
            app_ctx,
            year,
            horizons,
            member,
            top_n,
            mode,
            output,
            sectors,
            as_of_date,
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
        _cli_exit(1, f"House fetch incomplete: {exc}")
    _print_house_fetch_summary(summary)
    _cli_exit(0)


def _activate_house_generation(transaction_source, year: int) -> None:
    """Verify all artifacts parsed and activate the latest generation.

    Raises DataSourceError when no generation exists or artifacts remain
    unresolved, leaving the generation incomplete and new rows hidden
    from canonical reads.
    """
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


def _release_parent_db_for_ocr(app_ctx: AppContext) -> Path | None:
    """Checkpoint and close parent DuckDB handles so OCR can open the file.

    Returns the database path when a real parent handle was released, else
    None (test doubles hold no file lock). Both sources share one handle via
    get_context; close each distinct real handle once.
    """
    owners = (
        getattr(app_ctx, "transaction_source", None),
        getattr(app_ctx, "price_source", None),
    )
    seen: list = []
    for owner in owners:
        db = getattr(owner, "db", None)
        if isinstance(db, Database) and not any(db is prior for prior in seen):
            seen.append(db)
    if not seen:
        return None
    db_path = Path(seen[0].db_path)
    for db in seen:
        try:
            db.conn.execute("CHECKPOINT")
        except Exception:
            logger.debug("Pre-OCR checkpoint failed", exc_info=True)
        try:
            db.close()
        except Exception:
            logger.debug("Pre-OCR parent close failed", exc_info=True)
    return db_path


def _reacquire_parent_db_after_ocr(
    app_ctx: AppContext, db_path: Path | None
) -> None:
    """Reopen the parent handle after isolated OCR finishes."""
    if db_path is None:
        return
    fresh: Database | None = None
    for owner in (
        getattr(app_ctx, "transaction_source", None),
        getattr(app_ctx, "price_source", None),
    ):
        if owner is None:
            continue
        if not isinstance(getattr(owner, "db", None), Database):
            continue
        if fresh is None:
            fresh = Database(db_path, read_only=False)
        owner.db = fresh


def _run_gemini_ocr_year_subprocess(
    data_dir: str | Path, year: int, *, timeout: int = 7200
) -> tuple[int, str | None]:
    """Run one year's OCR in a child interpreter without the parent lock.

    Returns (inserted, failure_reason|None). The child prints
    ``Total inserted: N`` on success, matching the standalone entrypoint
    pattern; the parent holds no DuckDB handle while it runs.
    """
    repo_root = Path(__file__).resolve().parents[2]
    ocr_code = (
        "import sys;"
        f"sys.path.insert(0, {str(repo_root)!r});"
        "from scripts.ocr_zero_rows import run_gemini_ocr_for_year;"
        f"inserted = run_gemini_ocr_for_year({int(year)}, data_dir={str(data_dir)!r});"
        "print(f'Total inserted: {inserted}')"
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-c", ocr_code],
            text=True,
            capture_output=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return 0, "timeout"
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout)[-500:]
        return 0, f"exit {proc.returncode}: {detail}"
    match = re.search(r"Total inserted:\s*(\d+)", proc.stdout)
    if match is None:
        return 0, f"missing Total inserted in output: {(proc.stdout or '')[-500:]}"
    return int(match.group(1)), None


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
            _activate_house_generation(app_ctx.transaction_source, year)
            parse_success = True
        else:
            result = run_parse_pipeline(app_ctx.transaction_source, year)
            parse_success = result.success
    except Exception:
        logger.exception("Parse pipeline failed")
    ocr_inserted = 0
    if use_gemini_ocr:
        db_path = _release_parent_db_for_ocr(app_ctx)
        try:
            ocr_inserted, ocr_error = _run_gemini_ocr_year_subprocess(
                app_ctx.settings.data.data_dir, year
            )
            if ocr_error is not None:
                logger.warning("Gemini OCR failed for %d: %s", year, ocr_error)
                ocr_inserted = 0
            else:
                print(f"  Gemini OCR {year}: {ocr_inserted} transactions inserted")
        finally:
            _reacquire_parent_db_after_ocr(app_ctx, db_path)
    if use_gemini_ocr and ocr_inserted > 0 and not parse_success:
        try:
            _activate_house_generation(app_ctx.transaction_source, year)
            parse_success = True
        except Exception:
            logger.exception("Post-OCR generation activation failed")
    if not parse_success and use_gemini_ocr and ocr_inserted > 0:
        logger.warning(
            "Parse pipeline failed but Gemini OCR inserted %s rows", ocr_inserted
        )
    _cli_exit(0 if parse_success else 1)


@app.command()
def backtest(
    ctx: typer.Context,
    start: str = typer.Option(..., help="Backtest start date (YYYY-MM-DD)"),
    end: str = typer.Option(..., help="Backtest end date (YYYY-MM-DD)"),
    horizon: int = typer.Option(
        _BACKTEST_DEFAULTS["horizon"], help="Forward return horizon in days"
    ),
    top_n: int = typer.Option(
        _BACKTEST_DEFAULTS["top_n"], help="Top N recommendations per backtest date"
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

    Uses only disclosures public at each as-of date and enters on the next
    NYSE session. The declared --horizon is the evaluation holding horizon.
    """
    start_date, end_date = _parse_cli_date_range(start, end)

    _validate_positive_options(
        horizon=horizon,
        top_n=top_n,
        frequency_days=frequency_days,
    )

    app_ctx = get_context(ctx, data_dir, read_only=True)
    params = BacktestParams(
        start_date=start_date,
        end_date=end_date,
        horizon=horizon,
        lookback_days=CONSENSUS_LOOKBACK_DAYS,
        min_buyers=CONSENSUS_MIN_BUYERS,
        top_n=top_n,
        frequency_days=frequency_days,
    )
    result = run_backtest_pipeline(
        params, app_ctx.transaction_source, app_ctx.price_source
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
            benchmark_status = summary.attrs.get("spy_benchmark_status")
            if benchmark_status:
                benchmark_reason = summary.attrs.get("spy_benchmark_reason")
                detail = f" ({benchmark_reason})" if benchmark_reason else ""
                print(f"SPY buy/hold benchmark: {benchmark_status}{detail}")

            observations = data.get("date_observations", pd.DataFrame())
            recommendations = (
                int(observations["recommendation_count"].sum())
                if "recommendation_count" in observations.columns
                else 0
            )
            evaluable = (
                int(observations["evaluable_recommendation_count"].sum())
                if "evaluable_recommendation_count" in observations.columns
                else 0
            )
            print(
                f"\nDates evaluated: {data.get('evaluable_dates', 0)}/{data.get('total_as_of_dates', 0)}"
            )
            print(
                f"Recommendations issued: {recommendations}; individually evaluable: {evaluable}"
            )
        else:
            print("\n=== No backtest results produced ===")
    _cli_exit(0 if result.success else 1)


@app.command()
def portfolio(
    ctx: typer.Context,
    start: str = typer.Option(..., help="Simulation start date (YYYY-MM-DD)"),
    end: str = typer.Option(..., help="Simulation end date (YYYY-MM-DD)"),
    top_n: int = typer.Option(
        _BACKTEST_DEFAULTS["top_n"], help="Top N recommendations per backtest date"
    ),
    # Intentionally bi-weekly (not the backtest's 30d step): rebalance cadence
    # for the portfolio sim, independent of the sweep-calibrated backtest.
    frequency_days: int = typer.Option(14, help="Days between rolling backtest dates"),
    initial_capital: float = typer.Option(20000, help="Initial portfolio capital"),
    max_positions: int = typer.Option(5, help="Maximum concurrent positions"),
    hold_days: int = typer.Option(120, help="Hold period in days before forced exit"),
    entry_slippage_bps: float = typer.Option(
        0.0, help="Modeled entry slippage in basis points"
    ),
    exit_slippage_bps: float = typer.Option(
        0.0, help="Modeled exit slippage in basis points"
    ),
    data_dir: str = typer.Option("data", help="Data directory"),
):
    """
    Run portfolio-level simulation with overlapping positions and constraints.

    Unlike the backtest command which evaluates each recommendation independently,
    this simulates one shared cash account across overlapping holding periods.
    """
    start_date, end_date = _parse_cli_date_range(start, end)

    _validate_positive_options(
        top_n=top_n,
        frequency_days=frequency_days,
        initial_capital=initial_capital,
        max_positions=max_positions,
        hold_days=hold_days,
    )
    for name, value in (
        ("entry-slippage-bps", entry_slippage_bps),
        ("exit-slippage-bps", exit_slippage_bps),
    ):
        if not 0 <= value < 10_000:
            _cli_exit(1, f"Error: --{name} must be in [0, 10000)")

    app_ctx = get_context(ctx, data_dir, read_only=True)

    from datetime import timedelta

    from analyzer.portfolio_sim import PortfolioConfig, PortfolioSimulator

    tx_start = start_date - timedelta(days=CONSENSUS_LOOKBACK_DAYS)
    all_transactions = app_ctx.transaction_source.db.get_transactions_by_date_range(
        tx_start, end_date
    )
    if all_transactions.empty:
        _cli_exit(1, "Error: no transactions found for portfolio simulation")

    prices, recommendations = _load_portfolio_inputs(
        app_ctx,
        all_transactions,
        end_date,
        CONSENSUS_LOOKBACK_DAYS,
        CONSENSUS_MIN_BUYERS,
        top_n,
        frequency_days,
        start_date,
    )

    config = PortfolioConfig(
        initial_capital=initial_capital,
        max_positions=max_positions,
        hold_period_days=hold_days,
        rebalance_freq_days=frequency_days,
        entry_slippage_pct=entry_slippage_bps / 10_000.0,
        exit_slippage_pct=exit_slippage_bps / 10_000.0,
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


def _parse_cli_date_range(start, end):
    """Parse optional YYYY-MM-DD inputs and validate end >= start."""
    start_date = _parse_cli_date(start, "Error: dates must be in YYYY-MM-DD format")
    end_date = _parse_cli_date(end, "Error: dates must be in YYYY-MM-DD format")

    if start_date is not None and end_date is not None and end_date < start_date:
        _cli_exit(1, "Error: --end must be on or after --start")

    return start_date, end_date


def _load_portfolio_inputs(
    app_ctx,
    all_transactions,
    end_date,
    lookback_days,
    min_buyers,
    top_n,
    frequency_days,
    start_date,
):
    """Load execution prices and consensus recommendations."""
    from analyzer import analysis

    all_tickers = sorted(set(_get_consensus_price_tickers(all_transactions)) | {"SPY"})
    prices = app_ctx.transaction_source.db.get_prices(
        all_tickers, start_date, end_date
    )
    if prices.empty:
        _cli_exit(1, "Error: no price data available")

    as_of_dates = pd.date_range(start_date, end_date, freq=f"{frequency_days}D")
    all_recs = []
    for as_of in as_of_dates:
        # date_range never yields NaT; narrow the stubs' union explicitly.
        recs = analysis.backtest_recommendations(
            all_transactions,
            cast(pd.Timestamp, pd.Timestamp(as_of)),
            lookback_days=lookback_days,
            min_buyers=min_buyers,
            top_n=top_n,
        )
        if recs.empty:
            continue
        recs = recs.copy()
        recs["as_of_date"] = as_of
        all_recs.append(recs)

    if not all_recs:
        _cli_exit(1, "No recommendations produced for any backtest date")

    recommendations = pd.concat(all_recs, ignore_index=True)
    print(
        f"Collected {len(recommendations)} recommendations across {len(as_of_dates)} dates"
    )
    return prices, recommendations


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
        _cli_exit(1)

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
    _cli_exit(0)


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
                    all_years or refresh_metadata or archive_year == date.today().year
                ),
            )
            summaries.append(summary)
            _print_house_fetch_summary(summary)
    except Exception as exc:
        logger.warning("House PDF fetch failed: %s", exc)
        print(f"House fetch incomplete: {exc}")
        print("FAILED steps: fetch")
        _cli_exit(1)

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
                app_ctx.transaction_source.parse_cached_pdfs(archive_year, force=True)
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
        # The OCR helper opens its own DuckDB handles. A held parent handle
        # causes a same-process ConnectionException (different config) or a
        # child-process lock conflict, so release it and reuse the isolated-
        # subprocess pattern of the standalone entrypoint for every year.
        db_path = _release_parent_db_for_ocr(app_ctx)
        try:
            for archive_year in archive_years:
                inserted, ocr_error = _run_gemini_ocr_year_subprocess(
                    app_ctx.settings.data.data_dir, archive_year
                )
                if ocr_error is not None:
                    failed_steps.append(f"gemini_ocr:{archive_year}")
                    logger.warning(
                        "Gemini OCR failed for %d: %s", archive_year, ocr_error
                    )
                    continue
                print(
                    f"  Gemini OCR {archive_year}: {inserted} transactions inserted"
                )
        finally:
            _reacquire_parent_db_after_ocr(app_ctx, db_path)
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
                (
                    SELECT COUNT(*)
                    FROM transactions raw
                    WHERE raw.source IN ('house_pdf', 'gemini_ocr', 'senate_efd')
                      AND EXTRACT(YEAR FROM raw.disclosure_date) = ?
                      AND raw.transaction_date > raw.disclosure_date
                )
            FROM canonical_transactions
            WHERE EXTRACT(YEAR FROM disclosure_date) = ?
            """,
            [summary_year, summary_year],
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
        _cli_exit(1)
    _cli_exit(0)


@app.command()
def fetch_form4(
    ctx: typer.Context,
    ticker: str | None = typer.Option(
        None, help="Fetch Form 4 open-market purchases for one ticker (e.g. VST)"
    ),
    year: int = typer.Option(
        date.today().year, help="Filing year to scan for Form 4 purchases"
    ),
    sweep_days: int = typer.Option(
        0, "--sweep-days", help="Discover recent Form 4 purchases over this many days"
    ),
    data_dir: str = typer.Option(
        "data", help="Data directory for the canonical database"
    ),
):
    """Fetch SEC Form 4 open-market insider purchases (official source).

    Rows persist in the canonical DuckDB with source='form4'; only Table I
    transaction-code-P purchases are stored, fail-closed on missing fields.
    """
    from analyzer.form4 import Form4Error, Form4Source

    if ticker is not None and not ticker.strip():
        _cli_exit(1, "Error: --ticker must be non-empty")
    if year < 2000 or year > date.today().year + 1:
        _cli_exit(1, "Error: --year is out of plausible range")
    if sweep_days < 0:
        _cli_exit(1, "Error: --sweep-days must be non-negative")

    ingestion_generation = _new_ingestion_generation()
    src = Form4Source(
        data_dir=data_dir,
        read_only=False,
        ingestion_generation=ingestion_generation,
    )
    try:
        try:
            if sweep_days:
                if not hasattr(src, "fetch_and_save_sweep"):
                    _cli_exit(
                        1,
                        "Error: Form 4 sweep is unavailable; "
                        "fetch_and_save_sweep is not implemented",
                    )
                count = src.fetch_and_save_sweep(sweep_days)
                print(f"Saved {count} new Form 4 purchase rows from {sweep_days}-day sweep")
            elif ticker:
                count = src.fetch_and_save_ticker(ticker, year)
                print(
                    f"Saved {count} new Form 4 purchase rows for "
                    f"{ticker.strip().upper()} ({year})"
                )
            else:
                _cli_exit(
                    1,
                    "Error: --ticker is required unless --sweep-days is given",
                )
        except Form4Error as exc:
            _cli_exit(1, f"Error: {exc}")
    finally:
        src.close()

    _cli_exit(0)


@app.command()
def follow_member(
    ctx: typer.Context,
    member: str = typer.Option(..., "--member", help="Member name as filed (e.g. 'Nancy Pelosi')"),
    discount_pct: float = typer.Option(
        10.0, help="Alert threshold: flag positions this far below cost, in percent"
    ),
    min_cost: float = typer.Option(
        0.0, help="Ignore positions with total cost below this dollar amount"
    ),
    include_unpromoted: bool = typer.Option(
        False,
        "--include-unpromoted",
        help="Add unvalidated filings from not-yet-promoted House generations",
    ),
    as_of: str | None = typer.Option(
        None, help="Position date (YYYY-MM-DD). Defaults to today."
    ),
    data_dir: str = typer.Option(
        "data", help="Data directory for the canonical database"
    ),
):
    """Show a member's open positions with cost basis vs current price.

    Reconstructs holdings from all canonical filings (buys add implied
    shares, sells relieve FIFO) and flags positions trading below the
    member's cost — the follow-through that closed-window scoring misses.
    """
    from analyzer.positions import (
        PositionsError,
        build_positions,
        discount_alerts,
        load_member_trades,
        load_price_history,
    )

    as_of_date = _parse_cli_date(
        as_of or None, "Error: --as-of must be YYYY-MM-DD", date.today()
    )
    if discount_pct < 0:
        _cli_exit(1, "Error: --discount-pct must be non-negative")
    if not member:
        _cli_exit(1, "Error: provide at least one --member")

    db = _open_database(data_dir, read_only=True)
    try:
        try:
            trades = load_member_trades(
                db, member, include_unpromoted=include_unpromoted
            )
        except Exception as exc:
            _cli_exit(1, f"Error: {exc}")
        if include_unpromoted:
            print(
                "Warning: unpromoted rows are unvalidated and approximate; "
                "canonical rows take precedence.",
                file=sys.stderr,
            )
        if trades.empty:
            print(f"No canonical transactions for {member!r}")
            _cli_exit(0)
        histories = {}
        for ticker in trades["ticker"].dropna().unique():
            histories[str(ticker)] = load_price_history(db, str(ticker))
        try:
            positions = build_positions(trades, histories, as_of=as_of_date)
        except PositionsError as exc:
            _cli_exit(1, f"Error: {exc}")
    finally:
        db.close()

    if min_cost > 0:
        positions = positions.loc[positions["total_cost"] >= min_cost].copy()
    alerts = discount_alerts(positions, threshold_pct=-abs(discount_pct))

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(f"\n=== Open positions for {member} (as of {as_of_date}) ===")
        if positions.empty:
            print("No open positions.")
        else:
            show = positions[
                [
                    "ticker",
                    "shares",
                    "cost_basis",
                    "current_price",
                    "discount_pct",
                    "first_buy",
                    "last_activity",
                ]
            ].copy()
            show["shares"] = show["shares"].round(1)
            show["cost_basis"] = show["cost_basis"].round(2)
            show["current_price"] = show["current_price"].round(2)
            show["discount_pct"] = show["discount_pct"].round(1)
            print(show.to_string(index=False))
        print(f"\n=== On sale (>= {discount_pct:.0f}% below {member} cost) ===")
        if alerts.empty:
            print("None.")
        else:
            for row in alerts.itertuples():
                print(
                    f"{row.ticker}: cost ${row.cost_basis:.2f} -> "
                    f"${row.current_price:.2f} ({row.discount_pct:.1f}%)"
                )

    _cli_exit(0)


@app.command()
def setups(
    ctx: typer.Context,
    member: list[str] = typer.Option([], "--member", help="Member name; repeatable"),
    ticker: list[str] = typer.Option([], "--ticker", help="Ticker filter; repeatable"),
    as_of: str | None = typer.Option(
        None, "--as-of", help="Cutoff date (YYYY-MM-DD; defaults to today)"
    ),
    include_unpromoted: bool = typer.Option(
        False,
        "--include-unpromoted",
        help="Include unvalidated filings from unpromoted House generations",
    ),
    data_dir: str = typer.Option("data", "--data-dir", help="Data directory"),
):
    """Rank cross-source setups with per-actor price discipline.

    Candidates = holdings (per --member, persistent) + purchase events
    (all official sources, point-in-time). Weights are source priors
    (n_matured=0 v1 wiring). Fair region is enforced per actor inside
    scoring: current <= 1.05 x that actor's entry reference.
    """
    as_of_date = _parse_cli_date(
        as_of or None, "Error: --as-of must use YYYY-MM-DD", date.today()
    )

    try:
        from analyzer.actors import compute_weight
        from analyzer.pipeline import eligible_events
        from analyzer.positions import (
            coalesce_actor_variants,
            dedupe_member_names,
            gate_closed_events,
            holdings_candidates,
        )
        from analyzer.manager_watchlist import watchlist_actor_ids
        from analyzer.setups import score as score_setups

        def _congress_holdings(frame):
            """Congress rows initiate scoring; officer/manager holdings rows
            exist only for closure gating (their events carry the references)."""
            if frame.empty or "kind" not in frame.columns:
                return frame
            return frame.loc[frame["kind"] == "congress"].copy()
    except ImportError as exc:
        _cli_exit(1, f"Error: setups is unavailable ({exc})")

    try:
        db = _open_database(data_dir, read_only=True)
    except Exception as exc:
        _cli_exit(1, f"Error: could not open read-only database: {exc}")

    requested_tickers = {symbol.strip().upper() for symbol in ticker}
    try:
        frames = []
        candidate_frames = []
        # One human, one holdings load: drop requested names that denote an
        # already-requested member (variants, nicknames, middle names).
        member = dedupe_member_names(list(member))
        for selected_member in member:
            frame = holdings_candidates(
                db,
                selected_member,
                as_of_date,
                include_unpromoted=include_unpromoted,
            )
            if not frame.empty:
                frames.append(frame)
                candidate_frames.append(_congress_holdings(frame))
        events = eligible_events(db, as_of_date)
        frames = [frame for frame in frames if not frame.empty]
        if not events.empty:
            if requested_tickers:
                events = events.loc[
                    events["ticker"].astype("string").str.upper().isin(
                        requested_tickers
                    )
                ].copy()
            elif frames:
                held = {
                    str(value).strip().upper()
                    for frame in frames
                    for value in frame["ticker"].dropna()
                }
                if held:
                    events = events.loc[
                        events["ticker"].astype("string").str.upper().isin(held)
                    ].copy()
            else:
                events = events.iloc[0:0]
            if not events.empty:
                event_kinds: dict[str, str] = {}
                for actor_id in events["actor_id"].dropna():
                    text = str(actor_id)
                    if ":" not in text:
                        continue
                    kind, _, name = text.partition(":")
                    if name:
                        event_kinds.setdefault(name, kind)
                event_tickers = set(events["ticker"].astype("string").str.upper())
                for event_member in dedupe_member_names(
                    sorted(event_kinds), exclude=list(member)
                ):
                    frame = holdings_candidates(
                        db,
                        event_member,
                        as_of_date,
                        include_unpromoted=include_unpromoted,
                    )
                    if not frame.empty:
                        frame = frame.loc[
                            frame["ticker"]
                            .astype("string")
                            .str.upper()
                            .isin(event_tickers)
                        ].copy()
                        if not frame.empty:
                            frames.append(frame)
                            candidate_frames.append(_congress_holdings(frame))
                position_candidates = (
                    pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
                )
                events = coalesce_actor_variants(events, position_candidates)
                events = gate_closed_events(events, position_candidates)
                frames.append(events)
                candidate_frames.append(events)
        if not frames:
            print("No candidates.")
            _cli_exit(0)
        candidates = pd.concat(
            [frame for frame in candidate_frames if not frame.empty],
            ignore_index=True,
        )
        if requested_tickers:
            candidates = candidates.loc[
                candidates["ticker"].astype("string").str.upper().isin(
                    requested_tickers
                )
            ].copy()
        if candidates.empty:
            print("No candidates.")
            _cli_exit(0)

        symbols = sorted(
            {str(value).strip().upper() for value in candidates["ticker"].dropna()}
        )
        current_prices = {}
        for symbol in symbols:
            row = db.conn.execute(
                "SELECT close FROM prices WHERE ticker = ? AND date <= ? "
                "ORDER BY date DESC LIMIT 1",
                [symbol, as_of_date],
            ).fetchone()
            if row is not None and row[0] is not None:
                current_prices[symbol] = float(row[0])

        weights = {}
        for row in candidates.itertuples():
            actor_id = str(row.actor_id)
            if actor_id not in weights:
                weights[actor_id] = compute_weight(
                    str(row.kind), 0, 0.0, str(row.source)
                )
        ranked, blocked = score_setups(
            candidates,
            weights,
            current_prices,
            watchlist=watchlist_actor_ids(),
        )
    except typer.Exit:
        raise
    except Exception as exc:
        _cli_exit(1, f"Error: setup ranking failed: {exc}")
    finally:
        db.close()

    print("=== Ranked setups ===")
    print(ranked.to_string(index=False) if not ranked.empty else "None.")
    print("\n=== Blocked ===")
    print(blocked.to_string(index=False) if not blocked.empty else "None.")
    _cli_exit(0)


@app.command()
def fetch_13f(
    ctx: typer.Context,
    watchlist: str = typer.Option(
        "VST:VISTRA",
        help="Comma-separated TICKER:keyword pairs scoping 13F name matching",
    ),
    all_increases: bool = typer.Option(
        False,
        "--all-increases",
        help="Fetch and save all discovered 13F position increases",
    ),
    data_dir: str = typer.Option(
        "data", help="Data directory for the canonical database"
    ),
):
    """Fetch 13F-HR quarter-over-quarter position increases for watchlist managers.

    Rows persist in the canonical DuckDB with source='13f' and subtype
    INFERRED_POSITION_INCREASE (13F shows positions, not trades).
    """
    from analyzer.form13f import ThirteenFError, ThirteenFSource

    parsed: dict[str, set[str]] = {}
    for item in watchlist.split(","):
        if ":" not in item:
            _cli_exit(
                1,
                f"Error: --watchlist items must be TICKER:keyword, got {item!r}",
            )
        symbol, keyword = item.split(":", 1)
        if not symbol.strip() or not keyword.strip():
            _cli_exit(
                1,
                f"Error: --watchlist items must be TICKER:keyword, got {item!r}",
            )
        parsed.setdefault(symbol.strip().upper(), set()).add(keyword.strip())

    ingestion_generation = _new_ingestion_generation()
    src = ThirteenFSource(
        watchlist=parsed,
        data_dir=data_dir,
        read_only=False,
        ingestion_generation=ingestion_generation,
    )
    try:
        try:
            if all_increases:
                if not hasattr(src, "fetch_all_increases") or not hasattr(
                    src, "save_increases"
                ):
                    _cli_exit(
                        1,
                        "Error: 13F increase discovery is unavailable; "
                        "fetch_all_increases/save_increases is not implemented",
                    )
                count = src.save_increases(src.fetch_all_increases())
            else:
                count = src.fetch_and_save_all()
        except ThirteenFError as exc:
            _cli_exit(1, f"Error: {exc}")
        print(f"Saved {count} new 13F position-increase rows ({len(parsed)} tickers)")
    finally:
        src.close()

    _cli_exit(0)


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
        _cli_exit(1, "Error: specify exactly one of --politician NAME or --all")

    start_date, end_date = _parse_cli_date_range(start or None, end or None)

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
        _cli_exit(1, f"Error: {exc}")

    print(f"Wrote {len(df)} reconciliation records to {output}")
    print("No canonical transactions were saved.")
    _cli_exit(0)


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
        "data", help="Data directory for the canonical congressional database"
    ),
):
    """Fetch Senate PTR trades from efdsearch.senate.gov (official source).

    Senate rows are persisted in the canonical congressional DuckDB; source and
    chamber identity keep Senate refreshes isolated from House rows.
    """
    from datetime import timedelta

    from analyzer.senate_efd import SenateEFDSource

    date_error = "Error: dates must be YYYY-MM-DD and lookback > 0"
    end_date = _parse_cli_date(end or None, date_error, date.today())
    if lookback is not None:
        if lookback <= 0:
            _cli_exit(1, date_error)
        start_date = end_date - timedelta(days=lookback)
    else:
        start_date = _parse_cli_date(start or None, date_error)
        if start_date is None:
            try:
                start_date = end_date.replace(year=end_date.year - 1)
            except ValueError:
                _cli_exit(1, date_error)

    if start_date > end_date:
        _cli_exit(1, "Error: --start must be on or before --end")

    ingestion_generation = _new_ingestion_generation()
    src = SenateEFDSource(
        data_dir=data_dir,
        read_only=False,
        ingestion_generation=ingestion_generation,
    )
    try:
        count = src.fetch_and_save_all(start_date, end_date)
        print(
            f"Saved {count} new Senate eFD trades ({start_date} to {end_date}) into {data_dir}/"
        )
    finally:
        src.close()

    _cli_exit(0)


def main():
    try:
        app()
    except AnalyzerError as e:
        _cli_exit(1, f"Error: {e}")
    except KeyboardInterrupt:
        _cli_exit(130, "\nOperation cancelled by user")
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.exception(f"Unexpected error: {e}")
        _cli_exit(1)


if __name__ == "__main__":
    main()
