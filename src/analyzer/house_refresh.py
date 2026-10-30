"""House refresh orchestration on the caller's single owned database."""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

from analyzer.download import HouseFetchSummary
from analyzer.exceptions import DataResult

logger = logging.getLogger(__name__)


def house_refresh_counts(db) -> tuple[int, int]:
    """Return canonical and raw transaction counts for refresh reporting."""
    canonical = db.conn.execute(
        "SELECT COUNT(*) FROM canonical_transactions"
    ).fetchone()[0]
    raw = db.conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
    return canonical, raw


def acquire_house_archives(
    transaction_source,
    archive_years: list[int],
    *,
    all_years: bool,
    refresh_metadata: bool,
) -> DataResult[list[HouseFetchSummary]]:
    """Acquire every requested archive before parsing, retrying boundedly.

    For all-years refreshes, a twice-failed archive is quarantined from this
    refresh while later years continue. A single-year refresh remains fail-fast.
    The caller retains database ownership throughout acquisition and parsing.
    """
    summaries: list[HouseFetchSummary] = []
    first_error: Exception | None = None

    try:
        fail_fast = len(archive_years) == 1
        for archive_year in archive_years:
            should_refresh_metadata = (
                all_years
                or refresh_metadata
                or archive_year == date.today().year
            )
            try:
                summary = transaction_source.fetch_and_cache_pdfs(
                    archive_year, refresh_metadata=should_refresh_metadata
                )
            except Exception as year_exc:
                if fail_fast:
                    raise
                logger.warning(
                    "House archive %d fetch failed, retrying: %s",
                    archive_year,
                    year_exc,
                )
                try:
                    summary = transaction_source.fetch_and_cache_pdfs(
                        archive_year, refresh_metadata=should_refresh_metadata
                    )
                except Exception as retry_exc:
                    logger.warning(
                        "House archive %d failed twice, skipping: %s",
                        archive_year,
                        retry_exc,
                    )
                    if first_error is None:
                        first_error = retry_exc
                    continue
            summaries.append(summary)

        return DataResult(success=first_error is None, data=summaries, error=first_error)
    except Exception as exc:
        logger.warning("House PDF fetch failed: %s", exc)
        return DataResult(success=False, error=exc)


def parse_house_archives(
    transaction_source,
    archive_years: list[int],
    data_dir: str | Path,
    *,
    force: bool,
    gemini_workers: int = 0,
) -> list[int]:
    """Use one rolling window and one owner across every acquired archive."""
    from analyzer.ocr.coordinator import run_pipeline

    run_pipeline(data_dir, archive_years, database=transaction_source.db,
                 local_workers=transaction_source.parallel_workers,
                 gemini_workers=gemini_workers, refresh_recipe=force)
    return [year for year in archive_years
            if (generation := transaction_source.db.get_latest_house_generation(year)) is None
            or transaction_source.db.get_unresolved_house_doc_ids(year, generation)]


def complete_house_refresh(
    transaction_source,
    archive_years: list[int],
) -> tuple[
    dict[int, tuple[str | None, list[str]]],
    tuple[int, int, object, object, int],
]:
    """Activate resolved generations and collect metrics on the owned connection."""
    outcomes: dict[int, tuple[str | None, list[str]]] = {}
    for archive_year in archive_years:
        generation_id = transaction_source.db.get_latest_house_generation(archive_year)
        if generation_id is None:
            outcomes[archive_year] = (None, [])
            continue
        unresolved = transaction_source.db.get_unresolved_house_doc_ids(
            archive_year, generation_id
        )
        if not unresolved:
            transaction_source.db.mark_house_generation_parse_complete(
                archive_year, generation_id
            )
        outcomes[archive_year] = (generation_id, unresolved)

    summary_year = archive_years[-1]
    db = transaction_source.db
    canonical_after, raw_after = house_refresh_counts(db)
    max_date, max_disclosure_date, implausible_date_count = db.conn.execute(
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
    metrics = (
        canonical_after,
        raw_after,
        max_date,
        max_disclosure_date,
        implausible_date_count,
    )
    return outcomes, metrics
