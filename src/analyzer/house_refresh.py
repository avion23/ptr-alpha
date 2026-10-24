"""House refresh orchestration and bounded DuckDB ownership."""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

from analyzer.persistence.database import Database, open_database as _open_database
from analyzer.download import HouseFetchSummary, HouseTransactionSource
from analyzer.exceptions import DataResult
from analyzer.pipeline import run_parse_pipeline

logger = logging.getLogger(__name__)


class _DatabaseContextReleaseError(RuntimeError):
    pass


class _HouseAcquisitionDatabase:
    """Open a database only for each House acquisition query or promotion.

    PDF downloads can take hours. Keep no DuckDB connection open while they
    run; the only write operation in acquisition is archive promotion.
    """

    _WRITE_METHODS = frozenset({"promote_house_archive"})

    def __init__(self, data_dir: str | Path):
        self.data_dir = data_dir

    def __getattr__(self, name):
        def call(*args, **kwargs):
            db = _open_database(
                self.data_dir, read_only=name not in self._WRITE_METHODS
            )
            try:
                result = getattr(db, name)(*args, **kwargs)
            except BaseException:
                try:
                    db.close()
                except Exception as exc:
                    raise _DatabaseContextReleaseError(
                        f"Failed to close House acquisition database after {name}"
                    ) from exc
                raise
            else:
                try:
                    db.close()
                except Exception as exc:
                    raise _DatabaseContextReleaseError(
                        f"Failed to close House acquisition database after {name}"
                    ) from exc
            return result

        return call


def _set_shared_database(transaction_source, price_source, db) -> None:
    for owner in (transaction_source, price_source):
        if owner is not None:
            owner.db = db


def _release_shared_database(transaction_source, price_source) -> None:
    """Close the shared real DB handle and clear both source references."""
    db = getattr(transaction_source, "db", None)
    if isinstance(db, Database):
        db.close()
    _set_shared_database(transaction_source, price_source, None)


def _prepare_house_acquisition_database(
    transaction_source, price_source, data_dir: str | Path
) -> bool:
    """Release the initial writer and install lock-free acquisition access."""
    if not isinstance(transaction_source, HouseTransactionSource):
        return False
    db = getattr(transaction_source, "db", None)
    if not isinstance(db, Database):
        return False
    db.close()
    _set_shared_database(
        transaction_source, price_source, _HouseAcquisitionDatabase(data_dir)
    )
    return True


def _reopen_shared_database(
    transaction_source, price_source, data_dir: str | Path
) -> None:
    db = getattr(transaction_source, "db", None)
    if isinstance(db, _HouseAcquisitionDatabase):
        return
    db.close()
    _set_shared_database(
        transaction_source, price_source, _open_database(data_dir, False)
    )


def house_refresh_counts(db) -> tuple[int, int]:
    """Return canonical and raw transaction counts for refresh reporting."""
    canonical = db.conn.execute(
        "SELECT COUNT(*) FROM canonical_transactions"
    ).fetchone()[0]
    raw = db.conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
    return canonical, raw


def acquire_house_archives(
    transaction_source,
    price_source,
    archive_years: list[int],
    data_dir: str | Path,
    *,
    all_years: bool,
    refresh_metadata: bool,
) -> tuple[DataResult[list[HouseFetchSummary]], bool]:
    """Acquire every requested archive before parsing, retrying boundedly.

    For all-years refreshes, a twice-failed archive is quarantined from this
    refresh while later years continue. A single-year refresh remains fail-fast.
    The boolean reports whether this helper owns real DB handle transitions.
    """
    summaries: list[HouseFetchSummary] = []
    failed = False
    first_error: Exception | None = None
    database_lifecycle = False

    try:
        database_lifecycle = _prepare_house_acquisition_database(
            transaction_source, price_source, data_dir
        )
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
                if isinstance(year_exc, _DatabaseContextReleaseError):
                    raise
                if fail_fast:
                    raise
                logger.warning(
                    "House archive %d fetch failed, reopening database: %s",
                    archive_year,
                    year_exc,
                )
                _reopen_shared_database(transaction_source, price_source, data_dir)
                try:
                    summary = transaction_source.fetch_and_cache_pdfs(
                        archive_year, refresh_metadata=should_refresh_metadata
                    )
                except Exception as retry_exc:
                    if isinstance(retry_exc, _DatabaseContextReleaseError):
                        raise
                    logger.warning(
                        "House archive %d failed twice, skipping: %s",
                        archive_year,
                        retry_exc,
                    )
                    failed = True
                    if first_error is None:
                        first_error = retry_exc
                    _reopen_shared_database(
                        transaction_source, price_source, data_dir
                    )
                    continue
            summaries.append(summary)

        if failed:
            return (
                DataResult(
                    success=False,
                    data=summaries,
                    error=first_error,
                ),
                database_lifecycle,
            )
        return DataResult(success=True, data=summaries), database_lifecycle
    except Exception as exc:
        logger.warning("House PDF fetch failed: %s", exc)
        return DataResult(success=False, error=exc), database_lifecycle


def parse_house_archives(
    transaction_source,
    price_source,
    archive_years: list[int],
    data_dir: str | Path,
    *,
    database_lifecycle: bool,
    force: bool,
) -> list[int]:
    """Parse each acquired archive with a short-lived writer per year."""
    failed_years = []
    for archive_year in archive_years:
        if database_lifecycle:
            parse_db = _open_database(data_dir, False)
            _set_shared_database(transaction_source, price_source, parse_db)
        try:
            try:
                # Activation follows optional OCR, never the parse step.
                parse_result = run_parse_pipeline(
                    transaction_source,
                    archive_year,
                    force=force,
                    activate_generation=False,
                )
                if not parse_result.success:
                    failed_years.append(archive_year)
            except Exception as exc:
                failed_years.append(archive_year)
                logger.warning("PDF parse failed for %d: %s", archive_year, exc)
        finally:
            if database_lifecycle:
                _release_shared_database(transaction_source, price_source)
    return failed_years


def complete_house_refresh(
    transaction_source,
    price_source,
    archive_years: list[int],
    data_dir: str | Path,
    *,
    database_lifecycle: bool,
) -> tuple[
    dict[int, tuple[str | None, list[str]]],
    tuple[int, int, object, object, int],
]:
    """Activate only resolved generations and collect final report metrics.

    For each year, a missing generation has a ``None`` ID; an empty unresolved
    list means the generation was activated.
    """
    if database_lifecycle:
        _set_shared_database(
            transaction_source, price_source, _open_database(data_dir, False)
        )
    try:
        outcomes: dict[int, tuple[str | None, list[str]]] = {}
        for archive_year in archive_years:
            generation_id = transaction_source.db.get_latest_house_generation(
                archive_year
            )
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
    finally:
        if database_lifecycle:
            _release_shared_database(transaction_source, price_source)
