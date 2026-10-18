"""One-time cleanup proposal for review finding 2 (2026-10-18, Astra).

The canonical_transactions view now binds admission to the accepted parse
run's rowset (see src/analyzer/database.py, rowset admission binding). Rows
inserted by unaccepted or superseded House/OCR runs therefore stop being
served canonically, but they remain in the ``transactions`` table.

This script REPORTS the residual rowsets that the new admission rule
excludes -- i.e. live House/OCR rows whose own (source, artifact,
ingestion_generation) identity has no accepted parse run even though their
filing artifact is otherwise accepted. It never guesses: the predicate is
exactly the view's rowset acceptance rule.

Report mode (default) opens the database READ-ONLY and changes nothing.
With ``--apply`` it archives the offending rows into
``transaction_versions`` first (no evidence is destroyed) and then removes
them from ``transactions`` inside one logged transaction.

DO NOT run against data/congress.duckdb while a refresh holds the write
lock; run it in a maintenance window. Proposed, not executed, as part of
finding 2's remediation: the view-layer fix alone already restores correct
canonical admission, so this is optional table hygiene plus a measurement
of the live-contamination UNKNOWN.

Usage:
    python scripts/purge_unaccepted_house_rowsets.py [DB_PATH] [--apply]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import duckdb  # noqa: E402

from analyzer.database import _HOUSE_PARSE_SOURCE  # noqa: E402

logger = logging.getLogger("purge_unaccepted_house_rowsets")

# The view's rowset acceptance rule, correlated to one row identity.
ACCEPTED_RUN_EXISTS = f"""
    EXISTS (
        SELECT 1
        FROM pdf_parse_runs parse_run
        WHERE parse_run.doc_id = t.doc_id
          AND parse_run.ingestion_generation = t.ingestion_generation
          AND parse_run.artifact_sha256 IS NOT DISTINCT FROM t.artifact_sha256
          AND parse_run.status = 'success'
          AND t.source = {_HOUSE_PARSE_SOURCE}
          AND COALESCE(parse_run.transaction_count, 0) = (
                SELECT COUNT(*)
                FROM transactions tx_count
                WHERE tx_count.doc_id = t.doc_id
                  AND tx_count.artifact_sha256 IS NOT DISTINCT FROM t.artifact_sha256
                  AND tx_count.ingestion_generation = t.ingestion_generation
                  AND tx_count.source = t.source
          )
          AND NOT EXISTS (
                SELECT 1
                FROM transactions tx_invalid
                WHERE tx_invalid.doc_id = t.doc_id
                  AND tx_invalid.artifact_sha256 IS NOT DISTINCT FROM t.artifact_sha256
                  AND tx_invalid.ingestion_generation = t.ingestion_generation
                  AND tx_invalid.source = t.source
                  AND (
                      tx_invalid.transaction_date IS NULL
                      OR tx_invalid.disclosure_date IS NULL
                      OR tx_invalid.transaction_date > tx_invalid.disclosure_date
                      OR (
                          tx_invalid.notification_date IS NOT NULL
                          AND tx_invalid.notification_date < tx_invalid.transaction_date
                      )
                      OR tx_invalid.chamber IS NULL
                      OR TRIM(tx_invalid.chamber) = ''
                      OR tx_invalid.source_record_id IS NULL
                      OR TRIM(tx_invalid.source_record_id) = ''
                      OR tx_invalid.source_row_id IS NULL
                      OR TRIM(tx_invalid.source_row_id) = ''
                      OR tx_invalid.official_filing_date IS NULL
                  )
          )
    )
"""

# Only rows whose filing artifact exists and is itself accepted (the leak
# required an accepted artifact); everything else is legacy fallback data.
UNACCEPTED_ROWSETS = f"""
    SELECT t.id, t.doc_id, t.source, t.ingestion_generation, t.source_row_id
    FROM transactions t
    WHERE t.source IN ('house_pdf', 'gemini_ocr')
      AND t.ingestion_generation IS NOT NULL
      AND EXISTS (
          SELECT 1
          FROM house_pdf_artifacts artifact
          JOIN house_archive_generations generation
            ON generation.archive_year = artifact.archive_year
           AND generation.generation_id = artifact.generation_id
          WHERE artifact.doc_id = t.doc_id
            AND artifact.generation_id = t.ingestion_generation
            AND artifact.artifact_sha256 IS NOT DISTINCT FROM t.artifact_sha256
      )
      AND NOT {ACCEPTED_RUN_EXISTS}
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("db_path", nargs="?", default="data/congress.duckdb")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Archive the reported rows to transaction_versions and delete "
        "them from transactions. Default is report-only (read-only).",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    db_path = Path(args.db_path)
    conn = duckdb.connect(str(db_path), read_only=not args.apply)
    try:
        rows = conn.execute(UNACCEPTED_ROWSETS).fetchall()
        by_source: dict[str, int] = {}
        by_doc: dict[str, int] = {}
        for row_id, doc_id, source, _generation, _source_row_id in rows:
            by_source[source] = by_source.get(source, 0) + 1
            by_doc[doc_id] = by_doc.get(doc_id, 0) + 1
        logger.info(
            "Unaccepted-rowset rows excluded from canonical by the view: %d",
            len(rows),
        )
        logger.info("By source: %s", by_source)
        logger.info("By doc (%d docs): %s", len(by_doc), by_doc)
        if not rows:
            logger.info("Nothing to clean up.")
            return 0
        if not args.apply:
            logger.info(
                "Report-only mode; rerun with --apply to archive these rows "
                "into transaction_versions and remove them from transactions."
            )
            return 0
        conn.execute("BEGIN TRANSACTION")
        try:
            archived = conn.execute(
                f"INSERT INTO transaction_versions "
                f"SELECT t.*, CURRENT_TIMESTAMP FROM ({UNACCEPTED_ROWSETS}) reported "
                f"JOIN transactions t ON t.id = reported.id RETURNING id"  # nosec B608
            ).fetchall()
            deleted = conn.execute(
                f"DELETE FROM transactions WHERE id IN "
                f"(SELECT id FROM ({UNACCEPTED_ROWSETS}))"  # nosec B608
            ).fetchone()
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        logger.info(
            "Archived %d rows to transaction_versions and deleted %d rows "
            "from transactions.",
            len(archived),
            deleted[0] if deleted else 0,
        )
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
