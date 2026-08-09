from __future__ import annotations

import logging

import duckdb
import pandas as pd


logger = logging.getLogger(__name__)


class MetadataRepository:
    def __init__(self, conn: duckdb.DuckDBPyConnection) -> None:
        self.conn = conn

    @staticmethod
    def _with_archive_year(df: pd.DataFrame) -> pd.DataFrame:
        """Keep unknown legacy archive provenance explicitly NULL."""
        if "archive_year" in df.columns:
            return df
        normalized = df.copy()
        normalized["archive_year"] = None
        return normalized

    def upsert(self, df: pd.DataFrame) -> None:
        df = self._with_archive_year(df)
        self.conn.execute("""
            INSERT INTO metadata (
                doc_id, archive_year, first_name, last_name,
                filing_date, filing_type, fetched_at
            )
            SELECT doc_id, archive_year, first_name, last_name,
                   filing_date, filing_type, fetched_at
            FROM df
            ON CONFLICT (doc_id) DO UPDATE SET
                archive_year = EXCLUDED.archive_year,
                first_name = EXCLUDED.first_name,
                last_name = EXCLUDED.last_name,
                filing_date = EXCLUDED.filing_date,
                filing_type = EXCLUDED.filing_type,
                fetched_at = EXCLUDED.fetched_at
        """)

    def replace_archive(self, archive_year: int, df: pd.DataFrame) -> None:
        """Atomically replace one official House archive."""
        df = df.copy()
        df["archive_year"] = archive_year
        self.conn.execute("BEGIN TRANSACTION")
        try:
            self.conn.execute(
                "DELETE FROM metadata WHERE archive_year = ?",
                [archive_year],
            )
            self.upsert(df)
            self.conn.execute("COMMIT")
        except Exception:
            self.conn.execute("ROLLBACK")
            raise

    def get_by_archive(self, archive_year: int) -> pd.DataFrame:
        return self.conn.execute(
            """
            SELECT doc_id AS "DocID", archive_year AS "ArchiveYear",
                   first_name AS "First", last_name AS "Last",
                   filing_date AS "FilingDate", filing_type AS "FilingType"
            FROM metadata
            WHERE archive_year = ?
            ORDER BY doc_id
        """,
            [archive_year],
        ).fetchdf()

    def exists(self, archive_year: int) -> bool:
        count = self.conn.execute(
            "SELECT COUNT(*) FROM metadata WHERE archive_year = ?",
            [archive_year],
        ).fetchone()[0]
        return count > 0

    def clear(self, archive_year: int) -> None:
        self.conn.execute(
            "DELETE FROM metadata WHERE archive_year = ?",
            [archive_year],
        )
        logger.info("Cleared metadata for archive %s", archive_year)
