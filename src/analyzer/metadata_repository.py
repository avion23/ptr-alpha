from __future__ import annotations

import logging

import duckdb
import pandas as pd


logger = logging.getLogger(__name__)


class MetadataRepository:
    def __init__(self, conn: duckdb.DuckDBPyConnection) -> None:
        self.conn = conn

    def upsert(self, df: pd.DataFrame) -> None:
        self.conn.execute("""
            INSERT INTO metadata (doc_id, first_name, last_name, filing_date, filing_type, fetched_at)
            SELECT doc_id, first_name, last_name, filing_date, filing_type, fetched_at
            FROM df
            ON CONFLICT (doc_id) DO UPDATE SET
                first_name = EXCLUDED.first_name,
                last_name = EXCLUDED.last_name,
                filing_date = EXCLUDED.filing_date,
                filing_type = EXCLUDED.filing_type,
                fetched_at = EXCLUDED.fetched_at
        """)

    def get_by_year(self, year: int) -> pd.DataFrame:
        result = self.conn.execute(
            """
            SELECT doc_id AS "DocID", first_name AS "First", last_name AS "Last",
                   filing_date AS "FilingDate", filing_type AS "FilingType"
            FROM metadata
            WHERE EXTRACT(YEAR FROM filing_date) = ?
        """,
            [year],
        ).fetchdf()
        return result

    def exists(self, year: int) -> bool:
        count = self.conn.execute(
            """
            SELECT COUNT(*) FROM metadata
            WHERE EXTRACT(YEAR FROM filing_date) = ?
        """,
            [year],
        ).fetchone()[0]
        return count > 0

    def clear(self, year: int) -> None:
        self.conn.execute(
            """
            DELETE FROM metadata
            WHERE EXTRACT(YEAR FROM filing_date) = ?
        """,
            [year],
        )
        logger.info("Cleared metadata for year %s", year)
