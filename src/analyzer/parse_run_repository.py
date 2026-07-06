from __future__ import annotations

import duckdb


class ParseRunRepository:
    def __init__(self, conn: duckdb.DuckDBPyConnection) -> None:
        self.conn = conn

    def upsert(
        self,
        *,
        doc_id: str,
        year: int,
        parser_version: str,
        status: str,
        engines_attempted: str,
        raw_row_count: int,
        transaction_count: int,
        error_message: str | None = None,
    ) -> None:
        self.conn.execute("DELETE FROM pdf_parse_runs WHERE doc_id = ?", [doc_id])
        self.conn.execute("""
            INSERT INTO pdf_parse_runs (
                doc_id, year, parser_version, status, engines_attempted,
                raw_row_count, transaction_count, error_message
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, [doc_id, year, parser_version, status, engines_attempted,
              raw_row_count, transaction_count, error_message])
