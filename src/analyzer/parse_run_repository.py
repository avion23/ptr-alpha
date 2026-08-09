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
        _in_transaction: bool = False,
    ) -> None:
        if not _in_transaction:
            self.conn.execute("BEGIN TRANSACTION")
        try:
            # Replace only this parser fingerprint. Other deterministic/OCR
            # provenance for the same document remains auditable.
            self.conn.execute(
                "DELETE FROM pdf_parse_runs "
                "WHERE doc_id = ? AND parser_version = ?",
                [doc_id, parser_version],
            )
            self.conn.execute(
                """
                INSERT INTO pdf_parse_runs (
                    doc_id, year, parser_version, status, engines_attempted,
                    raw_row_count, transaction_count, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                [
                    doc_id,
                    year,
                    parser_version,
                    status,
                    engines_attempted,
                    raw_row_count,
                    transaction_count,
                    error_message,
                ],
            )
            if not _in_transaction:
                self.conn.execute("COMMIT")
        except Exception:
            if not _in_transaction:
                self.conn.execute("ROLLBACK")
            raise

    def get_cached_doc_ids(self, *, year: int, parser_version: str) -> set[str]:
        """doc_ids with a terminal (non-error) parse_run for this year + parser_version.

        These PDFs need not be re-parsed: their result is deterministic for the
        given parser_version. 'error' runs are excluded so failures get retried.
        """
        rows = self.conn.execute(
            "SELECT doc_id FROM pdf_parse_runs "
            "WHERE year = ? AND parser_version = ? "
            "AND status IN ('success', 'zero_rows', 'no_txs')",
            [year, parser_version],
        ).fetchall()
        return {str(r[0]) for r in rows}
