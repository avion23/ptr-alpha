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
        artifact_sha256: str | None = None,
        ingestion_generation: str | None = None,
        _in_transaction: bool = False,
    ) -> None:
        if not _in_transaction:
            self.conn.execute("BEGIN TRANSACTION")
        try:
            # Replace only this parser + artifact fingerprint. Prior artifact
            # generations and OCR provenance remain auditable.
            self.conn.execute(
                """
                DELETE FROM pdf_parse_runs
                WHERE doc_id = ? AND parser_version = ?
                  AND (
                    artifact_sha256 = ?
                    OR (artifact_sha256 IS NULL AND ? IS NULL)
                  )
                  AND (
                    ingestion_generation = ?
                    OR (ingestion_generation IS NULL AND ? IS NULL)
                  )
                """,
                [
                    doc_id,
                    parser_version,
                    artifact_sha256,
                    artifact_sha256,
                    ingestion_generation,
                    ingestion_generation,
                ],
            )
            self.conn.execute(
                """
                INSERT INTO pdf_parse_runs (
                    doc_id, year, parser_version, status, engines_attempted,
                    raw_row_count, transaction_count, error_message,
                    artifact_sha256, ingestion_generation
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    artifact_sha256,
                    ingestion_generation,
                ],
            )
            if not _in_transaction:
                self.conn.execute("COMMIT")
        except Exception:
            if not _in_transaction:
                self.conn.execute("ROLLBACK")
            raise

    def get_cached_doc_ids(
        self,
        *,
        year: int,
        parser_version: str,
        artifact_hashes: dict[str, str],
        ingestion_generation: str,
    ) -> set[str]:
        """Return terminal runs only when parser and artifact bytes match."""
        rows = self.conn.execute(
            """
            SELECT doc_id, artifact_sha256 FROM pdf_parse_runs
            WHERE year = ? AND parser_version = ?
              AND ingestion_generation = ?
              AND status IN ('success', 'no_txs')
            """,
            [year, parser_version, ingestion_generation],
        ).fetchall()
        return {
            str(doc_id)
            for doc_id, artifact_sha256 in rows
            if artifact_sha256
            and artifact_hashes.get(str(doc_id)) == str(artifact_sha256)
        }
