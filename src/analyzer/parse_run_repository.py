from __future__ import annotations

import duckdb


_TERMINAL_STATUSES = ("success", "no_txs")
_TERMINAL_STATUS_PREDICATE = (
    "status IN (" + ", ".join(f"'{status}'" for status in _TERMINAL_STATUSES) + ")"
)
_IDENTITY_PREDICATE = (
    "doc_id = ? AND parser_version = ? "
    "AND (artifact_sha256 = ? "
    "OR (artifact_sha256 IS NULL AND ? IS NULL)) "
    "AND (ingestion_generation = ? "
    "OR (ingestion_generation IS NULL AND ? IS NULL))"
)


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
        identity_params = [
            doc_id,
            parser_version,
            artifact_sha256,
            artifact_sha256,
            ingestion_generation,
            ingestion_generation,
        ]
        if not _in_transaction:
            self.conn.execute("BEGIN TRANSACTION")
        try:
            # A failed reparse must not erase the last terminal provenance for
            # the same immutable artifact and generation.  The transaction
            # rows are deliberately left untouched when a parse attempt
            # fails, so replacing a prior success with ``error`` would leave
            # persisted rows with no terminal run to bind them to.  A later
            # terminal attempt is still allowed to replace a prior failure.
            if status not in _TERMINAL_STATUSES:
                terminal = self.conn.execute(
                    f"""
                    SELECT 1 FROM pdf_parse_runs
                    WHERE {_IDENTITY_PREDICATE}
                      AND {_TERMINAL_STATUS_PREDICATE}
                    LIMIT 1
                    """,
                    identity_params,
                ).fetchone()
                if terminal:
                    # Keep the historical one-row-per-identity shape even if
                    # a legacy database already contains duplicate terminal
                    # rows.  The newest terminal row is the one retained.
                    self.conn.execute(
                        f"""
                        DELETE FROM pdf_parse_runs
                        USING (
                            SELECT rowid
                            FROM pdf_parse_runs
                            WHERE {_IDENTITY_PREDICATE}
                              AND {_TERMINAL_STATUS_PREDICATE}
                            QUALIFY row_number() OVER (
                                ORDER BY parsed_at DESC NULLS LAST, rowid DESC
                            ) > 1
                        ) AS duplicates
                        WHERE pdf_parse_runs.rowid = duplicates.rowid
                        """,
                        identity_params,
                    )
                    if not _in_transaction:
                        self.conn.execute("COMMIT")
                    return

            # Replace only this parser + artifact fingerprint. Prior artifact
            # generations and OCR provenance remain auditable.
            self.conn.execute(
                f"""
                DELETE FROM pdf_parse_runs
                WHERE {_IDENTITY_PREDICATE}
                """,
                identity_params,
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
            f"""
            SELECT doc_id, artifact_sha256 FROM pdf_parse_runs
            WHERE year = ? AND parser_version = ?
              AND ingestion_generation = ?
              AND {_TERMINAL_STATUS_PREDICATE}
            """,
            [year, parser_version, ingestion_generation],
        ).fetchall()
        return {
            str(doc_id)
            for doc_id, artifact_sha256 in rows
            if artifact_sha256
            and artifact_hashes.get(str(doc_id)) == str(artifact_sha256)
        }
