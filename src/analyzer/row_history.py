"""Small transactional helpers for official repairs (caller owns the transaction)."""

from contextlib import contextmanager


@contextmanager
def preserve_repair_history(conn):
    """Offline bulk repair: archive only changed/deleted rows, not the full DB."""
    tables = {r[0] for r in conn.execute("SHOW TABLES").fetchall()}
    bases = [t for t in ("transactions", "pdf_parse_runs") if t in tables]
    for table in bases:
        conn.execute(f"CREATE TEMP TABLE before_repair_{table} AS SELECT * FROM {table}")
    try:
        yield
        for table in bases:
            history = {"transactions": "transaction_versions", "pdf_parse_runs": "parse_run_versions"}[table]
            conn.execute(f"CREATE TABLE IF NOT EXISTS {history} AS SELECT *, CURRENT_TIMESTAMP AS retired_at FROM {table} WHERE FALSE")
            conn.execute(f"CREATE TEMP TABLE changed_repair_{table} AS SELECT * FROM before_repair_{table} EXCEPT SELECT * FROM {table}")
            conn.execute(f"INSERT INTO {history} SELECT *, CURRENT_TIMESTAMP FROM changed_repair_{table}")
            columns = {r[1] for r in conn.execute(f"PRAGMA table_info('{table}')").fetchall()}
            if table == "transactions" and "created_at" in columns:
                conn.execute("UPDATE transactions SET created_at = CURRENT_TIMESTAMP WHERE id IN (SELECT id FROM changed_repair_transactions)")
            elif table == "pdf_parse_runs":
                conn.execute("""UPDATE pdf_parse_runs SET parsed_at = CURRENT_TIMESTAMP
                    WHERE EXISTS (SELECT 1 FROM changed_repair_pdf_parse_runs old
                      WHERE old.doc_id = pdf_parse_runs.doc_id
                        AND old.parser_version = pdf_parse_runs.parser_version
                        AND old.ingestion_generation IS NOT DISTINCT FROM pdf_parse_runs.ingestion_generation
                        AND old.artifact_sha256 IS NOT DISTINCT FROM pdf_parse_runs.artifact_sha256)""")
        if "transactions" in bases:
            accept_corrections(conn)
    finally:
        for table in bases:
            conn.execute(f"DROP TABLE IF EXISTS before_repair_{table}")
            conn.execute(f"DROP TABLE IF EXISTS changed_repair_{table}")


def archive_rows(conn, table, predicate="TRUE", parameters=()):
    history = {"transactions": "transaction_versions", "pdf_parse_runs": "parse_run_versions",
               "house_archive_generations": "house_generation_versions"}[table]
    conn.execute(f"CREATE TABLE IF NOT EXISTS {history} AS SELECT *, CURRENT_TIMESTAMP AS retired_at FROM {table} WHERE FALSE")
    conn.execute(f"INSERT INTO {history} SELECT *, CURRENT_TIMESTAMP FROM {table} WHERE {predicate}", parameters)


def accept_corrections(conn):
    """Re-bind previously successful source runs after an official row repair.

    Never invent success for an unaccepted source. Archive the old acceptance
    before advancing it, so yesterday still binds yesterday's rowset.
    """
    tables = {r[0] for r in conn.execute("SHOW TABLES").fetchall()}
    if "pdf_parse_runs" not in tables:
        return
    from analyzer.database import _HOUSE_PARSE_SOURCE

    predicate = f"""status = 'success' AND EXISTS (
        SELECT 1 FROM transaction_versions t WHERE t.doc_id = pdf_parse_runs.doc_id
          AND t.ingestion_generation = pdf_parse_runs.ingestion_generation
          AND t.artifact_sha256 = pdf_parse_runs.artifact_sha256
          AND COALESCE(t.source, (SELECT source FROM transactions current_row WHERE current_row.id = t.id))
              = {_HOUSE_PARSE_SOURCE.replace('parse_run.', 'pdf_parse_runs.')}
          AND t.retired_at = CURRENT_TIMESTAMP)"""
    archive_rows(conn, "pdf_parse_runs", predicate)
    columns = {r[1] for r in conn.execute("PRAGMA table_info('pdf_parse_runs')").fetchall()}
    watermark = ", rowset_created_at = CURRENT_TIMESTAMP" if "rowset_created_at" in columns else ""
    source = _HOUSE_PARSE_SOURCE.replace('parse_run.', 'pdf_parse_runs.')
    conn.execute(f"""UPDATE pdf_parse_runs SET parsed_at = CURRENT_TIMESTAMP{watermark},
        transaction_count = (SELECT COUNT(*) FROM transactions t
          WHERE t.doc_id = pdf_parse_runs.doc_id
            AND t.ingestion_generation = pdf_parse_runs.ingestion_generation
            AND t.artifact_sha256 = pdf_parse_runs.artifact_sha256 AND t.source = {source})
        WHERE {predicate}""")
