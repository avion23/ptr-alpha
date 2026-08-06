"""Purge transaction phantom duplicates caused by NULL-ticker index bypass.

DuckDB unique indexes treat NULL values as distinct, so the historical
transactions unique index did not stop duplicate NULL-ticker rows. An audit
found 24,534 phantom rows; this script keeps the lowest id for each normalized
transaction key and optionally deletes the rest.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb


VICTIMS_CTE = """
    WITH ranked AS (
        SELECT
            id,
            ticker IS NULL AS ticker_is_null,
            ROW_NUMBER() OVER (
                PARTITION BY
                    doc_id,
                    COALESCE(ticker, ''),
                    transaction_date,
                    member,
                    transaction_type,
                    COALESCE(amount_raw, ''),
                    COALESCE(owner_code, ''),
                    COALESCE(asset_description, '')
                ORDER BY id
            ) AS row_num
        FROM transactions
    ), victims AS (
        SELECT id, ticker_is_null
        FROM ranked
        WHERE row_num > 1
    )
"""


def count_phantom_rows(conn: duckdb.DuckDBPyConnection) -> dict[bool, int]:
    rows = conn.execute(
        VICTIMS_CTE
        + """
        SELECT ticker_is_null, COUNT(*)
        FROM victims
        GROUP BY ticker_is_null
        ORDER BY ticker_is_null DESC
        """
    ).fetchall()
    return {bool(ticker_is_null): count for ticker_is_null, count in rows}


def purge_phantom_rows(conn: duckdb.DuckDBPyConnection) -> dict[str, int]:
    before_row = conn.execute("SELECT COUNT(*) FROM transactions").fetchone()
    if before_row is None:
        raise RuntimeError("COUNT(*) query returned no row before purge")
    before = before_row[0]
    deleted = sum(count_phantom_rows(conn).values())
    conn.execute(
        VICTIMS_CTE
        + """
        DELETE FROM transactions
        WHERE id IN (SELECT id FROM victims)
        """
    )
    after_row = conn.execute("SELECT COUNT(*) FROM transactions").fetchone()
    if after_row is None:
        raise RuntimeError("COUNT(*) query returned no row after purge")
    after = after_row[0]
    conn.execute("CHECKPOINT")
    return {"before": before, "deleted": deleted, "after": after}


def print_dry_run(counts: dict[bool, int]) -> None:
    print("DRY-RUN: phantom rows that would be deleted")
    print(f"  ticker_is_null=true: {counts.get(True, 0)}")
    print(f"  ticker_is_null=false: {counts.get(False, 0)}")
    print(f"  total: {sum(counts.values())}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Purge phantom transaction duplicates")
    parser.add_argument("db_path", nargs="?", default="data/congress.duckdb")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    # Dry-run must not take DuckDB's exclusive write lock on a live DB.
    conn = duckdb.connect(str(Path(args.db_path)), read_only=not args.execute)
    try:
        counts = count_phantom_rows(conn)
        if not args.execute:
            print_dry_run(counts)
            return

        stats = purge_phantom_rows(conn)
        print(f"before: {stats['before']}")
        print(f"deleted: {stats['deleted']}")
        print(f"after: {stats['after']}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
