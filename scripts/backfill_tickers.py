"""Backfill uniquely resolved company tickers; dry-run unless --apply is set."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from analyzer.asset_ticker_map import resolve_asset_ticker


def backfill_tickers(
    db_path: str | Path = "data/congress.duckdb", *, apply: bool = False
) -> tuple[int, int]:
    """Return (resolved, unresolved) counts; update only ticker and its origin."""
    connection = duckdb.connect(str(db_path), read_only=not apply)
    try:
        rows = connection.execute(
            """
            SELECT id, asset_description
            FROM transactions
            WHERE ticker IS NULL
              AND asset_description IS NOT NULL
              AND trim(asset_description) <> ''
            """
        ).fetchall()
        resolved = [
            (row_id, ticker)
            for row_id, description in rows
            if (ticker := resolve_asset_ticker(description)) is not None
        ]
        unresolved = len(rows) - len(resolved)

        if apply and resolved:
            connection.execute("BEGIN TRANSACTION")
            try:
                connection.executemany(
                    """
                    UPDATE transactions
                    SET ticker = ?, ticker_origin = 'resolved_asset_name'
                    WHERE id = ? AND ticker IS NULL
                    """,
                    [(ticker, row_id) for row_id, ticker in resolved],
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return len(resolved), unresolved
    finally:
        connection.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", default="data/congress.duckdb")
    parser.add_argument(
        "--apply", action="store_true", help="write resolved tickers (default: dry-run)"
    )
    args = parser.parse_args()
    resolved, unresolved = backfill_tickers(args.db_path, apply=args.apply)
    print("Mode: apply" if args.apply else "Mode: dry-run")
    print(f"Resolved: {resolved}")
    print(f"Unresolved: {unresolved}")


if __name__ == "__main__":
    main()
