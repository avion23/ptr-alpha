#!/usr/bin/env python3
"""Backfill missing tickers for existing transactions using company name matching.

Scans all no-ticker rows, applies _extract_ticker() (which now includes company
name matching), and updates the ticker column where a match is found.
"""

import duckdb
from analyzer.parsing.cells import _extract_ticker

DB_PATH = "data/congress.duckdb"


def backfill():
    conn = duckdb.connect(DB_PATH)
    before = conn.execute(
        "SELECT COUNT(*) FROM transactions WHERE ticker IS NULL OR ticker = ''"
    ).fetchone()[0]
    total = conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
    pct_before = before / total * 100 if total > 0 else 0.0
    print(f"Before: {before}/{total} no-ticker ({pct_before:.1f}%)")

    rows = conn.execute("""
        SELECT id, asset_description FROM transactions
        WHERE (ticker IS NULL OR ticker = '')
        AND asset_description IS NOT NULL
        AND asset_description != ''
    """).fetchall()
    print(f"Rows with asset_description to check: {len(rows)}")

    updated = 0
    for row_id, asset_desc in rows:
        ticker = _extract_ticker(asset_desc)
        if ticker:
            conn.execute(
                "UPDATE transactions SET ticker = ? WHERE id = ?",
                [ticker, row_id],
            )
            updated += 1

    conn.execute("CHECKPOINT")
    after = conn.execute(
        "SELECT COUNT(*) FROM transactions WHERE ticker IS NULL OR ticker = ''"
    ).fetchone()[0]
    print(f"Updated: {updated} rows")
    pct_after = after / total * 100 if total > 0 else 0.0
    print(f"After: {after}/{total} no-ticker ({pct_after:.1f}%)")
    print(f"Resolved: {before - after} new tickers")
    conn.close()


if __name__ == "__main__":
    backfill()
