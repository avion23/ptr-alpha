"""Run Docling OCR on zero-row PDFs that all text-layer parsers failed.

Usage:
    uv run python scripts/docling_stragglers.py [--max N] [--year Y]

This is the second pass after the pdfplumber-primary parse. It targets only
PDFs that returned 0 transactions from pdfplumber/camelot/pdftotext — these
are scanned-image PDFs that need OCR. Docling is run via `uvx` subprocess to
avoid polluting the project venv with ~500MB ML deps.

Inserts recovered transactions into the DB and updates pdf_parse_runs.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Only rewrite argv if no explicit flags were passed (prevent typer parsing in imports)
if not any(a.startswith("--") for a in sys.argv[1:]):
    sys.argv = ["ptr-alpha"]

import duckdb
import pandas as pd
from analyzer.parsing import extract_tables_with_docling, parse_pdf_table
from analyzer.database import Database
from analyzer.settings import Settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def find_zero_row_pdfs(con: duckdb.DuckDBPyConnection, year: int | None = None) -> list[dict]:
    """Find PDFs that returned 0 transactions in the latest parse run."""
    year_filter = f"AND year={year}" if year else ""
    rows = con.execute(f"""
        SELECT doc_id, year FROM pdf_parse_runs
        WHERE parser_version='v3' AND transaction_count=0
        {year_filter}
        ORDER BY year, doc_id
    """).fetchall()
    out = []
    for doc_id, yr in rows:
        p = Path(f"data/{yr}/pdfs/{doc_id}.pdf")
        if p.exists():
            out.append({"doc_id": str(doc_id), "year": int(yr), "path": p})
    return out


def parse_one(pdf_path: Path) -> list[dict]:
    """Run Docling on a single PDF, return parsed transactions."""
    tables = extract_tables_with_docling(pdf_path, timeout=300)
    txs = []
    for table in tables:
        txs.extend(parse_pdf_table(table))
    return txs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=None, help="Max PDFs to process")
    ap.add_argument("--year", type=int, default=None, help="Only process this year")
    args = ap.parse_args()

    settings = Settings()
    db_path = Path(settings.data.data_dir) / "congress.duckdb"

    # Read-only connection for finding stragglers
    ro = duckdb.connect(str(db_path), read_only=True)
    targets = find_zero_row_pdfs(ro, args.year)
    ro.close()

    if args.max:
        targets = targets[:args.max]

    log.info(f"Found {len(targets)} zero-row PDFs to process with Docling")

    # Read-write connection for inserting results
    db = Database(db_path, read_only=False)

    recovered = 0
    failed = 0
    total_tx = 0

    for i, t in enumerate(targets):
        doc_id = t["doc_id"]
        year = t["year"]
        try:
            txs = parse_one(t["path"])
            if txs:
                recovered += 1
                total_tx += len(txs)
                # Build DataFrame and insert via db API
                df_rows = []
                for tx in txs:
                    df_rows.append({
                        "doc_id": doc_id,
                        "member": None,  # Docling doesn't extract member
                        "ticker": tx.get("ticker"),
                        "transaction_type": tx.get("transaction_type"),
                        "transaction_date": tx.get("transaction_date"),
                        "disclosure_date": None,
                        "owner_code": tx.get("owner_code"),
                        "amount_raw": tx.get("amount_raw"),
                        "amount_midpoint": tx.get("amount_midpoint"),
                        "instrument_type": tx.get("instrument_type", "stock"),
                        "strike_price": tx.get("strike_price"),
                        "expiry_date": tx.get("expiry_date"),
                    })
                db.delete_transactions_for_doc(doc_id)
                if df_rows:
                    db.upsert_transactions(pd.DataFrame(df_rows))
                db.upsert_parse_run(
                    doc_id=doc_id, year=year, parser_version="v3-docling",
                    status="success", engines_attempted="docling",
                    raw_row_count=0, transaction_count=len(txs),
                )
                log.info(f"[{i+1}/{len(targets)}] {doc_id}: RECOVERED {len(txs)} tx")
            else:
                failed += 1
                db.upsert_parse_run(
                    doc_id=doc_id, year=year, parser_version="v3-docling",
                    status="zero_rows", engines_attempted="docling",
                    raw_row_count=0, transaction_count=0,
                )
                log.info(f"[{i+1}/{len(targets)}] {doc_id}: no tx found")
        except Exception as e:
            failed += 1
            log.warning(f"[{i+1}/{len(targets)}] {doc_id}: ERROR {e}")

    log.info(f"\nDone. Recovered {recovered}/{len(targets)} PDFs, {total_tx} total transactions, {failed} failed")
    db.close()


if __name__ == "__main__":
    main()
