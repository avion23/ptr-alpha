"""Run Docling OCR on zero-row PDFs that all text-layer parsers failed.

Architecture:
  1. Query DB (read-only) for zero-row doc_ids
  2. Run Docling subprocess on each PDF (parallel via ProcessPool)
  3. Collect results in memory
  4. Write all recovered transactions to DB in one shot at the end

This avoids DuckDB WAL contention (no concurrent writers) and processes
PDFs smallest-first for fast early wins.

Usage:
    uv run python scripts/docling_stragglers.py [--max N] [--year Y] [--workers N] [--size-limit KB]
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Prevent typer from parsing our args when analyzer modules import
if not any(a.startswith("--") for a in sys.argv[1:]):
    os.environ.setdefault("_PTR_NO_REWRITE_ARGV", "1")

import duckdb
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parent.parent


def find_zero_row_pdfs(con: duckdb.DuckDBPyConnection, year: int | None = None,
                       size_limit_kb: int | None = None) -> list[dict]:
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
        p = REPO / "data" / str(yr) / "pdfs" / f"{doc_id}.pdf"
        if p.exists():
            sz_kb = p.stat().st_size // 1024
            if size_limit_kb and sz_kb > size_limit_kb:
                continue
            out.append({"doc_id": str(doc_id), "year": int(yr), "path": str(p), "size_kb": sz_kb})
    # Sort smallest-first for fast early wins
    out.sort(key=lambda x: x["size_kb"])
    return out


def run_docling_on_pdf(task: dict) -> dict:
    """Run Docling on one PDF. Returns task + extracted markdown text (or None).

    This function runs in a subprocess worker. It does NOT touch DuckDB.
    """
    doc_id = task["doc_id"]
    pdf_path = task["path"]
    size_kb = task["size_kb"]
    # Scale timeout by size: 30s base + 1s per KB, capped at 600s
    timeout = min(600, 30 + size_kb)

    try:
        with tempfile.TemporaryDirectory(prefix=f"docling_{doc_id}_") as out_dir:
            cmd = ["uvx", "--from", "docling", "docling", "convert",
                   pdf_path, "--to", "md", "--output", out_dir]
            t0 = time.time()
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            elapsed = time.time() - t0
            if r.returncode != 0:
                return {**task, "ok": False, "elapsed": elapsed,
                        "error": f"rc={r.returncode} stderr={r.stderr[-200:]}"}
            mds = list(Path(out_dir).rglob("*.md"))
            if not mds:
                return {**task, "ok": False, "elapsed": elapsed, "error": "no markdown"}
            text = mds[0].read_text(encoding="utf-8", errors="ignore")
            return {**task, "ok": True, "elapsed": elapsed, "markdown": text}
    except subprocess.TimeoutExpired:
        return {**task, "ok": False, "elapsed": timeout, "error": "timeout"}
    except Exception as e:
        return {**task, "ok": False, "elapsed": 0, "error": str(e)}


def parse_markdown_to_txs(markdown: str) -> list[dict]:
    """Parse Docling markdown to transaction dicts using the project parser."""
    # Import here so workers don't need the full project venv
    sys.path.insert(0, str(REPO / "src"))
    from analyzer.parsing import _parse_docling_markdown, parse_pdf_table
    tables = _parse_docling_markdown(markdown)
    txs = []
    for table in tables:
        txs.extend(parse_pdf_table(table))
    return txs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=None, help="Max PDFs to process")
    ap.add_argument("--year", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4, help="Parallel Docling workers")
    ap.add_argument("--size-limit", type=int, default=400,
                    help="Skip PDFs larger than this (KB). 0=no limit")
    args = ap.parse_args()

    db_path = REPO / "data" / "congress.duckdb"

    # Step 1: find targets (read-only)
    ro = duckdb.connect(str(db_path), read_only=True)
    size_limit = args.size_limit if args.size_limit > 0 else None
    targets = find_zero_row_pdfs(ro, args.year, size_limit)
    ro.close()
    if args.max:
        targets = targets[:args.max]
    log.info(f"Found {len(targets)} zero-row PDFs to process (size limit: {size_limit}KB, workers: {args.workers})")

    # Step 2: run Docling in parallel, collect markdown results
    results: list[dict] = []
    t_start = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_docling_on_pdf, t): t for t in targets}
        for i, fut in enumerate(as_completed(futures)):
            res = fut.result()
            results.append(res)
            status = "OK" if res["ok"] else "FAIL"
            extra = res.get("error", "") or f"{len(res.get('markdown', ''))} chars"
            log.info(f"[{i+1}/{len(targets)}] {res['doc_id']} ({res['size_kb']}KB): "
                     f"{status} {res['elapsed']:.0f}s — {extra}")

    elapsed_total = time.time() - t_start
    ok_results = [r for r in results if r["ok"]]
    log.info(f"\nDocling pass done in {elapsed_total:.0f}s. {len(ok_results)}/{len(results)} succeeded.")

    # Step 3: parse markdown to transactions (in main process)
    all_tx_rows: list[dict] = []
    recovered_count = 0
    for r in ok_results:
        try:
            txs = parse_markdown_to_txs(r["markdown"])
        except Exception as e:
            log.warning(f"Parse failed for {r['doc_id']}: {e}")
            continue
        if txs:
            recovered_count += 1
            for tx in txs:
                all_tx_rows.append({
                    "doc_id": r["doc_id"],
                    "member": None,
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
    log.info(f"Parsed {len(all_tx_rows)} transactions from {recovered_count} PDFs")

    # Step 4: write to DB (single connection, one shot)
    if all_tx_rows:
        sys.path.insert(0, str(REPO / "src"))
        from analyzer.database import Database
        db = Database(db_path, read_only=False)
        df = pd.DataFrame(all_tx_rows)
        # Group by doc_id for proper upsert
        for doc_id in df["doc_id"].unique():
            sub = df[df["doc_id"] == doc_id]
            year = int(sub.iloc[0].get("year", 0)) if "year" in sub.columns else None
            # Find year from targets
            year = next((t["year"] for t in targets if t["doc_id"] == doc_id), 0)
            db.delete_transactions_for_doc(doc_id)
            db.upsert_transactions(sub)
            db.upsert_parse_run(
                doc_id=doc_id, year=year, parser_version="v3-docling",
                status="success", engines_attempted="docling",
                raw_row_count=0, transaction_count=len(sub),
            )
        # Mark the failed ones too
        failed_doc_ids = {r["doc_id"] for r in results if not r["ok"]}
        for t in targets:
            if t["doc_id"] in failed_doc_ids:
                db.upsert_parse_run(
                    doc_id=t["doc_id"], year=t["year"], parser_version="v3-docling",
                    status="zero_rows", engines_attempted="docling",
                    raw_row_count=0, transaction_count=0,
                )
        total = db.conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
        db.conn.execute("CHECKPOINT")
        db.close()
        log.info(f"Inserted {len(all_tx_rows)} transactions. DB total: {total}")
    else:
        log.info("No transactions recovered — nothing to insert.")


if __name__ == "__main__":
    main()
