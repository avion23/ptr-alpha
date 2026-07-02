"""Parallel Gemini OCR. Threaded fetch, single-writer DB."""
import argparse, json, os, re, time, threading, queue
from concurrent.futures import ThreadPoolExecutor, as_completed
import duckdb

from scripts.gemini_ocr_common import MODEL, PROMPT, call_gemini, validate_transactions
from scripts.ocr_zero_rows import get_filing_date, get_metadata_member, insert_transactions, record_parse_run

DB_PATH = "data/congress.duckdb"
PROGRESS_PATH = "data/ocr_progress.json"
MAX_WORKERS = 15

AMOUNT_MIDPOINTS = {"A":8000,"B":32500,"C":75000,"D":175000,"E":375000,
                    "F":750000,"G":3000000,"H":15000000,"I":37500000,"J":50000000}

def get_zero_row_pdfs():
    conn = duckdb.connect(DB_PATH, read_only=True)
    rows = conn.execute("""
        WITH latest AS (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY doc_id ORDER BY parsed_at DESC) as rn
            FROM pdf_parse_runs
        )
        SELECT l.doc_id, l.year FROM latest l
        WHERE l.rn = 1 AND l.status = 'zero_rows'
    """).fetchall()
    conn.close()
    return [(d, y, f"data/{y}/pdfs/{d}.pdf") for d, y in rows if os.path.exists(f"data/{y}/pdfs/{d}.pdf")]

def load_progress():
    if os.path.exists(PROGRESS_PATH):
        with open(PROGRESS_PATH) as f:
            return json.load(f)
    return {"completed": [], "errors": [], "no_txs": []}

def save_progress(p):
    tmp = PROGRESS_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(p, f)
    os.replace(tmp, PROGRESS_PATH)

def parse_output(output):
    if not output:
        return None, []
    lines = [l.strip() for l in output.split("\n") if l.strip()]
    member, transactions = None, []
    for line in lines:
        if line.upper().startswith("MEMBER:"):
            member = line.split(":", 1)[1].strip()
            continue
        if "EXAMPLE" in line.upper() or "Mega Corp" in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 5 and member:
            asset, ttype, tx_date, disc_date, amount = parts[:5]
            ttype_clean = None
            tl = ttype.lower()
            if "purchase" in tl or tl == "p": ttype_clean = "Purchase"
            elif "sale" in tl or tl == "s": ttype_clean = "Sale"
            elif "exchange" in tl or tl == "e": ttype_clean = "Exchange"
            if ttype_clean:
                transactions.append({
                    "asset": asset, "type": ttype_clean,
                    "date": tx_date, "notif_date": disc_date,
                    "amount_letter": amount.upper().strip()[0] if amount.strip() else None,
                    "amount_midpoint": AMOUNT_MIDPOINTS.get(amount.upper().strip()[0]) if amount.strip() else None,
                })
    return member, transactions

def normalize_date(s):
    from datetime import datetime
    if not s: return None
    s = s.strip()
    for fmt in ("%m/%d/%y", "%m/%d/%Y", "%m-%d-%y", "%m-%d-%Y"):
        try: return datetime.strptime(s, fmt).date().isoformat()
        except ValueError: continue
    return None

def extract_ticker(asset):
    m = re.search(r'\(([A-Z]{1,5}(\.[AB])?)\)', asset.upper())
    return m.group(1) if m else None

# ---- DB writer thread ----
write_q = queue.Queue()
SENTINEL = object()
def db_writer():
    """Single thread that owns the DuckDB connection."""
    batch = []
    flush_count = 0
    while True:
        item = write_q.get()
        if item is SENTINEL:
            if batch:
                _flush(batch)
            return
        batch.append(item)
        if len(batch) >= 50:
            _flush(batch)
            batch = []
            flush_count += 1
            print(f"  [writer] flushed batch #{flush_count}")

def _flush(batch):
    """Insert validated documents with delete-then-insert semantics."""
    for item in batch:
        try:
            if item.get("status") in ("error", "rejected"):
                con = duckdb.connect(DB_PATH)
                record_parse_run(
                    con, item["doc_id"], item["year"], item["status"],
                    item["raw_count"], 0, item.get("error", ""),
                    parser_version="v4-gemini-parallel",
                )
                con.close()
                continue
            count = insert_transactions(
                item["doc_id"], item["year"], item["member"], item["transactions"],
                db_path=DB_PATH, parser_version="v4-gemini-parallel", raw_count=item["raw_count"],
            )
            item["inserted"] = count
        except Exception as e:
            print(f"  insert err {item['doc_id']}: {e}")

def process_one(item, refresh=False):
    doc_id, year, pdf_path = item
    output, error = call_gemini(pdf_path, doc_id=doc_id, refresh=refresh, timeout=90)
    if output is None or error:
        write_q.put({"doc_id": doc_id, "year": year, "status": "error", "raw_count": 0, "error": str(error)[:1000]})
        return doc_id, year, "error", 0, error
    member, txs = parse_output(output)
    if not txs:
        write_q.put({"doc_id": doc_id, "year": year, "member": member, "transactions": [], "raw_count": 0})
        return doc_id, year, "no_txs", 0, []
    raw_count = len(txs)
    con = duckdb.connect(DB_PATH, read_only=True)
    filing_date = get_filing_date(con, doc_id)
    expected_member = get_metadata_member(con, doc_id)
    con.close()
    txs, rejections = validate_transactions(doc_id, member, txs, filing_date, expected_member)
    print(f"  {doc_id} validation rejections: {rejections}")
    if rejections.get("row_count_exceeds_cap"):
        write_q.put({"doc_id": doc_id, "year": year, "status": "rejected", "raw_count": raw_count, "error": "row_count_exceeds_cap"})
        return doc_id, year, "rejected", 0, []
    member = txs[0].get("member", member) if txs else member
    write_q.put({"doc_id": doc_id, "year": year, "member": member, "transactions": txs, "raw_count": raw_count})
    return doc_id, year, "ok" if txs else "no_txs", len(txs), txs

def main():
    parser = argparse.ArgumentParser(description="Parallel Gemini OCR for zero-row PDFs")
    parser.add_argument("--refresh", action="store_true", help="Ignore cached Gemini responses")
    args = parser.parse_args()

    progress = load_progress()
    done = set(progress["completed"]) | set(progress["no_txs"]) | set(progress["errors"])
    pending = [(d, y, p) for d, y, p in get_zero_row_pdfs() if d not in done]
    print(f"Total pending: {len(pending)} (parallelism: {MAX_WORKERS})")
    if not pending:
        return
    
    # Start DB writer thread
    writer_thread = threading.Thread(target=db_writer, daemon=True)
    writer_thread.start()
    
    t0 = time.time()
    completed = 0
    total_inserted = 0
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(process_one, item, args.refresh): item for item in pending}
        for fut in as_completed(futures):
            doc_id, year, status, n, _ = fut.result()
            completed += 1
            total_inserted += n
            if status == "ok":
                progress["completed"].append(doc_id)
            elif status == "no_txs":
                progress["no_txs"].append(doc_id)
            else:
                progress["errors"].append(doc_id)
            
            if completed % 10 == 0 or completed == len(pending):
                elapsed = time.time() - t0
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (len(pending) - completed) / rate if rate > 0 else 0
                print(f"  [{completed}/{len(pending)}] {doc_id} ({year}) {status} +{n} | "
                      f"total {total_inserted} | {elapsed:.0f}s, ETA {eta:.0f}s")
                save_progress(progress)
    
    # Signal writer to flush and exit
    write_q.put(SENTINEL)
    writer_thread.join()
    save_progress(progress)
    print(f"\nDone: {completed} PDFs, {total_inserted} tx in {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
