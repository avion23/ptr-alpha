"""Parallel Gemini OCR. Threaded fetch, single-writer DB."""
import json, os, re, sys, time, subprocess, threading, queue
from concurrent.futures import ThreadPoolExecutor, as_completed
import duckdb

DB_PATH = "data/congress.duckdb"
PROGRESS_PATH = "data/ocr_progress.json"
MODEL = "gemini/gemini-3.1-flash-lite"
MAX_WORKERS = 15

PROMPT = """This is a US House Periodic Transaction Report (PTR). The FIRST data row is an EXAMPLE labeled "Example: Mega Corp. Common Stock" - SKIP IT. Only extract REAL transactions below it.

Output format:
MEMBER: [full name of filer]
[asset name] | [Purchase/Sale/Exchange] | [MM/DD/YY] | [MM/DD/YY] | [amount range]

Amount ranges: A=$1K-15K, B=$15K-50K, C=$50K-100K, D=$100K-250K, E=$250K-500K, F=$500K-1M, G=$1M-5M, H=$5M-25M, I=$25M-50M, J=over $50M

One line per transaction. No markdown, no tables, no explanations."""

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
        return json.load(open(PROGRESS_PATH))
    return {"completed": [], "errors": [], "no_txs": []}

def save_progress(p):
    tmp = PROGRESS_PATH + ".tmp"
    json.dump(p, open(tmp, "w"))
    os.replace(tmp, PROGRESS_PATH)

def call_gemini(pdf_path):
    try:
        r = subprocess.run(
            ["llm", "-a", pdf_path, "-m", MODEL, PROMPT],
            capture_output=True, text=True, timeout=90
        )
        return r.stdout if r.returncode == 0 else None
    except subprocess.TimeoutExpired:
        return None

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
                    "tx_date": tx_date, "disc_date": disc_date,
                    "amount": amount.upper().strip()[0] if amount.strip() else None,
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
    con = duckdb.connect(DB_PATH)
    batch = []
    flush_count = 0
    while True:
        item = write_q.get()
        if item is SENTINEL:
            if batch:
                _flush(con, batch)
            con.execute("CHECKPOINT")
            con.close()
            return
        batch.append(item)
        if len(batch) >= 50:
            _flush(con, batch)
            batch = []
            flush_count += 1
            print(f"  [writer] flushed batch #{flush_count}")

def _flush(con, batch):
    """Insert a batch of transactions."""
    for tx in batch:
        try:
            con.execute("""
                INSERT OR IGNORE INTO transactions
                (doc_id, member, ticker, transaction_date, disclosure_date,
                 transaction_type, amount_raw, amount_midpoint, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, [tx["doc_id"], tx["member"], tx["ticker"], tx["tx_date"],
                  tx["disc_date"], tx["type"], tx["amount"],
                  AMOUNT_MIDPOINTS.get(tx["amount"]) if tx["amount"] else None])
        except Exception as e:
            print(f"  insert err {tx['doc_id']}: {e}")

def process_one(item):
    doc_id, year, pdf_path = item
    output = call_gemini(pdf_path)
    member, txs = parse_output(output)
    if not txs:
        return doc_id, year, "no_txs", 0, []
    parsed = []
    for tx in txs:
        ticker = extract_ticker(tx["asset"])
        tx_date = normalize_date(tx["tx_date"])
        disc_date = normalize_date(tx["disc_date"])
        if not tx_date: continue
        parsed.append({
            "doc_id": doc_id, "member": member, "ticker": ticker,
            "tx_date": tx_date, "disc_date": disc_date,
            "type": tx["type"], "amount": tx.get("amount"),
        })
    for p in parsed:
        write_q.put(p)
    return doc_id, year, "ok", len(parsed), parsed

def main():
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
        futures = {pool.submit(process_one, item): item for item in pending}
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
    writer_thread.join(timeout=30)
    save_progress(progress)
    print(f"\nDone: {completed} PDFs, {total_inserted} tx in {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
