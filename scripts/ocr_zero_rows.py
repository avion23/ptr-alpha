#!/usr/bin/env python3
"""Production OCR pipeline for 425 zero-row scanned PTR PDFs.

Uses `llm -a` with Gemini 3.1 Flash Lite to extract transactions.
Gemini auto-rotates PDFs and handles checkbox detection.
"""
import json, os, re, time, subprocess, duckdb

DB_PATH = "data/congress.duckdb"
PROGRESS_PATH = "data/ocr_progress.json"
COOLDOWN = 3  # seconds between requests (Lite model allows rapid fire)
MODEL = "gemini/gemini-3.1-flash-lite"

PROMPT = """This is a US House Periodic Transaction Report (PTR). The FIRST data row is an EXAMPLE labeled "Example: Mega Corp. Common Stock" - SKIP IT. Only extract REAL transactions below it.

Output format:
MEMBER: [full name of filer]
[asset name] | [Purchase/Sale/Exchange] | [MM/DD/YY] | [MM/DD/YY] | [amount range]

Amount ranges: A=$1K-15K, B=$15K-50K, C=$50K-100K, D=$100K-250K, E=$250K-500K, F=$500K-1M, G=$1M-5M, H=$5M-25M, I=$25M-50M, J=over $50M

One line per transaction. No markdown, no tables, no explanations."""

# Amount range midpoint estimates (for amount_midpoint column)
AMOUNT_MIDPOINTS = {
    "A": 8000, "B": 32500, "C": 75000, "D": 175000, "E": 375000,
    "F": 750000, "G": 3000000, "H": 15000000, "I": 37500000, "J": 50000000
}

def get_zero_row_pdfs():
    """Get all zero-row PDFs from DB that haven't been OCR'd yet."""
    conn = duckdb.connect(DB_PATH, read_only=True)
    rows = conn.execute("""
        WITH latest AS (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY doc_id ORDER BY parsed_at DESC) as rn
            FROM pdf_parse_runs
        )
        SELECT l.doc_id, l.year
        FROM latest l
        WHERE l.rn = 1 AND l.status = 'zero_rows'
    """).fetchall()
    conn.close()
    return [(d, y, f"data/{y}/pdfs/{d}.pdf") for d, y in rows
            if os.path.exists(f"data/{y}/pdfs/{d}.pdf")]

def load_progress():
    if os.path.exists(PROGRESS_PATH):
        with open(PROGRESS_PATH) as f:
            return json.load(f)
    return {"completed": [], "errors": [], "no_txs": []}

def save_progress(progress):
    with open(PROGRESS_PATH, "w") as f:
        json.dump(progress, f, indent=2)

def call_gemini(pdf_path):
    """Call Gemini via llm -a, return raw output text."""
    try:
        result = subprocess.run(
            ["llm", "-m", MODEL, "-a", pdf_path, PROMPT],
            capture_output=True, text=True, timeout=60
        )
        return result.stdout
    except subprocess.TimeoutExpired:
        return None
    except Exception:
        return None

def parse_output(output):
    """Parse Gemini output into structured data."""
    if not output:
        return None, []
    
    member = None
    transactions = []
    
    for line in output.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        
        # Extract MEMBER
        m = re.match(r"MEMBER:\s*(.+)", line, re.IGNORECASE)
        if m:
            member = m.group(1).strip()
            continue
        
        # Skip markdown table separators and headers
        if "|" not in line or "---" in line or "ASSET" in line.upper():
            continue
        
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 4:
            continue
        
        asset = parts[0]
        tx_type = parts[1] if len(parts) > 1 else ""
        tx_date = parts[2] if len(parts) > 2 else ""
        notif_date = parts[3] if len(parts) > 3 else ""
        amount = parts[4] if len(parts) > 4 else ""
        
        # Validate
        if not re.search(r"\d{2}/\d{2}/\d{2}", tx_date):
            continue
        if tx_type not in ("Purchase", "Sale", "Exchange", "Partial Sale", "P", "S", "E"):
            # Try fuzzy match
            tx_lower = tx_type.lower()
            if "purchase" in tx_lower or tx_lower == "p":
                tx_type = "Purchase"
            elif "sale" in tx_lower or tx_lower == "s":
                tx_type = "Sale"
            elif "exchange" in tx_lower or tx_lower == "e":
                tx_type = "Exchange"
            else:
                continue
        
        # Map amount letter
        amt_letter = ""
        amount_clean = amount.strip().upper()
        if amount_clean and amount_clean[0] in AMOUNT_MIDPOINTS:
            amt_letter = amount_clean[0]
        amt_mid = AMOUNT_MIDPOINTS.get(amt_letter)
        
        transactions.append({
            "asset": asset,
            "type": tx_type,
            "date": tx_date,
            "notif_date": notif_date,
            "amount_letter": amt_letter,
            "amount_midpoint": amt_mid,
        })
    
    return member, transactions

def normalize_date(date_str):
    """Convert MM/DD/YY or MM/DD/YYYY to YYYY-MM-DD for DuckDB."""
    if not date_str:
        return None
    m = re.match(r'(\d{1,2})/(\d{1,2})/(\d{2,4})', date_str.strip())
    if not m:
        return None
    month, day, year = m.groups()
    if len(year) == 2:
        year = "20" + year if int(year) < 50 else "19" + year
    return f"{year}-{int(month):02d}-{int(day):02d}"

def insert_transactions(doc_id, year, member, transactions):
    """Insert transactions into DB. Returns count inserted."""
    if not transactions:
        return 0
    
    conn = duckdb.connect(DB_PATH)
    count = 0
    errors = []
    for tx in transactions:
        try:
            # Convert dates to YYYY-MM-DD format
            tx_date = normalize_date(tx["date"])
            notif_date = normalize_date(tx["notif_date"]) or tx_date
            
            if not tx_date:
                errors.append(f"bad date: {tx['date']}")
                continue
            
            # Normalize type
            tx_type = tx["type"]
            if tx_type in ("P",):
                tx_type = "Purchase"
            elif tx_type in ("S",):
                tx_type = "Sale"
            elif tx_type in ("E",):
                tx_type = "Exchange"
            
            conn.execute("""
                INSERT OR IGNORE INTO transactions 
                (doc_id, member, ticker, transaction_date, disclosure_date, 
                 transaction_type, amount_raw, amount_midpoint, owner_code, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, [
                doc_id, member or "Unknown", tx["asset"],
                tx_date, notif_date,
                tx_type, tx["amount_letter"] or "Unknown", tx["amount_midpoint"],
                "JT"
            ])
            count += 1
        except Exception as e:
            errors.append(f"{tx.get('asset', '?')}: {e}")
    if errors:
        print(f"  INSERT ERRORS ({len(errors)}): {errors[:3]}", flush=True)
    conn.close()
    return count

def main():
    pdfs = get_zero_row_pdfs()
    progress = load_progress()
    completed = set(progress["completed"] + progress["errors"] + progress["no_txs"])
    
    remaining = [p for p in pdfs if p[0] not in completed]
    print(f"Total zero-row PDFs: {len(pdfs)}")
    print(f"Already processed: {len(completed)}")
    print(f"Remaining: {len(remaining)}")
    
    total_inserted = 0
    for i, (doc_id, year, path) in enumerate(remaining):
        idx = i + 1
        print(f"\n[{idx}/{len(remaining)}] {doc_id} ({year})...", flush=True)
        
        time.sleep(COOLDOWN)
        
        output = call_gemini(path)
        if output is None:
            progress["errors"].append(doc_id)
            save_progress(progress)
            print(f"  ERROR: timeout or API failure", flush=True)
            continue
        
        member, transactions = parse_output(output)
        if not transactions:
            progress["no_txs"].append(doc_id)
            save_progress(progress)
            print(f"  No transactions found", flush=True)
            continue
        
        inserted = insert_transactions(doc_id, year, member, transactions)
        total_inserted += inserted
        progress["completed"].append(doc_id)
        save_progress(progress)
        print(f"  Member: {member}", flush=True)
        print(f"  Inserted: {inserted}/{len(transactions)} transactions (total: {total_inserted})", flush=True)
        for tx in transactions[:3]:
            print(f"    {tx['asset']} | {tx['type']} | {tx['date']} | {tx['amount_letter']}", flush=True)
    
    print(f"\n=== DONE ===")
    print(f"Total inserted: {total_inserted}")


def run_gemini_ocr_for_year(year: int):
    """Process all zero-row PDFs for a specific year."""
    conn = duckdb.connect(DB_PATH, read_only=True)
    rows = conn.execute("""
        WITH latest AS (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY doc_id ORDER BY parsed_at DESC) as rn
            FROM pdf_parse_runs
        )
        SELECT l.doc_id, l.year
        FROM latest l
        WHERE l.rn = 1 AND l.status = 'zero_rows' AND l.year = ?
    """, [year]).fetchall()
    conn.close()
    
    pdfs = [(d, y, f"data/{y}/pdfs/{d}.pdf") for d, y in rows
            if os.path.exists(f"data/{y}/pdfs/{d}.pdf")]
    print(f"Zero-row PDFs for {year}: {len(pdfs)}")
    
    progress = load_progress()
    completed = set(progress["completed"] + progress["errors"] + progress["no_txs"])
    remaining = [p for p in pdfs if p[0] not in completed]
    print(f"Remaining: {len(remaining)}")
    
    total_inserted = 0
    for i, (doc_id, yr, path) in enumerate(remaining):
        print(f"\n[{i+1}/{len(remaining)}] {doc_id} ({yr})...", flush=True)
        time.sleep(COOLDOWN)
        output = call_gemini(path)
        if output is None:
            progress["errors"].append(doc_id)
            save_progress(progress)
            print(f"  ERROR: timeout", flush=True)
            continue
        member, transactions = parse_output(output)
        if not transactions:
            progress["no_txs"].append(doc_id)
            save_progress(progress)
            print(f"  No transactions found", flush=True)
            continue
        inserted = insert_transactions(doc_id, yr, member, transactions)
        total_inserted += inserted
        progress["completed"].append(doc_id)
        save_progress(progress)
        print(f"  Inserted: {inserted}/{len(transactions)}", flush=True)


if __name__ == "__main__":
    main()
