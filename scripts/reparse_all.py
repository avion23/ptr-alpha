"""Fast text-layer-only reparse of all cached PDFs.

Skips Docling and tesseract OCR; only uses pdfplumber/camelot/pdftotext.
Used to recover from DB corruption without re-running slow OCR.
"""
from __future__ import annotations
import os, sys, time
from pathlib import Path

# Must set env BEFORE importing analyzer (it may import OCR libs lazily)
os.environ["PTR_SKIP_DOCLING"] = "1"

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from analyzer.database import Database

# Import the text-layer engine functions
from analyzer.datasources import (
    _try_pdfplumber, _try_camelot_lattice, _try_camelot_stream, _try_pdftotext,
    _build_member_lookup, _filter_existing_pdfs, consolidate_transactions,
)
from analyzer.datasources import HouseTransactionSource
from analyzer.settings import Settings
from multiprocessing import Pool


def text_only_worker(pdf_path: Path):
    """Try text-layer engines only, no OCR."""
    for engine_fn, name in [
        (_try_pdfplumber, "pdfplumber"),
        (_try_camelot_lattice, "lattice"),
        (_try_camelot_stream, "stream"),
        (_try_pdftotext, "pdftotext"),
    ]:
        txs = engine_fn(pdf_path)
        if txs:
            return pdf_path, txs, [name]
    return pdf_path, [], []


def parse_year(year: int, db: Database, settings: Settings):
    """Parse all cached PDFs for a given year using text-layer only."""
    pdf_dir = Path(settings.data.data_dir) / str(year) / "pdfs"
    if not pdf_dir.exists():
        print(f"  {year}: no pdf dir, skipping")
        return 0
    
    # Get metadata from DB
    src = HouseTransactionSource(settings)
    metadata = src.fetch_metadata(year)
    src.close()
    
    ptrs = metadata[metadata["FilingType"] == "PTR.value"] if "PTR.value" in metadata["FilingType"].values else metadata
    pdf_paths, existing_docs = _filter_existing_pdfs(ptrs, pdf_dir)
    if not pdf_paths:
        print(f"  {year}: no PDFs found")
        return 0
    
    member_lookup = _build_member_lookup(existing_docs)
    print(f"  {year}: parsing {len(pdf_paths)} PDFs with {settings.data.get_workers()} workers...")
    t0 = time.time()
    
    with Pool(settings.data.get_workers()) as pool:
        results = pool.map(text_only_worker, pdf_paths)
    
    elapsed = time.time() - t0
    success = sum(1 for _, txs, _ in results if txs)
    zero = sum(1 for _, txs, _ in results if not txs)
    print(f"  {year}: {success} with rows, {zero} zero-rows in {elapsed:.1f}s")
    
    # Save parse runs
    for pdf_path, transactions, engines_attempted in results:
        doc_id = pdf_path.stem
        status = "success" if transactions else "zero_rows"
        db.upsert_parse_run(
            doc_id=doc_id, year=year,
            parser_version="v3-reparse",
            status=status,
            engines_attempted=",".join(engines_attempted) if engines_attempted else "text-only-failed",
            raw_row_count=0, transaction_count=len(transactions),
        )
    
    # Consolidate and save
    pdf_transactions = {pdf_path: txs for pdf_path, txs, _ in results}
    df = consolidate_transactions(pdf_transactions, member_lookup)
    if df.empty:
        print(f"  {year}: no transactions extracted")
        return 0
    
    # Delete any existing tx for these doc_ids first
    for doc_id in df["doc_id"].unique():
        db.delete_transactions_for_doc(doc_id)
    
    db.upsert_transactions(df, source="house_pdf")
    print(f"  {year}: saved {len(df)} transactions")
    return len(df)


if __name__ == "__main__":
    settings = Settings()
    db = Database(Path(settings.data.data_dir) / "congress.duckdb")
    
    years = [int(y) for y in sys.argv[1:]] if len(sys.argv) > 1 else [2021, 2022, 2023, 2024, 2025, 2026]
    total = 0
    t0 = time.time()
    for year in years:
        total += parse_year(year, db, settings)
        db.execute("CHECKPOINT") if hasattr(db, "execute") else None
    
    print(f"\nDone. {total} tx inserted in {time.time()-t0:.1f}s total")
    db.close()
