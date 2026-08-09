"""Reparse all cached PDFs through the production parser selection API.

Docling remains disabled to avoid its multi-gigabyte worker footprint. The
production cascade still controls text-engine comparison and final OCR fallback.
"""

from __future__ import annotations
import os
import sys
import time
from pathlib import Path

# Must set env BEFORE importing analyzer (it may import OCR libs lazily)
os.environ["PTR_SKIP_DOCLING"] = "1"

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from analyzer.database import Database
from analyzer.models import FilingType

from analyzer.datasources import (
    HouseTransactionSource,
    _build_member_lookup,
    _filter_existing_pdfs,
    consolidate_transactions,
)
from analyzer.parser_cascade import _parse_pdf_worker
from analyzer.download import preserve_existing_fields
from analyzer.settings import Settings
from multiprocessing import Pool


def parse_year(year: int, db: Database, settings: Settings):
    """Parse all cached PDFs for a year with the production cascade."""
    pdf_dir = Path(settings.data.data_dir) / str(year) / "pdfs"
    if not pdf_dir.exists():
        print(f"  {year}: no pdf dir, skipping")
        return 0

    # Get metadata from DB
    src = HouseTransactionSource(settings)
    metadata = src.fetch_metadata(year)
    src.close()

    ptrs = metadata[metadata["FilingType"] == FilingType.PTR.value]
    pdf_paths, existing_docs = _filter_existing_pdfs(ptrs, pdf_dir)
    if not pdf_paths:
        print(f"  {year}: no PDFs found")
        return 0

    member_lookup = _build_member_lookup(existing_docs)
    print(
        f"  {year}: parsing {len(pdf_paths)} PDFs with {settings.data.get_workers()} workers..."
    )
    t0 = time.time()

    with Pool(settings.data.get_workers()) as pool:
        results = pool.map(_parse_pdf_worker, pdf_paths)

    elapsed = time.time() - t0
    success = sum(1 for _, txs, _ in results if txs)
    zero = sum(1 for _, txs, _ in results if not txs)
    print(f"  {year}: {success} with rows, {zero} zero-rows in {elapsed:.1f}s")

    pdf_transactions = {pdf_path: txs for pdf_path, txs, _ in results}
    emitted_counts = {pdf_path.stem: len(txs) for pdf_path, txs, _ in results}
    attempted_doc_ids = list(emitted_counts)
    df = consolidate_transactions(pdf_transactions, member_lookup)

    # Carry forward previously-resolved ticker/amount before the delete+reinsert
    # so a weaker parse does not clobber good data already in the DB.
    df = preserve_existing_fields(df, db)

    # The database API owns atomic replacement of every attempted document,
    # including zero-output documents. Query afterward instead of treating the
    # emitted DataFrame length as proof of persistence.
    db.replace_transactions_for_docs(
        df,
        source="house_pdf",
        attempted_doc_ids=attempted_doc_ids,
    )
    persisted_counts = _persisted_house_counts(db, attempted_doc_ids)
    _verify_persisted_counts(year, emitted_counts, persisted_counts)

    for pdf_path, transactions, engines_attempted in results:
        doc_id = pdf_path.stem
        persisted = persisted_counts.get(doc_id, 0)
        db.upsert_parse_run(
            doc_id=doc_id,
            year=year,
            parser_version="v3-reparse",
            status="success" if persisted else "zero_rows",
            engines_attempted=",".join(engines_attempted)
            if engines_attempted
            else "production-cascade-failed",
            raw_row_count=len(transactions),
            transaction_count=persisted,
        )

    persisted_total = sum(persisted_counts.values())
    print(f"  {year}: persisted {persisted_total} verified transactions")
    return persisted_total


def _verify_persisted_counts(
    year: int, emitted_counts: dict[str, int], persisted_counts: dict[str, int]
) -> None:
    mismatches = {
        doc_id: (emitted, persisted_counts.get(doc_id, 0))
        for doc_id, emitted in emitted_counts.items()
        if emitted != persisted_counts.get(doc_id, 0)
    }
    if not mismatches:
        return
    sample = ", ".join(
        f"{doc_id}={emitted}/{persisted}"
        for doc_id, (emitted, persisted) in list(mismatches.items())[:10]
    )
    raise RuntimeError(
        f"{year}: emitted/persisted transaction mismatch in "
        f"{len(mismatches)} document(s): {sample}"
    )


def _persisted_house_counts(db: Database, doc_ids: list[str]) -> dict[str, int]:
    """Return verified House row counts and reject mixed-source replacements."""
    if not doc_ids:
        return {}
    placeholders = ", ".join("?" for _ in doc_ids)
    rows = db.conn.execute(
        f"""
        SELECT doc_id, source, COUNT(*)
        FROM transactions
        WHERE doc_id IN ({placeholders})
        GROUP BY doc_id, source
        """,  # nosec B608 -- placeholders only; values remain bound
        doc_ids,
    ).fetchall()
    unexpected = [
        (doc_id, source) for doc_id, source, _ in rows if source != "house_pdf"
    ]
    if unexpected:
        sample = ", ".join(f"{doc_id}:{source}" for doc_id, source in unexpected[:10])
        raise RuntimeError(
            f"replacement retained non-House transaction source(s): {sample}"
        )
    return {doc_id: count for doc_id, _, count in rows}


if __name__ == "__main__":
    settings = Settings()
    db = Database(Path(settings.data.data_dir) / "congress.duckdb")

    years = (
        [int(y) for y in sys.argv[1:]]
        if len(sys.argv) > 1
        else [2021, 2022, 2023, 2024, 2025, 2026]
    )
    total = 0
    t0 = time.time()
    for year in years:
        total += parse_year(year, db, settings)
        db.conn.execute("CHECKPOINT")

    print(f"\nDone. {total} tx inserted in {time.time() - t0:.1f}s total")
    db.close()
