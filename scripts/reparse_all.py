"""Reparse all cached PDFs through the production parser selection API.

Docling remains disabled to avoid its multi-gigabyte worker footprint. The
production cascade still controls text-engine comparison and final OCR fallback.
"""

from __future__ import annotations

import hashlib
import os
import sys
import time
from pathlib import Path

# Must set env BEFORE importing analyzer (it may import OCR libs lazily)
os.environ["PTR_SKIP_DOCLING"] = "1"

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from multiprocessing import Pool
from typing import cast

import pandas as pd

from analyzer.database import Database
from analyzer.download import (
    HouseTransactionSource,
    _build_member_lookup,
    _filter_existing_pdfs,
    preserve_existing_fields,
)
from analyzer.models import FilingType
from analyzer.parsing import consolidate_transactions
from analyzer.parser_cascade import (
    ParserCascadeError,
    _parse_pdf_worker,
)
from analyzer.settings import Settings


def _resilient_worker(
    pdf_path: Path,
) -> tuple[Path, list[dict], list[str], str | None]:
    """Isolate per-document failures so one unreadable PDF cannot abort the
    whole batch. Errors are recorded as parse-run rows; prior House rows for
    those documents stay preserved because they never enter
    ``replacement_doc_ids``."""
    try:
        pdf_path_out, txs, engines = _parse_pdf_worker(pdf_path)
        return pdf_path_out, txs, engines, None
    except (ParserCascadeError, OSError) as exc:
        return pdf_path, [], [], f"{type(exc).__name__}: {exc}"


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

    ptrs = cast(pd.DataFrame, metadata[metadata["FilingType"] == FilingType.PTR.value])
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
        results = pool.map(_resilient_worker, pdf_paths)

    elapsed = time.time() - t0
    errors = [(path, err) for path, _, _, err in results if err]
    success = sum(1 for _, txs, _, err in results if txs and not err)
    zero = sum(1 for _, txs, _, err in results if not txs and not err)
    print(
        f"  {year}: {success} with rows, {zero} zero-rows, {len(errors)} errors "
        f"in {elapsed:.1f}s"
    )
    for path, err in errors[:10]:
        print(f"    error {path.name}: {err}")

    pdf_transactions = {pdf_path: txs for pdf_path, txs, _, _ in results}
    emitted_counts = {pdf_path.stem: len(txs) for pdf_path, txs, _, _ in results}
    attempted_doc_ids = list(emitted_counts)
    replacement_doc_ids = [
        doc_id for doc_id, emitted in emitted_counts.items() if emitted > 0
    ]
    artifact_hashes = {
        pdf_path.stem: _artifact_sha256(pdf_path) for pdf_path, _, _, _ in results
    }
    ingestion_generation = (
        db.get_latest_house_generation(year) or f"legacy-untracked-{year}"
    )
    df = consolidate_transactions(pdf_transactions, member_lookup)
    consolidated_counts = (
        df["doc_id"].astype(str).value_counts().to_dict() if not df.empty else {}
    )
    # Documents whose parsed rows failed consolidation (e.g. uncoercible
    # dates) must never partially replace prior House rows: exclude them
    # from the replacement set, record an error run, and keep their
    # existing rows intact.
    drop_reasons = {
        doc_id: (
            f"consolidation kept {consolidated_counts.get(doc_id, 0)}/"
            f"{emitted_counts[doc_id]} parsed row(s); invalid dates or missing member metadata"
        )
        for doc_id in replacement_doc_ids
        if consolidated_counts.get(doc_id, 0) < emitted_counts[doc_id]
    }
    if drop_reasons:
        print(
            f"  {year}: consolidation shortfalls in {len(drop_reasons)} "
            "document(s), preserving prior rows: "
            + ", ".join(sorted(drop_reasons)[:10])
        )
        replacement_doc_ids = [
            doc_id for doc_id in replacement_doc_ids if doc_id not in drop_reasons
        ]
        if not df.empty and replacement_doc_ids:
            df = cast(
                pd.DataFrame,
                df[df["doc_id"].astype(str).isin(replacement_doc_ids)].copy(),
            )

    # Carry forward previously-resolved ticker/amount before the delete+reinsert
    # so a weaker parse does not clobber good data already in the DB.
    df = preserve_existing_fields(df, db)
    if not df.empty:
        df["ingestion_generation"] = ingestion_generation
        df["artifact_sha256"] = df["doc_id"].astype(str).map(artifact_hashes.get)
    parse_runs = []
    for pdf_path, transactions, engines_attempted, error in results:
        doc_id = pdf_path.stem
        if doc_id in drop_reasons:
            status = "error"
        elif transactions:
            status = "success"
        elif error:
            status = "error"
        else:
            status = "zero_rows"
        parse_runs.append(
            {
                "doc_id": doc_id,
                "year": year,
                "parser_version": "v3-reparse",
                "status": status,
                "engines_attempted": ",".join(engines_attempted)
                if engines_attempted and not error
                else "production-cascade-failed",
                "raw_row_count": len(transactions),
                "transaction_count": 0,
                "error_message": error or drop_reasons.get(doc_id),
                "artifact_sha256": artifact_hashes[doc_id],
                "ingestion_generation": ingestion_generation,
            }
        )

    # Empty deterministic results are ambiguous: record the attempt, but do not
    # replace prior House rows. Only nonzero successes (or a future explicit
    # verified no_txs result) belong in replacement_doc_ids.
    db.replace_transactions_for_docs(
        df,
        source="house_pdf",
        attempted_doc_ids=attempted_doc_ids,
        replacement_doc_ids=replacement_doc_ids,
        ingestion_generation=ingestion_generation,
        parse_runs=parse_runs,
    )
    persisted_house_counts = _persisted_house_generation_counts(
        db, attempted_doc_ids, ingestion_generation
    )
    _verify_persisted_counts(
        year,
        {doc_id: emitted_counts[doc_id] for doc_id in replacement_doc_ids},
        persisted_house_counts,
    )

    persisted_total = sum(
        persisted_house_counts.get(doc_id, 0) for doc_id in replacement_doc_ids
    )
    ambiguous_with_prior_house_rows = sum(
        bool(persisted_house_counts.get(doc_id, 0))
        for doc_id in attempted_doc_ids
        if doc_id not in replacement_doc_ids
    )
    print(
        f"  {year}: persisted {persisted_total} verified transactions; "
        f"preserved prior House rows for {ambiguous_with_prior_house_rows} "
        "ambiguous zero-row document(s)"
    )
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


def _artifact_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _persisted_house_generation_counts(
    db: Database, doc_ids: list[str], ingestion_generation: str
) -> dict[str, int]:
    """Query actual House counts for the targeted acquired generation."""
    if not doc_ids:
        return {}
    rows = db.conn.execute(
        """
        SELECT doc_id, COUNT(*)
        FROM transactions
        WHERE doc_id IN (SELECT UNNEST(CAST(? AS VARCHAR[])))
          AND source = 'house_pdf'
          AND ingestion_generation = ?
        GROUP BY doc_id
        """,
        [doc_ids, ingestion_generation],
    ).fetchall()
    return {str(doc_id): count for doc_id, count in rows}


if __name__ == "__main__":
    settings = Settings()
    db = Database(Path(settings.data.data_dir) / "congress.duckdb")

    try:
        years = [int(y) for y in sys.argv[1:]]
    except ValueError as exc:
        raise SystemExit(f"usage: reparse_all.py [year ...]: {exc}") from exc
    total = 0
    t0 = time.time()
    for year in years:
        total += parse_year(year, db, settings)
        db.conn.execute("CHECKPOINT")

    print(f"\nDone. {total} tx inserted in {time.time() - t0:.1f}s total")
    db.close()
