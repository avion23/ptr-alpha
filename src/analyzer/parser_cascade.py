"""Per-PDF parser cascade: tries multiple PDF engines until transactions are found."""

import os
import logging
from pathlib import Path

import camelot

from analyzer.parsing import (
    extract_tables_with_docling,
    extract_tables_with_ocr,
    extract_tables_with_pdfplumber,
    extract_tables_with_pdftotext,
    parse_pdf_table,
    _parse_ocr_text_to_rows,
)

logger = logging.getLogger(__name__)


# ── Per-PDF worker: cascades 5 PDF engines until transactions are found ─


def _is_valid_pdf(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    with open(path, "rb") as f:
        return f.read(5) == b"%PDF-"


def _result_quality(txs: list[dict]) -> float:
    if not txs:
        return 0.0
    valid = sum(
        1 for tx in txs if tx.get("transaction_date") and tx.get("amount_midpoint")
    )
    return valid / len(txs)


def _parse_pdf_worker(pdf_path: Path) -> tuple[Path, list[dict], list[str]]:
    """Cascade through pdfplumber → camelot lattice → camelot stream →
    pdftotext → Docling OCR → tesseract OCR. Each engine is tried only when
    all text-layer parsers returned 0 transactions; this minimises OCR work
    while allowing better text-layer engines to beat low-quality results."""
    skip_docling = os.environ.get("PTR_SKIP_DOCLING") == "1"
    engines_attempted: list[str] = []
    best_transactions: list[dict] = []
    best_engine = ""
    best_quality = 0.0

    for engine_fn, engine_name in [
        (_try_pdfplumber, "pdfplumber"),
        (_try_camelot_lattice, "lattice"),
        (_try_camelot_stream, "stream"),
        (_try_pdftotext, "pdftotext"),
    ]:
        transactions = engine_fn(pdf_path)
        engines_attempted.append(engine_name)
        if transactions:
            quality = _result_quality(transactions)
            if quality >= 0.7:
                engines_attempted.append(f"won:{engine_name}")
                return pdf_path, transactions, engines_attempted
            if (quality, len(transactions)) > (best_quality, len(best_transactions)):
                best_transactions = transactions
                best_engine = engine_name
                best_quality = quality

    if best_transactions:
        engines_attempted.append(f"won:{best_engine}")
        return pdf_path, best_transactions, engines_attempted

    transactions: list[dict] = []

    # Docling + tesseract: only run if all text-layer engines returned 0
    if not skip_docling:
        engines_attempted.append("docling")
        transactions = _try_docling(pdf_path)
        if transactions:
            engines_attempted.append("won:docling")
            return pdf_path, transactions, engines_attempted

    engines_attempted.append("ocr")
    transactions = _try_tesseract(pdf_path)
    if transactions:
        engines_attempted.append("won:ocr")

    return pdf_path, transactions, engines_attempted


def _try_pdfplumber(pdf_path: Path) -> list[dict]:
    """Benchmark winner for text-based PDFs (0.075s avg). Handles encrypted
    PDFs natively; returns 0 on scanned images."""
    try:
        pp_tables = extract_tables_with_pdfplumber(pdf_path)
    except Exception as e:
        logger.debug(f"pdfplumber failed for {pdf_path}: {e}")
        return []
    if not pp_tables:
        return []
    txs: list[dict] = []
    for table in pp_tables:
        txs.extend(parse_pdf_table(table))
    return txs


def _try_camelot_lattice(pdf_path: Path) -> list[dict]:
    """Previous primary parser; keeps PDFs with ruling lines. Lattice
    sometimes collapses to 1-column when null bytes are present — in that
    case we re-parse each cell as OCR text."""
    try:
        tables = camelot.read_pdf(str(pdf_path), pages="all", flavor="lattice")
    except Exception as e:
        logger.debug(f"Lattice failed for {pdf_path}: {e}")
        return []
    txs: list[dict] = []
    for table in tables:
        data = table.data
        if data and len(data[0]) == 1:
            for row in data:
                cell_text = row[0] if row else ""
                if cell_text:
                    ocr_rows = _parse_ocr_text_to_rows(cell_text)
                    if ocr_rows:
                        ocr_table = [
                            [
                                "Asset Name",
                                "Transaction Type",
                                "Transaction Date",
                                "Amount",
                            ]
                        ] + ocr_rows
                        txs.extend(parse_pdf_table(ocr_table))
        else:
            txs.extend(parse_pdf_table(data))
    return txs


def _try_camelot_stream(pdf_path: Path) -> list[dict]:
    """Fallback for unrulled tables. Try ALL detected tables and stop at
    the first one that yields transactions (Fix 2: don't just scan table[0])."""
    try:
        tables = camelot.read_pdf(str(pdf_path), pages="all", flavor="stream")
    except Exception as e:
        logger.debug(f"Stream failed for {pdf_path}: {e}")
        return []
    for table in tables:
        txs = parse_pdf_table(table.data)
        if txs:
            return txs
    return []


def _try_pdftotext(pdf_path: Path) -> list[dict]:
    """Handles encrypted PDFs where camelot/pdfplumber return nothing."""
    try:
        pdftext_tables = extract_tables_with_pdftotext(pdf_path)
    except Exception as e:
        logger.debug(f"pdftotext failed for {pdf_path}: {e}")
        return []
    for table in pdftext_tables:
        txs = parse_pdf_table(table)
        if txs:
            return txs
    return []


def _try_docling(pdf_path: Path) -> list[dict]:
    """OCR fallback for SCANNED IMAGE PDFs (no text layer). Slow (13-300s)
    but only runs when all text-layer parsers return nothing.

    Skip when PTR_SKIP_DOCLING=1 (set during the bulk first pass; stragglers
    are re-parsed with Docling in a second pass using reduced worker count
    to avoid OOM — each Docling proc uses ~2GB).
    """
    try:
        docling_tables = extract_tables_with_docling(pdf_path)
    except Exception as e:
        logger.debug(f"Docling failed for {pdf_path}: {e}")
        return []
    for table in docling_tables:
        txs = parse_pdf_table(table)
        if txs:
            return txs
    return []


def _try_tesseract(pdf_path: Path) -> list[dict]:
    """Last-resort OCR. Kept for backward compatibility when docling/uvx
    is unavailable."""
    try:
        ocr_tables = extract_tables_with_ocr(pdf_path)
    except Exception as e:
        logger.debug(f"OCR failed for {pdf_path}: {e}")
        return []
    txs: list[dict] = []
    for table in ocr_tables:
        txs.extend(parse_pdf_table(table))
    return txs
