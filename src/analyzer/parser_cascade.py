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


class ParserBackendError(RuntimeError):
    """One parser backend failed instead of returning a true empty result."""

    def __init__(self, engine: str, cause: Exception):
        super().__init__(f"{engine}: {cause}")
        self.engine = engine
        self.cause = cause


class ParserCascadeError(RuntimeError):
    """No rows were recovered and at least one backend failed."""


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


def _semantic_score(transactions: list[dict]) -> tuple[int, int, int, float]:
    complete = sum(
        1
        for tx in transactions
        if tx.get("transaction_date") and tx.get("transaction_type")
    )
    with_amount = sum(1 for tx in transactions if tx.get("amount_midpoint"))
    with_asset = sum(1 for tx in transactions if tx.get("asset_description"))
    return complete, with_amount, with_asset, _result_quality(transactions)


def _choose_best(candidates, engines_attempted):
    if not candidates:
        return []
    counts = {name: len(rows) for name, rows in candidates}
    if len(set(counts.values())) > 1:
        detail = ",".join(f"{name}={count}" for name, count in counts.items())
        engines_attempted.append(f"row_disagreement:{detail}")
    engine_preference = {
        "pdftotext": 5,
        "pdfplumber": 4,
        "lattice": 3,
        "stream": 2,
        "docling": 2,
        "ocr": 1,
    }
    name, transactions = max(
        candidates,
        key=lambda candidate: (
            _semantic_score(candidate[1]),
            len(candidate[1]),
            engine_preference.get(candidate[0], 0),
        ),
    )
    engines_attempted.append(f"won:{name}")
    return transactions


def _run_candidate(engine_fn, engine_name, pdf_path, engines_attempted, errors):
    engines_attempted.append(engine_name)
    try:
        transactions = engine_fn(pdf_path)
    except ParserBackendError as exc:
        errors.append(str(exc))
        engines_attempted.append(f"error:{engine_name}")
        return None
    if transactions:
        engines_attempted.append(
            f"candidate:{engine_name}:{len(transactions)}:{_result_quality(transactions):.3f}"
        )
        return engine_name, transactions
    return None


def _parse_pdf_worker(pdf_path: Path) -> tuple[Path, list[dict], list[str]]:
    """Compare every successful text engine, then every OCR engine if needed."""
    skip_docling = os.environ.get("PTR_SKIP_DOCLING") == "1"
    engines_attempted: list[str] = []
    errors: list[str] = []
    text_candidates = []
    for engine_fn, engine_name in [
        (_try_pdfplumber, "pdfplumber"),
        (_try_camelot_lattice, "lattice"),
        (_try_camelot_stream, "stream"),
        (_try_pdftotext, "pdftotext"),
    ]:
        candidate = _run_candidate(
            engine_fn, engine_name, pdf_path, engines_attempted, errors
        )
        if candidate:
            text_candidates.append(candidate)
    if text_candidates:
        return (
            pdf_path,
            _choose_best(text_candidates, engines_attempted),
            engines_attempted,
        )

    ocr_candidates = []
    if not skip_docling:
        candidate = _run_candidate(
            _try_docling, "docling", pdf_path, engines_attempted, errors
        )
        if candidate:
            ocr_candidates.append(candidate)
    candidate = _run_candidate(
        _try_tesseract, "ocr", pdf_path, engines_attempted, errors
    )
    if candidate:
        ocr_candidates.append(candidate)
    if ocr_candidates:
        return (
            pdf_path,
            _choose_best(ocr_candidates, engines_attempted),
            engines_attempted,
        )
    if errors:
        raise ParserCascadeError(
            f"{pdf_path}: parser backend failures with no recovered rows: {'; '.join(errors)}"
        )
    return pdf_path, [], engines_attempted


def _try_pdfplumber(pdf_path: Path) -> list[dict]:
    """Benchmark winner for text-based PDFs (0.075s avg). Handles encrypted
    PDFs natively; returns 0 on scanned images."""
    try:
        pp_tables = extract_tables_with_pdfplumber(pdf_path)
    except Exception as e:
        raise ParserBackendError("pdfplumber", e) from e
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
        raise ParserBackendError("lattice", e) from e
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
        raise ParserBackendError("stream", e) from e
    transactions: list[dict] = []
    for table in tables:
        transactions.extend(parse_pdf_table(table.data))
    return transactions


def _try_pdftotext(pdf_path: Path) -> list[dict]:
    """Handles encrypted PDFs where camelot/pdfplumber return nothing."""
    try:
        pdftext_tables = extract_tables_with_pdftotext(pdf_path)
    except Exception as e:
        raise ParserBackendError("pdftotext", e) from e
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
        raise ParserBackendError("docling", e) from e
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
        raise ParserBackendError("ocr", e) from e
    txs: list[dict] = []
    for table in ocr_tables:
        txs.extend(parse_pdf_table(table))
    return txs
