"""Per-PDF parser cascade: tries multiple PDF engines until transactions are found."""

import os
import re
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


def _normalize_identity_value(value) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())


def _normalized_asset_identity(transaction: dict) -> str:
    ticker = transaction.get("ticker")
    if ticker:
        return _normalize_identity_value(ticker)
    asset = str(transaction.get("asset_description") or "")
    asset = re.sub(r"\[Account:.*$", "", asset, flags=re.IGNORECASE)
    asset = re.sub(r"\[[A-Z]{2}\]", "", asset)
    return _normalize_identity_value(asset)


def _transaction_identity(transaction: dict) -> tuple:
    return (
        _normalized_asset_identity(transaction),
        _normalize_identity_value(transaction.get("transaction_date")),
        _normalize_identity_value(transaction.get("transaction_type")),
        _normalize_identity_value(
            transaction.get("amount_raw") or transaction.get("amount_midpoint")
        ),
        _normalize_identity_value(transaction.get("owner_code")),
        _normalize_identity_value(transaction.get("notification_date")),
        _normalize_identity_value(transaction.get("page_number")),
        _normalize_identity_value(transaction.get("source_row_id")),
    )


def _candidate_counts(transactions: list[dict]):
    counts = {}
    representatives = {}
    for transaction in transactions:
        identity = _transaction_identity(transaction)
        counts[identity] = counts.get(identity, 0) + 1
        representatives.setdefault(identity, transaction)
    return counts, representatives


def _multiset_subset(left: dict, right: dict) -> bool:
    return all(count <= right.get(identity, 0) for identity, count in left.items())


def _semantic_score(transactions: list[dict]) -> tuple[int, int, int, float]:
    counts, representatives = _candidate_counts(transactions)
    unique = list(representatives.values())
    complete = sum(
        1 for tx in unique if tx.get("transaction_date") and tx.get("transaction_type")
    )
    with_amount = sum(1 for tx in unique if tx.get("amount_midpoint"))
    with_asset = sum(1 for tx in unique if tx.get("asset_description"))
    return complete, with_amount, with_asset, _result_quality(unique)


def _reconcile_candidates(candidates, engines_attempted):
    """Merge complementary rows while retaining the maximum observed lot count."""
    if not candidates:
        return [], False
    counts_by_engine = {name: _candidate_counts(rows)[0] for name, rows in candidates}
    raw_counts = {name: len(rows) for name, rows in candidates}
    unique_counts = {name: len(counts) for name, counts in counts_by_engine.items()}
    if len(set(raw_counts.values())) > 1 or len(set(unique_counts.values())) > 1:
        detail = ",".join(
            f"{name}={raw_counts[name]}/{unique_counts[name]}u"
            for name, _ in candidates
        )
        engines_attempted.append(f"row_disagreement:{detail}")

    engine_preference = {
        "pdftotext": 5,
        "pdfplumber": 4,
        "lattice": 3,
        "stream": 2,
        "docling": 2,
        "ocr": 1,
    }
    best_name, best_rows = max(
        candidates,
        key=lambda candidate: (
            _semantic_score(candidate[1]),
            len(counts_by_engine[candidate[0]]),
            engine_preference.get(candidate[0], 0),
        ),
    )
    best_counts = counts_by_engine[best_name]
    complementary = any(
        not _multiset_subset(counts, best_counts)
        for name, counts in counts_by_engine.items()
        if name != best_name
    )
    if not complementary:
        engines_attempted.append(f"won:{best_name}")
        return best_rows, False

    maximum_counts = dict(best_counts)
    representatives = _candidate_counts(best_rows)[1]
    for _, rows in candidates:
        counts, current_representatives = _candidate_counts(rows)
        for identity, count in counts.items():
            maximum_counts[identity] = max(maximum_counts.get(identity, 0), count)
            representatives.setdefault(identity, current_representatives[identity])
    reconciled = []
    for identity, count in maximum_counts.items():
        reconciled.extend([representatives[identity]] * count)
    engines_attempted.append(f"complementary_rows:{len(reconciled)}")
    return reconciled, True


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
    """Compare text engines; require complete Tesseract coverage when uncertain."""
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

    reconciled_text, text_uncertain = _reconcile_candidates(
        text_candidates, engines_attempted
    )
    trusted = {name: rows for name, rows in text_candidates}
    pdfplumber_counts = _candidate_counts(trusted.get("pdfplumber", []))[0]
    pdftotext_counts = _candidate_counts(trusted.get("pdftotext", []))[0]
    trusted_overlap = sum(
        min(count, pdftotext_counts.get(identity, 0))
        for identity, count in pdfplumber_counts.items()
    )
    trusted_complete = (
        bool(pdfplumber_counts)
        and bool(pdftotext_counts)
        and trusted_overlap / sum(pdfplumber_counts.values()) >= 0.85
    )
    if trusted_complete:
        engines_attempted.append("trusted:pdfplumber_subset_pdftotext")
        engines_attempted.append("won:pdftotext")
        return pdf_path, trusted["pdftotext"], engines_attempted
    ocr_candidates = []
    if not skip_docling:
        candidate = _run_candidate(
            _try_docling, "docling", pdf_path, engines_attempted, errors
        )
        if candidate:
            ocr_candidates.append(candidate)
    tesseract_candidate = _run_candidate(
        _try_tesseract, "ocr", pdf_path, engines_attempted, errors
    )
    if tesseract_candidate:
        ocr_candidates.append(tesseract_candidate)

    if tesseract_candidate:
        all_candidates = list(text_candidates) + ocr_candidates
        reconciled, _ = _reconcile_candidates(all_candidates, engines_attempted)
        engines_attempted.append("won:reconciled_complete_ocr")
        return pdf_path, reconciled, engines_attempted

    if reconciled_text or ocr_candidates or errors:
        detail = "; ".join(errors) or "unconfirmed complementary/parser rows"
        raise ParserCascadeError(
            f"{pdf_path}: unresolved parser completeness: {detail}"
        )
    raise ParserCascadeError(f"{pdf_path}: no parser established complete row coverage")


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
