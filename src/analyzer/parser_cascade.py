"""Per-PDF parser cascade: tries multiple PDF engines until transactions are found."""

import os
import re
import logging
import time
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


class ParseBudgetExceeded(RuntimeError):
    """The watchdog stopped a document before the cascade could finish."""


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
    asset = str(transaction.get("asset_description") or "")
    reported_tickers = re.findall(r"\(([A-Z][A-Z0-9.\-/]{0,9})\)", asset)
    if reported_tickers:
        return _normalize_identity_value(reported_tickers[-1])
    ticker = transaction.get("ticker")
    if ticker:
        return _normalize_identity_value(ticker)
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


def _row_coordinate(transaction: dict) -> tuple[str, str] | None:
    """Split one source_row_id into (coordinate system, position).

    ``pdftotext:l47`` -> ("pdftotext:l", "47"); ``pdfplumber:p2:t0:r9`` ->
    ("pdfplumber:p2:t0:r", "9"). Rows from different systems describe the same
    document through different extraction passes, so identical content at two
    such coordinates cannot prove two distinct visual rows.
    """
    source_row_id = str(transaction.get("source_row_id") or "")
    match = re.fullmatch(r"(?P<system>.+?)(?P<position>\d+)", source_row_id)
    if not match:
        return None
    return match.group("system"), match.group("position")


def _coordinate_distinct_copies(copies: list[dict]) -> bool:
    """Whether repeated identical rows are genuinely distinct source lots.

    Copies that all live in ONE coordinate system are a parser reporting that
    many source rows -- identical lots are legitimate and preserved. Copies
    spread across DIFFERENT coordinate systems of the same engine (pdfplumber's
    table grid versus its text layout) are the merge-artifact signature of one
    visual row read twice (document 20006438): they cannot be coordinate-
    distinct and must escalate instead. Unverifiable rows escalate too.
    """
    if len(copies) < 2:
        return False
    coordinates = [_row_coordinate(transaction) for transaction in copies]
    if any(coordinate is None for coordinate in coordinates):
        return False
    return len({system for system, _ in coordinates}) == 1


def _multiplicity_expansion_is_genuine(
    candidate_rows: list[dict],
    candidate_counts: dict,
    observer_counts: list[dict],
) -> bool:
    """Whether a superset reading is justified beyond every observer's rows.

    A strict subset relationship alone cannot prove the superset complete:
    the extra observations must each be explained. Two explanations count:

    1. Another engine observed the same identity at least as often, so the
       row is corroborated (the count never exceeds the best observer).
    2. The extra copies of an identity the observers did see all live in one
       coordinate system of the winning engine -- genuine identical lots.
       Copies spread across that engine's different extraction passes (table
       grid vs text layout) are merge artifacts and escalate instead.

    Rows NO observer saw (new identities) are unexplained disagreement and
    must escalate instead of being silently accepted.
    """
    for identity, count in candidate_counts.items():
        max_observed = max(
            (counts.get(identity, 0) for counts in observer_counts), default=0
        )
        if count <= max_observed:
            continue
        if max_observed == 0:
            return False
        copies = [
            transaction
            for transaction in candidate_rows
            if _transaction_identity(transaction) == identity
        ]
        if not _coordinate_distinct_copies(copies):
            return False
    return True


def _pop_exact_identity(remaining: list[dict], identity: tuple) -> dict | None:
    for index, row in enumerate(remaining):
        if _transaction_identity(row) == identity:
            return remaining.pop(index)
    return None


def _pop_truncated_asset_match(
    remaining: list[dict], observer: dict
) -> dict | None:
    """Cover an observation whose asset cell was cut short by the observer.

    camelot's lattice/stream modes end an asset cell at a column boundary, so
    a wrapped name arrives truncated and with ``ticker=None`` ("Fifth Third
    Bancorp - Common Stock" instead of "... Common Stock (FITB) [ST]...").
    ``_normalized_asset_identity`` then falls back to the truncated text and
    the identity can never equal the winner's, which demoted every
    wrapped-name filing out of the trusted text path into OCR, where one
    unconfirmed signature page discarded the entire document.

    This is admitted only when the winner resolved a ticker the observer did
    not, every non-asset field agrees exactly, and the winner's asset text
    begins with what the observer captured -- i.e. the observer saw a prefix
    of a row the winner read in full. A row the winner never reported at all
    still fails, so a genuine engine disagreement cannot be absorbed.
    """
    if observer.get("ticker"):
        return None
    observed_asset = _normalize_identity_value(observer.get("asset_description"))
    if not observed_asset:
        return None
    observed_rest = _transaction_identity(observer)[1:]
    for index, row in enumerate(remaining):
        if not row.get("ticker"):
            continue
        if _transaction_identity(row)[1:] != observed_rest:
            continue
        if _normalize_identity_value(row.get("asset_description")).startswith(
            observed_asset
        ):
            return remaining.pop(index)
    return None


def _candidate_covers_all(candidate_rows: list[dict], candidates) -> bool:
    """Return whether one candidate covers every observed row multiplicity."""
    for _, observed_rows in candidates:
        remaining = list(candidate_rows)
        leftovers = []
        for row in observed_rows:
            if _pop_exact_identity(remaining, _transaction_identity(row)) is None:
                leftovers.append(row)
        for row in leftovers:
            if _pop_truncated_asset_match(remaining, row) is None:
                return False
    return True


def _semantic_score(transactions: list[dict]) -> tuple[int, int, int, float]:
    representatives = _candidate_counts(transactions)[1]
    unique = list(representatives.values())
    complete = sum(
        1 for tx in unique if tx.get("transaction_date") and tx.get("transaction_type")
    )
    with_amount = sum(1 for tx in unique if tx.get("amount_midpoint"))
    with_asset = sum(1 for tx in unique if tx.get("asset_description"))
    return complete, with_amount, with_asset, _result_quality(unique)


def _find_parse_budget(exc: BaseException) -> ParseBudgetExceeded | None:
    """Find a watchdog exception retained by one or more backend wrappers."""
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, ParseBudgetExceeded):
            return current
        cause = getattr(current, "cause", None)
        if not isinstance(cause, BaseException):
            cause = current.__cause__
        if not isinstance(cause, BaseException):
            cause = current.__context__
        current = cause
    return None


def _reconcile_candidates(candidates, engines_attempted):
    """Merge complementary rows while retaining the maximum observed lot count."""
    if not candidates:
        return []
    counts_by_engine = {name: _candidate_counts(rows)[0] for name, rows in candidates}
    raw_counts = {name: len(rows) for name, rows in candidates}
    unique_counts = {name: len(counts) for name, counts in counts_by_engine.items()}
    if len(set(raw_counts.values())) > 1 or len(set(unique_counts.values())) > 1:
        detail = ",".join(
            f"{name}={raw_counts[name]}/{unique_counts[name]}u"
            for name, _ in candidates
        )
        engines_attempted.append(f"row_disagreement:{detail}")

    best_name, best_rows = max(
        candidates,
        key=lambda candidate: (
            _semantic_score(candidate[1]),
            len(counts_by_engine[candidate[0]]),
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
        return best_rows

    maximum_counts = dict(best_counts)
    reconciled = list(best_rows)
    for _, rows in candidates:
        counts = {}
        for transaction in rows:
            identity = _transaction_identity(transaction)
            counts[identity] = counts.get(identity, 0) + 1
            if counts[identity] > maximum_counts.get(identity, 0):
                # Keep the observed lot, not copies of its first representative.
                reconciled.append(transaction)
        for identity, count in counts.items():
            maximum_counts[identity] = max(maximum_counts.get(identity, 0), count)
    engines_attempted.append(f"complementary_rows:{len(reconciled)}")
    return reconciled


def _call_parser_backend(engine_name, backend, *args, **kwargs):
    try:
        return backend(*args, **kwargs)
    except ParseBudgetExceeded:
        raise
    except Exception as exc:
        raise ParserBackendError(engine_name, exc) from exc


def _run_candidate(engine_fn, engine_name, pdf_path, engines_attempted, errors):
    engines_attempted.append(engine_name)
    started = time.monotonic()
    try:
        transactions = engine_fn(pdf_path)
    except ParseBudgetExceeded:
        raise
    except ParserBackendError as exc:
        budget = _find_parse_budget(exc)
        if budget is not None:
            raise budget from exc
        elapsed = time.monotonic() - started
        errors.append(str(exc))
        engines_attempted.append(f"error:{engine_name}")
        logger.debug(
            "Parser %s failed for %s after %.3fs: %s",
            engine_name,
            pdf_path.name,
            elapsed,
            exc,
        )
        return None

    elapsed = time.monotonic() - started
    quality = _result_quality(transactions)
    logger.debug(
        "Parser %s completed %s in %.3fs: rows=%d quality=%.3f",
        engine_name,
        pdf_path.name,
        elapsed,
        len(transactions),
        quality,
    )
    if transactions:
        # Table/OCR engines without explicit coordinates still have a stable
        # transaction sequence within this hashed artifact. Keep each lot distinct.
        for row_number, transaction in enumerate(transactions, start=1):
            if not transaction.get("source_row_id"):
                transaction["source_row_id"] = f"{engine_name}:r{row_number}"
        engines_attempted.append(
            f"candidate:{engine_name}:{len(transactions)}:{quality:.3f}"
        )
        return engine_name, transactions
    return None


def _has_document_text_coverage(pdf_path: Path) -> bool:
    """Text agreement cannot establish coverage of pages without text."""
    import pdfplumber

    with pdfplumber.open(str(pdf_path)) as pdf:
        return bool(pdf.pages) and all(
            (page.extract_text(layout=True) or "").replace("\x00", "").strip()
            for page in pdf.pages
        )


def _run_text_engines(
    pdf_path: Path,
) -> tuple[
    tuple[Path, list[dict], list[str]] | None,
    list[tuple[str, list[dict]]],
    list[dict],
    list[str],
    list[str],
]:
    """Run every text engine and return a trusted result when they prove coverage."""
    engines_attempted: list[str] = []
    errors: list[str] = []
    text_candidates: list[tuple[str, list[dict]]] = []
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

    reconciled_text = _reconcile_candidates(text_candidates, engines_attempted)
    trusted = {name: rows for name, rows in text_candidates}
    pdfplumber_rows = trusted.get("pdfplumber", [])
    pdftotext_rows = trusted.get("pdftotext", [])
    pdfplumber_counts = _candidate_counts(pdfplumber_rows)[0]
    pdftotext_counts = _candidate_counts(pdftotext_rows)[0]
    pdfplumber_subset = bool(pdfplumber_counts) and _multiset_subset(
        pdfplumber_counts, pdftotext_counts
    )
    pdftotext_subset = bool(pdftotext_counts) and _multiset_subset(
        pdftotext_counts, pdfplumber_counts
    )
    if pdfplumber_subset or pdftotext_subset:
        try:
            covered = _call_parser_backend(
                "text_coverage", _has_document_text_coverage, pdf_path
            )
        except ParserBackendError as exc:
            budget = _find_parse_budget(exc)
            if budget is not None:
                raise budget from exc
            errors.append(str(exc))
            engines_attempted.append("error:text_coverage")
            covered = False
        if not covered:
            engines_attempted.append("text_page_coverage_unconfirmed")
            return None, text_candidates, reconciled_text, engines_attempted, errors

    # A strict subset relationship is disagreement, not proof of completeness:
    # one engine silently dropped visible rows, so the surviving reading may
    # carry artifacts the dropped rows once exposed. Trust a superset only
    # when every row beyond the other engines' multiplicities is genuine
    # (corroborated, or coordinate-distinct identical lots); otherwise the
    # document escalates to the OCR rung (fail closed).
    trust_attempts = [
        (
            pdfplumber_subset,
            pdftotext_rows,
            pdftotext_counts,
            "trusted:pdfplumber_subset_pdftotext",
            "won:pdftotext",
        ),
        (
            pdftotext_subset,
            pdfplumber_rows,
            pdfplumber_counts,
            "trusted:pdftotext_subset_pdfplumber",
            "won:pdfplumber",
        ),
    ]
    for is_subset, rows, counts, trusted_tag, won_tag in trust_attempts:
        if not is_subset:
            continue
        observer_counts = [
            _candidate_counts(observer_rows)[0]
            for observer_name, observer_rows in text_candidates
            if observer_rows is not rows
        ]
        if not _multiplicity_expansion_is_genuine(rows, counts, observer_counts):
            engines_attempted.append(f"subset_expansion_unverified:{won_tag[4:]}")
            continue
        if not _candidate_covers_all(rows, text_candidates):
            engines_attempted.append(f"text_engines_miss_observed_rows:{won_tag[4:]}")
            continue
        engines_attempted.append(trusted_tag)
        engines_attempted.append(won_tag)
        result = pdf_path, rows, engines_attempted
        return result, text_candidates, reconciled_text, engines_attempted, errors
    if text_candidates:
        engines_attempted.append("text_row_disagreement_unresolved")
    return None, text_candidates, reconciled_text, engines_attempted, errors


def _parse_text_only_worker(pdf_path: Path) -> tuple[Path, list[dict], list[str]]:
    """Return only text-proven results; defer uncertain documents to OCR."""
    result, *_ = _run_text_engines(pdf_path)
    if result is not None:
        return result
    raise ParserCascadeError(f"{pdf_path}: text engines did not prove complete coverage")


def _parse_pdf_worker(pdf_path: Path) -> tuple[Path, list[dict], list[str]]:
    """Compare text engines; require complete Tesseract coverage when uncertain."""
    skip_docling = os.environ.get("PTR_SKIP_DOCLING") == "1"
    (
        trusted_result,
        text_candidates,
        reconciled_text,
        engines_attempted,
        errors,
    ) = _run_text_engines(pdf_path)
    if trusted_result is not None:
        return trusted_result

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
        prior_candidates = list(text_candidates) + [
            candidate for candidate in ocr_candidates if candidate[0] != "ocr"
        ]
        if _candidate_covers_all(tesseract_candidate[1], prior_candidates):
            all_candidates = list(text_candidates) + ocr_candidates
            reconciled = _reconcile_candidates(all_candidates, engines_attempted)
            engines_attempted.append("won:reconciled_complete_ocr")
            return pdf_path, reconciled, engines_attempted
        engines_attempted.append("ocr_incomplete_for_observed_rows")

    if reconciled_text or ocr_candidates or errors:
        detail = "; ".join(errors) or "unconfirmed complementary/parser rows"
        raise ParserCascadeError(
            f"{pdf_path}: unresolved parser completeness: {detail}"
        )
    raise ParserCascadeError(f"{pdf_path}: no parser established complete row coverage")


def _try_pdfplumber(pdf_path: Path) -> list[dict]:
    """Benchmark winner for text-based PDFs (0.075s avg). Handles encrypted
    PDFs natively; returns 0 on scanned images."""
    pp_tables = _call_parser_backend(
        "pdfplumber", extract_tables_with_pdfplumber, pdf_path
    )
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
    tables = _call_parser_backend(
        "lattice", camelot.read_pdf, str(pdf_path), pages="all", flavor="lattice"
    )
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
    """Fallback for unrulled tables; aggregate all detected tables."""
    tables = _call_parser_backend(
        "stream", camelot.read_pdf, str(pdf_path), pages="all", flavor="stream"
    )
    transactions: list[dict] = []
    for table in tables:
        transactions.extend(parse_pdf_table(table.data))
    return transactions


def _try_pdftotext(pdf_path: Path) -> list[dict]:
    """Extract and aggregate every pdftotext table."""
    pdftext_tables = _call_parser_backend(
        "pdftotext", extract_tables_with_pdftotext, pdf_path
    )
    transactions: list[dict] = []
    for table in pdftext_tables:
        transactions.extend(parse_pdf_table(table))
    return transactions


def _try_docling(pdf_path: Path) -> list[dict]:
    """Extract every Docling table from a scanned PDF.

    Docling is expensive (roughly 13-300s and about 2 GB per process), so bulk
    runs may set ``PTR_SKIP_DOCLING=1`` and handle stragglers separately.
    """
    docling_tables = _call_parser_backend(
        "docling", extract_tables_with_docling, pdf_path
    )
    transactions: list[dict] = []
    for table in docling_tables:
        transactions.extend(parse_pdf_table(table))
    return transactions


def _try_tesseract(pdf_path: Path) -> list[dict]:
    """Last-resort OCR. Kept for backward compatibility when docling/uvx
    is unavailable."""
    ocr_tables = _call_parser_backend("ocr", extract_tables_with_ocr, pdf_path)
    txs: list[dict] = []
    for table in ocr_tables:
        txs.extend(parse_pdf_table(table))
    return txs
