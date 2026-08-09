#!/usr/bin/env python3
"""Shared Gemini OCR parsing, cache, and validation helpers."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from analyzer.member_names import canonical_member_key

MODEL = "gemini/gemini-3.1-flash-lite"
GEMINI_PARSER_VERSION = "v5-gemini-validated"
CACHE_ENVELOPE_VERSION = 1
OUTPUT_SCHEMA_VERSION = 3
CACHE_DIR = "data/gemini_cache"
AMOUNT_MIDPOINTS = {
    "A": 8000,
    "B": 32500,
    "C": 75000,
    "D": 175000,
    "E": 375000,
    "F": 750000,
    "G": 3000000,
    "H": 15000000,
    "I": 37500000,
    "J": 50000000,
}

PROMPT = """This is a US House Periodic Transaction Report (PTR). The first data row may be an EXAMPLE labeled "Example: Mega Corp. Common Stock"; skip it. Extract every real transaction from every page.

Output format:
MEMBER: [full name of filer]
PAGES: [total PDF page count]
PAGE: [page number]
[asset name] | [Purchase/Sale/Exchange] | [MM/DD/YY] | [MM/DD/YY] | [amount range letter A-J]

For each page with no real transactions, output PAGE followed by exactly NO_TRANSACTIONS. Every PDF page must appear exactly once and must contain transactions or NO_TRANSACTIONS.
Amount ranges: A=$1K-15K, B=$15K-50K, C=$50K-100K, D=$100K-250K, E=$250K-500K, F=$500K-1M, G=$1M-5M, H=$5M-25M, I=$25M-50M, J=over $50M.
No markdown, no tables, no explanations."""
PROMPT_SHA256 = hashlib.sha256(PROMPT.encode()).hexdigest()


class GeminiOutputError(ValueError):
    """The model response does not conform to the extraction schema."""


@dataclass(frozen=True)
class ParsedGeminiOutput:
    member: str
    transactions: list[dict]
    raw_row_count: int
    no_transactions: bool
    page_count: int
    covered_pages: frozenset[int]


def _strict_date(value: object) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        return None
    text = value.strip()
    for fmt in ("%m/%d/%y", "%m/%d/%Y", "%m-%d-%y", "%m-%d-%Y"):
        try:
            parsed = datetime.strptime(text, fmt)
        except ValueError:
            continue
        if parsed.strftime(fmt) == text:
            return parsed.date()
    return None


def _normalize_tx_type(value: object) -> str | None:
    text = str(value or "").strip().lower()
    if text in {"p", "purchase"}:
        return "Purchase"
    if text in {"s", "sale", "partial sale"}:
        return "Sale"
    if text in {"e", "exchange"}:
        return "Exchange"
    return None


def _is_example(asset: str) -> bool:
    normalized = " ".join(asset.casefold().split()).rstrip(".")
    return normalized == "example: mega corp. common stock"


def parse_gemini_output(
    output: str, *, expected_page_count: int | None = None
) -> ParsedGeminiOutput:
    """Parse a complete response and require an outcome for every PDF page."""
    if not isinstance(output, str) or not output.strip():
        raise GeminiOutputError("empty_response")

    member: str | None = None
    declared_page_count: int | None = None
    current_page: int | None = None
    transactions: list[dict] = []
    raw_row_count = 0
    declared_pages: set[int] = set()
    page_has_rows: set[int] = set()
    page_has_no_transactions: set[int] = set()

    for line_number, raw_line in enumerate(output.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if line.upper().startswith("MEMBER:"):
            candidate = line.split(":", 1)[1].strip()
            if not candidate:
                raise GeminiOutputError(f"line {line_number}: empty member")
            if member is not None:
                raise GeminiOutputError(f"line {line_number}: duplicate member")
            member = candidate
            continue
        if line.upper().startswith("PAGES:"):
            if declared_page_count is not None:
                raise GeminiOutputError(f"line {line_number}: duplicate PAGES")
            try:
                declared_page_count = int(line.split(":", 1)[1].strip())
            except ValueError as exc:
                raise GeminiOutputError(
                    f"line {line_number}: invalid page count"
                ) from exc
            if declared_page_count <= 0:
                raise GeminiOutputError(f"line {line_number}: invalid page count")
            continue
        if line.upper().startswith("PAGE:"):
            if declared_page_count is None:
                raise GeminiOutputError(f"line {line_number}: PAGE before PAGES")
            try:
                current_page = int(line.split(":", 1)[1].strip())
            except ValueError as exc:
                raise GeminiOutputError(
                    f"line {line_number}: invalid page number"
                ) from exc
            if not 1 <= current_page <= declared_page_count:
                raise GeminiOutputError(f"line {line_number}: page out of range")
            if current_page in declared_pages:
                raise GeminiOutputError(f"line {line_number}: duplicate page")
            declared_pages.add(current_page)
            continue
        if line == "NO_TRANSACTIONS":
            if current_page is None:
                raise GeminiOutputError(
                    f"line {line_number}: NO_TRANSACTIONS before PAGE"
                )
            if current_page in page_has_rows:
                raise GeminiOutputError(f"line {line_number}: mixed page outcome")
            page_has_no_transactions.add(current_page)
            current_page = None
            continue
        if "|" not in line:
            raise GeminiOutputError(f"line {line_number}: unexpected text")
        if current_page is None:
            raise GeminiOutputError(f"line {line_number}: transaction before PAGE")
        if current_page in page_has_no_transactions:
            raise GeminiOutputError(f"line {line_number}: mixed page outcome")

        parts = [part.strip() for part in line.split("|")]
        if len(parts) != 5:
            raise GeminiOutputError(f"line {line_number}: expected 5 fields")
        asset, tx_type_raw, tx_date_raw, notification_date_raw, amount_raw = parts
        if _is_example(asset):
            continue

        raw_row_count += 1
        if not asset:
            raise GeminiOutputError(f"line {line_number}: empty asset")
        tx_type = _normalize_tx_type(tx_type_raw)
        if tx_type is None:
            raise GeminiOutputError(f"line {line_number}: invalid transaction type")
        if _strict_date(tx_date_raw) is None:
            raise GeminiOutputError(f"line {line_number}: invalid transaction date")
        if _strict_date(notification_date_raw) is None:
            raise GeminiOutputError(f"line {line_number}: invalid notification date")
        amount_letter = amount_raw.upper()
        if amount_letter not in AMOUNT_MIDPOINTS:
            raise GeminiOutputError(f"line {line_number}: invalid amount range")
        transactions.append(
            {
                "asset": asset,
                "type": tx_type,
                "date": tx_date_raw,
                "notif_date": notification_date_raw,
                "amount_letter": amount_letter,
                "amount_midpoint": AMOUNT_MIDPOINTS[amount_letter],
                "page_number": current_page,
            }
        )
        page_has_rows.add(current_page)

    if member is None:
        raise GeminiOutputError("missing member")
    if declared_page_count is None:
        raise GeminiOutputError("missing PAGES")
    if expected_page_count is not None and declared_page_count != expected_page_count:
        raise GeminiOutputError(
            f"page count mismatch: response={declared_page_count}, pdf={expected_page_count}"
        )
    covered_pages = page_has_rows | page_has_no_transactions
    expected_pages = set(range(1, declared_page_count + 1))
    if covered_pages != expected_pages:
        missing = sorted(expected_pages - covered_pages)
        raise GeminiOutputError(f"missing page outcomes: {missing}")
    return ParsedGeminiOutput(
        member,
        transactions,
        raw_row_count,
        not transactions,
        declared_page_count,
        frozenset(covered_pages),
    )


def cache_path(doc_id: str, cache_dir: str = CACHE_DIR) -> Path:
    safe_doc_id = str(doc_id).replace(os.sep, "_").replace("/", "_").replace("\\", "_")
    return Path(cache_dir) / f"{safe_doc_id}.json"


def pdf_sha256(pdf_path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(pdf_path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pdf_page_count(pdf_path: str | Path) -> int:
    result = subprocess.run(
        ["pdfinfo", str(pdf_path)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise OSError(result.stderr.strip() or f"pdfinfo exited {result.returncode}")
    for line in result.stdout.splitlines():
        if line.startswith("Pages:"):
            count = int(line.split(":", 1)[1].strip())
            if count > 0:
                return count
    raise OSError("pdfinfo did not report a positive page count")


@dataclass(frozen=True)
class PdfSnapshot:
    path: Path
    sha256: str
    page_count: int
    byte_count: int


@dataclass(frozen=True)
class ArtifactMetadata:
    sha256: str
    page_count: int
    byte_count: int
    source_path: str


@dataclass(frozen=True)
class CachedGeminiResponse:
    output: str
    parsed: ParsedGeminiOutput
    pdf_sha256: str
    pdf_page_count: int


@contextmanager
def snapshot_pdf(pdf_path: str | Path):
    """Read source bytes once and expose one immutable file to all subprocesses."""
    source_bytes = Path(pdf_path).read_bytes()
    digest = hashlib.sha256(source_bytes).hexdigest()
    with tempfile.TemporaryDirectory(prefix="ptr_ocr_snapshot_") as directory:
        snapshot_path = Path(directory) / "artifact.pdf"
        snapshot_path.write_bytes(source_bytes)
        snapshot_path.chmod(0o444)
        snapshot = PdfSnapshot(
            snapshot_path, digest, pdf_page_count(snapshot_path), len(source_bytes)
        )
        yield snapshot


def _cache_envelope(
    doc_id: str,
    pdf_digest: str,
    page_count: int,
    output: str,
    parser_version: str,
) -> dict:
    return {
        "cache_envelope_version": CACHE_ENVELOPE_VERSION,
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "doc_id": str(doc_id),
        "pdf_sha256": pdf_digest,
        "pdf_page_count": page_count,
        "model": MODEL,
        "prompt_sha256": PROMPT_SHA256,
        "parser_version": parser_version,
        "output": output,
    }


def _read_cached_snapshot(
    doc_id: str,
    snapshot: PdfSnapshot,
    cache_dir: str,
    parser_version: str,
) -> str | None:
    path = cache_path(doc_id, cache_dir)
    try:
        envelope = json.loads(path.read_text(encoding="utf-8"))
        output = str(envelope.get("output", ""))
        expected = _cache_envelope(
            str(doc_id),
            snapshot.sha256,
            snapshot.page_count,
            output,
            parser_version,
        )
        if envelope != expected:
            return None
        parse_gemini_output(output, expected_page_count=snapshot.page_count)
        return output
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
        GeminiOutputError,
    ):
        return None


def inspect_cached_response(
    doc_id: str,
    pdf_path: str | Path,
    cache_dir: str = CACHE_DIR,
    parser_version: str = GEMINI_PARSER_VERSION,
) -> CachedGeminiResponse | None:
    """Validate cache and return its parser/artifact identity without model I/O."""
    with snapshot_pdf(pdf_path) as snapshot:
        output = _read_cached_snapshot(str(doc_id), snapshot, cache_dir, parser_version)
        if output is None:
            return None
        parsed = parse_gemini_output(output, expected_page_count=snapshot.page_count)
        return CachedGeminiResponse(
            output, parsed, snapshot.sha256, snapshot.page_count
        )


def read_cached_response(
    doc_id: str,
    pdf_path: str | Path,
    cache_dir: str = CACHE_DIR,
    parser_version: str = GEMINI_PARSER_VERSION,
) -> str | None:
    """Return a valid cache entry bound to one immutable PDF snapshot."""
    cached = inspect_cached_response(doc_id, pdf_path, cache_dir, parser_version)
    return cached.output if cached is not None else None


def _write_cached_snapshot(
    doc_id: str,
    snapshot: PdfSnapshot,
    output: str,
    cache_dir: str,
    parser_version: str,
) -> None:
    parse_gemini_output(output, expected_page_count=snapshot.page_count)
    path = cache_path(doc_id, cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    envelope = _cache_envelope(
        str(doc_id), snapshot.sha256, snapshot.page_count, output, parser_version
    )
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(envelope, handle, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def write_cached_response(
    doc_id: str,
    pdf_path: str | Path,
    output: str,
    cache_dir: str = CACHE_DIR,
    parser_version: str = GEMINI_PARSER_VERSION,
) -> None:
    """Atomically persist output bound to one immutable PDF snapshot."""
    with snapshot_pdf(pdf_path) as snapshot:
        _write_cached_snapshot(str(doc_id), snapshot, output, cache_dir, parser_version)


def call_gemini(
    pdf_path: str,
    doc_id: str | None = None,
    refresh: bool = False,
    cache_dir: str = CACHE_DIR,
    timeout: int = 180,
    parser_version: str = GEMINI_PARSER_VERSION,
) -> tuple[str | None, str, ArtifactMetadata | None]:
    """Call Gemini against the same immutable bytes used for hash/cache checks."""
    try:
        with snapshot_pdf(pdf_path) as snapshot:
            metadata = ArtifactMetadata(
                snapshot.sha256,
                snapshot.page_count,
                snapshot.byte_count,
                str(Path(pdf_path)),
            )
            if doc_id and not refresh:
                cached = _read_cached_snapshot(
                    str(doc_id), snapshot, cache_dir, parser_version
                )
                if cached is not None:
                    return cached, "", metadata
            result = subprocess.run(
                [
                    "llm",
                    "-m",
                    MODEL,
                    "-a",
                    str(snapshot.path),
                    "-o",
                    "temperature",
                    "0",
                    PROMPT,
                ],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if result.returncode != 0:
                return (
                    None,
                    result.stderr.strip() or f"llm exited {result.returncode}",
                    metadata,
                )
            try:
                parse_gemini_output(
                    result.stdout, expected_page_count=snapshot.page_count
                )
            except GeminiOutputError as exc:
                return None, f"invalid_response: {exc}", metadata
            if doc_id:
                _write_cached_snapshot(
                    str(doc_id), snapshot, result.stdout, cache_dir, parser_version
                )
            return result.stdout, "", metadata
    except subprocess.TimeoutExpired:
        return None, "llm timed out", None
    except Exception as exc:
        return None, str(exc), None


def _tx_date(tx: dict):
    return tx.get("date") or tx.get("tx_date") or tx.get("transaction_date")


def _tx_type(tx: dict):
    return tx.get("type") or tx.get("transaction_type")


def _tx_amount(tx: dict):
    return tx.get("amount_letter") or tx.get("amount") or tx.get("amount_raw")


def _tx_asset(tx: dict):
    return tx.get("asset") or tx.get("asset_description") or ""


def _tx_notification_date(tx: dict):
    return (
        tx.get("notif_date") or tx.get("notification_date") or tx.get("disclosure_date")
    )


def validate_transactions(doc_id, member, transactions, filing_date, expected_member):
    """Validate OCR rows and return ``(valid_transactions, rejection_counts)``."""
    del doc_id
    rejections = defaultdict(int)
    raw_count = len(transactions)
    filing = _strict_date(filing_date)
    start = filing - timedelta(days=400) if filing else None
    end = filing + timedelta(days=7) if filing else None
    if not str(member or "").strip() and not str(expected_member or "").strip():
        return [], {"invalid_member": raw_count or 1}

    effective_member = str(member or expected_member).strip()
    if expected_member and canonical_member_key(member or "") != canonical_member_key(
        expected_member
    ):
        effective_member = str(expected_member).strip()
        rejections["member_mismatch"] += 1

    valid = []
    for tx in transactions:
        asset = str(_tx_asset(tx) or "").strip()
        if not asset or _is_example(asset):
            rejections["invalid_asset"] += 1
            continue
        tx_type = _normalize_tx_type(_tx_type(tx))
        if tx_type is None:
            rejections["invalid_transaction_type"] += 1
            continue
        parsed_date = _strict_date(_tx_date(tx))
        if parsed_date is None:
            rejections["invalid_transaction_date"] += 1
            continue
        notification_date = _strict_date(_tx_notification_date(tx))
        if notification_date is None:
            rejections["invalid_notification_date"] += 1
            continue
        amount_letter = str(_tx_amount(tx) or "").strip().upper()
        if amount_letter not in AMOUNT_MIDPOINTS:
            rejections["invalid_amount"] += 1
            continue
        if filing and start and end and (parsed_date < start or parsed_date > end):
            rejections["date_out_of_window"] += 1
            continue
        cleaned = dict(tx)
        cleaned.update(
            member=effective_member,
            asset=asset,
            type=tx_type,
            amount_letter=amount_letter,
            amount_midpoint=AMOUNT_MIDPOINTS[amount_letter],
        )
        valid.append(cleaned)

    return valid, dict(rejections)
