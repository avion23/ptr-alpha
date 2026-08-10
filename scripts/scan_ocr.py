#!/usr/bin/env python3
"""Local Tesseract OCR sweep over scanned House PTR PDFs.

Standalone fallback for scanned (image-only) House PTR filings.  No paid
APIs: every page is rasterized locally and read with Tesseract.

Output is staged under ``.staging/ocr/<generation>/`` for the staged
rebuild pipeline (luna-rebuild):

  manifest.json                 generation manifest (per-doc status, rows,
                                artifact SHAs, explicit unresolved list)
  rows/<doc_id>.jsonl           one JSON object per OCR row
  docs/<doc_id>.json            per-doc envelope (status, sha, coverage)

Row provenance: ``source_row_id`` = ``<doc_id>:page:<N>:row:<M>`` where
``N`` is the 1-based PDF page and ``M`` the 1-based row index across the
document.  Every staged row carries ``artifact_sha256`` (SHA-256 of the
source PDF bytes) and ``ingestion_generation``.

Fail-closed rules:

  * A document is ``resolved`` only when every PDF page produced at least
    one transaction row and the document has at least one strictly
    parseable transaction date.
  * Ground-truth canaries are enforced for pinned scan hashes; a canary
    document that misses its expected count / asset / date is demoted to
    ``unresolved`` with the exact discrepancy recorded.
  * OCR or rasterizer failures are recorded as unresolved, never promoted
    to success.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

GENERATION = "gen-live-20260809"
SCRIPT_VERSION = "1.0.0"
ENGINE = "local_tesseract"
SOURCE = "local_tesseract"
CHAMBER = "house"
RENDER_DPI = 300

# Ground-truth canaries for pinned 2026 House scans: (expected_row_count,
# asset_fragment, normalized txn date MM/DD/YY).  The hashes pin the exact
# artifacts these expectations were derived from (same set as
# tests/test_parsing.py scan_hashes / rebuild PINNED_SCAN_HASHES).
CANARY_TRUTH = {
    "9115808": (1, "spdr", "03/31/26"),
    "9115813": (9, "richmond", "04/15/26"),
    "9116141": (134, "whittier", "05/11/26"),
}
# Scans that are required to FAIL CLOSED (their canary is the unresolved
# state itself; e.g. 8221322 must stay in the unresolved work list).
CANARY_MUST_BE_UNRESOLVED = {"8221322"}

# Strict transaction-date token: M/D/YYYY or M/D/YY with / or - separators.
_STRICT_DATE_RE = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")
# Garbled-date residue: letters fused with a 4-digit year (e.g. "Jfae2026",
# "asi2026") that OCR produced from a printed MM/DD/YYYY cell.
_RESIDUE_DATE_RE = re.compile(r"(?<![0-9A-Za-z])[A-Za-z]{1,8}(?:19|20)\d{2}")
_EXAMPLE_RE = re.compile(r"example|mega corp", re.IGNORECASE)
_LEADING_MARK_RE = re.compile(
    r"^(?:[\s\-|_\[\](){}<>•·▪☐□■xX*~'\"`.,;:]+"
    r"|s\s*p|b\s*p)(?=\s|[|.:\]\-])",
    re.IGNORECASE,
)
_SP_RE = re.compile(r"^\s*(?:s\s*p|s\s*[xXp]|sp)", re.IGNORECASE)
_BP_RE = re.compile(r"^\s*(?:b\s*p|b\s*[xXp]|bp)", re.IGNORECASE)

TXN_TYPES = {"Purchase": "Purchase", "Sale": "Sale", "Exchange": "Exchange"}


def _run_tesseract(image_path: str | Path, args: list[str]) -> subprocess.CompletedProcess:
    """Run tesseract with cwd=dirname so child processes can open the file
    regardless of sandbox path remapping."""
    image_path = Path(image_path)
    return subprocess.run(
        ["tesseract", image_path.name, *args],
        cwd=str(image_path.parent),
        capture_output=True,
        timeout=600,
        check=False,
    )


def tesseract_lines(image_path: str | Path, psm: int = 3) -> list[tuple[int, str]]:
    """Return (approx_y, text) lines for a tesseract psm pass.

    Lines are reconstructed from the TSV word stream so each line keeps its
    vertical position (needed for sparse-pass asset recovery).  Falls back to
    the plain-text pass when TSV yields nothing.
    """
    words = tesseract_words(image_path, psm=psm)
    if words:
        groups: dict[tuple, list[dict]] = {}
        for word in words:
            key = (
                word["block_num"],
                word["par_num"],
                word["line_num"],
            )
            groups.setdefault(key, []).append(word)
        lines: list[tuple[int, str]] = []
        for key, group in groups.items():
            group.sort(key=lambda w: w["left"])
            y = int(sum(w["top"] for w in group) / len(group))
            text = " ".join(w["text"] for w in group if w["text"].strip())
            text = re.sub(r"\s+", " ", text).strip()
            if text:
                lines.append((y, text))
        lines.sort(key=lambda item: item[0])
        return lines
    result = _run_tesseract(image_path, ["-", "--psm", str(psm)])
    if result.returncode != 0:
        return []
    return [
        (0, raw.strip())
        for raw in result.stdout.decode("utf-8", errors="replace").splitlines()
        if raw.strip()
    ]


def tesseract_words(image_path: str | Path, psm: int = 3) -> list[dict]:
    """Return word-level TSV rows: {left, top, width, height, text}."""
    result = _run_tesseract(image_path, ["-", "--psm", str(psm), "tsv"])
    if result.returncode != 0:
        return []
    words: list[dict] = []
    header = None
    for line in result.stdout.decode("utf-8", errors="replace").splitlines():
        parts = line.split("\t")
        if header is None:
            header = parts
            continue
        if len(parts) < 12:
            continue
        entry = dict(zip(header, parts))
        text = entry.get("text", "").strip()
        if not text:
            continue
        try:
            entry["left"] = int(entry["left"])
            entry["top"] = int(entry["top"])
            entry["width"] = int(entry["width"])
            entry["height"] = int(entry["height"])
        except ValueError:
            continue
        words.append(entry)
    return words


def tesseract_plain_lines(image_path: str | Path, psm: int = 3) -> list[str]:
    """Plain-text pass lines (tesseract's own reconstruction, top-to-bottom).

    The plain pass keeps OCR'd dates intact ("3-31-26") where the TSV word
    stream fragments them, so it is the row-detection source of truth.
    """
    result = _run_tesseract(image_path, ["-", "--psm", str(psm)])
    if result.returncode != 0:
        return []
    return [
        raw.strip()
        for raw in result.stdout.decode("utf-8", errors="replace").splitlines()
        if raw.strip()
    ]


def _best_line_y(plain_line: str, tsv_lines: list[tuple[int, str]]) -> int | None:
    """Adopt the y of the TSV line sharing the most significant tokens."""
    plain_words = {
        w.casefold()
        for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9.-]{1,}", plain_line)
        if len(w) >= 2
    }
    best_y, best_score = None, 0
    for y, tsv_text in tsv_lines:
        tsv_words = {
            w.casefold()
            for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9.-]{1,}", tsv_text)
            if len(w) >= 2
        }
        score = len(plain_words & tsv_words)
        if score > best_score:
            best_score, best_y = score, y
    return best_y


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pdf_page_count(pdf_path: str | Path) -> int:
    result = subprocess.run(
        ["pdfinfo", str(pdf_path)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    for line in result.stdout.splitlines():
        if line.startswith("Pages:"):
            count = int(line.split(":", 1)[1].strip())
            if count > 0:
                return count
    raise RuntimeError(f"pdfinfo did not report a page count for {pdf_path}")


def is_scanned(pdf_path: str | Path) -> bool:
    """A PDF is scanned when no text layer is extractable."""
    result = subprocess.run(
        ["pdftotext", str(pdf_path), "-"],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    return not result.stdout.strip()


def _year_in_range(year_str: str) -> bool:
    if len(year_str) == 4:
        return year_str in {"2024", "2025", "2026"}
    return year_str in {"24", "25", "26"}


def _strict_dates(text: str) -> list[str]:
    out = []
    for match in _STRICT_DATE_RE.finditer(text):
        year = match.group(0).rsplit("/", 1)[-1].rsplit("-", 1)[-1]
        if _year_in_range(year):
            out.append(match.group(0))
    return out


def _has_residue_date(text: str) -> bool:
    return bool(_RESIDUE_DATE_RE.search(text))


def normalize_date(raw: str) -> str | None:
    """Normalize an OCR'd date to zero-padded MM/DD/YY.

    Accepts M/D/YY, M/D/YYYY with '/' or '-' separators; returns None for
    anything that is not a plausible transaction date (year 24-26).
    """
    raw = (raw or "").strip()
    match = _STRICT_DATE_RE.search(raw)
    if not match:
        return None
    token = match.group(0)
    parts = re.split(r"[/-]", token)
    if len(parts) != 3:
        return None
    month, day, year = parts
    if not _year_in_range(year):
        return None
    try:
        month_i, day_i = int(month), int(day)
        if not (1 <= month_i <= 12 and 1 <= day_i <= 31):
            return None
    except ValueError:
        return None
    yy = year[-2:]
    return f"{month_i:02d}/{day_i:02d}/{yy}"


def to_iso_date(raw: str) -> str | None:
    normalized = normalize_date(raw)
    if not normalized:
        return None
    month, day, year = normalized.split("/")
    return f"20{year}-{month}-{day}"


def _classify_type(line: str) -> str | None:
    if _SP_RE.match(line):
        return "Sale"
    if _BP_RE.match(line):
        return "Purchase"
    lowered = line.casefold()
    if re.search(r"\bbuy\b", lowered):
        return "Purchase"
    if re.search(r"\bsell\b|\bsale\b|\bsold\b", lowered):
        return "Sale"
    if re.search(r"\bexchange\b", lowered):
        return "Exchange"
    return None


def _strip_leading_marks(text: str) -> str:
    cleaned = _LEADING_MARK_RE.sub("", text)
    return re.sub(r"\s+", " ", cleaned).strip()


_TYPE_WORD_RE = re.compile(
    r"\b(?:buy|bay|sell|sold|sodr|sale|purchase|partial|exchange|"
    r"redemption|transfer)\b",
    re.IGNORECASE,
)


def _extract_asset(line: str, txn_date: str | None) -> str:
    """Asset = the portion of the row line before the transaction date."""
    if txn_date:
        idx = line.find(txn_date)
        if idx > 0:
            head = line[:idx]
        else:
            head = line
    else:
        head = line
    # Drop trailing residue/checkbox noise (e.g. " Jfae2026", " xX", " |").
    head = _RESIDUE_DATE_RE.sub(" ", head)
    head = re.sub(r"\b(?:x+|xx)\b", " ", head, flags=re.IGNORECASE)
    head = re.sub(r"[\|\[\]{}()<>_~^=]", " ", head)
    head = _TYPE_WORD_RE.sub(" ", head)
    return _strip_leading_marks(head)


@dataclass
class OcrRow:
    page_number: int
    row_index: int  # 1-based, document-wide
    asset_description: str
    transaction_type: str | None
    transaction_date_raw: str | None
    notification_date_raw: str | None
    amount_raw: str | None = None
    amount_midpoint: float | None = None
    owner_code: str | None = None
    date_unresolved: bool = False

    def transaction_date(self) -> str | None:
        return normalize_date(self.transaction_date_raw or "")

    def notification_date(self) -> str | None:
        return normalize_date(self.notification_date_raw or "")


@dataclass
class PageResult:
    page_number: int
    rows: list[OcrRow]
    note: str | None = None
    text: str = ""
    lines: list[str] = field(default_factory=list)



def _merge_asset_words(
    base_asset: str, page_words: list[dict], line_y: int | None, date_x: int | None
) -> str:
    """Use sparse (psm 11) word tokens near the row line to recover a garbled
    asset when the psm 3 line yields no usable asset text.

    The base asset is trusted when it contains at least one word of length
    >= 4; otherwise (e.g. "Buy St Str IL IIL") the sparse pass is consulted.
    """
    if base_asset and re.search(r"[A-Za-z]{4,}", base_asset):
        return base_asset
    nearby = [
        w
        for w in page_words
        if (line_y is None or abs(w["top"] - line_y) <= 120)
        and (date_x is None or w["left"] + w["width"] < date_x)
    ]
    nearby.sort(key=lambda w: w["left"])
    tokens = [
        w["text"] for w in nearby if re.search(r"[A-Za-z]", w["text"])
    ]
    candidate = _strip_leading_marks(" ".join(tokens))
    candidate = _TYPE_WORD_RE.sub(" ", candidate)
    candidate = re.sub(r"\s+", " ", candidate).strip()
    if re.search(r"[A-Za-z]{4,}", candidate):
        return candidate
    return base_asset


def _parse_row_line(
    line: str,
    *,
    page_number: int,
    row_index: int,
    page_words: list[dict] | None = None,
    line_y: int | None = None,
) -> OcrRow:
    strict = _strict_dates(line)
    txn_date_raw = strict[0] if strict else None
    notif_date_raw = strict[1] if len(strict) > 1 else None
    asset = _extract_asset(line, txn_date_raw)
    # The sparse-pass merge is constrained by the row's vertical band only;
    # a character index cannot serve as a pixel x cutoff.
    if page_words is not None and line_y is not None:
        asset = _merge_asset_words(asset, page_words, line_y, None)
    tx_type = _classify_type(line)
    row = OcrRow(
        page_number=page_number,
        row_index=row_index,
        asset_description=asset or "",
        transaction_type=tx_type,
        transaction_date_raw=txn_date_raw,
        notification_date_raw=notif_date_raw,
    )
    if not strict:
        row.date_unresolved = True
    return row


def _row_key(row: OcrRow) -> tuple:
    """Identity for cross-variant row dedup."""
    return (
        re.sub(r"[^a-z0-9]+", "", (row.asset_description or "").casefold()),
        normalize_date(row.transaction_date_raw or "") or "",
        normalize_date(row.notification_date_raw or "") or "",
    )


def _merge_page_results(primary: PageResult, secondary: PageResult) -> PageResult:
    """Use the resolution-estimated variant only when the dpi-pinned variant
    produced no usable rows.

    Tesseract's resolution estimate (PNG pHYs absent) shifts its binarization:
    some faint date cells read cleanly there but fragment in the dpi-pinned
    pass.  The estimated pass is far noisier overall, so its rows are trusted
    only as a fill-in for pages where the primary pass found nothing usable.
    """
    if not secondary.rows:
        return primary
    has_usable_primary = any(not row.date_unresolved for row in primary.rows)
    if primary.rows and has_usable_primary:
        return primary
    if primary.rows:
        # Primary rows exist but none carries a strict date: keep them, then
        # append only strict-dated rows from the estimated pass.
        seen = {_row_key(row) for row in primary.rows}
        added = [
            row
            for row in secondary.rows
            if not row.date_unresolved and _row_key(row) not in seen
        ]
        return PageResult(
            page_number=primary.page_number,
            rows=[*primary.rows, *added],
            text=primary.text or secondary.text,
            lines=primary.lines or secondary.lines,
        )
    return PageResult(
        page_number=primary.page_number,
        rows=secondary.rows,
        note=primary.note,
        text=primary.text or secondary.text,
        lines=primary.lines or secondary.lines,
    )


def extract_page_rows(
    image_path: str | Path, page_number: int, start_row_index: int
) -> PageResult:
    """OCR one rasterized page and return detected transaction rows."""
    plain_lines = tesseract_plain_lines(image_path, psm=3)
    tsv_lines = tesseract_lines(image_path, psm=3)
    words = tesseract_words(image_path, psm=3)
    sparse_words = tesseract_words(image_path, psm=11)
    if not plain_lines:
        return PageResult(page_number, [], "no OCR text")

    # Order lines visually (tesseract plain output can interleave blocks).
    ordered: list[tuple[int | None, str]] = []
    for line in plain_lines:
        ordered.append((_best_line_y(line, tsv_lines), line))
    ordered.sort(key=lambda item: (item[0] is None, item[0] or 0))

    rows: list[OcrRow] = []
    index = start_row_index
    for line_y, line in ordered:
        strict = _strict_dates(line)
        is_row = bool(strict) or _has_residue_date(line)
        is_example = bool(_EXAMPLE_RE.search(line))
        if is_example:
            continue
        if not is_row:
            continue
        row = _parse_row_line(
            line,
            page_number=page_number,
            row_index=index,
            page_words=sparse_words or words,
            line_y=line_y,
        )
        rows.append(row)
        index += 1
    page_text = " ".join(plain_lines).casefold()
    if not rows:
        return PageResult(
            page_number, [], "no transaction rows", page_text, plain_lines
        )
    return PageResult(page_number, rows, text=page_text, lines=plain_lines)




def sweep_document(pdf_path: str | Path) -> dict:
    """OCR a scanned PDF into per-page rows with fail-closed coverage."""
    pdf_path = Path(pdf_path)
    doc_id = pdf_path.stem
    artifact_sha256 = sha256_file(pdf_path)
    page_count = pdf_page_count(pdf_path)
    start = time.time()
    rows: list[OcrRow] = []
    uncovered: list[int] = []
    notes: list[str] = []
    no_tx_pages: list[int] = []
    cover_pages: list[int] = []
    with tempfile.TemporaryDirectory(prefix="scan_ocr_") as tmp:
        from pdf2image import convert_from_path

        images = convert_from_path(str(pdf_path), dpi=RENDER_DPI)
        for page_number, image in enumerate(images, start=1):
            image_path = Path(tmp) / f"page_{page_number:03d}.png"
            # Variant A: PNG carries the render dpi (poppler-identical OCR).
            image.save(str(image_path), dpi=(RENDER_DPI, RENDER_DPI))
            result = extract_page_rows(
                image_path, page_number, start_row_index=len(rows) + 1
            )
            # Variant B: no pHYs -> tesseract estimates resolution and uses a
            # different binarization; recovers faint date cells variant A
            # fragments.  Strict-dated rows are merged without duplication.
            estimated_path = Path(tmp) / f"page_{page_number:03d}_est.png"
            image.save(str(estimated_path))
            secondary = extract_page_rows(
                estimated_path, page_number, start_row_index=len(rows) + 1
            )
            result = _merge_page_results(result, secondary)
            rows.extend(result.rows)
            page_text = result.text or secondary.text or ""
            if not result.rows:
                _classify_empty_page(
                    page_number,
                    page_text,
                    result.lines or secondary.lines,
                    uncovered=uncovered,
                    no_tx_pages=no_tx_pages,
                    cover_pages=cover_pages,
                    notes=notes,
                    note=result.note,
                )
    strict_dated = sum(1 for r in rows if not r.date_unresolved)
    resolved = (
        not uncovered and strict_dated >= 1 and bool(rows)
    )
    reasons = list(notes)
    if not rows:
        reasons.append("no transaction rows recovered")
    elif uncovered:
        reasons.append(
            f"pages without transaction rows: {','.join(map(str, uncovered))}"
        )
    if strict_dated < 1 and rows:
        reasons.append("no strictly parseable transaction date")
    no_transactions = (
        not rows
        and not uncovered
        and bool(no_tx_pages)
        and all(
            p in no_tx_pages or p in cover_pages
            for p in range(1, page_count + 1)
        )
    )
    if no_transactions:
        reasons.append(
            "filing reports no transactions "
            f"(nothing-to-report pages: {','.join(map(str, no_tx_pages))})"
        )
    status = (
        "no_transactions"
        if no_transactions
        else ("resolved" if resolved else "unresolved")
    )
    return {
        "doc_id": doc_id,
        "artifact_sha256": artifact_sha256,
        "page_count": page_count,
        "rows": rows,
        "resolved": resolved,
        "status": status,
        "reasons": reasons,
        "uncovered_pages": uncovered,
        "cover_pages": cover_pages,
        "no_tx_pages": no_tx_pages,
        "elapsed_seconds": round(time.time() - start, 1),
    }


_ROW_LIKE_LINE_RE = re.compile(
    r"^\s*(?:sp|pc|sb|s[px]|x+|xx|bp)[\s|\]\[.}:_-]*", re.IGNORECASE
)
_INSTRUCTION_LINE_RE = re.compile(
    r"provide full name|not ticker symbol|initial report|amendment",
    re.IGNORECASE,
)


def _page_has_row_like_content(lines: list[str]) -> bool:
    """True when a row-less page carries unparsed transaction-row marks
    (leading checkbox mark followed by asset words), which indicates failed
    OCR rather than a legitimately empty page."""
    for line in lines:
        if _INSTRUCTION_LINE_RE.search(line):
            continue
        if _ROW_LIKE_LINE_RE.match(line):
            words = re.findall(r"[A-Za-z]{3,}", line)
            if len(words) >= 2:
                return True
    return False


def _classify_empty_page(
    page_number: int,
    page_text: str,
    page_lines: list[str],
    *,
    uncovered: list[int],
    no_tx_pages: list[int],
    cover_pages: list[int],
    notes: list[str],
    note: str | None,
) -> None:
    """Classify a page that produced no rows: nothing-to-report page, cover
    page (filer block, no transaction content), or uncovered (fail-closed)."""
    text = page_text
    nothing = "nothing to report" in text
    has_form_header = (
        "periodic transaction report" in text
        or "united states house" in text
    )
    filer_block = (
        "office telephone" in text
        or "member of the u.s. house" in text
        or "please see the attached" in text
    )
    if nothing:
        no_tx_pages.append(page_number)
        notes.append(f"page {page_number}: reports no transactions")
        return
    if (
        has_form_header
        and filer_block
        and not _page_has_row_like_content(page_lines)
    ):
        cover_pages.append(page_number)
        notes.append(f"page {page_number}: cover page (no transaction rows)")
        return
    uncovered.append(page_number)
    if note:
        notes.append(f"page {page_number}: {note}")


def canary_result(doc_id: str, result: dict) -> dict:
    rows = result["rows"]
    check = {
        "doc_id": doc_id,
        "expected": None,
        "actual": None,
        "passed": None,
        "detail": "",
    }
    if doc_id in CANARY_MUST_BE_UNRESOLVED:
        check["expected"] = "unresolved"
        check["actual"] = "unresolved" if not result["resolved"] else "resolved"
        check["passed"] = not result["resolved"]
        if not check["passed"]:
            check["detail"] = "canary requires fail-closed unresolved state"
        return check
    expected = CANARY_TRUTH.get(doc_id)
    if expected is None:
        return check
    expected_count, asset_fragment, expected_date = expected
    check["expected"] = {
        "row_count": expected_count,
        "asset_fragment": asset_fragment,
        "transaction_date": expected_date,
    }
    check["actual"] = {
        "row_count": len(rows),
        "asset_fragment": next(
            (
                r.asset_description.casefold()
                for r in rows
                if asset_fragment in r.asset_description.casefold()
            ),
            None,
        ),
        "transaction_date": next(
            (r.transaction_date() for r in rows if r.transaction_date()), None
        ),
    }
    asset_hit = any(
        asset_fragment in r.asset_description.casefold() for r in rows
    )
    date_hit = any(r.transaction_date() == expected_date for r in rows)
    check["passed"] = (
        len(rows) == expected_count and asset_hit and date_hit
    )
    if not check["passed"]:
        check["detail"] = (
            f"expected count={expected_count} asset~'{asset_fragment}' "
            f"date={expected_date}; got count={len(rows)} "
            f"asset_hit={asset_hit} date_hit={date_hit}"
        )
    return check


def apply_canaries(doc_results: dict[str, dict]) -> dict[str, dict]:
    """Demote canary docs that miss ground truth to unresolved (fail-closed)."""
    for doc_id, result in doc_results.items():
        check = canary_result(doc_id, result)
        result["canary"] = check
        if (
            check["expected"] is not None
            and check["passed"] is False
            and (result["resolved"] or doc_id in CANARY_TRUTH)
        ):
            result["resolved"] = False
            result["status"] = "unresolved"
            result["reasons"].append(f"canary failed: {check['detail']}")
    return doc_results


# --------------------------------------------------------------------------
# Staging output
# --------------------------------------------------------------------------

ROW_FIELDS = (
    "source",
    "chamber",
    "doc_id",
    "source_record_id",
    "source_row_id",
    "page_number",
    "row_index",
    "asset_description",
    "transaction_type",
    "transaction_date",
    "transaction_date_raw",
    "notification_date",
    "notification_date_raw",
    "amount_raw",
    "amount_midpoint",
    "owner_code",
    "artifact_sha256",
    "ingestion_generation",
)


def row_to_dict(row: OcrRow, doc_id: str, artifact_sha256: str) -> dict:
    return {
        "source": SOURCE,
        "chamber": CHAMBER,
        "doc_id": doc_id,
        "source_record_id": doc_id,
        "source_row_id": (
            f"{doc_id}:page:{row.page_number}:row:{row.row_index}"
        ),
        "page_number": row.page_number,
        "row_index": row.row_index,
        "asset_description": row.asset_description[:500],
        "transaction_type": row.transaction_type,
        "transaction_date": to_iso_date(row.transaction_date_raw or ""),
        "transaction_date_raw": row.transaction_date_raw,
        "notification_date": to_iso_date(row.notification_date_raw or ""),
        "notification_date_raw": row.notification_date_raw,
        "amount_raw": row.amount_raw,
        "amount_midpoint": row.amount_midpoint,
        "owner_code": row.owner_code,
        "artifact_sha256": artifact_sha256,
        "ingestion_generation": GENERATION,
    }


def write_staging(
    out_root: str | Path,
    doc_results: dict[str, dict],
    *,
    data_dir: str | Path,
    year: int,
) -> Path:
    out_root = Path(out_root)
    rows_dir = out_root / "rows"
    docs_dir = out_root / "docs"
    rows_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    resolved: dict[str, dict] = {}
    unresolved: dict[str, list[str]] = {}
    total_rows = 0
    staged_files: dict[str, str] = {}

    for doc_id in sorted(doc_results):
        result = doc_results[doc_id]
        rows = result["rows"]
        total_rows += len(rows)

        rows_path = rows_dir / f"{doc_id}.jsonl"
        with rows_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(
                        row_to_dict(row, doc_id, result["artifact_sha256"]),
                        sort_keys=True,
                    )
                    + "\n"
                )
        staged_files[f"rows/{doc_id}.jsonl"] = sha256_file(rows_path)

        envelope = {
            "doc_id": doc_id,
            "status": result.get("status", "resolved" if result["resolved"] else "unresolved"),
            "artifact_sha256": result["artifact_sha256"],
            "page_count": result["page_count"],
            "covered_pages": [
                p
                for p in range(1, result["page_count"] + 1)
                if p not in result["uncovered_pages"]
            ],
            "uncovered_pages": result["uncovered_pages"],
            "row_count": len(rows),
            "rows_file": f"rows/{doc_id}.jsonl",
            "rows_file_sha256": staged_files[f"rows/{doc_id}.jsonl"],
            "canary": result.get("canary"),
            "reasons": result["reasons"],
            "engine": ENGINE,
            "parser_version": SCRIPT_VERSION,
            "extracted_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        doc_path = docs_dir / f"{doc_id}.json"
        doc_path.write_text(
            json.dumps(envelope, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        staged_files[f"docs/{doc_id}.json"] = sha256_file(doc_path)

        if result.get("status", "resolved" if result["resolved"] else "unresolved") == "resolved":
            resolved[doc_id] = {
                "row_count": len(rows),
                "pages_covered": result["page_count"] - len(result["uncovered_pages"]),
            }
        else:
            unresolved[doc_id] = result["reasons"]

    manifest = {
        "generation": GENERATION,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "engine": ENGINE,
        "tesseract_version": _tesseract_version(),
        "script": "scripts/scan_ocr.py",
        "script_version": SCRIPT_VERSION,
        "render_dpi": RENDER_DPI,
        "data_dir": str(data_dir),
        "year": year,
        "doc_count": len(doc_results),
        "resolved_count": len(resolved),
        "unresolved_count": len(unresolved),
        "total_rows": total_rows,
        "resolved": resolved,
        "unresolved": unresolved,
        "canary": {
            doc_id: doc_results[doc_id].get("canary")
            for doc_id in sorted(doc_results)
            if doc_results[doc_id].get("canary", {}).get("expected") is not None
        },
        "staged_files_sha256": staged_files,
    }
    manifest_path = out_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    staged_files["manifest.json"] = sha256_file(manifest_path)
    return out_root


def _tesseract_version() -> str:
    try:
        result = subprocess.run(
            ["tesseract", "--version"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        first = result.stdout.splitlines()[0] if result.stdout else ""
        return first.strip() or "unknown"
    except Exception:  # noqa: BLE001 -- best-effort version probe
        return "unknown"


def discover_scans(data_dir: str | Path, year: int) -> list[Path]:
    pdf_dir = Path(data_dir) / str(year) / "pdfs"
    if not pdf_dir.is_dir():
        raise RuntimeError(f"no PDF directory at {pdf_dir}")
    scans = []
    for pdf in sorted(pdf_dir.glob("*.pdf")):
        try:
            if is_scanned(pdf):
                scans.append(pdf)
        except Exception:  # noqa: BLE001 -- fail-closed: treat as scanned
            scans.append(pdf)
    return scans


def _sweep_worker(pdf_path: str) -> dict:
    return sweep_document(Path(pdf_path))


def run_sweep(
    data_dir: str | Path,
    year: int,
    *,
    workers: int = 1,
    doc_ids: list[str] | None = None,
) -> dict[str, dict]:
    scans = discover_scans(data_dir, year)
    if doc_ids:
        wanted = set(doc_ids)
        scans = [p for p in scans if p.stem in wanted]
    if not scans:
        raise RuntimeError(f"no scanned PDFs found for {year}")
    with Pool(workers) as pool:
        results = pool.map(_sweep_worker, [str(p) for p in scans])
    doc_results = {r["doc_id"]: r for r in results}
    return apply_canaries(doc_results)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--data-dir", default=str(REPO_ROOT / "data"),
                        help="repo data root containing <year>/pdfs")
    parser.add_argument("--out", default=str(REPO_ROOT / ".staging" / "ocr" / GENERATION))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--doc-ids", nargs="*", default=None,
                        help="restrict sweep to these doc ids (debug)")
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir)
    if not (data_dir / str(args.year) / "pdfs").is_dir():
        # Fall back to the main checkout when running from a worktree.
        main_root = Path(__file__).resolve().parents[2]
        candidate = main_root / "data"
        if (candidate / str(args.year) / "pdfs").is_dir():
            data_dir = candidate

    doc_results = run_sweep(
        data_dir, args.year, workers=args.workers, doc_ids=args.doc_ids
    )
    out_root = write_staging(args.out, doc_results, data_dir=data_dir, year=args.year)

    resolved = sum(
        1 for r in doc_results.values() if r.get("status", "resolved") == "resolved"
    )
    no_tx = sum(
        1 for r in doc_results.values() if r.get("status") == "no_transactions"
    )
    rows = sum(len(r["rows"]) for r in doc_results.values())
    print(f"scans: {len(doc_results)}  resolved: {resolved}  "
          f"no_transactions: {no_tx}  unresolved: {len(doc_results) - resolved - no_tx}  "
          f"rows: {rows}")
    for doc_id in sorted(doc_results):
        r = doc_results[doc_id]
        status = {
            "resolved": "resolved  ",
            "no_transactions": "no_txs    ",
        }.get(r.get("status", ""), "UNRESOLVED")
        canary = r.get("canary", {})
        canary_txt = ""
        if canary.get("expected") is not None:
            canary_txt = f" canary={'PASS' if canary['passed'] else 'FAIL'}"
        print(
            f"  {status} {doc_id} pages={r['page_count']} rows={len(r['rows'])}"
            f"{canary_txt}"
        )
        for reason in r["reasons"][:6]:
            print(f"      - {reason}")
    print(f"staged to {out_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
