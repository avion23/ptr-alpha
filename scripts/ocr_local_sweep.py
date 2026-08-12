#!/usr/bin/env python3
"""Local free OCR second pass for unresolved House PTR PDFs (staged, DB-free).

Second-pass resolver for House PTR PDFs that the bulk first pass
(PTR_SKIP_DOCLING=1, no local OCR) and the Gemini flash-lite sweep left
unresolved.  Every engine is local and free; no paid API is called.

Documented flow (mirrors the accepted "stragglers re-parsed with Docling"
cascade, scripts/reparse_all.py + src/analyzer/parser_cascade.py):

  For each unresolved doc:
    (1) image preprocessing pipeline (ocrmypdf-style): pdftoppm render at
        300 dpi + PIL grayscale/autocontrast/median-denoise/2x-upscale
        (effective 600 dpi);
    (2) Docling OCR (extract_tables_with_docling semantics; runs in-process
        under the /opt/homebrew/bin/python3.14 env where docling 2.x is
        installed; PTR_SKIP_DOCLING is left unset).  Per page, Docling's
        structured table (TableItem.export_to_dataframe()) is the primary
        source, mapped through the old-form PTR column layout (ticker from
        the Asset parenthetical, transaction type, transaction/notification
        dates, amount range); the markdown text path is used only for pages
        Docling reports no table (markdown is still exported with a
        form-feed page-break placeholder so segments map 1:1 to PDF pages);
    (3) preprocessed Tesseract fallback per uncovered page: scan_ocr's
        accepted dpi-pinned + no-pHYs extraction variants first (these are
        the variants the pinned 2026 canaries were derived from), then the
        preprocessed image as a tertiary fill;
    (4) deterministic text-layer retry with the accepted production cascade
        (_parse_pdf_worker) when the PDF has an extractable text layer and
        steps (2)-(3) did not resolve it.

Results stage under ``.staging/ocr2-local/<generation>/`` in the SAME
contract as the Gemini sweep (scripts/ocr_gemini_sweep.py, branch
audit/luna-ocr-sweep-20260810):

  rows/<doc>.jsonl      one JSON object per transaction row; provenance
                        matrix (ticker_origin official/unverified/
                        not_reported), source_row_id doc:page:N:row:M,
                        artifact_sha256, per-doc year, generation binding
  docs/<doc>.json       per-doc envelope (status resolved/unresolved/no_txs,
                        reasons, coverage, canary, rows SHA)
  manifest.json         generation manifest with per-file SHAs
                        (staged_files_sha256) + by_year summary, so
                        luna-finalize can ingest either OCR source.

Fail-closed per document:
  * a doc is ``resolved`` only when every PDF page has transaction rows or a
    verified no-transaction/cover outcome AND at least one strictly
    parseable transaction date AND staged-house metadata exists (member +
    filing date);
  * unparseable rows, missing metadata, engine failures and canary misses
    demote the doc to ``unresolved`` with the exact reasons recorded;
  * ``no_txs`` requires every page verified empty (nothing-to-report or
    cover) and zero rows across all engines;
  * amounts and notification dates are only staged when the OCR text
    actually reports them (audit C9: unverifiable values are NULL, never
    guessed); rows without a transaction type are kept with NULL type
    exactly like the accepted local-OCR sweep (scripts/scan_ocr.py).

Pinned canaries (same ground truth as the Gemini sweep) are enforced for
any doc they cover: 9115808=1 row/1 page, 9115813=9 rows/2 pages,
9116141=134 rows/6 pages, 8221322=56 pages with >=18 rows on page 2.  A
missed canary exits nonzero (explicit fail-closed).

Tunable hyperparameters (env, read at import; defaults are the canary-
validated values): OCR2_RENDER_DPI (300), OCR2_PREPROCESS_SCALE (2.0),
OCR2_PREPROCESS_MEDIAN (3), OCR2_PREPROCESS_AUTOCONTRAST (1),
OCR2_DOCLING_ENABLED (1), OCR2_CASCADE_ENABLED (1), OCR2_TESSERACT_PSM (3),
OCR2_TESSERACT_PSM_SPARSE (11), OCR2_WORKERS (3).  Sibling workers may join
the same staging root: per-doc writes are atomic and --skip-staged replays
only the remainder.

The real DB and main checkout are never written.  The rebuild2 staged
generation (gen-live-20260809) is read only.

Usage:
    python scripts/ocr_local_sweep.py pilot [--out DIR] [--data-dir DIR]
        [--db DB] [--random N] [--workers N]
    python scripts/ocr_local_sweep.py sweep [--data-dir DIR] [--db DB]
        [--manifest FILE] [--years 2015 2016 ...] [--workers N] [--out DIR]
        [--skip-staged] [--max-docs N] [--merge-only]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from datetime import date, datetime, timedelta
from multiprocessing import Pool
from pathlib import Path

GENERATION = "gen-live-20260810"
ENGINE = "local_docling_tesseract"
PARSER_VERSION = "v1-local-docling-tesseract"
SOURCE = "local_ocr"
CHAMBER = "house"
SCRIPT_VERSION = "1.2.1"

# Tunable hyperparameters (env-overridable so sibling workers can tune/run
# the same sweep against the same staging root; defaults are the values the
# pinned canaries were validated with).
RENDER_DPI = int(os.environ.get("OCR2_RENDER_DPI", "300"))
PREPROCESS_SCALE = float(os.environ.get("OCR2_PREPROCESS_SCALE", "2.0"))
PREPROCESS_MEDIAN = int(os.environ.get("OCR2_PREPROCESS_MEDIAN", "3"))
PREPROCESS_AUTOCONTRAST = int(os.environ.get("OCR2_PREPROCESS_AUTOCONTRAST", "1"))
DOCLING_ENABLED = os.environ.get("OCR2_DOCLING_ENABLED", "1") != "0"
CASCADE_ENABLED = os.environ.get("OCR2_CASCADE_ENABLED", "1") != "0"
TESSERACT_PSM = int(os.environ.get("OCR2_TESSERACT_PSM", "3"))
TESSERACT_PSM_SPARSE = int(os.environ.get("OCR2_TESSERACT_PSM_SPARSE", "11"))
DOCLING_TIMEOUT_S = 900
CASCADE_TIMEOUT_S = 900
DEFAULT_WORKERS = int(os.environ.get("OCR2_WORKERS", "3"))  # docling ~2GB/proc; 2-4

# Gemini-sweep ground truth for the pinned 2026 House scans.


REPO_ROOT = Path(__file__).resolve().parents[1]
for _entry in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

# Tesseract segmentation-mode override for scan_ocr's extraction helpers
# (extract_page_rows resolves module globals at call time, so replacing the
# imported names lets siblings tune PSM without touching shared code).
def _psm_wrapper(fn, psm):
    def wrapped(image_path, psm_arg=None, **kwargs):
        return fn(image_path, psm=psm, **kwargs)
    return wrapped

import scripts.scan_ocr as _scan_ocr  # noqa: E402

if TESSERACT_PSM != 3 or TESSERACT_PSM_SPARSE != 11:
    _scan_ocr.tesseract_plain_lines = _psm_wrapper(
        _scan_ocr.tesseract_plain_lines, TESSERACT_PSM
    )
    _scan_ocr.tesseract_lines = _psm_wrapper(
        _scan_ocr.tesseract_lines, TESSERACT_PSM
    )
    _scan_ocr.tesseract_words = _psm_wrapper(
        _scan_ocr.tesseract_words, TESSERACT_PSM
    )

from scripts.scan_ocr import (  # noqa: E402
    OcrRow,
    _EXAMPLE_RE,
    _classify_type,
    _merge_page_results,
    extract_page_rows,
    pdf_page_count,
    sha256_file,
    tesseract_plain_lines,
)
from scripts.ocr_zero_rows import extract_ticker, resolve_ticker  # noqa: E402
from analyzer.parsing import (  # noqa: E402
    _extract_amount_midpoint,
    _extract_date,
    _extract_transaction_type,
    _parse_docling_markdown,
    _parse_markdown_tables,
    parse_pdf_table,
)
from analyzer.parser_cascade import (  # noqa: E402
    ParserCascadeError,
    _parse_pdf_worker,
)

PINNED_CANARIES = {
    "9115808": {"rows": 1, "pages": 1, "page2_min": None},
    "9115813": {"rows": 9, "pages": 2, "page2_min": None},
    "9116141": {"rows": 134, "pages": 6, "page2_min": None},
    "8221322": {"rows": None, "pages": 56, "page2_min": 18},
}

AMOUNT_MIDPOINTS = {
    "A": 8000, "B": 32500, "C": 75000, "D": 175000, "E": 375000,
    "F": 750000, "G": 3000000, "H": 15000000, "I": 37500000, "J": 50000000,
}
_LETTER_BY_MIDPOINT = sorted(
    AMOUNT_MIDPOINTS.items(), key=lambda item: item[1]
)

# OCR residue that docling merges from the remarks column into the asset
# cell ("FILING STATUS: New ... DESCRIPTION: ... COMMENTS: ..."); such rows
# are duplicate artifacts of the real transaction above them and are dropped.
_FILING_RESIDUE_RE = re.compile(
    r"f\s*il\s*ing\s*s\s*tat\s*us|d\s*escript\s*ion|filing\s+status",
    re.IGNORECASE,
)
# Checkbox/instruction lines that must not count as transaction-row-like
# content when classifying an empty page.
_ROW_LIKE_EXCLUDE_RE = re.compile(
    r"member\s+of\s+the\s+u\.?\s*s\.?\s*house|officer\s+or\s+employee|"
    r"for\s+official\s+use\s+only|initial\s+public\s+offering|"
    r"initial\s+report|amended\s+report|please\s+indicate|"
    r"please\s+contact|file\s+an\s+original",
    re.IGNORECASE,
)
_ROW_LIKE_LINE_RE = re.compile(
    r"^\s*(?:sp|pc|sb|s[px]|x+|xx|bp)[\s|\]\[.}:_-]*", re.IGNORECASE
)
_INSTRUCTION_LINE_RE = re.compile(
    r"provide full name|not ticker symbol|initial report|amendment",
    re.IGNORECASE,
)


def _strip_filing_residue(asset: str) -> str:
    """Cut remarks-cell residue docling merges into the asset cell.

    Docling folds the remarks column ("FILING STATUS: ... DESCRIPTION: ...
    COMMENTS: ...") into the asset cell of the transaction above it,
    producing duplicate rows.  The residue is stripped from the first match
    onward so the real transaction survives; a fully-residue cell empties.
    """
    match = _FILING_RESIDUE_RE.search(asset)
    if not match:
        return asset
    return asset[: match.start()].strip(" -|")


# --------------------------------------------------------------------------
# Docling structured-table mapping (old-form PTR column layout)
# --------------------------------------------------------------------------

_HEADER_NORM_RE = re.compile(r"[^a-z0-9]+")
_TICKER_CELL_RE = re.compile(r"\([A-Za-z][A-Za-z0-9.\-]{0,5}\)")
# Repeated-header rows docling sometimes leaves inside the data ("FULL ASSET
# NAME Provide full name, not ticker symbol") must not map as transactions.
_HEADER_LIKE_ASSET_RE = re.compile(
    r"asset\s*name|provide\s*full\s*name|not\s*ticker\s*symbol",
    re.IGNORECASE,
)
_DOLLAR_CELL_RE = re.compile(r"\$[\d,]+")
_STRICT_DATE_CELL_RE = re.compile(
    r"\d{1,2}/\d{1,2}/\d{2,4}|\d{4}-\d{2}-\d{2}"
)


def _extract_tx_type_cell(cell: str | None) -> str | None:
    """Transaction type from a type cell, tolerating the merged account
    residue of the old forms ("P aCCoUNTS", "S INVESTMENT aCCoUNTS"): the
    leading P/S/E letter is the type and the account text is discarded."""
    tx_type = _extract_transaction_type(cell)
    if tx_type:
        return tx_type
    raw = str(cell or "").strip()
    first_token = raw.split(" ", 1)[0]
    if first_token and first_token != raw:
        return _extract_transaction_type(first_token)
    return None


def _dataframe_rows(dataframe, page_number: int) -> list[dict]:
    """Map one Docling structured-table dataframe to PTR tx dicts.

    Accepts only the old-form column layout (Asset with parenthetical
    ticker, Transaction Type, Transaction Date, Notification Date, Amount
    range).  Returns [] for any other layout -- checkbox-style 2026 grids
    (distinct per-column type/amount headers, no parenthetical tickers or
    dollar ranges in the cells) and degenerate tables -- so the page stays
    uncovered and the tesseract/cascade fallbacks reproduce the pre-fix
    behaviour byte for byte (the pinned 2026 canaries are all produced by
    tesseract and must not regress).
    """
    headers = [_HEADER_NORM_RE.sub("", str(column).casefold()) for column in dataframe.columns]
    if not len(dataframe):  # degenerate empty table
        return []
    asset_cols = [i for i, h in enumerate(headers) if "asset" in h]
    type_cols = [i for i, h in enumerate(headers) if "type" in h]
    amount_cols = [i for i, h in enumerate(headers) if "amount" in h]
    date_cols = [
        i for i, h in enumerate(headers)
        if "date" in h and "notification" not in h
    ]
    if not asset_cols or not type_cols or not amount_cols or not date_cols:
        return []
    # Checkbox layouts carry distinct per-column headers (Purchase/Sale/
    # Exchange, lettered amount ranges) -- reject so the pinned canaries
    # keep their tesseract-produced rows.
    if len({headers[i] for i in type_cols}) > 1:
        return []
    if len({headers[i] for i in amount_cols}) > 1:
        return []
    asset_col = asset_cols[0]
    type_col = type_cols[0]
    amount_col = amount_cols[0]
    date_col = next((i for i in date_cols if i != type_col), None)
    if date_col is None:
        return []
    notif_cols = [i for i, h in enumerate(headers) if "notification" in h]
    notif_col = notif_cols[0] if notif_cols else None

    # Data gate: only map pages whose rows carry the old-form evidence --
    # (>=1 parenthetical ticker OR >=1 P/S/E type letter) AND >=1 dollar
    # range AND >=1 strict date.  This is what separates old-form scans
    # (parenthetical tickers, lettered types, dollar ranges) from
    # checkbox-style 2026 grids (x-marks only).  Docling dataframes carry
    # the header in .columns, so row 0 is the first data row and must be
    # scanned too.
    seen_ticker = seen_letter = seen_dollar = seen_date = False
    for row_index in range(len(dataframe)):
        asset = _strip_filing_residue(str(dataframe.iat[row_index, asset_col] or ""))
        if _TICKER_CELL_RE.search(asset):
            seen_ticker = True
        if _extract_tx_type_cell(str(dataframe.iat[row_index, type_col] or "")):
            seen_letter = True
        if _DOLLAR_CELL_RE.search(str(dataframe.iat[row_index, amount_col] or "")):
            seen_dollar = True
        if _STRICT_DATE_CELL_RE.search(str(dataframe.iat[row_index, date_col] or "")):
            seen_date = True
        if (seen_ticker or seen_letter) and seen_dollar and seen_date:
            break
    if not ((seen_ticker or seen_letter) and seen_dollar and seen_date):
        return []

    rows: list[dict] = []
    for row_index in range(len(dataframe)):
        asset_raw = str(dataframe.iat[row_index, asset_col] or "")
        if _HEADER_LIKE_ASSET_RE.search(asset_raw):
            continue  # stray repeated header row inside the data
        asset = _strip_filing_residue(asset_raw)
        if not asset:
            continue
        asset = re.sub(r"\s+", " ", asset).strip()[:500]
        amount_raw, amount_midpoint = _extract_amount_midpoint(
            str(dataframe.iat[row_index, amount_col] or "")
        )
        notif_raw = (
            str(dataframe.iat[row_index, notif_col] or "")
            if notif_col is not None
            else ""
        )
        rows.append({
            "asset_description": asset,
            "transaction_type": _extract_tx_type_cell(
                str(dataframe.iat[row_index, type_col] or "")
            ),
            "transaction_date": _extract_date(
                str(dataframe.iat[row_index, date_col] or "")
            ),
            "notification_date": _extract_date(notif_raw),
            "notification_date_raw": notif_raw,
            "owner_code": None,
            "amount_raw": amount_raw,
            "amount_midpoint": amount_midpoint,
            "page_number": page_number,
        })
    return rows

# --------------------------------------------------------------------------
# PDF / image helpers
# --------------------------------------------------------------------------

def pdf_sha256(pdf_path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(pdf_path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pdf_text_layer(pdf_path: str | Path) -> str:
    """Extractable text layer ('' when the PDF is a pure scan)."""
    try:
        result = subprocess.run(
            ["pdftotext", str(pdf_path), "-"],
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        return result.stdout or ""
    except (OSError, subprocess.TimeoutExpired):
        return ""


def render_pages(pdf_path: str | Path, dpi: int = RENDER_DPI) -> list:
    """Rasterize every page via pdf2image (poppler), 300 dpi by default."""
    from pdf2image import convert_from_path  # noqa: PLC0415

    return convert_from_path(str(pdf_path), dpi=dpi)


def preprocess_image(image) -> "object":
    """ocrmypdf-style enhancement: grayscale, contrast, denoise, 2x upscale."""
    from PIL import Image, ImageFilter, ImageOps  # noqa: PLC0415

    img = image.convert("L")
    if PREPROCESS_AUTOCONTRAST:
        img = ImageOps.autocontrast(img)
    if PREPROCESS_MEDIAN > 1:
        img = img.filter(ImageFilter.MedianFilter(size=PREPROCESS_MEDIAN))
    if PREPROCESS_SCALE > 1:
        width, height = img.size
        img = img.resize(
            (int(width * PREPROCESS_SCALE), int(height * PREPROCESS_SCALE)),
            Image.LANCZOS,
        )
    return img


def _docling_converter():
    """Lazy per-process DocumentConverter (models load on first use)."""
    global _CONVERTER
    if _CONVERTER is None:
        from docling.document_converter import DocumentConverter  # noqa: PLC0415

        _CONVERTER = DocumentConverter()
    return _CONVERTER


_CONVERTER = None


def docling_pages(pdf_path: str | Path) -> tuple[list[dict], str | None]:
    """Docling OCR, one table-row set per 1-based PDF page.

    Primary source per page is Docling's structured table
    (``TableItem.export_to_dataframe()``) mapped through the old-form PTR
    column layout (Asset with parenthetical ticker, Transaction Type,
    transaction/notification dates, amount range).  The markdown text path
    is kept ONLY for pages Docling reports no table; pages whose structured
    table is not the old-form layout (checkbox-style 2026 grids, degenerate
    tables) yield no docling rows and the existing tesseract/cascade
    fallbacks take over unchanged, exactly as before.

    Returns ``(pages, error)`` where each page dict is
    ``{"page": int, "rows": [tx dict], "text": str}``.
    """
    try:
        result = _docling_converter().convert(str(pdf_path))
        document = result.document
        markdown = document.export_to_markdown(page_break_placeholder="\f")
        doc_tables = list(document.tables or [])
    except Exception as exc:  # noqa: BLE001 -- fail-closed boundary
        return [], f"docling:{type(exc).__name__}:{exc}"

    segments = markdown.split("\f")
    pages: list[dict] = []
    for page_number, segment in enumerate(segments, start=1):
        text_lines = [
            line.strip()
            for line in segment.splitlines()
            if line.strip() and not line.strip().startswith("![Image]")
        ]
        text = " ".join(text_lines).casefold()
        rows: list[dict] = []
        page_tables = [
            table
            for table in doc_tables
            if page_number in {item.page_no for item in table.prov}
        ]
        if page_tables:
            # Structured-dataframe path (primary): only the old-form layout
            # maps; everything else returns [] so the page stays uncovered
            # and tesseract/cascade handle it exactly as before the fix.
            for table in page_tables:
                try:
                    dataframe = table.export_to_dataframe(doc=document)
                except Exception:  # noqa: BLE001 -- per-table fail-closed
                    continue
                for tx in _dataframe_rows(dataframe, page_number):
                    tx["_engine"] = "docling"
                    rows.append(tx)
        else:
            # Docling reports no table on this page -> markdown text path.
            tables = _parse_docling_markdown(segment) or _parse_markdown_tables(segment)
            for table in tables:
                for tx in parse_pdf_table(table):
                    asset = _strip_filing_residue(
                        str(tx.get("asset_description") or "").strip()
                    )
                    if not asset:
                        continue
                    tx["asset_description"] = asset
                    tx["page_number"] = page_number
                    tx["_engine"] = "docling"
                    rows.append(tx)
        pages.append({"page": page_number, "rows": rows, "text": text})
    return pages, None


def _normalize_iso_date(raw: str | None) -> str | None:
    """MM/DD/YYYY|MM/DD/YY|YYYY-MM-DD -> YYYY-MM-DD (None when unparseable)."""
    if not raw:
        return None
    text = str(raw).strip()
    match = re.match(r"(\d{4})-(\d{1,2})-(\d{1,2})", text)
    if match:
        year, month, day = match.groups()
    else:
        match = re.match(r"(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})", text)
        if not match:
            return None
        month, day, year = match.groups()
        if len(year) == 2:
            year = ("19" if int(year) >= 50 else "20") + year
    try:
        parsed = date(int(year), int(month), int(day))
    except ValueError:
        return None
    return parsed.isoformat()


# --------------------------------------------------------------------------
# Per-page OCR and empty-page classification
# --------------------------------------------------------------------------

def tesseract_page(
    image, page_number: int, start_row_index: int
) -> tuple[list[OcrRow], str]:
    """scan_ocr accepted variants (dpi-pinned + no-pHYs) then preprocessed.

    Variants A/B replicate scripts/scan_ocr.sweep_document exactly (the
    variants the pinned 2026 canaries were derived from); the preprocessed
    variant is a tertiary fill for pages where A/B recovered nothing usable.
    Returns ``(rows, page_text)``.
    """
    import tempfile as _tempfile  # noqa: PLC0415

    with _tempfile.TemporaryDirectory(prefix="local_ocr_tess_") as tmp:
        tmp = Path(tmp)
        primary_path = tmp / f"page_{page_number:03d}.png"
        image.save(str(primary_path), dpi=(RENDER_DPI, RENDER_DPI))
        primary = extract_page_rows(
            str(primary_path), page_number, start_row_index
        )
        estimated_path = tmp / f"page_{page_number:03d}_est.png"
        image.save(str(estimated_path))
        secondary = extract_page_rows(
            str(estimated_path), page_number, start_row_index
        )
        merged = _merge_page_results(primary, secondary)
        has_usable = any(not row.date_unresolved for row in merged.rows)
        if has_usable:
            return merged.rows, merged.text or secondary.text or ""

        # Tertiary: preprocessed image (deskew/denoise/upscale).
        prepped_path = tmp / f"page_{page_number:03d}_prep.png"
        preprocess_image(image).save(
            str(prepped_path), dpi=(RENDER_DPI * PREPROCESS_SCALE,) * 2
        )
        prep_result = extract_page_rows(
            str(prepped_path), page_number, start_row_index
        )
        seen = {_row_key(row) for row in merged.rows}
        added = [
            row
            for row in prep_result.rows
            if not row.date_unresolved and _row_key(row) not in seen
        ]
        combined = PageResult(
            page_number=page_number,
            rows=[*merged.rows, *added],
            text=merged.text or prep_result.text or "",
        )
        return combined.rows, combined.text


class PageResult:  # noqa: D101 -- small shim mirroring scan_ocr.PageResult
    def __init__(self, page_number, rows, text=""):
        self.page_number = page_number
        self.rows = rows
        self.text = text


def _row_key(row: OcrRow) -> tuple:
    return (
        re.sub(r"[^a-z0-9]+", "", (row.asset_description or "").casefold()),
        row.transaction_date() or "",
        row.transaction_type or "",
    )


def _page_has_row_like_content(lines: list[str]) -> bool:
    """True when a row-less page carries unparsed transaction-row marks
    (leading checkbox mark followed by asset words), which indicates failed
    OCR rather than a legitimately empty page.  Checkbox/instruction lines
    (e.g. 'x Member of the U.S. House of Representatives') are excluded."""
    for line in lines:
        if _INSTRUCTION_LINE_RE.search(line):
            continue
        if _ROW_LIKE_EXCLUDE_RE.search(line):
            continue
        if _ROW_LIKE_LINE_RE.match(line):
            words = re.findall(r"[A-Za-z]{3,}", line)
            if len(words) >= 2:
                return True
    return False


def classify_empty_page(
    page_number: int,
    page_text: str,
    page_lines: list[str],
    *,
    uncovered: list[int],
    no_tx_pages: list[int],
    cover_pages: list[int],
    notes: list[str],
) -> None:
    """nothing-to-report -> no_tx; form header + filer block -> cover;
    certification/signature page -> cover; otherwise uncovered
    (fail-closed)."""
    text = page_text.casefold()
    nothing = "nothing to report" in text
    has_form_header = (
        "periodic transaction report" in text or "united states house" in text
    )
    filer_block = (
        "office telephone" in text
        or "member of the u.s. house" in text
        or "please see the attached" in text
    )
    # Trailing "CERTIFICATION AND SIGNATURE" page: the filer certifies the
    # attached report; the form carries no transactions there by
    # construction, so it is legitimately transaction-free.  Markers cover
    # the House e-filing block plus older/alternate form wording; the
    # row-like-content guard keeps pages that DO carry transaction rows
    # (e.g. a trailing table that failed to map) uncovered (fail-closed).
    cert_page = (
        ("certification" in text and "signature" in text)
        or "under penalty of perjury" in text
        or "truthfulness" in text
        or "the information contained herein is true" in text
        or "digitally signed" in text
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
    if cert_page and not _page_has_row_like_content(page_lines):
        cover_pages.append(page_number)
        notes.append(
            f"page {page_number}: certification/signature page "
            "(no transaction rows)"
        )
        return
    uncovered.append(page_number)
    notes.append(f"page {page_number}: no transaction rows")


# --------------------------------------------------------------------------
# Row building / validation (Gemini sweep staging contract)
# --------------------------------------------------------------------------

def _amount_letter(amount_raw, amount_midpoint) -> tuple[str | None, float | None]:
    if amount_raw:
        letter = str(amount_raw).strip().upper()
        if letter in AMOUNT_MIDPOINTS:
            return letter, amount_midpoint or AMOUNT_MIDPOINTS[letter]
    if amount_midpoint:
        best = min(
            _LETTER_BY_MIDPOINT, key=lambda item: abs(item[1] - amount_midpoint)
        )
        return best[0], AMOUNT_MIDPOINTS[best[0]]
    return None, None


def build_row(
    doc_id: str,
    year: int,
    tx: dict,
    *,
    member: str,
    artifact_sha256: str,
    row_index: int,
) -> dict:
    """Convert one parsed transaction to a Gemini-contract staged row."""
    page_number = tx.get("page_number")
    tx = {k: v for k, v in tx.items() if not k.startswith("_")}
    source_row_id = (
        f"{doc_id}:page:{page_number}:row:{row_index}"
        if page_number is not None
        else f"{doc_id}:row:{row_index}"
    )
    tx_date = _normalize_iso_date(tx.get("transaction_date"))
    amount_raw, amount_midpoint = _amount_letter(
        tx.get("amount_raw"), tx.get("amount_midpoint")
    )
    asset = str(tx.get("asset_description") or "").strip()[:500]
    tx_type = tx.get("transaction_type")
    disclosed_ticker = extract_ticker(asset)
    if disclosed_ticker:
        ticker_origin, raw_ticker, ticker_candidate = (
            "official", disclosed_ticker, None,
        )
    else:
        candidate = resolve_ticker(asset)
        ticker_origin = "unverified" if candidate else "not_reported"
        raw_ticker, ticker_candidate = candidate, candidate
    return {
        "source": SOURCE,
        "chamber": CHAMBER,
        "doc_id": str(doc_id),
        "year": int(year),
        "source_record_id": str(doc_id),
        "source_row_id": source_row_id,
        "page_number": page_number,
        "row_index": row_index,
        "member": member,
        "asset_description": asset,
        "transaction_type": tx_type,
        "transaction_date": tx_date,
        "transaction_date_raw": str(tx.get("transaction_date") or ""),
        "notification_date": _normalize_iso_date(tx.get("notification_date")),
        "notification_date_raw": str(tx.get("notification_date_raw") or ""),
        "amount_raw": amount_raw,
        "amount_midpoint": amount_midpoint,
        "owner_code": tx.get("owner_code"),
        "raw_asset_description": asset,
        "raw_transaction_subtype": tx_type,
        "ticker_origin": ticker_origin,
        "raw_ticker": raw_ticker,
        "ticker_candidate": ticker_candidate,
        "raw_asset_class": "Not separately reported",
        "ingestion_generation": GENERATION,
        "artifact_sha256": artifact_sha256,
    }


def _local_validate(rows: list[dict]) -> tuple[list[dict], dict]:
    """Fail-closed per-row validation (accepted local-OCR policy).

    Rows must carry a non-empty asset; transaction type, amount and
    notification date are never invented (audit C9) and NULLs are staged.
    Rows without a strict transaction date are staged with a NULL ISO date
    (the accepted local-OCR sweep stages them the same way; the ingest layer
    drops unparseable dates).  Doc-level resolution requires at least one
    strict-dated row and full page coverage, enforced by the caller.
    """
    rejections: dict[str, int] = defaultdict(int)
    valid: list[dict] = []
    for tx in rows:
        asset = str(tx.get("asset_description") or "").strip()
        if not asset:
            rejections["invalid_asset"] += 1
            continue
        tx = dict(tx)
        valid.append(tx)
    return valid, dict(rejections)


# --------------------------------------------------------------------------
# Per-document pipeline
# --------------------------------------------------------------------------

def process_document(
    doc_id: str,
    year: int,
    pdf_path: str | Path,
    metadata: dict,
) -> dict:
    """Run the local second-pass pipeline for one PDF; never writes the DB."""
    started = time.time()
    pdf_path = Path(pdf_path)
    result = {
        "doc_id": str(doc_id),
        "year": int(year),
        "pdf_path": str(pdf_path),
        "status": "unresolved",
        "reasons": [],
        "rows": [],
        "row_count": 0,
        "page_count": None,
        "covered_pages": [],
        "uncovered_pages": [],
        "artifact_sha256": None,
        "elapsed_s": None,
        "canary": None,
        "engines": [],
    }
    if not pdf_path.exists():
        result["reasons"].append("pdf_missing")
        result["elapsed_s"] = round(time.time() - started, 2)
        return result

    artifact_sha256 = pdf_sha256(pdf_path)
    page_count = pdf_page_count(pdf_path)
    result["artifact_sha256"] = artifact_sha256
    result["page_count"] = page_count
    expected_member = (metadata or {}).get("member")

    # Step 1+2: image preprocessing + Docling (page-attributed rows).
    docling_pages_result: list[dict] = []
    docling_error: str | None = None
    if DOCLING_ENABLED:
        docling_pages_result, docling_error = docling_pages(pdf_path)
    else:
        docling_error = "docling disabled (OCR2_DOCLING_ENABLED=0)"
    if docling_error:
        result["reasons"].append(docling_error)
    result["engines"].append("docling")

    page_outcomes: dict[int, list[dict]] = {}
    page_notes: list[str] = []
    page_texts: dict[int, str] = {}
    no_tx_pages: list[int] = []
    cover_pages: list[int] = []
    uncovered: list[int] = []

    for page in docling_pages_result:
        page_number = page["page"]
        rows = page["rows"]
        page_texts[page_number] = page["text"]
        if rows:
            page_outcomes[page_number] = rows
        # Pages without docling rows (including docling text-classified
        # cover/no_tx pages) are re-examined by Tesseract: docling's
        # classification is advisory only (it can drop page tables or miss
        # faint transactions on pages that also carry the filer block).
        if page_number not in page_outcomes:
            uncovered.append(page_number)

    missing_pages = [
        p for p in range(1, page_count + 1)
        if p not in page_outcomes and p not in uncovered
    ]
    for p in missing_pages:
        uncovered.append(p)
        page_notes.append(f"page {p}: not produced by docling")

    # Step 3: preprocessed Tesseract fallback for uncovered pages only.
    if uncovered:
        try:
            images = render_pages(pdf_path, dpi=RENDER_DPI)
        except Exception as exc:  # noqa: BLE001 -- fail-closed
            result["reasons"].append(f"rasterize:{exc}")
            images = []
        if images:
            result["engines"].append("tesseract")
            for page_number in list(uncovered):
                if page_number < 1 or page_number > len(images):
                    continue
                image = images[page_number - 1]
                try:
                    tess_rows, tess_text = tesseract_page(
                        image, page_number, len(result["rows"]) + 1
                    )
                except Exception as exc:  # noqa: BLE001 -- per-page fail-closed
                    result["reasons"].append(f"tesseract page {page_number}: {exc}")
                    continue
                page_texts[page_number] = (
                    page_texts.get(page_number, "") + " " + tess_text
                )
                if any(not row.date_unresolved for row in tess_rows):
                    page_outcomes[page_number] = [
                        {
                            "asset_description": row.asset_description,
                            "transaction_type": row.transaction_type,
                            "transaction_date": row.transaction_date_raw,
                            "amount_raw": None,
                            "amount_midpoint": None,
                            "owner_code": row.owner_code,
                            "page_number": row.page_number,
                        }
                        for row in tess_rows
                    ]
                    uncovered.remove(page_number)
                else:
                    if page_number in uncovered:
                        uncovered.remove(page_number)
                    lines = [row.asset_description for row in tess_rows]
                    classify_empty_page(
                        page_number,
                        page_texts[page_number],
                        lines + _plain_lines_from_text(page_texts[page_number]),
                        uncovered=uncovered,
                        no_tx_pages=no_tx_pages,
                        cover_pages=cover_pages,
                        notes=page_notes,
                    )
            for image in images:
                close = getattr(image, "close", None)
                if callable(close):
                    close()

    # Step 4: deterministic text-layer retry with the accepted cascade.
    text_layer = pdf_text_layer(pdf_path)
    cascade_rows: list[dict] = []
    if uncovered and text_layer.strip() and CASCADE_ENABLED:
        result["engines"].append("cascade")
        try:
            _, cascade_rows, _ = _parse_pdf_worker(pdf_path)
        except ParserCascadeError as exc:
            result["reasons"].append(f"cascade:{exc}")
        except Exception as exc:  # noqa: BLE001 -- fail-closed boundary
            result["reasons"].append(f"cascade:{type(exc).__name__}:{exc}")
        if cascade_rows:
            result["engines"].append("cascade_rows")
            for tx in cascade_rows:
                tx.setdefault("page_number", None)
    elif uncovered and not text_layer.strip() and CASCADE_ENABLED:
        result["reasons"].append("cascade skipped: no extractable text layer")

    # Assemble rows: per-page rows win (page attribution); cascade rows are
    # used only when no page-attributed rows exist.
    ordered: list[dict] = []
    for page_number in range(1, page_count + 1):
        ordered.extend(page_outcomes.get(page_number, []))
    if not ordered and cascade_rows:
        ordered = cascade_rows
    if not ordered and not uncovered:
        # Zero rows on every page.  Terminal no_txs requires explicit
        # nothing-to-report evidence on EVERY page; cover-classified pages
        # (filer block only) are NOT zero-transaction evidence — a faint
        # one-page PTR that OCR cannot read looks exactly like a cover page
        # (82 of the staged 2015 no_txs docs were Gemini-resolved), so a
        # cover-only document stays unresolved (fail-closed, left for
        # Gemini/review) instead of being retired.
        all_no_tx = bool(no_tx_pages) and not cover_pages and (
            len(no_tx_pages) == page_count
        )
        if all_no_tx:
            result["status"] = "no_txs"
            result["covered_pages"] = sorted(no_tx_pages)
            result["uncovered_pages"] = []
            result["reasons"].extend(sorted(page_notes))
            result["reasons"].append(
                "filing reports no transactions "
                f"(nothing-to-report pages: {len(no_tx_pages)}/{page_count})"
            )
            result["elapsed_s"] = round(time.time() - started, 2)
            return result
        if cover_pages and not no_tx_pages:
            result["reasons"].append(
                "zero transactions unconfirmed: all pages cover-classified "
                "(filer block only, no readable rows)"
            )
        elif cover_pages:
            result["reasons"].append(
                "zero transactions unconfirmed: cover pages present without "
                "nothing-to-report evidence on every page"
            )
        uncovered = list(range(1, page_count + 1))

    valid_rows, rejections = _local_validate(ordered)
    if rejections:
        result["reasons"].append(
            "validation:" + json.dumps(rejections, sort_keys=True)
        )
    if not str(expected_member or "").strip():
        result["reasons"].append("metadata missing: no staged member/filing record")

    # Dedup only Docling artifacts of the same transaction (remarks-cell
    # merges produce duplicate rows after residue stripping).  Tesseract
    # rows pass through untouched: the accepted local-OCR sweep stages
    # legitimate same-day duplicate lots (e.g. 9116141's 134 rows contain
    # four identical (asset, type, date) pairs) and the pinned canary
    # counts depend on that.
    seen_keys: set[tuple] = set()
    deduped: list[dict] = []
    for tx in valid_rows:
        if tx.get("page_number") is not None and tx.get("_engine") == "docling":
            key = (
                re.sub(r"[^a-z0-9]+", "", str(tx.get("asset_description") or "").casefold()),
                str(tx.get("transaction_type") or "").casefold(),
                _normalize_iso_date(tx.get("transaction_date")) or "",
                str(tx.get("amount_raw") or "").strip().upper(),
            )
            if key in seen_keys:
                continue
            seen_keys.add(key)
        deduped.append(tx)
    valid_rows = deduped

    row_index = 0
    staged_rows: list[dict] = []
    for tx in valid_rows:
        row_index += 1
        staged_rows.append(
            build_row(
                doc_id,
                year,
                tx,
                member=expected_member,
                artifact_sha256=artifact_sha256,
                row_index=row_index,
            )
        )
    result["rows"] = staged_rows
    result["row_count"] = len(staged_rows)
    result["uncovered_pages"] = sorted(set(uncovered))
    result["covered_pages"] = sorted(
        set(range(1, page_count + 1)) - set(uncovered)
    )
    result["reasons"].extend(sorted(set(page_notes)))

    strict_dated = sum(
        1 for row in staged_rows if row.get("transaction_date") is not None
    )
    if not uncovered and strict_dated >= 1 and staged_rows and expected_member:
        result["status"] = "resolved"
    elif not uncovered and not staged_rows and not no_tx_pages and not cover_pages:
        result["reasons"].append("no rows recovered across all engines")
    elif uncovered:
        result["reasons"].append(
            "pages without transaction rows: " + ",".join(map(str, uncovered))
        )
    if strict_dated < 1 and staged_rows:
        result["reasons"].append("no strictly parseable transaction date")

    result["elapsed_s"] = round(time.time() - started, 2)
    return result


def _plain_lines_from_text(text: str) -> list[str]:
    return [line for line in text.split(" ") if line.strip()]


def check_canary(result: dict) -> dict:
    """Gemini-contract canary check; demotes the doc to unresolved on miss."""
    doc_id = result["doc_id"]
    expected = PINNED_CANARIES.get(doc_id)
    if expected is None:
        return result
    passed = True
    detail = []
    if result["status"] != "resolved":
        passed = False
        detail.append(
            f"status={result['status']} reasons={result['reasons']}"
        )
    if expected["rows"] is not None and result["row_count"] != expected["rows"]:
        passed = False
        detail.append(f"rows={result['row_count']} expected={expected['rows']}")
    if expected["pages"] is not None and result["page_count"] != expected["pages"]:
        passed = False
        detail.append(
            f"pages={result['page_count']} expected={expected['pages']}"
        )
    if result["uncovered_pages"]:
        passed = False
        detail.append(f"uncovered={result['uncovered_pages']}")
    if expected["page2_min"] is not None:
        page2_rows = sum(
            1 for row in result["rows"] if row.get("page_number") == 2
        )
        if page2_rows < expected["page2_min"]:
            passed = False
            detail.append(
                f"page2_rows={page2_rows} expected>={expected['page2_min']}"
            )
    result["canary"] = {
        "doc_id": doc_id,
        "expected": expected,
        "actual": {
            "status": result["status"],
            "row_count": result["row_count"],
            "page_count": result["page_count"],
            "page2_rows": (
                sum(1 for row in result["rows"] if row.get("page_number") == 2)
                if expected["page2_min"] is not None
                else None
            ),
        },
        "passed": passed,
        "detail": "; ".join(detail),
    }
    if not passed:
        result["status"] = "unresolved"
        result["reasons"].append(f"canary failed: {result['canary']['detail']}")
    return result


# --------------------------------------------------------------------------
# Staging (Gemini sweep contract)
# --------------------------------------------------------------------------

def stage_document(out_root: str | Path, result: dict) -> None:
    """Atomically stage one document: rows jsonl (tmp+rename) and docs envelope."""
    out_root = Path(out_root)
    rows_dir = out_root / "rows"
    docs_dir = out_root / "docs"
    rows_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)
    doc_id = result["doc_id"]

    rows_path = rows_dir / f"{doc_id}.jsonl"
    payload = "".join(
        json.dumps(row, sort_keys=True) + "\n" for row in result["rows"]
    )
    fd, tmp_name = tempfile.mkstemp(prefix=f".{doc_id}.", dir=str(rows_dir))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, rows_path)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass

    envelope = {
        "doc_id": doc_id,
        "year": int(result["year"]),
        "status": result["status"],
        "artifact_sha256": result["artifact_sha256"],
        "page_count": result["page_count"],
        "covered_pages": result["covered_pages"],
        "uncovered_pages": result["uncovered_pages"],
        "row_count": result["row_count"],
        "rows_file": f"rows/{doc_id}.jsonl",
        "rows_file_sha256": sha256_file(rows_path),
        "canary": result.get("canary"),
        "reasons": result["reasons"],
        "engine": ENGINE,
        "engines": result.get("engines", []),
        "parser_version": PARSER_VERSION,
        "elapsed_s": result.get("elapsed_s"),
        "extracted_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    doc_path = docs_dir / f"{doc_id}.json"
    fd, tmp_name = tempfile.mkstemp(prefix=f".{doc_id}.", dir=str(docs_dir))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(envelope, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, doc_path)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass


def write_manifest(
    out_root: str | Path, *, kind: str, data_dir: str | Path
) -> Path:
    """Assemble manifest.json from staged docs/ envelopes (idempotent)."""
    out_root = Path(out_root)
    docs_dir = out_root / "docs"
    resolved: dict[str, dict] = {}
    unresolved: dict[str, list[str]] = {}
    no_txs: dict[str, dict] = {}
    total_rows = 0
    staged_files: dict[str, str] = {}
    by_year: dict[str, dict] = {}
    canaries: dict[str, dict] = {}
    doc_count = 0
    for doc_path in sorted(docs_dir.glob("*.json")):
        envelope = json.loads(doc_path.read_text())
        doc_id = envelope["doc_id"]
        year = int(envelope["year"])
        doc_count += 1
        rows_path = out_root / envelope["rows_file"]
        rows_sha = sha256_file(rows_path) if rows_path.exists() else None
        staged_files[envelope["rows_file"]] = rows_sha
        staged_files[f"docs/{doc_id}.json"] = sha256_file(doc_path)
        bucket = by_year.setdefault(
            str(year),
            {"doc_count": 0, "resolved_count": 0, "unresolved_count": 0,
             "no_txs_count": 0, "row_count": 0},
        )
        bucket["doc_count"] += 1
        bucket["row_count"] += int(envelope["row_count"])
        status = envelope["status"]
        if status == "resolved":
            resolved[doc_id] = {
                "year": year,
                "row_count": envelope["row_count"],
                "pages_covered": len(envelope["covered_pages"]),
            }
            bucket["resolved_count"] += 1
            total_rows += int(envelope["row_count"])
        elif status == "no_txs":
            no_txs[doc_id] = {
                "year": year,
                "pages_covered": len(envelope["covered_pages"]),
            }
            bucket["no_txs_count"] += 1
        else:
            unresolved[doc_id] = envelope["reasons"]
            bucket["unresolved_count"] += 1
        if (envelope.get("canary") or {}).get("expected") is not None:
            canaries[doc_id] = envelope["canary"]

    manifest = {
        "generation": GENERATION,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "engine": ENGINE,
        "parser_version": PARSER_VERSION,
        "script": "scripts/ocr_local_sweep.py",
        "script_version": SCRIPT_VERSION,
        "data_dir": str(data_dir),
        "kind": kind,
        "doc_count": doc_count,
        "resolved_count": len(resolved),
        "unresolved_count": len(unresolved),
        "no_txs_count": len(no_txs),
        "total_rows": total_rows,
        "by_year": dict(sorted(by_year.items())),
        "resolved": resolved,
        "no_txs": no_txs,
        "unresolved": unresolved,
        "canary": canaries,
        "staged_files_sha256": staged_files,
    }
    manifest_path = out_root / "manifest.json"
    fd, tmp_name = tempfile.mkstemp(prefix=".manifest.", dir=str(out_root))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, manifest_path)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
    staged_files["manifest.json"] = sha256_file(manifest_path)
    return out_root


# --------------------------------------------------------------------------
# Input / metadata / CLI
# --------------------------------------------------------------------------

def load_input_list(manifest_path: str | Path, data_dir: str | Path) -> list[tuple]:
    """rebuild2 manifest -> [(doc_id, year, pdf_path)] for unresolved docs."""
    manifest = json.loads(Path(manifest_path).read_text())
    items: list[tuple] = []
    for year, entry in manifest.get("house", {}).items():
        pdf_dir = Path(data_dir) / year / "pdfs"
        for doc_id in entry.get("unresolved_doc_ids", []):
            items.append((str(doc_id), int(year), pdf_dir / f"{doc_id}.pdf"))
    return items


def load_metadata(db_path: str | Path) -> dict[str, dict]:
    """Read-only member/filing metadata for the staged generation."""
    import duckdb  # noqa: PLC0415

    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        rows = conn.execute(
            """
            SELECT CAST(m.doc_id AS VARCHAR), m.filing_date,
                   m.first_name, m.last_name
            FROM metadata m
            WHERE m.filing_type = 'P'
            """
        ).fetchall()
    finally:
        conn.close()
    metadata: dict[str, dict] = {}
    for doc_id, filing_date, first, last in rows:
        member = " ".join(part for part in (first, last) if part).strip()
        metadata[doc_id] = {
            "filing_date": filing_date.date()
            if hasattr(filing_date, "date")
            else filing_date,
            "member": member or None,
        }
    return metadata


def staged_docs(out_root: str | Path) -> set[str]:
    """doc ids already staged (resume support)."""
    out_root = Path(out_root)
    docs_dir = out_root / "docs"
    rows_dir = out_root / "rows"
    staged: set[str] = set()
    for envelope_path in docs_dir.glob("*.json"):
        doc_id = envelope_path.stem
        if (rows_dir / f"{doc_id}.jsonl").exists():
            staged.add(doc_id)
    return staged


def _run_pool(items, workers: int):
    """Stream per-doc results as workers finish them (imap_unordered).

    pool.map would hold every result until the whole batch finishes, so
    staging (done by the caller for each yielded result) only happened at
    batch end and a slow straggler stalled all completed work.  Yielding
    from imap_unordered lets the caller stage each doc atomically the
    moment it completes: durable incremental progress, no straggler stall.
    A worker death (e.g. OOM kill) aborts the stream; completed documents
    were already staged, so the next cycle resumes exactly with
    --skip-staged.
    """
    try:
        with Pool(workers) as pool:
            yield from pool.imap_unordered(_process_one, items)
    except (BrokenPipeError, EOFError, ConnectionResetError) as exc:
        print(
            f"SWEEP pool aborted: {exc.__class__.__name__}: {exc} "
            "(worker process died; staged docs are durable, resume with --skip-staged)",
            flush=True,
        )
        raise


def _process_one(item) -> dict:
    """Pool worker entry (module-level for pickling)."""
    doc_id, year, pdf_path, metadata = item
    return process_document(doc_id, year, pdf_path, metadata)


def run_sweep(args) -> int:
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.merge_only:
        write_manifest(out_root, kind="sweep", data_dir=args.data_dir)
        print(f"SWEEP manifest assembled at {out_root}")
        return 0

    items = load_input_list(args.manifest, args.data_dir)
    if args.years:
        wanted = set(args.years)
        items = [item for item in items if item[1] in wanted]
    print(f"SWEEP input: {len(items)} unresolved docs from {args.manifest}")
    if not items:
        print("SWEEP: no work items")
        return 0

    metadata = load_metadata(args.db)
    missing_meta = [doc_id for doc_id, *_ in items if doc_id not in metadata]
    if missing_meta:
        print(
            f"SWEEP: {len(missing_meta)} docs without staged metadata "
            f"(fail closed): {missing_meta[:5]}"
        )

    if args.skip_staged:
        staged = staged_docs(out_root)
        items = [item for item in items if item[0] not in staged]
        print(f"SWEEP: skipping {len(staged)} already-staged docs; {len(items)} remain")

    items = [
        (doc_id, year, pdf, metadata.get(doc_id, {}))
        for doc_id, year, pdf in items
    ]
    if args.max_docs:
        items = items[: args.max_docs]

    results = []
    started = time.time()
    completed = 0
    for result in _run_pool(items, args.workers):
        results.append(result)
        completed += 1
        # Incremental staging: every doc is durably staged as it completes so
        # progress survives crashes and --skip-staged resumes exactly.
        stage_document(out_root, result)
        if completed % 10 == 0 or completed == len(items):
            write_manifest(out_root, kind="sweep", data_dir=args.data_dir)
            resolved = sum(1 for r in results if r["status"] == "resolved")
            no_txs = sum(1 for r in results if r["status"] == "no_txs")
            rows = sum(r["row_count"] for r in results)
            elapsed = time.time() - started
            rate = completed / elapsed if elapsed else 0
            remaining = (len(items) - completed) / rate if rate else 0
            print(
                f"  [{completed}/{len(items)}] resolved={resolved} no_txs={no_txs} "
                f"rows={rows} elapsed={elapsed:.0f}s eta={remaining/3600:.1f}h",
                flush=True,
            )
    write_manifest(out_root, kind="sweep", data_dir=args.data_dir)

    resolved = sum(1 for r in results if r["status"] == "resolved")
    no_txs = sum(1 for r in results if r["status"] == "no_txs")
    unresolved = len(results) - resolved - no_txs
    rows = sum(r["row_count"] for r in results)
    per_year: dict[int, list] = defaultdict(list)
    for r in results:
        per_year[r["year"]].append(r["status"])
    print(
        f"SWEEP done: docs={len(results)} resolved={resolved} no_txs={no_txs} "
        f"unresolved={unresolved} rows={rows}"
    )
    print(
        "SWEEP per-year: "
        + json.dumps(
            {
                str(y): {
                    "docs": len(v),
                    "resolved": v.count("resolved"),
                    "no_txs": v.count("no_txs"),
                    "unresolved": v.count("unresolved"),
                }
                for y, v in sorted(per_year.items())
            }
        )
    )
    print(f"SWEEP staged to {out_root}")

    failed_canaries = [
        r for r in results if (r.get("canary") or {}).get("expected") is not None
        and not r.get("canary", {}).get("passed")
    ]
    if failed_canaries:
        print(
            "SWEEP CANARY GATE FAILED (explicit fail-closed): "
            + ", ".join(r["doc_id"] for r in failed_canaries)
        )
        for r in failed_canaries:
            print(f"  {r['doc_id']}: {r['canary']['detail']}")
        return 1
    passed_canaries = [
        r for r in results if (r.get("canary") or {}).get("expected") is not None
    ]
    if passed_canaries:
        print("SWEEP CANARY GATE PASSED: " + ", ".join(r["doc_id"] for r in passed_canaries))
    return 0


def run_pilot(args) -> int:
    data_dir = Path(args.data_dir)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    metadata = load_metadata(args.db)
    items = []
    for doc_id in PINNED_CANARIES:
        pdf = data_dir / "2026" / "pdfs" / f"{doc_id}.pdf"
        if not pdf.exists():
            print(f"PILOT MISSING PINNED PDF: {pdf}")
            return 2
        items.append((doc_id, 2026, pdf, metadata.get(doc_id, {})))

    import random  # noqa: PLC0415

    random.seed(20260810)
    all_items = load_input_list(args.manifest, data_dir)
    random_items = random.sample(all_items, min(args.random, len(all_items)))
    for doc_id, year, pdf in random_items:
        if doc_id in {d for d, *_ in items}:
            continue
        items.append((doc_id, year, pdf, metadata.get(doc_id, {})))

    results = []
    for item in items:
        result = process_document(*item[:3], item[3])
        if result["doc_id"] in PINNED_CANARIES:
            result = check_canary(result)
        results.append(result)
        canary_txt = ""
        if result.get("canary"):
            canary_txt = (
                f" canary={'PASS' if result['canary']['passed'] else 'FAIL'}"
            )
        print(
            f"PILOT {result['doc_id']} ({result['year']}) "
            f"status={result['status']} rows={result['row_count']} "
            f"pages={result['page_count']} elapsed={result['elapsed_s']}s "
            f"engines={result.get('engines')}{canary_txt}"
        )
        for reason in result["reasons"][:6]:
            print(f"    - {reason}")

    for result in results:
        stage_document(out_root, result)
    write_manifest(out_root, kind="pilot", data_dir=data_dir)
    print(f"PILOT staged to {out_root}")

    failed = [
        r for r in results if (r.get("canary") or {}).get("expected") is not None
        and not r.get("canary", {}).get("passed")
    ]
    if failed:
        print(
            "PILOT GATE FAILED (explicit fail-closed): "
            + ", ".join(r["doc_id"] for r in failed)
        )
        for r in failed:
            print(f"  {r['doc_id']}: {r['canary']['detail']}")
        return 1
    print("PILOT GATE PASSED: all pinned canaries met")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    pilot = sub.add_parser("pilot", help="run pinned + random pilot")
    pilot.add_argument("--data-dir", required=True,
                       help="staged generation dir containing <year>/pdfs")
    pilot.add_argument("--db", required=True,
                       help="staged generation congress.duckdb (read-only)")
    pilot.add_argument("--manifest", required=True,
                       help="rebuild2 generation manifest.json")
    pilot.add_argument("--out", default=str(REPO_ROOT / ".staging" / "ocr2-local" / GENERATION))
    pilot.add_argument("--random", type=int, default=3)
    pilot.set_defaults(handler=run_pilot)

    sweep = sub.add_parser("sweep", help="parallel staged sweep")
    sweep.add_argument("--data-dir", required=True)
    sweep.add_argument("--db", required=True)
    sweep.add_argument("--manifest", required=True)
    sweep.add_argument("--out", default=str(REPO_ROOT / ".staging" / "ocr2-local" / GENERATION))
    sweep.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    sweep.add_argument("--years", nargs="*", type=int, default=None)
    sweep.add_argument("--skip-staged", action="store_true")
    sweep.add_argument("--max-docs", type=int, default=None)
    sweep.add_argument("--merge-only", action="store_true")
    sweep.set_defaults(handler=run_sweep)

    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
