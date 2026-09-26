"""OCR fallback backend: pytesseract + pdf2image.

Last-resort engine when no text layer is available AND Docling subprocess
fails. Rasterizes each page to a 200dpi image, runs tesseract on it, then
extracts ticker / tx-type / date / amount rows from the resulting plaintext.
"""

import os
import re
import time
from collections import Counter
from pathlib import Path

from analyzer.models import TransactionType

# Bound every external OCR subprocess so one pathological PDF cannot stall a
# refresh worker indefinitely (observed: pool worker blocked for 50+ minutes
# at ~0 CPU waiting on an unbounded poppler/tesseract child).
_OCR_CALL_TIMEOUT = 90
_RASTERIZE_TIMEOUT = 120
_OCR_DOCUMENT_BUDGET = 600


class OcrBackendError(RuntimeError):
    """The local OCR backend could not execute reliably."""


class OcrIncompleteError(OcrBackendError):
    """OCR ran but did not establish complete page coverage."""

    def __init__(self, message: str, partial_tables: list[list[list[str]]]):
        super().__init__(message)
        self.partial_tables = partial_tables


def _orient_image(image, pytesseract):
    """Return an upright image when Tesseract detects a rotated scan."""
    # OSD needs only a coarse thumbnail; running it on the full 200dpi page
    # makes it nearly as expensive as the real OCR pass. 1200px is the floor
    # where OSD still reads sparse form text (600px fails with 'Too few
    # characters' and marks pages incomplete).
    try:
        thumbnail = image.copy()
        thumbnail.thumbnail((1200, 1200))
    except AttributeError:
        thumbnail = image
    try:
        osd = pytesseract.image_to_osd(thumbnail, timeout=_OCR_CALL_TIMEOUT)
        match = re.search(r"^Rotate:\s*(90|180|270)\s*$", osd, re.MULTILINE)
        if not match:
            return image
        # OSD reports the clockwise correction. PIL uses counter-clockwise
        # positive angles, so apply its inverse.
        return image.rotate(360 - int(match.group(1)), expand=True)
    except Exception as exc:
        raise OcrBackendError(f"orientation detection failed: {exc}") from exc


def _date_pattern() -> str:
    return r"(?:\d{1,2}/\d{1,2}/(?:\d{4}|\d{2})|\d{4}-\d{2}-\d{2})\b"


def _parse_tickerless_inline(stripped: str, amount_str: str | None):
    for match in re.finditer(r"(?<![A-Z0-9])(PP?|SS?|E)(?![A-Z0-9])", stripped.upper()):
        date_match = re.search(_date_pattern(), stripped[match.end() :])
        asset_name = re.sub(r"(?:\[[^]]+\]\s*)+$", "", stripped[: match.start()]).strip(
            " -|"
        )
        if not date_match or not asset_name:
            continue
        code = match.group(1)[0]
        tx_type = {
            "P": TransactionType.PURCHASE.value,
            "S": TransactionType.SALE.value,
            "E": TransactionType.EXCHANGE.value,
        }[code]
        return [asset_name, tx_type, date_match.group(0), amount_str or ""]
    return None


def _parse_ocr_text_to_rows(text: str) -> list[list[str]]:
    rows: list[list[str]] = []
    pending_asset: str | None = None
    pending_fields = ""

    for raw_line in [*text.splitlines(), ""]:
        stripped = raw_line.strip()
        ticker_match = re.search(r"\(([A-Za-z][A-Za-z0-9.\-]{0,5})\)", stripped)
        amount_match = re.search(r"\$[\d,]+\s*-\s*\$[\d,]+", stripped)
        amount_str = amount_match.group(0) if amount_match else None
        inline_row = (
            _parse_tickerless_inline(stripped, amount_str) if not ticker_match else None
        )
        is_field = re.match(r"^(?:[PSE]{1,2}\b|\d|\$|\[)", stripped)
        if not stripped or ticker_match or inline_row or not is_field:
            if pending_asset:
                row = _row_from_fields(pending_asset, pending_fields)
                if row is not None:
                    rows.append(row)
            pending_asset = None
            pending_fields = ""

        if not stripped or stripped.casefold() in {"cover page", "certification"}:
            continue

        if ticker_match:
            pending_asset = stripped[: ticker_match.end()].strip()
            pending_fields = stripped[ticker_match.end() :].strip()
            continue

        if inline_row is not None:
            rows.append(inline_row)
            continue
        if not is_field:
            pending_asset = stripped
            continue
        if pending_asset is None:
            continue
        pending_fields = f"{pending_fields} {stripped}".strip()

    return rows


def _row_from_fields(asset_name: str, rest: str):
    rest_clean = re.sub(r"\s+", " ", rest).strip().upper()
    amount = re.search(r"\$[\d,]+\s*-\s*\$[\d,]+", rest)
    tx_type, date_str = _tx_type_and_date(rest_clean, rest)
    if tx_type and date_str:
        return [asset_name, tx_type, date_str, amount.group(0) if amount else ""]
    return None


def _tx_type_and_date(rest_clean: str, rest: str) -> tuple[str | None, str | None]:
    tx_type: str | None = None
    # Strip leading asset/owner markers like '[ST]', '[SP]', '[JC]' that can
    # appear between the ticker and the tx code in OCR'd output. Without this,
    # a line like "(AAPL) [ST] P 01/15/2024" misses the tx code and the whole
    # row is dropped.
    body = re.sub(r"^(?:\[[^\]]*\]\s*)+", "", rest_clean).lstrip()
    if body.startswith(("P ", "PP ")):
        tx_type = TransactionType.PURCHASE.value
    elif body.startswith(("S ", "SS ")):
        tx_type = TransactionType.SALE.value
    elif body.startswith("E "):
        tx_type = TransactionType.EXCHANGE.value

    # Accept 1- or 2-digit month/day to match cells-level extractor behavior.
    date_match = re.search(_date_pattern(), rest)
    return tx_type, date_match.group(0) if date_match else None


def _reconcile_rows(*row_sets: list[list[str]]) -> list[list[str]]:
    reconciled: list[list[str]] = []
    seen = Counter()
    for rows in row_sets:
        counts = Counter()
        for row in rows:
            key = tuple(
                re.sub(r"\s+", " ", str(cell)).strip().casefold() for cell in row
            )
            counts[key] += 1
            if counts[key] <= seen[key]:
                continue
            reconciled.append(row)
        seen |= counts
    return reconciled


def _confirmed_nontransaction_page(text: str, image) -> bool:
    if not text.strip():
        # Empty OCR is not evidence of a blank scan. Check the pixels too.
        if not hasattr(image, "convert"):
            return False
        with image.convert("L") as gray:
            return gray.getextrema()[0] >= 250
    if re.search(
        r"\$|\b(?:asset|transaction date|transaction type)\b|^\s*[PSE]\b",
        text,
        re.I | re.M,
    ):
        return False
    return bool(re.search(r"(?im)^\s*(?:cover page|certification)\s*$", text))


def extract_tables_with_ocr(pdf_path: Path) -> list[list[list[str]]]:
    # Enforce the cap at execution time, including when workers inherit it.
    os.environ["OMP_THREAD_LIMIT"] = "1"
    try:
        import pytesseract
        from pdf2image import convert_from_path
        from pdf2image.exceptions import PDFPopplerTimeoutError
    except ImportError as exc:
        raise OcrBackendError(f"OCR dependencies unavailable: {exc}") from exc

    try:
        images = convert_from_path(str(pdf_path), dpi=200, timeout=_RASTERIZE_TIMEOUT)
    except (OSError, ValueError, PDFPopplerTimeoutError) as exc:
        raise OcrBackendError(f"failed to rasterize {pdf_path}: {exc}") from exc
    if not images:
        raise OcrBackendError(f"rasterizer returned no pages for {pdf_path}")

    all_rows: list[list[str]] = []
    incomplete_pages: list[str] = []
    started_at = time.monotonic()
    try:
        for page_number, image in enumerate(images, start=1):
            if time.monotonic() - started_at > _OCR_DOCUMENT_BUDGET:
                incomplete_pages.extend(
                    f"page {n}: ocr deadline exceeded"
                    for n in range(page_number, len(images) + 1)
                )
                break
            oriented_image = image
            first_rows: list[list[str]] = []
            try:
                first_text = pytesseract.image_to_string(image, timeout=_OCR_CALL_TIMEOUT)
                first_rows = _parse_ocr_text_to_rows(first_text)
                if not first_rows and _confirmed_nontransaction_page(first_text, image):
                    continue
                oriented_image = _orient_image(image, pytesseract)
                page_rows = first_rows
                oriented_text = first_text
                if oriented_image is not image:
                    oriented_text = pytesseract.image_to_string(
                        oriented_image, timeout=_OCR_CALL_TIMEOUT
                    )
                    oriented_rows = _parse_ocr_text_to_rows(oriented_text)
                    page_rows = _reconcile_rows(first_rows, oriented_rows)
                all_rows.extend(page_rows)
                if not page_rows and not _confirmed_nontransaction_page(
                    oriented_text, oriented_image
                ):
                    incomplete_pages.append(f"page {page_number}: no transaction rows")
            except Exception as exc:
                # Retain diagnosable first-pass rows, but never promote them to success.
                all_rows.extend(first_rows)
                incomplete_pages.append(f"page {page_number}: {exc}")
            finally:
                if oriented_image is not image:
                    oriented_close = getattr(oriented_image, "close", None)
                    if callable(oriented_close):
                        oriented_close()
    finally:
        for image in images:
            image_close = getattr(image, "close", None)
            if callable(image_close):
                image_close()

    table = (
        [["Asset Name", "Transaction Type", "Transaction Date", "Amount"]] + all_rows
        if all_rows
        else []
    )
    if incomplete_pages:
        partial_tables = [table] if table else []
        raise OcrIncompleteError(
            f"incomplete OCR for {pdf_path}: {'; '.join(incomplete_pages)}",
            partial_tables,
        )
    if not table:
        raise OcrIncompleteError(f"OCR produced no rows for {pdf_path}", [])
    return [table]
