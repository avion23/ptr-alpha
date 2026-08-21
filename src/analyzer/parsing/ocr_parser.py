"""OCR fallback backend: pytesseract + pdf2image.

Last-resort engine when no text layer is available AND Docling subprocess
fails. Rasterizes each page to a 200dpi image, runs tesseract on it, then
extracts ticker / tx-type / date / amount rows from the resulting plaintext.
"""

import re
import time
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
    try:
        osd = pytesseract.image_to_osd(image, timeout=_OCR_CALL_TIMEOUT)
        match = re.search(r"^Rotate:\s*(90|180|270)\s*$", osd, re.MULTILINE)
        if not match:
            return image
        # OSD reports the clockwise correction. PIL uses counter-clockwise
        # positive angles, so apply its inverse.
        return image.rotate(360 - int(match.group(1)), expand=True)
    except Exception as exc:
        raise OcrBackendError(f"orientation detection failed: {exc}") from exc


def _date_pattern() -> str:
    return r"(?:\d{1,2}/\d{1,2}/(?:\d{2}|\d{4})|\d{4}-\d{2}-\d{2})"


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
    pending_tx: dict | None = None

    for raw_line in text.strip().splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        ticker_match = re.search(r"\(([A-Za-z][A-Za-z0-9.\-]{0,5})\)", stripped)
        amount_match = re.search(r"\$[\d,]+\s*-\s*\$[\d,]+", stripped)
        amount_str = amount_match.group(0) if amount_match else None

        if ticker_match:
            row, pending_tx = _handle_ticker_line(
                stripped, ticker_match, amount_str, pending_tx
            )
            if row is not None:
                rows.append(row)
            continue

        inline_row = _parse_tickerless_inline(stripped, amount_str)
        if inline_row is not None:
            rows.append(inline_row)
            pending_tx = None
            continue
        if pending_tx and not re.search(_date_pattern(), stripped):
            rows.append(
                [
                    stripped,
                    pending_tx["tx_type"],
                    pending_tx["date_str"],
                    pending_tx.get("amount") or "",
                ]
            )
            pending_tx = None
            continue
        pending_tx = _handle_continuation_line(stripped, amount_str)

    return rows


def _handle_ticker_line(
    stripped: str,
    ticker_match: re.Match,
    amount_str: str | None,
    pending_tx: dict | None,
):
    asset_name = stripped[: ticker_match.end()].strip()
    rest = stripped[ticker_match.end() :].strip()
    rest_clean = re.sub(r"\s+", " ", rest).strip().upper()

    tx_type, date_str = _tx_type_and_date(rest_clean, rest)
    if tx_type and date_str:
        return [asset_name, tx_type, date_str, amount_str or ""], None
    if pending_tx:
        return [
            asset_name,
            pending_tx["tx_type"],
            pending_tx["date_str"],
            pending_tx.get("amount") or "",
        ], None
    return None, None


def _tx_type_and_date(rest_clean: str, rest: str) -> tuple[str | None, str | None]:
    tx_type: str | None = None
    # Strip leading asset/owner markers like '[ST]', '[SP]', '[JC]' that can
    # appear between the ticker and the tx code in OCR'd output. Without this,
    # a line like "(AAPL) [ST] P 01/15/2024" misses the tx code and the whole
    # row is dropped.
    body = re.sub(r"^(?:\[[^\]]*\]\s*)+", "", rest_clean).lstrip()
    if body.startswith("P ") or body.startswith("PP "):
        tx_type = TransactionType.PURCHASE.value
    elif body.startswith("S ") or body.startswith("SS "):
        tx_type = TransactionType.SALE.value
    elif body.startswith("E "):
        tx_type = TransactionType.EXCHANGE.value

    # Accept 1- or 2-digit month/day to match cells-level extractor behavior.
    date_match = re.search(_date_pattern(), rest)
    return tx_type, date_match.group(0) if date_match else None


def _handle_continuation_line(stripped: str, amount_str: str | None) -> dict | None:
    rest_clean = re.sub(r"\s+", " ", stripped).upper()

    has_s = (
        " S " in rest_clean
        or rest_clean.startswith("S ")
        or re.search(r"[A-Z0-9]S\s+\d", rest_clean)
    )
    has_p = (
        " P " in rest_clean
        or rest_clean.startswith("P ")
        or re.search(r"[A-Z0-9]P\s+\d", rest_clean)
    )

    if has_s and not has_p:
        tx_type = TransactionType.SALE.value
    elif has_p:
        tx_type = TransactionType.PURCHASE.value
    else:
        tx_type = None

    if tx_type is None:
        return None

    date_match = re.search(_date_pattern(), stripped)
    if not date_match:
        return None
    return {"tx_type": tx_type, "date_str": date_match.group(0), "amount": amount_str}


def _reconcile_rows(*row_sets: list[list[str]]) -> list[list[str]]:
    reconciled: list[list[str]] = []
    seen = set()
    for rows in row_sets:
        for row in rows:
            key = tuple(
                re.sub(r"\s+", " ", str(cell)).strip().casefold() for cell in row
            )
            if key in seen:
                continue
            seen.add(key)
            reconciled.append(row)
    return reconciled


def extract_tables_with_ocr(pdf_path: Path) -> list[list[list[str]]]:
    try:
        from pdf2image import convert_from_path
        import pytesseract
    except ImportError as exc:
        raise OcrBackendError(f"OCR dependencies unavailable: {exc}") from exc

    try:
        images = convert_from_path(str(pdf_path), dpi=200, timeout=_RASTERIZE_TIMEOUT)
    except Exception as exc:
        raise OcrBackendError(f"failed to rasterize {pdf_path}: {exc}") from exc
    if not images:
        raise OcrBackendError(f"rasterizer returned no pages for {pdf_path}")

    all_rows: list[list[str]] = []
    incomplete_pages: list[str] = []
    started_at = time.monotonic()
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
            oriented_image = _orient_image(image, pytesseract)
            page_rows = first_rows
            if oriented_image is not image:
                oriented_text = pytesseract.image_to_string(
                    oriented_image, timeout=_OCR_CALL_TIMEOUT
                )
                oriented_rows = _parse_ocr_text_to_rows(oriented_text)
                page_rows = _reconcile_rows(first_rows, oriented_rows)
            all_rows.extend(page_rows)
            if not page_rows:
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
