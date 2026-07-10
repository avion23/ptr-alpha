"""OCR fallback backend: pytesseract + pdf2image.

Last-resort engine when no text layer is available AND Docling subprocess
fails. Rasterizes each page to a 200dpi image, runs tesseract on it, then
extracts ticker / tx-type / date / amount rows from the resulting plaintext.
"""

import logging
import re
from pathlib import Path

from analyzer.models import TransactionType

logger = logging.getLogger(__name__)


def _orient_image(image, pytesseract):
    """Return an upright image when Tesseract detects a rotated scan."""
    try:
        osd = pytesseract.image_to_osd(image)
        match = re.search(r'^Rotate:\s*(90|180|270)\s*$', osd, re.MULTILINE)
        if not match:
            return image
        # OSD reports the clockwise correction. PIL uses counter-clockwise
        # positive angles, so apply its inverse.
        return image.rotate(360 - int(match.group(1)), expand=True)
    except Exception as e:
        logger.debug("OCR orientation detection failed: %s", e)
        return image


def _parse_ocr_text_to_rows(text: str) -> list[list[str]]:
    rows: list[list[str]] = []
    lines = text.strip().split('\n')
    pending_tx: dict | None = None

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        ticker_match = re.search(r'\(([A-Za-z][A-Za-z0-9.\-]{0,5})\)', stripped)
        amount_match = re.search(r'\$[\d,]+\s*-\s*\$[\d,]+', stripped)
        amount_str = amount_match.group(0) if amount_match else None

        if ticker_match:
            row, pending_tx = _handle_ticker_line(stripped, ticker_match, amount_str, pending_tx)
            if row is not None:
                rows.append(row)
        else:
            pending_tx = _handle_continuation_line(stripped, amount_str)

    return rows


def _handle_ticker_line(stripped: str, ticker_match: re.Match, amount_str: str | None, pending_tx: dict | None):
    asset_name = stripped[:ticker_match.end()].strip()
    rest = stripped[ticker_match.end():].strip()
    rest_clean = re.sub(r'\s+', ' ', rest).strip().upper()

    tx_type, date_str = _tx_type_and_date(rest_clean, rest)
    if tx_type and date_str:
        return [asset_name, tx_type, date_str, amount_str or ""], None
    if pending_tx:
        return [asset_name, pending_tx['tx_type'], pending_tx['date_str'], pending_tx.get('amount') or ""], None
    return None, None


def _tx_type_and_date(rest_clean: str, rest: str) -> tuple[str | None, str | None]:
    tx_type: str | None = None
    # Strip leading asset/owner markers like '[ST]', '[SP]', '[JC]' that can
    # appear between the ticker and the tx code in OCR'd output. Without this,
    # a line like "(AAPL) [ST] P 01/15/2024" misses the tx code and the whole
    # row is dropped.
    body = re.sub(r'^(?:\[[^\]]*\]\s*)+', '', rest_clean).lstrip()
    if body.startswith('P ') or body.startswith('PP '):
        tx_type = TransactionType.PURCHASE.value
    elif body.startswith('S ') or body.startswith('SS '):
        tx_type = TransactionType.SALE.value
    elif body.startswith('E '):
        tx_type = TransactionType.EXCHANGE.value

    # Accept 1- or 2-digit month/day to match cells-level extractor behavior.
    date_match = re.search(r'(\d{1,2}/\d{1,2}/\d{4}|\d{4}-\d{2}-\d{2})', rest)
    return tx_type, date_match.group(1) if date_match else None


def _handle_continuation_line(stripped: str, amount_str: str | None) -> dict | None:
    rest_clean = re.sub(r'\s+', ' ', stripped).upper()

    has_s = ' S ' in rest_clean or rest_clean.startswith('S ') or re.search(r'[A-Z0-9]S\s+\d', rest_clean)
    has_p = ' P ' in rest_clean or rest_clean.startswith('P ') or re.search(r'[A-Z0-9]P\s+\d', rest_clean)

    if has_s and not has_p:
        tx_type = TransactionType.SALE.value
    elif has_p:
        tx_type = TransactionType.PURCHASE.value
    else:
        tx_type = None

    if tx_type is None:
        return None

    date_match = re.search(r'(\d{1,2}/\d{1,2}/\d{4}|\d{4}-\d{2}-\d{2})', stripped)
    if not date_match:
        return None
    return {'tx_type': tx_type, 'date_str': date_match.group(1), 'amount': amount_str}


def extract_tables_with_ocr(pdf_path: Path) -> list[list[list[str]]]:
    try:
        from pdf2image import convert_from_path
        import pytesseract
    except ImportError as e:
        logger.warning(f"OCR dependencies not available: {e}")
        return []

    try:
        images = convert_from_path(str(pdf_path), dpi=200)
    except (OSError, ValueError) as e:
        logger.warning(f"Failed to convert PDF to images for OCR {pdf_path}: {e}")
        return []

    all_rows = []
    for image in images:
        oriented_image = image
        try:
            text = pytesseract.image_to_string(image)
            page_rows = _parse_ocr_text_to_rows(text)
            if not page_rows:
                oriented_image = _orient_image(image, pytesseract)
                if oriented_image is not image:
                    text = pytesseract.image_to_string(oriented_image)
                    page_rows = _parse_ocr_text_to_rows(text)
            all_rows.extend(page_rows)
        except Exception as e:
            # Tesseract can raise runtime errors per page (missing binary,
            # corrupt image, decode failures). Skip the page rather than abort
            # the whole PDF — other pages may OCR cleanly.
            logger.warning(f"OCR failed for one page of {pdf_path}: {e}")
            continue
        finally:
            # pdf2image returns PIL images that hold raster data; close them
            # promptly to avoid memory pressure on large multi-page PDFs.
            if oriented_image is not image:
                oriented_close = getattr(oriented_image, "close", None)
                if callable(oriented_close):
                    oriented_close()
            image_close = getattr(image, "close", None)
            if callable(image_close):
                image_close()
    if not all_rows:
        return []

    table = [['Asset Name', 'Transaction Type', 'Transaction Date', 'Amount']] + all_rows
    return [table]
