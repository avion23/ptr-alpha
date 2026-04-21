import logging
import pandas as pd
import re
from .exceptions import ParsingError

logger = logging.getLogger(__name__)

def clean_text(text):
    if text is None:
        return ""
    return re.sub(r'\s+', ' ', str(text)).strip()

def extract_ticker_from_name(asset_name):
    if not asset_name:
        return None

    match = re.search(r'\(([A-Z][A-Z.\-]{1,5})\)', asset_name)
    if match:
        ticker = match.group(1).strip()
        if 1 <= len(ticker) <= 5:
            return ticker
    return None

def _extract_ticker(asset_cell):
    if not asset_cell:
        return None
    ticker_match = re.search(r'\(([A-Z][A-Z.\-]{0,9})\)', asset_cell)
    return ticker_match.group(1) if ticker_match else None

def _extract_transaction_type(tx_type_cell):
    if not tx_type_cell:
        return None
    tx_type_lower = tx_type_cell.lower()
    if 'purchase' in tx_type_lower:
        return 'Purchase'
    elif 'sale' in tx_type_lower:
        return 'Sale'
    return None

def _extract_date(date_cell):
    if not date_cell:
        return None
    # Support both MM/DD/YYYY and YYYY-MM-DD formats
    date_match = re.search(r'(\d{2}/\d{2}/\d{4}|\d{4}-\d{2}-\d{2})', date_cell)
    return date_match.group(1) if date_match else None

def _process_row(row):
    try:
        asset_cell = str(row[0])
        tx_type_cell = str(row[1])
        date_cell = str(row[2])

        ticker = _extract_ticker(asset_cell)
        tx_type = _extract_transaction_type(tx_type_cell)
        tx_date = _extract_date(date_cell)

        if ticker and tx_type and tx_date:
            return {
                'ticker': ticker,
                'transaction_type': tx_type,
                'transaction_date': tx_date,
            }
        return None
    except IndexError:
        return None

def parse_pdf_table(table):
    if not table or len(table) < 2:
        return []

    return [tx for tx in (_process_row(row) for row in table[1:]) if tx]


def normalize_house_metadata(content):
    if not content:
        raise ParsingError("Empty metadata content")

    lines = content.strip().split('\n')
    if len(lines) < 2:
        raise ParsingError("Insufficient metadata lines")

    header = [col.strip() for col in lines[0].split('\t')]
    data = []
    for line in lines[1:]:
        if line.strip():
            row = [col.strip() for col in line.split('\t')]
            if len(row) >= len(header):
                data.append(row[:len(header)])

    if not data:
        raise ParsingError("No data rows in metadata")

    df = pd.DataFrame(data, columns=header)

    if 'FilingDate' not in df.columns:
        raise ParsingError("Missing FilingDate column in metadata")

    df['FilingDate'] = pd.to_datetime(df['FilingDate'], errors='coerce')
    df = df.dropna(subset=['FilingDate'])

    if df.empty:
        raise ParsingError("No valid filing dates in metadata")

    return df

def consolidate_transactions(pdf_transactions, member_metadata):
    if not pdf_transactions:
        return pd.DataFrame()

    all_transactions = []
    for pdf_path, transactions in pdf_transactions.items():
        if not transactions:
            continue

        doc_id = pdf_path.stem
        member_info = member_metadata.get(doc_id)
        if not member_info:
            continue

        for tx in transactions:
            all_transactions.append({
                'doc_id': doc_id,
                'member': f"{member_info['First']} {member_info['Last']}",
                'transaction_date': tx['transaction_date'],
                'disclosure_date': member_info['FilingDate'],
                'ticker': tx['ticker'],
                'transaction_type': tx['transaction_type'],
            })

    if not all_transactions:
        return pd.DataFrame()

    df = pd.DataFrame(all_transactions)
    df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
    df['disclosure_date'] = pd.to_datetime(df['disclosure_date'], errors='coerce')

    return df.dropna(subset=['transaction_date'])


def _parse_ocr_text_to_rows(text):
    rows = []
    lines = text.strip().split('\n')

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Look for ticker pattern
        ticker_match = re.search(r'\(([A-Z][A-Z0-9.\-]{1,5})\)', line)
        if not ticker_match:
            continue

        asset_name = line[:ticker_match.end()].strip()
        rest = line[ticker_match.end():].strip()

        # Handle single-letter transaction codes: P=Purchase, S=Sale, PP=Purchase
        tx_type = None
        rest_clean = re.sub(r'\s+', ' ', rest)
        if re.match(r'^P\s', rest_clean) or rest_clean.startswith('PP '):
            tx_type = 'Purchase'
        elif re.match(r'^S\s', rest_clean) or rest_clean.startswith('S '):
            tx_type = 'Sale'

        # Look for date in rest
        date_match = re.search(r'(\d{2}/\d{2}/\d{4}|\d{4}-\d{2}-\d{2})', rest)
        date_str = date_match.group(1) if date_match else None

        if tx_type and date_str:
            rows.append([asset_name, tx_type, date_str])

    return rows


def extract_tables_with_ocr(pdf_path):
    try:
        from pdf2image import convert_from_path
        import pytesseract
    except ImportError as e:
        logger.warning(f"OCR dependencies not available: {e}")
        return []

    try:
        images = convert_from_path(str(pdf_path), dpi=300)
    except Exception as e:
        logger.warning(f"Failed to convert PDF to images for OCR {pdf_path}: {e}")
        return []

    all_rows = []
    for image in images:
        text = pytesseract.image_to_string(image)
        all_rows.extend(_parse_ocr_text_to_rows(text))

    if not all_rows:
        return []

    table = [['Asset Name', 'Transaction Type', 'Transaction Date']] + all_rows
    return [table]