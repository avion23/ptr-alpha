import logging
import re
from pathlib import Path

import pandas as pd

from analyzer.exceptions import ParsingError
from analyzer.models import TransactionType

logger = logging.getLogger(__name__)

def clean_text(text: str | None) -> str:
    if text is None:
        return ""
    return re.sub(r'\s+', ' ', str(text)).strip()

def _extract_ticker(asset_cell: str | None) -> str | None:
    if not asset_cell:
        return None
    ticker_match = re.search(r'\(([A-Z][A-Z.\-]{0,5})\)', asset_cell)
    return ticker_match.group(1) if ticker_match else None

def _extract_transaction_type(tx_type_cell: str | None) -> str | None:
    if not tx_type_cell:
        return None
    raw = tx_type_cell.strip()
    s = raw.lower()
    # Handle "(partial)" suffix: "P (partial)", "S (partial)", "Purchase (partial)", etc.
    s_stripped = re.sub(r'\s*\(partial\)\s*$', '', s).strip()
    if s_stripped in ('p', 'purchase', 'buy'):
        return TransactionType.PURCHASE.value
    if s_stripped in ('s', 'sale', 'sold'):
        return TransactionType.SALE.value
    if 'purchase' in s or 'buy' in s:
        return TransactionType.PURCHASE.value
    if 'sale' in s or 'sell' in s or 'sold' in s:
        return TransactionType.SALE.value
    if s_stripped.startswith('p') and len(s_stripped) <= 2:
        return TransactionType.PURCHASE.value
    if s_stripped.startswith('s') and len(s_stripped) <= 2:
        return TransactionType.SALE.value
    return None

def _extract_date(date_cell: str | None) -> str | None:
    if not date_cell:
        return None
    # Support both MM/DD/YYYY and YYYY-MM-DD formats
    date_match = re.search(r'(\d{2}/\d{2}/\d{4}|\d{4}-\d{2}-\d{2})', date_cell)
    return date_match.group(1) if date_match else None


def _extract_owner_code(owner_cell: str | None) -> str | None:
    owner = clean_text(owner_cell).upper()
    if not owner:
        return None
    if owner.startswith("DEPENDENT"):
        return "DC"
    if owner.startswith("SPOUSE"):
        return "SP"
    if owner.startswith("JOINT"):
        return "J"
    if owner.startswith("SELF"):
        return "S"
    if owner in ("DC", "SP", "J", "S"):
        return owner
    return owner[:8]


def _extract_instrument_type(asset_cell: str | None) -> str:
    """Detect whether an asset description is a stock, call option, or put option.

    Handles common PTR formats:
      - "NVIDIA Corp Common Stock Call Option (NVDA)"
      - "NVDA Call $120 Exp 12/20/2024"
      - "Call Option" / "Put Option" as separate field in asset text
      - Bare "call" / "put" keywords
    """
    if not asset_cell:
        return 'stock'
    text = asset_cell.lower()
    # Put detection — check before call since "put call" is unlikely but "call put" might appear
    if re.search(r'\bput\s*(?:option|opt)\b', text) or re.search(r'\bput\b', text):
        return 'put'
    # Call detection — broad match including "call option", "call opt"
    if re.search(r'\bcall\s*(?:option|opt)\b', text) or re.search(r'\bcall\b', text):
        return 'call'
    # Generic "option" without call/put qualifier — try to infer from context
    if re.search(r'\boption\b', text):
        # Heuristic: if there's a strike/expiry pattern, it's an option but type unknown — default to 'call'
        if re.search(r'\b(?:strike|exp|strike\s*price|expir)\b', text):
            return 'call'
    return 'stock'


def _extract_option_details(asset_cell: str | None) -> dict:
    """Extract strike price and expiry date from an option asset description.

    Returns dict with optional 'strike_price' (float) and 'expiry_date' (str MM/DD/YYYY).
    Handles formats:
      - "Strike $150" / "Strike: 150.00"
      - "$120" preceding "Exp" in "NVDA Call $120 Exp 12/20/2024"
      - "Exp MM/DD/YYYY" / "Expire MM/DD/YYYY" / "Expiring MM/DD/YYYY"
      - "Exp 12/20/2024" (bare exp abbreviation)
    """
    details: dict = {}
    if not asset_cell:
        return details
    # Strike price: "Strike $150" or "Strike: 150.00"
    strike_match = re.search(r'(?:strike[:\s]*\$?)(\d+(?:\.\d+)?)', asset_cell, re.IGNORECASE)
    if strike_match:
        details['strike_price'] = float(strike_match.group(1))
    else:
        # Fallback: dollar amount before Exp/expiry, e.g. "$120 Exp 12/20/2024"
        strike_fallback = re.search(r'\$(\d+(?:\.\d+)?)\s+(?:exp|strike)', asset_cell, re.IGNORECASE)
        if strike_fallback:
            details['strike_price'] = float(strike_fallback.group(1))

    # Expiry date: "Exp MM/DD/YYYY" or "Expire: MM/DD/YYYY" or "Expiring MM/DD/YYYY"
    exp_match = re.search(r'(?:exp(?:ir(?:e|ation|ing)?)?[:\s]+(\d{2}/\d{2}/\d{4}))', asset_cell, re.IGNORECASE)
    if exp_match:
        details['expiry_date'] = exp_match.group(1)
    return details


def _extract_amount_midpoint(amount_cell: str | None) -> tuple[str | None, float | None]:
    amount = clean_text(amount_cell)
    if not amount:
        return None, None
    values = [float(value.replace(",", "")) for value in re.findall(r'\$([0-9][0-9,]*)', amount)]
    if not values:
        return amount, None
    return amount, sum(values[:2]) / min(len(values), 2)


def _normalize_header(header: str) -> str:
    return re.sub(r"[^a-z0-9]", "", clean_text(header).lower())


def _column_index(headers: list[str], candidates: set[str]) -> int | None:
    for idx, header in enumerate(headers):
        normalized = _normalize_header(header)
        if normalized in candidates:
            return idx
    return None


def _column_indexes(header: list[str]) -> dict[str, int]:
    headers = [str(cell) for cell in header]
    indexes = {
        "asset": _column_index(headers, {"asset", "assetname", "description", "desc", "desciption"}),
        "owner": _column_index(headers, {"owner", "ownership", "ownercode", "reportedby"}),
        "type": _column_index(headers, {"type", "transactiontype", "txtype", "transaction", "txtype"}),
        "date": _column_index(headers, {"date", "transactiondate", "txdate", "notifdate", "notificationdate"}),
        "amount": _column_index(headers, {
            "amount", "transactionamount", "value", "transactionvalue",
            "valueamount", "price", "cost", "proceeds", "tradedamount",
        }),
    }
    if indexes["asset"] is None or indexes["type"] is None or indexes["date"] is None:
        return {"asset": 0, "type": 1, "date": 2}
    return indexes


def _get_cell(row: list, index: int | None) -> str | None:
    if index is None or index >= len(row):
        return None
    return str(row[index])


def _find_amount_in_row(row: list) -> str | None:
    """Fallback: scan all cells for a '$X,XXX - $X,XXX' or '$X,XXX' amount pattern."""
    amount_re = re.compile(r'\$\d[\d,]*(?:\s*-\s*\$\d[\d,]*)?')
    for cell in row:
        if cell is None:
            continue
        text = str(cell).strip()
        match = amount_re.search(text)
        if match:
            return match.group(0)
    return None


def _process_row(row: list, indexes: dict[str, int] | None = None) -> dict | None:
    try:
        indexes = indexes or {"asset": 0, "type": 1, "date": 2}
        asset_cell = _get_cell(row, indexes.get("asset"))
        tx_type_cell = _get_cell(row, indexes.get("type"))
        date_cell = _get_cell(row, indexes.get("date"))

        ticker = _extract_ticker(asset_cell)
        tx_type = _extract_transaction_type(tx_type_cell)
        tx_date = _extract_date(date_cell)

        if ticker and tx_type and tx_date:
            amount_cell = _get_cell(row, indexes.get("amount"))
            # Fallback: if amount column not mapped, search all cells for $ pattern
            if amount_cell is None and indexes.get("amount") is None:
                amount_cell = _find_amount_in_row(row)
            amount_raw, amount_midpoint = _extract_amount_midpoint(amount_cell)
            instrument_type = _extract_instrument_type(asset_cell)
            option_details = _extract_option_details(asset_cell) if instrument_type != 'stock' else {}
            return {
                'ticker': ticker,
                'transaction_type': tx_type,
                'transaction_date': tx_date,
                'owner_code': _extract_owner_code(_get_cell(row, indexes.get("owner"))),
                'amount_raw': amount_raw,
                'amount_midpoint': amount_midpoint,
                'instrument_type': instrument_type,
                'strike_price': option_details.get('strike_price'),
                'expiry_date': option_details.get('expiry_date'),
            }
        return None
    except IndexError:
        return None

KNOWN_HEADERS = {
    "asset", "assetname", "description", "desc", "desciption",
    "owner", "ownership", "ownertype", "ownercode", "reportedby",
    "type", "transactiontype", "txtype", "transaction",
    "date", "transactiondate", "txdate", "notifdate", "notificationdate",
    "amount", "transactionamount", "value", "transactionvalue",
    "valueamount", "price", "cost", "proceeds", "tradedamount",
}


def _find_header_row(table: list, max_scan: int = 3) -> int | None:
    """Scan the first `max_scan` rows for one that contains known column headers."""
    for i, row in enumerate(table[:max_scan]):
        matches = sum(1 for cell in row if _normalize_header(str(cell)) in KNOWN_HEADERS)
        if matches >= 2:
            return i
    return None


def parse_pdf_table(table: list) -> list[dict]:
    if not table or len(table) < 2:
        return []

    header_idx = _find_header_row(table)
    if header_idx is None:
        header_idx = 0
    indexes = _column_indexes(table[header_idx])
    data_rows = table[header_idx + 1:]
    return [tx for tx in (_process_row(row, indexes) for row in data_rows) if tx]


def normalize_house_metadata(content: str) -> pd.DataFrame:
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

def consolidate_transactions(pdf_transactions: dict[Path, list[dict]], member_metadata: dict[str, dict]) -> pd.DataFrame:
    if not pdf_transactions:
        return pd.DataFrame()

    all_transactions = []
    for pdf_path, transactions in pdf_transactions.items():
        if not transactions:
            continue

        doc_id = pdf_path.stem
        member_info = member_metadata.get(doc_id)
        if not member_info:
            logger.warning(f"No metadata found for doc_id={doc_id}, skipping {len(transactions)} transaction(s)")
            continue

        for tx in transactions:
            all_transactions.append({
                'doc_id': doc_id,
                'member': f"{member_info['First']} {member_info['Last']}",
                'transaction_date': tx['transaction_date'],
                'disclosure_date': member_info['FilingDate'],
                'ticker': tx['ticker'],
                'transaction_type': tx['transaction_type'],
                'owner_code': tx.get('owner_code'),
                'amount_raw': tx.get('amount_raw'),
                'amount_midpoint': tx.get('amount_midpoint'),
                'instrument_type': tx.get('instrument_type', 'stock'),
                'strike_price': tx.get('strike_price'),
                'expiry_date': tx.get('expiry_date'),
            })

    if not all_transactions:
        return pd.DataFrame()

    df = pd.DataFrame(all_transactions)
    df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
    df['disclosure_date'] = pd.to_datetime(df['disclosure_date'], errors='coerce')

    return df.dropna(subset=['transaction_date', 'disclosure_date'])


def _parse_ocr_text_to_rows(text: str) -> list[list[str]]:
    rows = []
    lines = text.strip().split('\n')

    pending_tx = None

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue

        ticker_match = re.search(r'\(([A-Z][A-Z0-9.\-]{0,5})\)', stripped)
        amount_match = re.search(r'\$[\d,]+\s*-\s*\$[\d,]+', stripped)
        amount_str = amount_match.group(0) if amount_match else None

        if ticker_match:
            asset_name = stripped[:ticker_match.end()].strip()
            rest = stripped[ticker_match.end():].strip()
            rest_clean = re.sub(r'\s+', ' ', rest).strip().upper()

            tx_type = None
            date_str = None

            if rest_clean.startswith('P ') or rest_clean.startswith('PP '):
                tx_type = TransactionType.PURCHASE.value
            elif rest_clean.startswith('S ') or rest_clean.startswith('SS '):
                tx_type = TransactionType.SALE.value

            date_match = re.search(r'(\d{2}/\d{2}/\d{4}|\d{4}-\d{2}-\d{2})', rest)
            if date_match:
                date_str = date_match.group(1)

            if tx_type and date_str:
                rows.append([asset_name, tx_type, date_str, amount_str or ""])
            elif pending_tx:
                rows.append([asset_name, pending_tx['tx_type'], pending_tx['date_str'], pending_tx.get('amount') or ""])

            pending_tx = None

        else:
            rest_clean = re.sub(r'\s+', ' ', stripped).upper()

            has_s = ' S ' in rest_clean or rest_clean.startswith('S ') or re.search(r'[A-Z0-9]S\s+\d', rest_clean)
            has_p = ' P ' in rest_clean or rest_clean.startswith('P ') or re.search(r'[A-Z0-9]P\s+\d', rest_clean)

            if has_s and not has_p:
                tx_type = TransactionType.SALE.value
            elif has_p:
                tx_type = TransactionType.PURCHASE.value
            else:
                tx_type = None

            if tx_type:
                date_match = re.search(r'(\d{2}/\d{2}/\d{4}|\d{4}-\d{2}-\d{2})', stripped)
                if date_match:
                    pending_tx = {'tx_type': tx_type, 'date_str': date_match.group(1), 'amount': amount_str}

    return rows


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
        text = pytesseract.image_to_string(image)
        all_rows.extend(_parse_ocr_text_to_rows(text))

    if not all_rows:
        return []

    table = [['Asset Name', 'Transaction Type', 'Transaction Date', 'Amount']] + all_rows
    return [table]
