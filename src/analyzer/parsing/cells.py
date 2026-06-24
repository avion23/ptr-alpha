"""Cell-level extraction helpers for PTR table cells.

Each helper extracts a single piece of information from a raw cell string:
ticker, transaction type, date, owner code, amount, instrument type, option
details. Used by row-level processing in `rows.py` after column mapping.
"""

import re

from analyzer.models import TransactionType


def clean_text(text: str | None) -> str:
    if text is None:
        return ""
    return re.sub(r'\s+', ' ', str(text)).strip()


def _extract_ticker(asset_cell: str | None) -> str | None:
    if not asset_cell:
        return None
    ticker_match = re.search(r'\(([A-Za-z][A-Za-z0-9.\-]{0,5})\)', asset_cell)
    if ticker_match:
        return ticker_match.group(1).upper()
    # Also match $TICKER and TICKER: formats
    dollar_match = re.search(r'\$([A-Za-z][A-Za-z0-9.\-]{0,5})', asset_cell)
    if dollar_match:
        return dollar_match.group(1).upper()
    colon_match = re.search(r'\b([A-Za-z][A-Za-z0-9.\-]{0,5}):', asset_cell)
    if colon_match:
        return colon_match.group(1).upper()
    return None


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
    if s_stripped in ('e', 'exchange'):
        return TransactionType.EXCHANGE.value
    if 'purchase' in s or 'buy' in s:
        return TransactionType.PURCHASE.value
    if 'sale' in s or 'sell' in s or 'sold' in s:
        return TransactionType.SALE.value
    if 'exchange' in s:
        return TransactionType.EXCHANGE.value
    if s_stripped.startswith('p') and len(s_stripped) <= 2:
        return TransactionType.PURCHASE.value
    if s_stripped.startswith('s') and len(s_stripped) <= 2:
        return TransactionType.SALE.value
    return None


def _extract_date(date_cell: str | None) -> str | None:
    if not date_cell:
        return None
    # Support MM/DD/YYYY, YYYY-MM-DD, and MM/DD/YY formats
    date_match = re.search(r'(\d{1,2}/\d{1,2}/\d{4}|\d{4}-\d{2}-\d{2})', date_cell)
    if date_match:
        return date_match.group(1)
    # MM/DD/YY format — normalize 2-digit year
    date_short = re.search(r'(\d{1,2}/\d{1,2}/(\d{2}))', date_cell)
    if date_short:
        full = date_short.group(1)
        yy = int(date_short.group(2))
        year_prefix = "19" if yy >= 50 else "20"
        date_part = full.rsplit('/', 1)[0]
        return f"{date_part}/{year_prefix}{date_short.group(2)}"
    return None


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
      - "Stock Option (NVDA)" — a standalone phrase without explicit call/put
    """
    if not asset_cell:
        return 'stock'
    text = asset_cell.lower()

    # Put detection — use compound patterns first; avoid matching bare "put" (too many false positives)
    if re.search(r'\bput\s+option\b', text):
        return 'put'
    if re.search(r'\bput\s+opt\b', text):
        return 'put'
    if re.search(r'\bstock\s+put\b', text):
        return 'put'
    if re.search(r'\bput\b.*\b(?:strike|exp|expir)\b', text):
        return 'put'

    # Call detection — use compound patterns first; avoid matching bare "call" (too many false positives)
    if re.search(r'\bcall\s+option\b', text):
        return 'call'
    if re.search(r'\bcall\s+opt\b', text):
        return 'call'
    if re.search(r'\bstock\s+call\b', text):
        return 'call'
    if re.search(r'\bcall\b.*\b(?:strike|exp|expir)\b', text):
        return 'call'

    # "Stock Option" standalone — a common PTR label; default to 'call' as most
    # stock options in congressional disclosures are call options
    if re.search(r'\bstock\s+option\b', text):
        return 'call'

    # Generic "option" without call/put qualifier — try to infer from context
    if re.search(r'\boption\b', text):
        if re.search(r'\b(?:strike|exp|expir)\b', text):
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
      - "Strike Price $150.00"
      - "NVDA Call Option $120 Exp 12/20/2024"
    """
    details: dict = {}
    if not asset_cell:
        return details

    # Strike price: "Strike Price $150", "Strike $150", "Strike: 150.00"
    strike_match = re.search(r'strike\s*(?:price)?[:\s]*\$?(\d+(?:\.\d+)?)', asset_cell, re.IGNORECASE)
    if strike_match:
        details['strike_price'] = float(strike_match.group(1))
    else:
        # Fallback: dollar amount before Exp/expiry, e.g. "$120 Exp 12/20/2024"
        strike_fallback = re.search(r'\$(\d+(?:\.\d+)?)\s+(?:exp|strike)', asset_cell, re.IGNORECASE)
        if strike_fallback:
            details['strike_price'] = float(strike_fallback.group(1))

    # Expiry date: "Exp MM/DD/YYYY" / "Expire: MM/DD/YYYY" / "Expiring MM/DD/YYYY"
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
