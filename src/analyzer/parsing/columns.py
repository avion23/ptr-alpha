"""Column mapping and header row detection for PTR table parsing.

Maps raw header strings to canonical column roles (asset, owner, type, date,
amount). Handles 1-row and 2-row headers by merging cells vertically when
core columns are missing, then falls back to substring matching.
"""

import re

from analyzer.parsing.cells import clean_text


KNOWN_HEADERS = {
    "asset", "assetname", "description", "desc", "desciption",
    "owner", "ownership", "ownertype", "ownercode", "reportedby",
    "type", "transactiontype", "txtype", "transaction",
    "date", "transactiondate", "txdate", "notifdate", "notificationdate",
    "amount", "transactionamount", "value", "transactionvalue",
    "valueamount", "price", "cost", "proceeds", "tradedamount",
}

_ASSET_CANDS = {"asset", "assetname", "description", "desc", "desciption"}
_OWNER_CANDS = {"owner", "ownership", "ownercode", "reportedby"}
_TYPE_CANDS = {"type", "transactiontype", "txtype", "transaction", "txtype"}
_DATE_CANDS = {"date", "transactiondate", "txdate", "notifdate", "notificationdate"}
_AMOUNT_CANDS = {
    "amount", "transactionamount", "value", "transactionvalue",
    "valueamount", "price", "cost", "proceeds", "tradedamount",
}


def _normalize_header(header: str) -> str:
    return re.sub(r"[^a-z0-9]", "", clean_text(header).lower())


def _column_index(headers: list[str], candidates: set[str]) -> int | None:
    for idx, header in enumerate(headers):
        normalized = _normalize_header(header)
        if normalized in candidates:
            return idx
    return None


def _column_index_substring(headers: list[str], candidates: set[str]) -> int | None:
    """Like _column_index but matches if any candidate is a substring of the normalized header."""
    for idx, header in enumerate(headers):
        normalized = _normalize_header(header)
        for candidate in candidates:
            if candidate in normalized:
                return idx
    return None


def _column_indexes(header: list[str], next_row: list[str] | None = None) -> dict[str, int]:
    headers = [str(cell) for cell in header]
    indexes = _indexes_from_headers(headers)
    # If core columns are missing, try merging with next row (2-row header case)
    if (indexes["asset"] is None or indexes["type"] is None or indexes["date"] is None) and next_row:
        merged = _merge_two_row_headers(headers, next_row)
        indexes = _indexes_from_headers(merged)
        # Substring fallback for all columns (e.g., "Owner Asset" contains "owner")
        for key, cands in [
            ("asset", _ASSET_CANDS),
            ("owner", _OWNER_CANDS),
            ("type", _TYPE_CANDS),
            ("date", _DATE_CANDS),
            ("amount", _AMOUNT_CANDS),
        ]:
            if indexes[key] is None:
                indexes[key] = _column_index_substring(merged, cands)
    if indexes["asset"] is None or indexes["type"] is None or indexes["date"] is None:
        return {"asset": 0, "type": 1, "date": 2}
    return indexes


def _indexes_from_headers(headers: list[str]) -> dict[str, int | None]:
    return {
        "asset": _column_index(headers, _ASSET_CANDS),
        "owner": _column_index(headers, _OWNER_CANDS),
        "type": _column_index(headers, _TYPE_CANDS),
        "date": _column_index(headers, _DATE_CANDS),
        "amount": _column_index(headers, _AMOUNT_CANDS),
    }


def _merge_two_row_headers(headers: list[str], next_row: list) -> list[str]:
    merged: list[str] = []
    for i, cell in enumerate(headers):
        top = cell.strip()
        bottom = str(next_row[i]).strip() if i < len(next_row) else ""
        if top and bottom:
            merged.append(f"{top} {bottom}")
        elif bottom:
            merged.append(bottom)
        else:
            merged.append(top)
    return merged


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


def _find_header_row(table: list, max_scan: int = 3) -> int | None:
    """Scan the first `max_scan` rows for one that contains known column headers."""
    for i, row in enumerate(table[:max_scan]):
        matches = sum(1 for cell in row if _normalize_header(str(cell)) in KNOWN_HEADERS)
        if matches >= 2:
            return i
    return None
