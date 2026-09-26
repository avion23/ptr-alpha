"""Row-level processing: turn a parsed table into a list of transaction dicts.

`parse_pdf_table` handles header detection, column mapping, and continuation-row
merging. `_process_row` extracts a single transaction dict (or None) from one
table row using the column index map from `columns.py`.
"""

import re

from analyzer.parsing.cells import (
    _extract_amount_midpoint,
    _extract_date,
    _extract_instrument_type,
    _extract_option_details,
    _extract_owner_code,
    _extract_ticker,
    _extract_transaction_type,
    clean_text,
)
from analyzer.parsing.columns import (
    _column_indexes,
    _find_amount_in_row,
    _find_header_row,
    _get_cell,
)


def _process_row(
    row: list, indexes: dict[str, int | None] | None = None, next_row: list | None = None
) -> tuple[dict | None, bool]:
    try:
        indexes = indexes or {"asset": 0, "type": 1, "date": 2}
        asset_cell = _get_cell(row, indexes.get("asset"))
        if not clean_text(asset_cell):
            return None, False
        tx_type_cell = _get_cell(row, indexes.get("type"))
        date_cell = _get_cell(row, indexes.get("date"))

        ticker = _extract_ticker(asset_cell)
        tx_type = _extract_transaction_type(tx_type_cell)
        tx_date = _extract_date(date_cell)

        if (not tx_type or not tx_date) and next_row:
            combined = _merge_complementary_row(row, next_row, indexes)
            if combined is not None:
                tx, _ = _process_row(combined, indexes)
                return tx, bool(tx)

        if tx_type and tx_date:
            return _build_row_dict(
                row,
                indexes,
                asset_cell,
                ticker,
                tx_type,
                tx_date,
            ), False
        return None, False
    except IndexError:
        return None, False


def _try_merge_continuation(row, next_row, indexes, asset_cell):
    """Merge complementary fields without consuming an independent asset.

    Returns updated (ticker, asset_cell, tx_type_cell, date_cell, tx_type, tx_date).
    """
    combined = _merge_complementary_row(row, next_row, indexes)
    if combined is None:
        return None, asset_cell, None, None, None, None
    asset_cell = _get_cell(combined, indexes.get("asset"))
    ticker = _extract_ticker(asset_cell)
    tx_type_cell = _get_cell(combined, indexes.get("type"))
    date_cell = _get_cell(combined, indexes.get("date"))
    return (
        ticker,
        asset_cell,
        tx_type_cell,
        date_cell,
        _extract_transaction_type(tx_type_cell),
        _extract_date(date_cell),
    )


def _merge_complementary_row(row, candidate, indexes):
    asset_index = indexes.get("asset")
    if asset_index is None:
        return None
    asset = clean_text(_get_cell(row, asset_index))
    suffix = clean_text(_get_cell(candidate, asset_index))
    if not asset or _is_filing_detail_row(suffix):
        return None
    # Only identifiable security suffixes may extend an asset, never another issuer.
    if suffix and not re.fullmatch(
        r"(?:Class\s+[AB]\s*)?(?:\([A-Za-z][A-Za-z0-9.\-]{0,5}\))?(?:\s*\[(?:ST|OP)\])?",
        suffix, re.IGNORECASE,
    ):
        return None
    combined = list(row) + [""] * max(0, len(candidate) - len(row))
    for index, value in enumerate(candidate):
        if index == asset_index:
            continue
        previous = clean_text(_get_cell(row, index))
        current = clean_text(value)
        if previous and current and previous != current:
            return None
        if current:
            combined[index] = value
    combined[asset_index] = f"{asset} {suffix}".strip()
    return combined


def _build_row_dict(
    row, indexes, asset_cell, ticker, tx_type, tx_date, *, amount_owner_row=None
) -> dict:
    # Fix 5: in continuation-row merges, amount and owner live in the next row
    # (the original row only had partial asset text). Use amount_owner_row when provided.
    source = amount_owner_row if amount_owner_row is not None else row
    amount_cell = _get_cell(source, indexes.get("amount"))
    # An unmapped amount must occupy its own cell, outside known columns.
    if amount_cell is None and indexes.get("amount") is None:
        amount_cell = _find_amount_in_row(
            [cell for index, cell in enumerate(source) if index not in indexes.values()]
        )
    amount_raw, amount_midpoint = _extract_amount_midpoint(amount_cell)
    instrument_type = _extract_instrument_type(asset_cell)
    option_details = (
        _extract_option_details(asset_cell) if instrument_type != "stock" else {}
    )
    return {
        "ticker": ticker,
        "transaction_type": tx_type,
        "transaction_date": tx_date,
        "owner_code": _extract_owner_code(_get_cell(source, indexes.get("owner"))),
        "amount_raw": amount_raw,
        "amount_midpoint": amount_midpoint,
        "instrument_type": instrument_type,
        "strike_price": option_details.get("strike_price"),
        "expiry_date": option_details.get("expiry_date"),
        "asset_description": clean_text(asset_cell)[:500] if asset_cell else None,
        "source_row_id": clean_text(_get_cell(row, indexes.get("source_row_id")))
        or None,
    }


def parse_pdf_table(table: list) -> list[dict]:
    if not table:
        return []

    header_idx = _find_header_row(table)
    if header_idx is None:
        # Some extraction backends return only data rows.  Treating row zero as
        # a header silently discarded the first disclosure in that case.
        return _extract_transactions(table, {"asset": 0, "type": 1, "date": 2})

    # Pass next row for 2-row header detection
    next_header_row = table[header_idx + 1] if header_idx + 1 < len(table) else None
    indexes = _column_indexes(table[header_idx], next_header_row)
    indexes["source_row_id"] = _source_row_id_index(table[header_idx])

    data_start = _data_start_offset(table, header_idx, next_header_row)
    data_rows = table[data_start:]
    return _extract_transactions(data_rows, indexes)


def _source_row_id_index(header: list) -> int | None:
    for index, cell in enumerate(header):
        normalized = "".join(
            character for character in clean_text(cell).lower() if character.isalnum()
        )
        if normalized == "sourcerowid":
            return index
    return None


def _data_start_offset(
    table: list, header_idx: int, next_header_row: list | None
) -> int:
    """Determine how many rows to skip after the header (1 or 2)."""
    data_start = header_idx + 1
    if next_header_row is None:
        return data_start
    # If merged headers were needed (core columns were None before merge), skip 2 rows
    pre_merge_indexes = _column_indexes(table[header_idx])
    if (
        pre_merge_indexes.get("asset") is None
        or pre_merge_indexes.get("type") is None
        or pre_merge_indexes.get("date") is None
    ):
        return header_idx + 2
    return data_start


def _extract_transactions(data_rows: list, indexes: dict[str, int | None]) -> list[dict]:
    results: list[dict] = []
    i = 0
    while i < len(data_rows):
        row = data_rows[i]
        next_rows = data_rows[i + 1 : i + 4]
        tx, consumed = _process_with_continuations(row, next_rows, indexes)
        if tx:
            results.append(tx)
        i += consumed + 1
    return results


def _process_with_continuations(
    row: list, next_rows: list[list], indexes: dict[str, int | None]
) -> tuple[dict | None, int]:
    """Process a row and up to three physical continuation rows."""
    if _is_filing_detail_row(_get_cell(row, indexes.get("asset"))):
        return None, 0

    combined = list(row)
    consumed = 0
    for candidate in next_rows:
        tx, _ = _process_row(combined, indexes)
        if tx and any(
            clean_text(_get_cell(candidate, indexes.get(key)))
            for key in ("type", "date")
        ):
            break
        merged = _merge_complementary_row(combined, candidate, indexes)
        if merged is None:
            break
        combined = merged
        consumed += 1
    tx, _ = _process_row(combined, indexes)
    return tx, consumed


def _is_filing_detail_row(asset_cell: str | None) -> bool:
    """Identify filing metadata that must not prefix the next asset row."""
    text = clean_text(asset_cell).upper()
    if not text:
        return False
    if text.startswith(("DESCRIPTION:", "D:")) and (
        "[ST]" in text or "[OP]" in text
    ):
        return False
    return text.startswith(
        (
            "FILING STATUS:",
            "F S:",
            "SOURCE OF:",
            "S O:",
            "SUBHOLDING OF:",
            "DESCRIPTION:",
            "D:",
            "LOCATION:",
            "L:",
        )
    )
