"""Row-level processing: turn a parsed table into a list of transaction dicts.

`parse_pdf_table` handles header detection, column mapping, and continuation-row
merging. `_process_row` extracts a single transaction dict (or None) from one
table row using the column index map from `columns.py`.
"""

from analyzer.parsing.cells import (
    _extract_amount_midpoint,
    _extract_date,
    _extract_instrument_type,
    _extract_option_details,
    _extract_owner_code,
    _extract_ticker,
    _extract_transaction_type,
)
from analyzer.parsing.columns import (
    _column_indexes,
    _find_amount_in_row,
    _find_header_row,
    _get_cell,
)


def _process_row(row: list, indexes: dict[str, int] | None = None, next_row: list | None = None) -> dict | None:
    try:
        indexes = indexes or {"asset": 0, "type": 1, "date": 2}
        asset_cell = _get_cell(row, indexes.get("asset"))
        tx_type_cell = _get_cell(row, indexes.get("type"))
        date_cell = _get_cell(row, indexes.get("date"))

        ticker = _extract_ticker(asset_cell)
        tx_type = _extract_transaction_type(tx_type_cell)
        tx_date = _extract_date(date_cell)

        if not ticker and not tx_type and not tx_date and next_row:
            ticker, asset_cell, tx_type_cell, date_cell, tx_type, tx_date = _try_merge_continuation(
                row, next_row, indexes, asset_cell
            )

        if ticker and tx_type and tx_date:
            return _build_row_dict(row, indexes, asset_cell, ticker, tx_type, tx_date)
        return None
    except IndexError:
        return None


def _try_merge_continuation(row, next_row, indexes, asset_cell):
    """Merge a row with the next row when the current row has no ticker/tx/date.

    Returns updated (ticker, asset_cell, tx_type_cell, date_cell, tx_type, tx_date).
    """
    next_asset = _get_cell(next_row, indexes.get("asset"))
    if not next_asset:
        return None, asset_cell, None, None, None, None
    merged = f"{asset_cell or ''} {next_asset}".strip()
    ticker = _extract_ticker(merged)
    if not ticker:
        return None, asset_cell, None, None, None, None
    asset_cell = merged
    tx_type_cell = _get_cell(next_row, indexes.get("type"))
    date_cell = _get_cell(next_row, indexes.get("date"))
    return (
        ticker,
        asset_cell,
        tx_type_cell,
        date_cell,
        _extract_transaction_type(tx_type_cell),
        _extract_date(date_cell),
    )


def _build_row_dict(row, indexes, asset_cell, ticker, tx_type, tx_date) -> dict:
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


def parse_pdf_table(table: list) -> list[dict]:
    if not table or len(table) < 2:
        return []

    header_idx = _find_header_row(table)
    if header_idx is None:
        header_idx = 0

    # Pass next row for 2-row header detection
    next_header_row = table[header_idx + 1] if header_idx + 1 < len(table) else None
    indexes = _column_indexes(table[header_idx], next_header_row)

    data_start = _data_start_offset(table, header_idx, next_header_row)
    data_rows = table[data_start:]
    return _extract_transactions(data_rows, indexes)


def _data_start_offset(table: list, header_idx: int, next_header_row: list | None) -> int:
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


def _extract_transactions(data_rows: list, indexes: dict[str, int]) -> list[dict]:
    results: list[dict] = []
    skip_next = False
    for i, row in enumerate(data_rows):
        if skip_next:
            skip_next = False
            continue
        next_row = data_rows[i + 1] if i + 1 < len(data_rows) else None
        tx = _process_row(row, indexes, next_row)
        if tx:
            results.append(tx)
            # If we merged with next_row, skip it to avoid duplicate
            if next_row and _should_skip_next(row, next_row, indexes):
                skip_next = True
    return results


def _should_skip_next(row, next_row, indexes) -> bool:
    """True when the current row's asset cell had no ticker and merging with
    next_row produced one — so next_row is a continuation, not a new transaction."""
    cur_asset = _get_cell(row, indexes.get("asset"))
    if _extract_ticker(cur_asset):
        return False
    next_asset = _get_cell(next_row, indexes.get("asset"))
    if not next_asset:
        return False
    merged = f"{cur_asset or ''} {next_asset}".strip()
    return bool(_extract_ticker(merged))
