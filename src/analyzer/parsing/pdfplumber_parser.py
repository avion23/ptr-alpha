"""pdfplumber backend: text-layer table extraction.

Benchmark winner for text-based PDFs (0.075s avg). Handles encrypted PDFs
natively. Returns 0 tables on scanned image-only PDFs (no text layer) — caller
falls back to OCR for those. Returns tables in the same format as camelot:
list of 2D string grids.
"""

import logging
from pathlib import Path

from analyzer.parsing.columns import _column_indexes, _find_header_row
from analyzer.parsing.pdftotext_parser import _parse_pdftotext_lines
from analyzer.parsing.rows import parse_pdf_table

logger = logging.getLogger(__name__)


def extract_tables_with_pdfplumber(pdf_path: Path) -> list[list[list[str]]]:
    """Extract transaction tables using pdfplumber (fast text-layer parser).

    Benchmark winner for text-based PDFs: 0.075s/PDF avg, 8/12 success on
    text-based PTRs vs camelot's slower lattice/stream engines. Handles
    encrypted PDFs natively. Returns 0 tables on scanned image-only PDFs
    (no text layer) — caller should fall back to OCR for those.

    Returns tables in the same format as camelot: list of 2D string grids.
    """
    try:
        import pdfplumber
    except ImportError as e:
        logger.debug(f"pdfplumber not available: {e}")
        return []

    tables_out: list[list[list[str]]] = []
    layout_rows: list[list[str]] = []
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                page_text = (page.extract_text(layout=True) or "").replace("\x00", "")
                layout_rows.extend(
                    _parse_pdftotext_lines(
                        page_text.splitlines(),
                        source_prefix=f"pdfplumber:p{page_number}",
                    )
                )

                # page.extract_tables() returns list[list[list[str|None]]]
                page_tables = page.extract_tables()
                for table_number, tbl in enumerate(page_tables):
                    cleaned = _clean_table(tbl)
                    if cleaned is not None:
                        tables_out.append(
                            _attach_source_row_ids(
                                cleaned,
                                f"pdfplumber:p{page_number}:t{table_number}",
                            )
                        )
    except Exception as e:
        logger.debug(f"pdfplumber failed for {pdf_path}: {e}")
        return []

    table_row_count = sum(len(parse_pdf_table(table)) for table in tables_out)
    if layout_rows and len(layout_rows) >= table_row_count:
        return [
            [
                [
                    "Asset Name",
                    "Owner",
                    "Transaction Type",
                    "Transaction Date",
                    "Amount",
                    "Source Row ID",
                ]
            ]
            + layout_rows
        ]
    return tables_out


def _clean_table(tbl: list[list[str | None]]) -> list[list[str]] | None:
    if not tbl or len(tbl) < 2:
        return None
    cleaned = [
        [
            ("" if cell is None else str(cell).replace("\x00", "").strip())
            for cell in row
        ]
        for row in tbl
    ]
    # Drop fully-empty rows
    cleaned = [row for row in cleaned if any(c for c in row)]
    if len(cleaned) < 2:
        return None
    return _expand_flattened_transaction_rows(cleaned)


def _attach_source_row_ids(
    table: list[list[str]], source_prefix: str
) -> list[list[str]]:
    header_idx = _find_header_row(table)
    if header_idx is None:
        return table
    return [
        row
        + [
            "Source Row ID"
            if row_index == header_idx
            else f"{source_prefix}:r{row_index}"
        ]
        for row_index, row in enumerate(table)
    ]


def _expand_flattened_transaction_rows(table: list[list[str]]) -> list[list[str]]:
    """Realign transactions that pdfplumber collapses into one table cell.

    Some encrypted House PDFs alternate normal transaction rows with rows whose
    complete visible text is placed in the first (``ID``) cell. Leaving those
    rows untouched loses the transaction and lets adjacent filing-detail rows
    pollute the next transaction's asset description.
    """
    header_idx = _find_header_row(table)
    if header_idx is None:
        return table

    next_header = table[header_idx + 1] if header_idx + 1 < len(table) else None
    indexes = _column_indexes(table[header_idx], next_header)
    required = (indexes.get("asset"), indexes.get("type"), indexes.get("date"))
    if any(index is None for index in required):
        return table

    expanded: list[list[str]] = []
    for row in table:
        populated = [cell for cell in row if cell.strip()]
        if len(populated) != 1:
            expanded.append(row)
            continue

        parsed = _parse_pdftotext_lines(populated[0].splitlines())
        if not parsed:
            expanded.append(row)
            continue

        for asset, owner, tx_type, tx_date, amount in parsed:
            aligned = [""] * len(row)
            aligned[indexes["asset"]] = asset
            aligned[indexes["type"]] = tx_type
            aligned[indexes["date"]] = tx_date
            if indexes.get("owner") is not None:
                aligned[indexes["owner"]] = owner
            if indexes.get("amount") is not None:
                aligned[indexes["amount"]] = amount
            expanded.append(aligned)
    return expanded
