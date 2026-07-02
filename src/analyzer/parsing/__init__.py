"""PDF table extraction for House PTR disclosures.

Engine pipeline (benchmark-driven, see data/pdf_converter_benchmark_*.json):
  1. pdfplumber   — primary, 0.075s/PDF, MIT, handles text + encrypted PDFs
  2. camelot lattice/stream — fallback for rulled/stream tables
  3. pdftotext    — fallback for encrypted PDFs where pdfplumber returns 0
  4. Docling      — OCR fallback for SCANNED IMAGE PDFs (no text layer).
                    MIT license, IBM TableFormer; runs via `uvx` subprocess.
                    Benchmark winner over Marker (better accuracy 75% vs 58%,
                    MIT vs GPL, no Cyrillic-E OCR bug).
  5. pytesseract  — last resort when docling is unavailable

Benchmark results on 30 PTR PDFs (12 successful parses + 18 known failures):
  - pdfplumber: 8/12 SUCCESS recovery, 4/18 FAILED recovery, 0.075s avg
  - Docling:    9/12 SUCCESS recovery, 6/18 FAILED recovery, 43.4s avg
  - Marker:     7/12 SUCCESS recovery, 6/18 FAILED recovery, 49.7s avg
  (FAILED recovery is bounded by ~11/18 being bonds with no ticker by design)

Decision: hybrid pdfplumber-primary + Docling-OCR. pdfplumber is 580x faster
than Docling on text PDFs; Docling is the only tool that recovers the 4 large
scanned 2024-2025 filings (8220660/0674/1212/0750, ~263 transactions).

This package is organised as:
  - cells.py     — cell-level extraction (ticker, date, tx type, owner, ...)
  - columns.py   — column mapping & header detection
  - rows.py      — `parse_pdf_table`, row processing
  - metadata.py  — `normalize_house_metadata`, `consolidate_transactions`
  - pdfplumber_parser.py — pdfplumber backend
  - docling_parser.py    — Docling OCR + markdown parsers
  - pdftotext_parser.py  — pdftotext backend (encrypted PDFs)
  - ocr_parser.py        — pytesseract fallback

Public API is re-exported here so `from analyzer.parsing import X` keeps
working after the split.
"""

# Cell-level helpers
from analyzer.parsing.cells import (
    clean_text,
    _extract_amount_midpoint,
    _extract_date,
    _extract_instrument_type,
    _extract_option_details,
    _extract_owner_code,
    _extract_ticker,
    _extract_transaction_type,
)

# Column mapping
from analyzer.parsing.columns import (
    KNOWN_HEADERS,
    _column_index,
    _column_index_substring,
    _column_indexes,
    _find_amount_in_row,
    _find_header_row,
    _get_cell,
    _merge_two_row_headers,
    _normalize_header,
)

# Row processing
from analyzer.parsing.rows import (
    _build_row_dict,
    _data_start_offset,
    _extract_transactions,
    _process_row,
    _try_merge_continuation,
    parse_pdf_table,
)

# Top-level metadata/consolidation
from analyzer.parsing.metadata import (
    consolidate_transactions,
    normalize_house_metadata,
)

# Backends
from analyzer.parsing.pdfplumber_parser import extract_tables_with_pdfplumber
from analyzer.parsing.docling_parser import (
    _parse_docling_markdown,
    _parse_markdown_tables,
    extract_tables_with_docling,
)
from analyzer.parsing.pdftotext_parser import extract_tables_with_pdftotext
from analyzer.parsing.ocr_parser import (
    _parse_ocr_text_to_rows,
    extract_tables_with_ocr,
)

__all__ = [
    "clean_text",
    "_extract_amount_midpoint",
    "_extract_date",
    "_extract_instrument_type",
    "_extract_option_details",
    "_extract_owner_code",
    "_extract_ticker",
    "_extract_transaction_type",
    "KNOWN_HEADERS",
    "_column_index",
    "_column_index_substring",
    "_column_indexes",
    "_find_amount_in_row",
    "_find_header_row",
    "_get_cell",
    "_merge_two_row_headers",
    "_normalize_header",
    "_build_row_dict",
    "_data_start_offset",
    "_extract_transactions",
    "_process_row",
    "_try_merge_continuation",
    "parse_pdf_table",
    "consolidate_transactions",
    "normalize_house_metadata",
    "extract_tables_with_pdfplumber",
    "_parse_docling_markdown",
    "_parse_markdown_tables",
    "extract_tables_with_docling",
    "extract_tables_with_pdftotext",
    "_parse_ocr_text_to_rows",
    "extract_tables_with_ocr",
]
