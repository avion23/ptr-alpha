"""Backward-compatible re-exports. Import from submodules directly for new code."""
from analyzer.parser_cascade import (
    _is_valid_pdf,
    _parse_pdf_worker,
    _result_quality,
    _try_camelot_lattice,
    _try_camelot_stream,
    _try_docling,
    _try_pdfplumber,
    _try_pdftotext,
    _try_tesseract,
)
from analyzer.download import (
    HouseTransactionSource,
    _build_member_lookup,
    _filter_existing_pdfs,
    _read_first_text_from_zip,
)
from analyzer.parsing import consolidate_transactions
from analyzer.price_source import (
    YFinancePriceSource,
    _clean_tickers,
    _resolve_tickers,
    _validate_and_log_prices,
)

__all__ = [
    "_is_valid_pdf",
    "_parse_pdf_worker",
    "_result_quality",
    "_try_camelot_lattice",
    "_try_camelot_stream",
    "_try_docling",
    "_try_pdfplumber",
    "_try_pdftotext",
    "_try_tesseract",
    "HouseTransactionSource",
    "_build_member_lookup",
    "_filter_existing_pdfs",
    "_read_first_text_from_zip",
    "consolidate_transactions",
    "YFinancePriceSource",
    "_clean_tickers",
    "_resolve_tickers",
    "_validate_and_log_prices",
]
