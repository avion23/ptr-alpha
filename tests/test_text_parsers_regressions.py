from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import subprocess

import pytest

from analyzer import parser_cascade as cascade
from analyzer.parsing.pdfplumber_parser import extract_tables_with_pdfplumber
from analyzer.parsing.pdftotext_parser import _parse_pdftotext_lines
from analyzer.parsing.rows import parse_pdf_table


HEADER = ["Asset Name", "Owner", "Transaction Type", "Transaction Date", "Amount"]


def row(asset):
    return [asset, "", "P", "01/02/2024", "$1,001 - $15,000"]


def page(text, tables=()):
    return SimpleNamespace(extract_text=lambda **kwargs: text,
                           extract_tables=lambda: tables)


def test_pdfplumber_retains_complementary_table_rows(monkeypatch):
    pages = [page("Apple (AAPL) P 01/02/2024 01/03/2024 $1,001 - $15,000",
                  [[HEADER, row("IBM (IBM)")]])]
    monkeypatch.setattr("pdfplumber.open", lambda _: nullcontext(SimpleNamespace(pages=pages)))
    tables = extract_tables_with_pdfplumber(Path("filing.pdf"))
    assert {tx["ticker"] for table in tables for tx in parse_pdf_table(table)} == {"AAPL", "IBM"}


def test_pdfplumber_failure_keeps_backend_cause(tmp_path):
    path = tmp_path / "invalid.pdf"
    path.write_bytes(b"not a PDF")
    with pytest.raises(cascade.ParserBackendError) as error:
        cascade._try_pdfplumber(path)
    assert error.value.engine == "pdfplumber"
    assert error.value.cause is not None


def test_pdftotext_nonzero_keeps_backend_cause(tmp_path):
    path = tmp_path / "invalid.pdf"
    path.write_bytes(b"not a PDF")
    with pytest.raises(cascade.ParserBackendError) as error:
        cascade._try_pdftotext(path)
    assert error.value.engine == "pdftotext"
    assert isinstance(error.value.cause, subprocess.CalledProcessError)
    assert error.value.cause.returncode != 0


@pytest.mark.parametrize("ticker", ["F", "GE"])
def test_short_wrapped_ticker(ticker):
    rows = _parse_pdftotext_lines([
        "Unlisted asset P 01/02/2024 01/03/2024 $1,001 - $15,000",
        f"  ({ticker})", "  [ST]",
    ])
    assert rows[0][0] == f"Unlisted asset ({ticker}) [ST]"


def test_text_agreement_on_partial_document_requires_ocr(monkeypatch):
    text_rows = parse_pdf_table([HEADER, row("Apple (AAPL)")])
    all_rows = text_rows + parse_pdf_table([HEADER, row("IBM (IBM)")])
    pages = [page("Apple transaction"), page("")]
    monkeypatch.setattr("pdfplumber.open", lambda _: nullcontext(SimpleNamespace(pages=pages)))
    for name in ("_try_pdfplumber", "_try_pdftotext"):
        monkeypatch.setattr(cascade, name, lambda _: text_rows)
    for name in ("_try_camelot_lattice", "_try_camelot_stream"):
        monkeypatch.setattr(cascade, name, lambda _: [])
    monkeypatch.setenv("PTR_SKIP_DOCLING", "1")
    monkeypatch.setattr(cascade, "_try_tesseract", lambda _: all_rows)
    _, transactions, engines = cascade._parse_pdf_worker(Path("mixed.pdf"))
    assert "ocr" in engines
    assert len(transactions) == 2


def test_flattened_owner_row_and_layout_are_one_lot(monkeypatch):
    text = "SP Apple (AAPL) P 01/02/2024 01/03/2024 $1,001 - $15,000"
    pages = [page("  " + text, [[HEADER, [text, "", "", "", ""]]])]
    monkeypatch.setattr("pdfplumber.open", lambda _: nullcontext(SimpleNamespace(pages=pages)))
    tables = extract_tables_with_pdfplumber(Path("filing.pdf"))
    transactions = [tx for table in tables for tx in parse_pdf_table(table)]
    assert len(transactions) == 1
    assert transactions[0]["owner_code"] == "SP"
