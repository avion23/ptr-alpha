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


def test_pdfplumber_layout_notification_and_safe_context(monkeypatch):
    text = (
        "  JT Private Foods LLC [OL] P 09/29/2026 09/30/2026 $50,001 - $100,000\n"
        "  F S: New\n  L: Bethesda, MD, US\n"
        "  D: Mandatorily convertible SAFE (Simple Agreement\n  for Future Equity)\n"
    )
    pages = [page(text)]
    monkeypatch.setattr("pdfplumber.open", lambda _: nullcontext(SimpleNamespace(pages=pages)))
    tx = parse_pdf_table(extract_tables_with_pdfplumber(Path("test.pdf"))[0])[0]
    assert tx["notification_date"] == "09/30/2026"
    assert tx["instrument_type"] == "private"
    assert "Simple Agreement for Future Equity" in tx["asset_description"]


def test_table_notification_and_linked_description_not_next_issuer():
    header = HEADER + ["Notification Date"]
    transactions = parse_pdf_table([
        header,
        ["Private Foods LLC [OL]", "JT", "P", "09/29/2026", "$50,001 - $100,000", "09/30/2026"],
        ["F S: New", "", "", "", "", ""],
        ["Location: Bethesda, MD, US", "", "", "", "", ""],
        ["Description: Mandatorily convertible SAFE (Simple Agreement\nfor Future Equity)", "", "", "", "", ""],
        ["Apple (AAPL) [ST]", "", "P", "10/01/2026", "$1,001 - $15,000", "10/02/2026"],
    ])
    assert len(transactions) == 2
    assert transactions[0]["notification_date"] == "09/30/2026"
    assert transactions[0]["instrument_type"] == "private"
    assert "Simple Agreement for Future Equity" in transactions[0]["asset_description"]
    assert transactions[1]["asset_description"] == "Apple (AAPL) [ST]"


@pytest.mark.parametrize("asset,expected", [
    ("Treasury Bill (6-Week, Matures 12/3/2026) [GS]", "treasury"),
    ("Municipal Bond [MB]", "bond"),
    ("Private LLC [OL] Description: convertible SAFE", "private"),
    ("Private LLC [ST] Description: Simple Agreement for Future Equity", "private"),
    ("D: convertible SAFE Apple (AAPL) [ST]", "stock"),
    ("D: call option Apple (AAPL) [ST]", "stock"),
    ("Apple (AAPL) [ST]", "stock"),
    ("Fund Corporation (FUND) [ST]", "stock"),
])
def test_explicit_nonstock_and_stock_boundaries(asset, expected):
    from analyzer.parsing.cells import _extract_instrument_type
    assert _extract_instrument_type(asset) == expected


def test_flattened_prior_description_cannot_hijack_next_stock():
    transactions = parse_pdf_table([
        HEADER,
        row("Private LLC [OL]"),
        ["D: convertible SAFE Apple (AAPL) [ST]", "", "P", "01/03/2024", "$1,001 - $15,000"],
    ])
    assert len(transactions) == 2
    assert transactions[0]["asset_description"] == "Private LLC [OL]"
    assert transactions[1]["ticker"] == "AAPL"
    assert transactions[1]["instrument_type"] == "stock"


def test_linked_table_description_retains_transaction_source_identity():
    transactions = parse_pdf_table([
        HEADER + ["Source Row ID"],
        row("Private LLC [OL]") + ["p1:r1"],
        ["D: Mandatorily convertible SAFE", "", "", "", "", "p1:r2"],
    ])
    assert len(transactions) == 1
    assert transactions[0]["instrument_type"] == "private"
    assert transactions[0]["source_row_id"] == "p1:r1"
