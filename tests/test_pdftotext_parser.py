from pathlib import Path

from analyzer.parsing.pdftotext_parser import extract_tables_with_pdftotext
from analyzer.parsing.rows import parse_pdf_table


def test_pdftotext_preserves_owner_code(monkeypatch):
    text = (
        "  SP Apple Inc. (AAPL) P 01/02/2024 01/03/2024 "
        "$1,001 - $15,000\n"
    )
    monkeypatch.setattr(
        "analyzer.parsing.pdftotext_parser._run_pdftotext", lambda _path: text
    )

    tables = extract_tables_with_pdftotext(Path("disclosure.pdf"))
    transactions = parse_pdf_table(tables[0])

    assert transactions[0]["owner_code"] == "SP"
    assert transactions[0]["amount_midpoint"] == 8000.5


def test_pdftotext_no_owner_keeps_column_alignment(monkeypatch):
    text = (
        "Apple Inc. (AAPL) P 01/02/2024 01/03/2024 "
        "$1,001 - $15,000\n"
    )
    monkeypatch.setattr(
        "analyzer.parsing.pdftotext_parser._run_pdftotext", lambda _path: text
    )

    tables = extract_tables_with_pdftotext(Path("disclosure.pdf"))
    transactions = parse_pdf_table(tables[0])

    assert transactions[0]["owner_code"] is None
    assert transactions[0]["ticker"] == "AAPL"


def test_indented_uppercase_asset_is_not_misclassified_as_owner(monkeypatch):
    text = (
        "    IBM (IBM) P 01/02/2024 01/03/2024 "
        "$1,001 - $15,000\n"
    )
    monkeypatch.setattr(
        "analyzer.parsing.pdftotext_parser._run_pdftotext", lambda _path: text
    )

    tables = extract_tables_with_pdftotext(Path("disclosure.pdf"))
    transactions = parse_pdf_table(tables[0])

    assert transactions[0]["ticker"] == "IBM"
    assert transactions[0]["owner_code"] is None
