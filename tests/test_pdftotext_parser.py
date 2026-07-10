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


def test_wrapped_amount_and_source_accounts_are_preserved(monkeypatch):
    text = (
        "                      AT&T Inc. (T) [ST] P 05/15/2026 05/28/2026 $50,001 -\n"
        "                                                                 $100,000\n"
        "                      F      S     : New\n"
        "                      S           O : Trust > Retirement Account\n"
        "\n"
        "                      AT&T Inc. (T) [ST] P 05/15/2026 05/28/2026 $1,001 - $15,000\n"
        "                      F      S     : New\n"
        "                      S           O : Trust > Joint Brokerage\n"
    )
    monkeypatch.setattr(
        "analyzer.parsing.pdftotext_parser._run_pdftotext", lambda _path: text
    )

    tables = extract_tables_with_pdftotext(Path("disclosure.pdf"))
    transactions = parse_pdf_table(tables[0])

    assert len(transactions) == 2
    assert transactions[0]["amount_raw"] == "$50,001 - $100,000"
    assert transactions[0]["amount_midpoint"] == 75000.5
    assert transactions[0]["asset_description"].endswith(
        "[Account: Trust > Retirement Account]"
    )
    assert transactions[1]["asset_description"].endswith(
        "[Account: Trust > Joint Brokerage]"
    )


def test_asset_text_in_owner_column_is_not_an_owner_code():
    table = [
        ["Asset Name", "Owner", "Transaction Type", "Transaction Date", "Amount"],
        ["AT&T Inc. (T)", "AT&T INC", "P", "05/15/2026", "$1,001 - $15,000"],
    ]

    transaction = parse_pdf_table(table)[0]

    assert transaction["owner_code"] is None


def test_wrapped_asset_line_can_also_contain_amount_upper_bound(monkeypatch):
    text = (
        "                      Alphabet Inc. - Class A Common S 05/15/2026 05/28/2026 $15,001 -\n"
        "                      Stock (GOOGL) [ST]                                      $50,000\n"
    )
    monkeypatch.setattr(
        "analyzer.parsing.pdftotext_parser._run_pdftotext", lambda _path: text
    )

    table = extract_tables_with_pdftotext(Path("disclosure.pdf"))[0]
    transaction = parse_pdf_table(table)[0]

    assert transaction["ticker"] == "GOOGL"
    assert transaction["amount_raw"] == "$15,001 - $50,000"
    assert transaction["amount_midpoint"] == 32500.5
    assert transaction["asset_description"] == "Alphabet Inc. - Class A Common Stock (GOOGL) [ST]"
