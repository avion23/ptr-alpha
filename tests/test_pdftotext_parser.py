from pathlib import Path

from analyzer.parsing.pdftotext_parser import extract_tables_with_pdftotext
from analyzer.parsing.rows import parse_pdf_table


def test_pdftotext_preserves_owner_code(monkeypatch):
    text = "  SP Apple Inc. (AAPL) P 01/02/2024 01/03/2024 $1,001 - $15,000\n"
    monkeypatch.setattr(
        "analyzer.parsing.pdftotext_parser._run_pdftotext", lambda _path: text
    )

    tables = extract_tables_with_pdftotext(Path("disclosure.pdf"))
    transactions = parse_pdf_table(tables[0])

    assert transactions[0]["owner_code"] == "SP"
    assert transactions[0]["amount_midpoint"] == 8000.5
    assert transactions[0]["source_row_id"] == "pdftotext:l0"


def test_pdftotext_no_owner_keeps_column_alignment(monkeypatch):
    text = "Apple Inc. (AAPL) P 01/02/2024 01/03/2024 $1,001 - $15,000\n"
    monkeypatch.setattr(
        "analyzer.parsing.pdftotext_parser._run_pdftotext", lambda _path: text
    )

    tables = extract_tables_with_pdftotext(Path("disclosure.pdf"))
    transactions = parse_pdf_table(tables[0])

    assert transactions[0]["owner_code"] is None
    assert transactions[0]["ticker"] == "AAPL"


def test_indented_uppercase_asset_is_not_misclassified_as_owner(monkeypatch):
    text = "    IBM (IBM) P 01/02/2024 01/03/2024 $1,001 - $15,000\n"
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
    assert (
        transaction["asset_description"]
        == "Alphabet Inc. - Class A Common Stock (GOOGL) [ST]"
    )


def test_wrapped_amount_after_non_asset_detail_lines(monkeypatch):
    text = (
        "  Alibaba Group Holding Limited S 02/11/2022 03/04/2022 $50,001 - b\n"
        "                                                              c\n"
        "  American Depositary Shares each                              $100,000\n"
        "  representing eight Ordinary shares\n"
        "  (BABA) [ST]\n"
        "  FILING STATUS: New\n"
    )
    monkeypatch.setattr(
        "analyzer.parsing.pdftotext_parser._run_pdftotext", lambda _path: text
    )

    transaction = parse_pdf_table(
        extract_tables_with_pdftotext(Path("disclosure.pdf"))[0]
    )[0]

    assert transaction["amount_raw"] == "$50,001 - $100,000"
    assert transaction["amount_midpoint"] == 75000.5


def test_wrapped_amount_before_instrument_marker(monkeypatch):
    text = (
        "  IBM Common Stock P 06/03/2025 06/05/2025 $15,001 -\n"
        "  (IBM)                                      $50,000 [ST]\n"
        "  S O: Trust Account\n"
    )
    monkeypatch.setattr(
        "analyzer.parsing.pdftotext_parser._run_pdftotext", lambda _path: text
    )

    transaction = parse_pdf_table(
        extract_tables_with_pdftotext(Path("disclosure.pdf"))[0]
    )[0]

    assert transaction["amount_raw"] == "$15,001 - $50,000"
    assert (
        transaction["asset_description"]
        == "IBM Common Stock (IBM) [ST] [Account: Trust Account]"
    )


def test_spouse_dc_over_amount_can_wrap_to_a_later_line(monkeypatch):
    text = (
        "  SP U.S. Treasury Note due 05/31/2024 P 05/30/2023 06/02/2023 "
        "Spouse/DC Over\n"
        "     [GS]                                      $1,000,000\n"
        "     F S: New\n"
    )
    monkeypatch.setattr(
        "analyzer.parsing.pdftotext_parser._run_pdftotext", lambda _path: text
    )

    transaction = parse_pdf_table(
        extract_tables_with_pdftotext(Path("disclosure.pdf"))[0]
    )[0]

    assert transaction["owner_code"] == "SP"
    assert transaction["ticker"] is None
    assert transaction["asset_description"] == "U.S. Treasury Note due 05/31/2024 [GS]"
    assert transaction["amount_raw"] == "Spouse/DC Over $1,000,000"
    assert transaction["amount_midpoint"] == 1_000_000


def test_asset_name_starting_with_id_is_not_treated_as_header(monkeypatch):
    text = (
        "  IDACORP, Inc. Common Stock (IDA) P 03/03/2026 04/06/2026 $1,001 - $15,000\n"
    )
    monkeypatch.setattr(
        "analyzer.parsing.pdftotext_parser._run_pdftotext", lambda _path: text
    )

    transaction = parse_pdf_table(
        extract_tables_with_pdftotext(Path("disclosure.pdf"))[0]
    )[0]

    assert transaction["ticker"] == "IDA"
    assert transaction["transaction_date"] == "03/03/2026"


def test_pdfplumber_nul_source_accounts_keep_distinct_rows_through_persistence(
    monkeypatch, tmp_path
):
    from types import SimpleNamespace

    import pandas as pd

    from analyzer.database import Database
    from analyzer.parsing import consolidate_transactions
    from analyzer.parsing.pdfplumber_parser import extract_tables_with_pdfplumber

    accounts = [
        "Morgan Stanley - E*TRADE #2",
        "Morgan Stanley - E*TRADE IRA",
        "Morgan Stanley - E*TRADE - Fields Law Firm 2, LLC",
    ]
    text = "".join(
        "  NVIDIA Corporation - Common Stock P 06/26/2026 06/26/2026 "
        "$1,001 - $15,000\n"
        "  (NVDA) [ST]\n"
        f"  S\x00\x00 O\x00: {account}\n\n"
        for account in accounts
    )

    class FakePage:
        def extract_text(self, *, layout):
            assert layout is True
            return text

        def extract_tables(self):
            return []

    class FakePdf:
        pages = [FakePage()]

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    monkeypatch.setitem(
        __import__("sys").modules,
        "pdfplumber",
        SimpleNamespace(open=lambda _path: FakePdf()),
    )
    path = Path("20034894.pdf")
    transactions = parse_pdf_table(extract_tables_with_pdfplumber(path)[0])

    assert len(transactions) == 3
    assert [tx["ticker"] for tx in transactions] == ["NVDA"] * 3
    assert len({tx["asset_description"] for tx in transactions}) == 3
    assert [tx["source_row_id"] for tx in transactions] == [
        "pdfplumber:p1:l0",
        "pdfplumber:p1:l4",
        "pdfplumber:p1:l8",
    ]

    df = consolidate_transactions(
        {path: transactions},
        {
            "20034894": {
                "First": "Cleo",
                "Last": "Fields",
                "FilingDate": pd.Timestamp("2026-07-16"),
            }
        },
    )
    assert df["source_row_id"].tolist() == [
        "pdfplumber:p1:l0",
        "pdfplumber:p1:l4",
        "pdfplumber:p1:l8",
    ]

    db = Database(tmp_path / "account-identity.duckdb")
    try:
        inserted = db.upsert_transactions(df, source="house_pdf")
        stored = db.get_transactions_for_doc("20034894")
    finally:
        db.close()

    assert inserted == 3
    assert len(stored) == 3
    assert set(stored["asset_description"]) == {
        f"NVIDIA Corporation - Common Stock (NVDA) [ST] [Account: {account}]"
        for account in accounts
    }


def test_multiline_asset_description_reaches_ticker_line(monkeypatch):
    text = (
        "  JT Sea Limited American Depositary S 12/05/2025 01/07/2026 "
        "$1,001 - $15,000\n"
        "     Shares, each representing one Class A\n"
        "     Ordinary Share (SE) [ST]\n"
        "     S O: Morgan Stanley - Select UMA Account # 1\n"
    )
    monkeypatch.setattr(
        "analyzer.parsing.pdftotext_parser._run_pdftotext", lambda _path: text
    )

    transaction = parse_pdf_table(
        extract_tables_with_pdftotext(Path("20033756.pdf"))[0]
    )[0]

    assert transaction["ticker"] == "SE"
    assert transaction["asset_description"] == (
        "Sea Limited American Depositary Shares, each representing one Class A "
        "Ordinary Share (SE) [ST] "
        "[Account: Morgan Stanley - Select UMA Account # 1]"
    )
