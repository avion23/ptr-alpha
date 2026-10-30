from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import subprocess

import pytest

from analyzer.parsing import cascade
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
    _, transactions, engines = cascade.parse_pdf_worker(Path("mixed.pdf"))
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


def test_pdfplumber_layout_enrichment_matches_truncated_table_row(monkeypatch):
    asset_rows = [
        "JP Morgan 3-year auto callable buffer note [CS]",
        "JP Morgan 3-year one-time callable barrier note [CS]",
    ]
    details = (
        "  F S: New\n"
        "  S O: Advisory Account holding Structured Notes\n"
        "  D: Structured Note\n"
    )
    text = "".join(
        f"  {asset} P 12/30/2025 12/30/2025 $5,000,001 - $25,000,000\n{details}"
        for asset in asset_rows
    )
    header = HEADER + ["Notification Date"]
    table = [header] + [
        [asset, "", "P", "12/30/2025", "$5,000,001 - $25,000,000", "12/30/2025"]
        for asset in asset_rows
    ]
    monkeypatch.setattr(
        "pdfplumber.open",
        lambda _: nullcontext(SimpleNamespace(pages=[page(text, [table])])),
    )

    transactions = [
        tx
        for extracted in extract_tables_with_pdfplumber(Path("filing.pdf"))
        for tx in parse_pdf_table(extracted)
    ]

    assert len(transactions) == 2
    assert all("Description: Structured Note" in tx["asset_description"] for tx in transactions)


def test_pdfplumber_layout_enrichment_preserves_repeated_physical_rows(monkeypatch):
    asset = "JP Morgan callable note [CS]"
    details = "  D: Structured Note\n"
    text = "".join(
        f"  {asset} P 12/30/2025 12/30/2025 $5,000,001 - $25,000,000\n{details}"
        for _ in range(2)
    )
    header = HEADER + ["Notification Date"]
    table = [header] + [
        [asset, "", "P", "12/30/2025", "$5,000,001 - $25,000,000", "12/30/2025"]
        for _ in range(2)
    ]
    monkeypatch.setattr(
        "pdfplumber.open",
        lambda _: nullcontext(SimpleNamespace(pages=[page(text, [table])])),
    )

    transactions = [
        tx
        for extracted in extract_tables_with_pdfplumber(Path("filing.pdf"))
        for tx in parse_pdf_table(extracted)
    ]

    assert len(transactions) == 2
    assert [tx["source_row_id"] for tx in transactions] == [
        "pdfplumber:p1:l0",
        "pdfplumber:p1:l2",
    ]


def test_pdfplumber_keeps_conflicting_or_ambiguous_layout_observations(monkeypatch):
    from analyzer.parsing import pdfplumber_parser

    asset = "JP Morgan callable note [CS]"
    owner_text = (
        f"  {asset} P 12/30/2025 12/30/2025 $5,000,001 - $25,000,000\n"
        "  D: Structured Note\n"
    )
    extended = [
        f"{asset} Description: Series A",
        f"{asset} Description: Series B",
    ]
    ambiguous_text = "".join(
        f"  {name} P 12/30/2025 12/30/2025 $5,000,001 - $25,000,000\n"
        for name in extended
    )
    header = HEADER + ["Notification Date"]
    conflicting_owner_table = [
        header,
        [asset, "SP", "P", "12/30/2025", "$5,000,001 - $25,000,000", "12/30/2025"],
    ]
    ambiguous_table = [
        header,
        [asset, "", "P", "12/30/2025", "$5,000,001 - $25,000,000", "12/30/2025"],
    ]
    extension_checks = []
    is_metadata_extension = pdfplumber_parser._has_recognized_metadata_extension

    def record_metadata_extension(shorter, longer):
        matched = is_metadata_extension(shorter, longer)
        extension_checks.append(matched)
        return matched

    monkeypatch.setattr(
        pdfplumber_parser,
        "_has_recognized_metadata_extension",
        record_metadata_extension,
    )

    for table, source_text, expected_rows in (
        (conflicting_owner_table, owner_text, 2),
        (ambiguous_table, ambiguous_text, 3),
    ):
        extension_checks.clear()
        monkeypatch.setattr(
            "pdfplumber.open",
            lambda _, table=table, source_text=source_text: nullcontext(
                SimpleNamespace(pages=[page(source_text, [table])])
            ),
        )
        transactions = [
            tx
            for extracted in extract_tables_with_pdfplumber(Path("filing.pdf"))
            for tx in parse_pdf_table(extracted)
        ]
        assert len(transactions) == expected_rows
        if expected_rows == 3:
            assert extension_checks == [True, True]


@pytest.mark.parametrize(
    ("table_asset", "layout_asset"),
    [
        ("Private Fund I", "Private Fund II"),
        ("Asset A", "Asset AB"),
        ("Asset A", "Asset A Series B"),
    ],
)
def test_arbitrary_asset_prefixes_remain_unresolved_in_cascade(
    monkeypatch, table_asset, layout_asset
):
    header = HEADER + ["Notification Date"]
    amount = "$5,000,001 - $25,000,000"
    table = [
        header,
        [table_asset, "", "P", "12/30/2025", amount, "12/30/2025"],
    ]
    text = f"  {layout_asset} P 12/30/2025 12/30/2025 {amount}\n"
    monkeypatch.setattr(
        "pdfplumber.open",
        lambda _: nullcontext(SimpleNamespace(pages=[page(text, [table])])),
    )
    pdftotext_row = parse_pdf_table(
        [
            header + ["Source Row ID"],
            [layout_asset, "", "P", "12/30/2025", amount, "12/30/2025", "pdftotext:l0"],
        ]
    )[0]
    monkeypatch.setattr(cascade, "_try_camelot_lattice", lambda _: [])
    monkeypatch.setattr(cascade, "_try_camelot_stream", lambda _: [])
    monkeypatch.setattr(cascade, "_try_pdftotext", lambda _: [pdftotext_row])
    monkeypatch.setattr(cascade, "_has_document_text_coverage", lambda _: True)
    monkeypatch.setattr(cascade, "_try_tesseract", lambda _: [pdftotext_row])
    monkeypatch.setenv("PTR_SKIP_DOCLING", "1")

    with pytest.raises(cascade.ParserCascadeError):
        cascade.parse_pdf_worker(Path("conflicting-assets.pdf"))


def test_pdfplumber_does_not_merge_metadata_observations_across_pages(monkeypatch):
    header = HEADER + ["Notification Date"]
    amount = "$5,000,001 - $25,000,000"
    asset = "Private Fund [CS]"
    table = [
        header,
        [asset, "", "P", "12/30/2025", amount, "12/30/2025"],
    ]
    layout_text = (
        f"  {asset} Description: Structured Note P 12/30/2025 "
        f"12/30/2025 {amount}\n"
    )
    monkeypatch.setattr(
        "pdfplumber.open",
        lambda _: nullcontext(SimpleNamespace(pages=[page("", [table]), page(layout_text)])),
    )

    transactions = [
        tx
        for extracted in extract_tables_with_pdfplumber(Path("filing.pdf"))
        for tx in parse_pdf_table(extracted)
    ]

    assert len(transactions) == 2
    assert {tx["source_row_id"].split(":")[1] for tx in transactions} == {"p1", "p2"}


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
    ("Bitcoin [cT]", "other"),
    ("United States Treasury Securities [gS]", "Government Securities"),
    ("Private startup shares [oT]", "other"),
    ("Apple (AAPL) [sT]", "stock"),
    ("Alphabet Inc Note Coupon 4.80000% Mat 2/15/2036 [CS] Description: Corporate Note", "bond"),
    ("JPMorgan Chase BK N A INSTL CTF DEP ... [CS] Description: Corporate Bond", "bond"),
    ("Pacific Co B1 Bond [CS] ... Real estate bond", "bond"),
    ("FEDERAL NATIONAL MORTGAGE ASSOCIATION POOL ... [CS]", "bond"),
    ("Lord Abbett Short Duration High Yield I (LSYIX) bond fund [CS]", "bond"),
    ("Bitcoin [oP]", "option"),
    ("Municipal Bond [MB]", "bond"),
    ("Private LLC [OL] Description: convertible SAFE", "private"),
    ("Private LLC [ST] Description: Simple Agreement for Future Equity", "private"),
    ("D: convertible SAFE Apple (AAPL) [ST]", "stock"),
    ("D: call option Apple (AAPL) [ST]", "stock"),
    ("Apple (AAPL) [ST]", "stock"),
    ("Fund Corporation (FUND) [ST]", "stock"),
])
def test_explicit_nonstock_and_stock_boundaries(asset, expected):
    from analyzer.parsing.cells import extract_instrument_type
    assert extract_instrument_type(asset) == expected


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
