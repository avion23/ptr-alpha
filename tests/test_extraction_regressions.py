import pytest

from analyzer.parsing.cells import (
    _extract_amount_midpoint, _extract_option_details, _extract_ticker,
    _extract_transaction_type,
)
from analyzer.parsing.columns import _indexes_from_headers
from analyzer.parsing.rows import parse_pdf_table


HEADER = ["Asset", "Type", "Date", "Amount"]


def test_finding_1_independent_asset_is_not_consumed():
    rows = parse_pdf_table([HEADER, ["Apple (AAPL)", "", "", ""],
                           ["Microsoft (MSFT)", "P", "01/02/2026", "$2,000"]])
    assert [row["ticker"] for row in rows] == ["MSFT"]


def test_finding_2_notification_is_not_transaction_date():
    indexes = _indexes_from_headers(["Member", "Notification Date", "Transaction Date"])
    assert indexes["date"] == 2
    assert indexes["notification_date"] == 1
    assert _indexes_from_headers(["Notification Date"])["date"] is None


def test_finding_3_generic_description_is_not_issuer():
    assert _extract_ticker("Target date fund 2045") is None
    assert _extract_ticker("Target Corporation") == "TGT"
    assert _extract_ticker("Apple Inc. Common Stock") == "AAPL"


def test_finding_4_contract_money_is_not_transaction_amount():
    rows = parse_pdf_table([["Asset", "Type", "Date", "Extra Col"],
                           ["Apple (AAPL) call option strike $100", "P", "01/02/2026", "$1,000 - $2,000"]])
    assert rows[0]["amount_midpoint"] == 1500
    assert parse_pdf_table([["Apple (AAPL) call option strike $100", "P", "01/02/2026"]])[0]["amount_midpoint"] is None


@pytest.mark.parametrize("year,expected", [("2026", "2026"), ("26", "2026"), ("99", "1999"), ("20260", None)])
def test_finding_5_complete_expiry_year(year, expected):
    details = _extract_option_details(f"call option exp 12/19/{year}")
    assert details.get("expiry_date") == (f"12/19/{expected}" if expected else None)


def test_finding_6_grouped_strike():
    assert _extract_option_details("call option strike $1,200.50")["strike_price"] == 1200.5


def test_finding_6_amount_cents():
    assert _extract_amount_midpoint("$1,000.50 - $2,000.50")[1] == 1500.5


@pytest.mark.parametrize("token", ["$1,20", "$1.200,50", "$100.123"])
def test_finding_6_reject_unsupported_currency(token):
    assert _extract_amount_midpoint(token)[1] is None
    assert _extract_option_details(f"call option strike {token}").get("strike_price") is None


@pytest.mark.parametrize("label", ["Sale / Purchase", "P / S", "Exchange / Buy"])
def test_finding_7_conflicting_directions(label):
    assert _extract_transaction_type(label) is None


def test_finding_8_complementary_fields_and_conflicts():
    rows = parse_pdf_table([HEADER, ["Example (XYZ)", "P", "", "$1,000"],
                           ["", "", "01/02/2026", ""]])
    assert len(rows) == 1
    assert rows[0]["transaction_date"] == "01/02/2026"
    assert rows[0]["amount_midpoint"] == 1000
    assert parse_pdf_table([HEADER, ["Example (XYZ)", "P", "", ""],
                            ["", "S", "01/02/2026", ""]]) == []


def test_finding_9_trailing_ticker_is_collected():
    rows = parse_pdf_table([HEADER, ["Example Corporation", "P", "01/02/2026", "$1,000"],
                           ["(XYZ)", "", "", ""]])
    assert len(rows) == 1
    assert rows[0]["ticker"] == "XYZ"
