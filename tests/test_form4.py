from datetime import date

import pytest

from analyzer.form4 import Form4Error, parse_form4_xml
from analyzer.transaction_repository import _BASE_WRITE_COLUMNS


_EXPECTED_COLUMNS = _BASE_WRITE_COLUMNS + (
    "source_record_id",
    "source_row_id",
    "source_report_path",
    "raw_owner",
    "official_filing_date",
    "ingestion_generation",
    "raw_transaction_subtype",
    "raw_asset_description",
)


@pytest.fixture
def burke_form4_xml():
    return b"""<?xml version="1.0" encoding="UTF-8"?>
<ownershipDocument>
  <documentType>4</documentType>
  <periodOfReport>2026-09-01</periodOfReport>
  <issuer>
    <issuerCik>0001692819</issuerCik>
    <issuerName>Vistra Corp.</issuerName>
    <issuerTradingSymbol>VST</issuerTradingSymbol>
  </issuer>
  <reportingOwner>
    <reportingOwnerId><rptOwnerName>James Burke</rptOwnerName></reportingOwnerId>
    <reportingOwnerRelationship><isOfficer>1</isOfficer><officerTitle>CEO</officerTitle></reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <securityTitle><value>Common Stock</value></securityTitle>
      <transactionDate><value>2026-08-31</value></transactionDate>
      <transactionCoding><transactionCode>P</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>2200</value></transactionShares>
        <transactionPricePerShare><value>135.99</value></transactionPricePerShare>
      </transactionAmounts>
    </nonDerivativeTransaction>
    <nonDerivativeTransaction>
      <securityTitle><value>Common Stock</value></securityTitle>
      <transactionDate><value>2026-08-31</value></transactionDate>
      <transactionCoding><transactionCode>S</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>12</value></transactionShares>
        <transactionPricePerShare><value>136.01</value></transactionPricePerShare>
      </transactionAmounts>
    </nonDerivativeTransaction>
    <nonDerivativeTransaction>
      <securityTitle><value>Common Stock</value></securityTitle>
      <transactionDate><value>2026-09-01</value></transactionDate>
      <transactionCoding><transactionCode>P</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>4465</value></transactionShares>
        <transactionPricePerShare><value>135.25</value></transactionPricePerShare>
      </transactionAmounts>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
  <derivativeTable>
    <derivativeTransaction>
      <securityTitle><value>Stock Option</value></securityTitle>
      <transactionDate><value>2026-09-01</value></transactionDate>
      <transactionCoding><transactionCode>P</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>100</value></transactionShares>
        <transactionPricePerShare><value>0</value></transactionPricePerShare>
      </transactionAmounts>
    </derivativeTransaction>
  </derivativeTable>
</ownershipDocument>"""


def parse(content):
    return parse_form4_xml(
        content,
        accession="0001268406-26-000009",
        source_report_path=(
            "https://www.sec.gov/Archives/edgar/data/1692819/"
            "000126840626000009/form4.xml"
        ),
        official_filing_date=date(2026, 9, 2),
        ingestion_generation="test-generation",
    )


def test_only_non_derivative_open_market_purchases_are_emitted(burke_form4_xml):
    rows = parse(burke_form4_xml)

    assert tuple(rows.columns) == _EXPECTED_COLUMNS
    assert len(rows) == 2
    assert rows["member"].tolist() == ["James Burke", "James Burke"]
    assert rows["ticker"].tolist() == ["VST", "VST"]
    assert rows["transaction_date"].tolist() == [date(2026, 8, 31), date(2026, 9, 1)]
    assert rows["amount_midpoint"].tolist() == pytest.approx(
        [2200 * 135.99, 4465 * 135.25]
    )
    assert rows["transaction_type"].tolist() == ["Purchase", "Purchase"]
    assert rows["instrument_type"].tolist() == ["Common Stock", "Common Stock"]
    assert rows["source"].tolist() == ["form4", "form4"]
    assert rows["source_record_id"].tolist() == [
        "0001268406-26-000009",
        "0001268406-26-000009",
    ]
    assert rows["source_row_id"].tolist() == [
        "nonDerivativeTransaction:000001",
        "nonDerivativeTransaction:000003",
    ]
    assert rows["source_report_path"].str.contains("000126840626000009").all()
    assert rows["raw_owner"].tolist() == ["James Burke", "James Burke"]
    assert rows["official_filing_date"].tolist() == [
        date(2026, 9, 2),
        date(2026, 9, 2),
    ]
    assert rows["ingestion_generation"].tolist() == [
        "test-generation",
        "test-generation",
    ]
    assert rows["raw_transaction_subtype"].tolist() == ["P", "P"]
    assert rows["raw_asset_description"].str.contains("is_10b5_1=false").all()


def test_purchase_without_price_fails_closed(burke_form4_xml):
    xml = burke_form4_xml.replace(
        b"<transactionPricePerShare><value>135.99</value></transactionPricePerShare>",
        b"<transactionPricePerShare><value></value></transactionPricePerShare>",
        1,
    )

    with pytest.raises(Form4Error, match="missing price"):
        parse(xml)


def test_10b5_1_checkbox_is_detected_both_ways(burke_form4_xml):
    assert parse(burke_form4_xml)["asset_description"].str.contains(
        "is_10b5_1=false"
    ).all()
    unchecked = burke_form4_xml.replace(
        b"<periodOfReport>2026-09-01</periodOfReport>",
        b"<periodOfReport>2026-09-01</periodOfReport><aff10b5One>0</aff10b5One>",
    )
    assert parse(unchecked)["asset_description"].str.contains(
        "is_10b5_1=false"
    ).all()
    planned = burke_form4_xml.replace(
        b"<periodOfReport>2026-09-01</periodOfReport>",
        b"<periodOfReport>2026-09-01</periodOfReport><aff10b5One>1</aff10b5One>",
    )
    assert parse(planned)["asset_description"].str.contains("is_10b5_1=true").all()


def test_10b5_1_affirmative_footnote_is_detected(burke_form4_xml):
    with_footnote = burke_form4_xml.replace(
        b"</ownershipDocument>",
        b"<footnotes><footnote id='F1'>These purchases were made pursuant to a "
        b"Rule 10b5-1 trading plan.</footnote></footnotes></ownershipDocument>",
    )

    assert parse(with_footnote)["asset_description"].str.contains(
        "is_10b5_1=true"
    ).all()
