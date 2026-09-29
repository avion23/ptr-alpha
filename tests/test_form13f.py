from datetime import date

import pytest

from analyzer.form13f import ThirteenFError, ThirteenFSource, _parse_information_table


@pytest.fixture
def quarter_xmls():
    previous = """<?xml version="1.0"?>
    <informationTable xmlns="http://www.sec.gov/edgar/document/thirteenf/informationtable">
      <infoTable><nameOfIssuer>Vistra Corp</nameOfIssuer><cusip>111111111</cusip><sshPrnamt>100</sshPrnamt><sshPrnamtType>SH</sshPrnamtType></infoTable>
      <infoTable><nameOfIssuer>Alpha Bio Inc</nameOfIssuer><cusip>222222222</cusip><sshPrnamt>200</sshPrnamt><sshPrnamtType>SH</sshPrnamtType></infoTable>
    </informationTable>"""
    current = """<?xml version="1.0"?>
    <informationTable xmlns="http://www.sec.gov/edgar/document/thirteenf/informationtable">
      <infoTable><nameOfIssuer>VISTRA CORP</nameOfIssuer><cusip>111111111</cusip><sshPrnamt>150</sshPrnamt><sshPrnamtType>SH</sshPrnamtType></infoTable>
      <infoTable><nameOfIssuer>Alpha Bio Inc</nameOfIssuer><cusip>222222222</cusip><sshPrnamt>125</sshPrnamt><sshPrnamtType>SH</sshPrnamtType></infoTable>
      <infoTable><nameOfIssuer>Nova Labs</nameOfIssuer><cusip>333333333</cusip><sshPrnamt>300</sshPrnamt><sshPrnamtType>SH</sshPrnamtType></infoTable>
      <infoTable><nameOfIssuer>Orion Energy Holdings</nameOfIssuer><cusip>444444444</cusip><sshPrnamt>400</sshPrnamt><sshPrnamtType>SH</sshPrnamtType></infoTable>
    </informationTable>"""
    return previous, current


def test_emits_only_watchlist_increases_and_new_positions(tmp_path, quarter_xmls):
    source = ThirteenFSource(
        data_dir=tmp_path / "data",
        db=object(),
        watchlist={
            "VST": {"VISTRA"},
            "NOVA": {"NOVA LABS"},
            "ORION": {"ORION"},
            "ORI": {"ORION ENERGY"},
        },
        managers={"test": {"name": "Test Manager", "cik": "1234567"}},
        ingestion_generation="test-generation",
        price_lookup=lambda ticker, quarter_end: 10 if ticker == "VST" else None,
    )
    previous_xml, current_xml = quarter_xmls
    previous = _parse_information_table(previous_xml)
    current = _parse_information_table(current_xml)
    filing = {
        "report_date": "2026-06-30",
        "filing_date": "2026-08-14",
        "accession": "0001234567-26-000002",
    }
    rows = source._position_changes(
        {"name": "Test Manager"},
        filing,
        current,
        previous,
        "https://www.sec.gov/Archives/edgar/data/1234567/000123456726000002/infotable.xml",
    )

    assert [(row["ticker"], row["amount_raw"]) for row in rows] == [
        ("VST", "50 shares"),
        ("NOVA", "300 shares"),
    ]
    assert all(
        row["raw_transaction_subtype"] == "INFERRED_POSITION_INCREASE" for row in rows
    )
    assert rows[0]["transaction_date"] == date(2026, 6, 30)
    assert rows[0]["disclosure_date"] == date(2026, 8, 14)
    assert rows[0]["amount_midpoint"] == 500
    assert rows[1]["amount_midpoint"] is None
    assert rows[0]["source_record_id"] == "0001234567-26-000002"
    assert rows[0]["source_row_id"] == "111111111"
    assert rows[0]["ingestion_generation"] == "test-generation"
    source.close()


def test_empty_information_table_fails_closed():
    with pytest.raises(ThirteenFError, match="empty or has no usable share rows"):
        _parse_information_table(
            '<informationTable xmlns="http://www.sec.gov/edgar/document/thirteenf/informationtable" />'
        )
