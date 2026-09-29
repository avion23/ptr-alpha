from datetime import date

import pandas as pd
import pytest

from analyzer.form13f import (
    ThirteenFError,
    ThirteenFSource,
    _parse_information_table,
    candidates_from_increases,
)


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


def test_fetch_all_increases_keeps_unmatched_and_new_positions(
    tmp_path, quarter_xmls, monkeypatch
):
    source = ThirteenFSource(
        data_dir=tmp_path / "data",
        db=object(),
        watchlist={"VST": {"VISTRA"}},
        ticker_map={"444444444": "ORION"},
        managers={"test": {"name": "Test Manager", "cik": "1234567"}},
    )
    previous_xml, current_xml = quarter_xmls
    previous = _parse_information_table(previous_xml)
    previous["333333333"] = {"issuer": "Nova Labs", "shares": 250}
    current = _parse_information_table(current_xml)
    filings = {
        "filings": {
            "recent": {
                "form": ["13F-HR", "13F-HR"],
                "reportDate": ["2026-03-31", "2026-06-30"],
                "filingDate": ["2026-05-15", "2026-08-14"],
                "accessionNumber": ["0001234567-26-000001", "0001234567-26-000002"],
            }
        }
    }
    monkeypatch.setattr(source, "_load_submissions", lambda cik: filings)
    monkeypatch.setattr(
        source,
        "_load_filing",
        lambda cik, filing: (
            previous if filing["report_date"] == "2026-03-31" else current,
            "unused",
        ),
    )

    increases = source.fetch_all_increases()

    assert list(zip(increases["ticker"], increases["added_shares"])) == [
        ("VST", 50),
        (None, 50),
        ("ORION", 400),
    ]
    unknown = increases.loc[increases["ticker"].isna()].iloc[0]
    assert unknown["cusip"] == "333333333"
    assert unknown["raw_asset_description"] == "Nova Labs"
    assert increases["event_date"].eq(date(2026, 6, 30)).all()
    assert increases["disclosure_date"].eq(date(2026, 8, 14)).all()
    source.close()


def test_candidates_from_increases_has_shared_schema_and_corroborates():
    increases = pd.DataFrame(
        {
            "ticker": ["VST", None],
            "manager_key": ["thiel_macro", "appaloosa"],
            "event_date": [date(2026, 6, 30), date(2026, 6, 30)],
            "disclosure_date": [date(2026, 8, 14), date(2026, 8, 14)],
        }
    )

    candidates = candidates_from_increases(increases)

    assert list(candidates.columns) == [
        "ticker",
        "actor_id",
        "kind",
        "source",
        "entry_ref",
        "event_date",
        "disclosure_date",
        "corroboration",
        "as_of",
    ]
    assert candidates["actor_id"].tolist() == [
        "manager:thiel_macro",
        "manager:appaloosa",
    ]
    assert candidates["kind"].eq("manager").all()
    assert candidates["source"].eq("13f").all()
    assert candidates["entry_ref"].tolist() == [None, None]
    assert candidates["corroboration"].tolist() == [True, True]
    assert candidates["ticker"].isna().tolist() == [False, True]
    assert candidates.loc[1, "ticker"] is None
    assert candidates["as_of"].dt.tz is not None


def test_fetch_all_increases_isolates_manager_failures(tmp_path, monkeypatch, caplog):
    source = ThirteenFSource(
        data_dir=tmp_path / "data",
        db=object(),
        watchlist={"VST": {"VISTRA"}},
        managers={
            "bad": {"name": "Bad Manager", "cik": "1234567"},
            "good": {"name": "Good Manager", "cik": "7654321"},
        },
    )

    def manager_increases(key, manager):
        if key == "bad":
            raise ThirteenFError("broken filing")
        return [
            {
                "manager_key": key,
                "manager": manager["name"],
                "ticker": "VST",
                "event_date": date(2026, 6, 30),
                "disclosure_date": date(2026, 8, 14),
                "cusip": "111111111",
                "added_shares": 50,
                "raw_asset_description": "Vistra Corp",
                "source": "13f",
            }
        ]

    monkeypatch.setattr(source, "_manager_increases", manager_increases)
    increases = source.fetch_all_increases()
    assert increases["manager_key"].tolist() == ["good"]
    assert "Skipping 13F manager bad" in caplog.text

    source.managers = {"bad": {"name": "Bad Manager", "cik": "1234567"}}
    with pytest.raises(ThirteenFError, match="All 13F managers failed"):
        source.fetch_all_increases()
    source.close()
