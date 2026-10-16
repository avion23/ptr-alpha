import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from analyzer import asset_ticker_map
from analyzer.form13f import (
    ThirteenFError,
    ThirteenFSource,
    _parse_information_table,
    candidates_from_increases,
)
from analyzer.pipeline import eligible_events


@pytest.fixture
def sec_company_tickers(monkeypatch):
    rows = asset_ticker_map._company_rows(
        {
            "0": {"ticker": "MSFT", "title": "Microsoft Corporation"},
            "1": {"ticker": "AMZN", "title": "Amazon.com, Inc."},
            "2": {
                "ticker": "SPCX",
                "title": "Space Exploration Technologies Corp.",
            },
        }
    )
    monkeypatch.setattr(asset_ticker_map, "_load_sec_company_tickers", lambda: rows)
    asset_ticker_map._sec_company_name_index.cache_clear()
    yield
    asset_ticker_map._sec_company_name_index.cache_clear()


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


@pytest.mark.parametrize("discovery", [False, True])
def test_fetch_records_acceptance(tmp_path, monkeypatch, quarter_xmls, discovery):
    from datetime import datetime
    from analyzer.database import Database

    recent = {
        "form": ["13F-HR", "13F-HR"],
        "reportDate": ["2026-06-30", "2026-03-31"],
        "filingDate": ["2026-08-14", "2026-05-15"],
        "accessionNumber": ["0001234567-26-000002", "0001234567-26-000001"],
        "acceptanceDateTime": ["2026-08-14T21:30:00Z", "2026-05-15T20:00:00Z"],
    }
    previous, current = map(_parse_information_table, quarter_xmls)
    with Database(tmp_path / "test.duckdb") as db:
        source = ThirteenFSource(
            data_dir=tmp_path, db=db, watchlist={"VST": {"VISTRA"}},
            managers={"test": {"name": "Test Manager", "cik": "1234567"}},
            ingestion_generation="test", ticker_map={"111111111": "VST"},
        )
        calls = []
        def submissions(cik):
            calls.append("submissions")
            return {"filings": {"recent": recent}}
        def filing(cik, record):
            calls.append("filing")
            return (current if record["report_date"] == "2026-06-30" else previous), "https://example.test/report.xml"
        monkeypatch.setattr(source, "_load_submissions", submissions)
        monkeypatch.setattr(source, "_load_filing", filing)
        try:
            if discovery:
                rows = source._manager_increases("test", source.managers["test"])
                source.save_increases(pd.DataFrame(rows))
            else:
                rows = source._manager_transactions(source.managers["test"])
                from analyzer.form13f import _OUTPUT_COLUMNS
                source.save_to_db(pd.DataFrame(rows, columns=_OUTPUT_COLUMNS))
            expected = datetime(2026, 8, 14, 17, 30)
            assert rows and all(row["acceptance_datetime"] == expected for row in rows)
            assert db.conn.execute("SELECT DISTINCT acceptance_datetime FROM transactions").fetchall() == [(expected,)]
            assert calls == ["submissions", "filing", "filing"]
        finally:
            source.close()


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


@pytest.mark.parametrize(
    "case", ["numeric", "ambiguous", "no_table", "malformed", "too_many"]
)
def test_load_filing_identifies_schema_not_filename(
    tmp_path, quarter_xmls, monkeypatch, case
):
    index = json.loads(
        (Path(__file__).parent / "fixtures/form13f/berkshire-index.json").read_text()
    )
    table = quarter_xmls[0].encode()
    contents = {
        "53405.xml": table,
        "primary_doc.xml": b'<edgarSubmission xmlns="http://www.sec.gov/edgar/thirteenffiler" />',
    }
    if case == "ambiguous":
        index["directory"]["item"].append({"name": "infotable.xml"})
        contents["infotable.xml"] = table
    elif case == "no_table":
        contents["53405.xml"] = b"<unrelated><infoTable /></unrelated>"
    elif case == "malformed":
        contents["primary_doc.xml"] = b"<broken"
    elif case == "too_many":
        index["directory"]["item"].extend({"name": f"{n}.xml"} for n in range(7))
    calls = []

    def get(url):
        name = url.rsplit("/", 1)[-1]
        calls.append(name)
        return SimpleNamespace(json=lambda: index, content=contents.get(name, b""))

    filing = {"accession": "0001193125-26-226661"}
    with ThirteenFSource(
        data_dir=tmp_path, db=object(), watchlist={"VST": {"VISTRA"}}
    ) as source:
        monkeypatch.setattr(source, "_get", get)
        if case == "numeric":
            holdings, report_url = source._load_filing("1067983", filing)
            assert holdings == _parse_information_table(table)
            assert report_url.endswith("/1067983/000119312526226661/53405.xml")
            assert calls == ["index.json", "53405.xml", "primary_doc.xml"]
        else:
            match = {
                "ambiguous": "2 information-table",
                "no_table": "0 information-table",
                "malformed": "Unparseable SEC filing XML",
                "too_many": "Too many XML files",
            }[case]
            with pytest.raises(ThirteenFError, match=match):
                source._load_filing("1067983", filing)
            if case == "too_many":
                assert calls == ["index.json"]


def test_fetch_all_increases_drops_unmatched_and_keeps_mapped_positions(
    tmp_path, quarter_xmls, monkeypatch
):
    monkeypatch.setattr("analyzer.form13f.resolve_asset_ticker", lambda issuer: None)
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
        ("ORION", 400),
    ]
    assert increases["event_date"].eq(date(2026, 6, 30)).all()
    assert increases["disclosure_date"].eq(date(2026, 8, 14)).all()
    source.close()


def test_increases_resolve_tickers_drop_unknown_and_dedupe_cusips(
    tmp_path, sec_company_tickers
):
    source = ThirteenFSource(
        data_dir=tmp_path / "data",
        db=object(),
        watchlist={"VST": {"VISTRA"}},
        managers={"test": {"name": "Test Manager", "cik": "1234567"}},
    )
    current = _parse_information_table(
        """<informationTable>
        <infoTable><nameOfIssuer>Microsoft Corporation</nameOfIssuer><cusip>594918104</cusip><sshPrnamt>100</sshPrnamt><sshPrnamtType>SH</sshPrnamtType></infoTable>
        <infoTable><nameOfIssuer>Microsoft Corporation</nameOfIssuer><cusip>594918104</cusip><sshPrnamt>50</sshPrnamt><sshPrnamtType>SH</sshPrnamtType></infoTable>
        <infoTable><nameOfIssuer>Amazon.com, Inc.</nameOfIssuer><cusip>023135106</cusip><sshPrnamt>250</sshPrnamt><sshPrnamtType>SH</sshPrnamtType></infoTable>
        <infoTable><nameOfIssuer>Space Exploration Techn Corp</nameOfIssuer><cusip>84615Q103</cusip><sshPrnamt>225</sshPrnamt><sshPrnamtType>SH</sshPrnamtType></infoTable>
        <infoTable><nameOfIssuer>Private Project Holdings</nameOfIssuer><cusip>999999999</cusip><sshPrnamt>75</sshPrnamt><sshPrnamtType>SH</sshPrnamtType></infoTable>
        </informationTable>"""
    )

    rows = source._position_increases(
        "test",
        {"name": "Test Manager"},
        {
            "report_date": "2026-06-30",
            "filing_date": "2026-08-14",
            "accession": "0001234567-26-000002",
        },
        current,
        {"594918104": {"issuer": "Microsoft Corporation", "shares": 100}},
        "unused",
    )

    assert [(row["cusip"], row["ticker"], row["added_shares"]) for row in rows] == [
        ("594918104", "MSFT", 50),
        ("023135106", "AMZN", 250),
        ("84615Q103", "SPCX", 225),
    ]
    assert len({row["cusip"] for row in rows}) == len(rows)
    source.close()


def test_eligible_events_resolves_stored_null_13f_tickers(
    sec_company_tickers,
):
    trades = pd.DataFrame(
        [
            {
                "member": "Citadel Advisors LLC",
                "ticker": None,
                "raw_asset_description": "Microsoft Corporation",
                "transaction_date": date(2026, 3, 31),
                "disclosure_date": date(2026, 5, 15),
                "transaction_type": "Purchase",
                "instrument_type": "Common Stock",
                "source": "13f",
            },
            {
                "member": "Appaloosa LP",
                "ticker": None,
                "raw_asset_description": "Amazon.com, Inc.",
                "transaction_date": date(2026, 3, 31),
                "disclosure_date": date(2026, 5, 15),
                "transaction_type": "Purchase",
                "instrument_type": "Common Stock",
                "source": "13f",
            },
            {
                "member": "Appaloosa LP",
                "ticker": None,
                "raw_asset_description": "Space Exploration Techn Corp",
                "transaction_date": date(2026, 6, 30),
                "disclosure_date": date(2026, 8, 14),
                "transaction_type": "Purchase",
                "instrument_type": "Common Stock",
                "source": "13f",
            },
            {
                "member": "Citadel Advisors LLC",
                "ticker": "ABT",
                "raw_asset_description": "ABBOTT LABORATORIES",
                "transaction_date": date(2026, 6, 30),
                "disclosure_date": date(2026, 9, 2),
                "transaction_type": "Purchase",
                "instrument_type": "Common Stock",
                "source": "13f",
            },
            {
                "member": "Unknown Manager",
                "ticker": None,
                "raw_asset_description": "Private Project Holdings",
                "transaction_date": date(2026, 3, 31),
                "disclosure_date": date(2026, 5, 15),
                "transaction_type": "Purchase",
                "instrument_type": "Common Stock",
                "source": "13f",
            },
        ]
    )

    class ReadOnlyDB:
        conn = None

        def get_transactions_by_date_range(self, start, end):
            return trades.copy()

    events = eligible_events(ReadOnlyDB(), date(2026, 9, 3))

    assert events["ticker"].tolist() == ["MSFT", "AMZN", "SPCX", "ABT"]
    assert events["corroboration"].tolist() == [True, True, True, True]


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
        "position_evidence",
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
