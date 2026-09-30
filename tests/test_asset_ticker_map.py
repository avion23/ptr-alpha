from __future__ import annotations

import duckdb
import pytest

from analyzer import asset_ticker_map
from scripts.backfill_tickers import backfill_tickers


@pytest.fixture
def sec_company_tickers(monkeypatch):
    fixture = {
        "0": {"cik_str": 1, "ticker": "ZTS", "title": "Zoetis Inc."},
        "1": {"cik_str": 2, "ticker": "MSFT", "title": "Microsoft Corporation"},
        "2": {"cik_str": 3, "ticker": "ACME", "title": "Acme Holdings, Inc."},
        "3": {"cik_str": 4, "ticker": "ACM.A", "title": "Acme Holdings Inc."},
        "4": {"cik_str": 5, "ticker": "O", "title": "Realty Income Corporation"},
        "5": {"cik_str": 6, "ticker": "ABCB", "title": "Ameris Bancorp"},
        "6": {"cik_str": 7, "ticker": "TBBK", "title": "Bancorp"},
    }
    rows = asset_ticker_map._company_rows(fixture)
    monkeypatch.setattr(asset_ticker_map, "_load_sec_company_tickers", lambda: rows)
    asset_ticker_map._sec_company_name_index.cache_clear()
    yield fixture
    asset_ticker_map._sec_company_name_index.cache_clear()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("ZOETIS INC CMN CLASS A", "ZOETIS"),
        ("Acme, Inc. Class B", "ACME"),
        ("Acme Corp. COMMON STOCK", "ACME"),
        ("Acme LLC CMN", "ACME"),
        ("Acme Ltd. Class C", "ACME"),
        ("Acme Co. Common Stock Class A", "ACME CO"),
    ],
)
def test_normalize_company_name_strips_security_suffixes(raw, expected):
    assert asset_ticker_map._normalize_company_name(raw) == expected


def test_normalize_ampersand_as_and():
    assert (
        asset_ticker_map._normalize_company_name("Air Products & Chemicals, Inc.")
        == "AIR PRODUCTS AND CHEMICALS"
    )


def test_resolves_sec_name_from_fixture(sec_company_tickers):
    assert asset_ticker_map.resolve_asset_ticker("ZOETIS INC CMN CLASS A") == "ZTS"
    assert (
        asset_ticker_map.resolve_asset_ticker("Microsoft Corporation Common Stock")
        == "MSFT"
    )


def test_resolves_curated_company_alias(sec_company_tickers):
    assert asset_ticker_map.resolve_asset_ticker("Par Petroleum Corporation") == "PARR"


@pytest.mark.parametrize(
    ("description", "expected"),
    [
        ("Realty Income Corporation Common Stock (O)", "O"),
        (
            "Realty Income Corporation Common Stock (O) "
            "[Account: Joint Brokerage - Bank of America]",
            "O",
        ),
        (
            "Realty Income Corporation Common Stock (O) "
            "Account: Joint Brokerage - Bank of America",
            "O",
        ),
        (
            "Realty Income Corporation Common Stock (O) "
            "Custodian=Joint Brokerage - Bank of America",
            "O",
        ),
        ("Private investment [Account = Bank of America]", None),
        ("Private investment (Account: Bank of America)", None),
        ("Account: Bank of America", None),
        ("Zoetis, Inc. Common Stock", "ZTS"),
        ("California municipal Bonds [Account: Bank of America]", None),
        ("Hybrid MTN [Account: Bank of America]", None),
        ("Tesla Growth FD", None),
        ("Zoetis and Microsoft Corporation", None),
        # A generic name nested inside a longer company name is one company.
        ("Ameris Bancorp - Common Stock", "ABCB"),
    ],
)
def test_resolves_only_one_held_company(sec_company_tickers, description, expected):
    assert asset_ticker_map.resolve_asset_ticker(description) == expected


def test_ambiguous_sec_name_returns_none(sec_company_tickers):
    assert asset_ticker_map.resolve_asset_ticker("Acme Holdings Inc. Class A") is None


@pytest.mark.parametrize(
    "description",
    [
        "JT",
        "U.S. Treasury Bills [GS]",
        "JPM 100% US TREAS INSTL SWEEP FD #199",
        "Vanguard S&P 500 ETF",
    ],
)
def test_funds_and_treasuries_are_unresolvable(sec_company_tickers, description):
    assert asset_ticker_map.resolve_asset_ticker(description) is None


def _make_database(path):
    connection = duckdb.connect(str(path))
    connection.execute(
        """
        CREATE TABLE transactions (
            id INTEGER PRIMARY KEY,
            ticker VARCHAR,
            ticker_origin VARCHAR,
            asset_description VARCHAR,
            amount_raw VARCHAR,
            member VARCHAR
        )
        """
    )
    connection.executemany(
        "INSERT INTO transactions VALUES (?, ?, ?, ?, ?, ?)",
        [
            (1, None, "missing", "Zoetis Inc CMN Class A", "$1,001 - $15,000", "A"),
            (2, None, "missing", "U.S. Treasury Bill [GS]", "$15,001 - $50,000", "B"),
            (3, "AAPL", "official", "Apple Inc.", "$1,001 - $15,000", "C"),
        ],
    )
    connection.close()


def _snapshot(path):
    connection = duckdb.connect(str(path), read_only=True)
    try:
        columns = [
            row[1]
            for row in connection.execute(
                "PRAGMA table_info('transactions')"
            ).fetchall()
        ]
        rows = connection.execute("SELECT * FROM transactions ORDER BY id").fetchall()
        return columns, rows
    finally:
        connection.close()


def test_backfill_dry_run_writes_nothing(tmp_path, sec_company_tickers):
    db_path = tmp_path / "dry-run.duckdb"
    _make_database(db_path)
    before = _snapshot(db_path)

    assert backfill_tickers(db_path) == (1, 1)
    assert _snapshot(db_path) == before


def test_backfill_apply_changes_only_ticker_and_origin(tmp_path, sec_company_tickers):
    db_path = tmp_path / "apply.duckdb"
    _make_database(db_path)
    columns_before, rows_before = _snapshot(db_path)

    assert backfill_tickers(db_path, apply=True) == (1, 1)
    columns_after, rows_after = _snapshot(db_path)

    assert columns_after == columns_before
    ticker_index = columns_before.index("ticker")
    origin_index = columns_before.index("ticker_origin")
    for before, after in zip(rows_before, rows_after, strict=True):
        assert [
            value
            for index, value in enumerate(after)
            if index not in {ticker_index, origin_index}
        ] == [
            value
            for index, value in enumerate(before)
            if index not in {ticker_index, origin_index}
        ]
    assert rows_after[0][ticker_index] == "ZTS"
    assert rows_after[0][origin_index] == "resolved_asset_name"
    assert rows_after[1] == rows_before[1]
    assert rows_after[2] == rows_before[2]
