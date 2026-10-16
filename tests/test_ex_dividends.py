"""Ex-div entry gate (R9): block tickers with an ex-div in the 7-day window.

Missing dividend rows mean unknown, never clean: without the map the gate
is off and scoring is byte-identical to before. No network in this file —
provider responses are stubbed.
"""

from datetime import date
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from analyzer.setups import score

AS_OF = date(2024, 6, 1)


def _row(**overrides):
    base = {
        "ticker": "AAPL",
        "actor_id": "congress:PELOSI",
        "kind": "congress",
        "source": "house_pdf",
        "entry_ref": 100.0,
        "event_date": date(2024, 5, 30),
        "disclosure_date": date(2024, 5, 30),
        "corroboration": False,
        "position_evidence": True,
        "as_of": AS_OF,
    }
    base.update(overrides)
    return base


def _score(ex_dividends):
    candidates = pd.DataFrame([_row()])
    return score(
        candidates,
        {"congress:PELOSI": 1.0},
        {"AAPL": (100.0, AS_OF)},
        ex_dividends=ex_dividends,
    )


def test_ex_div_inside_window_blocks():
    ranked, blocked = _score({"AAPL": [date(2024, 5, 28)]})
    assert ranked.empty
    assert blocked.iloc[0]["reason"] == "ex-div window (ex 2024-05-28)"


def test_ex_div_seven_days_back_is_outside_window():
    ranked, _ = _score({"AAPL": [date(2024, 5, 25)]})  # window is (as_of-7d, as_of]
    assert not ranked.empty
    ranked, _ = _score({"AAPL": [date(2024, 5, 26)]})
    assert ranked.empty


def test_ex_div_outside_window_passes():
    ranked, _ = _score({"AAPL": [date(2024, 5, 24)]})
    assert not ranked.empty


def test_ex_div_after_as_of_passes():
    ranked, _ = _score({"AAPL": [date(2024, 6, 2)]})
    assert not ranked.empty


def test_other_ticker_ex_div_does_not_block():
    ranked, _ = _score({"MSFT": [date(2024, 5, 28)]})
    assert not ranked.empty


def test_missing_map_is_byte_identical_to_before():
    ranked_none, blocked_none = _score(None)
    ranked_empty, _ = _score({})
    assert ranked_none.equals(ranked_empty)
    assert blocked_none.empty


def test_malformed_ex_dates_are_ignored_not_fatal():
    ranked, _ = _score({"AAPL": ["not-a-date", None]})
    assert not ranked.empty


def test_audit_records_ex_div_block():
    candidates = pd.DataFrame([_row()])
    audit: dict = {}
    score(
        candidates,
        {"congress:PELOSI": 1.0},
        {"AAPL": (100.0, AS_OF)},
        ex_dividends={"AAPL": [date(2024, 5, 28)]},
        audit=audit,
    )
    assert audit["AAPL"]["ex_div_block"] == "2024-05-28"
    assert audit["AAPL"]["ranking_gate"] is False


def test_dividends_table_created_writable_only(tmp_path):
    import duckdb

    from analyzer.database import Database

    path = tmp_path / "d.duckdb"
    db = Database(path, read_only=False)
    cols = {
        row[1] for row in db.conn.execute("PRAGMA table_info(dividends)").fetchall()
    }
    assert {"ticker", "ex_date", "cash_amount"} <= cols
    db.close()
    # Read-only open of the migrated file: no schema change attempted.
    before = path.read_bytes()
    db = Database(path, read_only=True)
    db.conn.execute("SELECT COUNT(*) FROM dividends").fetchone()
    db.close()
    assert path.read_bytes() == before


def test_upsert_dividends_is_idempotent(tmp_path):
    from analyzer.database import Database
    from analyzer.price_source import upsert_dividends

    db = Database(tmp_path / "d.duckdb", read_only=False)
    try:
        frame = pd.DataFrame(
            [
                {"ticker": "abt", "ex_date": date(2026, 10, 15), "cash_amount": 0.59},
                {"ticker": "ABT", "ex_date": date(2026, 10, 15), "cash_amount": 0.59},
            ]
        )
        assert upsert_dividends(db, frame) == 1
        assert upsert_dividends(db, frame) == 1  # no duplicates
        assert upsert_dividends(db, pd.DataFrame()) == 0
    finally:
        db.close()


def test_fetch_dividend_events_parses_actions_without_network():
    from analyzer.price_source import fetch_dividend_events

    actions = pd.DataFrame(
        {"Dividends": [0.0, 0.59], "Stock Splits": [0.0, 0.0]},
        index=pd.to_datetime(["2026-07-15", "2026-10-15"]),
    )
    ticker = SimpleNamespace(actions=actions)
    empty = SimpleNamespace(actions=pd.DataFrame())

    def fake_ticker(symbol):
        return ticker if symbol == "ABT" else empty

    with patch("analyzer.price_source.yf.Ticker", side_effect=fake_ticker):
        with patch("analyzer.price_source.time.sleep", return_value=None):
            events = fetch_dividend_events(["ABT", "ZZZ"], sleep_s=0)
    assert len(events) == 1
    assert events.iloc[0]["ticker"] == "ABT"
    assert events.iloc[0]["ex_date"] == date(2026, 10, 15)
    assert events.iloc[0]["cash_amount"] == 0.59


def test_cli_ex_div_map_is_none_without_table():
    import duckdb

    from analyzer.cli import _ex_div_map

    conn = duckdb.connect()
    conn.execute("CREATE TABLE prices (ticker VARCHAR, date DATE, close DOUBLE)")
    db = SimpleNamespace(conn=conn)
    try:
        assert _ex_div_map(db, ["AAPL"], date(2024, 6, 1)) is None
    finally:
        conn.close()


def test_cli_ex_div_map_returns_window_only():
    import duckdb

    from analyzer.cli import _ex_div_map

    conn = duckdb.connect()
    conn.execute("CREATE TABLE dividends (ticker VARCHAR, ex_date DATE, cash_amount DOUBLE)")
    conn.execute(
        "INSERT INTO dividends VALUES ('AAPL', '2024-05-28', 0.24), ('AAPL', '2024-01-01', 0.24)"
    )
    db = SimpleNamespace(conn=conn)
    try:
        mapping = _ex_div_map(db, ["AAPL", "MSFT"], date(2024, 6, 1))
        assert mapping == {"AAPL": [date(2024, 5, 28)]}
    finally:
        conn.close()
