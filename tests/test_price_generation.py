from datetime import date
from pathlib import Path
from unittest.mock import patch

import duckdb
import pandas as pd
import pytest

from analyzer.persistence.database import Database
from analyzer.pipeline import eligible_events
from analyzer.price_source import (
    _merge_price_generation,
    _set_generation_state,
    _store_generation,
    canonical_price_hash,
    certify_adjusted_price_generations,
    ensure_price_generation_schema,
    price_generation_state,
    price_generation_schema_exists,
    publish_price_generation_certification,
    resolve_setup_quotes,
)
from analyzer.setups import score


def test_generation_absence_is_not_a_failed_database_read():
    from types import SimpleNamespace

    conn = duckdb.connect(":memory:")
    db = SimpleNamespace(conn=conn)
    try:
        assert not price_generation_schema_exists(db)
        conn.execute("CREATE TABLE prices (ticker VARCHAR, date DATE, close DOUBLE)")
        assert price_generation_state(db, "AAA")["generation_id"] is None
        conn.execute("DROP TABLE prices")
        with pytest.raises(duckdb.Error):
            price_generation_state(db, "AAA")
    finally:
        conn.close()
    with pytest.raises(duckdb.Error):
        price_generation_schema_exists(db)


def _setup_quote_candidate(ticker, as_of):
    return {
        "ticker": ticker,
        "actor_id": "officer:CEO",
        "kind": "officer",
        "corroboration": False,
        "entry_ref": 101.0,
        "event_date": as_of,
        "disclosure_date": as_of,
        "position_evidence": True,
        "as_of": as_of,
    }


def _add_served_quote(db, ticker, status, day, close):
    dates = pd.to_datetime([day])
    values = pd.Series([close], index=dates)
    _store_generation(
        db,
        ticker,
        "served",
        values,
        pd.Series(1.0, index=dates),
        "adjusted-close-v1",
    )
    _set_generation_state(db, ticker, "served", status)


def test_certified_newer_setup_quote_ranks_but_unknown_generation_stays_stale():
    db = Database(":memory:")
    as_of = date(2026, 10, 26)
    base_day = pd.Timestamp("2026-10-16")
    try:
        ensure_price_generation_schema(db)
        db.upsert_prices(
            pd.DataFrame(
                {"CERT": [100.0], "UNKNOWN": [100.0]}, index=[base_day]
            )
        )
        _add_served_quote(db, "CERT", "comparable", "2026-10-23", 100.0)
        _add_served_quote(db, "UNKNOWN", "unknown", "2026-10-23", 100.0)

        quotes = resolve_setup_quotes(db, ["CERT", "UNKNOWN"], as_of)
        assert quotes == {
            "CERT": (100.0, date(2026, 10, 23)),
            "UNKNOWN": (100.0, date(2026, 10, 16)),
        }
        candidates = pd.DataFrame([
            _setup_quote_candidate("CERT", as_of),
            _setup_quote_candidate("UNKNOWN", as_of),
        ])
        ranked, blocked = score(
            candidates, {"officer:CEO": 1.0}, quotes, enable_13f_corroboration=False
        )
        assert ranked.ticker.tolist() == ["CERT"]
        assert blocked.to_dict("records") == [
            {"ticker": "UNKNOWN", "actor_id": "officer:CEO", "reason": "stale quote"}
        ]
    finally:
        db.close()


def test_served_setup_quote_still_obeys_future_guard():
    db = Database(":memory:")
    decision_date = date(2026, 10, 22)
    try:
        ensure_price_generation_schema(db)
        db.upsert_prices(pd.DataFrame(
            {"AAA": [100.0]}, index=pd.to_datetime(["2026-10-16"])
        ))
        _add_served_quote(db, "AAA", "comparable", "2026-10-24", 100.0)
        quotes = resolve_setup_quotes(db, ["AAA"], date(2026, 10, 26))
        ranked, blocked = score(
            pd.DataFrame([_setup_quote_candidate("AAA", decision_date)]),
            {"officer:CEO": 1.0}, quotes, enable_13f_corroboration=False,
        )
        assert ranked.empty
        assert blocked.iloc[0].reason == "future quote"
    finally:
        db.close()


def test_fresh_base_setup_quotes_keep_value_and_tie_ordering():
    db = Database(":memory:")
    as_of = date(2026, 10, 26)
    try:
        ensure_price_generation_schema(db)
        db.upsert_prices(pd.DataFrame(
            {"AAA": [100.0], "BBB": [100.0]},
            index=pd.to_datetime(["2026-10-26"]),
        ))
        _add_served_quote(db, "AAA", "comparable", "2026-10-26", 200.0)
        _add_served_quote(db, "BBB", "comparable", "2026-10-26", 200.0)
        quotes = resolve_setup_quotes(db, ["BBB", "AAA"], as_of)
        ranked, blocked = score(
            pd.DataFrame([
                _setup_quote_candidate("BBB", as_of),
                _setup_quote_candidate("AAA", as_of),
            ]),
            {"officer:CEO": 1.0}, quotes, enable_13f_corroboration=False,
        )
        assert ranked.ticker.tolist() == ["AAA", "BBB"]
        assert ranked.current.tolist() == [100.0, 100.0]
        assert blocked.empty
    finally:
        db.close()


def _certification_fetch(closes, factors=None):
    dates = pd.date_range("2024-01-02", periods=len(closes), freq="B")
    if factors is None:
        factors = [1.0] * len(closes)
    return pd.DataFrame({
        "date": dates,
        "close": closes,
        "adjustment_factor": factors,
    })


def test_independent_adjusted_fetches_must_agree_after_scale_normalization():
    identical = _certification_fetch([100.0, 101.0, 102.0])
    result = certify_adjusted_price_generations(
        {"fetch-a": identical, "fetch-b": identical.copy()},
        pd.DataFrame(columns=["ex_date", "cash_amount"]),
    )
    assert result["certified"]
    assert result["stats"]["max_normalized_fetch_error"] == 0.0

    divergent = _certification_fetch([100.0, 101.0, 103.0])
    result = certify_adjusted_price_generations(
        {"fetch-a": identical, "fetch-b": divergent},
        pd.DataFrame(columns=["ex_date", "cash_amount"]),
    )
    assert not result["certified"]
    assert "independent fetches disagree" in result["reason"]


def test_calibrated_fetch_tolerance_accepts_rounding_noise_not_larger_divergence():
    baseline = _certification_fetch([100.0, 101.0, 102.0])
    no_dividends = pd.DataFrame(columns=["ex_date", "cash_amount"])
    for relative_error, certified in ((1.1e-6, True), (1.3e-6, False)):
        noisy = _certification_fetch([100.0, 101.0 * (1.0 + relative_error), 102.0])
        result = certify_adjusted_price_generations(
            {"fetch-a": baseline, "fetch-b": noisy}, no_dividends
        )
        assert result["certified"] is certified


def test_dividend_reconstruction_uses_prior_session_raw_close():
    # $1 distribution from a $100 pre-ex-date close creates a 0.99 backward
    # factor; after the ex-date the factor is 1 and adjusted closes are flat.
    fetch = _certification_fetch([99.0, 99.0, 99.0], [0.99, 1.0, 1.0])
    events = pd.DataFrame({
        "ex_date": [pd.Timestamp("2024-01-03")],
        "cash_amount": [1.0],
    })
    result = certify_adjusted_price_generations(
        {"fetch-a": fetch, "fetch-b": fetch.copy()}, events
    )
    assert result["certified"]
    assert result["stats"]["max_dividend_reconstruction_error"] == 0.0


def test_split_factor_must_explain_raw_split_like_jump():
    # A 2:1 split halves raw close while factor doubles, leaving adjusted close
    # continuous; no dividend table action is needed for this split-free check.
    explained = _certification_fetch([50.0, 50.0, 51.0], [0.5, 1.0, 1.0])
    result = certify_adjusted_price_generations(
        {"fetch-a": explained, "fetch-b": explained.copy()},
        pd.DataFrame(columns=["ex_date", "cash_amount"]),
    )
    assert result["certified"]
    assert result["stats"]["explained_split_steps"] == 2

    unexplained = _certification_fetch([100.0, 50.0, 51.0], [1.0, 1.0, 1.0])
    result = certify_adjusted_price_generations(
        {"fetch-a": unexplained, "fetch-b": unexplained.copy()},
        pd.DataFrame(columns=["ex_date", "cash_amount"]),
    )
    assert not result["certified"]
    assert "raw split-like jump lacks factor evidence" in result["reason"]


def test_certified_spy_generation_is_published_through_state_machinery(tmp_path):
    db = Database(tmp_path / "certified-spy.duckdb")
    dates = pd.date_range("2024-01-02", periods=3, freq="B")
    values = pd.Series([100.0, 101.0, 102.0], index=dates)
    factors = pd.Series(1.0, index=dates)
    try:
        ensure_price_generation_schema(db)
        _store_generation(db, "SPY", "fetch-a", values, factors, "yfinance-adjusted-close-v1")
        _store_generation(db, "SPY", "fetch-b", values, factors, "yfinance-adjusted-close-v1")
        _set_generation_state(db, "SPY", "legacy-spy", "unknown", "fetch-a", "legacy raw")
        old = price_generation_state(db, "SPY")
        fetch = pd.DataFrame({
            "date": dates,
            "close": values.to_numpy(),
            "adjustment_factor": factors.to_numpy(),
        })
        certification = certify_adjusted_price_generations(
            {"fetch-a": fetch, "fetch-b": fetch.copy()},
            pd.DataFrame(columns=["ex_date", "cash_amount"]),
        )
        assert certification["certified"]

        published = publish_price_generation_certification(db, "SPY", certification, old)

        assert published["generation_id"] == "fetch-b"
        assert price_generation_state(db, "SPY")["generation_id"] == "fetch-b"
        assert price_generation_state(db, "SPY")["comparability_status"] == "comparable"
        assert "certified yfinance-adjusted-close-v1" in price_generation_state(db, "SPY")["reason"]
    finally:
        db.close()


def _store_generation_rowwise(db, ticker, generation, series, factors, basis):
    """Pre-bulk implementation, retained only as a parity oracle."""
    ensure_price_generation_schema(db)
    value_hash = canonical_price_hash(ticker, series, basis, generation)
    db.conn.execute(
        "INSERT INTO price_generations(ticker,generation,basis,value_hash) "
        "VALUES (?,?,?,?) ON CONFLICT(ticker,generation) DO UPDATE "
        "SET value_hash=excluded.value_hash,basis=excluded.basis",
        [ticker, generation, basis, value_hash],
    )
    rows = []
    for day, close in series.dropna().sort_index().items():
        factor = factors.get(day, float("nan"))
        rows.append((ticker, generation, pd.Timestamp(day).date(), float(close),
                     float(factor) if pd.notna(factor) else None, basis))
    if rows:
        db.conn.executemany(
            "INSERT INTO price_generation_values VALUES (?,?,?,?,?,?) "
            "ON CONFLICT(ticker,generation,date) DO UPDATE SET "
            "close=excluded.close, adjustment_factor=excluded.adjustment_factor, "
            "basis=excluded.basis", rows,
        )
    return value_hash


@pytest.mark.parametrize("empty", [False, True])
def test_bulk_generation_matches_rowwise_headers_values_hash_and_upsert(tmp_path, empty):
    db = Database(tmp_path / "parity.duckdb")
    dates = pd.to_datetime(["2024-01-04", "2024-01-02", "2024-01-03", "2024-01-05"])
    prices = pd.Series([103.125, 100.25, float("nan"), 104.5], index=dates)
    factors = pd.Series([0.49, float("nan"), 1.0, 0.98], index=dates)
    if empty:
        prices = prices.iloc[:0]
    try:
        ensure_price_generation_schema(db)
        # Both implementations see the same transaction timestamp, so even the
        # complete generation header (including created_at) must match exactly.
        db.conn.execute("BEGIN")
        snapshots = []
        for store in (_store_generation_rowwise, _store_generation):
            hashes = [store(db, "AAPL", "gen", prices, factors, "adjusted-close-v1")]
            original_created = db.conn.execute(
                "SELECT created_at FROM price_generations"
            ).fetchone()
            hashes.append(store(db, "AAPL", "gen", prices + 0.125,
                                factors * 0.5, "updated-basis"))
            assert db.conn.execute("SELECT created_at FROM price_generations").fetchone() == original_created
            snapshots.append((hashes, db.conn.execute(
                "SELECT * FROM price_generations ORDER BY ticker,generation"
            ).fetchall(), db.conn.execute(
                "SELECT * FROM price_generation_values ORDER BY ticker,generation,date"
            ).fetchall()))
            db.conn.execute("DELETE FROM price_generation_values")
            db.conn.execute("DELETE FROM price_generations")
        assert snapshots[0] == snapshots[1]
        assert len(snapshots[1][2]) == (0 if empty else 3)
        if not empty:
            assert snapshots[1][2][0][3:] == (100.375, None, "updated-basis")
        db.conn.execute("ROLLBACK")
    finally:
        db.close()


def test_bulk_generation_preserves_outer_transaction_rollback(tmp_path):
    db = Database(tmp_path / "rollback.duckdb")
    prices = pd.Series([100.0], index=pd.to_datetime(["2024-01-02"]))
    try:
        ensure_price_generation_schema(db)
        _store_generation(db, "AAPL", "gen", prices, prices * 0.01, "old-basis")
        _set_generation_state(db, "AAPL", "gen", "comparable")
        before = [db.conn.execute(f"SELECT * FROM {table}").fetchall() for table in (
            "price_generations", "price_generation_values", "price_generation_state"
        )]
        db.conn.execute("BEGIN")
        _store_generation(db, "AAPL", "gen", prices + 1, prices * 0.005, "new-basis")
        _store_generation(db, "AAPL", "new-gen", prices, prices * 0.01, "new-basis")
        _set_generation_state(db, "AAPL", "new-gen", "unknown")
        db.conn.execute("ROLLBACK")
        after = [db.conn.execute(f"SELECT * FROM {table}").fetchall() for table in (
            "price_generations", "price_generation_values", "price_generation_state"
        )]
        assert after == before
    finally:
        db.close()


def test_failed_generation_does_not_publish_state(tmp_path):
    db = Database(tmp_path / "failure.duckdb")
    try:
        ensure_price_generation_schema(db)
        _set_generation_state(db, "AAPL", "served", "comparable")
        before = db.conn.execute("SELECT * FROM price_generation_state").fetchall()
        with pytest.raises(duckdb.ConstraintException, match="NOT NULL"):
            _store_generation(db, "AAPL", "bad-gen",
                              pd.Series([100.0], index=pd.DatetimeIndex([pd.NaT])),
                              pd.Series(dtype=float), "adjusted-close-v1")
        # As before, a header may exist after an autocommit failure, but the
        # helper must never publish it as the served generation.
        assert db.conn.execute("SELECT generation FROM price_generations").fetchall() == [("bad-gen",)]
        assert db.conn.execute("SELECT * FROM price_generation_state").fetchall() == before
        assert db.conn.execute("SELECT * FROM price_generation_values").fetchall() == []
    finally:
        db.close()


def test_generation_hash_and_divergent_merge_are_explicit(tmp_path: Path):
    db = Database(tmp_path / "prices.duckdb")
    try:
        ensure_price_generation_schema(db)
        dates = pd.date_range("2024-01-02", periods=2, freq="B")
        old = pd.DataFrame({"AAPL": [100.0, 101.0]}, index=dates)
        db.upsert_prices(old)
        generation = "gen-old"
        _store_generation(
            db, "AAPL", generation, old.AAPL, pd.Series([1.0, 1.0], index=dates),
            "adjusted-close-v1",
        )
        _set_generation_state(db, "AAPL", generation, "comparable")

        same_hash = canonical_price_hash("AAPL", old.AAPL, "adjusted-close-v1", generation)
        assert same_hash == canonical_price_hash(
            "AAPL", old.AAPL.copy(), "adjusted-close-v1", generation
        )
        assert same_hash != canonical_price_hash(
            "AAPL", old.AAPL, "adjusted-close-v1", "gen-other"
        )

        incoming = pd.DataFrame({"AAPL": [50.0, 50.5]}, index=dates)
        factors = pd.DataFrame({"AAPL": [1.0, 1.0]}, index=dates)
        merged, states = _merge_price_generation(
            db, incoming, factors, "adjusted-close-v1", persist=True,
            cached_prices=old,
        )

        assert merged.AAPL.tolist() == [100.0, 101.0]
        assert db.get_prices(["AAPL"], dates[0].date(), dates[-1].date()).AAPL.tolist() == [100.0, 101.0]
        assert states["AAPL"]["comparability_status"] == "unknown"
        assert "2 overlaps exceed 0.1%" in states["AAPL"]["reason"]
        assert states["AAPL"]["alternate_generation"]
        assert price_generation_state(db, "AAPL")["comparability_status"] == "unknown"
        version_count = db.conn.execute(
            "SELECT COUNT(*) FROM price_generation_values WHERE ticker='AAPL'"
        ).fetchone()[0]
        assert version_count == 4
    finally:
        db.close()


def test_within_tolerance_values_merge_into_one_generation(tmp_path: Path):
    db = Database(tmp_path / "prices.duckdb")
    try:
        ensure_price_generation_schema(db)
        dates = pd.date_range("2024-01-02", periods=2, freq="B")
        old = pd.DataFrame({"AAPL": [100.0]}, index=dates[:1])
        db.upsert_prices(old)
        incoming = pd.DataFrame({"AAPL": [100.05, 101.0]}, index=dates)
        factors = pd.DataFrame({"AAPL": [1.0, 1.0]}, index=dates)

        merged, states = _merge_price_generation(
            db, incoming, factors, "adjusted-close-v1", persist=True,
            cached_prices=old,
        )

        assert merged.AAPL.tolist() == [100.05, 101.0]
        assert states["AAPL"]["comparability_status"] == "comparable"
        assert db.conn.execute("SELECT COUNT(*) FROM prices WHERE ticker='AAPL'").fetchone()[0] == 2
    finally:
        db.close()


def _form4_trade(transaction_day: str, disclosure_day: str) -> pd.DataFrame:
    return pd.DataFrame([{
        "member": "Jane Doe",
        "ticker": "AAPL",
        "transaction_date": pd.Timestamp(transaction_day),
        "disclosure_date": pd.Timestamp(disclosure_day),
        "transaction_type": "Purchase",
        "instrument_type": "Common Stock",
        "source": "form4",
        "amount_midpoint": 2000.0,
        "asset_description": "Common Stock; 10 shares @ $200; is_10b5_1=false",
    }])


def _events_with_factor(db, factor):
    dates = pd.date_range("2024-01-02", periods=4, freq="B")
    prices = pd.DataFrame({"AAPL": [98.0, 99.0, 100.0, 100.0]}, index=dates)
    db.upsert_prices(prices)
    generation = "split-dividend-gen"
    factors = pd.Series([factor] * len(dates), index=dates, dtype=float)
    _store_generation(db, "AAPL", generation, prices.AAPL, factors, "adjusted-close-v1")
    _set_generation_state(db, "AAPL", generation, "comparable")
    trades = _form4_trade("2024-01-02", "2024-01-05")
    with patch.object(db, "get_transactions_by_date_range", return_value=trades):
        return eligible_events(db, date(2024, 1, 5), days_back=None)


def test_form4_reference_normalizes_to_split_dividend_basis_and_passes_gate(tmp_path: Path):
    db = Database(tmp_path / "prices.duckdb")
    try:
        ensure_price_generation_schema(db)
        # 2:1 split (0.5) and 2% dividend adjustment (0.98) in Adj Close / Close.
        events = _events_with_factor(db, 0.49)
        event = events.iloc[0]
        assert event.filed_entry_ref == 200.0
        assert abs(event.normalized_entry_ref - 98.0) < 1e-10
        assert event.entry_ref == event.normalized_entry_ref
        assert event.comparability_status == "comparable"
        assert event.price_generation_id == "split-dividend-gen"

        ranked, blocked = score(
            events,
            {"officer:JANE DOE": 1.0},
            {"AAPL": (100.0, "2024-01-05")},
        )
        assert ranked.ticker.tolist() == ["AAPL"]
        assert blocked.empty
    finally:
        db.close()


def test_missing_action_factor_blocks_with_explicit_unknown_reason(tmp_path: Path):
    db = Database(tmp_path / "prices.duckdb")
    try:
        ensure_price_generation_schema(db)
        events = _events_with_factor(db, float("nan"))
        event = events.iloc[0]
        assert event.comparability_status == "unknown"
        assert pd.isna(event.entry_ref)
        assert "price comparability unknown" in event.blocked_reason
        ranked, blocked = score(
            events,
            {"officer:JANE DOE": 1.0},
            {"AAPL": (100.0, "2024-01-05")},
        )
        assert ranked.empty
        assert blocked.iloc[0].reason == event.blocked_reason
    finally:
        db.close()


def test_batched_generation_evidence_preserves_exact_stored_identity(tmp_path):
    from analyzer.price_source import price_generation_basis_map, price_generation_states

    db = Database(tmp_path / "case-identity.duckdb")
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
    old = pd.Series([100.0, 101.0], index=dates)
    factors = pd.Series(0.5, index=dates)
    try:
        db.upsert_prices(pd.DataFrame({"AAA": old}))
        _store_generation(db, "AAA", "old-AAA", old, factors, "old-basis")
        _set_generation_state(db, "AAA", "old-AAA", "comparable")
        # SQL identities are case/whitespace sensitive, even on legacy schemas.
        for ticker in ("aaa", " AAA "):
            _store_generation(db, ticker, "old-AAA", old, factors, "wrong-basis")
            _set_generation_state(db, ticker, "wrong-generation", "unknown")
        assert set(price_generation_states(db)) == {"AAA", "aaa", " AAA "}
        assert price_generation_basis_map(db)[("AAA", "old-AAA")] == "old-basis"
        _, states = _merge_price_generation(
            db, pd.DataFrame({"AAA": [50.0, 51.0]}, index=dates),
            pd.DataFrame(), "new-basis", persist=True, cached_prices=pd.DataFrame(),
        )
        assert states["AAA"]["generation_id"] == "old-AAA"
        assert price_generation_state(db, " aaa ")["generation_id"] == "old-AAA"
        assert db.conn.execute(
            "SELECT basis FROM price_generations WHERE ticker='AAA' AND generation='old-AAA'"
        ).fetchone() == ("old-basis",)
        assert db.conn.execute(
            "SELECT count(*) FROM price_generations WHERE ticker='AAA' AND generation='wrong-generation'"
        ).fetchone() == (0,)
    finally:
        db.close()


@pytest.mark.parametrize("outer_transaction", [False, True])
def test_merge_generation_failure_retains_per_ticker_write_order(tmp_path, outer_transaction):
    db = Database(tmp_path / "failure-order.duckdb")
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
    try:
        old = pd.DataFrame({"AAA": [100.0, 101.0], "BBB": [200.0, 201.0]}, index=dates)
        db.upsert_prices(old)
        ensure_price_generation_schema(db)
        db.conn.execute("DROP TABLE price_generation_values")
        db.conn.execute(
            "CREATE TABLE price_generation_values (ticker VARCHAR, generation VARCHAR, "
            "date DATE, close DOUBLE, adjustment_factor DOUBLE CHECK (adjustment_factor <= 1), "
            "basis VARCHAR, PRIMARY KEY(ticker,generation,date))"
        )
        for ticker in old:
            _set_generation_state(db, ticker, "old-" + ticker, "comparable")
        if outer_transaction:
            db.conn.execute("BEGIN")
        incoming_dates = pd.to_datetime(["2024-01-03", "2024-01-04"])
        with pytest.raises(duckdb.ConstraintException):
            _merge_price_generation(
                db, pd.DataFrame({"AAA": [101.0, 102.0], "BBB": [201.0, 202.0],
                                  "CCC": [50.0, 51.0]}, index=incoming_dates),
                pd.DataFrame({"AAA": [0.5, 0.5], "BBB": [2.0, 2.0],
                              "CCC": [0.5, 0.5]}, index=incoming_dates),
                "new-basis", persist=True, cached_prices=pd.DataFrame(),
            )
        if outer_transaction:
            db.conn.execute("ROLLBACK")
            assert db.conn.execute("SELECT count(*) FROM prices").fetchone() == (4,)
            assert db.conn.execute("SELECT count(*) FROM price_generations").fetchone() == (0,)
            assert db.conn.execute(
                "SELECT served_generation FROM price_generation_state WHERE ticker='AAA'"
            ).fetchone() == ("old-AAA",)
        else:
            assert db.conn.execute(
                "SELECT ticker, close FROM prices WHERE date='2024-01-04' ORDER BY ticker"
            ).fetchall() == [("AAA", 102.0), ("BBB", 202.0)]
            assert db.conn.execute(
                "SELECT DISTINCT ticker FROM price_generation_values ORDER BY ticker"
            ).fetchall() == [("AAA",)]
            states = dict(db.conn.execute(
                "SELECT ticker, served_generation FROM price_generation_state"
            ).fetchall())
            assert states["AAA"].startswith("yahoo:")
            assert states["BBB"] == "old-BBB"
            assert db.conn.execute(
                "SELECT ticker FROM price_generations ORDER BY ticker"
            ).fetchall() == [("AAA",), ("BBB",)]
    finally:
        db.close()


@pytest.mark.parametrize("conflict", [False, True])
def test_canonical_merge_coerces_legacy_text_factors(tmp_path, conflict):
    from analyzer.price_source import price_generation_factors
    from scripts.refresh_prices import merge_temp_into_canonical

    canonical = tmp_path / "text-factors.duckdb"
    source = tmp_path / "source.duckdb"
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
    db = Database(canonical)
    try:
        old = pd.Series([100.0, 101.0], index=dates)
        db.upsert_prices(pd.DataFrame({"AAA": old}))
        _store_generation(db, "AAA", "old-AAA", old, old * 0.005, "old-basis")
        _set_generation_state(db, "AAA", "old-AAA", "comparable")
        db.conn.execute("ALTER TABLE price_generation_values ALTER adjustment_factor TYPE VARCHAR")
        db.conn.execute("UPDATE price_generation_values SET adjustment_factor='unknown'")
        assert price_generation_factors(db, {"AAA": "old-AAA"})["AAA"].isna().all()
    finally:
        db.close()
    db = Database(source)
    try:
        db.upsert_prices(pd.DataFrame(
            {"AAA": [50.0 if conflict else 101.0, 102.0]},
            index=pd.to_datetime(["2024-01-03", "2024-01-04"]),
        ))
    finally:
        db.close()
    stats = merge_temp_into_canonical(source, canonical, ["AAA"])
    assert stats["prices_added"] == (0 if conflict else 1)
    db = Database(canonical, read_only=True)
    try:
        state = price_generation_state(db, "AAA")
        assert state["comparability_status"] == ("unknown" if conflict else "comparable")
        if conflict:
            assert state["generation_id"] == "old-AAA"
        assert db.conn.execute(
            "SELECT adjustment_factor FROM price_generation_values "
            "WHERE ticker='AAA' AND generation=? AND date='2024-01-02'",
            [state["generation_id"]],
        ).fetchone() == (None,)
    finally:
        db.close()
