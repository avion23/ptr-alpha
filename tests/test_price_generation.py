from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from analyzer.database import Database
from analyzer.pipeline import eligible_events
from analyzer.price_source import (
    _merge_price_generation,
    _set_generation_state,
    _store_generation,
    canonical_price_hash,
    ensure_price_generation_schema,
    price_generation_state,
)
from analyzer.setups import score


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
