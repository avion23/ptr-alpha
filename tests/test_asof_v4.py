from datetime import date
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from analyzer.database import Database
from analyzer.row_history import archive_rows
from tests.test_asof_v2 import load_script, seed


@pytest.mark.parametrize("read_only", [False, True])
def test_production_jpm_transaction_start_inversion(tmp_path, read_only):
    # Reduced production row 267210, frozen audit DB (not the live DB):
    # doc 20026658, deterministic success started 89ms before row creation.
    path = tmp_path / "db.duckdb"
    db = Database(path)
    seed(db, doc="20026658", promoted="2026-09-03",
         accepted="2026-09-03 14:03:01.367326",
         created="2026-09-03 14:03:01.456549", disclosure="2025-01-27")
    db.conn.execute("UPDATE transactions SET member='Marjorie Taylor Greene', "
                    "transaction_date='2025-01-08', source_row_id='pdftotext:l196'")
    db.close()
    db = Database(path, read_only=read_only)
    try:
        assert db.get_transactions(2025).ticker.tolist() == ["JPM"]
        with db.canonical_as_of(date(2026, 9, 3)):
            assert db.get_transactions(2025).ticker.tolist() == ["JPM"]
        with db.canonical_as_of(date(2026, 9, 2)):
            assert db.get_transactions(2025).empty
    finally:
        db.close()


@pytest.mark.parametrize("legacy", [False, True])
def test_same_day_rewrite_cannot_reuse_old_acceptance(tmp_path, legacy):
    db = Database(tmp_path / "db.duckdb")
    try:
        row = seed(db, accepted=str(date.today()), created=str(date.today()))
        if not legacy:
            db.conn.execute("UPDATE pdf_parse_runs SET rowset_created_at = (SELECT MAX(created_at) FROM transactions)")
        row["ticker"] = "MSFT"
        db.upsert_transactions(pd.DataFrame([row]), source="house_pdf")
        with db.canonical_as_of(date.today()):
            assert db.get_transactions(2026).empty
    finally:
        db.close()


@pytest.mark.parametrize("ticker,expected", [("MICROSOFT CORPORATION", "MSFT"), ("US TREASURY BILL", None)])
def test_cleanup_ticker_preserves_pre_mutation_value(tmp_path, ticker, expected):
    path = tmp_path / "db.duckdb"
    db = Database(path)
    seed(db, accepted="2026-08-04", ticker=ticker)
    db.close()
    load_script("cleanup_tickers").main(path)
    db = Database(path)
    try:
        with db.canonical_as_of(date(2026, 9, 21)):
            assert db.get_transactions(2026).ticker.tolist() == [ticker]
        with db.canonical_as_of(date.today()):
            rows = db.get_transactions(2026)
            assert len(rows) == 1
            assert rows.ticker.isna().all() if expected is None else rows.ticker.tolist() == [expected]
    finally:
        db.close()


def test_failed_retry_archives_older_duplicate_success(tmp_path):
    db = Database(tmp_path / "db.duckdb")
    try:
        seed(db, accepted="2026-08-04")
        db.conn.execute("INSERT INTO pdf_parse_runs SELECT * REPLACE (TIMESTAMP '2026-10-01' AS parsed_at) FROM pdf_parse_runs")
        db.parse_runs.upsert(doc_id="delayed", year=2026, parser_version="pdftotext",
                             status="error", engines_attempted="pdftotext", raw_row_count=0,
                             transaction_count=0, artifact_sha256="sha", ingestion_generation="delayed")
        assert db.conn.execute("SELECT COUNT(*) FROM parse_run_versions").fetchone()[0] == 1
        with db.canonical_as_of(date(2026, 9, 21)):
            assert db.get_transactions(2026).ticker.tolist() == ["JPM"]
    finally:
        db.close()


def test_cleanup_duplicate_deletion_preserves_old_rowset(tmp_path):
    path = tmp_path / "db.duckdb"
    db = Database(path)
    row = seed(db, accepted="2026-08-04", ticker="MICROSOFT CORPORATION")
    row.update(ticker="MSFT", source_row_id="already-resolved")
    db.upsert_transactions(pd.DataFrame([row]), source="house_pdf")
    db.conn.execute("UPDATE transactions SET created_at='2026-08-02'")
    db.conn.execute("UPDATE pdf_parse_runs SET transaction_count=2")
    db.close()
    load_script("cleanup_tickers").main(path)
    db = Database(path)
    try:
        with db.canonical_as_of(date(2026, 9, 21)):
            assert set(db.get_transactions(2026).ticker) == {"MICROSOFT CORPORATION", "MSFT"}
        with db.canonical_as_of(date.today()):
            assert db.get_transactions(2026).ticker.tolist() == ["MSFT"]
    finally:
        db.close()


@pytest.mark.parametrize("repair", ["cleanup_tickers", "backfill_tickers"])
def test_partial_ticker_repair_preserves_complete_legacy_rowset(tmp_path, repair):
    path = tmp_path / "db.duckdb"
    original = "MICROSOFT CORPORATION" if repair == "cleanup_tickers" else None
    db = Database(path)
    row = seed(db, accepted="2026-08-04", created="2026-08-02 12:00:00", ticker=original)
    db.conn.execute("UPDATE transactions SET asset_description='Microsoft Corporation'")
    row.update(source_row_id="unchanged", ticker="JPM")
    db.upsert_transactions(pd.DataFrame([row]), source="house_pdf")
    db.conn.execute("UPDATE transactions SET created_at='2026-08-02 12:01:00' WHERE source_row_id='unchanged'")
    db.conn.execute("UPDATE pdf_parse_runs SET transaction_count=2")
    with db.canonical_as_of(date(2026, 9, 21)):
        assert db.get_transactions(2026).sort_values("source_row_id").ticker.fillna("NULL").tolist() == [original or "NULL", "JPM"]
    db.close()

    if repair == "cleanup_tickers":
        load_script(repair).main(path)
    else:
        assert load_script(repair).backfill_tickers(path, apply=True) == (1, 0)
    db = Database(path)
    try:
        assert db.conn.execute("SELECT COUNT(*) FROM transaction_versions").fetchone()[0] == 1
        with db.canonical_as_of(date(2026, 9, 21)):
            assert db.get_transactions(2026).sort_values("source_row_id").ticker.fillna("NULL").tolist() == [original or "NULL", "JPM"]
        with db.canonical_as_of(date.today()):
            assert db.get_transactions(2026).sort_values("source_row_id").ticker.tolist() == ["MSFT", "JPM"]
        assert db.get_transactions(2026).sort_values("source_row_id").ticker.tolist() == ["MSFT", "JPM"]
    finally:
        db.close()


def test_bulk_repair_does_not_make_repaired_doc_valid_in_the_past(tmp_path):
    path = tmp_path / "congress.duckdb"
    db = Database(path)
    row = seed(db, accepted="2026-08-04")
    row.update(source_row_id="bad-row", transaction_date="2026-08-05")
    db.upsert_transactions(pd.DataFrame([row]), source="house_pdf")
    db.conn.execute("UPDATE transactions SET created_at='2026-08-02'")
    db.conn.execute("UPDATE pdf_parse_runs SET transaction_count=2")
    with db.canonical_as_of(date(2026, 9, 21)):
        assert db.get_transactions(2026).empty
    db.close()
    (tmp_path / "manifest.json").write_text(json.dumps({"house": {}}))
    load_script("rebuild_staged").repair_audit_gaps(SimpleNamespace(staging=str(tmp_path)))
    db = Database(path)
    try:
        with db.canonical_as_of(date(2026, 9, 21)):
            assert db.get_transactions(2026).empty
        with db.canonical_as_of(date.today()):
            assert db.get_transactions(2026).ticker.tolist() == ["JPM"]
        assert db.conn.execute("SELECT COUNT(*) FROM transaction_versions").fetchone()[0] == 1
        assert db.conn.execute("SELECT COUNT(*) FROM parse_run_versions").fetchone()[0] >= 1
    finally:
        db.close()


def test_phantom_purge_archives_deleted_row_and_acceptance(tmp_path):
    db = Database(tmp_path / "db.duckdb")
    try:
        row = seed(db, accepted="2026-08-04", ticker=None)
        row["source_row_id"] = "duplicate"
        db.upsert_transactions(pd.DataFrame([row]), source="house_pdf")
        db.conn.execute("UPDATE transactions SET created_at='2026-08-02'")
        db.conn.execute("UPDATE pdf_parse_runs SET transaction_count=2")
        load_script("purge_phantom_rows").purge_phantom_rows(db.conn)
        with db.canonical_as_of(date(2026, 9, 21)):
            assert db.conn.execute("SELECT COUNT(*) FROM canonical_transactions").fetchone()[0] == 2
        with db.canonical_as_of(date.today()):
            assert len(db.get_transactions(2026)) == 1
    finally:
        db.close()


def test_initialization_migration_preserves_old_values(tmp_path):
    path = tmp_path / "db.duckdb"
    db = Database(path)
    seed(db, accepted="2026-08-04")
    db.conn.execute("UPDATE transactions SET owner_code=NULL, amount_raw=NULL")
    db.close()
    db = Database(path)
    try:
        with db.canonical_as_of(date(2026, 9, 21)):
            assert db.conn.execute("SELECT owner_code,amount_raw FROM canonical_transactions").fetchall() == [(None, None)]
        with db.canonical_as_of(date.today()):
            assert db.conn.execute("SELECT owner_code,amount_raw FROM canonical_transactions").fetchall() == [("", "")]
    finally:
        db.close()


def test_generation_status_mutation_does_not_change_historical_fallback(tmp_path):
    db = Database(tmp_path / "db.duckdb")
    try:
        row = seed(db)  # accepted Oct 1, not Sep 21
        db.conn.execute("UPDATE house_archive_generations SET parse_status='complete', completed_at='2026-08-05'")
        row.update(source_row_id="legacy", ingestion_generation="legacy-generation", ticker="MSFT")
        db.upsert_transactions(pd.DataFrame([row]), source="house_pdf")
        db.conn.execute("UPDATE transactions SET created_at='2026-08-02'")
        with db.canonical_as_of(date(2026, 9, 21)):
            assert db.get_transactions(2026).empty
        db.conn.execute("BEGIN TRANSACTION")
        archive_rows(db.conn, "house_archive_generations")
        db.conn.execute("UPDATE house_archive_generations SET parse_status='incomplete'")
        db.conn.execute("COMMIT")
        with db.canonical_as_of(date(2026, 9, 21)):
            assert db.get_transactions(2026).empty
        with db.canonical_as_of(date.today()):
            assert db.get_transactions(2026).ticker.tolist() == ["JPM"]
    finally:
        db.close()
