"""Regression tests for the staged rebuild parser boundary."""

import json
from datetime import date
from types import SimpleNamespace
from typing import cast

import pandas as pd
import pytest


def test_required_house_scope_matches_production_inputs_not_full_archive():
    from scripts import rebuild_staged

    required = set(rebuild_staged.REQUIRED_HOUSE_YEARS)
    assert set(range(2021, 2026)).issubset(required)
    assert date.today().year in required
    assert max(2015, date.today().year - 1) in required
    assert 2015 not in required
    assert required.issubset(set(rebuild_staged.HOUSE_YEARS))


def test_senate_ingest_rejects_generation_provenance_rewrite(tmp_path):
    from scripts import rebuild_staged

    (tmp_path / "transactions.jsonl").write_text(
        json.dumps({"ingestion_generation": "wrong-generation"}) + "\n"
    )
    (tmp_path / "report_inventory.jsonl").write_text(
        json.dumps(
            {
                "ingestion_generation": "manifest-generation",
                "raw_row_count": 0,
                "accepted_row_count": 0,
                "rejected_row_count": 0,
            }
        )
        + "\n"
    )

    class FakeDatabase:
        def persist_source_refresh(self, **_kwargs):
            raise AssertionError("mismatched provenance must fail before persistence")

    with pytest.raises(ValueError, match="generation provenance mismatch"):
        rebuild_staged._ingest_senate(
            tmp_path,
            {"generation": "manifest-generation"},
            cast(rebuild_staged.Database, FakeDatabase()),
        )


def test_primary_text_preflight_only_prioritizes_reconcilable_candidates(monkeypatch):
    from scripts import rebuild_staged

    row = {
        "asset_description": "Apple Inc. (AAPL)",
        "transaction_type": "Purchase",
        "transaction_date": pd.Timestamp("2026-01-02"),
        "amount_raw": "$1,001 - $15,000",
        "owner_code": "JT",
        "notification_date": pd.Timestamp("2026-01-03"),
    }
    monkeypatch.setattr(
        rebuild_staged._parser_cascade,
        "_try_pdfplumber",
        lambda _path: [row],
    )
    monkeypatch.setattr(
        rebuild_staged._parser_cascade,
        "_try_pdftotext",
        lambda _path: [dict(row)],
    )
    assert rebuild_staged._primary_text_engines_reconcile(
        rebuild_staged.Path("matching.pdf")
    )

    conflicting = dict(row)
    conflicting["transaction_date"] = pd.Timestamp("2026-01-04")
    monkeypatch.setattr(
        rebuild_staged._parser_cascade,
        "_try_pdftotext",
        lambda _path: [conflicting],
    )
    assert not rebuild_staged._primary_text_engines_reconcile(
        rebuild_staged.Path("conflicting.pdf")
    )


def test_house_parse_keeps_winning_fallback_rows_and_quarantines_total_failure(
    monkeypatch, tmp_path
):
    from scripts import rebuild_staged

    fallback_path = tmp_path / "fallback.pdf"
    failed_path = tmp_path / "failed.pdf"
    fallback_path.write_bytes(b"%PDF-fallback\n%%EOF")
    failed_path.write_bytes(b"%PDF-failed\n%%EOF")

    class FakeSource:
        def fetch_metadata(self, _year):
            return pd.DataFrame(
                {
                    "DocID": ["fallback", "failed"],
                    "FilingType": ["P", "P"],
                }
            )

        def close(self):
            pass

    class FakeQuery:
        def __init__(self, rows=()):
            self.rows = list(rows)

        def fetchall(self):
            return self.rows

    class FakeConnection:
        def execute(self, sql, *_args):
            if "SELECT DISTINCT doc_id" in sql:
                return FakeQuery([("failed",)])
            return FakeQuery()

    class FakeParseRuns:
        def get_cached_doc_ids(self, **_kwargs):
            return set()

    class FakeDatabase:
        def __init__(self):
            self.conn = FakeConnection()
            self.parse_runs = FakeParseRuns()
            self.replacement = None

        def get_latest_house_generation(self, year):
            assert year == 2026
            return "generation-2026"

        def replace_transactions_for_docs(self, dataframe, **kwargs):
            self.replacement = (dataframe.copy(), kwargs)
            return SimpleNamespace(by_doc_total={"fallback": 1, "failed": 0})

    class FakePool:
        def __init__(self, _workers):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def map(self, worker, paths, chunksize=1):
            assert chunksize == 1
            return [worker(path) for path in paths]

        def imap_unordered(self, worker, paths, chunksize=1):
            assert chunksize == 1
            for path in paths:
                yield worker(path)

    fallback_rows = [
        {
            "doc_id": "fallback",
            "transaction_date": pd.Timestamp("2026-06-20"),
            "disclosure_date": pd.Timestamp("2026-06-26"),
            "source_row_id": "pdftotext:r1",
            "asset_description": "Apple Inc. (AAPL)",
        }
    ]

    def fake_parser(path):
        if path == fallback_path:
            return path, fallback_rows, ["lattice", "error:lattice", "won:pdftotext"]
        raise rebuild_staged.ParserCascadeError("unresolved parser completeness")

    def fake_consolidate(pdf_transactions, _member_lookup):
        assert pdf_transactions[fallback_path] == fallback_rows
        assert pdf_transactions[failed_path] == []
        return pd.DataFrame(fallback_rows)

    fake_database = FakeDatabase()
    monkeypatch.setattr(rebuild_staged, "_house_source", lambda _staging: FakeSource())
    monkeypatch.setattr(
        rebuild_staged,
        "_filter_existing_pdfs",
        lambda _ptrs, _pdf_dir: (
            [failed_path, fallback_path],
            pd.DataFrame({"DocID": ["failed", "fallback"]}),
        ),
    )
    monkeypatch.setattr(rebuild_staged, "_build_member_lookup", lambda _docs: {})
    monkeypatch.setattr(
        rebuild_staged,
        "_settings_for",
        lambda _staging: SimpleNamespace(
            data=SimpleNamespace(get_workers=lambda: 1)
        ),
    )
    monkeypatch.setattr(rebuild_staged, "Pool", FakePool)
    monkeypatch.setattr(
        rebuild_staged,
        "_primary_text_engines_reconcile",
        lambda path: path == fallback_path,
    )
    monkeypatch.setattr(rebuild_staged, "_parse_pdf_worker", fake_parser)
    monkeypatch.setattr(rebuild_staged, "consolidate_transactions", fake_consolidate)
    monkeypatch.setattr(
        rebuild_staged, "preserve_existing_fields", lambda dataframe, _db: dataframe
    )

    result = rebuild_staged._parse_house_year_tolerant(
        tmp_path, cast(rebuild_staged.Database, fake_database), 2026
    )

    assert result["parse_run_statuses"] == {"success": 1, "error": 1}
    assert fake_database.replacement is not None
    stored_df, kwargs = fake_database.replacement
    assert stored_df["doc_id"].tolist() == ["fallback"]
    assert kwargs["attempted_doc_ids"] == ["fallback", "failed"]
    assert kwargs["replacement_doc_ids"] == ["fallback"]

    parse_runs = {run["doc_id"]: run for run in kwargs["parse_runs"]}
    assert parse_runs["fallback"]["status"] == "success"
    assert parse_runs["fallback"]["error_message"] is None
    assert parse_runs["fallback"]["engines_attempted"] == (
        "lattice,error:lattice,won:pdftotext"
    )
    assert parse_runs["failed"]["status"] == "error"
    assert parse_runs["failed"]["error_message"] == "unresolved parser completeness"
    assert parse_runs["failed"]["engines_attempted"] == "cascade-failed"


def _seed_house_inventory_database(tmp_path):
    from analyzer.database import Database

    db = Database(tmp_path / "house-inventory.duckdb")
    db.conn.execute(
        """
        INSERT INTO house_archive_generations (
            archive_year, generation_id, metadata_count, ptr_count
        ) VALUES (2026, 'house-generation', 3, 3)
        """
    )
    db.conn.executemany(
        """
        INSERT INTO house_generation_metadata (
            archive_year, generation_id, doc_id, first_name, last_name,
            filing_date, filing_type
        ) VALUES (2026, 'house-generation', ?, ?, ?, '2026-01-03', 'P')
        """,
        [("deterministic", "Jane", "Doe"), ("ocr", "John", "Doe"), ("empty", "Alex", "Doe")],
    )
    db.conn.executemany(
        """
        INSERT INTO house_pdf_artifacts (
            archive_year, generation_id, doc_id, artifact_sha256
        ) VALUES (2026, 'house-generation', ?, ?)
        """,
        [("deterministic", "a" * 64), ("ocr", "b" * 64), ("empty", "c" * 64)],
    )
    db.conn.execute(
        """
        INSERT INTO transactions (
            doc_id, member, ticker, transaction_date, disclosure_date,
            transaction_type, source, chamber, source_record_id, source_row_id,
            official_filing_date, ingestion_generation, artifact_sha256
        ) VALUES (
            'deterministic', 'Jane Doe', 'AAPL', '2026-01-01', '2026-01-03',
            'Purchase', 'house_pdf', 'house', 'deterministic', 'pdf:r1',
            '2026-01-03', 'house-generation', ?
        )
        """,
        ["a" * 64],
    )
    db.conn.execute(
        """
        INSERT INTO transactions (
            doc_id, member, ticker, transaction_date, disclosure_date,
            transaction_type, source, chamber, source_record_id, source_row_id,
            official_filing_date, ingestion_generation, artifact_sha256
        ) VALUES (
            'ocr', 'John Doe', 'MSFT', '2026-01-01', '2026-01-03',
            'Purchase', 'gemini_ocr', 'House', 'ocr', 'ocr:r1',
            '2026-01-03', 'house-generation', ?
        )
        """,
        ["b" * 64],
    )
    db.upsert_parse_run(
        "deterministic", 2026, "v5-deterministic", "success", "pdf", 1, 1,
        artifact_sha256="a" * 64, ingestion_generation="house-generation",
    )
    db.upsert_parse_run(
        "ocr", 2026, "v5-gemini-validated", "success", "ocr", 1, 1,
        artifact_sha256="b" * 64, ingestion_generation="house-generation",
    )
    db.upsert_parse_run(
        "empty", 2026, "v5-deterministic", "no_txs", "pdf", 0, 0,
        artifact_sha256="c" * 64, ingestion_generation="house-generation",
    )
    return db


def test_house_inventory_rows_include_mixed_sources_and_no_txs(tmp_path):
    from scripts import rebuild_staged

    db = _seed_house_inventory_database(tmp_path)
    try:
        rows = rebuild_staged._house_inventory_rows(
            db, 2026, "house-generation"
        )
    finally:
        db.close()

    assert [row["source_record_id"] for row in rows] == [
        "deterministic", "empty", "ocr"
    ]
    by_id = {row["source_record_id"]: row for row in rows}
    assert by_id["deterministic"]["source"] == "house_pdf"
    assert by_id["deterministic"]["outcome"] == "parsed"
    assert by_id["ocr"]["source"] == "gemini_ocr"
    assert by_id["ocr"]["outcome"] == "parsed"
    assert by_id["empty"]["source"] == "house_pdf"
    assert by_id["empty"]["outcome"] == "no_txs"
    assert by_id["empty"]["accepted_row_count"] == 0


def test_house_inventory_rows_reject_ambiguous_terminal_runs(tmp_path):
    from scripts import rebuild_staged

    db = _seed_house_inventory_database(tmp_path)
    try:
        db.upsert_parse_run(
            "ocr", 2026, "v5-gemini-alt", "success", "pdf", 1, 1,
            artifact_sha256="b" * 64, ingestion_generation="house-generation",
        )
        with pytest.raises(RuntimeError, match="ambiguous terminal parse runs"):
            rebuild_staged._house_inventory_rows(db, 2026, "house-generation")
    finally:
        db.close()


def test_house_inventory_rows_reject_stale_source_bindings(tmp_path):
    from scripts import rebuild_staged

    db = _seed_house_inventory_database(tmp_path)
    try:
        db.conn.execute(
            """
            INSERT INTO transactions (
                doc_id, member, ticker, transaction_date, disclosure_date,
                transaction_type, source, chamber, source_record_id, source_row_id,
                official_filing_date, ingestion_generation, artifact_sha256
            ) VALUES (
                'ocr', 'John Doe', 'TSLA', '2026-01-01', '2026-01-03',
                'Purchase', 'house_pdf', 'house', 'ocr', 'stale:r1',
                '2026-01-03', 'house-generation', ?
            )
            """,
            ["b" * 64],
        )
        with pytest.raises(RuntimeError, match="stale source/artifact bindings"):
            rebuild_staged._house_inventory_rows(db, 2026, "house-generation")
    finally:
        db.close()


def test_refresh_house_completion_replaces_each_source_inventory(tmp_path):
    from scripts import rebuild_staged

    db = _seed_house_inventory_database(tmp_path)
    try:
        house = {
            "generation_id": "house-generation",
            "ptr_count": 3,
        }
        unresolved, report_count = rebuild_staged._refresh_house_completion(
            db, house, 2026
        )
        assert unresolved == []
        assert report_count == 3
        assert house["parse_status"] == "complete"
        assert db.source_reports.reconcile(
            "house-generation", "house_pdf", "house"
        ) == {
            "found": 2,
            "parsed": 1,
            "no_txs": 1,
            "paper_only": 0,
            "unavailable": 0,
            "failed": 0,
        }
        assert db.source_reports.reconcile(
            "house-generation", "gemini_ocr", "house"
        ) == {
            "found": 1,
            "parsed": 1,
            "no_txs": 0,
            "paper_only": 0,
            "unavailable": 0,
            "failed": 0,
        }
    finally:
        db.close()


def _seed_semantically_stale_generation_database(tmp_path):
    from analyzer.database import Database

    db = Database(tmp_path / "canonical-semantic.duckdb")
    generations = [
        ("g1", "artifact-g1", "2026-07-01 00:00:00", None),
        ("g2", "artifact-g2", "2026-07-02 00:00:00", date(2026, 1, 1)),
    ]
    for generation, artifact_sha, promoted_at, notification_date in generations:
        db.conn.execute(
            """
            INSERT INTO house_archive_generations (
                archive_year, generation_id, metadata_sha256,
                metadata_count, ptr_count, parse_status, promoted_at
            ) VALUES (2026, ?, 'metadata', 1, 1, 'complete', ?)
            """,
            [generation, promoted_at],
        )
        db.conn.execute(
            """
            INSERT INTO house_generation_metadata (
                archive_year, generation_id, doc_id, first_name, last_name,
                filing_date, filing_type, fetched_at
            ) VALUES (
                2026, ?, 'semantic-doc', 'Jane', 'Doe',
                '2026-01-03', 'P', '2026-01-04'
            )
            """,
            [generation],
        )
        db.conn.execute(
            """
            INSERT INTO house_pdf_artifacts (
                archive_year, doc_id, generation_id, artifact_sha256
            ) VALUES (2026, 'semantic-doc', ?, ?)
            """,
            [generation, artifact_sha],
        )
        db.upsert_transactions(
            pd.DataFrame(
                [
                    {
                        "doc_id": "semantic-doc",
                        "member": "Jane Doe",
                        "ticker": "OLD" if generation == "g1" else "NEW",
                        "transaction_date": date(2026, 1, 2),
                        "disclosure_date": date(2026, 1, 3),
                        "notification_date": notification_date,
                        "transaction_type": "Purchase",
                        "chamber": "house",
                        "source_record_id": "semantic-doc",
                        "source_row_id": f"{generation}:r1",
                        "official_filing_date": date(2026, 1, 3),
                        "ingestion_generation": generation,
                        "artifact_sha256": artifact_sha,
                    }
                ]
            ),
            source="house_pdf",
        )
        db.upsert_parse_run(
            doc_id="semantic-doc",
            year=2026,
            parser_version="v4-deterministic",
            status="success",
            engines_attempted="pdfplumber",
            raw_row_count=1,
            transaction_count=1,
            artifact_sha256=artifact_sha,
            ingestion_generation=generation,
        )
    return db


def test_canonical_view_check_uses_latest_semantically_accepted_generation(tmp_path):
    from scripts import rebuild_staged

    db = _seed_semantically_stale_generation_database(tmp_path)
    try:
        assert rebuild_staged._latest_accepted_house_generations(db) == {2026: "g1"}
        assert rebuild_staged._canonical_view_diff_counts(db) == (0, 0)

        db.conn.execute(
            """
            CREATE OR REPLACE TEMP VIEW canonical_transactions AS
            SELECT * FROM transactions
            """
        )
        assert rebuild_staged._canonical_view_diff_counts(db) == (1, 0)

        db.conn.execute(
            """
            CREATE OR REPLACE TEMP VIEW canonical_transactions AS
            SELECT * FROM transactions WHERE FALSE
            """
        )
        assert rebuild_staged._canonical_view_diff_counts(db) == (0, 1)
    finally:
        db.close()
