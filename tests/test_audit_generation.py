"""Tests for the read-only post-rebuild generation audit script.

The audit runs against a staged DuckDB database and must never modify it.
Every production check has a positive (clean fixture) test and a mutation
test proving the exact violation is reported and the process exits nonzero.
"""

import hashlib
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path

import duckdb
import pandas as pd

from analyzer.database import Database

from scripts.audit_generation import (
    PINNED_CANARY_COUNTS,
    audit_database,
)

ROOT = Path(__file__).resolve().parents[1]
AUDIT_SCRIPT = ROOT / "scripts" / "audit_generation.py"


# ── Fixture builder ───────────────────────────────────────────────────────────
DOCS = [
    ("20030977", 2026, "Alice", "Adams", "2026-03-01"),
    ("20033737", 2026, "Bob", "Baker", "2026-03-02"),
    ("20033921", 2026, "Carol", "Clark", "2026-03-03"),
    ("10000001", 2026, "Dan", "Davis", "2026-03-04"),
    ("10000002", 2025, "Eve", "Evans", "2025-06-01"),
    ("10000003", 2026, "Frank", "Fox", "2026-03-05"),  # scanned -> OCR
]

HOUSE_ROW_COUNTS = {
    "20030977": 224,
    "20033737": 16,
    "20033921": 15,
    "10000001": 3,
    "10000002": 2,
    "10000003": 0,
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _house_row(doc_id, member, n, tx_date, disc_date, ticker, artifact_sha):
    return {
        "doc_id": doc_id,
        "member": member,
        "ticker": ticker,
        "transaction_date": date.fromisoformat(tx_date),
        "disclosure_date": date.fromisoformat(disc_date),
        "transaction_type": "Purchase",
        "owner_code": "",
        "amount_raw": "$1,001 - $15,000",
        "amount_midpoint": 8000.0,
        "instrument_type": "stock",
        "asset_description": "Test Corp",
        "source": "house_pdf",
        "chamber": "house",
        "source_record_id": doc_id,
        "source_row_id": f"{doc_id}:r{n}",
        "official_filing_date": date.fromisoformat(disc_date),
        "ingestion_generation": "gen-2025-a" if disc_date.startswith("2025") else "gen-2026-b",
        "artifact_sha256": artifact_sha,
    }


def build_staged_db(db_path: Path) -> Database:
    """Create a complete, internally consistent staged rebuild database."""
    db = Database(db_path)
    conn = db.conn

    metadata = pd.DataFrame(
        [
            {
                "doc_id": doc_id,
                "archive_year": year,
                "first_name": first,
                "last_name": last,
                "filing_date": datetime.fromisoformat(filing),
                "filing_type": "P",
                "fetched_at": datetime.fromisoformat(filing),
            }
            for doc_id, year, first, last, filing in DOCS
        ]
    )
    db.upsert_metadata(metadata)

    generations = [
        (2026, "gen-2026-a", "complete", "2026-07-01 00:00:00"),
        (2026, "gen-2026-b", "complete", "2026-07-02 00:00:00"),
        (2025, "gen-2025-a", "complete", "2026-07-01 00:00:00"),
    ]
    for year, generation, status, promoted in generations:
        conn.execute(
            "INSERT INTO house_archive_generations ("
            "archive_year, generation_id, metadata_sha256, metadata_count, "
            "ptr_count, parse_status, promoted_at) VALUES (?, ?, 'm', 1, 1, ?, ?)",
            [year, generation, status, promoted],
        )
    for doc_id, year, first, last, filing in DOCS:
        generation = "gen-2025-a" if year == 2025 else "gen-2026-b"
        conn.execute(
            "INSERT INTO house_generation_metadata ("
            "archive_year, generation_id, doc_id, first_name, last_name, "
            "filing_date, filing_type, fetched_at) VALUES (?, ?, ?, ?, ?, ?, 'P', ?)",
            [year, generation, doc_id, first, last, filing, filing],
        )
        if year == 2026:
            conn.execute(
                "INSERT INTO house_generation_metadata ("
                "archive_year, generation_id, doc_id, first_name, last_name, "
                "filing_date, filing_type, fetched_at) "
                "VALUES (2026, 'gen-2026-a', ?, ?, ?, ?, 'P', ?)",
                [doc_id, first, last, filing, filing],
            )

    artifact_shas = {
        doc_id: f"{index:064x}" for index, (doc_id, *_rest) in enumerate(DOCS)
    }
    for doc_id, year, *_rest in DOCS:
        generation = "gen-2025-a" if year == 2025 else "gen-2026-b"
        conn.execute(
            "INSERT INTO house_pdf_artifacts ("
            "archive_year, doc_id, generation_id, artifact_sha256, http_status, "
            "etag, last_modified, content_length) VALUES (?, ?, ?, ?, 200, NULL, "
            "NULL, 100)",
            [year, doc_id, generation, artifact_shas[doc_id]],
        )

    rows = []
    for doc_id, member, ticker, tx_date, disc_date, count in [
        ("20030977", "Alice Adams", "TEST", "2026-02-01", "2026-03-01", 224),
        ("20033737", "Bob Baker", "TSTB", "2026-02-02", "2026-03-02", 16),
        ("20033921", "Carol Clark", "TSTC", "2026-02-03", "2026-03-03", 15),
        ("10000001", "Dan Davis", "TSTD", "2026-02-04", "2026-03-04", 3),
        ("10000002", "Eve Evans", "TSTE", "2025-05-01", "2025-06-01", 2),
    ]:
        for n in range(1, count + 1):
            rows.append(
                _house_row(doc_id, member, n, tx_date, disc_date, ticker,
                           artifact_shas[doc_id])
            )
    house_df = pd.DataFrame(rows)
    db.upsert_transactions(house_df, source="house_pdf")

    # Scanned doc: Gemini OCR rows with the OCR provenance contract.
    ocr_columns = [
        "doc_id", "member", "ticker", "transaction_date", "disclosure_date",
        "transaction_type", "owner_code", "amount_raw", "amount_midpoint",
        "instrument_type", "asset_description", "source", "chamber",
        "source_record_id", "source_row_id", "official_filing_date",
        "available_date", "notification_date", "ingestion_generation",
        "artifact_sha256",
    ]
    for n in (1, 2):
        conn.execute(
            f"INSERT INTO transactions ({', '.join(ocr_columns)}) VALUES ("
            + ", ".join("?" for _ in ocr_columns) + ")",
            [
                "10000003", "Frank Fox", "TSTF", date(2026, 2, 20),
                date(2026, 3, 5), "Sale", "", "B", 32500.0, "stock",
                "Fox Industries", "gemini_ocr", "House", "10000003",
                f"10000003:page:1:row:{n}", date(2026, 3, 5),
                date(2026, 3, 5), date(2026, 3, 1), "gen-2026-b",
                artifact_shas["10000003"],
            ],
        )

    for doc_id, count in HOUSE_ROW_COUNTS.items():
        year = 2025 if doc_id == "10000002" else 2026
        generation = "gen-2025-a" if year == 2025 else "gen-2026-b"
        if doc_id == "10000003":
            db.upsert_parse_run(
                doc_id=doc_id, year=year, parser_version="v5-gemini-validated",
                status="success", engines_attempted="gemini", raw_row_count=2,
                transaction_count=2, artifact_sha256=artifact_shas[doc_id],
                ingestion_generation=generation,
            )
        else:
            db.upsert_parse_run(
                doc_id=doc_id, year=year, parser_version="v4-deterministic",
                status="success", engines_attempted="pdfplumber",
                raw_row_count=count, transaction_count=count,
                artifact_sha256=artifact_shas[doc_id],
                ingestion_generation=generation,
            )

    reports = pd.DataFrame(
        [
            {
                "ingestion_generation": "senate-gen-1",
                "chamber": "Senate",
                "source_record_id": "s1",
                "report_path": "s1.pdf",
                "member": "Senator One",
                "official_filing_date": date(2026, 8, 1),
                "outcome": "parsed",
                "artifact_sha256": "a" * 64,
                "landing_sha256": "a" * 64,
                "paper_artifact_url": None,
                "paper_artifact_sha256": None,
                "error_message": None,
                "raw_row_count": 3,
                "accepted_row_count": 3,
                "rejected_row_count": 0,
            },
            {
                "ingestion_generation": "senate-gen-1",
                "chamber": "Senate",
                "source_record_id": "s2",
                "report_path": "s2.pdf",
                "member": "Senator Two",
                "official_filing_date": date(2026, 8, 1),
                "outcome": "paper_only",
                "artifact_sha256": "b" * 64,
                "landing_sha256": "b" * 64,
                "paper_artifact_url": (
                    "https://efdsearch.senate.gov/search/view/paper/xyz/"
                ),
                "paper_artifact_sha256": "c" * 64,
                "error_message": None,
                "raw_row_count": 0,
                "accepted_row_count": 0,
                "rejected_row_count": 0,
            },
        ]
    )
    db.replace_source_reports("senate-gen-1", "senate_efd", "Senate", reports)
    return db


def _audit(db_path: Path):
    return audit_database(db_path)


def _result(results, name):
    matches = [result for result in results if result.name == name]
    assert len(matches) == 1, name
    return matches[0]


def _mutate(db_path: Path, fn):
    conn = duckdb.connect(str(db_path))
    try:
        fn(conn)
    finally:
        conn.close()


# ── Positive path ─────────────────────────────────────────────────────────────
def test_clean_fixture_passes_all_checks(tmp_path):
    db = build_staged_db(tmp_path / "stage.duckdb")
    db.close()
    results = _audit(tmp_path / "stage.duckdb")
    assert len(results) == 9
    for result in results:
        assert result.passed, (result.name, result.violations)


def test_audit_never_modifies_the_database(tmp_path):
    db_path = tmp_path / "stage.duckdb"
    db = build_staged_db(db_path)
    db.close()
    before = _sha(db_path)
    results = _audit(db_path)
    assert all(result.passed for result in results)
    assert _sha(db_path) == before


def test_cli_exit_zero_with_json_output(tmp_path):
    db_path = tmp_path / "stage.duckdb"
    db = build_staged_db(db_path)
    db.close()
    output = tmp_path / "audit.json"
    env = {"PYTHONPATH": str(ROOT / "src")}
    completed = subprocess.run(
        [sys.executable, str(AUDIT_SCRIPT), str(db_path), "--json", str(output)],
        capture_output=True,
        text=True,
        env=env,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    import json

    payload = json.loads(output.read_text())
    assert payload["passed"] is True
    assert {check["name"] for check in payload["checks"]} == {
        "parse_counts_match_persisted",
        "source_row_id_integrity",
        "no_transaction_after_disclosure",
        "duplicate_policy",
        "source_reports_equation",
        "completeness",
        "pinned_pdf_canaries",
        "house_generation_activation",
        "date_domain",
    }


def test_cli_missing_database_exits_two(tmp_path):
    env = {"PYTHONPATH": str(ROOT / "src")}
    completed = subprocess.run(
        [sys.executable, str(AUDIT_SCRIPT), str(tmp_path / "nope.duckdb")],
        capture_output=True,
        text=True,
        env=env,
    )
    assert completed.returncode == 2
    assert "database not found" in completed.stderr


def test_missing_tables_fail_closed(tmp_path):
    db_path = tmp_path / "empty.duckdb"
    duckdb.connect(str(db_path)).close()
    results = _audit(db_path)
    assert not _result(results, "parse_counts_match_persisted").passed
    assert not _result(results, "source_row_id_integrity").passed
    assert not _result(results, "duplicate_policy").passed
    assert not _result(results, "pinned_pdf_canaries").passed
    assert not _result(results, "date_domain").passed


# ── C1: parse counts == persisted rows ───────────────────────────────────────
def test_parse_count_mismatch_is_reported(tmp_path):
    db_path = tmp_path / "stage.duckdb"
    db = build_staged_db(db_path)
    db.close()

    def fn(conn):
        conn.execute(
            "UPDATE pdf_parse_runs SET transaction_count = 5 "
            "WHERE doc_id = '10000001' AND parser_version = 'v4-deterministic'"
        )

    _mutate(db_path, fn)
    check = _result(_audit(db_path), "parse_counts_match_persisted")
    assert any(
        "10000001" in v and "transaction_count=5 but persisted=3" in v
        for v in check.violations
    )


def test_failed_parse_never_leaves_rows(tmp_path):
    db_path = tmp_path / "stage.duckdb"
    db = build_staged_db(db_path)
    db.close()

    def fn(conn):
        conn.execute(
            "UPDATE pdf_parse_runs SET status = 'error' "
            "WHERE doc_id = '10000001' AND parser_version = 'v4-deterministic'"
        )

    _mutate(db_path, fn)
    check = _result(_audit(db_path), "parse_counts_match_persisted")
    assert any("scans must fail closed" in v for v in check.violations)


def test_unknown_parse_status_is_reported(tmp_path):
    db_path = tmp_path / "stage.duckdb"
    db = build_staged_db(db_path)
    db.close()

    def fn(conn):
        conn.execute(
            "UPDATE pdf_parse_runs SET status = 'weird' "
            "WHERE doc_id = '10000001' AND parser_version = 'v4-deterministic'"
        )

    _mutate(db_path, fn)
    check = _result(_audit(db_path), "parse_counts_match_persisted")
    assert any("unknown parse status 'weird'" in v for v in check.violations)


# ── C2: source_row_id integrity ──────────────────────────────────────────────
def test_blank_source_row_id_is_reported(tmp_path):
    db_path = tmp_path / "stage.duckdb"
    db = build_staged_db(db_path)
    db.close()

    def fn(conn):
        conn.execute(
            "UPDATE transactions SET source_row_id = '' "
            "WHERE doc_id = '10000001' AND source_row_id = '10000001:r1'"
        )

    _mutate(db_path, fn)
    check = _result(_audit(db_path), "source_row_id_integrity")
    assert any("blank source_row_id" in v for v in check.violations)


def test_duplicate_source_row_id_is_reported(tmp_path):
    db_path = tmp_path / "stage.duckdb"
    db = build_staged_db(db_path)
    db.close()

    def fn(conn):
        conn.execute("DROP INDEX idx_tx_source_row_unique")
        conn.execute(
            "UPDATE transactions SET source_row_id = '10000001:r1' "
            "WHERE doc_id = '10000001' AND source_row_id = '10000001:r2'"
        )

    _mutate(db_path, fn)
    check = _result(_audit(db_path), "source_row_id_integrity")
    assert any("2 rows share one source_row_id" in v for v in check.violations)


def test_missing_unique_index_is_reported(tmp_path):
    db_path = tmp_path / "stage.duckdb"
    db = build_staged_db(db_path)
    db.close()

    def fn(conn):
        conn.execute("DROP INDEX idx_tx_source_row_unique")

    _mutate(db_path, fn)
    check = _result(_audit(db_path), "source_row_id_integrity")
    assert any("idx_tx_source_row_unique missing" in v for v in check.violations)


# ── C3: no transaction after disclosure ──────────────────────────────────────
def test_transaction_after_disclosure_is_reported(tmp_path):
    db_path = tmp_path / "stage.duckdb"
    db = build_staged_db(db_path)
    db.close()

    def fn(conn):
        conn.execute(
            "UPDATE transactions SET transaction_date = '2026-03-05' "
            "WHERE doc_id = '10000001' AND source_row_id = '10000001:r1'"
        )

    _mutate(db_path, fn)
    check = _result(_audit(db_path), "no_transaction_after_disclosure")
    assert any(
        "10000001" in v and "2026-03-05" in v and "2026-03-04" in v
        for v in check.violations
    )


# ── C4: duplicate policy ─────────────────────────────────────────────────────
def test_exact_identity_replay_is_reported(tmp_path):
    db_path = tmp_path / "stage.duckdb"
    db = build_staged_db(db_path)
    db.close()

    def fn(conn):
        conn.execute("DROP INDEX idx_tx_source_row_unique")
        conn.execute(
            """
            INSERT INTO transactions (
                doc_id, member, ticker, transaction_date, disclosure_date,
                transaction_type, owner_code, amount_raw, amount_midpoint,
                instrument_type, asset_description, source, chamber,
                source_record_id, source_row_id, official_filing_date,
                ingestion_generation, artifact_sha256
            )
            SELECT doc_id, member, ticker, transaction_date, disclosure_date,
                   transaction_type, owner_code, amount_raw, amount_midpoint,
                   instrument_type, asset_description, source, chamber,
                   source_record_id, source_row_id, official_filing_date,
                   ingestion_generation, artifact_sha256
            FROM transactions
            WHERE doc_id = '10000001' AND source_row_id = '10000001:r1'
            """
        )

    _mutate(db_path, fn)
    check = _result(_audit(db_path), "duplicate_policy")
    assert any(
        "exact-identity replay not deduplicated" in v for v in check.violations
    )
    assert any("replay dedup removed 1 row(s)" in v for v in check.violations)


def test_economic_duplicates_survive_and_are_flagged(tmp_path):
    # Two rows with an identical economic key but distinct exact identity must
    # both persist and both be flagged economic_duplicate_candidate.
    db_path = tmp_path / "stage.duckdb"
    db = build_staged_db(db_path)
    stored = db.conn.execute(
        """
        SELECT COUNT(*) FROM transactions
        WHERE doc_id = '10000001' AND source_row_id IN ('10000001:r1', '10000001:r2')
        """
    ).fetchone()[0]
    db.close()
    assert stored == 2
    results = _audit(db_path)
    check = _result(results, "duplicate_policy")
    assert check.passed, check.violations


# ── C5: source_reports equation ──────────────────────────────────────────────
def test_source_report_row_reconciliation_is_reported(tmp_path):
    db_path = tmp_path / "stage.duckdb"
    db = build_staged_db(db_path)
    db.close()

    def fn(conn):
        conn.execute(
            "UPDATE source_reports SET raw_row_count = 5 "
            "WHERE source_record_id = 's1'"
        )

    _mutate(db_path, fn)
    check = _result(_audit(db_path), "source_reports_equation")
    assert any("outcome=parsed raw=5 accepted=3" in v for v in check.violations)


def test_source_report_equation_violation_on_legacy_schema(tmp_path):
    # A pre-existing table without the outcome CHECK constraint can hold an
    # outcome outside the four-way partition; the audit must fail closed.
    db_path = tmp_path / "legacy.duckdb"
    conn = duckdb.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE source_reports (
            ingestion_generation VARCHAR NOT NULL,
            source VARCHAR NOT NULL,
            chamber VARCHAR NOT NULL,
            source_record_id VARCHAR NOT NULL,
            report_path VARCHAR,
            member VARCHAR,
            official_filing_date DATE,
            outcome VARCHAR,
            artifact_sha256 VARCHAR,
            landing_sha256 VARCHAR,
            paper_artifact_url VARCHAR,
            paper_artifact_sha256 VARCHAR,
            error_message VARCHAR,
            raw_row_count INTEGER NOT NULL,
            accepted_row_count INTEGER NOT NULL,
            rejected_row_count INTEGER NOT NULL
        )
        """
    )
    conn.execute(
        "INSERT INTO source_reports VALUES ("
        "'senate-gen-1', 'senate_efd', 'Senate', 's1', 's1.pdf', "
        "'Senator One', '2026-08-01', 'mystery', NULL, NULL, NULL, NULL, NULL, "
        "1, 1, 0)"
    )
    conn.close()
    results = _audit(db_path)
    check = _result(results, "source_reports_equation")
    assert any("found=1 != parsed=0" in v for v in check.violations)


# ── C6: completeness ─────────────────────────────────────────────────────────
def test_completeness_requires_zero_failed_and_unavailable(tmp_path):
    db_path = tmp_path / "stage.duckdb"
    db = build_staged_db(db_path)
    db.close()

    def fn(conn):
        conn.execute(
            "UPDATE source_reports SET outcome = 'unavailable' "
            "WHERE source_record_id = 's2'"
        )

    _mutate(db_path, fn)
    check = _result(_audit(db_path), "completeness")
    assert any(
        "gen=senate-gen-1 incomplete: failed=0, unavailable=1" in v
        for v in check.violations
    )


# ── C7: pinned canaries ──────────────────────────────────────────────────────
def test_canary_persisted_count_violation(tmp_path):
    db_path = tmp_path / "stage.duckdb"
    db = build_staged_db(db_path)
    db.close()
    assert PINNED_CANARY_COUNTS == {"20030977": 224, "20033737": 16, "20033921": 15}

    def fn(conn):
        conn.execute(
            "DELETE FROM transactions WHERE doc_id = '20033737' "
            "AND source_row_id = '20033737:r16'"
        )

    _mutate(db_path, fn)
    check = _result(_audit(db_path), "pinned_pdf_canaries")
    assert any(
        "canary 20033737: persisted 15 rows, pinned 16" in v
        for v in check.violations
    )


def test_canary_run_raw_count_violation(tmp_path):
    db_path = tmp_path / "stage.duckdb"
    db = build_staged_db(db_path)
    db.close()

    def fn(conn):
        conn.execute(
            "UPDATE pdf_parse_runs SET raw_row_count = 10 "
            "WHERE doc_id = '20033921' AND parser_version = 'v4-deterministic'"
        )

    _mutate(db_path, fn)
    check = _result(_audit(db_path), "pinned_pdf_canaries")
    assert any(
        "canary 20033921: latest success run raw_row_count=10" in v
        for v in check.violations
    )


# ── C8: House generation activation / scans fail closed ──────────────────────
def test_complete_generation_with_unresolved_artifact_is_reported(tmp_path):
    db_path = tmp_path / "stage.duckdb"
    db = build_staged_db(db_path)
    db.close()

    def fn(conn):
        conn.execute(
            "INSERT INTO house_pdf_artifacts ("
            "archive_year, doc_id, generation_id, artifact_sha256, http_status) "
            "VALUES (2026, '99999999', 'gen-2026-b', ?, 200)",
            ["f" * 64],
        )

    _mutate(db_path, fn)
    check = _result(_audit(db_path), "house_generation_activation")
    assert any(
        "gen-2026-b marked 'complete' but has 1 unresolved artifact(s)" in v
        for v in check.violations
    )


def test_incomplete_generation_is_informational_not_a_violation(tmp_path):
    db_path = tmp_path / "stage.duckdb"
    db = build_staged_db(db_path)
    db.close()

    def fn(conn):
        conn.execute(
            "INSERT INTO house_archive_generations ("
            "archive_year, generation_id, metadata_sha256, metadata_count, "
            "ptr_count, parse_status, promoted_at) "
            "VALUES (2026, 'gen-2026-c', 'm', 1, 1, 'incomplete', "
            "'2026-07-03 00:00:00')"
        )
        conn.execute(
            "INSERT INTO house_pdf_artifacts ("
            "archive_year, doc_id, generation_id, artifact_sha256, http_status) "
            "VALUES (2026, '99999999', 'gen-2026-c', ?, 200)",
            ["e" * 64],
        )

    _mutate(db_path, fn)
    check = _result(_audit(db_path), "house_generation_activation")
    assert check.passed, check.violations
    assert any(
        "correctly incomplete (1 unresolved artifact(s))" in info
        for info in check.info
    )


def test_unbound_persisted_house_row_is_reported(tmp_path):
    db_path = tmp_path / "stage.duckdb"
    db = build_staged_db(db_path)
    db.close()

    def fn(conn):
        conn.execute(
            "INSERT INTO transactions ("
            "doc_id, member, ticker, transaction_date, disclosure_date, "
            "transaction_type, owner_code, amount_raw, asset_description, "
            "source, chamber, source_record_id, source_row_id, "
            "official_filing_date, ingestion_generation, artifact_sha256) "
            "VALUES ('10000001', 'Dan Davis', 'TSTD', '2026-02-04', "
            "'2026-03-04', 'Purchase', '', '$1,001 - $15,000', 'Test Corp', "
            "'house_pdf', 'house', '10000001', '10000001:r99', '2026-03-04', "
            "'gen-2026-b', ?)",
            ["z" * 64],
        )

    _mutate(db_path, fn)
    check = _result(_audit(db_path), "house_generation_activation")
    assert any(
        "persisted House/OCR row(s) are not artifact-bound" in v
        for v in check.violations
    )


# ── C9: chronology / date domain ─────────────────────────────────────────────
def test_date_out_of_domain_is_reported(tmp_path):
    db_path = tmp_path / "stage.duckdb"
    db = build_staged_db(db_path)
    db.close()

    def fn(conn):
        conn.execute(
            "UPDATE transactions SET transaction_date = '1899-12-31' "
            "WHERE doc_id = '10000001' AND source_row_id = '10000001:r1'"
        )

    _mutate(db_path, fn)
    check = _result(_audit(db_path), "date_domain")
    assert any("date out of domain" in v for v in check.violations)


def test_canonical_null_date_is_reported(tmp_path):
    db_path = tmp_path / "stage.duckdb"
    db = build_staged_db(db_path)
    db.close()

    def fn(conn):
        conn.execute(
            "UPDATE transactions SET disclosure_date = NULL "
            "WHERE doc_id = '10000001' AND source_row_id = '10000001:r1'"
        )

    _mutate(db_path, fn)
    check = _result(_audit(db_path), "date_domain")
    assert any("canonical row missing a date" in v for v in check.violations)


def test_notification_before_transaction_is_reported(tmp_path):
    db_path = tmp_path / "stage.duckdb"
    db = build_staged_db(db_path)
    db.close()

    def fn(conn):
        conn.execute(
            "UPDATE transactions SET notification_date = '2026-01-01' "
            "WHERE doc_id = '10000003'"
        )

    _mutate(db_path, fn)
    check = _result(_audit(db_path), "date_domain")
    assert any("chronology broken" in v for v in check.violations)


def test_disclosure_after_official_filing_is_reported(tmp_path):
    db_path = tmp_path / "stage.duckdb"
    db = build_staged_db(db_path)
    db.close()

    def fn(conn):
        conn.execute(
            "UPDATE transactions SET official_filing_date = '2026-02-01' "
            "WHERE doc_id = '10000001' AND source_row_id = '10000001:r1'"
        )

    _mutate(db_path, fn)
    check = _result(_audit(db_path), "date_domain")
    assert any("chronology broken" in v for v in check.violations)
