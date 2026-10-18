"""Regression tests for finding 2 of the 2026-10-18 OCR pipeline review.

Canonical admission must be bound to the ACCEPTED parse run's rowset:
exactly the rows of an accepted (source, artifact, ingestion_generation,
parse_run) identity. An unaccepted OCR run's rows for an accepted artifact
must never enter canonical_transactions -- live or as-of -- regardless of
their engines_attempted family, while a legitimately accepted dual run
(v6-dual-validated) must still be admitted. Superseded/rejected evidence
stays in transactions/transaction_versions/parse-run records.
"""

from datetime import date

import pandas as pd
import pytest

from analyzer.database import Database


def seed_accepted_deterministic(db, *, doc="20034202", ticker="JPM"):
    """One promoted generation whose deterministic rowset is accepted."""
    db.conn.execute(
        "INSERT INTO house_archive_generations "
        "(archive_year, generation_id, parse_status, promoted_at) "
        "VALUES (2026, 'g1', 'incomplete', '2026-08-01')"
    )
    db.conn.execute(
        "INSERT INTO house_generation_metadata "
        "(archive_year, generation_id, doc_id, filing_type) "
        "VALUES (2026, 'g1', ?, 'P')",
        [doc],
    )
    db.conn.execute(
        "INSERT INTO house_pdf_artifacts "
        "(archive_year, generation_id, doc_id, artifact_sha256) "
        "VALUES (2026, 'g1', ?, 'sha')",
        [doc],
    )
    db.upsert_transactions(
        pd.DataFrame(
            [
                {
                    "doc_id": doc,
                    "member": "Jane Doe",
                    "ticker": ticker,
                    "transaction_date": date(2026, 3, 6),
                    "disclosure_date": date(2026, 3, 18),
                    "transaction_type": "Purchase",
                    "source_record_id": doc,
                    "source_row_id": "pdftotext:l26",
                    "chamber": "house",
                    "official_filing_date": date(2026, 3, 18),
                    "ingestion_generation": "g1",
                    "artifact_sha256": "sha",
                }
            ]
        ),
        source="house_pdf",
    )
    db.conn.execute(
        "UPDATE transactions SET created_at = '2026-08-02' WHERE source = 'house_pdf'"
    )
    db.parse_runs.upsert(
        doc_id=doc,
        year=2026,
        parser_version="pdftotext",
        status="success",
        engines_attempted="pdftotext",
        raw_row_count=1,
        transaction_count=1,
        artifact_sha256="sha",
        ingestion_generation="g1",
    )
    db.conn.execute(
        "UPDATE pdf_parse_runs SET parsed_at = '2026-08-04', rowset_created_at = NULL "
        "WHERE parser_version = 'pdftotext'"
    )


def insert_rogue_ocr_row(db, *, doc="20034202", ticker="MSFT"):
    """The wrong-security OCR row from the reproduced leak scenario."""
    db.upsert_transactions(
        pd.DataFrame(
            [
                {
                    "doc_id": doc,
                    "member": "Jane Doe",
                    "ticker": ticker,
                    "transaction_date": date(2026, 3, 6),
                    "disclosure_date": date(2026, 3, 18),
                    "transaction_type": "Purchase",
                    "source_record_id": doc,
                    "source_row_id": "ocr-row",
                    "chamber": "house",
                    "official_filing_date": date(2026, 3, 18),
                    "ingestion_generation": "g1",
                    "artifact_sha256": "sha",
                }
            ]
        ),
        source="gemini_ocr",
    )
    db.conn.execute(
        "UPDATE transactions SET created_at = '2026-08-02' WHERE source = 'gemini_ocr'"
    )


@pytest.mark.parametrize("record_unaccepted_run", [True, False])
def test_unaccepted_ocr_rows_never_enter_canonical(
    tmp_path, record_unaccepted_run
):
    """Astra repro: accepted deterministic run + rogue OCR row for the same
    artifact (with an unaccepted OCR run recorded, or none) must yield ONLY
    the accepted deterministic row, live and historically."""
    db = Database(tmp_path / "db.duckdb")
    try:
        seed_accepted_deterministic(db)
        insert_rogue_ocr_row(db)
        if record_unaccepted_run:
            # The OCR attempt happened but was never accepted: its rows must
            # not ride in on the deterministic run's artifact acceptance.
            db.parse_runs.upsert(
                doc_id="20034202",
                year=2026,
                parser_version="gemini-v1",
                status="error",
                engines_attempted="gemini/v1",
                raw_row_count=1,
                transaction_count=1,
                error_message="boom",
                artifact_sha256="sha",
                ingestion_generation="g1",
            )
            db.conn.execute(
                "UPDATE pdf_parse_runs SET parsed_at = '2026-08-05' "
                "WHERE parser_version = 'gemini-v1'"
            )
        assert db.get_transactions(2026).ticker.tolist() == ["JPM"]
        with db.canonical_as_of(date(2026, 9, 21)):
            assert db.get_transactions(2026).ticker.tolist() == ["JPM"]
        with db.canonical_as_of(date(2026, 10, 15)):
            assert db.get_transactions(2026).ticker.tolist() == ["JPM"]
        # Evidence is preserved for audit surfaces, only excluded from canonical.
        assert db.conn.execute(
            "SELECT COUNT(*) FROM transactions WHERE source = 'gemini_ocr'"
        ).fetchone()[0] == 1
    finally:
        db.close()


def test_unaccepted_ocr_run_with_count_mismatch_excluded(tmp_path):
    """Even a *successful* OCR run does not admit its rows unless its counted
    rowset matches the live family rowset: the artifact acceptance rule is
    applied per source family, not per artifact."""
    db = Database(tmp_path / "db.duckdb")
    try:
        seed_accepted_deterministic(db)
        insert_rogue_ocr_row(db, ticker="MSFT")
        db.upsert_transactions(
            pd.DataFrame(
                [
                    {
                        "doc_id": "20034202",
                        "member": "Jane Doe",
                        "ticker": "GOOG",
                        "transaction_date": date(2026, 3, 6),
                        "disclosure_date": date(2026, 3, 18),
                        "transaction_type": "Sale",
                        "source_record_id": "20034202",
                        "source_row_id": "ocr-row-2",
                        "chamber": "house",
                        "official_filing_date": date(2026, 3, 18),
                        "ingestion_generation": "g1",
                        "artifact_sha256": "sha",
                    }
                ]
            ),
            source="gemini_ocr",
        )
        db.conn.execute(
            "UPDATE transactions SET created_at = '2026-08-02' "
            "WHERE source = 'gemini_ocr' AND source_row_id = 'ocr-row-2'"
        )
        # The run claims one row; two OCR rows are persisted. Not acceptable.
        db.parse_runs.upsert(
            doc_id="20034202",
            year=2026,
            parser_version="gemini-v1",
            status="success",
            engines_attempted="gemini/v1",
            raw_row_count=1,
            transaction_count=1,
            artifact_sha256="sha",
            ingestion_generation="g1",
        )
        db.conn.execute(
            "UPDATE pdf_parse_runs SET parsed_at = '2026-08-05' "
            "WHERE parser_version = 'gemini-v1'"
        )
        assert db.get_transactions(2026).ticker.tolist() == ["JPM"]
        with db.canonical_as_of(date(2026, 10, 15)):
            assert db.get_transactions(2026).ticker.tolist() == ["JPM"]
    finally:
        db.close()


def test_dual_validated_run_rows_are_admitted(tmp_path):
    """A legitimately accepted dual-agreed run (v6-dual-validated,
    engines_attempted gemini+visionb) MUST appear in canonical views, live and
    as-of, once its own rowset is accepted."""
    db = Database(tmp_path / "db.duckdb")
    try:
        seed_accepted_deterministic(db, ticker=None)
        db.upsert_transactions(
            pd.DataFrame(
                [
                    {
                        "doc_id": "20034202",
                        "member": "Jane Doe",
                        "ticker": "JPM",
                        "transaction_date": date(2026, 3, 6),
                        "disclosure_date": date(2026, 3, 18),
                        "transaction_type": "Purchase",
                        "source_record_id": "20034202",
                        "source_row_id": "dual-row",
                        "chamber": "house",
                        "official_filing_date": date(2026, 3, 18),
                        "ingestion_generation": "g1",
                        "artifact_sha256": "sha",
                    }
                ]
            ),
            source="gemini_ocr",
        )
        db.conn.execute(
            "UPDATE transactions SET created_at = '2026-08-02' "
            "WHERE source = 'gemini_ocr'"
        )
        db.parse_runs.upsert(
            doc_id="20034202",
            year=2026,
            parser_version="v6-dual-validated",
            status="success",
            engines_attempted="gemini+visionb",
            raw_row_count=1,
            transaction_count=1,
            artifact_sha256="sha",
            ingestion_generation="g1",
        )
        db.conn.execute(
            "UPDATE pdf_parse_runs SET parsed_at = '2026-10-01' "
            "WHERE parser_version = 'v6-dual-validated'"
        )
        def both_rowsets_visible(frame):
            house = frame[frame.source == "house_pdf"]
            ocr = frame[frame.source == "gemini_ocr"]
            assert len(house) == 1 and house.ticker.isna().all()
            assert len(ocr) == 1 and ocr.ticker.iloc[0] == "JPM"

        both_rowsets_visible(db.get_transactions(2026))
        with db.canonical_as_of(date(2026, 9, 21)):
            # Before the dual run's acceptance only the deterministic rowset.
            frame = db.get_transactions(2026)
            assert frame.source.tolist() == ["house_pdf"]
        with db.canonical_as_of(date(2026, 10, 2)):
            both_rowsets_visible(db.get_transactions(2026))
    finally:
        db.close()


def test_stray_house_rows_without_own_family_run_excluded(tmp_path):
    """The binding is family-symmetric: house_pdf rows with no accepted
    house_pdf run are excluded even though an OCR rowset is accepted."""
    db = Database(tmp_path / "db.duckdb")
    try:
        seed_accepted_deterministic(db)
        # Demote the deterministic run to a failed attempt; keep its rows.
        db.conn.execute("DELETE FROM pdf_parse_runs WHERE parser_version = 'pdftotext'")
        db.parse_runs.upsert(
            doc_id="20034202",
            year=2026,
            parser_version="pdftotext",
            status="error",
            engines_attempted="pdftotext",
            raw_row_count=1,
            transaction_count=1,
            error_message="boom",
            artifact_sha256="sha",
            ingestion_generation="g1",
        )
        db.conn.execute(
            "UPDATE pdf_parse_runs SET parsed_at = '2026-08-04' "
            "WHERE parser_version = 'pdftotext'"
        )
        # The OCR rowset is fully accepted...
        db.upsert_transactions(
            pd.DataFrame(
                [
                    {
                        "doc_id": "20034202",
                        "member": "Jane Doe",
                        "ticker": "JPM",
                        "transaction_date": date(2026, 3, 6),
                        "disclosure_date": date(2026, 3, 18),
                        "transaction_type": "Purchase",
                        "source_record_id": "20034202",
                        "source_row_id": "ocr-row",
                        "chamber": "house",
                        "official_filing_date": date(2026, 3, 18),
                        "ingestion_generation": "g1",
                        "artifact_sha256": "sha",
                    }
                ]
            ),
            source="gemini_ocr",
        )
        db.conn.execute(
            "UPDATE transactions SET created_at = '2026-08-02' WHERE source = 'gemini_ocr'"
        )
        db.parse_runs.upsert(
            doc_id="20034202",
            year=2026,
            parser_version="gemini-v1",
            status="success",
            engines_attempted="gemini/v1",
            raw_row_count=1,
            transaction_count=1,
            artifact_sha256="sha",
            ingestion_generation="g1",
        )
        db.conn.execute(
            "UPDATE pdf_parse_runs SET parsed_at = '2026-08-04' "
            "WHERE parser_version = 'gemini-v1'"
        )
        # ...so only its rows are canonical; the stray deterministic row is not.
        assert db.get_transactions(2026).ticker.tolist() == ["JPM"]
        assert db.get_transactions(2026).source.tolist() == ["gemini_ocr"]
    finally:
        db.close()
