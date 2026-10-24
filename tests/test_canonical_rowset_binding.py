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


def insert_gemini_interpretation(
    db,
    *,
    doc="20034202",
    generation="g1",
    artifact="sha",
    status="success",
    row_count=1,
    transaction_count=None,
    parsed_at="2026-08-05",
):
    if row_count:
        db.upsert_transactions(
            pd.DataFrame(
                [
                    {
                        "doc_id": doc,
                        "member": "Jane Doe",
                        "ticker": f"G{index}",
                        "transaction_date": date(2026, 3, 6),
                        "disclosure_date": date(2026, 3, 18),
                        "transaction_type": "Purchase",
                        "source_record_id": doc,
                        "source_row_id": f"gemini-row-{index}",
                        "chamber": "house",
                        "official_filing_date": date(2026, 3, 18),
                        "ingestion_generation": generation,
                        "artifact_sha256": artifact,
                    }
                    for index in range(1, row_count + 1)
                ]
            ),
            source="gemini_ocr",
        )
        db.conn.execute(
            "UPDATE transactions SET created_at = '2026-08-02' "
            "WHERE doc_id = ? AND source = 'gemini_ocr' "
            "AND ingestion_generation = ? AND artifact_sha256 IS NOT DISTINCT FROM ?",
            [doc, generation, artifact],
        )
    db.parse_runs.upsert(
        doc_id=doc,
        year=2026,
        parser_version="gemini-v1",
        status=status,
        engines_attempted="gemini/v1",
        raw_row_count=row_count,
        transaction_count=(row_count if transaction_count is None else transaction_count),
        artifact_sha256=artifact,
        ingestion_generation=generation,
    )
    db.conn.execute(
        "UPDATE pdf_parse_runs SET parsed_at = ?, rowset_created_at = ? "
        "WHERE doc_id = ? AND parser_version = 'gemini-v1' "
        "AND ingestion_generation = ? AND artifact_sha256 IS NOT DISTINCT FROM ?",
        [parsed_at, "2026-08-02" if row_count else parsed_at, doc, generation, artifact],
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


def test_dual_validated_gemini_rows_prefer_whole_house_document(tmp_path):
    """An eligible Gemini interpretation wins as a complete rowset for the
    same accepted artifact; raw House evidence remains available for audit."""
    db = Database(tmp_path / "db.duckdb")
    try:
        seed_accepted_deterministic(db, ticker=None)
        db.upsert_transactions(pd.DataFrame(
            [
                {
                    "doc_id": "20034202",
                    "member": "Jane Doe",
                    "ticker": "JPM",
                    "transaction_date": date(2026, 3, 6),
                    "disclosure_date": date(2026, 3, 18),
                    "transaction_type": "Purchase",
                    "source_record_id": "20034202",
                    "source_row_id": f"dual-row-{index}",
                    "chamber": "house",
                    "official_filing_date": date(2026, 3, 18),
                    "ingestion_generation": "g1",
                    "artifact_sha256": "sha",
                }
                for index in (1, 2)
            ]
        ), source="gemini_ocr")
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
            raw_row_count=2,
            transaction_count=2,
            artifact_sha256="sha",
            ingestion_generation="g1",
        )
        db.conn.execute(
            "UPDATE pdf_parse_runs SET parsed_at = '2026-10-01' "
            "WHERE parser_version = 'v6-dual-validated'"
        )
        def gemini_rowset_selected(frame):
            assert frame.source.tolist() == ["gemini_ocr", "gemini_ocr"]
            assert frame.ticker.tolist() == ["JPM", "JPM"]
            assert sorted(frame.source_row_id.tolist()) == [
                "dual-row-1",
                "dual-row-2",
            ]

        gemini_rowset_selected(db.get_transactions(2026))
        assert db.conn.execute(
            "SELECT source, COUNT(*) FROM transactions GROUP BY source ORDER BY source"
        ).fetchall() == [("gemini_ocr", 2), ("house_pdf", 1)]
        assert [row[1] for row in db.conn.execute(
            "PRAGMA table_info('canonical_transactions')"
        ).fetchall()] == [
            row[1] for row in db.conn.execute("PRAGMA table_info('transactions')").fetchall()
        ]
        with db.canonical_as_of(date(2026, 9, 21)):
            # Before Gemini acceptance, the accepted House interpretation wins.
            frame = db.get_transactions(2026)
            assert frame.source.tolist() == ["house_pdf"]
        with db.canonical_as_of(date(2026, 10, 2)):
            gemini_rowset_selected(db.get_transactions(2026))
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


@pytest.mark.parametrize("status", ["error", "invalid", "success"])
def test_nonaccepted_gemini_zero_row_attempt_does_not_shadow_house(tmp_path, status):
    db = Database(tmp_path / "db.duckdb")
    try:
        seed_accepted_deterministic(db)
        insert_gemini_interpretation(
            db, status=status, row_count=0, transaction_count=0
        )

        frame = db.get_transactions(2026)

        assert frame.source.tolist() == ["house_pdf"]
        assert db.conn.execute(
            "SELECT COUNT(*) FROM transactions WHERE source = 'gemini_ocr'"
        ).fetchone()[0] == 0
    finally:
        db.close()


def test_accepted_gemini_no_txs_does_not_shadow_house_on_same_artifact(tmp_path):
    db = Database(tmp_path / "db.duckdb")
    try:
        seed_accepted_deterministic(db)
        insert_gemini_interpretation(
            db, status="no_txs", row_count=0, transaction_count=0
        )
        assert db.conn.execute(
            "SELECT COUNT(*) FROM transactions WHERE source = 'house_pdf'"
        ).fetchone()[0] == 1

        with db.canonical_as_of(date(2026, 8, 4)):
            assert db.get_transactions(2026).source.tolist() == ["house_pdf"]
        with db.canonical_as_of(date(2026, 8, 5)):
            assert db.get_transactions(2026).source.tolist() == ["house_pdf"]
        assert db.get_transactions(2026).source.tolist() == ["house_pdf"]
        assert db.conn.execute(
            "SELECT COUNT(*) FROM transactions WHERE source = 'house_pdf'"
        ).fetchone()[0] == 1
    finally:
        db.close()


def test_gemini_no_txs_false_zero_does_not_shadow_house(tmp_path):
    db = Database(tmp_path / "db.duckdb")
    try:
        seed_accepted_deterministic(db)
        insert_gemini_interpretation(
            db, status="no_txs", row_count=1, transaction_count=0
        )

        frame = db.get_transactions(2026)

        assert frame.source.tolist() == ["house_pdf"]
        assert db.conn.execute(
            "SELECT COUNT(*) FROM transactions WHERE source = 'gemini_ocr'"
        ).fetchone()[0] == 1
    finally:
        db.close()


@pytest.mark.parametrize(
    ("generation", "artifact"), [("g1", "other-sha"), ("g2", "sha")]
)
def test_gemini_from_different_artifact_or_generation_cannot_shadow_house(
    tmp_path, generation, artifact
):
    db = Database(tmp_path / "db.duckdb")
    try:
        seed_accepted_deterministic(db)
        insert_gemini_interpretation(
            db,
            generation=generation,
            artifact=artifact,
            status="success",
            row_count=1,
        )

        frame = db.get_transactions(2026)

        assert frame.source.tolist() == ["house_pdf"]
        assert db.conn.execute(
            "SELECT COUNT(*) FROM transactions WHERE source = 'gemini_ocr'"
        ).fetchone()[0] == 1
    finally:
        db.close()


def test_legacy_null_artifact_rows_keep_legacy_admission_without_precedence(tmp_path):
    db = Database(tmp_path / "db.duckdb")
    try:
        rows = [
            {
                "doc_id": "legacy-doc",
                "member": "Jane Doe",
                "ticker": "JPM",
                "transaction_date": date(2026, 3, 6),
                "disclosure_date": date(2026, 3, 18),
                "transaction_type": "Purchase",
                "source_record_id": "legacy-doc",
                "source_row_id": "legacy-house",
                "chamber": "house",
                "official_filing_date": date(2026, 3, 18),
            },
            {
                "doc_id": "legacy-doc",
                "member": "Jane Doe",
                "ticker": "MSFT",
                "transaction_date": date(2026, 3, 6),
                "disclosure_date": date(2026, 3, 18),
                "transaction_type": "Purchase",
                "source_record_id": "legacy-doc",
                "source_row_id": "legacy-gemini",
                "chamber": "house",
                "official_filing_date": date(2026, 3, 18),
            },
        ]
        db.upsert_transactions(pd.DataFrame([rows[0]]), source="house_pdf")
        db.upsert_transactions(pd.DataFrame([rows[1]]), source="gemini_ocr")

        frame = db.get_transactions(2026)

        assert set(frame.source) == {"house_pdf", "gemini_ocr"}
        assert frame.artifact_sha256.isna().all()
        assert frame.ingestion_generation.isna().all()
    finally:
        db.close()


def test_orphan_fallback_gemini_cannot_shadow_house_without_accepted_artifact(
    tmp_path,
):
    db = Database(tmp_path / "db.duckdb")
    try:
        base = {
            "doc_id": "orphan-doc",
            "member": "Jane Doe",
            "transaction_date": date(2026, 3, 6),
            "disclosure_date": date(2026, 3, 18),
            "transaction_type": "Purchase",
            "source_record_id": "orphan-doc",
            "chamber": "house",
            "official_filing_date": date(2026, 3, 18),
            "ingestion_generation": "orphan-generation",
            "artifact_sha256": "orphan-sha",
        }
        house = {**base, "ticker": "JPM", "source_row_id": "house-row"}
        gemini = {**base, "ticker": "MSFT", "source_row_id": "gemini-row"}
        db.upsert_transactions(pd.DataFrame([house]), source="house_pdf")
        db.upsert_transactions(pd.DataFrame([gemini]), source="gemini_ocr")

        frame = db.get_transactions(2026)

        assert sorted(frame.source.tolist()) == ["gemini_ocr", "house_pdf"]
        assert set(frame.ticker) == {"JPM", "MSFT"}
        assert db.conn.execute(
            "SELECT COUNT(*) FROM house_pdf_artifacts"
        ).fetchone()[0] == 0
    finally:
        db.close()


def test_latest_gemini_zero_row_artifact_selection_does_not_fall_back_to_older_house(
    tmp_path,
):
    """A selected newer zero-row artifact does not revive an older House
    artifact; this is existing document selection, not negative-rowset priority."""
    db = Database(tmp_path / "db.duckdb")
    try:
        seed_accepted_deterministic(db)
        db.conn.execute(
            "INSERT INTO house_archive_generations "
            "(archive_year, generation_id, parse_status, promoted_at) "
            "VALUES (2026, 'g2', 'incomplete', '2026-08-05')"
        )
        db.conn.execute(
            "INSERT INTO house_generation_metadata "
            "(archive_year, generation_id, doc_id, filing_type) "
            "VALUES (2026, 'g2', '20034202', 'P')"
        )
        db.conn.execute(
            "INSERT INTO house_pdf_artifacts "
            "(archive_year, generation_id, doc_id, artifact_sha256) "
            "VALUES (2026, 'g2', '20034202', 'sha-g2')"
        )
        insert_gemini_interpretation(
            db,
            generation="g2",
            artifact="sha-g2",
            status="no_txs",
            row_count=0,
            transaction_count=0,
        )

        with db.canonical_as_of(date(2026, 8, 4)):
            assert db.get_transactions(2026).ticker.tolist() == ["JPM"]
        assert db.get_transactions(2026).empty
    finally:
        db.close()


def test_asof_house_run_cannot_authorize_same_count_replacement(tmp_path):
    db = Database(tmp_path / "db.duckdb")
    try:
        seed_accepted_deterministic(db)
        replacement = pd.DataFrame(
            [
                {
                    "doc_id": "20034202",
                    "member": "Jane Doe",
                    "ticker": "MSFT",
                    "transaction_date": date(2026, 3, 6),
                    "disclosure_date": date(2026, 3, 18),
                    "transaction_type": "Purchase",
                    "source_record_id": "20034202",
                    "source_row_id": "pdftotext:l26",
                    "chamber": "house",
                    "official_filing_date": date(2026, 3, 18),
                    "ingestion_generation": "g1",
                    "artifact_sha256": "sha",
                }
            ]
        )
        db.replace_transactions_for_docs(
            replacement,
            source="house_pdf",
            attempted_doc_ids=["20034202"],
            replacement_doc_ids=["20034202"],
            ingestion_generation="g1",
        )
        db.conn.execute(
            "UPDATE transactions SET created_at = '2026-08-05' "
            "WHERE source = 'house_pdf'"
        )
        assert db.conn.execute(
            "SELECT COUNT(*) FROM transactions WHERE source = 'house_pdf'"
        ).fetchone()[0] == 1

        with db.canonical_as_of(date(2026, 8, 4)):
            assert db.get_transactions(2026).ticker.tolist() == ["JPM"]
        with db.canonical_as_of(date(2026, 8, 6)):
            assert db.get_transactions(2026).empty
    finally:
        db.close()
