import csv
import json
from pathlib import Path

import duckdb

from scripts.reconcile_untickered import reconcile


def test_exact_keys_and_proposal_manifest_do_not_write_database(tmp_path: Path):
    db_path = tmp_path / "canonical.duckdb"
    connection = duckdb.connect(str(db_path))
    connection.execute(
        """CREATE TABLE canonical_transactions (
            doc_id VARCHAR, member VARCHAR, transaction_date DATE,
            amount_raw VARCHAR, ticker VARCHAR
        )"""
    )
    connection.executemany(
        "INSERT INTO canonical_transactions VALUES (?, ?, ?, ?, NULL)",
        [
            ("9115822", "Ro Khanna", "2020-09-10", "$1-15k"),
            ("date-mismatch", "Ro Khanna", "2020-09-10", "$1-15k"),
            ("band-mismatch", "Ro Khanna", "2020-09-10", "$15-50k"),
        ],
    )
    connection.close()

    input_path = tmp_path / "third_party.csv"
    with input_path.open("w", newline="", encoding="utf-8") as source:
        writer = csv.DictWriter(
            source,
            fieldnames=[
                "member",
                "transaction_date",
                "ticker",
                "amount_band",
                "source_doc_id",
            ],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "member": "Ro Khanna",
                    "transaction_date": "2020-09-10",
                    "ticker": "ZTS",
                    "amount_band": "$1-15k",
                    "source_doc_id": "9115822",
                },
                {
                    "member": "Ro Khanna",
                    "transaction_date": "2020-09-11",
                    "ticker": "BADDATE",
                    "amount_band": "$1-15k",
                    "source_doc_id": "date-mismatch",
                },
                {
                    "member": "Ro Khanna",
                    "transaction_date": "2020-09-10",
                    "ticker": "BADBAND",
                    "amount_band": "$1-15k",
                    "source_doc_id": "band-mismatch",
                },
            ]
        )

    report_path = tmp_path / "report.json"
    manifest_path = tmp_path / "manifest.json"
    report, manifest = reconcile(db_path, input_path, report_path, manifest_path)

    assert report["matched_count"] == 1
    assert report["unmatched_count"] == 2
    assert report["unused_third_party_row_count"] == 2
    assert {row["reason"] for row in report["unmatched"]} == {"no_exact_match"}
    assert manifest["proposed_fill_count"] == 1
    assert manifest["records"] == [
        {
            "doc_id": "9115822",
            "member": "Ro Khanna",
            "transaction_date": "2020-09-10",
            "amount_band": "$1-15k",
            "ticker": "ZTS",
            "source_doc_id": "9115822",
            "ticker_origin": "reconciled_third_party",
        }
    ]
    assert manifest["reconciliation_only"] is True
    assert "must not enter canonical data silently" in manifest["policy_note"]
    assert json.loads(report_path.read_text(encoding="utf-8")) == report
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == manifest

    connection = duckdb.connect(str(db_path), read_only=True)
    try:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM canonical_transactions WHERE ticker IS NULL"
            ).fetchone()[0]
            == 3
        )
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM canonical_transactions WHERE ticker IS NOT NULL"
            ).fetchone()[0]
            == 0
        )
    finally:
        connection.close()


def test_json_input_uses_same_exact_match_contract(tmp_path: Path):
    db_path = tmp_path / "canonical.duckdb"
    connection = duckdb.connect(str(db_path))
    connection.execute(
        "CREATE TABLE canonical_transactions AS SELECT 'doc' doc_id, 'Member' member, "
        "DATE '2024-01-01' transaction_date, '$1-15k' amount_raw, NULL::VARCHAR ticker"
    )
    connection.close()
    input_path = tmp_path / "trades.json"
    input_path.write_text(
        json.dumps(
            [
                {
                    "member": "Member",
                    "transaction_date": "2024-01-01",
                    "ticker": "ABC",
                    "amount_band": "$1-15k",
                    "source_doc_id": "doc",
                }
            ]
        ),
        encoding="utf-8",
    )

    report, manifest = reconcile(
        db_path, input_path, tmp_path / "report.json", tmp_path / "manifest.json"
    )

    assert (report["matched_count"], report["unmatched_count"]) == (1, 0)
    assert manifest["records"][0]["ticker_origin"] == "reconciled_third_party"


def test_tickered_canonical_collision_suppresses_untickered_proposal(tmp_path: Path):
    db_path = tmp_path / "canonical.duckdb"
    connection = duckdb.connect(str(db_path))
    connection.execute(
        """CREATE TABLE canonical_transactions (
            doc_id VARCHAR, member VARCHAR, transaction_date DATE,
            amount_raw VARCHAR, ticker VARCHAR
        )"""
    )
    connection.executemany(
        "INSERT INTO canonical_transactions VALUES (?, ?, ?, ?, ?)",
        [
            ("doc", "Member", "2024-01-01", "$25k-$40k", "AAPL"),
            ("doc", "Member", "2024-01-01", "$25k-$40k", None),
        ],
    )
    connection.close()
    input_path = tmp_path / "trades.json"
    input_path.write_text(
        json.dumps(
            [
                {
                    "member": "Member",
                    "transaction_date": "2024-01-01",
                    "ticker": "AAPL",
                    "amount_band": "$25k-$40k",
                    "source_doc_id": "doc",
                }
            ]
        ),
        encoding="utf-8",
    )

    report, manifest = reconcile(
        db_path, input_path, tmp_path / "report.json", tmp_path / "manifest.json"
    )

    assert report["canonical_row_count"] == 1
    assert report["matched_count"] == 0
    assert report["unmatched"][0]["reason"] == "ambiguous_canonical_key"
    assert manifest["records"] == []
