"""metadata archive_year must stay queryable across replace cycles.

Regression: DuckDB 1.5.5's ART index on metadata(archive_year) silently
stops matching rows the table still holds after repeated bulk delete/upsert
cycles. `WHERE archive_year = ?` then returns fewer rows than `GROUP BY`
sees, and MetadataRepository.replace_archive reads its stale id set through
the same index, so the corruption compounds. The fix is to keep no index on
archive_year at all; these tests pin both the absence of the index and the
agreement between the two access paths.
"""

import pandas as pd
import pytest

from analyzer.database import Database
from analyzer.metadata_repository import MetadataRepository


def _rows(year: int, doc_ids: list[str]) -> pd.DataFrame:
    # Column names match the lowercase schema replace_archive consumes
    # (download._download_and_upsert_metadata renames before calling).
    return pd.DataFrame(
        {
            "doc_id": doc_ids,
            "first_name": ["F"] * len(doc_ids),
            "last_name": ["L"] * len(doc_ids),
            "filing_date": [pd.Timestamp(year, 1, 2)] * len(doc_ids),
            "filing_type": ["P"] * len(doc_ids),
            "fetched_at": [pd.Timestamp(2026, 1, 1)] * len(doc_ids),
        }
    )


def _index_scan_count(repo: MetadataRepository, year: int) -> int:
    return repo.conn.execute(
        "SELECT COUNT(*) FROM metadata WHERE archive_year = ?", [year]
    ).fetchone()[0]


def _table_scan_count(repo: MetadataRepository, year: int) -> int:
    return repo.conn.execute(
        "SELECT COUNT(*) FROM metadata GROUP BY archive_year HAVING archive_year = ?",
        [year],
    ).fetchone()[0]


def test_metadata_has_no_archive_year_index(tmp_path):
    db = Database(tmp_path / "m.duckdb", read_only=False)
    try:
        names = {
            row[0]
            for row in db.conn.execute(
                "SELECT index_name FROM duckdb_indexes() WHERE table_name = 'metadata'"
            ).fetchall()
        }
        assert "idx_metadata_archive_year" not in names
    finally:
        db.close()


def test_replace_archive_keeps_access_paths_agreeing(tmp_path):
    db = Database(tmp_path / "m.duckdb", read_only=False)
    try:
        repo = MetadataRepository(db.conn)
        # Churn the archive through several generations, including a shrink
        # (the shape that triggered the delete-side index failure).
        generations = [
            [f"doc-{year}-{i}" for i in range(400)] for year in range(2024, 2027)
        ]
        for year, doc_ids in enumerate(generations):
            repo.replace_archive(2024 + year, _rows(2024 + year, doc_ids))
        # Second pass over the same archives, each with fewer docs.
        for year, doc_ids in enumerate(generations):
            repo.replace_archive(
                2024 + year, _rows(2024 + year, doc_ids[: len(doc_ids) // 2])
            )

        for year in range(2024, 2027):
            expected = len(generations[year - 2024]) // 2
            assert _index_scan_count(repo, year) == expected
            assert _table_scan_count(repo, year) == expected
            assert repo.get_by_archive(year).shape[0] == expected
    finally:
        db.close()


def test_replace_archive_drops_docs_absent_from_archive(tmp_path):
    db = Database(tmp_path / "m.duckdb", read_only=False)
    try:
        repo = MetadataRepository(db.conn)
        repo.replace_archive(2026, _rows(2026, ["a", "b", "c"]))
        repo.replace_archive(2026, _rows(2026, ["b"]))
        assert _index_scan_count(repo, 2026) == 1
        assert _table_scan_count(repo, 2026) == 1
        assert repo.get_by_archive(2026)["DocID"].tolist() == ["b"]
    finally:
        db.close()


def test_writable_reopen_after_replace_is_consistent(tmp_path):
    """A close/reopen between cycles must not resurrect stale index state."""
    path = tmp_path / "m.duckdb"
    db = Database(path, read_only=False)
    repo = MetadataRepository(db.conn)
    repo.replace_archive(2026, _rows(2026, [f"doc-{i}" for i in range(300)]))
    db.close()

    db = Database(path, read_only=False)
    try:
        repo = MetadataRepository(db.conn)
        assert _index_scan_count(repo, 2026) == 300
        repo.replace_archive(2026, _rows(2026, ["only"]))
        assert _index_scan_count(repo, 2026) == 1
        assert _table_scan_count(repo, 2026) == 1
    finally:
        db.close()
