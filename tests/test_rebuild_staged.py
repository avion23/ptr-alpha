"""Regression tests for the staged rebuild parser boundary."""

from types import SimpleNamespace
from typing import cast

import pandas as pd


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
        def fetchall(self):
            return []

    class FakeConnection:
        def execute(self, *_args):
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

        def map(self, worker, paths):
            return [worker(path) for path in paths]

    fallback_rows = [
        {
            "doc_id": "fallback",
            "disclosure_date": pd.Timestamp("2026-06-26"),
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
            [fallback_path, failed_path],
            pd.DataFrame({"DocID": ["fallback", "failed"]}),
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
