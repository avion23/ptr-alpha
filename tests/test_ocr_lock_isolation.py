"""Concurrency regression tests for CLI Gemini OCR paths.

Both CLI OCR paths must use the isolated-subprocess pattern of the
standalone ``run_gemini_ocr_for_year`` entrypoint: the parent DuckDB handle
is released before OCR opens the file, so neither the in-process
ConnectionException (read_only second connect, parse path) nor the
child-process lock conflict (refresh path) can fire. No network: OCR is
mocked; temp files only.
"""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import duckdb
from typer.testing import CliRunner

from analyzer.cli import (
    _reacquire_parent_db_after_ocr,
    _release_parent_db_for_ocr,
    _run_gemini_ocr_year_subprocess,
    app,
)
from analyzer.database import Database
from analyzer.exceptions import StepResult


def _make_app_ctx(data_dir: str) -> MagicMock:
    from analyzer.settings import Settings

    settings = Settings()
    settings.data.data_dir = data_dir
    shared_db = Database(Path(data_dir) / "congress.duckdb", read_only=False)
    ctx = MagicMock()
    ctx.settings = settings
    ctx.transaction_source.db = shared_db
    ctx.transaction_source.fetch_and_cache_pdfs.return_value = MagicMock(
        archive_year=2026,
        metadata_count=1,
        ptr_count=1,
        valid_pdf_count=1,
        downloaded_count=0,
        skipped_count=1,
        orphan_pdf_count=0,
        removed_doc_count=0,
        quarantined_pdf_count=0,
        generation_id="generation-2026",
        generation_status="acquired",
    )
    ctx.price_source.db = shared_db
    return ctx


class TestOcrLockIsolation(unittest.TestCase):
    def test_release_lets_both_crash_configs_open(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            app_ctx = _make_app_ctx(temp_dir)
            db_path = str(Path(temp_dir) / "congress.duckdb")
            released = _release_parent_db_for_ocr(app_ctx)
            self.assertIsNotNone(released)
            try:
                # Parse crash config: in-process read_only second connect.
                read_only = duckdb.connect(db_path, read_only=True)
                read_only.execute("SELECT 1").fetchall()
                read_only.close()
                # Refresh crash config: fresh read-write handle (child pattern).
                writer = duckdb.connect(db_path)
                writer.execute("SELECT 1").fetchall()
                writer.close()
            finally:
                _reacquire_parent_db_after_ocr(app_ctx, released)
            rows = app_ctx.transaction_source.db.conn.execute("SELECT 1").fetchall()
            self.assertEqual(rows, [(1,)])

    def test_held_parent_reproduces_both_crashes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            app_ctx = _make_app_ctx(temp_dir)
            db_path = str(Path(temp_dir) / "congress.duckdb")
            try:
                with self.assertRaises(duckdb.ConnectionException):
                    second = duckdb.connect(db_path, read_only=True)
                    try:
                        second.execute("SELECT 1").fetchall()
                    finally:
                        second.close()
                code = (
                    "import duckdb;"
                    f"c=duckdb.connect({db_path!r});"
                    "c.execute('SELECT 1').fetchall()"
                )
                proc = subprocess.run(
                    [sys.executable, "-c", code],
                    text=True,
                    capture_output=True,
                )
                self.assertNotEqual(proc.returncode, 0)
                self.assertIn("Conflicting lock", proc.stderr)
            finally:
                _release_parent_db_for_ocr(app_ctx)

    def test_subprocess_helper_reports_insert_count(self):
        with (
            patch("analyzer.cli.subprocess.run") as run,
            patch("analyzer.cli.Path"),
        ):
            run.return_value = MagicMock(
                returncode=0, stdout="Total inserted: 7\n", stderr=""
            )
            inserted, error = _run_gemini_ocr_year_subprocess("data", 2026)
        self.assertIsNone(error)
        self.assertEqual(inserted, 7)

    def test_parse_gemini_ocr_releases_parent_and_exits_zero(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as temp_dir:
            app_ctx = _make_app_ctx(temp_dir)
            db_path = str(Path(temp_dir) / "congress.duckdb")
            lock_seen: dict = {}

            def fake_ocr(data_dir, year, timeout=7200):
                # Mocked OCR insert: both crash configs must open cleanly.
                read_only = duckdb.connect(db_path, read_only=True)
                try:
                    read_only.execute("SELECT 1").fetchall()
                finally:
                    read_only.close()
                writer = duckdb.connect(db_path)
                try:
                    writer.execute("SELECT 1").fetchall()
                finally:
                    writer.close()
                lock_seen["ok"] = True
                return 1, None

            with (
                patch("analyzer.cli.get_context", return_value=app_ctx),
                patch(
                    "analyzer.cli.run_parse_pipeline",
                    return_value=StepResult(success=True),
                ),
                patch(
                    "analyzer.cli._run_gemini_ocr_year_subprocess",
                    side_effect=fake_ocr,
                ),
                patch(
                    "analyzer.cli._activate_house_generation",
                    side_effect=lambda *a: lock_seen.setdefault("activated", True),
                ),
            ):
                result = runner.invoke(app, ["parse", "--year", "2026", "--gemini-ocr"])
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertTrue(lock_seen.get("ok"))
            app_ctx.transaction_source.db.close()

    def test_refresh_gemini_ocr_releases_parent_and_exits_zero(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as temp_dir:
            app_ctx = _make_app_ctx(temp_dir)
            db_path = str(Path(temp_dir) / "congress.duckdb")

            def fake_ocr(data_dir, year, timeout=7200):
                read_only = duckdb.connect(db_path, read_only=True)
                try:
                    read_only.execute("SELECT 1").fetchall()
                finally:
                    read_only.close()
                return 2, None

            with (
                patch("analyzer.cli.get_context", return_value=app_ctx),
                patch(
                    "analyzer.cli.run_parse_pipeline",
                    return_value=StepResult(success=True),
                ),
                patch(
                    "analyzer.cli._run_gemini_ocr_year_subprocess",
                    side_effect=fake_ocr,
                ),
                patch.object(
                    Database,
                    "get_latest_house_generation",
                    return_value="generation-2026",
                ),
                patch.object(
                    Database, "get_unresolved_house_doc_ids", return_value=[]
                ),
                patch.object(
                    Database, "mark_house_generation_parse_complete", return_value=None
                ),
            ):
                result = runner.invoke(
                    app, ["refresh", "--year", "2026", "--gemini-ocr"]
                )
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertNotIn("ConnectionException", result.output)
            self.assertNotIn("Conflicting lock", result.output)


if __name__ == "__main__":
    unittest.main()
