"""Concurrency regression tests for CLI Gemini OCR paths.

Both CLI OCR paths must use the isolated-subprocess pattern of the
standalone ``run_gemini_ocr_for_year`` entrypoint: the parent DuckDB handle
is released before OCR opens the file, so neither the in-process
ConnectionException (read_only second connect, parse path) nor the
child-process lock conflict (refresh path) can fire. No network: OCR is
mocked; temp files only.
"""

import os
import subprocess
import sys
import tempfile
import time
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
from analyzer.persistence.database import Database
from analyzer.exceptions import StepResult

_REAL_POPEN = subprocess.Popen


def _pid_is_running(pid: int) -> bool:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except FileNotFoundError:
        return False
    return stat.rsplit(") ", 1)[-1][:1] != "Z"


def _sleeping_tree_script() -> str:
    return (
        "import os,subprocess,sys,time\n"
        "grandchild=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'])\n"
        "with open(sys.argv[1]+'.tmp','w') as pids:\n"
        "    pids.write(f'{os.getpid()} {grandchild.pid}')\n"
        "os.replace(sys.argv[1]+'.tmp',sys.argv[1])\n"
        "time.sleep(60)\n"
    )


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
    def test_parent_database_close_failure_propagates(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            app_ctx = _make_app_ctx(temp_dir)
            db = app_ctx.transaction_source.db
            real_close = db.close
            with (
                patch.object(db, "close", side_effect=RuntimeError("close failed")),
                self.assertRaisesRegex(RuntimeError, "close failed"),
            ):
                _release_parent_db_for_ocr(app_ctx)
            real_close()

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
            patch("analyzer.cli.subprocess.Popen") as popen,
            patch("analyzer.cli.Path"),
        ):
            popen.return_value = MagicMock(
                returncode=0,
                communicate=MagicMock(
                    return_value=('{"schema":"ocr-year-summary-v1","inserted":7}\n', "")
                ),
            )
            inserted, error = _run_gemini_ocr_year_subprocess("data", 2026)
        self.assertIsNone(error)
        self.assertEqual(inserted, 7)

    @unittest.skipUnless(
        os.name == "posix" and Path("/proc").exists(), "POSIX process groups required"
    )
    def test_timeout_kills_and_reaps_owned_process_group(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            pid_file = Path(temp_dir) / "pids"
            script = _sleeping_tree_script()
            spawned = []

            def spawn_safe_child(_command, **kwargs):
                self.assertTrue(kwargs["start_new_session"])
                child = _REAL_POPEN(
                    [sys.executable, "-c", script, str(pid_file)], **kwargs
                )
                spawned.append(child)
                return child

            with patch("analyzer.cli.subprocess.Popen", side_effect=spawn_safe_child):
                result = _run_gemini_ocr_year_subprocess("data", 2026, timeout=0.2)

            self.assertEqual(result, (0, "timeout"))
            self.assertEqual(len(spawned), 1)
            self.assertIsNotNone(spawned[0].poll())
            direct_pid, descendant_pid = map(int, pid_file.read_text().split())
            self.assertEqual(spawned[0].pid, direct_pid)
            deadline = time.monotonic() + 3
            while _pid_is_running(descendant_pid) and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertFalse(_pid_is_running(descendant_pid))

    @unittest.skipUnless(
        os.name == "posix" and Path("/proc").exists(), "POSIX process groups required"
    )
    def test_exception_kills_and_reaps_owned_process_group(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            pid_file = Path(temp_dir) / "pids"
            spawned = []

            def spawn_safe_child(_command, **kwargs):
                child = _REAL_POPEN(
                    [sys.executable, "-c", _sleeping_tree_script(), str(pid_file)],
                    **kwargs,
                )
                communicate = child.communicate
                failed = False

                def fail_once(*args, **communicate_kwargs):
                    nonlocal failed
                    if not failed:
                        failed = True
                        deadline = time.monotonic() + 3
                        while not pid_file.exists() and time.monotonic() < deadline:
                            time.sleep(0.01)
                        self.assertTrue(pid_file.exists())
                        raise OSError("simulated communicate failure")
                    return communicate(*args, **communicate_kwargs)

                child.communicate = fail_once
                spawned.append(child)
                return child

            with (
                patch("analyzer.cli.subprocess.Popen", side_effect=spawn_safe_child),
                self.assertRaisesRegex(OSError, "simulated communicate failure"),
            ):
                _run_gemini_ocr_year_subprocess("data", 2026)

            self.assertIsNotNone(spawned[0].poll())
            direct_pid, descendant_pid = map(int, pid_file.read_text().split())
            self.assertEqual(spawned[0].pid, direct_pid)
            deadline = time.monotonic() + 3
            while _pid_is_running(descendant_pid) and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertFalse(_pid_is_running(descendant_pid))

    @unittest.skipUnless(
        os.name == "posix" and Path("/proc").exists(), "POSIX process groups required"
    )
    def test_parent_sigterm_cleans_owned_process_group(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            pid_file = Path(temp_dir) / "pids"
            worker = r"""
import os, signal, subprocess, sys, threading, time
from analyzer.cli import _run_gemini_ocr_year_subprocess
real_popen = subprocess.Popen
pid_file = sys.argv[1]
spawned = []
script = %r
def spawn_safe_child(command, **kwargs):
    assert command[1:3] == ['-m', 'analyzer.ocr.year_cli']
    assert kwargs['start_new_session']
    child = real_popen([sys.executable, '-c', script, pid_file], **kwargs)
    spawned.append(child)
    return child
subprocess.Popen = spawn_safe_child
previous = signal.getsignal(signal.SIGTERM)
def interrupt_parent():
    deadline = time.monotonic() + 3
    while not os.path.exists(pid_file) and time.monotonic() < deadline:
        time.sleep(0.01)
    os.kill(os.getpid(), signal.SIGTERM)
sender = threading.Thread(target=interrupt_parent)
sender.start()
try:
    _run_gemini_ocr_year_subprocess('data', 2026)
except SystemExit as exc:
    assert exc.code == 128 + signal.SIGTERM
else:
    raise AssertionError('SIGTERM did not cancel the helper')
sender.join(timeout=3)
assert not sender.is_alive()
assert signal.getsignal(signal.SIGTERM) == previous
assert spawned and spawned[0].poll() is not None
""" % _sleeping_tree_script()
            worker_proc = _REAL_POPEN(
                [sys.executable, "-c", worker, str(pid_file)],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            try:
                stdout, stderr = worker_proc.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                worker_proc.kill()
                worker_proc.communicate()
                self.fail("SIGTERM cleanup worker did not finish")
            self.assertEqual(worker_proc.returncode, 0, stderr or stdout)
            direct_pid, descendant_pid = map(int, pid_file.read_text().split())
            self.assertFalse(_pid_is_running(direct_pid))
            deadline = time.monotonic() + 3
            while _pid_is_running(descendant_pid) and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertFalse(_pid_is_running(descendant_pid))

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
