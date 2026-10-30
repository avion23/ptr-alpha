"""OCR pipeline observability: doctor staging/cron checks + freshness.

Review finding 12 (ocr-pipeline-astra-review-20261018.md): ``glm_stage.py
stage`` exits 0 even when individual documents error and ``backlog_ocr.sh``
returns only the doctor status, so an entirely broken OCR drain can look
successful. These tests pin the fail-loud monitors that close that gap:
staging disposition counts with retry/audit starvation, nightly drain
engagement, OCR staging freshness, and House ingestion freshness.

Every case runs in temporary directories with injected clocks; no network,
and ``data/congress.duckdb`` is never opened.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from contextlib import redirect_stdout
from datetime import date, datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import duckdb
import pytest

from analyzer.freshness import (
    FreshnessCheck,
    end_to_end_checks,
    house_ingestion_freshness,
    monitoring_checks,
    ocr_backlog_freshness,
)
from analyzer.persistence.price_repository import previous_nyse_session
from scripts import doctor as doctor_mod

REPO = Path(__file__).resolve().parents[1]

# Fixed clock: 2026-11-01 12:00 UTC (a Sunday; only used as an epoch).
NOW = datetime(2026, 11, 1, 12, 0, 0, tzinfo=timezone.utc).timestamp()
TODAY = date(2026, 11, 1)


def _stamp(age_days: float) -> str:
    return datetime.fromtimestamp(NOW - age_days * 86_400, tz=timezone.utc).isoformat(
        timespec="seconds"
    ).replace("+00:00", "Z")


def _write_record(
    root: Path,
    year: int,
    doc_id: str,
    *,
    status: str = "success",
    audit: str | None = None,
    age_days: float = 0.0,
    error: str | None = None,
    epoch: float = NOW,
    asset_review: bool = False,
    audit_severity: str | None = None,
) -> Path:
    stamp = datetime.fromtimestamp(
        epoch - age_days * 86_400, tz=timezone.utc
    ).isoformat(timespec="seconds").replace("+00:00", "Z")
    record = {
        "doc_id": doc_id,
        "year": year,
        "parse_status": status,
        "audit": audit,
        "parsed_at_utc": stamp,
    }
    if error is not None:
        record["error"] = error
    if asset_review:
        record["asset_review"] = True
    if audit_severity is not None:
        record["audit_severity"] = audit_severity
    path = root / str(year) / f"{doc_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def _write_run_dir(
    log_root: Path, day: str, *, log_text: str | None = "doctor exit=0\n", make_log: bool = True
) -> Path:
    run_dir = log_root / day
    run_dir.mkdir(parents=True, exist_ok=True)
    if make_log and log_text is not None:
        (run_dir / "run.log").write_text(log_text, encoding="utf-8")
    return run_dir


def _house_session(today: date) -> date:
    return previous_nyse_session(today - timedelta(days=1)).date()


def _house_conn(newest_disclosure) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    con.execute(
        "CREATE TABLE canonical_transactions (source VARCHAR, disclosure_date DATE)"
    )
    if newest_disclosure is not None:
        con.execute(
            "INSERT INTO canonical_transactions VALUES ('house_pdf', ?)",
            [newest_disclosure],
        )
    return con


class OcrBacklogFreshnessTest(TestCase):
    def test_missing_staging_root_is_unknown_not_green(self):
        check = ocr_backlog_freshness(Path(tempfile.mkdtemp()) / "absent", now=NOW)
        self.assertEqual("UNKNOWN", check.mark)
        self.assertFalse(check.ok)
        self.assertEqual("UNKNOWN", check.state)

    def test_none_staging_root_is_unknown(self):
        check = ocr_backlog_freshness(None, now=NOW)
        self.assertEqual("UNKNOWN", check.mark)

    def test_fresh_staged_record_passes(self):
        root = Path(tempfile.mkdtemp())
        record = root / "2017" / "1.json"
        record.parent.mkdir(parents=True)
        record.write_text("{}")
        past = NOW - 1 * 86_400
        os.utime(record, (past, past))
        check = ocr_backlog_freshness(root, now=NOW)
        self.assertTrue(check.ok, check.detail)
        self.assertIn("limit 3d", check.detail)

    def test_silent_staging_fails(self):
        root = Path(tempfile.mkdtemp())
        record = root / "2017" / "1.json"
        record.parent.mkdir(parents=True)
        record.write_text("{}")
        past = NOW - 4 * 86_400
        os.utime(record, (past, past))
        check = ocr_backlog_freshness(root, now=NOW)
        self.assertFalse(check.ok)
        self.assertIn("4d old", check.detail)

    def test_root_level_files_do_not_count_as_staged_records(self):
        # promoted.jsonl / audit-results.json live in the root, not a year
        # dir: audit activity must not masquerade as staging activity.
        root = Path(tempfile.mkdtemp())
        (root / "promoted.jsonl").write_text("{}\n")
        os.utime(root / "promoted.jsonl", (NOW, NOW))
        check = ocr_backlog_freshness(root, now=NOW)
        self.assertEqual("UNKNOWN", check.mark)


class HouseIngestionFreshnessTest(TestCase):
    def test_recent_disclosure_passes(self):
        session = _house_session(TODAY)
        check = house_ingestion_freshness(_house_conn(session), today=TODAY)
        self.assertTrue(check.ok, check.detail)

    def test_boundary_21d_passes_22d_fails(self):
        session = _house_session(TODAY)
        ok = house_ingestion_freshness(
            _house_conn(session - timedelta(days=21)), today=TODAY
        )
        stale = house_ingestion_freshness(
            _house_conn(session - timedelta(days=22)), today=TODAY
        )
        self.assertTrue(ok.ok, ok.detail)
        self.assertIn("limit 21d", ok.detail)
        self.assertFalse(stale.ok)
        self.assertIn("22d before", stale.detail)

    def test_gemini_ocr_rows_count_as_house_ingestion(self):
        session = _house_session(TODAY)
        con = _house_conn(None)
        con.execute(
            "INSERT INTO canonical_transactions VALUES ('gemini_ocr', ?)",
            [session - timedelta(days=2)],
        )
        self.assertTrue(house_ingestion_freshness(con, today=TODAY).ok)

    def test_no_house_rows_fails_with_fix_command(self):
        check = house_ingestion_freshness(_house_conn(None), today=TODAY)
        self.assertFalse(check.ok)
        self.assertIn("refresh --all-years", check.detail)


class OcrStagingCheckTest(TestCase):
    def test_missing_root_is_unknown(self):
        check = doctor_mod._ocr_staging_check(
            Path(tempfile.mkdtemp()) / "absent", now=NOW
        )
        self.assertEqual("UNKNOWN", check.mark)
        self.assertFalse(check.ok)

    def test_healthy_tree_passes_with_counts(self):
        root = Path(tempfile.mkdtemp())
        _write_record(root, 2017, "20001", status="success", audit="pass")
        _write_record(root, 2017, "20002", status="success", audit="fail", age_days=1)
        _write_record(root, 2017, "20003", status="no_txs", age_days=1)
        check = doctor_mod._ocr_staging_check(root, now=NOW)
        self.assertTrue(check.ok, check.detail)
        self.assertIn("2 success / 1 no_txs / 0 error", check.detail)
        self.assertIn(
            "audit pass/critical-fail/cosmetic-fail/unaudited 1/1/0/1", check.detail
        )
        self.assertIn("promoted-consumed-absent 0", check.detail)

    def test_retry_starvation_fails_and_names_the_artifact(self):
        root = Path(tempfile.mkdtemp())
        _write_record(
            root, 2016, "30001", status="error", age_days=8, error="boom"
        )
        check = doctor_mod._ocr_staging_check(root, now=NOW)
        self.assertFalse(check.ok)
        self.assertIn("retry starved", check.detail)
        self.assertIn(str(root / "2016" / "30001.json"), check.detail)

    def test_error_record_exactly_7d_old_is_not_yet_starved(self):
        root = Path(tempfile.mkdtemp())
        _write_record(root, 2016, "30001", status="error", age_days=7)
        self.assertTrue(doctor_mod._ocr_staging_check(root, now=NOW).ok)

    def test_audit_starvation_fails_and_names_the_sampler(self):
        root = Path(tempfile.mkdtemp())
        _write_record(root, 2015, "40001", status="success", audit=None, age_days=15)
        check = doctor_mod._ocr_staging_check(root, now=NOW)
        self.assertFalse(check.ok)
        self.assertIn("audit:null", check.detail)
        self.assertIn("audit_sample.py", check.detail)

    def test_success_record_exactly_14d_old_with_null_audit_passes(self):
        root = Path(tempfile.mkdtemp())
        _write_record(root, 2015, "40001", status="success", audit=None, age_days=14)
        self.assertTrue(doctor_mod._ocr_staging_check(root, now=NOW).ok)

    def test_audited_success_records_never_starve(self):
        root = Path(tempfile.mkdtemp())
        _write_record(root, 2015, "40001", status="success", audit="pass", age_days=40)
        _write_record(root, 2015, "40002", status="success", audit="fail", age_days=40)
        self.assertTrue(doctor_mod._ocr_staging_check(root, now=NOW).ok)

    def test_unreadable_record_fails_loudly(self):
        root = Path(tempfile.mkdtemp())
        bad = root / "2017" / "50001.json"
        bad.parent.mkdir(parents=True)
        bad.write_text("{not json", encoding="utf-8")
        check = doctor_mod._ocr_staging_check(root, now=NOW)
        self.assertFalse(check.ok)
        self.assertIn("unreadable staging record", check.detail)
        self.assertIn("50001.json", check.detail)

    def test_record_without_timestamp_counts_as_starved(self):
        root = Path(tempfile.mkdtemp())
        path = root / "2017" / "60001.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"parse_status": "error", "audit": None}))
        check = doctor_mod._ocr_staging_check(root, now=NOW)
        self.assertFalse(check.ok)
        self.assertIn("retry starved", check.detail)

    def test_promoted_consumed_absent_is_reported(self):
        root = Path(tempfile.mkdtemp())
        _write_record(root, 2017, "70001")  # still present (refused/consumed lag)
        (root / "promoted.jsonl").write_text(
            " \t\n"
            + json.dumps({"doc_id": "70002", "year": 2017})
            + "\n"
            + "  \n"
            + json.dumps({"doc_id": "70001", "year": 2017})
            + "\n",
            encoding="utf-8",
        )
        check = doctor_mod._ocr_staging_check(root, now=NOW)
        self.assertIn("promoted-consumed-absent 1", check.detail)

    def test_malformed_promoted_journal_is_unknown_not_zero(self):
        for bad_entry in ('{"doc_id":', "{}", "[]"):
            with self.subTest(bad_entry=bad_entry):
                root = Path(tempfile.mkdtemp()) / "staging"
                journal = root / "promoted.jsonl"
                journal.parent.mkdir(parents=True)
                journal.write_text(
                    json.dumps({"doc_id": "70002", "year": 2017})
                    + "\n \t\n"
                    + bad_entry
                    + "\n",
                    encoding="utf-8",
                )
                before = journal.read_bytes()

                check = doctor_mod._ocr_staging_check(root, now=NOW)

                self.assertEqual("UNKNOWN", check.mark, check.detail)
                self.assertFalse(check.ok)
                self.assertIn("promoted-consumed-absent UNKNOWN", check.detail)
                self.assertIn("promoted journal evidence unavailable", check.detail)
                self.assertNotIn("promoted-consumed-absent 0", check.detail)
                self.assertEqual(
                    before, journal.read_bytes(), "monitor must not rewrite the journal"
                )

    def test_unreadable_existing_promoted_journal_is_unknown(self):
        root = Path(tempfile.mkdtemp())
        journal = root / "promoted.jsonl"
        journal.write_text('{"doc_id": "70002", "year": 2017}\n', encoding="utf-8")
        original_read_text = Path.read_text

        def unreadable(path, *args, **kwargs):
            if path == journal:
                raise PermissionError("fixture denied")
            return original_read_text(path, *args, **kwargs)

        with patch.object(Path, "read_text", new=unreadable):
            check = doctor_mod._ocr_staging_check(root, now=NOW)

        self.assertEqual("UNKNOWN", check.mark, check.detail)
        self.assertFalse(check.ok)
        self.assertIn("cannot read", check.detail)
        self.assertNotIn("promoted-consumed-absent 0", check.detail)

    def test_journal_unknown_fails_doctor_without_skipping_other_checks(self):
        root = Path(tempfile.mkdtemp())
        (root / "promoted.jsonl").write_text('{"doc_id":\n', encoding="utf-8")
        passing = FreshnessCheck("fixture.monitoring", "still checked", True)

        with (
            patch.object(doctor_mod, "monitoring_checks", return_value=[passing]),
            patch.object(
                doctor_mod,
                "_stale_artifacts",
                return_value=FreshnessCheck("artifacts.consumed", "clear", True),
            ),
            patch.object(
                doctor_mod,
                "_staging_backup_check",
                return_value=FreshnessCheck("staging.backup.freshness", "fresh", True),
            ),
            patch.object(
                doctor_mod,
                "_ocr_cron_check",
                return_value=FreshnessCheck("cron.ocr.engagement", "fresh", True),
            ),
            patch.object(
                doctor_mod,
                "_heartbeat",
                return_value=FreshnessCheck("doctor.alive", "fresh", True),
            ),
            patch.object(
                doctor_mod,
                "_crontab_contract",
                return_value=FreshnessCheck("crontab.contract", "synthetic", True),
            ),
        ):
            results = doctor_mod.check(
                SimpleNamespace(), staging_root=root, ocr_log_root=root / "logs"
            )

        self.assertEqual(
            [
                "fixture.monitoring",
                "artifacts.consumed",
                "staging.ocr.backlog",
                "staging.backup.freshness",
                "cron.ocr.engagement",
                "doctor.alive",
                "crontab.contract",
            ],
            [result.name for result in results],
        )
        self.assertEqual("UNKNOWN", results[2].mark)

        output = StringIO()
        connection = SimpleNamespace(close=lambda: None)
        (root / "canonical.duckdb").touch()
        with (
            patch.object(doctor_mod, "CANONICAL", root / "canonical.duckdb"),
            patch("duckdb.connect", return_value=connection),
            patch.object(doctor_mod, "check", return_value=results),
            redirect_stdout(output),
        ):
            exit_code = doctor_mod.main()

        self.assertEqual(1, exit_code, output.getvalue())
        self.assertIn("[UNKNOWN] staging.ocr.backlog", output.getvalue())
        self.assertIn("[PASS] fixture.monitoring", output.getvalue())
        self.assertIn("[PASS] cron.ocr.engagement", output.getvalue())
        self.assertIn("doctor: 1 CHECK(S) FAILED", output.getvalue())


def test_staging_audit_severity_split_uses_sticky_worst_and_legacy_default(tmp_path):
    root = tmp_path / "staging"
    _write_record(root, 2024, "pass", audit="pass")
    _write_record(root, 2024, "critical", audit="fail", audit_severity="cosmetic")
    _write_record(root, 2024, "cosmetic", audit="fail", audit_severity="cosmetic")
    _write_record(root, 2024, "legacy", audit="fail")
    _write_record(root, 2024, "unaudited")
    (root / "audit-results.json").write_text(
        json.dumps(
            [
                {"doc_id": "critical", "verdict": "fail", "severity": "cosmetic"},
                {"doc_id": "critical", "verdict": "fail", "severity": "critical"},
                {"doc_id": "cosmetic", "verdict": "fail", "severity": "cosmetic"},
                {"doc_id": "legacy", "verdict": "fail"},
            ]
        ),
        encoding="utf-8",
    )

    check = doctor_mod._ocr_staging_check(root, now=NOW)

    assert check.ok, check.detail  # cosmetic fails are visible, not blocking
    assert "audit pass/critical-fail/cosmetic-fail/unaudited 1/2/1/1" in check.detail
    assert "cosmetic fails are tracked, not blocking" in check.detail


def test_asset_review_info_counts_only_successful_unaudited_records(tmp_path):
    root = tmp_path / "staging"
    _write_record(root, 2024, "flagged-unreviewed", asset_review=True)
    _write_record(root, 2024, "flagged-audited", audit="pass", asset_review=True)
    _write_record(root, 2024, "flagged-history", asset_review=True)
    _write_record(root, 2024, "flagged-error", status="error", asset_review=True)
    _write_record(root, 2024, "plain-unreviewed")
    (root / "audit-results.json").write_text(
        json.dumps([{"doc_id": "flagged-history", "verdict": "fail"}]),
        encoding="utf-8",
    )

    check = doctor_mod._ocr_staging_check(root, now=NOW)

    assert (
        "INFO: asset_review flagged, unaudited: 1 (audit sampler draws these first)"
        in check.detail
    )


def test_staging_backup_freshness_missing_empty_stale_and_fresh(tmp_path):
    fix = "run bash cron/backlog_ocr.sh manually or wait for the 03:30 UTC cron; the wrapper snapshots after staging"
    missing = doctor_mod._staging_backup_check(tmp_path / "missing", now=NOW)
    assert missing.mark == "FAIL"
    assert fix in missing.detail

    empty_root = tmp_path / "empty"
    empty_root.mkdir()
    empty = doctor_mod._staging_backup_check(empty_root, now=NOW)
    assert empty.mark == "FAIL"
    assert fix in empty.detail

    stale_root = tmp_path / "stale"
    stale_root.mkdir()
    stale_tar = stale_root / "glm-backlog-old.tar.gz"
    stale_tar.write_bytes(b"snapshot")
    os.utime(stale_tar, (NOW - 8.1 * 86_400, NOW - 8.1 * 86_400))
    stale = doctor_mod._staging_backup_check(stale_root, now=NOW)
    assert stale.mark == "FAIL"
    assert "8.1d old" in stale.detail
    assert fix in stale.detail

    fresh_root = tmp_path / "fresh"
    fresh_root.mkdir()
    fresh_tar = fresh_root / "glm-backlog-new.tar.gz"
    fresh_tar.write_bytes(b"snapshot")
    os.utime(fresh_tar, (NOW - 8 * 86_400, NOW - 8 * 86_400))
    fresh = doctor_mod._staging_backup_check(fresh_root, now=NOW)
    assert fresh.mark == "PASS", fresh.detail
    assert "limit 8d" in fresh.detail


class OcrCronEngagementTest(TestCase):
    def test_fresh_completed_run_passes(self):
        log_root = Path(tempfile.mkdtemp())
        _write_run_dir(log_root, "20261031", log_text="doctor exit=0\n")
        check = doctor_mod._ocr_cron_check(log_root, today=TODAY)
        self.assertTrue(check.ok, check.detail)
        self.assertIn("doctor exit=0", check.detail)

    def test_never_fired_fails_with_install_command_when_cron_line_absent(self):
        with patch.object(
            doctor_mod.subprocess,
            "run",
            return_value=SimpleNamespace(returncode=0, stdout="0 2 * * * prices\n"),
        ):
            check = doctor_mod._ocr_cron_check(
                Path(tempfile.mkdtemp()) / "absent", today=TODAY
            )
        self.assertFalse(check.ok)
        self.assertIn("never fired", check.detail)
        self.assertIn("cron/install.sh", check.detail)

    def test_never_fired_says_awaiting_schedule_when_cron_line_present(self):
        with patch.object(
            doctor_mod.subprocess,
            "run",
            return_value=SimpleNamespace(
                returncode=0,
                stdout="30 3 * * * /srv/insider-trading/cron/backlog_ocr.sh\n",
            ),
        ):
            check = doctor_mod._ocr_cron_check(
                Path(tempfile.mkdtemp()) / "absent", today=TODAY
            )
        self.assertFalse(check.ok)
        self.assertIn(
            "installed in crontab; awaiting first scheduled firing at 03:30 UTC",
            check.detail,
        )
        self.assertNotIn("cron/install.sh", check.detail)

    def test_run_dir_boundary_2d_passes_3d_is_dead(self):
        log_root = Path(tempfile.mkdtemp())
        _write_run_dir(log_root, "20261030")  # 2d before TODAY
        self.assertTrue(doctor_mod._ocr_cron_check(log_root, today=TODAY).ok)
        dead_root = Path(tempfile.mkdtemp())
        _write_run_dir(dead_root, "20261029")  # 3d before TODAY
        check = doctor_mod._ocr_cron_check(dead_root, today=TODAY)
        self.assertFalse(check.ok)
        self.assertIn("drain is dead", check.detail)
        self.assertIn("20261029", check.detail)

    def test_missing_run_log_fails_as_truncated(self):
        log_root = Path(tempfile.mkdtemp())
        _write_run_dir(log_root, "20261101", make_log=False)
        check = doctor_mod._ocr_cron_check(log_root, today=TODAY)
        self.assertTrue(check.ok, check.detail)
        self.assertIn("in progress", check.detail)

    def test_recent_run_without_doctor_exit_is_in_progress(self):
        log_root = Path(tempfile.mkdtemp())
        run_dir = _write_run_dir(
            log_root,
            "20261101",
            log_text="ocr backlog start\n--- staging year 2017 ---\n",
        )
        started = NOW - 2 * 3600
        os.utime(run_dir, (started, started))
        check = doctor_mod._ocr_cron_check(log_root, today=TODAY, now=NOW)
        self.assertTrue(check.ok, check.detail)
        self.assertIn("in progress", check.detail)
        self.assertIn("2.0h old", check.detail)

    def test_old_run_without_doctor_exit_fails_as_truncated(self):
        log_root = Path(tempfile.mkdtemp())
        run_dir = _write_run_dir(
            log_root,
            "20261101",
            log_text="ocr backlog start\n--- staging year 2017 ---\n",
        )
        started = NOW - 8 * 3600
        os.utime(run_dir, (started, started))
        check = doctor_mod._ocr_cron_check(log_root, today=TODAY, now=NOW)
        self.assertFalse(check.ok)
        self.assertIn("no 'doctor exit=' line", check.detail)
        self.assertIn("truncated", check.detail)

    def test_nonzero_doctor_exit_in_log_fails(self):
        log_root = Path(tempfile.mkdtemp())
        _write_run_dir(log_root, "20261101", log_text="doctor exit=3\n")
        check = doctor_mod._ocr_cron_check(log_root, today=TODAY)
        self.assertFalse(check.ok)
        self.assertIn("doctor exit=3", check.detail)

    def test_junk_entries_ignored_newest_wins(self):
        log_root = Path(tempfile.mkdtemp())
        (log_root / "notes").mkdir()
        _write_run_dir(log_root, "20261001", log_text="doctor exit=0\n")
        _write_run_dir(log_root, "20261101", log_text="doctor exit=0\n")
        check = doctor_mod._ocr_cron_check(log_root, today=TODAY)
        self.assertTrue(check.ok, check.detail)
        self.assertIn("20261101", check.detail)


def _ocr_start(stamp: str) -> str:
    return f"ocr backlog start: {stamp} UTC (repo /fixture @ abc123)\n"


def _ocr_skip(log_root: Path, stamp: str = "2026-11-01T03:30:00+00:00") -> str:
    return _ocr_start(stamp) + (
        f"another drain is running (lock: {log_root}/drain.lock); exiting\n"
    )


@pytest.mark.parametrize(
    "winner,modified,held,expected",
    [
        ("doctor exit=0\n", "2026-11-01T04:28:00+00:00", False, True),
        ("doctor exit=3\n", "2026-11-01T04:28:00+00:00", False, False),
        ("--- staging year 2020 ---\n", "2026-11-01T04:28:00+00:00", False, False),
        # A completion before the skipped launch does not prove overlap.
        ("doctor exit=0\n", "2026-11-01T02:28:00+00:00", False, False),
        # Touching the log well past the winner's run budget is not completion evidence.
        ("doctor exit=0\n", "2026-11-01T11:28:00+00:00", False, False),
        # A lock still held at noon cannot revive an overnight truncated winner.
        ("--- staging year 2020 ---\n", "2026-11-01T04:28:00+00:00", True, False),
    ],
)
def test_ocr_cross_midnight_lock_skip_requires_fresh_overlapping_winner(
    tmp_path, winner, modified, held, expected
):
    root = tmp_path / "logs"
    run = _write_run_dir(
        root, "20261031",
        log_text=_ocr_start("2026-10-31T22:33:00+00:00") + winner,
    )
    stamp = datetime.fromisoformat(modified).timestamp()
    os.utime(run / "run.log", (stamp, stamp))
    _write_run_dir(root, "20261101", log_text=_ocr_skip(root))
    with patch.object(doctor_mod, "_ocr_drain_lock_held", return_value=held):
        check = doctor_mod._ocr_cron_check(root, today=TODAY, now=NOW)
    assert check.ok is expected, check.detail
    assert "lock-skipped" in check.detail
    if expected:
        assert "overlapping winner 20261031 completed with doctor exit=0" in check.detail


@pytest.mark.parametrize("held", [False, True])
def test_ocr_lock_skip_only_never_proves_work(tmp_path, held):
    _write_run_dir(tmp_path, "20261101", log_text=_ocr_skip(tmp_path))
    with patch.object(doctor_mod, "_ocr_drain_lock_held", return_value=held):
        check = doctor_mod._ocr_cron_check(tmp_path, today=TODAY, now=NOW)
    assert not check.ok, check.detail


@pytest.mark.parametrize("held,staging,expected", [(True, True, True), (False, True, False), (True, False, False)])
def test_ocr_lock_skip_active_winner_needs_bounded_timestamp_staging_and_lock(
    tmp_path, held, staging, expected
):
    _write_run_dir(
        tmp_path, "20261031",
        log_text=_ocr_start("2026-10-31T22:33:00+00:00")
        + ("--- staging year 2020 ---\n" if staging else ""),
    )
    _write_run_dir(tmp_path, "20261101", log_text=_ocr_skip(tmp_path))
    now = datetime(2026, 11, 1, 4, tzinfo=timezone.utc).timestamp()
    with patch.object(doctor_mod, "_ocr_drain_lock_held", return_value=held):
        check = doctor_mod._ocr_cron_check(tmp_path, today=TODAY, now=now)
    assert check.ok is expected, check.detail


@pytest.mark.parametrize("tail,held,expected", [
    ("", False, False),
    ("--- staging year 2020 ---\n", False, False),
    ("--- staging year 2020 ---\n", True, True),
    ("doctor exit=4\n", False, False),
    ("doctor exit=0\n", False, True),
    ("  [PASS] cron.ocr.engagement: old doctor exit=0\n", False, False),
])
def test_ocr_latest_appended_invocation_cannot_inherit_old_success(
    tmp_path, tail, held, expected
):
    run = _write_run_dir(
        tmp_path, "20261101",
        log_text=_ocr_start("2026-11-01T01:00:00+00:00") + "doctor exit=0\n"
        + _ocr_start("2026-11-01T11:00:00+00:00") + tail,
    )
    os.utime(run, (NOW, NOW))
    with patch.object(doctor_mod, "_ocr_drain_lock_held", return_value=held):
        check = doctor_mod._ocr_cron_check(tmp_path, today=TODAY, now=NOW)
    assert check.ok is expected, check.detail


@pytest.mark.parametrize("trailing", ["", "doctor exit=0\n"])
def test_ocr_same_file_skip_does_not_retimestamp_an_earlier_completion(tmp_path, trailing):
    _write_run_dir(
        tmp_path, "20261101",
        log_text=_ocr_start("2026-11-01T01:00:00+00:00") + "doctor exit=0\n"
        + _ocr_skip(tmp_path) + trailing,
    )
    check = doctor_mod._ocr_cron_check(tmp_path, today=TODAY, now=NOW)
    assert not check.ok, check.detail


def test_ocr_stale_winner_and_repeated_skips_do_not_keep_cron_alive(tmp_path):
    run = _write_run_dir(
        tmp_path, "20261029",
        log_text=_ocr_start("2026-10-29T22:33:00+00:00") + "doctor exit=0\n",
    )
    os.utime(run / "run.log", (NOW, NOW))
    _write_run_dir(tmp_path, "20261031", log_text=_ocr_skip(tmp_path, "2026-10-31T03:30:00+00:00"))
    _write_run_dir(tmp_path, "20261101", log_text=_ocr_skip(tmp_path))
    assert not doctor_mod._ocr_cron_check(tmp_path, today=TODAY, now=NOW).ok


def test_ocr_latest_failed_winner_is_not_hidden_by_older_success(tmp_path):
    run = _write_run_dir(
        tmp_path, "20261031",
        log_text=_ocr_start("2026-10-31T12:00:00+00:00") + "doctor exit=0\n"
        + _ocr_start("2026-10-31T22:33:00+00:00") + "doctor exit=2\n",
    )
    stamp = datetime(2026, 11, 1, 4, tzinfo=timezone.utc).timestamp()
    os.utime(run / "run.log", (stamp, stamp))
    _write_run_dir(tmp_path, "20261101", log_text=_ocr_skip(tmp_path))
    assert not doctor_mod._ocr_cron_check(tmp_path, today=TODAY, now=NOW).ok


def test_ocr_truncated_timestamped_run_is_not_revived_by_directory_mtime(tmp_path):
    run = _write_run_dir(
        tmp_path, "20261101",
        log_text=_ocr_start("2026-11-01T01:00:00+00:00") + "--- staging year 2020 ---\n",
    )
    os.utime(run, (NOW, NOW))
    with patch.object(doctor_mod, "_ocr_drain_lock_held", return_value=False):
        check = doctor_mod._ocr_cron_check(tmp_path, today=TODAY, now=NOW)
    assert not check.ok, check.detail
    assert "truncated" in check.detail


def test_ocr_skip_for_another_lock_cannot_use_recent_directory_grace(tmp_path):
    _write_run_dir(tmp_path, "20261101", log_text=_ocr_skip(tmp_path / "other"))
    assert not doctor_mod._ocr_cron_check(tmp_path, today=TODAY, now=NOW).ok


class MonitoringChecksContractTest(TestCase):
    """freshness.py's consumers (doctor + ``health``) must share one surface."""

    def test_monitoring_checks_include_the_ocr_surface(self):
        con = duckdb.connect(":memory:")
        con.execute(
            "CREATE TABLE canonical_transactions (source VARCHAR, disclosure_date DATE)"
        )
        session = _house_session(TODAY)
        con.execute(
            "INSERT INTO canonical_transactions VALUES ('house_pdf', ?)", [session]
        )
        names = [c.name for c in monitoring_checks(con, staging_root=None, today=TODAY)]
        self.assertEqual(
            [
                "prices.freshness",
                "dividends.populated",
                "form4.freshness",
                "receipts.freshness",
                "dividends.freshness",
                "house.ingestion.freshness",
                "ocr.staging.freshness",
            ],
            names,
        )
        # staging_root=None → UNKNOWN, never green.
        ocr = [c for c in monitoring_checks(con, staging_root=None, today=TODAY)
               if c.name == "ocr.staging.freshness"][0]
        self.assertEqual("UNKNOWN", ocr.mark)

    def test_legacy_end_to_end_surface_is_unchanged(self):
        con = duckdb.connect(":memory:")
        self.assertEqual(
            4, len(end_to_end_checks(con, receipts_path=None)),
            "the legacy 4-check contract must not grow silently",
        )


class DoctorIntegrationTest(TestCase):
    """The wired product (scripts/doctor.py), driven through env overrides.

    Fixtures are stamped against the real clock because the subprocess uses
    the real ``date.today()``/``time.time()``; exact threshold boundaries
    are covered by the unit tests above with injected clocks.
    """

    def _build_canonical(self, path: Path) -> None:
        con = duckdb.connect(str(path))
        con.execute("CREATE TABLE prices (ticker VARCHAR, date DATE, close DOUBLE)")
        session = previous_nyse_session(date.today() - timedelta(days=1)).date()
        con.execute("INSERT INTO prices VALUES ('AAPL', ?, 1.0)", [session])
        con.execute(
            "CREATE TABLE canonical_transactions (source VARCHAR, disclosure_date DATE)"
        )
        con.execute("INSERT INTO canonical_transactions VALUES ('form4', ?)", [session])
        con.execute(
            "INSERT INTO canonical_transactions VALUES ('house_pdf', ?)", [session]
        )
        con.execute(
            "CREATE TABLE dividends (ticker VARCHAR, ex_date DATE, cash_amount DOUBLE)"
        )
        con.execute("INSERT INTO dividends VALUES ('AAPL', '2026-10-01', 0.5)")
        con.close()

    def _run(self, staging_root: Path | None, log_root: Path | None) -> tuple[int, str]:
        tmp = Path(tempfile.mkdtemp(prefix="doctor-ocr-"))
        canonical = tmp / "congress.duckdb"
        self._build_canonical(canonical)
        heartbeat = tmp / "doctor-last-run"
        heartbeat.write_text(str(time.time()))
        env = dict(os.environ)
        env["DOCTOR_CANONICAL"] = str(canonical)
        env["DOCTOR_HEARTBEAT"] = str(heartbeat)
        if staging_root is not None:
            env["DOCTOR_STAGING_ROOT"] = str(staging_root)
        env["DOCTOR_OCR_LOG_ROOT"] = str(log_root or tmp / "ocr-logs-absent")
        # Never inspect the operator's installed crontab from a process test.
        bin_dir = tmp / "bin"
        bin_dir.mkdir()
        crontab = bin_dir / "crontab"
        contract = (
            (REPO / "cron" / "crontab.txt").read_text().replace("@REPO@", str(REPO))
        )
        crontab.write_text("#!/bin/sh\ncat <<'CRONTAB_FIXTURE'\n" + contract + "\nCRONTAB_FIXTURE\n")
        crontab.chmod(0o700)
        # Keep mandatory validation PATH shims immediately after this fixture.
        env["PATH"] = str(bin_dir) + os.pathsep + env["PATH"]
        result = subprocess.run(
            [sys.executable, str(REPO / "scripts" / "doctor.py")],
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
            cwd=str(REPO),
            check=False,
        )
        return result.returncode, result.stdout + result.stderr

    def test_healthy_staging_and_drain_show_as_pass(self):
        staging = Path(tempfile.mkdtemp(prefix="stag-"))
        _write_record(staging, 2017, "20001", status="success", audit="pass",
                      epoch=time.time())
        _write_record(staging, 2017, "20002", status="no_txs", epoch=time.time())
        logs = Path(tempfile.mkdtemp(prefix="logs-"))
        _write_run_dir(
            logs, date.today().strftime("%Y%m%d"), log_text="doctor exit=0\n"
        )
        rc, output = self._run(staging, logs)
        self.assertIn("[PASS] staging.ocr.backlog", output)
        self.assertIn("promoted-consumed-absent 0", output)
        self.assertIn("[PASS] cron.ocr.engagement", output)
        self.assertIn("[PASS] ocr.staging.freshness", output)
        self.assertIn("[PASS] house.ingestion.freshness", output)
        self.assertIn("[PASS] dividends.populated", output)
        self.assertIn("[UNKNOWN] dividends.freshness", output)

    def test_retry_starved_staging_fails_the_whole_doctor(self):
        staging = Path(tempfile.mkdtemp(prefix="stag-"))
        _write_record(
            staging, 2016, "30001", status="error", age_days=9, error="x",
            epoch=time.time(),
        )
        logs = Path(tempfile.mkdtemp(prefix="logs-"))
        _write_run_dir(
            logs, date.today().strftime("%Y%m%d"), log_text="doctor exit=0\n"
        )
        rc, output = self._run(staging, logs)
        self.assertEqual(1, rc, output)
        self.assertIn("[FAIL] staging.ocr.backlog", output)
        self.assertIn("retry starved", output)

    def test_dead_drain_cron_fails(self):
        staging = Path(tempfile.mkdtemp(prefix="stag-"))
        logs = Path(tempfile.mkdtemp(prefix="logs-"))
        dead = (date.today() - timedelta(days=5)).strftime("%Y%m%d")
        _write_run_dir(logs, dead)
        rc, output = self._run(staging, logs)
        self.assertEqual(1, rc, output)
        self.assertIn("[FAIL] cron.ocr.engagement", output)
        self.assertIn("drain is dead", output)

    def test_missing_staging_dir_is_unknown_not_a_crash(self):
        rc, output = self._run(
            Path(tempfile.mkdtemp(prefix="absent-")) / "nope", None
        )
        self.assertIn("[UNKNOWN] staging.ocr.backlog", output)
        self.assertIn("[UNKNOWN] ocr.staging.freshness", output)
        self.assertNotIn("Traceback", output)
        self.assertEqual(1, rc, "UNKNOWN is fail-loud, not green")


class CrontabContractTest(TestCase):
    def test_missing_crontab_command_is_unknown(self):
        with patch.object(doctor_mod.subprocess, "run", side_effect=FileNotFoundError):
            result = doctor_mod._crontab_contract()

        self.assertEqual("UNKNOWN", result.mark)
        self.assertFalse(result.ok)
        self.assertIn("deployment not verified", result.detail)

    def test_nonzero_crontab_result_is_unknown(self):
        with patch.object(
            doctor_mod.subprocess,
            "run",
            return_value=SimpleNamespace(
                returncode=1, stdout="", stderr="permission denied"
            ),
        ):
            result = doctor_mod._crontab_contract()

        self.assertEqual("UNKNOWN", result.mark)
        self.assertFalse(result.ok)
        self.assertIn("permission denied", result.detail)

    def test_matching_crontab_passes(self):
        contract = doctor_mod.CRONTAB_CONTRACT.read_text().replace(
            "@REPO@", str(doctor_mod.REPO)
        )
        with patch.object(
            doctor_mod.subprocess,
            "run",
            return_value=SimpleNamespace(returncode=0, stdout=contract, stderr=""),
        ):
            result = doctor_mod._crontab_contract()

        self.assertEqual("PASS", result.mark)
        self.assertTrue(result.ok)

    def test_drifted_crontab_fails(self):
        with patch.object(
            doctor_mod.subprocess,
            "run",
            return_value=SimpleNamespace(
                returncode=0, stdout="# no repo cron jobs\n", stderr=""
            ),
        ):
            result = doctor_mod._crontab_contract()

        self.assertEqual("FAIL", result.mark)
        self.assertFalse(result.ok)
        self.assertIn("drifted", result.detail)

    def test_main_exits_nonzero_when_a_check_is_unknown(self):
        unknown = FreshnessCheck.unknown("crontab.contract", "deployment not verified")
        connection = SimpleNamespace(close=lambda: None)
        canonical = Path(tempfile.mkdtemp()) / "canonical.duckdb"
        canonical.touch()
        with (
            patch.object(doctor_mod, "CANONICAL", canonical),
            patch("duckdb.connect", return_value=connection),
            patch.object(doctor_mod, "check", return_value=[unknown]),
        ):
            self.assertEqual(1, doctor_mod.main())
