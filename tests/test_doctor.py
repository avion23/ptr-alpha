"""Integration: scripts/doctor.py end to end against a synthetic canonical.

The 2026-10-18 fix shipped doctor.py with no tests and its merge sibling
crashed on first real use (ATTACH parameter binding). This runs the actual
script in a subprocess so the wired product, not a reimplementation, is what
is verified."""

import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone
import tempfile
from unittest import TestCase

import duckdb

REPO = Path(__file__).resolve().parents[1]


def _build_canonical(path: Path, *, fresh_dividends: bool) -> None:
    con = duckdb.connect(str(path))
    con.execute("CREATE TABLE prices (ticker VARCHAR, date DATE, close DOUBLE)")
    from analyzer.price_repository import previous_nyse_session

    fresh = previous_nyse_session(date.today() - timedelta(days=1)).date()
    con.execute("INSERT INTO prices VALUES ('AAPL', ?, 1.0)", [fresh])
    con.execute(
        "CREATE TABLE canonical_transactions (source VARCHAR, disclosure_date DATE)"
    )
    con.execute(
        "INSERT INTO canonical_transactions VALUES ('form4', ?)",
        [date.today() - timedelta(days=1)],
    )
    con.execute(
        "CREATE TABLE dividends (ticker VARCHAR, ex_date DATE, cash_amount DOUBLE)"
    )
    if fresh_dividends:
        con.execute("INSERT INTO dividends VALUES ('AAPL', '2026-10-01', 0.5)")
    con.close()


class DoctorIntegrationTest(TestCase):
    def _run(self, canonical: Path, heartbeat_age_s: int) -> tuple[int, str]:
        env = dict(os.environ)
        env["DOCTOR_CANONICAL"] = str(canonical)
        alerts = Path(tempfile.mkdtemp(prefix="doctor-alerts-"))
        heartbeat = alerts / "doctor-last-run"
        heartbeat.write_text(str(datetime.now(timezone.utc).timestamp()))
        stamp = datetime.now(timezone.utc).timestamp() - heartbeat_age_s
        os.utime(heartbeat, (stamp, stamp))
        env["DOCTOR_HEARTBEAT"] = str(heartbeat)
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

    def test_fed_pipeline_and_fresh_heartbeat_pass_contract_checks(self):
        tmp = Path(tempfile.mkdtemp(prefix="doctor-canon-"))
        canonical = tmp / "congress.duckdb"
        _build_canonical(canonical, fresh_dividends=True)
        rc, output = self._run(canonical, heartbeat_age_s=0)
        self.assertIn("[PASS] dividends.populated", output)
        self.assertIn("[PASS] doctor.alive", output)
        self.assertIn("[PASS] crontab.contract", output)
        self.assertNotIn("unavailable", output)  # synthetic schema is sufficient

    def test_starved_dividends_and_dead_heartbeat_fail(self):
        tmp = Path(tempfile.mkdtemp(prefix="doctor-canon-"))
        canonical = tmp / "congress.duckdb"
        _build_canonical(canonical, fresh_dividends=False)
        rc, output = self._run(canonical, heartbeat_age_s=48 * 3600)
        self.assertEqual(1, rc, output)
        self.assertIn("[FAIL] dividends.populated", output)
        self.assertIn("[FAIL] doctor.alive", output)
        self.assertIn("48.0h old", output)

    def test_never_fired_heartbeat_fails_closed(self):
        tmp = Path(tempfile.mkdtemp(prefix="doctor-canon-"))
        canonical = tmp / "congress.duckdb"
        _build_canonical(canonical, fresh_dividends=True)
        env = dict(os.environ)
        env["DOCTOR_CANONICAL"] = str(canonical)
        env["DOCTOR_HEARTBEAT"] = str(tmp / "absent-heartbeat")
        result = subprocess.run(
            [sys.executable, str(REPO / "scripts" / "doctor.py")],
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
            cwd=str(REPO),
            check=False,
        )
        self.assertEqual(1, result.returncode)
        self.assertIn("has never fired", result.stdout + result.stderr)
