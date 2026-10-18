"""Cron contract tripwire (2026-10-18 incident).

The incident's cron wrappers lived under gitignored data/: untracked,
unreviewed, untestable, invisible to repo-level audits. This test fails the
suite unless every repo cron job is tracked under cron/, referenced by the
tracked crontab source of truth, portable, executable, and ends in a
canonical consumption/verification step."""

import os
import shutil
import subprocess
from pathlib import Path
from unittest import TestCase

ROOT = Path(__file__).resolve().parents[1]
CRON_DIR = ROOT / "cron"
CRONTAB_SOURCE = CRON_DIR / "crontab.txt"
CONSUMPTION_MARKERS = ("--merge-into", "scripts/doctor.py")


def _job_lines() -> list[tuple[str, str]]:
    jobs = []
    for raw in CRONTAB_SOURCE.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 6 or "@REPO@/cron/" not in fields[5]:
            continue  # env lines like CRON_TZ=UTC
        jobs.append((" ".join(fields[:5]), fields[5]))
    return jobs


class CrontabSourceOfTruthTest(TestCase):
    def test_source_of_truth_exists_with_managed_block(self):
        text = CRONTAB_SOURCE.read_text()
        self.assertIn("# BEGIN insider-trading", text)
        self.assertIn("# END insider-trading", text)
        self.assertIn("CRON_TZ=UTC", text)

    def test_at_least_the_two_known_jobs_are_declared(self):
        jobs = _job_lines()
        self.assertGreaterEqual(len(jobs), 2)
        targets = {path.rsplit("/", 1)[-1] for _, path in jobs}
        self.assertIn("weekly_prices.sh", targets)
        self.assertIn("daily_doctor.sh", targets)


class WrapperContractTest(TestCase):
    def test_every_declared_job_is_a_tracked_executable_wrapper(self):
        for schedule, target in _job_lines():
            with self.subTest(target=target):
                self.assertTrue(
                    target.startswith("@REPO@/cron/"),
                    f"cron jobs must live in cron/: {target}",
                )
                wrapper = ROOT / target.replace("@REPO@", str(ROOT))
                self.assertTrue(wrapper.exists(), target)
                self.assertTrue(
                    os.access(wrapper, os.X_OK), f"not executable: {target}"
                )

    def test_wrappers_are_portable_not_machine_hardcoded(self):
        for _, target in _job_lines():
            wrapper = ROOT / target.replace("@REPO@", str(ROOT))
            text = wrapper.read_text()
            with self.subTest(target=target):
                self.assertNotIn(
                    "/home/ubuntu",
                    text,
                    "wrappers must derive the repo root, not hardcode a machine",
                )
                self.assertIn("BASH_SOURCE", text)

    def test_wrappers_end_in_a_consumption_or_verification_step(self):
        for _, target in _job_lines():
            wrapper = ROOT / target.replace("@REPO@", str(ROOT))
            text = wrapper.read_text()
            with self.subTest(target=target):
                self.assertTrue(
                    any(marker in text for marker in CONSUMPTION_MARKERS),
                    f"{target} writes without a canonical consumption/"
                    "verification step; wire --merge-into or doctor",
                )

    def test_wrappers_stamp_the_checked_out_git_sha(self):
        for _, target in _job_lines():
            wrapper = ROOT / target.replace("@REPO@", str(ROOT))
            with self.subTest(target=target):
                self.assertIn("rev-parse", wrapper.read_text())


class EscalationWiringTest(TestCase):
    """A doctor failure must route somewhere concrete, not into syslog limbo
    (this box has no MTA: no mail/sendmail)."""

    def test_doctor_wrapper_writes_heartbeat_alert_log_and_sticky_flag(self):
        text = (CRON_DIR / "daily_doctor.sh").read_text()
        self.assertIn("doctor-last-run", text)  # dead-man heartbeat
        self.assertIn("ptr-alerts.log", text)  # operator's central cron log
        self.assertIn("ACTIVE", text)  # sticky acknowledge flag

    def test_doctor_checks_heartbeat_and_crontab_drift(self):
        text = (ROOT / "scripts" / "doctor.py").read_text()
        self.assertIn("doctor.alive", text)
        self.assertIn("crontab.contract", text)
        self.assertIn("analyzer.freshness", text)

    def test_install_dry_run_changes_nothing(self):
        if not shutil.which("crontab"):
            self.skipTest("crontab unavailable")
        result = subprocess.run(
            ["bash", str(CRON_DIR / "install.sh")],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn("dry run", result.stdout)
