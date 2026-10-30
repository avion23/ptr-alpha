"""Cron contract tripwire (2026-10-18 incident).

The incident's cron wrappers lived under gitignored data/: untracked,
unreviewed, untestable, invisible to repo-level audits. This test fails the
suite unless every repo cron job is tracked under cron/, referenced by the
tracked crontab source of truth, portable, executable, and ends in a
canonical consumption/verification step."""

import os
import shutil
import subprocess
import sys
import tempfile
from datetime import date, timedelta
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

    def test_price_refresh_schedule_and_quote_age_budget(self):
        schedule = next(
            schedule
            for schedule, target in _job_lines()
            if target.endswith("/weekly_prices.sh")
        )
        self.assertEqual("17 0 * * 2,4,6", schedule)

        # Cron weekday numbers are Sunday=0; datetime uses Monday=0.
        cron_days = {int(day) for day in schedule.split()[4].split(",")}
        refresh_days = {(day + 6) % 7 for day in cron_days}

        def last_weekday_before(day: date) -> date:
            quote = day - timedelta(days=1)
            while quote.weekday() >= 5:
                quote -= timedelta(days=1)
            return quote

        # Start Monday with Friday's close from the preceding run. Measure
        # both just before and just after each scheduled refresh for a full
        # weekly cycle; this is pure date arithmetic, no market/API calls.
        monday = date(2026, 10, 19)
        quote_date = monday - timedelta(days=3)
        ages = []
        for offset in range(7):
            day = monday + timedelta(days=offset)
            ages.append((day - quote_date).days)
            if day.weekday() in refresh_days:
                quote_date = last_weekday_before(day)
            ages.append((day - quote_date).days)

        self.assertEqual(4, max(ages))
        self.assertLess(max(ages), 5)


class WrapperContractTest(TestCase):
    def test_price_retries_get_unique_refresh_and_snapshot_paths(self):
        text = (CRON_DIR / "weekly_prices.sh").read_text()
        self.assertIn('mktemp -d "$log_root/$(date -u +%Y%m%dT%H%M%S)-XXXXXX"', text)
        self.assertIn('--db "$temp_db"', text)
        self.assertIn('--generation "prices-cron-$run_id"', text)
        self.assertIn('--out "$snapshot_dir"', text)
        # The doctor still sees each flat per-attempt temp DB for freshness.
        doctor = (ROOT / "scripts" / "doctor.py").read_text()
        self.assertIn('PRICES_CRON.glob("*/prices.duckdb")', doctor)

    def test_price_attempts_survive_faults_and_concurrent_retries(self):
        source = (CRON_DIR / "weekly_prices.sh").read_text()
        self.assertNotIn("prune_runs", source)
        self.assertNotIn("rm -rf", source)
        self.assertIn("operator-explicit GC only", source)

        # Execute only a copy under a temporary repo; fake git/date/python3
        # ensure no live refresh, database, cron, or network access occurs.
        with tempfile.TemporaryDirectory() as temp:
            repo = Path(temp)
            wrapper = repo / "cron" / "weekly_prices.sh"
            wrapper.parent.mkdir()
            shutil.copy2(CRON_DIR / "weekly_prices.sh", wrapper)
            bin_dir = repo / "fake-bin"
            bin_dir.mkdir()
            fake_python = bin_dir / "python3"
            fake_python.write_text(
                f"#!{sys.executable}\n"
                "import os, sys, time\n"
                "from pathlib import Path\n"
                "script = Path(sys.argv[1]).name\n"
                "if script == 'refresh_prices.py' and os.getenv('BARRIER_DIR'):\n"
                "    db = Path(sys.argv[sys.argv.index('--db') + 1])\n"
                "    barrier = Path(os.environ['BARRIER_DIR'])\n"
                "    (barrier / (db.parent.name + '.ready')).touch()\n"
                "    deadline = time.monotonic() + 10\n"
                "    while len(list(barrier.glob('*.ready'))) < 2:\n"
                "        if time.monotonic() > deadline: sys.exit(98)\n"
                "        time.sleep(0.01)\n"
                "rc = os.getenv('REFRESH_RC' if script == 'refresh_prices.py' "
                "else 'SNAPSHOT_RC', '0')\n"
                "sys.exit(int(rc))\n"
            )
            fake_python.chmod(0o755)
            (bin_dir / "git").write_text("#!/bin/sh\nprintf 'testsha\\n'\n")
            (bin_dir / "date").write_text(
                "#!/bin/sh\n"
                'case "$*" in\n'
                "  *+%Y%m%dT%H%M%S*) printf '20261024T120000\\n' ;;\n"
                "  *--iso-8601=seconds*) printf '2026-10-24T12:00:00+00:00\\n' ;;\n"
                "  *) exit 2 ;;\n"
                "esac\n"
            )
            (bin_dir / "git").chmod(0o755)
            (bin_dir / "date").chmod(0o755)
            env = {
                key: value
                for key, value in os.environ.items()
                if "proxy" not in key.lower()
            }
            env["PATH"] = f"{bin_dir}:/usr/bin:/bin"
            log_root = repo / "data" / "logs" / "prices-cron"
            old_names = [
                f"20261024T120000-{suffix}"
                for suffix in ("aaaaaa", "bbbbbb", "cccccc", "dddddd", "eeeeee")
            ]
            for index, name in enumerate(old_names):
                attempt = log_root / name
                (attempt / "snapshot").mkdir(parents=True)
                (attempt / "report.json").write_text(f"report-{index}")
                (attempt / "snapshot" / "manifest.json").write_text(f"bundle-{index}")
                (attempt / "run.log").write_text(f"evidence-{index}")
                if index == 0:
                    (attempt / "ACTIVE").write_text("still running")

            preserved = {
                path.relative_to(log_root): path.read_bytes()
                for path in log_root.rglob("*")
                if path.is_file()
            }

            fault = subprocess.run(
                ["bash", str(wrapper)],
                cwd=repo,
                env={**env, "REFRESH_RC": "17"},
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
            self.assertEqual(17, fault.returncode, fault.stderr)
            snapshot_fault = subprocess.run(
                ["bash", str(wrapper)],
                cwd=repo,
                env={**env, "SNAPSHOT_RC": "19"},
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
            self.assertEqual(19, snapshot_fault.returncode, snapshot_fault.stderr)

            barrier = repo / "barrier"
            barrier.mkdir()
            concurrent_env = {**env, "BARRIER_DIR": str(barrier)}
            runs = [
                subprocess.Popen(
                    ["bash", str(wrapper)],
                    cwd=repo,
                    env=concurrent_env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                for _ in range(2)
            ]
            for run in runs:
                stdout, stderr = run.communicate(timeout=20)
                self.assertEqual(0, run.returncode, f"{stdout}\n{stderr}")

            self.assertEqual(2, len(list(barrier.glob("*.ready"))))
            for relative, contents in preserved.items():
                self.assertEqual(contents, (log_root / relative).read_bytes())
            self.assertEqual(9, len([p for p in log_root.iterdir() if p.is_dir()]))

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


class BacklogOcrContractTest(TestCase):
    def test_scheduled_drain_uses_one_coordinator_with_explicit_lanes(self):
        text = (CRON_DIR / "backlog_ocr.sh").read_text()
        self.assertIn("from analyzer.ocr.coordinator import run_pipeline", text)
        self.assertIn("run_pipeline(sys.argv[1], local_workers=3, gemini_workers=0,", text)
        self.assertIn("glm_workers=2)", text)
        self.assertNotIn("glm_stage.py stage", text)
        self.assertNotIn("--from-manifest", text)
        self.assertIn('exit "$coordinator_rc"', text)
        self.assertIn('exit "$backup_rc"', text)
        self.assertIn('exit "$doctor_rc"', text)
        self.assertNotIn("for year in 2013", text)
        self.assertNotIn("for year in 2018", text)
