"""Unit tests for scripts/go_live.py: stage ordering, gating, manifest,
dry-run, staging safety, and the invariants audit.

The orchestrator contract under test:

* stages run in the fixed order house -> senate -> capitol -> prices ->
  invariants -> validation, each invoking existing merged CLIs/scripts with
  exact args;
* a failing stage stops promotion: later stages are recorded as not_run and
  the manifest final status is ``not_established``;
* only when every stage passes is the final status ``established``;
* --dry-run prints the plan and creates nothing;
* the real data directory and pre-existing generation directories are refused;
* the invariants audit fails closed on bad rows and phantom duplicates.
"""

import json
import shutil
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest import mock

from scripts import go_live

REPO_ROOT = Path(__file__).resolve().parents[1]


def make_context(staging_dir, **overrides) -> go_live.Context:
    defaults = dict(
        repo_root=go_live._REPO_ROOT,
        staging_dir=Path(staging_dir),
        generation="test-gen",
        house_year=2024,
        senate_start=date(2023, 1, 1),
        senate_end=date(2024, 1, 1),
        train_start=date(2022, 1, 1),
        train_end=date(2023, 12, 31),
        test_start=date(2024, 1, 1),
        test_end=date(2025, 6, 30),
        dry_run=False,
        verbose=False,
        runner=("python3", "-c", go_live._CLI_RUNNER),
        env={"PYTHONPATH": str(go_live._SRC), "PTR_SKIP_DOCLING": "1"},
    )
    defaults.update(overrides)
    return go_live.Context(**defaults)


class TestStageOrderingAndCommands(unittest.TestCase):
    def test_stage_order_is_fixed_contract(self):
        self.assertEqual(
            go_live.STAGE_ORDER,
            ("house", "senate", "capitol", "prices", "invariants", "validation"),
        )

    def test_stage_commands_use_merged_cli_exact_args(self):
        staging = Path("/staging/root/gen")
        ctx = make_context(staging)
        commands = go_live.build_stage_commands(ctx)

        runner = list(ctx.runner)
        self.assertEqual(
            commands["house"],
            [
                runner
                + [
                    "refresh",
                    "--year",
                    "2024",
                    "--data-dir",
                    str(staging),
                    "--gemini-ocr",
                ]
            ],
        )
        self.assertEqual(
            commands["senate"],
            [
                runner
                + [
                    "fetch-senate-efd",
                    "--start",
                    "2023-01-01",
                    "--end",
                    "2024-01-01",
                    "--data-dir",
                    str(staging / "senate"),
                ]
            ],
        )
        self.assertEqual(
            commands["capitol"],
            [
                runner
                + [
                    "fetch-capitol",
                    "--all",
                    "--output",
                    str(staging / "capitol_recon.json"),
                    "--generation",
                    "test-gen",
                    "--data-dir",
                    str(staging),
                ]
            ],
        )
        self.assertEqual(
            commands["prices"],
            [
                runner
                + [
                    "analyze",
                    "--year",
                    "2024",
                    "--mode",
                    "ranks",
                    "--data-dir",
                    str(staging),
                ],
                runner
                + [
                    "snapshot",
                    "--data-dir",
                    str(staging),
                    "--output",
                    str(staging / "price_snapshot.json"),
                ],
            ],
        )
        self.assertEqual(
            commands["invariants"],
            [
                [
                    sys.executable,
                    str(go_live._REPO_ROOT / "scripts" / "purge_phantom_rows.py"),
                    str(staging / "congress.duckdb"),
                ],
                [
                    sys.executable,
                    str(go_live._REPO_ROOT / "scripts" / "purge_phantom_rows.py"),
                    str(staging / "senate" / "congress.duckdb"),
                ],
            ],
        )
        self.assertEqual(
            commands["validation"],
            [
                runner
                + [
                    "validate",
                    "--train-start",
                    "2022-01-01",
                    "--train-end",
                    "2023-12-31",
                    "--test-start",
                    "2024-01-01",
                    "--test-end",
                    "2025-06-30",
                    "--data-dir",
                    str(staging),
                ]
            ],
        )


class TestOrderingAndGating(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.staging = Path(self.tmp) / "gen"
        self.staging.mkdir()
        self.ctx = make_context(self.staging)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def _recorded_run(self, fail_at=None):
        """Patch _execute_command to record order and simulate failures."""
        calls = []

        def command_label(command):
            if len(command) > 3:
                return command[3]
            return Path(command[1]).name

        def fake_execute(ctx, command, log_path):
            calls.append(command_label(command))
            if fail_at is not None and any(part == fail_at for part in command):
                return 1
            return 0

        with mock.patch.object(go_live, "_execute_command", side_effect=fake_execute):
            results = go_live.run_stages(self.ctx)
        return calls, results

    def test_stages_execute_in_declared_order(self):
        # Provide the artifacts the post-command gates require so every stage
        # can pass; the invariants audit is replaced by a stub.
        (self.staging / "capitol_recon.json").write_text("{}")
        (self.staging / "price_snapshot.json").write_text("{}")
        (self.staging / "validation_results.json").write_text("{}")
        with mock.patch.object(
            go_live,
            "run_invariants_audit",
            return_value={"all_passed": True, "checks": {}},
        ):
            calls, results = self._recorded_run()
        self.assertEqual(
            calls,
            [
                "refresh",
                "fetch-senate-efd",
                "fetch-capitol",
                "analyze",
                "snapshot",
                "purge_phantom_rows.py",
                "purge_phantom_rows.py",
                "validate",
            ],
        )
        self.assertEqual(
            [r.status for r in results.values()],
            ["passed"] * len(go_live.STAGE_ORDER),
        )

    def test_failing_stage_stops_promotion(self):
        calls, results = self._recorded_run(fail_at="fetch-senate-efd")
        # house ran; senate failed; nothing after senate ran
        self.assertEqual(calls, ["refresh", "fetch-senate-efd"])
        self.assertEqual(results["senate"].status, "failed")
        for stage_id in ("capitol", "prices", "invariants", "validation"):
            self.assertEqual(
                results[stage_id].status,
                go_live.STATUS_NOT_RUN,
                f"{stage_id} must not run after a failure",
            )
        payload = go_live._manifest_payload(self.ctx, results)
        self.assertEqual(payload["final_status"], "not_established")

    def test_all_passed_manifest_has_hashes_and_established(self):
        # Provide the artifacts the artifact gates require.
        (self.staging / "capitol_recon.json").write_text("{}")
        (self.staging / "price_snapshot.json").write_text('{"value_hash": "x"}')
        (self.staging / "validation_results.json").write_text("{}")
        with (
            mock.patch.object(go_live, "_execute_command", return_value=0),
            mock.patch.object(
                go_live,
                "run_invariants_audit",
                return_value={"all_passed": True, "checks": {}},
            ),
        ):
            results = go_live.run_stages(self.ctx)
        payload = go_live._manifest_payload(self.ctx, results)
        self.assertEqual(payload["final_status"], "established")
        for stage_id in go_live.STAGE_ORDER:
            self.assertEqual(payload["stages"][stage_id]["status"], "passed")
        self.assertIn(
            "capitol_recon.json",
            payload["stages"]["capitol"]["artifacts_sha256"],
        )
        self.assertEqual(
            len(payload["stages"]["capitol"]["artifacts_sha256"]["capitol_recon.json"]),
            64,
        )
        self.assertIn(
            "price_snapshot.json",
            payload["stages"]["prices"]["artifacts_sha256"],
        )
        self.assertIn(
            "validation_results.json",
            payload["stages"]["validation"]["artifacts_sha256"],
        )
        self.assertIn("generation", payload)
        self.assertIn("git", payload)

    def test_missing_required_artifact_fails_stage(self):
        # No capitol artifact: the capitol stage must fail even with exit 0.
        with mock.patch.object(go_live, "_execute_command", return_value=0):
            results = go_live.run_stages(self.ctx)
        self.assertEqual(results["capitol"].status, "failed")
        self.assertIn("missing_artifact", results["capitol"].detail)
        payload = go_live._manifest_payload(self.ctx, results)
        self.assertEqual(payload["final_status"], "not_established")


class TestDryRun(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.staging = Path(self.tmp) / "drygen"
        self.ctx = make_context(self.staging, dry_run=True)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_dry_run_creates_nothing_and_marks_would_run(self):
        with mock.patch.object(
            go_live,
            "_execute_command",
            side_effect=AssertionError("dry-run must not execute commands"),
        ):
            results = go_live.run_stages(self.ctx)
        self.assertFalse(self.staging.exists())
        for stage_id in go_live.STAGE_ORDER:
            self.assertEqual(results[stage_id].status, go_live.STATUS_WOULD_RUN)
        payload = go_live._manifest_payload(self.ctx, results)
        self.assertEqual(payload["final_status"], "not_established")

    def test_dry_run_main_exits_zero_without_writing_manifest(self):
        exit_code = go_live.main(
            [
                "--dry-run",
                "--staging-root",
                str(Path(self.tmp) / "root"),
                "--generation",
                "drygen",
            ]
        )
        self.assertEqual(exit_code, 0)
        self.assertFalse((Path(self.tmp) / "root").exists())


class TestStagingSafety(unittest.TestCase):
    def test_refuses_real_data_dir(self):
        with self.assertRaises(SystemExit):
            go_live._validate_staging(make_context(go_live._REPO_ROOT / "data"))

    def test_refuses_existing_generation_dir(self):
        tmp = tempfile.mkdtemp()
        try:
            existing = Path(tmp) / "existing"
            existing.mkdir()
            with self.assertRaises(SystemExit):
                go_live._validate_staging(make_context(existing))
        finally:
            shutil.rmtree(tmp)

    def test_accepts_fresh_staging_dir(self):
        tmp = tempfile.mkdtemp()
        try:
            fresh = Path(tmp) / "fresh"
            go_live._validate_staging(make_context(fresh))  # must not raise
            self.assertFalse(fresh.exists())
        finally:
            shutil.rmtree(tmp)


class TestInvariantsAudit(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.staging = Path(self.tmp) / "gen"
        self.staging.mkdir()
        self.ctx = make_context(self.staging)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def _seed_db(self, db_path, *, bad_chronology=False, duplicate_rows=False):
        import pandas as pd

        from analyzer.database import Database

        db_path.parent.mkdir(parents=True, exist_ok=True)
        db = Database(db_path)
        rows = [
            {
                "doc_id": "DOC-1",
                "member": "Alice Smith",
                "ticker": "AAPL",
                "disclosure_date": date(2024, 1, 5),
                "transaction_date": date(2024, 1, 2),
                "transaction_type": "Purchase",
                "source": "house_pdf",
                "chamber": "house",
                "source_record_id": "sr-1",
                "source_row_id": 1,
                "ingestion_generation": "gen-1",
            },
            {
                "doc_id": "DOC-1",
                "member": "Alice Smith",
                "ticker": "MSFT",
                "disclosure_date": date(2024, 1, 5),
                "transaction_date": date(2024, 1, 2),
                "transaction_type": "Purchase",
                "source": "house_pdf",
                "chamber": "house",
                "source_record_id": "sr-2",
                "source_row_id": 2,
                "ingestion_generation": "gen-1",
            },
        ]
        if duplicate_rows:
            # A raw insert with a distinct source identity tuple bypasses the
            # repository dedup while sharing the purge_phantom_rows normalized
            # key (doc_id/ticker/transaction_date/member/transaction_type).
            db.conn.execute(
                """
                INSERT INTO transactions (
                    doc_id, member, ticker, transaction_date, disclosure_date,
                    transaction_type, source, chamber, source_record_id,
                    source_row_id, ingestion_generation
                ) VALUES (?, ?, ?, ?, ?, ?, 'house_pdf', 'house', ?, ?, ?)
                """,
                [
                    "DOC-1",
                    "Alice Smith",
                    "AAPL",
                    date(2024, 1, 2),
                    date(2024, 1, 5),
                    "Purchase",
                    "sr-dup",
                    "999",
                    "gen-1",
                ],
            )
        if bad_chronology:
            rows[0]["transaction_date"] = date(2024, 1, 10)  # after disclosure
        db.upsert_transactions(pd.DataFrame(rows), source="house_pdf")
        # An activated complete generation for the staged house year.
        db.conn.execute(
            "INSERT INTO house_archive_generations "
            "(archive_year, generation_id, parse_status, promoted_at) "
            "VALUES (?, ?, 'complete', CURRENT_TIMESTAMP)",
            [2024, "gen-1"],
        )
        db.close()

    def test_invariants_pass_on_clean_db(self):
        from analyzer.database import Database

        self._seed_db(self.staging / "congress.duckdb")
        # Senate DB is not produced by the house stage; create an empty one so
        # the senate side of the audit has something to open.
        Database(self.staging / "senate" / "congress.duckdb").close()
        audit = go_live.run_invariants_audit(self.ctx)
        self.assertTrue(audit["all_passed"], audit)

    def test_invariants_fail_closed_on_bad_chronology(self):
        self._seed_db(self.staging / "congress.duckdb", bad_chronology=True)
        audit = go_live.run_invariants_audit(self.ctx)
        self.assertFalse(audit["all_passed"])
        self.assertFalse(audit["checks"]["main_no_invalid_chronology"]["passed"])

    def test_invariants_fail_closed_on_phantom_duplicates(self):
        self._seed_db(self.staging / "congress.duckdb", duplicate_rows=True)
        audit = go_live.run_invariants_audit(self.ctx)
        self.assertFalse(audit["all_passed"])
        phantom = audit["checks"].get("phantom_rows_gen")
        self.assertIsNotNone(phantom)
        self.assertFalse(phantom["passed"])
        self.assertNotIn("phantom duplicate rows=0", phantom["detail"])

    def test_invariants_fail_closed_when_db_missing(self):
        audit = go_live.run_invariants_audit(self.ctx)
        self.assertFalse(audit["all_passed"])
        self.assertFalse(audit["checks"]["main_db_exists"]["passed"])


class TestManifestWriting(unittest.TestCase):
    def test_write_manifest_atomic_and_complete(self):
        tmp = tempfile.mkdtemp()
        try:
            staging = Path(tmp) / "gen"
            staging.mkdir()
            ctx = make_context(staging)
            results = {
                sid: go_live.StageResult(stage_id=sid, status="passed")
                for sid in go_live.STAGE_ORDER
            }
            path = go_live.write_manifest(ctx, results)
            payload = json.loads(path.read_text())
            self.assertEqual(payload["generation"], "test-gen")
            self.assertEqual(payload["final_status"], "established")
            self.assertEqual(set(payload["stages"]), set(go_live.STAGE_ORDER))
            self.assertEqual(payload["schema_version"], 1)
            self.assertFalse(list(staging.glob("*.tmp")))
        finally:
            shutil.rmtree(tmp)


if __name__ == "__main__":
    unittest.main()
