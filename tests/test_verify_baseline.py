"""Tests for scripts/verify_baseline.py (the frozen regression baseline gate).

All tests are hermetic: the full suite is never executed and the real
data/congress.duckdb is never read. subprocess-based helpers are
monkeypatched.
"""

import json
from pathlib import Path

from scripts import verify_baseline as vb


def _write_junit(path: Path, cases, suite_time: float = 41.0):
    """cases: list of (classname, name, status) with status in
    passed|failed|error|skipped."""
    body = []
    for index, (classname, name, status) in enumerate(cases):
        if status == "passed":
            inner = ""
        elif status == "failed":
            inner = '<failure message="boom"/>'
        elif status == "error":
            inner = '<error message="boom"/>'
        else:
            inner = '<skipped message="skip"/>'
        body.append(
            f'<testcase classname="{classname}" name="{name}" time="{0.1 * (index + 1)}">'
            f"{inner}</testcase>"
        )
    path.write_text(
        f'<testsuites><testsuite name="pytest" tests="{len(cases)}" '
        f'failures="0" errors="0" skipped="0" time="{suite_time}">'
        f"<properties/>"
        + "".join(body)
        + "</testsuite></testsuites>"
    )


class TestNodeIdMapping:
    def test_class_level(self):
        assert vb.node_id_to_junit_classname(
            "tests/test_parsing.py::TestParsing::test_real_pdf_safety_canaries"
        ) == ("tests.test_parsing.TestParsing", "test_real_pdf_safety_canaries")

    def test_module_level(self):
        assert vb.node_id_to_junit_classname(
            "tests/test_senate_efd_reconciliation.py::test_katie_britt_jpm_canary_preserves_provenance_and_normalizes_sale"
        ) == (
            "tests.test_senate_efd_reconciliation",
            "test_katie_britt_jpm_canary_preserves_provenance_and_normalizes_sale",
        )


class TestParseJunit:
    def test_counts_and_canary_statuses(self, tmp_path):
        xml = tmp_path / "junit.xml"
        _write_junit(
            xml,
            [
                ("tests.test_parsing.TestParsing", "test_real_pdf_safety_canaries", "passed"),
                ("tests.test_parsing.TestParsing", "test_real_option_canary_parses_strike_and_two_digit_expiry", "passed"),
                ("tests.test_integration.TestIntegration", "test_end_to_end_house_analysis", "passed"),
                ("tests.test_parsing.TestParsing", "test_something_else", "failed"),
                ("tests.test_parsing.TestParsing", "test_skipped_thing", "skipped"),
                ("tests.test_parsing.TestParsing", "test_errored_thing", "error"),
            ],
            suite_time=40.5,
        )
        result = vb.parse_junit(xml)
        assert result.duration_s == 40.5
        assert result.counts.collected == 6
        assert result.counts.passed == 3
        assert result.counts.failed == 1
        assert result.counts.errors == 1
        assert result.counts.skipped == 1
        canary = result.canaries[
            "tests/test_parsing.py::TestParsing::test_real_pdf_safety_canaries"
        ]
        assert canary == {"status": "passed", "duration_s": 0.1}

    def test_missing_junit_yields_missing_canaries(self, tmp_path):
        result = vb.parse_junit(tmp_path / "nope.xml")
        assert result.counts.collected == 0
        assert all(
            c["status"] == "missing"
            for c in result.canaries.values()
        )
        assert result.exit_code == -1


class TestParseSummary:
    def test_subtests_and_summary_line(self):
        stdout = (
            "........................................\n"
            "760 passed, 3 skipped, 3 warnings, 79 subtests passed in 41.16s\n"
        )
        line, subtests = vb.parse_summary_line(stdout)
        assert subtests == 79
        assert line == "760 passed, 3 skipped, 3 warnings, 79 subtests passed in 41.16s"

    def test_no_subtests(self):
        stdout = "760 passed, 3 skipped in 41.16s\n"
        line, subtests = vb.parse_summary_line(stdout)
        assert subtests is None
        assert line == "760 passed, 3 skipped in 41.16s"


class TestSha256:
    def test_known_hash(self, tmp_path):
        path = tmp_path / "f.bin"
        path.write_bytes(b"hello baseline")
        assert vb.sha256_file(path) == vb.hashlib.sha256(b"hello baseline").hexdigest()


class TestDbLocation:
    def test_worktree_falls_back_to_primary_repo(self, tmp_path):
        primary = tmp_path / "main"
        worktree = tmp_path / "worktree"
        (primary / "data").mkdir(parents=True)
        (primary / "data" / "congress.duckdb").write_bytes(b"db")
        worktree.mkdir()
        assert vb.locate_db(worktree, primary, None) == primary / "data" / "congress.duckdb"
        assert vb.locate_db(primary, primary, None) == primary / "data" / "congress.duckdb"
        custom = worktree / "custom.duckdb"
        custom.write_bytes(b"custom")
        assert vb.locate_db(worktree, primary, custom) == custom
        # explicit path wins over the primary repo db
        assert vb.locate_db(worktree, primary, custom) == custom

    def test_missing_db(self, tmp_path):
        primary = tmp_path / "main"
        primary.mkdir()
        assert vb.locate_db(primary, primary, None) is None


class TestRevisionCheck:
    def test_exact_match(self, tmp_path, monkeypatch):
        monkeypatch.setattr(vb, "git_revision", lambda root: "abc123")
        ok, detail = vb.revision_check(tmp_path, "abc123", allow_baseline_delta=True)
        assert ok

    def test_baseline_delta_allowed(self, tmp_path, monkeypatch):
        monkeypatch.setattr(vb, "git_revision", lambda root: "def456")
        monkeypatch.setattr(
            vb,
            "git_diff_names",
            lambda root, base: ["scripts/verify_baseline.py", "tests/baseline/golden_manifest.json"],
        )
        ok, detail = vb.revision_check(tmp_path, "abc123", allow_baseline_delta=True)
        assert ok
        assert "baseline tooling" in detail

    def test_production_delta_rejected(self, tmp_path, monkeypatch):
        monkeypatch.setattr(vb, "git_revision", lambda root: "def456")
        monkeypatch.setattr(
            vb,
            "git_diff_names",
            lambda root, base: ["src/analyzer/cli.py"],
        )
        ok, _ = vb.revision_check(tmp_path, "abc123", allow_baseline_delta=True)
        assert not ok

    def test_delta_rejected_when_disallowed(self, tmp_path, monkeypatch):
        monkeypatch.setattr(vb, "git_revision", lambda root: "def456")
        ok, _ = vb.revision_check(tmp_path, "abc123", allow_baseline_delta=False)
        assert not ok


def _env(root, revision=vb.EXPECTED_REVISION, db_sha="deadbeef"):
    return vb.EnvCheck(
        root=Path(root),
        primary=Path(root),
        base_revision=vb.EXPECTED_REVISION,
        revision=revision,
        branch="main",
        tree_clean=True,
        db_path=Path("/fake/data/congress.duckdb"),
        db_sha256=db_sha,
    )


def _manifest(**overrides):
    manifest = {
        "schema_version": 1,
        "git": {
            "base_revision": vb.EXPECTED_REVISION,
            "revision": vb.EXPECTED_REVISION,
            "branch": "main",
            "tree_clean": True,
        },
        "database": {"relative_path": "data/congress.duckdb", "sha256": "deadbeef"},
        "suite": {
            "counts": {
                "collected": 772,
                "passed": 769,
                "failed": 0,
                "errors": 0,
                "skipped": 3,
                "subtests_passed": 79,
            },
            "duration_range_s": [40.0, 42.0],
            "duration_tolerance": 0.25,
            "canaries": {
                "tests/test_parsing.py::TestParsing::test_real_pdf_safety_canaries": {
                    "status": "passed",
                    "duration_s": 4.9,
                }
            },
        },
    }
    manifest.update(overrides)
    return manifest


def _result(**overrides):
    result = vb.SuiteResult(
        duration_s=41.0,
        counts=vb.SuiteCounts(
            collected=772, passed=769, failed=0, errors=0, skipped=3, subtests_passed=79
        ),
        canaries={
            "tests/test_parsing.py::TestParsing::test_real_pdf_safety_canaries": {
                "status": "passed",
                "duration_s": 4.9,
            }
        },
        exit_code=0,
        summary_line="769 passed, 3 skipped, 79 subtests passed in 41.0s",
    )
    for key, value in overrides.items():
        setattr(result, key, value)
    return result


class TestVerify:
    def test_clean_pass(self, tmp_path, monkeypatch):
        monkeypatch.setattr(vb, "git_revision", lambda root: vb.EXPECTED_REVISION)
        problems = vb.verify(_manifest(), _result(), _env(tmp_path), allow_baseline_delta=True)
        assert problems == []

    def test_count_drift(self, tmp_path, monkeypatch):
        monkeypatch.setattr(vb, "git_revision", lambda root: vb.EXPECTED_REVISION)
        result = _result()
        result.counts.passed = 768
        problems = vb.verify(_manifest(), result, _env(tmp_path), True)
        assert any("count gate: passed" in p for p in problems)

    def test_canary_status_drift(self, tmp_path, monkeypatch):
        monkeypatch.setattr(vb, "git_revision", lambda root: vb.EXPECTED_REVISION)
        result = _result()
        result.canaries[
            "tests/test_parsing.py::TestParsing::test_real_pdf_safety_canaries"
        ] = {"status": "failed", "duration_s": 4.9}
        problems = vb.verify(_manifest(), result, _env(tmp_path), True)
        assert any("canary gate" in p for p in problems)

    def test_missing_canary(self, tmp_path, monkeypatch):
        monkeypatch.setattr(vb, "git_revision", lambda root: vb.EXPECTED_REVISION)
        result = _result()
        result.canaries = {}
        problems = vb.verify(_manifest(), result, _env(tmp_path), True)
        assert any("canary gate" in p for p in problems)

    def test_duration_outside_range(self, tmp_path, monkeypatch):
        monkeypatch.setattr(vb, "git_revision", lambda root: vb.EXPECTED_REVISION)
        result = _result(duration_s=60.0)
        problems = vb.verify(_manifest(), result, _env(tmp_path), True)
        assert any("duration gate" in p for p in problems)

    def test_revision_mismatch(self, tmp_path, monkeypatch):
        monkeypatch.setattr(vb, "git_revision", lambda root: "other123")
        problems = vb.verify(_manifest(), _result(), _env(tmp_path), True)
        assert any("revision gate" in p for p in problems)

    def test_db_hash_mismatch(self, tmp_path, monkeypatch):
        monkeypatch.setattr(vb, "git_revision", lambda root: vb.EXPECTED_REVISION)
        problems = vb.verify(_manifest(), _result(), _env(tmp_path, db_sha="cafebabe"), True)
        assert any("database gate" in p for p in problems)

    def test_pytest_exit_code(self, tmp_path, monkeypatch):
        monkeypatch.setattr(vb, "git_revision", lambda root: vb.EXPECTED_REVISION)
        problems = vb.verify(_manifest(), _result(exit_code=1), _env(tmp_path), True)
        assert any("suite gate" in p for p in problems)


class TestRecord:
    def test_record_writes_manifest(self, tmp_path, monkeypatch):
        counter = {"n": 0}

        def fake_suite(root, junit_path):
            counter["n"] += 1
            return _result(duration_s=41.0 + counter["n"])

        monkeypatch.setattr(vb, "run_suite", fake_suite)
        monkeypatch.setattr(vb, "git_tree_clean", lambda root: True)
        monkeypatch.setattr(
            vb, "check_environment",
            lambda root, primary, er, edb, override: (
                _env(tmp_path),
                [],
            ),
        )
        manifest_path = tmp_path / "golden.json"
        args = _args(record=True, record_runs=3, duration_tolerance=0.25)
        args.manifest = manifest_path
        assert vb.cmd_record(args, tmp_path, tmp_path) == 0
        manifest = json.loads(manifest_path.read_text())
        assert manifest["schema_version"] == 1
        assert manifest["git"]["base_revision"] == vb.EXPECTED_REVISION
        assert manifest["git"]["revision"] == vb.EXPECTED_REVISION
        assert manifest["database"]["sha256"] == "deadbeef"
        assert manifest["suite"]["counts"]["collected"] == 772
        assert manifest["suite"]["duration_range_s"][0] < manifest["suite"]["duration_range_s"][1]
        assert (
            manifest["suite"]["canaries"][
                "tests/test_parsing.py::TestParsing::test_real_pdf_safety_canaries"
            ]["status"]
            == "passed"
        )

    def test_record_refuses_on_suite_failure(self, tmp_path, monkeypatch):
        def failing_suite(root, junit_path):
            return _result(exit_code=1)

        monkeypatch.setattr(vb, "run_suite", failing_suite)
        monkeypatch.setattr(vb, "git_tree_clean", lambda root: True)
        monkeypatch.setattr(
            vb, "check_environment",
            lambda root, primary, er, edb, override: (_env(tmp_path), []),
        )
        manifest_path = tmp_path / "golden.json"
        args = _args(record=True)
        args.manifest = manifest_path
        assert vb.cmd_record(args, tmp_path, tmp_path) == 1
        assert not manifest_path.exists()

    def test_record_refuses_on_dirty_env(self, tmp_path, monkeypatch):
        monkeypatch.setattr(vb, "run_suite", lambda root, junit_path: _result())
        monkeypatch.setattr(vb, "git_tree_clean", lambda root: True)
        monkeypatch.setattr(
            vb, "check_environment",
            lambda root, primary, er, edb, override: (
                _env(tmp_path),
                ["revision mismatch: HEAD=xyz expected=a1829c1"],
            ),
        )
        manifest_path = tmp_path / "golden.json"
        args = _args(record=True)
        args.manifest = manifest_path
        assert vb.cmd_record(args, tmp_path, tmp_path) == 1
        assert not manifest_path.exists()


class TestVerifyFlow:
    def test_gate_passes(self, tmp_path, monkeypatch):
        manifest_path = tmp_path / "golden.json"
        manifest_path.write_text(json.dumps(_manifest()))
        monkeypatch.setattr(vb, "run_suite", lambda root, junit_path: _result())
        monkeypatch.setattr(vb, "git_tree_clean", lambda root: True)
        monkeypatch.setattr(vb, "git_revision", lambda root: vb.EXPECTED_REVISION)
        monkeypatch.setattr(
            vb, "check_environment",
            lambda root, primary, er, edb, override: (_env(tmp_path), []),
        )
        args = _args(record=False)
        args.manifest = manifest_path
        assert vb.cmd_verify(args, tmp_path, tmp_path) == 0

    def test_gate_fails_on_drift(self, tmp_path, monkeypatch):
        manifest_path = tmp_path / "golden.json"
        manifest_path.write_text(json.dumps(_manifest()))
        monkeypatch.setattr(vb, "run_suite", lambda root, junit_path: _result(exit_code=1))
        monkeypatch.setattr(vb, "git_tree_clean", lambda root: True)
        monkeypatch.setattr(vb, "git_revision", lambda root: vb.EXPECTED_REVISION)
        monkeypatch.setattr(
            vb, "check_environment",
            lambda root, primary, er, edb, override: (_env(tmp_path), []),
        )
        args = _args(record=False)
        args.manifest = manifest_path
        assert vb.cmd_verify(args, tmp_path, tmp_path) == 1

    def test_missing_manifest(self, tmp_path, monkeypatch):
        args = _args(record=False)
        args.manifest = tmp_path / "nope.json"
        assert vb.cmd_verify(args, tmp_path, tmp_path) == 1


def _args(record: bool, **kwargs):
    args = type("Args", (), {"record": record, "verify": not record})()
    args.record_runs = kwargs.get("record_runs", 3)
    args.duration_tolerance = kwargs.get("duration_tolerance", 0.25)
    args.expect_revision = vb.EXPECTED_REVISION
    args.expect_db_sha256 = vb.EXPECTED_DB_SHA256
    args.manifest = None
    args.db_path = None
    args.allow_baseline_delta = True
    return args
