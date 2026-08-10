#!/usr/bin/env python3
"""Frozen regression baseline for go-live CI gating.

Runs the integrated full pytest suite and records a golden manifest
(test counts, suite duration range, key canary results, git revision,
real database hash). `--verify` gates the current tree against that
manifest; any drift fails the gate with exit code 1.

Environment invariants enforced in both modes:
  * HEAD sits on the frozen revision 6226675f36a5b7db060efa2d8ec9eedb50432dcb
    (or differs from it only by the baseline tooling files themselves).
  * The working tree is clean before and after the suite run (main stays
    clean at the frozen revision).
  * data/congress.duckdb SHA-256 is unchanged
    (9ec6be9263dc30aab07585d0110d2daf8568a14e4244f39d07c5b2bc130d476d).

Usage:
  scripts/verify_baseline.py --record    # run suite N times, write golden manifest
  scripts/verify_baseline.py --verify    # run suite once, gate against manifest (default)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 1

EXPECTED_REVISION = "6226675f36a5b7db060efa2d8ec9eedb50432dcb"
EXPECTED_DB_SHA256 = "9ec6be9263dc30aab07585d0110d2daf8568a14e4244f39d07c5b2bc130d476d"

MANIFEST_REL_PATH = Path("tests/baseline/golden_manifest.json")
DB_REL_PATH = Path("data/congress.duckdb")

# Baseline tooling files that may legitimately ride on top of the frozen
# revision: their presence alone never fails the revision gate.
ALLOWED_BASELINE_FILES = frozenset(
    {
        "scripts/verify_baseline.py",
        "tests/test_verify_baseline.py",
        str(MANIFEST_REL_PATH),
    }
)

# Key canary tests whose individual outcome is recorded in the golden manifest.
CANARY_TESTS = [
    "tests/test_integration.py::TestIntegration::test_end_to_end_house_analysis",
    "tests/test_parsing.py::TestParsing::test_real_option_canary_parses_strike_and_two_digit_expiry",
    "tests/test_parsing.py::TestParsing::test_real_pdf_safety_canaries",
    "tests/test_senate_efd_reconciliation.py::test_katie_britt_jpm_canary_preserves_provenance_and_normalizes_sale",
    "tests/test_senate_efd_reconciliation.py::test_rick_scott_coupon_canary_is_non_equity_not_a_ticker",
    "tests/test_validation.py::TestNeweyWest::test_zero_alpha_canary_is_exactly_null",
    "tests/test_validation.py::TestCorrectedSelection::test_all_zero_canary_has_no_deployable_config",
    "tests/test_validation.py::TestMemberPermutationCanary::test_member_label_permutation_is_bijective_and_preserves_values",
]

DEFAULT_RECORD_RUNS = 3
DEFAULT_DURATION_TOLERANCE = 0.25
SUITE_TIMEOUT_S = 1800


@dataclass
class SuiteCounts:
    collected: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    subtests_passed: int | None = None

    def as_dict(self) -> dict:
        d = {
            "collected": self.collected,
            "passed": self.passed,
            "failed": self.failed,
            "errors": self.errors,
            "skipped": self.skipped,
        }
        if self.subtests_passed is not None:
            d["subtests_passed"] = self.subtests_passed
        return d


@dataclass
class SuiteResult:
    duration_s: float
    counts: SuiteCounts
    canaries: dict[str, dict]
    exit_code: int = 0
    summary_line: str | None = None


@dataclass
class EnvCheck:
    root: Path
    primary: Path
    base_revision: str
    revision: str
    branch: str
    tree_clean: bool
    db_path: Path | None
    db_sha256: str | None


# ---------------------------------------------------------------------------
# git / filesystem helpers
# ---------------------------------------------------------------------------


def git(cwd: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args], cwd=str(cwd), capture_output=True, text=True, check=True
    )
    return proc.stdout.strip()


def repo_roots(cwd: Path) -> tuple[Path, Path]:
    """Return (toplevel, primary_repo_root)."""
    toplevel = Path(git(cwd, "rev-parse", "--show-toplevel")).resolve()
    common = Path(git(cwd, "rev-parse", "--git-common-dir")).resolve()
    primary = common.parent
    return toplevel, primary


def db_candidates(
    toplevel: Path, primary: Path, explicit: Path | None
) -> list[Path]:
    candidates = []
    if explicit is not None:
        candidates.append(explicit)
    candidates.append(primary / DB_REL_PATH)
    candidates.append(toplevel / DB_REL_PATH)
    return candidates


def locate_db(toplevel: Path, primary: Path, explicit: Path | None) -> Path | None:
    for candidate in db_candidates(toplevel, primary, explicit):
        if candidate.is_file():
            return candidate
    return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(root: Path) -> str:
    return git(root, "rev-parse", "HEAD")


def git_branch(root: Path) -> str:
    branch = git(root, "rev-parse", "--abbrev-ref", "HEAD")
    return branch if branch and branch != "HEAD" else "detached"


def git_tree_clean(root: Path) -> bool:
    return git(root, "status", "--porcelain") == ""


def git_diff_names(root: Path, base_rev: str) -> list[str]:
    """Files whose content differs between base_rev and HEAD (repo-relative)."""
    return git(root, "diff", "--name-only", base_rev, "HEAD").splitlines()


def revision_check(root: Path, recorded_rev: str, allow_baseline_delta: bool) -> tuple[bool, str]:
    head = git_revision(root)
    if head == recorded_rev:
        return True, f"HEAD {head} matches frozen revision"
    if allow_baseline_delta:
        try:
            diff = sorted(set(git_diff_names(root, recorded_rev)))
        except subprocess.CalledProcessError:
            return (
                False,
                f"HEAD {head} is not the frozen revision {recorded_rev} "
                f"(cannot diff against {recorded_rev})",
            )
        if diff and set(diff) <= set(ALLOWED_BASELINE_FILES):
            return (
                True,
                f"HEAD {head} differs from {recorded_rev} only by baseline tooling: {diff}",
            )
        if not diff:
            return True, f"HEAD {head} tree identical to {recorded_rev}"
    return False, f"HEAD {head} is not the frozen revision {recorded_rev}"


# ---------------------------------------------------------------------------
# pytest junit parsing
# ---------------------------------------------------------------------------


def node_id_to_junit_classname(node_id: str) -> tuple[str, str]:
    """Map 'tests/test_x.py::Class::test_y' to junit (classname, name)."""
    parts = node_id.split("::")
    module = parts[0][:-3].replace("/", ".")
    classname = module if len(parts) == 2 else f"{module}.{parts[1]}"
    return classname, parts[-1]


def _case_status(case: ET.Element) -> str:
    if case.find("failure") is not None:
        return "failed"
    if case.find("error") is not None:
        return "error"
    if case.find("skipped") is not None:
        return "skipped"
    return "passed"


def parse_junit(xml_path: Path) -> SuiteResult:
    if not xml_path.is_file():
        return SuiteResult(
            duration_s=0.0,
            counts=SuiteCounts(),
            canaries={node: {"status": "missing", "duration_s": None} for node in CANARY_TESTS},
            exit_code=-1,
            summary_line="junit xml missing",
        )
    root = ET.parse(xml_path).getroot()
    if root.tag == "testsuites":
        suite = root.find("testsuite")
    else:
        suite = root
    cases = list(suite.iter("testcase")) if suite is not None else []
    skipped = [c for c in cases if c.find("skipped") is not None]
    failed = [c for c in cases if c.find("failure") is not None]
    errors = [c for c in cases if c.find("error") is not None]
    counts = SuiteCounts(
        collected=len(cases),
        passed=len(cases) - len(skipped) - len(failed) - len(errors),
        failed=len(failed),
        errors=len(errors),
        skipped=len(skipped),
    )
    canaries: dict[str, dict] = {}
    for node in CANARY_TESTS:
        expected_classname, expected_name = node_id_to_junit_classname(node)
        match = next(
            (
                c
                for c in cases
                if c.attrib.get("classname") == expected_classname
                and c.attrib.get("name") == expected_name
            ),
            None,
        )
        if match is None:
            canaries[node] = {"status": "missing", "duration_s": None}
        else:
            canaries[node] = {
                "status": _case_status(match),
                "duration_s": float(match.attrib.get("time", 0) or 0),
            }
    duration_s = float(suite.attrib.get("time", 0)) if suite is not None else 0.0
    return SuiteResult(duration_s=duration_s, counts=counts, canaries=canaries)


def parse_summary_line(stdout: str) -> tuple[str | None, int | None]:
    subtests = None
    match = re.search(r"(\d+)\s+subtests?\s+passed", stdout)
    if match:
        subtests = int(match.group(1))
    summary_line = None
    for line in stdout.splitlines():
        if re.search(r"in\s+[\d.]+\s*s", line) and any(
            token in line for token in ("passed", "failed", "no tests")
        ):
            summary_line = line.strip()
    return summary_line, subtests


def run_suite(root: Path, junit_path: Path, timeout: float = SUITE_TIMEOUT_S) -> SuiteResult:
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/",
        "-q",
        "-p",
        "no:cacheprovider",
        "--junitxml",
        str(junit_path),
    ]
    try:
        proc = subprocess.run(cmd, cwd=str(root), capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        result = SuiteResult(
            duration_s=timeout,
            counts=SuiteCounts(),
            canaries={node: {"status": "missing", "duration_s": None} for node in CANARY_TESTS},
            exit_code=2,
            summary_line=f"pytest timed out after {timeout:.0f}s",
        )
        return result
    result = parse_junit(junit_path)
    result.exit_code = proc.returncode
    summary_line, subtests = parse_summary_line(proc.stdout)
    result.summary_line = summary_line
    if subtests is not None:
        result.counts.subtests_passed = subtests
    return result


# ---------------------------------------------------------------------------
# environment checks
# ---------------------------------------------------------------------------


def check_environment(
    root: Path,
    primary: Path,
    expect_revision: str,
    expect_db_sha256: str,
    db_override: Path | None,
) -> tuple[EnvCheck, list[str]]:
    problems: list[str] = []
    revision = git_revision(root)
    branch = git_branch(root)
    tree_clean = git_tree_clean(root)
    db_path = locate_db(root, primary, db_override)
    db_sha256 = sha256_file(db_path) if db_path is not None else None

    ok, detail = revision_check(root, expect_revision, allow_baseline_delta=True)
    if not ok:
        problems.append(f"revision gate: {detail}")
    if not tree_clean:
        problems.append("working tree is not clean before suite run")
    if db_path is None:
        problems.append(f"real database not found: {DB_REL_PATH}")
    elif db_sha256 != expect_db_sha256:
        problems.append(
            f"real database hash mismatch: {db_sha256} expected {expect_db_sha256}"
        )
    return (
        EnvCheck(
            root=root,
            primary=primary,
            base_revision=expect_revision,
            revision=revision,
            branch=branch,
            tree_clean=tree_clean,
            db_path=db_path,
            db_sha256=db_sha256,
        ),
        problems,
    )


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------


def build_manifest(env: EnvCheck, results: list[SuiteResult], tolerance: float) -> dict:
    durations = [r.duration_s for r in results]
    final = results[-1]
    return {
        "schema_version": SCHEMA_VERSION,
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": {
            "base_revision": env.base_revision,
            "revision": env.revision,
            "branch": env.branch,
            "tree_clean": env.tree_clean,
        },
        "database": {
            "relative_path": str(DB_REL_PATH),
            "sha256": env.db_sha256,
        },
        "suite": {
            "counts": final.counts.as_dict(),
            "duration_range_s": [min(durations), max(durations)],
            "duration_tolerance": tolerance,
            "canaries": final.canaries,
        },
    }


def write_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".golden-", suffix=".json")
    try:
        with open(fd, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_name, str(path))
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def load_manifest(path: Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


# ---------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------


def verify(
    manifest: dict,
    result: SuiteResult,
    env: EnvCheck,
    allow_baseline_delta: bool,
) -> list[str]:
    problems: list[str] = []

    expected_revision = manifest["git"]["base_revision"]
    ok, detail = revision_check(env.root, expected_revision, allow_baseline_delta)
    if not ok:
        problems.append(f"revision gate: {detail}")

    expected_db = manifest["database"]["sha256"]
    if env.db_sha256 is None:
        problems.append("database gate: real database not found")
    elif env.db_sha256 != expected_db:
        problems.append(
            f"database gate: hash {env.db_sha256} != golden {expected_db}"
        )

    if result.exit_code != 0:
        problems.append(f"suite gate: pytest exited {result.exit_code}")

    expected_counts = manifest["suite"]["counts"]
    counts = result.counts
    for key in ("collected", "passed", "failed", "errors", "skipped"):
        observed = getattr(counts, key)
        expected = expected_counts.get(key, 0)
        if observed != expected:
            problems.append(
                f"count gate: {key} observed={observed} golden={expected}"
            )

    golden_canaries = manifest["suite"]["canaries"]
    if set(golden_canaries) != set(result.canaries):
        problems.append(
            "canary gate: canary set changed "
            f"(golden={sorted(golden_canaries)}, observed={sorted(result.canaries)})"
        )
    for node, golden in golden_canaries.items():
        observed = result.canaries.get(node)
        if observed is None or observed["status"] != golden["status"]:
            problems.append(
                f"canary gate: {node} observed={observed} golden={golden}"
            )

    low, high = manifest["suite"]["duration_range_s"]
    tolerance = manifest["suite"]["duration_tolerance"]
    lo, hi = low * (1 - tolerance), high * (1 + tolerance)
    if not (lo <= result.duration_s <= hi):
        problems.append(
            f"duration gate: suite {result.duration_s:.1f}s outside "
            f"[{lo:.1f}s, {hi:.1f}s] (golden {low:.1f}s-{high:.1f}s, tol {tolerance:.0%})"
        )
    return problems


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_env(env: EnvCheck) -> None:
    print(f"  repo root:      {env.root}")
    print(f"  primary root:   {env.primary}")
    print(f"  revision:       {env.revision}")
    print(f"  branch:         {env.branch}")
    print(f"  tree clean:     {'yes' if env.tree_clean else 'NO'}")
    db = env.db_path if env.db_path is not None else f"missing ({DB_REL_PATH})"
    print(f"  database:       {db}")
    print(f"  database sha256:{' ' + env.db_sha256 if env.db_sha256 else ''}")


def _print_result(result: SuiteResult, index: int, total: int) -> None:
    summary = result.summary_line or "no pytest summary"
    print(
        f"  run {index}/{total}: exit={result.exit_code} junit={result.duration_s:.1f}s "
        f"-> {summary}"
    )


def cmd_record(args: argparse.Namespace, root: Path, primary: Path) -> int:
    manifest_path = args.manifest or root / MANIFEST_REL_PATH
    env, problems = check_environment(
        root, primary, args.expect_revision, args.expect_db_sha256, args.db_path
    )
    print("== baseline environment ==")
    _print_env(env)
    if problems:
        print("ENVIRONMENT CHECK FAILED:")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print(f"== running integrated suite ({args.record_runs} runs) ==")
    results: list[SuiteResult] = []
    with tempfile.TemporaryDirectory(prefix="baseline-") as tmp:
        for i in range(args.record_runs):
            junit_path = Path(tmp) / f"run-{i}.xml"
            result = run_suite(root, junit_path)
            results.append(result)
            _print_result(result, i + 1, args.record_runs)
            if result.exit_code != 0:
                print("RECORD FAILED: suite did not pass; no manifest written")
                return 1

    if not git_tree_clean(root):
        print("RECORD FAILED: working tree is dirty after suite run")
        return 1

    manifest = build_manifest(env, results, args.duration_tolerance)
    write_manifest(manifest_path, manifest)
    print(f"== golden manifest recorded ==")
    print(f"  path:          {manifest_path}")
    print(f"  base revision: {manifest['git']['base_revision']}")
    print(f"  revision:      {manifest['git']['revision']}")
    print(f"  db sha256: {manifest['database']['sha256']}")
    print(f"  counts:    {manifest['suite']['counts']}")
    print(
        f"  duration:  {manifest['suite']['duration_range_s'][0]:.1f}s - "
        f"{manifest['suite']['duration_range_s'][1]:.1f}s "
        f"(tolerance {args.duration_tolerance:.0%})"
    )
    for node, canary in manifest["suite"]["canaries"].items():
        print(f"  canary:    {node} -> {canary['status']}")
    return 0


def cmd_verify(args: argparse.Namespace, root: Path, primary: Path) -> int:
    manifest_path = args.manifest or root / MANIFEST_REL_PATH
    if not manifest_path.is_file():
        print(f"VERIFY FAILED: golden manifest not found at {manifest_path}")
        print("  run `scripts/verify_baseline.py --record` first")
        return 1
    manifest = load_manifest(manifest_path)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        print(
            f"VERIFY FAILED: manifest schema {manifest.get('schema_version')} "
            f"!= script schema {SCHEMA_VERSION}"
        )
        return 1

    env, problems = check_environment(
        root, primary, args.expect_revision, args.expect_db_sha256, args.db_path
    )
    print("== baseline environment ==")
    _print_env(env)
    if problems:
        print("ENVIRONMENT CHECK FAILED:")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print("== running integrated suite ==")
    with tempfile.TemporaryDirectory(prefix="baseline-") as tmp:
        junit_path = Path(tmp) / "run.xml"
        result = run_suite(root, junit_path)
    _print_result(result, 1, 1)

    problems = verify(manifest, result, env, args.allow_baseline_delta)
    if not git_tree_clean(root):
        problems.append("working tree is dirty after suite run")

    if problems:
        print("BASELINE GATE: FAIL")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("BASELINE GATE: PASS")
    print(f"  counts:    {result.counts.as_dict()}")
    print(f"  duration:  {result.duration_s:.1f}s")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="verify_baseline.py",
        description="Frozen regression baseline gate for go-live.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--record", action="store_true", help="record the golden manifest")
    mode.add_argument(
        "--verify",
        action="store_true",
        help="verify the current tree against the golden manifest (default)",
    )
    parser.add_argument(
        "--record-runs",
        type=int,
        default=DEFAULT_RECORD_RUNS,
        help=f"suite runs used to establish the duration range (default {DEFAULT_RECORD_RUNS})",
    )
    parser.add_argument(
        "--duration-tolerance",
        type=float,
        default=DEFAULT_DURATION_TOLERANCE,
        help=f"fractional tolerance around the recorded duration range (default {DEFAULT_DURATION_TOLERANCE})",
    )
    parser.add_argument(
        "--expect-revision",
        default=EXPECTED_REVISION,
        help=f"frozen revision the record/verify run must sit on (default {EXPECTED_REVISION[:12]})",
    )
    parser.add_argument(
        "--expect-db-sha256",
        default=EXPECTED_DB_SHA256,
        help="expected SHA-256 of data/congress.duckdb",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help=f"golden manifest path (default <repo>/tests/baseline/golden_manifest.json)",
    )
    parser.add_argument("--db-path", type=Path, default=None, help="path to data/congress.duckdb")
    parser.add_argument(
        "--no-baseline-delta",
        action="store_false",
        dest="allow_baseline_delta",
        help="require HEAD to be exactly the frozen revision (no tooling delta allowance)",
    )
    parser.set_defaults(allow_baseline_delta=True)
    args = parser.parse_args(argv)

    try:
        root, primary = repo_roots(Path.cwd())
    except subprocess.CalledProcessError as exc:
        print(f"not inside a git repository: {exc.stderr.strip()}")
        return 2

    try:
        if args.record:
            return cmd_record(args, root, primary)
        return cmd_verify(args, root, primary)
    except KeyboardInterrupt:
        print("interrupted")
        return 130



if __name__ == "__main__":
    sys.exit(main())
