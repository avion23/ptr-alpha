#!/usr/bin/env python3
"""Reproducible go-live orchestrator for the congressional PTR generation.

Runs the six accepted production stages in a fixed order with fail-closed
gating against an isolated staging root (``--staging-root``). Every stage
invokes existing merged CLIs/scripts with exact arguments; a non-zero exit
code stops promotion immediately and the generation manifest records the
failure. The real ``data/`` database is never opened: the staging DB is
always ``<staging-root>/<generation>/congress.duckdb``.

Stages (order is a hard contract):

1. house   - ``ptr-alpha refresh --year <YEAR> --data-dir <STAGING>
             --gemini-ocr``: fetch House Clerk PDFs, parse via the
             production cascade, run Gemini OCR on zero-row PDFs, and
             activate the archive generation only when nothing is unresolved.
2. senate  - ``ptr-alpha fetch-senate-efd --start <S> --end <E>
             --data-dir <STAGING>/senate``: official eFD sweep into an
             isolated senate DB (chamber separation is exact).
3. capitol - ``ptr-alpha fetch-capitol --all --output <STAGING>/capitol_recon.json
             --generation <GEN> --data-dir <STAGING>``: third-party
             reconciliation artifact; canonical rows are never written.
4. prices  - ``ptr-alpha analyze --year <YEAR> --mode ranks --data-dir <STAGING>``
             (acquisition through the merged YFinancePriceSource pipeline)
             then ``ptr-alpha snapshot --data-dir <STAGING> --output
             <STAGING>/price_snapshot.json`` (value-hashed frozen prices).
5. invariants - read-only audit of the staged DBs through the merged
             ``analyzer.database`` schema plus the merged
             ``scripts/purge_phantom_rows.py`` dry-run (phantom duplicates
             must be zero). No merged CLI exposes DB invariants, so the
             checks run inline against the merged schema.
6. validation - ``ptr-alpha validate --train-start 2022-01-01 --train-end
             2023-12-31 --test-start 2024-01-01 --test-end 2025-06-30
             --data-dir <STAGING>``: purged retrospective harness with the
             canonical append-only evaluation ledger.

The generation manifest (<STAGING>/<GENERATION>/manifest.json) records every
stage's exact command, exit code, duration, produced-artifact SHA-256 hashes,
and the audit checks. The final status is ``not_established`` unless every
stage gate passes, in which case it is ``established``; no profitability
claim is ever emitted here.

Usage:
    python3 scripts/go_live.py [--staging-root DIR] [--generation ID]
        [--house-year YYYY] [--senate-start YYYY-MM-DD]
        [--senate-end YYYY-MM-DD] [--dry-run] [--verbose]
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"

# Runs the merged CLI from this repository's own tree (PYTHONPATH is set to
# <repo>/src:<repo>), so the executed code is exactly the merged revision the
# worktree is on; an installed console script could point at another tree.
_CLI_RUNNER = "import sys; from analyzer.cli import main; sys.exit(main())"

STAGE_ORDER = ("house", "senate", "capitol", "prices", "invariants", "validation")

# Merged ptr-alpha validate default windows (README + cli.py defaults).
DEFAULT_TRAIN_START = _dt.date(2022, 1, 1)
DEFAULT_TRAIN_END = _dt.date(2023, 12, 31)
DEFAULT_TEST_START = _dt.date(2024, 1, 1)
DEFAULT_TEST_END = _dt.date(2025, 6, 30)

STATUS_PASSED = "passed"
STATUS_FAILED = "failed"
STATUS_NOT_RUN = "not_run"
STATUS_WOULD_RUN = "would_run"
FINAL_NOT_ESTABLISHED = "not_established"
FINAL_ESTABLISHED = "established"


@dataclass(frozen=True)
class Context:
    repo_root: Path
    staging_dir: Path
    generation: str
    house_year: int
    senate_start: _dt.date
    senate_end: _dt.date
    train_start: _dt.date
    train_end: _dt.date
    test_start: _dt.date
    test_end: _dt.date
    dry_run: bool
    verbose: bool
    runner: tuple[str, ...] = ("python3", "-c", _CLI_RUNNER)
    env: dict = field(default_factory=dict)


@dataclass
class StageResult:
    stage_id: str
    status: str
    commands: list[list[str]] = field(default_factory=list)
    exit_codes: list[int] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)
    duration_seconds: float = 0.0
    detail: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Command construction (pure; exact merged CLI args)
# --------------------------------------------------------------------------- #


def build_house_commands(ctx: Context) -> list[list[str]]:
    return [
        [
            *ctx.runner,
            "refresh",
            "--year",
            str(ctx.house_year),
            "--data-dir",
            str(ctx.staging_dir),
            "--gemini-ocr",
        ]
    ]


def build_senate_commands(ctx: Context) -> list[list[str]]:
    return [
        [
            *ctx.runner,
            "fetch-senate-efd",
            "--start",
            ctx.senate_start.isoformat(),
            "--end",
            ctx.senate_end.isoformat(),
            "--data-dir",
            str(ctx.staging_dir / "senate"),
        ]
    ]


def build_capitol_commands(ctx: Context) -> list[list[str]]:
    return [
        [
            *ctx.runner,
            "fetch-capitol",
            "--all",
            "--output",
            str(ctx.staging_dir / "capitol_recon.json"),
            "--generation",
            ctx.generation,
            "--data-dir",
            str(ctx.staging_dir),
        ]
    ]


def build_prices_commands(ctx: Context) -> list[list[str]]:
    # Acquisition through the merged analysis pipeline (YFinancePriceSource
    # refreshes missing prices into the staging DB cache), then freeze the
    # value-hashed snapshot artifact.
    return [
        [
            *ctx.runner,
            "analyze",
            "--year",
            str(ctx.house_year),
            "--mode",
            "ranks",
            "--data-dir",
            str(ctx.staging_dir),
        ],
        [
            *ctx.runner,
            "snapshot",
            "--data-dir",
            str(ctx.staging_dir),
            "--output",
            str(ctx.staging_dir / "price_snapshot.json"),
        ],
    ]


def build_validation_commands(ctx: Context) -> list[list[str]]:
    return [
        [
            *ctx.runner,
            "validate",
            "--train-start",
            ctx.train_start.isoformat(),
            "--train-end",
            ctx.train_end.isoformat(),
            "--test-start",
            ctx.test_start.isoformat(),
            "--test-end",
            ctx.test_end.isoformat(),
            "--data-dir",
            str(ctx.staging_dir),
        ]
    ]


def build_stage_commands(ctx: Context) -> dict[str, list[list[str]]]:
    return {
        "house": build_house_commands(ctx),
        "senate": build_senate_commands(ctx),
        "capitol": build_capitol_commands(ctx),
        "prices": build_prices_commands(ctx),
        "invariants": _build_invariants_commands(ctx),
        "validation": build_validation_commands(ctx),
    }


def _build_invariants_commands(ctx: Context) -> list[list[str]]:
    # The merged script dry-runs the phantom-duplicate invariant without
    # taking DuckDB's exclusive write lock (read_only=True without --execute).
    return [
        [
            sys.executable,
            str(_REPO_ROOT / "scripts" / "purge_phantom_rows.py"),
            str(ctx.staging_dir / "congress.duckdb"),
        ],
        [
            sys.executable,
            str(_REPO_ROOT / "scripts" / "purge_phantom_rows.py"),
            str(ctx.staging_dir / "senate" / "congress.duckdb"),
        ],
    ]


# --------------------------------------------------------------------------- #
# Artifacts and hashing
# --------------------------------------------------------------------------- #


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stage_artifact_paths(stage_id: str, ctx: Context) -> list[Path]:
    staging = ctx.staging_dir
    if stage_id == "house":
        return sorted(staging.glob(f"{ctx.house_year}/pdfs/*.pdf"))
    if stage_id == "senate":
        senate_dir = staging / "senate"
        if not senate_dir.exists():
            return []
        return sorted(
            p
            for p in senate_dir.rglob("*")
            if p.is_file() and p.suffix not in (".tmp", ".wal")
        )
    if stage_id == "capitol":
        return [staging / "capitol_recon.json"]
    if stage_id == "prices":
        return [staging / "price_snapshot.json"]
    if stage_id == "invariants":
        dbs = [staging / "congress.duckdb", staging / "senate" / "congress.duckdb"]
        return [p for p in dbs if p.exists()]
    if stage_id == "validation":
        paths = [staging / "validation_results.json"]
        ledger = staging / ".ptr-alpha-evaluation-ledger-v2.json"
        if ledger.exists():
            paths.append(ledger)
        return paths
    return []


def _hash_artifacts(stage_id: str, ctx: Context) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in _stage_artifact_paths(stage_id, ctx):
        if path.is_file():
            hashes[str(path.relative_to(ctx.staging_dir))] = _sha256_file(path)
    return hashes


# --------------------------------------------------------------------------- #
# Invariants audit (inline against the merged schema; no merged CLI exists)
# --------------------------------------------------------------------------- #


def _run_phantom_check(ctx: Context, db_path: Path, checks: dict) -> None:
    name = f"phantom_rows_{db_path.parent.name or 'staging'}"
    command = [
        sys.executable,
        str(_REPO_ROOT / "scripts" / "purge_phantom_rows.py"),
        str(db_path),
    ]
    try:
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            cwd=ctx.repo_root,
            env=ctx.env,
        )
    except OSError as exc:
        checks[name] = {"passed": False, "detail": f"exec failed: {exc}"}
        return
    if proc.returncode != 0:
        checks[name] = {
            "passed": False,
            "detail": f"exit={proc.returncode}: {proc.stderr.strip()[:300]}",
        }
        return
    total = None
    for line in proc.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("total:"):
            total = stripped.split(":", 1)[1].strip()
    if total is None:
        checks[name] = {
            "passed": False,
            "detail": f"unparsed output: {proc.stdout.strip()[:300]}",
        }
        return
    checks[name] = {
        "passed": total == "0",
        "detail": f"phantom duplicate rows={total}",
    }


def _db_checks(ctx: Context, checks: dict) -> None:
    from analyzer.database import Database

    main_db = ctx.staging_dir / "congress.duckdb"
    senate_db = ctx.staging_dir / "senate" / "congress.duckdb"

    def audit(db_path: Path, label: str, is_main: bool) -> None:
        if not db_path.exists():
            checks[f"{label}_db_exists"] = {
                "passed": False,
                "detail": f"missing: {db_path}",
            }
            return
        try:
            db = Database(db_path, read_only=True)
        except Exception as exc:  # DatabaseError from a corrupt/missing file
            checks[f"{label}_db_open"] = {"passed": False, "detail": str(exc)[:300]}
            return
        try:
            conn = db.conn

            def check(name: str, ok: bool, detail: str) -> None:
                checks[f"{label}_{name}"] = {"passed": bool(ok), "detail": detail}

            # House-generation checks apply only to the main (house) staging
            # DB; the isolated senate DB legitimately has no house generations.
            if is_main:
                # Parse-run counts must equal persisted rows per generation.
                rows = conn.execute(
                    """
                    SELECT p.ingestion_generation, COUNT(*)
                    FROM pdf_parse_runs p
                    WHERE p.status = 'success'
                      AND p.transaction_count != (
                          SELECT COUNT(*) FROM transactions t
                          WHERE t.doc_id = p.doc_id
                            AND t.source IN ('house_pdf', 'gemini_ocr')
                            AND t.ingestion_generation = p.ingestion_generation
                      )
                    GROUP BY p.ingestion_generation
                    """
                ).fetchall()
                check(
                    "house_parse_count_equals_persisted_rows",
                    not rows,
                    f"mismatched_generations={[(r[0], r[1]) for r in rows][:10]}",
                )

                # A complete generation must have no unresolved artifacts.
                unresolved = conn.execute(
                    """
                    SELECT archive_year, generation_id
                    FROM house_archive_generations
                    WHERE parse_status = 'complete'
                      AND (
                          SELECT COUNT(*) FROM house_archive_quarantine q
                          WHERE q.archive_year = house_archive_generations.archive_year
                            AND q.generation_id = house_archive_generations.generation_id
                      ) > 0
                    """
                ).fetchall()
                check(
                    "house_complete_generation_has_no_unresolved",
                    not unresolved,
                    f"unresolved={[(r[0], r[1]) for r in unresolved][:10]}",
                )

                # Canonical view must include every row of the active house
                # generation (a complete generation is fully visible).
                active = conn.execute(
                    """
                    SELECT generation_id FROM house_archive_generations
                    WHERE archive_year = ? AND parse_status = 'complete'
                    ORDER BY promoted_at DESC, generation_id DESC LIMIT 1
                    """,
                    [ctx.house_year],
                ).fetchone()
                if active is None:
                    check(
                        "house_generation_active",
                        False,
                        f"no complete generation for {ctx.house_year}",
                    )
                else:
                    generation_id = active[0]
                    raw_count = conn.execute(
                        "SELECT COUNT(*) FROM transactions "
                        "WHERE ingestion_generation = ?",
                        [generation_id],
                    ).fetchone()[0]
                    canonical_count = conn.execute(
                        "SELECT COUNT(*) FROM canonical_transactions "
                        "WHERE ingestion_generation = ?",
                        [generation_id],
                    ).fetchone()[0]
                    check(
                        "canonical_house_generation_visible",
                        int(raw_count) == int(canonical_count),
                        f"generation={generation_id} raw={raw_count} canonical={canonical_count}",
                    )

            # Chronology: transaction never after disclosure; dates never null.
            bad_chronology = conn.execute(
                """
                SELECT COUNT(*) FROM transactions
                WHERE transaction_date IS NOT NULL AND disclosure_date IS NOT NULL
                  AND transaction_date > disclosure_date
                """
            ).fetchone()[0]
            check(
                "no_invalid_chronology",
                int(bad_chronology) == 0,
                f"rows={bad_chronology}",
            )
            null_dates = conn.execute(
                """
                SELECT COUNT(*) FROM transactions
                WHERE transaction_date IS NULL OR disclosure_date IS NULL
                """
            ).fetchone()[0]
            check(
                "no_null_transaction_or_disclosure_dates",
                int(null_dates) == 0,
                f"rows={null_dates}",
            )

            # Source identity tuple must be unique.
            duplicates = conn.execute(
                """
                SELECT source, chamber, source_record_id, source_row_id,
                       ingestion_generation, COUNT(*) AS n
                FROM transactions
                WHERE source_record_id IS NOT NULL
                GROUP BY source, chamber, source_record_id, source_row_id,
                         ingestion_generation
                HAVING COUNT(*) > 1
                """
            ).fetchall()
            check(
                "duplicate_source_identity_policy",
                not duplicates,
                f"duplicate_keys={[(r[0], r[1], r[2], r[3], r[4], r[5]) for r in duplicates][:10]}",
            )

            # source_row_id must be distinct per doc.
            dup_row_ids = conn.execute(
                """
                SELECT doc_id, source_row_id, COUNT(*) AS n
                FROM transactions
                WHERE source_row_id IS NOT NULL
                GROUP BY doc_id, source_row_id
                HAVING COUNT(*) > 1
                """
            ).fetchall()
            check(
                "source_row_id_distinct_per_doc",
                not dup_row_ids,
                f"dups={[(r[0], r[1], r[2]) for r in dup_row_ids][:10]}",
            )

            # Senate report equation: found == parsed + paper_only + unavailable
            # + failed, and none failed or unavailable.
            for gen in [
                row[0]
                for row in conn.execute(
                    "SELECT DISTINCT ingestion_generation FROM source_reports "
                    "WHERE source = 'senate_efd'"
                ).fetchall()
            ]:
                eq = db.source_reports.reconcile(gen, "senate_efd", "senate")
                ok = (
                    (
                        eq["found"]
                        == eq["parsed"]
                        + eq["paper_only"]
                        + eq["unavailable"]
                        + eq["failed"]
                    )
                    and eq["failed"] == 0
                    and eq["unavailable"] == 0
                )
                check(
                    f"report_equation_senate_efd_{gen}",
                    ok,
                    f"reconcile={eq}",
                )
        finally:
            db.close()

    audit(main_db, "main", is_main=True)
    audit(senate_db, "senate", is_main=False)


def run_invariants_audit(ctx: Context) -> dict:
    checks: dict = {}
    for db_path in (
        ctx.staging_dir / "congress.duckdb",
        ctx.staging_dir / "senate" / "congress.duckdb",
    ):
        if db_path.exists():
            _run_phantom_check(ctx, db_path, checks)
        else:
            checks[f"phantom_rows_{db_path.parent.name or 'staging'}"] = {
                "passed": False,
                "detail": f"missing: {db_path}",
            }
    _db_checks(ctx, checks)
    return {
        "all_passed": bool(checks) and all(c["passed"] for c in checks.values()),
        "checks": checks,
    }


# --------------------------------------------------------------------------- #
# Stage execution
# --------------------------------------------------------------------------- #


def _log_dir(ctx: Context) -> Path:
    return ctx.staging_dir / "logs"


def _execute_command(ctx: Context, command: list[str], log_path: Path) -> int:
    if ctx.verbose:
        print(f"  $ {' '.join(shlex.quote(part) for part in command)}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log_path.open("ab") as log:
            log.write(
                (
                    "\n$ " + " ".join(shlex.quote(part) for part in command) + "\n"
                ).encode()
            )
            proc = subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                cwd=ctx.repo_root,
                env=ctx.env,
            )
    except OSError as exc:
        print(f"  ERROR: failed to launch: {exc}", file=sys.stderr)
        return 127
    return proc.returncode


def _execute_stage(
    stage_id: str, commands: list[list[str]], ctx: Context
) -> StageResult:
    result = StageResult(stage_id=stage_id, status=STATUS_PASSED, commands=commands)
    started = time.monotonic()
    for index, command in enumerate(commands):
        log_path = _log_dir(ctx) / f"{stage_id}.log"
        exit_code = _execute_command(ctx, command, log_path)
        result.exit_codes.append(exit_code)
        if exit_code != 0:
            result.status = STATUS_FAILED
            result.detail["failed_command_index"] = index
            result.detail["exit_code"] = exit_code
            break
    result.duration_seconds = round(time.monotonic() - started, 2)

    if stage_id == "invariants":
        audit = run_invariants_audit(ctx)
        result.detail["invariants"] = audit
        if result.status == STATUS_PASSED and not audit["all_passed"]:
            result.status = STATUS_FAILED

    # Post-command artifact gates: the stage must have produced its artifact.
    required = {
        "capitol": ctx.staging_dir / "capitol_recon.json",
        "prices": ctx.staging_dir / "price_snapshot.json",
        "validation": ctx.staging_dir / "validation_results.json",
    }.get(stage_id)
    if (
        required is not None
        and result.status == STATUS_PASSED
        and not required.is_file()
    ):
        result.status = STATUS_FAILED
        result.detail["missing_artifact"] = str(required.relative_to(ctx.staging_dir))

    if result.status == STATUS_PASSED:
        result.artifacts = _hash_artifacts(stage_id, ctx)
    return result


def run_stages(ctx: Context) -> dict[str, StageResult]:
    commands_by_stage = build_stage_commands(ctx)
    results: dict[str, StageResult] = {}
    for stage_id in STAGE_ORDER:
        commands = commands_by_stage[stage_id]
        if ctx.dry_run:
            results[stage_id] = StageResult(
                stage_id=stage_id, status=STATUS_WOULD_RUN, commands=commands
            )
            continue
        results[stage_id] = _execute_stage(stage_id, commands, ctx)
        if results[stage_id].status == STATUS_FAILED:
            # Failures stop promotion: later stages are recorded as not_run
            # (never executed) so the manifest shows the full ordered plan.
            for blocked in STAGE_ORDER[STAGE_ORDER.index(stage_id) + 1 :]:
                results[blocked] = StageResult(
                    stage_id=blocked,
                    status=STATUS_NOT_RUN,
                    commands=commands_by_stage[blocked],
                )
            break
    return results


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #


def _git_state(repo_root: Path) -> dict:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=repo_root,
            check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                cwd=repo_root,
                check=True,
            ).stdout.strip()
        )
        return {"revision": revision, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"revision": "unknown", "dirty": None}


def _manifest_payload(ctx: Context, results: dict[str, StageResult]) -> dict:
    stages = {}
    for stage_id, result in results.items():
        stages[stage_id] = {
            "status": result.status,
            "commands": result.commands,
            "exit_codes": result.exit_codes,
            "duration_seconds": result.duration_seconds,
            "artifacts_sha256": result.artifacts,
            "detail": result.detail,
        }
    all_passed = (
        bool(results)
        and len(results) == len(STAGE_ORDER)
        and all(r.status == STATUS_PASSED for r in results.values())
    )
    return {
        "schema_version": 1,
        "purpose": (
            "Reproducible go-live generation manifest. The final status is "
            f"{FINAL_NOT_ESTABLISHED} unless every stage gate passes; stage "
            "failures stop promotion and later stages are not_run. No "
            "profitability claim is emitted by this orchestrator."
        ),
        "generation": ctx.generation,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git": _git_state(ctx.repo_root),
        "staging_root": str(ctx.staging_dir),
        "scope": {
            "house_year": ctx.house_year,
            "senate_start": ctx.senate_start.isoformat(),
            "senate_end": ctx.senate_end.isoformat(),
            "validation": {
                "train_start": ctx.train_start.isoformat(),
                "train_end": ctx.train_end.isoformat(),
                "test_start": ctx.test_start.isoformat(),
                "test_end": ctx.test_end.isoformat(),
            },
        },
        "stages": stages,
        "final_status": FINAL_ESTABLISHED if all_passed else FINAL_NOT_ESTABLISHED,
    }


def write_manifest(ctx: Context, results: dict[str, StageResult]) -> Path:
    payload = _manifest_payload(ctx, results)
    path = ctx.staging_dir / "manifest.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    tmp.replace(path)
    return path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _iso_date(value: str) -> _dt.date:
    try:
        return _dt.date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{value!r} is not a YYYY-MM-DD date") from exc


def build_context(args: argparse.Namespace) -> Context:
    repo_root = _REPO_ROOT
    generation = args.generation or time.strftime("go-live-%Y%m%dT%H%M%S")
    staging_root = Path(args.staging_root)
    if not staging_root.is_absolute():
        staging_root = repo_root / staging_root
    staging_dir = (staging_root / generation).resolve()
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(_SRC), str(repo_root), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    env["PTR_SKIP_DOCLING"] = "1"
    runner = (sys.executable, "-c", _CLI_RUNNER)
    return Context(
        repo_root=repo_root,
        staging_dir=staging_dir,
        generation=generation,
        house_year=args.house_year,
        senate_start=args.senate_start,
        senate_end=args.senate_end,
        train_start=args.train_start,
        train_end=args.train_end,
        test_start=args.test_start,
        test_end=args.test_end,
        dry_run=args.dry_run,
        verbose=args.verbose,
        runner=runner,
        env=env,
    )


def _validate_staging(ctx: Context) -> None:
    real_data = (ctx.repo_root / "data").resolve()
    if ctx.staging_dir == real_data:
        raise SystemExit(
            "error: --staging-root must not be the real data directory "
            f"({real_data}); the real congress.duckdb is never opened"
        )
    if ctx.staging_dir.exists():
        raise SystemExit(
            f"error: staging generation directory already exists: {ctx.staging_dir} "
            "(refusing to clobber; choose a new --generation or --staging-root)"
        )


def _print_plan(ctx: Context, results: dict[str, StageResult]) -> None:
    print(f"Go-live plan: generation={ctx.generation} staging={ctx.staging_dir}")
    for stage_id in STAGE_ORDER:
        result = results[stage_id]
        print(f"  [{stage_id}]")
        for command in result.commands:
            print(f"    $ {' '.join(shlex.quote(part) for part in command)}")
    print(
        "  final status: "
        + (
            "would be established if every gate passes"
            if ctx.dry_run
            else FINAL_NOT_ESTABLISHED
        )
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the reproducible go-live generation pipeline (house, senate, "
            "capitol, prices, invariants, validation) with fail-closed gating "
            "against an isolated staging root."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--staging-root",
        default="data/.staging/go-live",
        help="Root under which <generation>/ is created; never the real data dir",
    )
    parser.add_argument(
        "--generation",
        default=None,
        help="Generation id (default: go-live-<local timestamp>)",
    )
    parser.add_argument(
        "--house-year",
        type=int,
        default=_dt.date.today().year,
        help="House archive year for the fetch/parse/OCR stage",
    )
    parser.add_argument(
        "--senate-start",
        type=_iso_date,
        default=None,
        help="Senate eFD sweep start (default: one year before --senate-end)",
    )
    parser.add_argument(
        "--senate-end",
        type=_iso_date,
        default=_dt.date.today(),
        help="Senate eFD sweep end (default: today)",
    )
    parser.add_argument(
        "--train-start",
        type=_iso_date,
        default=DEFAULT_TRAIN_START,
        help="Validation harness training window start",
    )
    parser.add_argument(
        "--train-end",
        type=_iso_date,
        default=DEFAULT_TRAIN_END,
        help="Validation harness training window end",
    )
    parser.add_argument(
        "--test-start",
        type=_iso_date,
        default=DEFAULT_TEST_START,
        help="Validation harness test window start",
    )
    parser.add_argument(
        "--test-end",
        type=_iso_date,
        default=DEFAULT_TEST_END,
        help="Validation harness test window end",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the exact commands each stage would run; create nothing",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print every command before executing it",
    )
    args = parser.parse_args(argv)

    if args.senate_start is None:
        args.senate_start = args.senate_end - _dt.timedelta(days=365)
    if args.senate_start > args.senate_end:
        parser.error("--senate-start must be on or before --senate-end")

    ctx = build_context(args)

    if not ctx.dry_run:
        _validate_staging(ctx)
        ctx.staging_dir.mkdir(parents=True, exist_ok=False)

    results = run_stages(ctx)

    if ctx.dry_run:
        _print_plan(ctx, results)
        return 0

    manifest_path = write_manifest(ctx, results)
    final_status = _manifest_payload(ctx, results)["final_status"]
    print(f"Generation manifest: {manifest_path}")
    print(f"Final status: {final_status}")
    failed = [sid for sid, r in results.items() if r.status == STATUS_FAILED]
    if failed:
        print(
            "Promotion stopped: " + ", ".join(f"{sid} failed" for sid in failed),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
