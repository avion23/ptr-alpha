"""Fail-loud freshness doctor for the canonical database and cron artifacts.

Exits nonzero when the pipeline has silently stopped feeding canonical:
stale prices, empty dividends, dead form4/receipt feeds, stale House
ingestion, silent OCR staging, cron artifacts (staging/temp databases)
older than the grace window, a dead daily-doctor heartbeat, a dead nightly
OCR-drain cron, or a live crontab that drifted from the tracked contract
(cron/crontab.txt). Read-only on every database and directory it opens;
safe to run any time, designed for a daily cron.

End-product freshness lives in src/analyzer/freshness.py so the operator's
``health`` command and this cron share one implementation. The OCR
staging/backlog checks exist because ``glm_stage.py stage`` exits 0 even
when individual documents error and ``backlog_ocr.sh`` returns only the
doctor status (review finding 12, 2026-10-18): without these checks an
entirely broken OCR drain looks successful. Missing evidence is reported
UNKNOWN (counts as a failure), never green.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from datetime import date, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from analyzer.freshness import FreshnessCheck, monitoring_checks  # noqa: E402
from scripts.audit_sample import _doc_fail_severity, _verdict_of  # noqa: E402

CANONICAL = Path(os.environ.get("DOCTOR_CANONICAL", str(REPO / "data" / "congress.duckdb")))
RECEIPTS = REPO / "data" / "recommendations" / "receipts.log"
PRICES_CRON = REPO / "data" / "logs" / "prices-cron"
# Active GLM staging root (data/staging/glm-backlog). The retired Gemini
# staging DBs (data/.staging/rescrape) are gone; that scan was dropped with
# them (review finding 12).
STAGING_ROOT = Path(
    os.environ.get("DOCTOR_STAGING_ROOT", str(REPO / "data" / "staging" / "glm-backlog"))
)
BACKUP_ROOT = REPO / "data" / "backups" / "glm-backlog"
OCR_LOG_ROOT = Path(
    os.environ.get(
        "DOCTOR_OCR_LOG_ROOT", str(REPO / "data" / "logs" / "ocr-backlog-cron")
    )
)
HEARTBEAT = Path(
    os.environ.get("DOCTOR_HEARTBEAT", str(REPO / "data" / "alerts" / "doctor-last-run"))
)
CRONTAB_CONTRACT = REPO / "cron" / "crontab.txt"

# Weekly prices cadence: tolerate one missed week before failing.
MAX_PRICE_LAG_SESSIONS = 8
ARTIFACT_MAX_AGE_DAYS = 10
# Daily doctor cadence plus slack before "the monitor itself is dead".
HEARTBEAT_MAX_AGE_HOURS = 36
# Staging error records must be retried nightly; one week of an unchanged
# error record means the retry path is starved (finding 12).
OCR_ERROR_MAX_AGE_DAYS = 7
# Staged successes await the pixel audit; two weeks at audit:null means the
# sampler/auditor is not running (finding 12).
OCR_AUDIT_MAX_AGE_DAYS = 14
# The OCR drain cron runs nightly; two days without a run means it is dead.
OCR_CRON_RUN_MAX_AGE_DAYS = 2
OCR_CRON_IN_PROGRESS_MAX_AGE_HOURS = 7
STAGING_BACKUP_MAX_AGE_DAYS = 8

_DOCTOR_EXIT_RE = re.compile(r"doctor exit=(\d+)")


def check(
    canonical,
    staging_root: Path | None = None,
    ocr_log_root: Path | None = None,
) -> list[FreshnessCheck]:
    staging_root = staging_root or STAGING_ROOT
    ocr_log_root = ocr_log_root or OCR_LOG_ROOT
    results: list[FreshnessCheck] = [
        *monitoring_checks(canonical, receipts_path=RECEIPTS, staging_root=staging_root),
        # 5. No unconsumed cron artifacts older than the grace window.
        _stale_artifacts(),
        # 6. GLM staging observability: counts + retry/audit starvation.
        _ocr_staging_check(staging_root),
        # 7. Nightly drain must leave a recent staging snapshot.
        _staging_backup_check(),
        # 8. Nightly OCR drain engagement (dead-cron + truncated-run tripwire).
        _ocr_cron_check(ocr_log_root),
        # 9. Dead-man switch: the daily doctor cron must be beating. Catches a
        # broken/edited-away wrapper while doctor itself still runs.
        _heartbeat(),
        # 10. Deployment drift: the live crontab must match the tracked contract.
        _crontab_contract(),
    ]
    return results


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def _stale_artifacts() -> FreshnessCheck:
    stale_artifacts: list[str] = []
    cutoff = time.time() - ARTIFACT_MAX_AGE_DAYS * 86_400
    if PRICES_CRON.exists():
        for temp in PRICES_CRON.glob("*/prices.duckdb"):
            if temp.stat().st_mtime < cutoff:
                stale_artifacts.append(str(temp.relative_to(REPO)))
    detail = (
        f"{len(stale_artifacts)} unconsumed artifacts older than "
        f"{ARTIFACT_MAX_AGE_DAYS}d" + (": " + ", ".join(stale_artifacts[:4]))
        if stale_artifacts
        else "no stale cron artifacts"
    )
    return FreshnessCheck("artifacts.consumed", detail, not stale_artifacts)


def _heartbeat() -> FreshnessCheck:
    if HEARTBEAT.exists():
        age_h = (time.time() - HEARTBEAT.stat().st_mtime) / 3600
        return FreshnessCheck(
            "doctor.alive",
            f"heartbeat {age_h:.1f}h old (limit {HEARTBEAT_MAX_AGE_HOURS}h)",
            age_h <= HEARTBEAT_MAX_AGE_HOURS,
        )
    return FreshnessCheck(
        "doctor.alive",
        f"no heartbeat at {_rel(HEARTBEAT)}: the daily "
        "doctor cron has never fired here",
        False,
    )


# --------------------------------------------------------------------------
# GLM OCR staging observability (review finding 12)
# --------------------------------------------------------------------------


def _parse_stamp(value) -> float | None:
    """Parsed an ISO-8601 Z/offset stamp to an epoch, or None."""
    if not value or not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _ocr_staging_check(
    staging_root: Path, now: float | None = None
) -> FreshnessCheck:
    """Scan data/staging/glm-backlog/<year>/*.json.

    Reports disposition counts (success / no_txs / error / audit pass /
    critical-fail / cosmetic-fail / unaudited / promoted-consumed-absent) and
    FAILs on the two starvation modes an exit-0 stage run can hide: error
    records that no nightly retry has touched for a week, and staged successes
    still audit:null after two weeks (the pixel-audit sampler is not running).
    Unreadable records are an evidence gap and fail loudly with the file named.
    """
    name = "staging.ocr.backlog"
    if not staging_root.exists():
        return FreshnessCheck.unknown(
            name,
            f"no staging root at {_rel(staging_root)}: GLM OCR staging has "
            "never run here (create it: scripts/glm_stage.py stage)",
        )
    now = now if now is not None else time.time()
    counts = {"success": 0, "no_txs": 0, "error": 0, "other": 0}
    audit_pass = audit_critical_fail = audit_cosmetic_fail = audit_none = 0
    asset_review_unaudited = 0
    starved_errors: list[str] = []
    audit_starved: list[str] = []
    unreadable: list[str] = []

    # Match audit_sample.report's sticky, per-document worst-fail semantics.
    # Its helper is the source of truth for severity and legacy handling.
    audit_entries_by_doc: dict[str, list[dict]] = {}
    try:
        results = json.loads(
            (staging_root / "audit-results.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        results = []
    if isinstance(results, list):
        for entry in results:
            if (
                isinstance(entry, dict)
                and "doc_id" in entry
                and _verdict_of(entry) in ("pass", "fail")
            ):
                audit_entries_by_doc.setdefault(str(entry["doc_id"]), []).append(entry)

    for path in sorted(staging_root.glob("[0-9][0-9][0-9][0-9]/*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            unreadable.append(_rel(path))
            continue
        if not isinstance(record, dict):
            unreadable.append(_rel(path))
            continue
        status = record.get("parse_status")
        counts[status if status in ("success", "no_txs", "error") else "other"] += 1
        audit = record.get("audit")
        doc_id = record.get("doc_id")
        audit_entries = (
            audit_entries_by_doc.get(str(doc_id), []) if doc_id is not None else []
        )
        fail_entries = [
            entry for entry in audit_entries if _verdict_of(entry) == "fail"
        ]
        if (
            status == "success"
            and record.get("asset_review") is True
            and audit not in ("pass", "fail")
            and not audit_entries
        ):
            asset_review_unaudited += 1
        if fail_entries:
            severity, _ = _doc_fail_severity(fail_entries)
            if severity == "cosmetic":
                audit_cosmetic_fail += 1
            else:
                audit_critical_fail += 1
        elif audit == "fail":
            # Staging is stamped before the results history is appended. If a
            # process stopped between those writes, honor the staged verdict;
            # absent/invalid severity remains critical like a legacy result.
            severity, _ = _doc_fail_severity(
                [{"verdict": "fail", "severity": record.get("audit_severity")}]
            )
            if severity == "cosmetic":
                audit_cosmetic_fail += 1
            else:
                audit_critical_fail += 1
        elif audit == "pass" or any(
            _verdict_of(entry) == "pass" for entry in audit_entries
        ):
            audit_pass += 1
        else:
            audit_none += 1
        stamp = _parse_stamp(record.get("parsed_at_utc"))
        age_days = None if stamp is None else (now - stamp) / 86_400
        if status == "error" and (age_days is None or age_days > OCR_ERROR_MAX_AGE_DAYS):
            starved_errors.append(_rel(path))
        if status == "success" and audit is None and (
            age_days is None or age_days > OCR_AUDIT_MAX_AGE_DAYS
        ):
            audit_starved.append(_rel(path))

    consumed_absent = _promoted_consumed_absent(staging_root)

    summary = (
        f"{counts['success']} success / {counts['no_txs']} no_txs / "
        f"{counts['error']} error"
        + (f" / {counts['other']} other" if counts["other"] else "")
        + f"; audit pass/critical-fail/cosmetic-fail/unaudited "
        f"{audit_pass}/{audit_critical_fail}/{audit_cosmetic_fail}/{audit_none}"
        + "; cosmetic fails are tracked, not blocking"
        + f"; promoted-consumed-absent {consumed_absent}"
    )
    asset_review_info = (
        f"\n    INFO: asset_review flagged, unaudited: {asset_review_unaudited} "
        "(audit sampler draws these first)"
    )
    problems: list[str] = []
    if unreadable:
        problems.append(
            f"{len(unreadable)} unreadable staging record(s) (evidence gap): "
            + ", ".join(unreadable[:3])
            + " — inspect or remove them"
        )
    if starved_errors:
        problems.append(
            f"{len(starved_errors)} error record(s) older than "
            f"{OCR_ERROR_MAX_AGE_DAYS}d — nightly retry starved; delete the "
            "record to re-stage it (select_candidates skips existing "
            "records): " + ", ".join(starved_errors[:3])
        )
    if audit_starved:
        problems.append(
            f"{len(audit_starved)} staged success record(s) older than "
            f"{OCR_AUDIT_MAX_AGE_DAYS}d still audit:null — pixel-audit "
            "sampler not running (scripts/audit_sample.py sample / record): "
            + ", ".join(audit_starved[:3])
        )
    if problems:
        return FreshnessCheck(
            name, summary + "; " + "; ".join(problems) + asset_review_info, False
        )
    return FreshnessCheck(name, summary + asset_review_info, True)


def _staging_backup_check(
    backup_root: Path = BACKUP_ROOT, now: float | None = None
) -> FreshnessCheck:
    """Fail if the nightly staging-corpus snapshot is missing or stale."""
    name = "staging.backup.freshness"
    fix = (
        "run bash cron/backlog_ocr.sh manually or wait for the 03:30 UTC cron; "
        "the wrapper snapshots after staging"
    )
    if not backup_root.is_dir():
        return FreshnessCheck(
            name, f"no staging backup directory at {_rel(backup_root)}; {fix}", False
        )
    try:
        backups = [
            path for path in backup_root.glob("glm-backlog-*.tar.gz") if path.is_file()
        ]
        newest = (
            max(backups, key=lambda path: path.stat().st_mtime) if backups else None
        )
        modified = newest.stat().st_mtime if newest is not None else None
    except OSError as exc:
        return FreshnessCheck(
            name,
            f"cannot inspect staging backups at {_rel(backup_root)} "
            f"({type(exc).__name__}: {exc}); {fix}",
            False,
        )
    if newest is None or modified is None:
        return FreshnessCheck(
            name,
            f"no glm-backlog-*.tar.gz snapshot in {_rel(backup_root)}; {fix}",
            False,
        )
    now = now if now is not None else time.time()
    age_days = max(0.0, (now - modified) / 86_400)
    if age_days > STAGING_BACKUP_MAX_AGE_DAYS:
        return FreshnessCheck(
            name,
            f"newest staging backup {_rel(newest)} is {age_days:.1f}d old "
            f"(limit {STAGING_BACKUP_MAX_AGE_DAYS}d); {fix}",
            False,
        )
    return FreshnessCheck(
        name,
        f"newest staging backup {_rel(newest)} is {age_days:.1f}d old "
        f"(limit {STAGING_BACKUP_MAX_AGE_DAYS}d)",
        True,
    )


def _promoted_consumed_absent(staging_root: Path) -> int:
    """Promoted-journal entries whose staging record is gone (consumed)."""
    journal = staging_root / "promoted.jsonl"
    try:
        lines = journal.read_text(encoding="utf-8").splitlines()
    except OSError:
        return 0
    consumed = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        year = entry.get("year")
        doc_id = entry.get("doc_id")
        if year is None or doc_id is None:
            continue
        if not (staging_root / str(year) / f"{doc_id}.json").exists():
            consumed += 1
    return consumed


def _ocr_cron_check(
    ocr_log_root: Path, today: date | None = None, now: float | None = None
) -> FreshnessCheck:
    """The nightly drain (cron/backlog_ocr.sh) must leave fresh evidence.

    A recent run directory without its final ``doctor exit=`` line is still
    in progress: backlog_ocr.sh writes that line after this doctor check.
    Older incomplete runs are truncated; completed runs older than two days
    indicate a dead cron. Log inspection is read-only.
    """
    name = "cron.ocr.engagement"
    if not ocr_log_root.exists():
        return FreshnessCheck(name, _ocr_cron_never_fired_detail(ocr_log_root), False)
    run_dirs = sorted(
        (
            d
            for d in ocr_log_root.iterdir()
            if d.is_dir() and re.fullmatch(r"\d{8}", d.name)
        ),
        key=lambda d: d.name,
    )
    if not run_dirs:
        return FreshnessCheck(name, _ocr_cron_never_fired_detail(ocr_log_root), False)
    newest = run_dirs[-1]
    today = today or date.today()
    try:
        run_date = datetime.strptime(newest.name, "%Y%m%d").date()
    except ValueError:  # unreachable given the fullmatch above
        return FreshnessCheck(name, _ocr_cron_never_fired_detail(ocr_log_root), False)
    age_days = (today - run_date).days
    now = now if now is not None else time.time()
    try:
        run_age_hours = max(0.0, (now - newest.stat().st_mtime) / 3600)
    except OSError as exc:
        return FreshnessCheck(
            name,
            f"cannot inspect OCR-drain run {newest.name} ({type(exc).__name__}: {exc})",
            False,
        )
    run_log = newest / "run.log"
    try:
        text = run_log.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        text = ""
    except OSError as exc:
        return FreshnessCheck(
            name,
            f"cannot read {run_log.name} for run {newest.name} "
            f"({type(exc).__name__}: {exc}): truncated run — "
            "re-run cron/backlog_ocr.sh",
            False,
        )
    exits = _DOCTOR_EXIT_RE.findall(text)
    if not exits:
        if run_age_hours <= OCR_CRON_IN_PROGRESS_MAX_AGE_HOURS:
            lock_held = _ocr_drain_lock_held(ocr_log_root)
            progress = (
                "in progress (lock held)"
                if lock_held
                else "in progress (recent run directory)"
            )
            return FreshnessCheck(
                name,
                f"OCR-drain run {newest.name} {progress}; no 'doctor exit=' "
                f"line yet (run directory {run_age_hours:.1f}h old, "
                f"in-progress limit {OCR_CRON_IN_PROGRESS_MAX_AGE_HOURS}h)",
                True,
            )
        return FreshnessCheck(
            name,
            f"no 'doctor exit=' line in {_rel(run_log)} for run "
            f"{newest.name} ({run_age_hours:.1f}h old): run truncated before "
            "the verification step — re-run cron/backlog_ocr.sh; "
            + _ocr_cron_recovery_hint(),
            False,
        )
    if age_days > OCR_CRON_RUN_MAX_AGE_DAYS:
        return FreshnessCheck(
            name,
            f"newest OCR-drain run {newest.name} is {age_days}d old (limit "
            f"{OCR_CRON_RUN_MAX_AGE_DAYS}d): the nightly backlog_ocr.sh "
            f"drain is dead — {_ocr_cron_recovery_hint()}",
            False,
        )
    last_exit = int(exits[-1])
    if last_exit != 0:
        return FreshnessCheck(
            name,
            f"newest OCR-drain run {newest.name} ended with doctor "
            f"exit={last_exit} ({age_days}d old): that nightly run reported "
            f"failures — see {_rel(run_log)}",
            False,
        )
    return FreshnessCheck(
        name,
        f"newest OCR-drain run {newest.name} ({age_days}d old, limit "
        f"{OCR_CRON_RUN_MAX_AGE_DAYS}d); run.log reached "
        f"'doctor exit={last_exit}'",
        True,
    )


def _ocr_drain_lock_held(ocr_log_root: Path) -> bool:
    """Best-effort non-blocking flock probe; acquires only when momentarily free."""
    lock_path = ocr_log_root / "drain.lock"
    if not lock_path.exists():
        return False
    try:
        probe = subprocess.run(
            ["flock", "-n", str(lock_path), "true"],
            capture_output=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return probe.returncode == 1


def _backlog_ocr_crontab_status() -> bool | None:
    """Whether the live crontab has an active backlog_ocr.sh line."""
    try:
        probe = subprocess.run(
            ["crontab", "-l"], capture_output=True, text=True, timeout=15, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if probe.returncode != 0:
        return None
    return any(
        "cron/backlog_ocr.sh" in line and not line.lstrip().startswith("#")
        for line in probe.stdout.splitlines()
    )


def _ocr_cron_recovery_hint() -> str:
    installed = _backlog_ocr_crontab_status()
    if installed is True:
        return "cron line is installed; inspect the cron daemon and OCR run logs"
    if installed is False:
        return "install with: bash cron/install.sh -y"
    return "live crontab unavailable; check crontab -l and OCR run logs"


def _ocr_cron_never_fired_detail(ocr_log_root: Path) -> str:
    installed = _backlog_ocr_crontab_status()
    if installed is True:
        hint = "installed in crontab; awaiting first scheduled firing at 03:30 UTC"
    elif installed is False:
        hint = "install with: bash cron/install.sh -y"
    else:
        hint = "live crontab unavailable; check crontab -l"
    return (
        f"no OCR-drain cron runs under {_rel(ocr_log_root)}: "
        f"backlog_ocr.sh has never fired here although cron/crontab.txt "
        f"declares it daily; {hint}"
    )


def _active_repo_crontab_lines(text: str, repo_prefix: str) -> list[str]:
    lines = []
    for raw in text.splitlines():
        line = " ".join(raw.split())
        if not line or line.startswith("#"):
            continue
        if repo_prefix in line:
            lines.append(line)
    return sorted(lines)


def _crontab_contract() -> FreshnessCheck:
    expected = _active_repo_crontab_lines(
        CRONTAB_CONTRACT.read_text().replace("@REPO@", str(REPO)), str(REPO)
    ) if CRONTAB_CONTRACT.exists() else None
    if expected is None:
        return FreshnessCheck(
            "crontab.contract", f"missing {CRONTAB_CONTRACT.relative_to(REPO)}", False
        )
    try:
        probe = subprocess.run(
            ["crontab", "-l"], capture_output=True, text=True, timeout=15, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return FreshnessCheck(
            "crontab.contract",
            f"live crontab unavailable ({type(exc).__name__}); deployment not verified",
            True,
        )
    if probe.returncode != 0:
        return FreshnessCheck(
            "crontab.contract",
            f"live crontab unavailable ({probe.stderr.strip()}); deployment not verified",
            True,
        )
    live = _active_repo_crontab_lines(probe.stdout, str(REPO))
    if live == expected:
        return FreshnessCheck(
            "crontab.contract", f"{len(live)} repo cron lines match cron/crontab.txt", True
        )
    missing = [line for line in expected if line not in live]
    extra = [line for line in live if line not in expected]
    return FreshnessCheck(
        "crontab.contract",
        f"live crontab drifted from cron/crontab.txt; missing={missing} extra={extra}",
        False,
    )


def main() -> int:
    if not CANONICAL.exists():
        print(f"doctor: canonical database missing: {CANONICAL}")
        return 1
    import duckdb

    canonical = duckdb.connect(str(CANONICAL), read_only=True)
    try:
        results = check(canonical)
    finally:
        canonical.close()

    failed = 0
    print(f"doctor ({date.today().isoformat()}):")
    for row in results:
        print(f"  [{row.mark}] {row.name}: {row.detail}")
        failed += not row.ok
    print(f"doctor: {'ALL PASS' if not failed else f'{failed} CHECK(S) FAILED'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
