"""Fail-loud freshness doctor for the canonical database and cron artifacts.

Exits nonzero when the pipeline has silently stopped feeding canonical:
stale prices, empty dividends, dead form4/receipt feeds, cron artifacts
(staging/temp databases) older than the grace window, a dead daily-doctor
heartbeat, or a live crontab that drifted from the tracked contract
(cron/crontab.txt). Read-only on every database it opens and on the
crontab; safe to run any time, designed for a daily cron.

End-product freshness lives in src/analyzer/freshness.py so the operator's
``health`` command and this cron share one implementation.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from analyzer.freshness import end_to_end_checks  # noqa: E402

CANONICAL = Path(os.environ.get("DOCTOR_CANONICAL", str(REPO / "data" / "congress.duckdb")))
RECEIPTS = REPO / "data" / "recommendations" / "receipts.log"
PRICES_CRON = REPO / "data" / "logs" / "prices-cron"
GEMINI_STAGING = REPO / "data" / ".staging" / "rescrape"  # active staging only
GEMINI_ARCHIVE = REPO / "data" / "archive"  # retired roots: kept, never flagged
HEARTBEAT = Path(
    os.environ.get("DOCTOR_HEARTBEAT", str(REPO / "data" / "alerts" / "doctor-last-run"))
)
CRONTAB_CONTRACT = REPO / "cron" / "crontab.txt"

# Weekly prices cadence: tolerate one missed week before failing.
MAX_PRICE_LAG_SESSIONS = 8
ARTIFACT_MAX_AGE_DAYS = 10
# Daily doctor cadence plus slack before "the monitor itself is dead".
HEARTBEAT_MAX_AGE_HOURS = 36


def check(canonical) -> list[tuple[str, str, bool]]:
    results: list[tuple[str, str, bool]] = [
        (c.name, c.detail, c.ok)
        for c in end_to_end_checks(canonical, receipts_path=RECEIPTS)
    ]

    # 5. No unconsumed cron artifacts older than the grace window.
    stale_artifacts: list[str] = []
    cutoff = time.time() - ARTIFACT_MAX_AGE_DAYS * 86_400
    if PRICES_CRON.exists():
        for temp in PRICES_CRON.glob("*/prices.duckdb"):
            if temp.stat().st_mtime < cutoff:
                stale_artifacts.append(str(temp.relative_to(REPO)))
    if GEMINI_STAGING.exists():
        for db in GEMINI_STAGING.glob("*/years/*/congress.duckdb"):
            if db.stat().st_mtime < cutoff:
                stale_artifacts.append(str(db.relative_to(REPO)) + " (staging)")
    results.append(
        (
            "artifacts.consumed",
            (
                f"{len(stale_artifacts)} unconsumed artifacts older than "
                f"{ARTIFACT_MAX_AGE_DAYS}d" + (": " + ", ".join(stale_artifacts[:4]))
                if stale_artifacts
                else "no stale cron artifacts"
            ),
            not stale_artifacts,
        )
    )

    # 6. Dead-man switch: the daily doctor cron must be beating. Catches a
    # broken/edited-away wrapper while doctor itself still runs.
    if HEARTBEAT.exists():
        age_h = (time.time() - HEARTBEAT.stat().st_mtime) / 3600
        results.append(
            (
                "doctor.alive",
                f"heartbeat {age_h:.1f}h old (limit {HEARTBEAT_MAX_AGE_HOURS}h)",
                age_h <= HEARTBEAT_MAX_AGE_HOURS,
            )
        )
    else:
        try:
            shown = HEARTBEAT.relative_to(REPO)
        except ValueError:
            shown = HEARTBEAT
        results.append(
            (
                "doctor.alive",
                f"no heartbeat at {shown}: the daily "
                "doctor cron has never fired here",
                False,
            )
        )

    # 7. Deployment drift: the live crontab must match the tracked contract.
    results.append(_crontab_contract())
    return results


def _active_repo_crontab_lines(text: str, repo_prefix: str) -> list[str]:
    lines = []
    for raw in text.splitlines():
        line = " ".join(raw.split())
        if not line or line.startswith("#"):
            continue
        if repo_prefix in line:
            lines.append(line)
    return sorted(lines)


def _crontab_contract() -> tuple[str, str, bool]:
    expected = _active_repo_crontab_lines(
        CRONTAB_CONTRACT.read_text().replace("@REPO@", str(REPO)), str(REPO)
    ) if CRONTAB_CONTRACT.exists() else None
    if expected is None:
        return ("crontab.contract", f"missing {CRONTAB_CONTRACT.relative_to(REPO)}", False)
    try:
        probe = subprocess.run(
            ["crontab", "-l"], capture_output=True, text=True, timeout=15, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return (
            "crontab.contract",
            f"live crontab unavailable ({type(exc).__name__}); deployment not verified",
            True,
        )
    if probe.returncode != 0:
        return (
            "crontab.contract",
            f"live crontab unavailable ({probe.stderr.strip()}); deployment not verified",
            True,
        )
    live = _active_repo_crontab_lines(probe.stdout, str(REPO))
    if live == expected:
        return ("crontab.contract", f"{len(live)} repo cron lines match cron/crontab.txt", True)
    missing = [line for line in expected if line not in live]
    extra = [line for line in live if line not in expected]
    return (
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
    for name, detail, ok in results:
        mark = "PASS" if ok else "FAIL"
        failed += not ok
        print(f"  [{mark}] {name}: {detail}")
    print(f"doctor: {'ALL PASS' if not failed else f'{failed} CHECK(S) FAILED'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
