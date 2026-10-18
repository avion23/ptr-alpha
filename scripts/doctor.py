"""Fail-loud freshness doctor for the canonical database and cron artifacts.

Exits nonzero when the pipeline has silently stopped feeding canonical:
stale prices, empty dividends, dead form4/receipt feeds, or cron artifacts
(staging/temp databases) older than the grace window — the signature of a
writer that stops one step short of canonical. Read-only on every database
it opens; safe to run any time, designed for a daily cron.
"""

from __future__ import annotations

import os
import sys
import time
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.refresh_prices import refresh_end_date  # noqa: E402

CANONICAL = Path(os.environ.get("DOCTOR_CANONICAL", str(REPO / "data" / "congress.duckdb")))
RECEIPTS = REPO / "data" / "recommendations" / "receipts.log"
PRICES_CRON = REPO / "data" / "logs" / "prices-cron"
GEMINI_STAGING = REPO / "data" / ".staging" / "rescrape"

# Weekly prices cadence: tolerate one missed week before failing.
MAX_PRICE_LAG_SESSIONS = 8
FORM4_MAX_AGE_DAYS = 7
RECEIPTS_MAX_AGE_DAYS = 8
ARTIFACT_MAX_AGE_DAYS = 10


def cast_date(value) -> date:
    return value.date() if hasattr(value, "date") else value


def check(canonical) -> list[tuple[str, str, bool]]:
    results: list[tuple[str, str, bool]] = []

    def record(name: str, detail: str, ok: bool) -> None:
        results.append((name, detail, ok))

    # 1. Price freshness vs the last completed NYSE session.
    today = date.today()
    expected = refresh_end_date(today)
    max_price = canonical.execute("SELECT MAX(date) FROM prices").fetchone()[0]
    max_price = cast_date(max_price)
    lag_sessions = len(
        __import__("pandas").bdate_range(max_price, expected)
    ) - 1
    record(
        "prices.freshness",
        f"canonical max {max_price}, last completed session {expected} "
        f"({lag_sessions} sessions back, limit {MAX_PRICE_LAG_SESSIONS})",
        lag_sessions <= MAX_PRICE_LAG_SESSIONS,
    )

    # 2. Dividends populated (ex-div gate is starved while this is 0).
    has_div = canonical.execute(
        "SELECT COUNT(*) FROM duckdb_tables() WHERE table_name = 'dividends'"
    ).fetchone()[0]
    div_rows = (
        canonical.execute("SELECT COUNT(*) FROM dividends").fetchone()[0]
        if has_div
        else 0
    )
    record(
        "dividends.populated",
        f"dividends rows: {div_rows}" + ("" if has_div else " (table absent)"),
        div_rows > 0,
    )

    # 3. Form4 feed alive.
    newest_f4 = canonical.execute(
        "SELECT MAX(disclosure_date) FROM canonical_transactions "
        "WHERE source = 'form4'"
    ).fetchone()[0]
    newest_f4 = cast_date(newest_f4) if newest_f4 else None
    f4_age = (today - newest_f4).days if newest_f4 else 10_000
    record(
        "form4.freshness",
        f"newest form4 disclosure {newest_f4} ({f4_age}d old, "
        f"limit {FORM4_MAX_AGE_DAYS}d)",
        f4_age <= FORM4_MAX_AGE_DAYS,
    )

    # 4. Receipt ledger alive.
    receipt_age = 10_000
    if RECEIPTS.exists():
        last_line = RECEIPTS.read_text().strip().splitlines()[-1]
        published = __import__("json").loads(last_line).get("published_at", "")
        if published:
            receipt_age = int(time.time() - __import__("datetime").datetime
                              .fromisoformat(published.replace("Z", "+00:00"))
                              .timestamp()) // 86_400
    record(
        "receipts.freshness",
        f"newest receipt {receipt_age}d old (limit {RECEIPTS_MAX_AGE_DAYS}d)",
        receipt_age <= RECEIPTS_MAX_AGE_DAYS,
    )

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
    record(
        "artifacts.consumed",
        (
            f"{len(stale_artifacts)} unconsumed artifacts older than "
            f"{ARTIFACT_MAX_AGE_DAYS}d" + (": " + ", ".join(stale_artifacts[:4]))
            if stale_artifacts
            else "no stale cron artifacts"
        ),
        not stale_artifacts,
    )

    return results


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
