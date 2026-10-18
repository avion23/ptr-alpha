"""End-to-end freshness of the canonical end products (monitoring only).

The 2026-10-18 cron incident: automated writers exited 0 for weeks while
their output never reached canonical, because every component measured
success at its own boundary and the end-product surface (prices, dividends,
form4, receipts) was measured nowhere. This module owns that surface so the
daily doctor (scripts/doctor.py) and the operator-facing ``health`` command
report from one implementation instead of drifting apart.

Review finding 12 (ocr-pipeline-astra-review-20261018.md) extended the
surface with the GLM OCR pipeline: staged-record freshness
(``ocr_backlog_freshness``) and House ingestion freshness
(``house_ingestion_freshness``). Consumers use ``monitoring_checks`` so the
doctor and ``health`` cannot drift apart.

Every function is read-only and returns a FreshnessCheck rather than
raising: a monitor that crashes on an unexpected schema stops monitoring.
Missing evidence is UNKNOWN (``ok=False``), never green.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
import json
from pathlib import Path
import time

import pandas as pd

from analyzer.price_repository import previous_nyse_session

MAX_PRICE_LAG_SESSIONS = 8
FORM4_MAX_AGE_DAYS = 7
RECEIPTS_MAX_AGE_DAYS = 8
# Nightly OCR drain (cron/backlog_ocr.sh) stages fresh docs every night;
# a newest-staged-record older than this means staging went silent.
OCR_BACKLOG_MAX_AGE_DAYS = 3
# House filings land weekly-ish and the all-years refresh runs weekly:
# 21 days without a house_pdf/gemini_ocr disclosure means ingestion stopped.
HOUSE_INGESTION_MAX_AGE_DAYS = 21


@dataclass(frozen=True)
class FreshnessCheck:
    name: str
    detail: str
    ok: bool
    # "OK" for a decisive verdict; "UNKNOWN" when the evidence to decide is
    # missing entirely. UNKNOWN is fail-loud (ok=False) but displayed
    # distinctly so an operator sees "no evidence" vs "evidence of failure".
    state: str = "OK"

    @property
    def mark(self) -> str:
        if self.state == "UNKNOWN":
            return "UNKNOWN"
        return "PASS" if self.ok else "FAIL"

    @classmethod
    def unknown(cls, name: str, detail: str) -> "FreshnessCheck":
        return cls(name, detail, False, state="UNKNOWN")


def _as_date(value) -> date | None:
    if value is None:
        return None
    return value if isinstance(value, date) else value.date()


def price_freshness(
    conn, today: date | None = None, max_lag: int = MAX_PRICE_LAG_SESSIONS
) -> FreshnessCheck:
    """Canonical prices must be within ``max_lag`` sessions of the last
    completed NYSE session (weekly refresh cadence tolerates one miss)."""
    today = today or date.today()
    expected = previous_nyse_session(today - pd.Timedelta(days=1)).date()
    max_price = _as_date(conn.execute("SELECT MAX(date) FROM prices").fetchone()[0])
    if max_price is None:
        return FreshnessCheck(
            "prices.freshness", "prices table is empty", False
        )
    lag = len(pd.bdate_range(max_price, expected)) - 1
    return FreshnessCheck(
        "prices.freshness",
        f"canonical max {max_price}, last completed session {expected} "
        f"({lag} sessions back, limit {max_lag})",
        lag <= max_lag,
    )


def dividends_populated(conn) -> FreshnessCheck:
    """The ex-div entry gate (charter R9) reads canonical dividends; 0 rows
    means the gate is starved even when the code ships."""
    has_div = conn.execute(
        "SELECT COUNT(*) FROM duckdb_tables() WHERE table_name = 'dividends'"
    ).fetchone()[0]
    rows = conn.execute("SELECT COUNT(*) FROM dividends").fetchone()[0] if has_div else 0
    return FreshnessCheck(
        "dividends.populated",
        f"dividends rows: {rows}" + ("" if has_div else " (table absent)"),
        rows > 0,
    )


def form4_freshness(
    conn, today: date | None = None, max_age_days: int = FORM4_MAX_AGE_DAYS
) -> FreshnessCheck:
    today = today or date.today()
    newest = _as_date(
        conn.execute(
            "SELECT MAX(disclosure_date) FROM canonical_transactions "
            "WHERE source = 'form4'"
        ).fetchone()[0]
    )
    age = (today - newest).days if newest else 10_000
    return FreshnessCheck(
        "form4.freshness",
        f"newest form4 disclosure {newest} ({age}d old, limit {max_age_days}d)",
        age <= max_age_days,
    )


def receipts_freshness(
    receipts_path: Path | str | None,
    now: float | None = None,
    max_age_days: int = RECEIPTS_MAX_AGE_DAYS,
) -> FreshnessCheck:
    """Decision receipts are the end product of the manual analyze cadence."""
    name = "receipts.freshness"
    if receipts_path is None or not Path(receipts_path).exists():
        return FreshnessCheck(name, f"no receipt ledger at {receipts_path}", False)
    last_line = Path(receipts_path).read_text().strip().splitlines()[-1]
    published = json.loads(last_line).get("published_at", "")
    if not published:
        return FreshnessCheck(name, "newest receipt has no published_at", False)
    published_at = datetime.fromisoformat(published.replace("Z", "+00:00"))
    now = now if now is not None else datetime.now(timezone.utc).timestamp()
    age_days = int(now - published_at.timestamp()) // 86_400
    return FreshnessCheck(
        name,
        f"newest receipt {age_days}d old (limit {max_age_days}d)",
        age_days <= max_age_days,
    )


def house_ingestion_freshness(
    conn,
    today: date | None = None,
    max_age_days: int = HOUSE_INGESTION_MAX_AGE_DAYS,
) -> FreshnessCheck:
    """House ingestion (house_pdf / gemini_ocr rows) vs the last completed
    NYSE session. House disclosures arrive weekly-ish and the all-years
    refresh runs weekly, so ``max_age_days`` calendar days of silence means
    House acquisition/OCR stopped (review finding 12)."""
    name = "house.ingestion.freshness"
    today = today or date.today()
    expected = previous_nyse_session(today - pd.Timedelta(days=1)).date()
    newest = _as_date(
        conn.execute(
            "SELECT MAX(disclosure_date) FROM canonical_transactions "
            "WHERE source IN ('house_pdf', 'gemini_ocr')"
        ).fetchone()[0]
    )
    if newest is None:
        return FreshnessCheck(
            name,
            "no house_pdf/gemini_ocr disclosure in canonical — House "
            "ingestion has never produced a row; run the refresh "
            "pipeline (refresh --all-years)",
            False,
        )
    age = (expected - newest).days
    return FreshnessCheck(
        name,
        f"newest house_pdf/gemini_ocr disclosure {newest} ({age}d before "
        f"last completed session {expected}; limit {max_age_days}d; fix: "
        "refresh --all-years)",
        age <= max_age_days,
    )


def ocr_backlog_freshness(
    staging_root: Path | str | None,
    now: float | None = None,
    max_age_days: int = OCR_BACKLOG_MAX_AGE_DAYS,
) -> FreshnessCheck:
    """Age of the newest staged OCR record under
    ``data/staging/glm-backlog/<year>/*.json``.

    The nightly drain stages new documents every night while a backlog
    remains, so a silent staging tree means the drain stopped (quota,
    credentials, dead cron). A missing staging root is UNKNOWN — no
    evidence, never green. Uses file mtimes, not record contents, so a
    staging-record schema change cannot blind the monitor.
    """
    name = "ocr.staging.freshness"
    if staging_root is None:
        return FreshnessCheck.unknown(
            name, "no staging root configured for the OCR freshness check"
        )
    root = Path(staging_root)
    if not root.exists():
        return FreshnessCheck.unknown(
            name,
            f"no staging root at {root}: GLM OCR staging has never run "
            "here (scripts/glm_stage.py stage creates it)",
        )
    now = now if now is not None else time.time()
    newest = max(
        (path.stat().st_mtime for path in root.glob("[0-9][0-9][0-9][0-9]/*.json")),
        default=None,
    )
    if newest is None:
        return FreshnessCheck.unknown(
            name,
            f"no staged records under {root}: backlog fully promoted and "
            "consumed, or staging was wiped — see staging.ocr.backlog",
        )
    age_days = int(max(0.0, now - newest)) // 86_400
    return FreshnessCheck(
        name,
        f"newest staged OCR record {age_days}d old under {root} "
        f"(limit {max_age_days}d; nightly drain: cron/backlog_ocr.sh)",
        age_days <= max_age_days,
    )


def end_to_end_checks(
    conn,
    receipts_path: Path | str | None = None,
    today: date | None = None,
) -> list[FreshnessCheck]:
    """The full end-product surface; never raises, failures are data."""
    today = today or date.today()

    def guard(name: str, run) -> FreshnessCheck:
        try:
            return run()
        except Exception as exc:  # a broken check must stay a visible row
            return FreshnessCheck(
                name, f"unavailable: {type(exc).__name__}: {exc}", False
            )

    return [
        guard("prices.freshness", lambda: price_freshness(conn, today)),
        guard("dividends.populated", lambda: dividends_populated(conn)),
        guard("form4.freshness", lambda: form4_freshness(conn, today)),
        guard("receipts.freshness", lambda: receipts_freshness(receipts_path)),
    ]


def monitoring_checks(
    conn,
    receipts_path: Path | str | None = None,
    staging_root: Path | str | None = None,
    today: date | None = None,
) -> list[FreshnessCheck]:
    """The full monitoring surface shared by scripts/doctor.py and the
    operator ``health`` command: canonical end-product freshness plus the
    OCR pipeline observability checks (review finding 12, 2026-10-18).
    Never raises, failures are data."""
    today = today or date.today()

    def guard(name: str, run) -> FreshnessCheck:
        try:
            return run()
        except Exception as exc:  # a broken check must stay a visible row
            return FreshnessCheck(
                name, f"unavailable: {type(exc).__name__}: {exc}", False
            )

    return [
        *end_to_end_checks(conn, receipts_path=receipts_path, today=today),
        guard("house.ingestion.freshness", lambda: house_ingestion_freshness(conn, today)),
        guard("ocr.staging.freshness", lambda: ocr_backlog_freshness(staging_root)),
    ]
