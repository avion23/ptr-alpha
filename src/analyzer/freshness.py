"""End-to-end freshness of the canonical end products (monitoring only).

The 2026-10-18 cron incident: automated writers exited 0 for weeks while
their output never reached canonical, because every component measured
success at its own boundary and the end-product surface (prices, dividends,
form4, receipts) was measured nowhere. This module owns that surface so the
daily doctor (scripts/doctor.py) and the operator-facing ``health`` command
report from one implementation instead of drifting apart.

Every function is read-only and returns a FreshnessCheck rather than
raising: a monitor that crashes on an unexpected schema stops monitoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
import json
from pathlib import Path

import pandas as pd

from analyzer.price_repository import previous_nyse_session

MAX_PRICE_LAG_SESSIONS = 8
FORM4_MAX_AGE_DAYS = 7
RECEIPTS_MAX_AGE_DAYS = 8


@dataclass(frozen=True)
class FreshnessCheck:
    name: str
    detail: str
    ok: bool

    @property
    def mark(self) -> str:
        return "PASS" if self.ok else "FAIL"


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
