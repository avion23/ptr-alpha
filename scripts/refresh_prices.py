#!/usr/bin/env python3
"""Refresh price data for eligible congressional-disclosure assets.

Fetch closes from yfinance for every strategy-eligible asset referenced by a
source transactions database and persist them into a throwaway temp database.
The canonical ``data/congress.duckdb`` is never opened; the only writable
database is the explicit ``--db`` path (a fresh temp DB).

Enforcements:
* exact NYSE sessions: the window end defaults to the latest completed NYSE
  session (``previous_nyse_session(today)``) and an explicit ``--end`` must be
  a session in the NYSE calendar;
* nonpositive quarantine: non-finite, zero, and negative closes are rejected
  before persistence and the persisted temp DB is re-verified afterwards;
* ticker/asset eligibility: only syntax-valid, non-quarantined, non-suspicious,
  non-reserved equity identifiers are requested (SPY benchmark always added);
* stale = unavailable: a ticker whose last close predates the window end by
  more than ``--max-staleness-days`` is reported as stale, so downstream
  consumers treat it as unavailable for recent windows;
* read-only against temp DBs: the source DB is opened read-only and the
  refresh refuses to run against an existing output DB (use ``--force``);
* unavailable recovery: when the price source's fail-closed gate rejects a
  batch (e.g. a cluster of delisted assets without yfinance history), the
  already-persisted tickers are kept and only the missing ones are retried
  individually; assets that still yield no price data in the window are
  recorded as ``unavailable`` in the report instead of aborting the refresh.

The script writes a JSON report (``--report``) and nothing else; snapshot the
temp DB with ``scripts/snapshot_prices.py`` to produce a value-hashed manifest.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from analyzer.database import Database
from analyzer.exceptions import DataSourceError
from analyzer.price_repository import (
    nyse_sessions,
    previous_nyse_session,
)
from analyzer.price_source import YFinancePriceSource
from analyzer.settings import DataSettings, Settings
from analyzer.ticker_resolver import TickerResolver

logger = logging.getLogger(__name__)

DEFAULT_PRICE_START = date(2014, 1, 1)
DEFAULT_MAX_STALENESS_DAYS = 30

_VALID_TICKER_RE = re.compile(r"^[A-Z]{1,5}([.-][A-Z]{1,2})?$")
# Non-equity identifiers rejected by the canonical transaction validation
# matrix (database._validate_ticker_origin_matrix) must not request prices.
RESERVED_NON_EQUITY_TOKENS = frozenset(
    {"COUPON", "BOND", "BONDS", "NOTE", "NOTES", "STOCK", "TICKER"}
)
BENCHMARK_TICKER = "SPY"


@dataclass(frozen=True, slots=True)
class RefreshReport:
    generation: str
    created_at: str
    window_start: str
    window_end: str
    max_staleness_days: int
    requested_assets: int
    eligible_assets: list[str]
    excluded_assets: dict[str, list[str]]
    resolved_tickers: int
    unresolved_tickers: list[str]
    unavailable_tickers: list[str]
    price_rows: int
    rejected_observations: int
    first_date: str
    last_date: str
    stale_tickers: list[str]
    source_db: str
    temp_db: str

    def to_dict(self) -> dict:
        return {
            "generation": self.generation,
            "created_at": self.created_at,
            "window": {"start": self.window_start, "end": self.window_end},
            "max_staleness_days": self.max_staleness_days,
            "requested_assets": self.requested_assets,
            "eligible_assets": self.eligible_assets,
            "excluded_assets": self.excluded_assets,
            "resolved_tickers": self.resolved_tickers,
            "unresolved_tickers": self.unresolved_tickers,
            "unavailable_tickers": self.unavailable_tickers,
            "price_rows": self.price_rows,
            "rejected_observations": self.rejected_observations,
            "first_date": self.first_date,
            "last_date": self.last_date,
            "stale_tickers": self.stale_tickers,
            "source_db": self.source_db,
            "temp_db": self.temp_db,
        }


def refresh_end_date(today: date | None = None) -> date:
    """Latest completed market session: the most recent NYSE session on or
    before today. A weekend or market holiday never yields a "today" end."""
    return previous_nyse_session(today or date.today()).date()


def _clean_asset(token: str) -> str | None:
    if token is None:
        return None
    token = str(token).strip().upper()
    if not token or token == "NAN":
        return None
    return token


def select_eligible_assets(
    tickers: list[str], *, include_benchmark: bool = True
) -> tuple[list[str], dict[str, list[str]]]:
    """Filter distinct raw tickers down to refresh-eligible equity assets.

    Eligibility is asset-level, not strategy-level: temporal rename/acquisition
    resolution needs a trade date and is delegated to the price source and to
    backtest-time callers. Here we only reject tokens that are provably not
    equity identities: syntax-invalid strings, reserved non-equity tokens, and
    TickerResolver-quarantined/suspicious parser artifacts. The SPY benchmark
    is always appended when ``include_benchmark`` is set.
    """
    resolver = TickerResolver()
    eligible: list[str] = []
    excluded: dict[str, list[str]] = {}
    seen: set[str] = set()
    for raw in tickers:
        token = _clean_asset(raw)
        if token is None or token in seen:
            continue
        seen.add(token)
        if _VALID_TICKER_RE.fullmatch(token) is None:
            excluded.setdefault("invalid_syntax", []).append(token)
            continue
        if token in RESERVED_NON_EQUITY_TOKENS:
            excluded.setdefault("reserved_non_equity", []).append(token)
            continue
        if resolver.resolve(token).status == "quarantined":
            excluded.setdefault("quarantined_or_suspicious", []).append(token)
            continue
        eligible.append(token)
    if include_benchmark and BENCHMARK_TICKER not in eligible:
        eligible.append(BENCHMARK_TICKER)
    return sorted(eligible), {reason: sorted(tokens) for reason, tokens in excluded.items()}


def _verify_persisted_prices(db: Database, start: date, end: date) -> int:
    """Return the number of non-finite/non-positive rows in the temp DB within
    the refresh window. The price repository quarantines those on upsert, so a
    healthy refresh reports zero."""
    rows = db.conn.execute(
        """
        SELECT COUNT(*) FROM prices
        WHERE date BETWEEN ? AND ?
          AND (close <= 0 OR NOT isfinite(close))
        """,
        [start, end],
    ).fetchone()[0]
    return int(rows)


def _compute_staleness(
    db: Database, end: date, max_staleness_days: int
) -> list[str]:
    """Tickers whose last close predates the window end by more than the
    staleness budget. Stale means unavailable for recent windows."""
    rows = db.conn.execute(
        """
        SELECT ticker, MAX(date) AS last_date FROM prices GROUP BY ticker
        """
    ).fetchall()
    stale = [
        str(ticker)
        for ticker, last_date in rows
        if last_date is not None and (end - pd.Timestamp(last_date).date()).days > max_staleness_days
    ]
    return sorted(stale)


def _source_tickers(source_db: Path) -> list[str]:
    db = Database(source_db, read_only=True)
    try:
        rows = db.conn.execute(
            "SELECT DISTINCT ticker FROM transactions WHERE ticker IS NOT NULL"
        ).fetchall()
    finally:
        db.close()
    return [str(row[0]) for row in rows]


def _persisted_tickers(db: Database, start: date, end: date) -> set[str]:
    rows = db.conn.execute(
        """
        SELECT DISTINCT ticker FROM prices
        WHERE date BETWEEN ? AND ? AND close > 0 AND isfinite(close)
        """,
        [start, end],
    ).fetchall()
    return {str(row[0]) for row in rows}


def refresh_prices(
    source_db: Path,
    temp_db: Path,
    *,
    start: date = DEFAULT_PRICE_START,
    end: date | None = None,
    max_staleness_days: int = DEFAULT_MAX_STALENESS_DAYS,
    force: bool = False,
) -> RefreshReport:
    """Refresh eligible asset prices into a throwaway temp database.

    ``source_db`` is opened read-only; ``temp_db`` must not already exist
    unless ``force`` is set (the canonical database is never touched).
    """
    if end is None:
        end = refresh_end_date()
    if end < start:
        raise ValueError(f"refresh end {end} precedes start {start}")
    expected_sessions = nyse_sessions(start, end)
    if expected_sessions.empty:
        raise ValueError(f"no NYSE sessions in window {start}..{end}")
    if pd.Timestamp(end) not in expected_sessions:
        raise ValueError(
            f"refresh end {end} is not a completed NYSE session; "
            f"latest completed session is {expected_sessions[-1].date()}"
        )
    if max_staleness_days < 0:
        raise ValueError("max_staleness_days must be non-negative")

    source_db = Path(source_db)
    temp_db = Path(temp_db)
    if not source_db.exists():
        raise FileNotFoundError(f"source database not found: {source_db}")
    if temp_db.exists() and not force:
        raise FileExistsError(
            f"temp database already exists: {temp_db} (use --force to overwrite)"
        )

    tickers = _source_tickers(source_db)
    eligible, excluded = select_eligible_assets(tickers)
    if not eligible:
        raise ValueError("no eligible assets found in the source database")

    temp_db.parent.mkdir(parents=True, exist_ok=True)
    if temp_db.exists() and force:
        temp_db.unlink()
    for sidecar in ("", ".wal"):
        leftover = Path(str(temp_db) + sidecar)
        if leftover.exists():
            leftover.unlink()

    settings = Settings(data=DataSettings(data_dir=str(temp_db.parent)))
    db = Database(temp_db, read_only=False)
    try:
        price_source = YFinancePriceSource(settings, read_only=False, db=db)
        try:
            try:
                price_source.get_prices(eligible, start, end)
            except DataSourceError:
                # The batch fetch persisted every ticker it could. Only the
                # genuinely unresolvable assets remain missing; retry each one
                # individually so transient failures are recovered instead of
                # aborting the refresh over a cluster of delisted assets.
                for ticker in sorted(set(eligible) - _persisted_tickers(db, start, end)):
                    try:
                        price_source.get_prices([ticker], start, end)
                    except DataSourceError:
                        pass
        finally:
            price_source.close()
        # Assets with no price history in the window (including those whose
        # empty download was masked by the cached benchmark column) are
        # recorded explicitly as unavailable.
        unavailable = sorted(set(eligible) - _persisted_tickers(db, start, end))
        matrix = db.get_prices(eligible, start, end)
        rejected = _verify_persisted_prices(db, start, end)
        stale = _compute_staleness(db, end, max_staleness_days)
    finally:
        db.close()

    resolved = [t for t in eligible if t in matrix.columns]
    unresolved = sorted(set(eligible) - set(matrix.columns))
    first_date = (
        str(matrix.index.min().date()) if not matrix.empty else ""
    )
    last_date = str(matrix.index.max().date()) if not matrix.empty else ""

    return RefreshReport(
        generation="",
        created_at=datetime.now().isoformat(),
        window_start=str(start),
        window_end=str(end),
        max_staleness_days=max_staleness_days,
        requested_assets=len(tickers),
        eligible_assets=eligible,
        excluded_assets=excluded,
        resolved_tickers=len(resolved),
        unresolved_tickers=unresolved,
        unavailable_tickers=unavailable,
        price_rows=int(matrix.notna().sum().sum()),
        rejected_observations=rejected,
        first_date=first_date,
        last_date=last_date,
        stale_tickers=stale,
        source_db=str(source_db),
        temp_db=str(temp_db),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Refresh eligible asset prices into a throwaway temp database"
    )
    parser.add_argument("--source-db", required=True, type=Path, help="read-only transactions source DB")
    parser.add_argument("--db", required=True, type=Path, help="temp output DB (must not exist)")
    parser.add_argument("--start", type=date.fromisoformat, default=DEFAULT_PRICE_START)
    parser.add_argument("--end", type=date.fromisoformat, default=None, help="defaults to the latest completed NYSE session")
    parser.add_argument("--max-staleness-days", type=int, default=DEFAULT_MAX_STALENESS_DAYS)
    parser.add_argument("--report", type=Path, default=None, help="write the JSON refresh report here")
    parser.add_argument("--force", action="store_true", help="overwrite an existing temp DB")
    args = parser.parse_args(argv)

    try:
        report = refresh_prices(
            args.source_db,
            args.db,
            start=args.start,
            end=args.end,
            max_staleness_days=args.max_staleness_days,
            force=args.force,
        )
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        print(f"refresh failed: {exc}", file=sys.stderr)
        return 1

    print(
        f"refresh: rows={report.price_rows} resolved={report.resolved_tickers}/"
        f"{len(report.eligible_assets)} range={report.first_date}..{report.last_date} "
        f"window_end={report.window_end} rejected={report.rejected_observations} "
        f"stale={len(report.stale_tickers)}"
    )
    if report.unresolved_tickers:
        print(
            "  unresolved: "
            + ", ".join(report.unresolved_tickers[:20])
            + ("..." if len(report.unresolved_tickers) > 20 else "")
        )
    if report.unavailable_tickers:
        print(
            "  unavailable (no price history in window): "
            + ", ".join(report.unavailable_tickers[:20])
            + ("..." if len(report.unavailable_tickers) > 20 else "")
        )
    if report.stale_tickers:
        print(
            "  stale (unavailable for recent windows): "
            + ", ".join(report.stale_tickers[:20])
            + ("..." if len(report.stale_tickers) > 20 else "")
        )
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report.to_dict(), indent=2))
        print(f"report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
