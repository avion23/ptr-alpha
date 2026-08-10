#!/usr/bin/env python3
"""Freeze a value-hashed price snapshot manifest from a temp price database.

Opens the temp database **read-only** (never writes to any database) and
writes a staged snapshot directory:

* ``manifest.json`` — value-hashed manifest binding the price data, the price
  pipeline code, and the repo configuration (SHA-256 over each, plus an
  aggregate over per-file digests);
* ``snapshot.json`` — the canonical ``PriceSnapshot`` manifest (existing
  ``price_snapshot.load_snapshot`` format);
* ``prices.parquet`` — the long (ticker, date, close) price artifact for
  downstream database loading.

Enforcements mirror ``scripts/refresh_prices.py``: the window end defaults to
the latest completed NYSE session and must be one, coverage gaps are counted
against the exact NYSE trading calendar, nonpositive observations never enter
the manifest (they are quarantined at persistence and re-verified), and
tickers whose last close is older than ``--max-staleness-days`` relative to
the window end are reported as stale (unavailable for recent windows).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from analyzer.database import Database
from analyzer.price_repository import nyse_sessions, previous_nyse_session
from analyzer.price_snapshot import create_snapshot, save_snapshot

REPO_ROOT = Path(__file__).resolve().parent.parent

# Price pipeline code whose behavior is bound into the snapshot.
CODE_FILES = [
    "src/analyzer/price_source.py",
    "src/analyzer/price_repository.py",
    "src/analyzer/price_snapshot.py",
    "src/analyzer/ticker_resolver.py",
    "src/analyzer/database.py",
    "scripts/refresh_prices.py",
    "scripts/snapshot_prices.py",
]
# Configuration inputs that can change price semantics.
CONFIG_FILES = [
    "config.toml",
    "pyproject.toml",
]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_files(rel_paths: list[str]) -> tuple[dict[str, str], str]:
    """Per-file SHA-256 digests plus an aggregate over existing files.

    Missing files are recorded as ``"missing"`` so the manifest states exactly
    which inputs were bound; the aggregate covers only files that exist.
    """
    digests: dict[str, str] = {}
    for rel in rel_paths:
        path = REPO_ROOT / rel
        digests[rel] = file_sha256(path) if path.is_file() else "missing"
    existing = sorted(
        (rel, digest) for rel, digest in digests.items() if digest != "missing"
    )
    aggregate = hashlib.sha256()
    for rel, digest in existing:
        aggregate.update(rel.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(digest.encode("utf-8"))
        aggregate.update(b"\n")
    return digests, aggregate.hexdigest()


def _stale_tickers(coverage_by_ticker: dict, end: date, max_staleness_days: int) -> list[str]:
    stale: list[str] = []
    for ticker, cov in coverage_by_ticker.items():
        last = cov.get("last")
        if not last:
            continue
        last_date = pd.Timestamp(last).date()
        if (end - last_date).days > max_staleness_days:
            stale.append(ticker)
    return sorted(stale)


def _snapshot_end_date(end: date | None) -> date:
    if end is not None:
        return end
    return previous_nyse_session(date.today()).date()


def build_manifest(
    db_path: Path,
    *,
    start: date,
    end: date,
    generation: str,
    out_dir: Path,
    max_staleness_days: int = 30,
    code_files: list[str] | None = None,
    config_files: list[str] | None = None,
) -> dict:
    """Build and persist the staged snapshot artifacts; return the manifest."""
    end = _snapshot_end_date(end)
    if end < start:
        raise ValueError(f"snapshot end {end} precedes start {start}")
    sessions = nyse_sessions(start, end)
    if sessions.empty:
        raise ValueError(f"no NYSE sessions in window {start}..{end}")
    if pd.Timestamp(end) not in sessions:
        raise ValueError(
            f"snapshot end {end} is not a completed NYSE session; "
            f"latest completed session is {sessions[-1].date()}"
        )

    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"price database not found: {db_path}")

    db = Database(db_path, read_only=True)
    try:
        tickers = sorted(
            str(row[0])
            for row in db.conn.execute(
                "SELECT DISTINCT ticker FROM prices WHERE date BETWEEN ? AND ?",
                [start, end],
            ).fetchall()
        )
        snapshot = create_snapshot(db, tickers, start, end)
        rows = db.conn.execute(
            """
            SELECT ticker, date, close FROM prices
            WHERE date BETWEEN ? AND ? AND close > 0 AND isfinite(close)
            ORDER BY ticker, date
            """,
            [start, end],
        ).fetchdf()
    finally:
        db.close()

    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = out_dir / "snapshot.json"
    save_snapshot(snapshot, snapshot_path)
    if rows.empty:
        prices_long = pd.DataFrame(columns=["ticker", "date", "close"])
    else:
        prices_long = rows
    prices_long.to_parquet(out_dir / "prices.parquet", index=False)

    code_digests, code_hash = hash_files(code_files or CODE_FILES)
    config_digests, config_hash = hash_files(config_files or CONFIG_FILES)

    manifest = {
        "generation": generation,
        "created_at": datetime.now().isoformat(),
        "git_sha": snapshot.git_sha,
        "yfinance_version": snapshot.yfinance_version,
        "python_version": snapshot.python_version,
        "window": {"start": str(start), "end": str(end)},
        "max_staleness_days": max_staleness_days,
        "data_hash": snapshot.value_hash,
        "value_hash": snapshot.value_hash,
        "code_hash": code_hash,
        "code_files": code_digests,
        "config_hash": config_hash,
        "config_files": config_digests,
        "tickers": {
            "requested": snapshot.requested_tickers,
            "resolved": snapshot.resolved_tickers,
            "unresolved": list(snapshot.unresolved_tickers),
            "stale": _stale_tickers(snapshot.coverage_by_ticker, end, max_staleness_days),
        },
        "price_rows": snapshot.price_rows,
        "first_date": snapshot.first_date,
        "last_date": snapshot.last_date,
        "artifacts": {
            "manifest": "manifest.json",
            "snapshot": "snapshot.json",
            "prices": "prices.parquet",
        },
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Freeze a value-hashed price snapshot manifest (read-only)"
    )
    parser.add_argument("--db", required=True, type=Path, help="temp price DB, opened read-only")
    parser.add_argument("--start", type=date.fromisoformat, default=date(2014, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=None, help="defaults to the latest completed NYSE session")
    parser.add_argument("--generation", required=True, help="generation id, e.g. gen-live-20260809")
    parser.add_argument("--out", required=True, type=Path, help="staged snapshot output directory")
    parser.add_argument("--max-staleness-days", type=int, default=30)
    args = parser.parse_args(argv)

    try:
        manifest = build_manifest(
            args.db,
            start=args.start,
            end=args.end,
            generation=args.generation,
            out_dir=args.out,
            max_staleness_days=args.max_staleness_days,
        )
    except (ValueError, FileNotFoundError) as exc:
        print(f"snapshot failed: {exc}", file=sys.stderr)
        return 1

    print(
        f"snapshot: generation={manifest['generation']} rows={manifest['price_rows']} "
        f"tickers={manifest['tickers']['resolved']}/{manifest['tickers']['requested']} "
        f"range={manifest['first_date']}..{manifest['last_date']} "
        f"data_hash={manifest['data_hash'][:16]} code_hash={manifest['code_hash'][:16]} "
        f"config_hash={manifest['config_hash'][:16]}"
    )
    print(f"staged: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
