from __future__ import annotations

import json
import subprocess  # nosec B404
import sys
import uuid
from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from pathlib import Path

import pandas as pd


def _get_git_sha() -> str:
    try:
        result = subprocess.run(  # nosec B603 B607
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _get_yfinance_version() -> str:
    try:
        import yfinance
        return yfinance.__version__
    except ImportError:
        return "not installed"


@dataclass(frozen=True, slots=True)
class PriceSnapshot:
    snapshot_id: str
    created_at: str
    git_sha: str
    yfinance_version: str
    python_version: str
    requested_tickers: int
    resolved_tickers: int
    unresolved_tickers: tuple[str, ...]
    price_rows: int
    first_date: str
    last_date: str
    coverage_by_ticker: dict[str, dict] = field(default_factory=dict)


def _compute_ticker_coverage(
    prices: pd.DataFrame, ticker: str, requested_start: date, requested_end: date
) -> dict:
    if ticker not in prices.columns:
        return {
            "first": None,
            "last": None,
            "days": 0,
            "gaps": 0,
        }

    series = prices[ticker].dropna()
    if series.empty:
        return {
            "first": None,
            "last": None,
            "days": 0,
            "gaps": 0,
        }

    first = series.index.min()
    last = series.index.max()
    days = len(series)

    # Detect gaps: count business days between min and max not covered by data
    expected_dates = pd.bdate_range(first, last)
    actual_dates = set(series.index)
    gaps = sum(1 for d in expected_dates if d not in actual_dates)

    return {
        "first": str(first.date()) if hasattr(first, "date") else str(first),
        "last": str(last.date()) if hasattr(last, "date") else str(last),
        "days": days,
        "gaps": gaps,
    }


def create_snapshot(
    db,
    tickers: list[str],
    start: date,
    end: date,
) -> PriceSnapshot:
    prices = db.get_prices(tickers, start, end)

    if prices.empty:
        coverage_by_ticker = {}
        resolved = 0
        unresolved = tuple(tickers)
        total_rows = 0
        first_date = ""
        last_date = ""
    else:
        resolved_tickers_set = set(prices.columns)
        unresolved = tuple(t for t in tickers if t not in resolved_tickers_set)
        resolved = len(resolved_tickers_set)
        total_rows = len(prices)

        coverage_by_ticker = {}
        for t in tickers:
            coverage_by_ticker[t] = _compute_ticker_coverage(prices, t, start, end)

        all_dates = prices.index
        first_date = str(all_dates.min().date())
        last_date = str(all_dates.max().date())

    snapshot = PriceSnapshot(
        snapshot_id=str(uuid.uuid4()),
        created_at=datetime.now().isoformat(),
        git_sha=_get_git_sha(),
        yfinance_version=_get_yfinance_version(),
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        requested_tickers=len(tickers),
        resolved_tickers=resolved,
        unresolved_tickers=unresolved,
        price_rows=total_rows,
        first_date=first_date,
        last_date=last_date,
        coverage_by_ticker=coverage_by_ticker,
    )
    return snapshot


def save_snapshot(snapshot: PriceSnapshot, path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(asdict(snapshot), f, indent=2)


def load_snapshot(path: str = "data/price_snapshot.json") -> PriceSnapshot:
    with open(path) as f:
        data = json.load(f)
    # JSON arrays deserialize as lists; convert to tuple for frozen dataclass
    if isinstance(data.get("unresolved_tickers"), list):
        data["unresolved_tickers"] = tuple(data["unresolved_tickers"])
    return PriceSnapshot(**data)


def compare_snapshots(old: PriceSnapshot, new: PriceSnapshot) -> dict:
    old_tickers = set(old.coverage_by_ticker.keys())
    new_tickers = set(new.coverage_by_ticker.keys())

    added = sorted(new_tickers - old_tickers)
    removed = sorted(old_tickers - new_tickers)

    changed = []
    for ticker in sorted(old_tickers & new_tickers):
        old_cov = old.coverage_by_ticker[ticker]
        new_cov = new.coverage_by_ticker[ticker]
        if old_cov != new_cov:
            changed.append({
                "ticker": ticker,
                "old": old_cov,
                "new": new_cov,
            })

    return {
        "old_snapshot_id": old.snapshot_id,
        "new_snapshot_id": new.snapshot_id,
        "old_created_at": old.created_at,
        "new_created_at": new.created_at,
        "added_tickers": added,
        "removed_tickers": removed,
        "changed_coverage": changed,
        "requested_tickers_diff": new.requested_tickers - old.requested_tickers,
        "resolved_tickers_diff": new.resolved_tickers - old.resolved_tickers,
        "price_rows_diff": new.price_rows - old.price_rows,
    }
