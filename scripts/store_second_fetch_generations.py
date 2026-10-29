"""Store the post-rebaseline second fetch as certification evidence.

Companion to ``rebaseline_price_generations.py``. After the rebaseline each
scope ticker keeps exactly one adjusted generation (the post-revision
keeper), so ``certify_adjusted_price_generations`` cannot run — it requires
at least two stored fetches. This script reads a fresh throwaway refresh
(fetched independently with an earlier ``--end`` so its content hash differs
from the keeper) and persists the per-ticker generation through the
production writer :func:`analyzer.price_source.store_generations`.

Only scope tickers that still hold fewer than two adjusted generations are
topped up; nothing else is written. Dry-run is the default.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd

from analyzer.persistence.database import Database
from analyzer.price_source import store_generations

DEFAULT_TEMP_DB = Path(
    "/tmp/opencode/reco-20261025/rebaseline-h2/prices.duckdb"
)
DEFAULT_DATABASE = Path("/home/ubuntu/insider-trading/data/congress.duckdb")
DEFAULT_CHANGE_LOG = Path(
    "/home/ubuntu/insider-trading/data/logs/rebaseline-20261027/second-fetch.json"
)
ADJUSTED_BASIS = "yfinance-adjusted-close-v1"


def _scope_keepers(con: duckdb.DuckDBPyConnection) -> list[tuple[str, str]]:
    """Scope tickers (reason marks the rebaseline) and their keeper."""
    return con.execute(
        """
        SELECT ticker, served_generation FROM price_generation_state
        WHERE reason LIKE 'rebaseline 2026-10-27:%'
        ORDER BY ticker
        """
    ).fetchall()


def _adjusted_counts(
    con: duckdb.DuckDBPyConnection, tickers: list[str]
) -> dict[str, int]:
    placeholders = ", ".join("?" for _ in tickers)
    rows = con.execute(
        f"""
        SELECT ticker, COUNT(DISTINCT generation) FROM price_generations
        WHERE basis = ? AND ticker IN ({placeholders})
        GROUP BY ticker
        """,
        [ADJUSTED_BASIS, *tickers],
    ).fetchall()
    return dict(rows)


def _temp_generation(
    temp: duckdb.DuckDBPyConnection, ticker: str, generation: str
) -> tuple[pd.Series, pd.Series] | None:
    rows = temp.execute(
        """
        SELECT date, close, adjustment_factor FROM price_generation_values
        WHERE ticker = ? AND generation = ? ORDER BY date
        """,
        [ticker, generation],
    ).fetchdf()
    if rows.empty:
        return None
    frame = rows.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    series = frame.set_index("date")["close"].astype(float)
    factors = frame.set_index("date")["adjustment_factor"].astype(float)
    return series, factors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temp-db", type=Path, default=DEFAULT_TEMP_DB)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--change-log", type=Path, default=DEFAULT_CHANGE_LOG)
    parser.add_argument(
        "--apply", action="store_true", help="store the evidence generations"
    )
    args = parser.parse_args()

    canonical = Database(str(args.database), read_only=True)
    scope = _scope_keepers(canonical.conn)
    if not scope:
        raise SystemExit("no rebaselined tickers found — nothing to top up")
    counts = _adjusted_counts(canonical.conn, [ticker for ticker, _ in scope])
    canonical.close()

    targets = [
        (ticker, keeper)
        for ticker, keeper in scope
        if counts.get(ticker, 0) < 2
    ]
    skipped = len(scope) - len(targets)
    print(
        json.dumps(
            {
                "rebaselined_tickers": len(scope),
                "already_two_or_more": skipped,
                "targets": len(targets),
            }
        )
    )
    if not targets:
        return 0
    if not args.temp_db.is_file():
        raise SystemExit(f"temp fetch DB missing: {args.temp_db}")

    temp = duckdb.connect(str(args.temp_db), read_only=True)
    entries = []
    missing_gen: list[str] = []
    collisions: list[str] = []
    for ticker, keeper in targets:
        row = temp.execute(
            """
            SELECT generation FROM price_generations
            WHERE ticker = ? AND basis = ?
            ORDER BY created_at DESC LIMIT 1
            """,
            [ticker, ADJUSTED_BASIS],
        ).fetchone()
        if row is None:
            missing_gen.append(ticker)
            continue
        generation = row[0]
        if generation == keeper:
            collisions.append(ticker)
            continue
        loaded = _temp_generation(temp, ticker, generation)
        if loaded is None:
            missing_gen.append(ticker)
            continue
        series, factors = loaded
        entries.append((ticker, generation, series, factors, ADJUSTED_BASIS))
    temp.close()

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "temp_db": str(args.temp_db),
        "entries": len(entries),
        "unusable_targets": {
            "missing_temp_generation": missing_gen,
            "collision_with_keeper": collisions,
        },
    }
    if not args.apply:
        sample = [
            {"ticker": ticker, "generation": generation}
            for ticker, generation, _, _, _ in entries[:10]
        ]
        print(json.dumps({**summary, "sample": sample}, indent=1))
        return 0
    if not entries:
        raise SystemExit("no usable temp generations — nothing to store")

    canonical = Database(str(args.database))
    conn = canonical.conn
    try:
        conn.execute("BEGIN TRANSACTION")
        store_generations(canonical, entries)
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        canonical.close()
        raise
    conn.execute("CHECKPOINT")
    summary["stored"] = len(entries)
    summary["unusable_targets"] = {
        "missing_temp_generation": missing_gen,
        "collision_with_keeper": collisions,
    }
    summary["state_after"] = conn.execute(
        "SELECT comparability, COUNT(*) FROM price_generation_state GROUP BY 1 ORDER BY 2 DESC"
    ).fetchall()
    canonical.close()
    args.change_log.parent.mkdir(parents=True, exist_ok=True)
    args.change_log.write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps(summary, indent=1, default=str))
    print(f"change log: {args.change_log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
