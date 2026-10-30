"""Rebaseline adjusted price generations onto the post-revision vendor vintage.

Operator-approved repair for the 2026-10-24..27 corpus-wide vendor revision
(1,478 conflicted tickers; NVDA e1418-vs-6a82 constant 0.1119% factor from
2020-12-01; TSM 0.2683% with no dividend event in that window). For every
ticker whose ``price_generation_state`` carries a divergent
``alternate_generation``:

1. The alternate becomes the keeper (it is the latest merge's fresh fetch).
2. Every other adjusted-basis generation is kept only when it certifies
   against the keeper with the production comparison
   (:func:`analyzer.price_source.certify_adjusted_price_generations`);
   disagreeing vintages are deleted from ``price_generations`` and
   ``price_generation_values``.
3. ``prices`` rows for the ticker are replaced by the keeper's closes (the
   merge contract is additive-only, so replacement is the only way to move
   the base onto the new vintage).
4. State becomes ``(keeper, unknown, NULL reason-free alternate, note)`` —
   fail-closed until ``certify_price_generations.py`` re-certifies on at
   least two agreeing fetches.

Dry-run is the default. ``--apply`` requires a backup of the canonical DB
whose byte size matches the live file, and changes only ``prices``,
``price_generations``, ``price_generation_values`` and
``price_generation_state``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd

from analyzer.price_source import certify_adjusted_price_generations
from analyzer.persistence.ownership import database_owner, resolve_read_path

DEFAULT_DATABASE = Path("/home/ubuntu/insider-trading/data/congress.duckdb")
REQUIRED_BACKUP = Path(
    "/home/ubuntu/insider-trading/data/backups/congress-pre-rebaseline-20261027.duckdb"
)
DEFAULT_CHANGE_LOG = Path(
    "/home/ubuntu/insider-trading/data/logs/rebaseline-20261027/changes.json"
)
ADJUSTED_BASIS = "yfinance-adjusted-close-v1"
PENDING_REASON = (
    "rebaseline 2026-10-27: pre-revision generations retired; "
    "pending recertification on post-revision fetches"
)
STATE_NOTE = "pre-revision vintage retired"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scope(
    con: duckdb.DuckDBPyConnection, limit: int | None
) -> list[tuple[str, str]]:
    query = (
        "SELECT ticker, alternate_generation FROM price_generation_state "
        "WHERE alternate_generation IS NOT NULL ORDER BY ticker"
    )
    if limit is not None:
        if limit < 0:
            raise SystemExit("--limit must be non-negative")
        query += f" LIMIT {int(limit)}"
    return con.execute(query).fetchall()


def _dividends(con: duckdb.DuckDBPyConnection) -> dict[str, pd.DataFrame]:
    frame = con.execute(
        "SELECT ticker, ex_date, cash_amount FROM dividends ORDER BY ticker, ex_date"
    ).fetchdf()
    return {
        str(ticker): rows[["ex_date", "cash_amount"]].reset_index(drop=True)
        for ticker, rows in frame.groupby("ticker", sort=False)
    }


def _generations(
    con: duckdb.DuckDBPyConnection, tickers: list[str]
) -> dict[str, dict[str, pd.DataFrame]]:
    placeholders = ", ".join("?" for _ in tickers)
    rows = con.execute(
        f"""
        SELECT v.ticker, v.generation, v.date, v.close, v.adjustment_factor
        FROM price_generation_values v
        JOIN price_generations g
          ON g.ticker = v.ticker AND g.generation = v.generation
        WHERE g.basis = ? AND v.ticker IN ({placeholders})
        ORDER BY v.ticker, v.generation, v.date
        """,
        [ADJUSTED_BASIS, *tickers],
    ).fetchdf()
    out: dict[str, dict[str, pd.DataFrame]] = {}
    if rows.empty:
        return out
    for (ticker, generation), frame in rows.groupby(
        ["ticker", "generation"], sort=False
    ):
        out.setdefault(ticker, {})[generation] = frame[
            ["date", "close", "adjustment_factor"]
        ].reset_index(drop=True)
    return out


def build_plan(
    con: duckdb.DuckDBPyConnection, limit: int | None
) -> tuple[list[dict], dict]:
    scope = _scope(con, limit)
    if not scope:
        return [], {"dividend_rows": 0}
    dividends = _dividends(con)
    tickers = [ticker for ticker, _ in scope]
    generations = _generations(con, tickers)
    plan: list[dict] = []
    no_dividends = pd.DataFrame(columns=["ex_date", "cash_amount"])
    for ticker, keeper in scope:
        ticker_generations = generations.get(ticker, {})
        if keeper not in ticker_generations:
            raise SystemExit(
                f"{ticker}: keeper {keeper} has no adjusted values — aborting"
            )
        kept: list[str] = []
        deleted: list[str] = []
        for generation, frame in ticker_generations.items():
            if generation == keeper:
                continue
            verdict = certify_adjusted_price_generations(
                {keeper: ticker_generations[keeper], generation: frame},
                dividends.get(ticker, no_dividends),
            )
            (kept if verdict.get("certified") else deleted).append(generation)
        plan.append(
            {
                "ticker": ticker,
                "keeper": keeper,
                "kept": sorted(kept),
                "deleted": sorted(deleted),
            }
        )
    return plan, {"dividend_rows": sum(map(len, dividends.values()))}


def _plan_counts(con: duckdb.DuckDBPyConnection, plan: list[dict]) -> dict:
    scope_frame = pd.DataFrame(
        [(entry["ticker"], entry["keeper"]) for entry in plan],
        columns=["ticker", "keeper"],
    )
    delete_frame = pd.DataFrame(
        [
            (entry["ticker"], generation)
            for entry in plan
            for generation in entry["deleted"]
        ],
        columns=["ticker", "generation"],
    )
    con.register("_plan_scope", scope_frame)
    con.register("_plan_delete", delete_frame)
    counts = {
        "values_deleted": con.execute(
            """
            SELECT COUNT(*) FROM price_generation_values v
            JOIN _plan_delete d
              ON d.ticker = v.ticker AND d.generation = v.generation
            """
        ).fetchone()[0],
        "prices_deleted": con.execute(
            "SELECT COUNT(*) FROM prices p JOIN _plan_scope s ON s.ticker = p.ticker"
        ).fetchone()[0],
        "prices_inserted": con.execute(
            """
            SELECT COUNT(*) FROM price_generation_values v
            JOIN _plan_scope s
              ON s.ticker = v.ticker AND s.keeper = v.generation
            """
        ).fetchone()[0],
    }
    con.unregister("_plan_scope")
    con.unregister("_plan_delete")
    return counts


def apply_plan(
    database: Path, backup: Path, limit: int | None, change_log: Path
) -> tuple[dict, dict]:
    with (
        database_owner(database, purpose="price rebaseline maintenance"),
        duckdb.connect(str(database)) as con,
    ):
        plan, summary = _plan_summary(con, limit)
        result = _apply_plan_owned(database, backup, plan, change_log, con)
        return summary, result


def _plan_summary(con, limit):
    plan, plan_meta = build_plan(con, limit)
    return plan, {
        "scope_tickers": len(plan),
        "tickers_with_deletions": sum(1 for entry in plan if entry["deleted"]),
        **_plan_counts(con, plan),
        **plan_meta,
    }


def _apply_plan_owned(database, backup, plan, change_log, con):
    if backup != REQUIRED_BACKUP:
        raise SystemExit(f"--backup must be exactly {REQUIRED_BACKUP}")
    if not backup.is_file() or backup.stat().st_size != database.stat().st_size:
        raise SystemExit(
            "required backup is missing or its size differs from the canonical DB"
        )

    stats: dict = {
        "applied_at": datetime.now(timezone.utc).isoformat(),
        "backup": str(backup),
        "backup_bytes": backup.stat().st_size,
        "scope_tickers": len(plan),
        "tickers_with_deletions": sum(1 for entry in plan if entry["deleted"]),
        "values_deleted": 0,
        "headers_deleted": 0,
        "prices_deleted": 0,
        "prices_inserted": 0,
        "state_rows_written": 0,
    }
    try:
        con.execute("BEGIN TRANSACTION")
        con.execute("CREATE TEMP TABLE _rebase_scope (ticker VARCHAR, keeper VARCHAR)")
        con.execute(
            "CREATE TEMP TABLE _rebase_delete (ticker VARCHAR, generation VARCHAR)"
        )
        con.executemany(
            "INSERT INTO _rebase_scope VALUES (?,?)",
            [(entry["ticker"], entry["keeper"]) for entry in plan],
        )
        con.executemany(
            "INSERT INTO _rebase_delete VALUES (?,?)",
            [
                (entry["ticker"], generation)
                for entry in plan
                for generation in entry["deleted"]
            ],
        )

        stats["values_deleted"] = con.execute(
            """
            SELECT COUNT(*) FROM price_generation_values v
            JOIN _rebase_delete d
              ON d.ticker = v.ticker AND d.generation = v.generation
            """
        ).fetchone()[0]
        con.execute(
            """
            DELETE FROM price_generation_values v USING _rebase_delete d
            WHERE v.ticker = d.ticker AND v.generation = d.generation
            """
        )
        stats["headers_deleted"] = con.execute(
            """
            SELECT COUNT(*) FROM price_generations g
            JOIN _rebase_delete d
              ON d.ticker = g.ticker AND d.generation = g.generation
            WHERE g.basis = ?
            """,
            [ADJUSTED_BASIS],
        ).fetchone()[0]
        con.execute(
            """
            DELETE FROM price_generations g USING _rebase_delete d
            WHERE g.ticker = d.ticker AND g.generation = d.generation
              AND g.basis = ?
            """,
            [ADJUSTED_BASIS],
        )

        stats["prices_deleted"] = con.execute(
            """
            SELECT COUNT(*) FROM prices p
            JOIN _rebase_scope s ON s.ticker = p.ticker
            """
        ).fetchone()[0]
        con.execute(
            "DELETE FROM prices p USING _rebase_scope s WHERE p.ticker = s.ticker"
        )
        stats["prices_inserted"] = con.execute(
            """
            SELECT COUNT(*) FROM price_generation_values v
            JOIN _rebase_scope s
              ON s.ticker = v.ticker AND s.keeper = v.generation
            """
        ).fetchone()[0]
        con.execute(
            """
            INSERT INTO prices (ticker, date, close)
            SELECT v.ticker, v.date, v.close
            FROM price_generation_values v
            JOIN _rebase_scope s
              ON s.ticker = v.ticker AND s.keeper = v.generation
            """
        )

        con.execute(
            """
            UPDATE price_generation_state SET
                served_generation = alternate_generation,
                alternate_generation = NULL,
                comparability = 'unknown',
                reason = ?
            WHERE ticker IN (SELECT ticker FROM _rebase_scope)
            """,
            [PENDING_REASON],
        )
        stats["state_rows_written"] = con.execute(
            """
            SELECT COUNT(*) FROM price_generation_state s
            JOIN _rebase_scope sc ON sc.ticker = s.ticker
            WHERE s.reason = ?
            """,
            [PENDING_REASON],
        ).fetchone()[0]
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    con.execute("CHECKPOINT")
    stats["table_counts_after"] = _table_counts(con)
    stats["state_after"] = con.execute(
        "SELECT comparability, COUNT(*) FROM price_generation_state GROUP BY 1 ORDER BY 2 DESC"
    ).fetchall()
    stats["tickers"] = plan
    change_log.parent.mkdir(parents=True, exist_ok=True)
    change_log.write_text(json.dumps(stats, indent=1, default=str))
    return {key: value for key, value in stats.items() if key != "tickers"}


def _table_counts(con: duckdb.DuckDBPyConnection) -> dict:
    counts = {}
    for table in (
        "prices",
        "dividends",
        "price_generations",
        "price_generation_values",
        "price_generation_state",
    ):
        counts[table] = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--backup", type=Path, default=REQUIRED_BACKUP)
    parser.add_argument("--change-log", type=Path, default=DEFAULT_CHANGE_LOG)
    parser.add_argument("--apply", action="store_true", help="publish the rebaseline")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="restrict to the first N tickers (smoke test)",
    )
    args = parser.parse_args()

    if args.apply:
        summary, result = apply_plan(args.database, args.backup, args.limit, args.change_log)
    else:
        with duckdb.connect(str(resolve_read_path(args.database)), read_only=True) as con:
            _, summary = _plan_summary(con, args.limit)
    print(json.dumps(summary, indent=1, default=str))
    if not args.apply:
        return 0
    print(json.dumps(result, indent=1, default=str))
    print(f"change log: {args.change_log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
