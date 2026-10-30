"""Certify in-DB adjusted generations; optionally publish only state rows.

Dry-run is the default. ``--apply`` requires the specifically requested backup
and changes only ``price_generation_state`` through its production owner.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import duckdb
import pandas as pd
from analyzer.persistence.ownership import database_owner, resolve_read_path

from analyzer.price_source import (
    _PRICE_BASIS,
    certify_adjusted_price_generations,
    publish_price_generation_certification,
)

DEFAULT_DATABASE = Path("/home/ubuntu/insider-trading/data/congress.duckdb")
REQUIRED_BACKUP = Path(
    "/home/ubuntu/insider-trading/data/backups/congress-pre-cert-20261025.duckdb"
)
DEFAULT_CHANGE_LOG = Path("/tmp/opencode/reco-20261025/fix-w1/state-changes.json")


def _read_certifications(connection):
    headers = connection.execute(
        "SELECT ticker, generation FROM price_generations WHERE basis=? "
        "ORDER BY ticker,generation",
        [_PRICE_BASIS],
    ).fetchall()
    by_ticker: dict[str, list[str]] = {}
    for ticker, generation in headers:
        by_ticker.setdefault(ticker, []).append(generation)

    states = {
        ticker: {
            "generation_id": served,
            "comparability_status": status,
            "alternate_generation": alternate,
            "reason": reason,
        }
        for ticker, served, status, alternate, reason in connection.execute(
            "SELECT ticker,served_generation,comparability,alternate_generation,reason "
            "FROM price_generation_state"
        ).fetchall()
    }
    all_dividends = connection.execute(
        "SELECT ticker,ex_date,cash_amount FROM dividends ORDER BY ticker,ex_date"
    ).fetchdf()
    dividends_by_ticker = {
        str(ticker): frame[["ex_date", "cash_amount"]].reset_index(drop=True)
        for ticker, frame in all_dividends.groupby("ticker", sort=False)
    }

    results = {}
    for ticker, generation_ids in by_ticker.items():
        rows = connection.execute(
            "SELECT generation,date,close,adjustment_factor "
            "FROM price_generation_values WHERE ticker=? AND basis=? "
            "ORDER BY generation,date",
            [ticker, _PRICE_BASIS],
        ).fetchdf()
        generations = {
            generation: frame[["date", "close", "adjustment_factor"]].reset_index(drop=True)
            for generation, frame in rows.groupby("generation", sort=False)
        }
        results[ticker] = {
            "certification": certify_adjusted_price_generations(
                generations,
                dividends_by_ticker.get(
                    ticker, pd.DataFrame(columns=["ex_date", "cash_amount"])
                ),
            ),
            "old_state": states.get(ticker, {
                "generation_id": None,
                "comparability_status": "unknown",
                "alternate_generation": None,
                "reason": "price comparability unknown: persisted state missing",
            }),
        }
    return results


def _table_counts(connection):
    tables = (
        "prices", "dividends", "price_generations", "price_generation_values",
        "price_generation_state", "canonical_transactions",
    )
    return {
        table: int(connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0])
        for table in tables
    }


def _apply(database: Path, results: dict, backup: Path, change_log: Path):
    with (
        database_owner(database, purpose="price certification maintenance"),
        duckdb.connect(str(database)) as connection,
    ):
        return _apply_owned(database, results, backup, change_log, connection)


def _apply_owned(database, results, backup, change_log, connection):
    if backup != REQUIRED_BACKUP:
        raise SystemExit(f"--backup must be exactly {REQUIRED_BACKUP}")
    if not backup.is_file() or backup.stat().st_size != database.stat().st_size:
        raise SystemExit("required backup is missing or its size differs from the canonical DB")

    db = SimpleNamespace(conn=connection)
    before_counts = _table_counts(connection)
    changes = []
    try:
        connection.execute("BEGIN TRANSACTION")
        for ticker in sorted(results):
            certification = results[ticker]["certification"]
            old_state = results[ticker]["old_state"]
            new_state = publish_price_generation_certification(
                db, ticker, certification, old_state
            )
            if new_state != old_state:
                changes.append({
                    "ticker": ticker,
                    "old_served_generation": old_state.get("generation_id"),
                    "new_served_generation": new_state["generation_id"],
                    "old_comparability": old_state.get("comparability_status"),
                    "new_comparability": new_state["comparability_status"],
                    "evidence": certification.get("reason"),
                    "persisted_reason": new_state.get("reason"),
                })
        connection.execute("COMMIT")
    except Exception:
        connection.execute("ROLLBACK")
        raise
    after_counts = _table_counts(connection)
    changed_tables = {
        table: (before_counts[table], after_counts[table])
        for table in before_counts
        if before_counts[table] != after_counts[table]
    }
    if set(changed_tables) - {"price_generation_state"}:
        raise RuntimeError(f"unexpected table row-count changes: {changed_tables}")
    change_log.parent.mkdir(parents=True, exist_ok=True)
    previous = json.loads(change_log.read_text()) if change_log.exists() else {}
    prior_changes = previous.get("changes", [])
    prior_runs = previous.get("runs")
    if prior_runs is None and previous:
        prior_runs = [{
            "stage": "provisional-whole-history-check",
            "table_counts_before": previous.get("table_counts_before"),
            "table_counts_after": previous.get("table_counts_after"),
            "state_changes": len(prior_changes),
        }]
    prior_runs = prior_runs or []
    corrected = [{"stage": "local-per-session-reconciliation", **change} for change in changes]
    change_log.write_text(json.dumps({
        "backup": str(backup),
        "backup_bytes": backup.stat().st_size,
        "database_bytes": database.stat().st_size,
        "table_counts_before": before_counts,
        "table_counts_after": after_counts,
        "runs": prior_runs + [{
            "stage": "local-per-session-reconciliation",
            "table_counts_before": before_counts,
            "table_counts_after": after_counts,
            "state_changes": len(changes),
        }],
        "changes": prior_changes + corrected,
    }, indent=2, sort_keys=True) + "\n")
    return changes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--apply", action="store_true", help="publish certification state")
    parser.add_argument("--backup", type=Path, default=REQUIRED_BACKUP)
    parser.add_argument("--change-log", type=Path, default=DEFAULT_CHANGE_LOG)
    args = parser.parse_args()

    connection = duckdb.connect(str(resolve_read_path(args.database)), read_only=True)
    try:
        results = _read_certifications(connection)
    finally:
        connection.close()

    certified = sum(row["certification"]["certified"] for row in results.values())
    failed = Counter()
    failure_examples = {}
    for row in results.values():
        certification = row["certification"]
        if certification["certified"]:
            continue
        reason = certification["reason"]
        message = reason.split(": ", 1)[-1] if reason.startswith("yahoo:") else reason
        if message.startswith("dividend reconstruction"):
            category = "dividend reconstruction"
        elif message.startswith("dividend "):
            category = "missing dividend session"
        elif message.startswith("independent fetches disagree"):
            category = "adjusted-close fetch divergence"
        elif message.startswith("independent adjustment factors disagree"):
            category = "adjustment-factor fetch divergence"
        elif message.startswith("independent fetches "):
            category = "insufficient shared dates"
        elif message.startswith("only 1 independent"):
            category = "single adjusted fetch"
        elif message.startswith("raw split-like jump"):
            category = "unexplained raw split-like jump"
        elif message.startswith("unexplained adjustment-factor step"):
            category = "unexplained adjustment-factor step"
        elif message.startswith("duplicate dates"):
            category = "invalid generation rows"
        else:
            category = message.split(":", 1)[0]
        failed[category] += 1
        failure_examples.setdefault(category, reason)
    output = {
        "tickers_with_adjusted_generations": len(results),
        "certified_tickers": certified,
        "failed_tickers": len(results) - certified,
        "failure_reason_groups": dict(sorted(failed.items())),
        "failure_reason_examples": dict(sorted(failure_examples.items())),
        "spy": results.get("SPY", {}).get("certification"),
        "controls": {
            ticker: results[ticker]["certification"]
            for ticker in ("AAPL", "MSFT", "AAOI")
            if ticker in results
        },
    }
    if args.apply:
        changes = _apply(args.database, results, args.backup, args.change_log)
        output["state_changes"] = len(changes)
        output["change_log"] = str(args.change_log)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
