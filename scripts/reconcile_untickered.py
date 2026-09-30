#!/usr/bin/env python3
"""Propose ticker fills for untickered canonical rows using exact third-party keys.

Third-party aggregates are reconciliation-only, never canonical evidence. Their
parsing can suggest a ticker, but a human/root-owned persistence policy must
review the proposal; this script only reads the database and writes artifacts.

Input CSV/JSON fields: member, transaction_date (YYYY-MM-DD), ticker,
amount_band, source_doc_id. Amount bands and member/doc IDs match exactly.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import duckdb

_INPUT_FIELDS = (
    "member",
    "transaction_date",
    "ticker",
    "amount_band",
    "source_doc_id",
)
_POLICY_NOTE = (
    "Third-party aggregates are reconciliation-only, not canonical disclosures. "
    "A proposed ticker fill must not enter canonical data silently; root must "
    "decide and apply any persistence policy."
)


def _parse_date(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("transaction_date must be an ISO YYYY-MM-DD string")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(
            f"invalid transaction_date {value!r}; expected YYYY-MM-DD"
        ) from exc
    if parsed.isoformat() != value:
        raise ValueError(f"invalid transaction_date {value!r}; expected YYYY-MM-DD")
    return value


def _load_trades(path: Path) -> list[dict[str, str]]:
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            if reader.fieldnames is None or not set(_INPUT_FIELDS).issubset(
                reader.fieldnames
            ):
                raise ValueError(
                    f"CSV must contain columns: {', '.join(_INPUT_FIELDS)}"
                )
            records = list(reader)
    elif path.suffix.lower() == ".json":
        records = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(records, list) or any(
            not isinstance(row, dict) for row in records
        ):
            raise ValueError("JSON input must be an array of trade objects")
    else:
        raise ValueError("third-party input must be a .csv or .json file")

    trades = []
    for index, row in enumerate(records, start=1):
        missing = set(_INPUT_FIELDS) - row.keys()
        if missing:
            raise ValueError(
                f"third-party row {index} missing fields: {sorted(missing)}"
            )
        for field in ("member", "amount_band"):
            if not isinstance(row[field], str) or not row[field].strip():
                raise ValueError(
                    f"third-party row {index} has an empty or invalid {field}"
                )
        source_doc_id = row["source_doc_id"]
        if isinstance(source_doc_id, bool) or not isinstance(source_doc_id, (str, int)):
            raise TypeError(f"third-party row {index} has an invalid source_doc_id")
        source_doc_id = str(source_doc_id)
        if not source_doc_id.strip():
            raise ValueError(f"third-party row {index} has an empty source_doc_id")
        ticker = row["ticker"]
        if not isinstance(ticker, str) or not ticker.strip():
            raise ValueError(f"third-party row {index} has an empty or invalid ticker")
        trades.append(
            {
                "member": row["member"],
                "transaction_date": _parse_date(row["transaction_date"]),
                "ticker": ticker.strip().upper(),
                "amount_band": row["amount_band"],
                "source_doc_id": source_doc_id,
            }
        )
    return trades


def _canonical_rows(db_path: Path) -> list[dict]:
    connection = duckdb.connect(str(db_path), read_only=True)
    try:
        rows = connection.execute(
            """SELECT doc_id, member, transaction_date, amount_raw, ticker
               FROM canonical_transactions"""
        ).fetchall()
    finally:
        connection.close()
    return [
        {
            "doc_id": doc_id,
            "member": member,
            "transaction_date": transaction_date,
            "amount_band": amount_raw,
            "ticker": ticker,
        }
        for doc_id, member, transaction_date, amount_raw, ticker in rows
    ]


def _key(doc_id: object, member: object, transaction_date: object, amount_band: object):
    if (
        doc_id is None
        or member is None
        or transaction_date is None
        or amount_band is None
        or not str(doc_id).strip()
        or not isinstance(member, str)
        or not member.strip()
        or not isinstance(amount_band, str)
        or not amount_band.strip()
    ):
        return None
    if isinstance(transaction_date, date):
        tx_date = transaction_date.isoformat()
    else:
        try:
            tx_date = _parse_date(transaction_date)
        except ValueError:
            return None
    return str(doc_id), member, tx_date, amount_band


def _reconcile(canonical_rows: list[dict], trades: list[dict[str, str]]):
    trades_by_key = defaultdict(list)
    for index, trade in enumerate(trades):
        trades_by_key[
            _key(
                trade["source_doc_id"],
                trade["member"],
                trade["transaction_date"],
                trade["amount_band"],
            )
        ].append((index, trade))

    canonical_key_counts = defaultdict(int)
    for row in canonical_rows:
        key = _key(
            row["doc_id"], row["member"], row["transaction_date"], row["amount_band"]
        )
        if key is not None:
            canonical_key_counts[key] += 1

    untickered_rows = [row for row in canonical_rows if row["ticker"] is None]
    proposed = []
    unmatched = []
    used_trade_indexes = set()
    for row in untickered_rows:
        key = _key(
            row["doc_id"], row["member"], row["transaction_date"], row["amount_band"]
        )
        if key is None:
            reason = "incomplete_canonical_key"
        elif canonical_key_counts[key] != 1:
            reason = "ambiguous_canonical_key"
        elif len(trades_by_key[key]) == 0:
            reason = "no_exact_match"
        elif len(trades_by_key[key]) != 1:
            reason = "ambiguous_third_party_key"
        else:
            trade_index, trade = trades_by_key[key][0]
            used_trade_indexes.add(trade_index)
            proposed.append(
                {
                    "doc_id": str(row["doc_id"]),
                    "member": row["member"],
                    "transaction_date": key[2],
                    "amount_band": row["amount_band"],
                    "ticker": trade["ticker"],
                    "source_doc_id": trade["source_doc_id"],
                    "ticker_origin": "reconciled_third_party",
                }
            )
            continue
        unmatched.append(
            {
                "doc_id": row["doc_id"],
                "member": row["member"],
                "transaction_date": str(row["transaction_date"])
                if row["transaction_date"] is not None
                else None,
                "amount_band": row["amount_band"],
                "reason": reason,
            }
        )

    report = {
        "schema_version": 1,
        "report_type": "untickered_reconciliation",
        "reconciliation_only": True,
        "canonical_row_count": len(untickered_rows),
        "matched_count": len(proposed),
        "unmatched_count": len(unmatched),
        "third_party_row_count": len(trades),
        "used_third_party_row_count": len(used_trade_indexes),
        "unused_third_party_row_count": len(trades) - len(used_trade_indexes),
        "unmatched": unmatched,
        "policy_note": _POLICY_NOTE,
    }
    manifest = {
        "schema_version": 1,
        "artifact_type": "untickered_ticker_fill_proposals",
        "reconciliation_only": True,
        "proposed_fill_count": len(proposed),
        "records": proposed,
        "policy_note": _POLICY_NOTE,
    }
    return report, manifest


def reconcile(db_path: Path, input_path: Path, report_path: Path, manifest_path: Path):
    """Read canonical rows and write report/proposals; never write to the DB."""
    protected_paths = {db_path.resolve(), input_path.resolve()}
    output_paths = {report_path.resolve(), manifest_path.resolve()}
    if len(output_paths) != 2 or protected_paths & output_paths:
        raise ValueError(
            "report and manifest paths must be distinct from each other, the DB, and input"
        )
    report, manifest = _reconcile(_canonical_rows(db_path), _load_trades(input_path))
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return report, manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db", required=True, type=Path, help="canonical DuckDB path (read-only)"
    )
    parser.add_argument(
        "--third-party", required=True, type=Path, help="CSV or JSON trade list"
    )
    parser.add_argument(
        "--report", required=True, type=Path, help="reconciliation report JSON path"
    )
    parser.add_argument(
        "--manifest", required=True, type=Path, help="ticker-fill proposal JSON path"
    )
    args = parser.parse_args()
    try:
        report, manifest = reconcile(
            args.db, args.third_party, args.report, args.manifest
        )
    except (OSError, TypeError, ValueError, duckdb.Error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"matched={report['matched_count']} unmatched={report['unmatched_count']}")
    print(
        f"report={args.report} manifest={args.manifest} "
        f"proposals={manifest['proposed_fill_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
