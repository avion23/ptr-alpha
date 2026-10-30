"""Quarantine frozen model receipts, restoring exact pre-run OCR data under a hold.

Dry-run is the default. Both modes lease and inspect the WORKING database, never
the published reader path. Apply archives complete current rows/runs and appends
needs-review decisions in one transaction; it never publishes or releases a hold.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if __name__ == "__main__":
    sys.path.insert(0, str(_REPO_ROOT / "src"))

import duckdb

from analyzer.ocr.contracts import ExtractionLane, ExtractionOutcome, ResultBundle
from analyzer.ocr.workers import _publish, load_result
from analyzer.persistence.ownership import database_owner, publication_pointer
from analyzer.persistence.work_repository import WorkRepository
from analyzer.sql_fragments import HOUSE_PARSE_SOURCE


def quarantine(conn, scope: list[dict], baseline: Path, data_dir: Path,
               reason: str, *, apply: bool = False) -> dict:
    """Caller owns the working database lease; no schema initialization/repair."""
    if not reason.strip() or not scope:
        raise ValueError("A nonempty frozen scope and operator reason are required")
    scope = [{key: item[key] for key in ("result_id", "task_id", "attempt_id", "path", "lane")}
             for item in scope]
    if len({item["result_id"] for item in scope}) != len(scope):
        raise ValueError("Duplicate frozen receipt")
    baseline = baseline.expanduser().resolve(strict=True)
    scope_id = hashlib.sha256(json.dumps(scope, sort_keys=True).encode()).hexdigest()
    repair = {"operation": "quarantine_model_admissions", "scope_sha256": scope_id,
              "baseline": str(baseline), "reason": reason}
    # ATTACH does not accept parameter placeholders; escape the operator's path.
    conn.execute("ATTACH '" + str(baseline).replace("'", "''") + "' AS quarantine_baseline (READ_ONLY)")
    conn.execute("BEGIN TRANSACTION")
    try:
        repository = WorkRepository(conn)
        plans = []
        docs = []
        already = []
        for frozen in scope:
            receipt = {key: frozen[key] for key in ("result_id", "task_id", "attempt_id", "path")}
            source = load_result(receipt["path"])
            work = repository.validate_receipt(receipt)
            if source.work != work or work.lane == ExtractionLane.LOCAL or work.lane.value != frozen["lane"]:
                raise ValueError("Frozen receipt is not the bound model extraction")
            if hashlib.sha256(json.dumps(source.to_dict(), sort_keys=True, separators=(",", ":"),
                                         allow_nan=False).encode()).hexdigest() != receipt["result_id"]:
                raise ValueError("Frozen receipt result identity mismatch")
            if work.doc_id in docs:
                raise ValueError("Scope contains multiple receipts for one document")
            docs.append(work.doc_id)
            decision = ResultBundle(work, ExtractionOutcome.NEEDS_REVIEW, [], evidence={
                "source_receipt": receipt, "quarantine": repair}, detail=reason)
            effective = repository.effective_admission_decision(receipt)
            current = conn.execute("SELECT attempt_id, result_id, status FROM extraction_work WHERE task_id = ?",
                                   [work.task_id]).fetchone()
            if current != (work.attempt_id, effective["result_id"], effective["state"]):
                raise ValueError(f"Scope changed: newer task state for {work.doc_id}")
            if effective["result_id"] != receipt["result_id"]:
                event = effective["detail"]
                if (effective["state"] != "needs_review" or not isinstance(event, dict)
                        or event.get("detail") != repair or event.get("decision") != decision.to_dict()):
                    raise ValueError(f"Scope changed: newer review for {work.doc_id}")
                already.append(work.doc_id)
                continue
            if effective["state"] != "accepted":
                raise ValueError(f"Frozen receipt is not accepted: {work.doc_id}")
            finished = conn.execute("SELECT finished_at FROM extraction_results WHERE result_id = ?",
                                    [receipt["result_id"]]).fetchone()[0]
            # Frozen admissions created their complete rowset in the receipt's
            # transaction. Refuse later replacements, including same-count ones.
            count, changed = conn.execute("""SELECT COUNT(*), COUNT(*) FILTER (WHERE
                created_at IS DISTINCT FROM ? OR ingestion_generation IS DISTINCT FROM ?
                OR artifact_sha256 IS DISTINCT FROM ?) FROM transactions
                WHERE doc_id = ? AND source = 'gemini_ocr'""",
                [finished, work.generation_id, work.artifact_sha256, work.doc_id]).fetchone()
            runs = conn.execute("""SELECT status, transaction_count, rowset_created_at
                FROM pdf_parse_runs WHERE doc_id = ? AND parser_version = ?
                  AND ingestion_generation = ? AND artifact_sha256 = ?""",
                [work.doc_id, work.extraction_recipe, work.generation_id, work.artifact_sha256]).fetchall()
            if changed or not count or runs != [("success", count, finished)]:
                raise ValueError(f"Scope changed: current rowset no longer matches receipt for {work.doc_id}")
            plans.append((receipt, decision))

        pending_docs = [decision.work.doc_id for _, decision in plans]
        tx_filter = "doc_id IN (SELECT unnest(?)) AND source = 'gemini_ocr'"
        run_filter = "doc_id IN (SELECT unnest(?)) AND " + HOUSE_PARSE_SOURCE + " = 'gemini_ocr'"
        # A changed historical acceptance could otherwise silently be undone.
        for receipt, decision in plans:
            doc = decision.work.doc_id
            current_runs = conn.execute("SELECT * FROM pdf_parse_runs parse_run WHERE " + run_filter
                                       + " AND parser_version <> ?", [[doc], decision.work.extraction_recipe]).fetchall()
            prior_runs = conn.execute("SELECT * FROM quarantine_baseline.pdf_parse_runs parse_run WHERE "
                                     + run_filter, [[doc]]).fetchall()
            if sorted(map(repr, current_runs)) != sorted(map(repr, prior_runs)):
                raise ValueError(f"Scope changed: other OCR parse runs changed for {doc}")

        counts = {"scope": len(scope), "already_quarantined": len(already), "pending": len(plans),
                  "doc_ids": sorted(docs)}
        for label, table, predicate in (("rows", "transactions", tx_filter),
                                         ("parse_runs", "pdf_parse_runs parse_run", run_filter)):
            counts["archive_" + label] = conn.execute("SELECT COUNT(*) FROM " + table + " WHERE " + predicate,
                                                       [pending_docs]).fetchone()[0]
            counts["restore_" + label] = conn.execute("SELECT COUNT(*) FROM quarantine_baseline." + table
                                                       + " WHERE " + predicate, [pending_docs]).fetchone()[0]
        counts["baseline_canonical_rows"] = conn.execute("""SELECT COUNT(*) FROM
            quarantine_baseline.canonical_transactions WHERE doc_id IN (SELECT unnest(?))""", [docs]).fetchone()[0]

        # On replay, verify the prior repair still has its exact baseline rows
        # and runs; never undo later edits just because its event still exists.
        for table, predicate in (("transactions", tx_filter), ("pdf_parse_runs parse_run", run_filter)):
            left = "SELECT * FROM " + table + " WHERE " + predicate
            right = "SELECT * FROM quarantine_baseline." + table + " WHERE " + predicate
            for first, second in ((left, right), (right, left)):
                if conn.execute("SELECT COUNT(*) FROM (" + first + " EXCEPT ALL " + second + ")",
                                [already, already]).fetchone()[0]:
                    raise ValueError("Scope changed after quarantine; refusing replay")

        if not apply:
            conn.execute("ROLLBACK")
            return {**counts, "applied": False}

        # Existing history tables hold every column, including IDs, raw fields,
        # exact types and original watermarks. Do not call accept_corrections.
        for table, history, predicate in (("transactions", "transaction_versions", tx_filter),
                                           ("pdf_parse_runs", "parse_run_versions", run_filter.replace("parse_run.", "pdf_parse_runs."))):
            conn.execute("INSERT INTO " + history + " SELECT *, CURRENT_TIMESTAMP FROM " + table
                         + " WHERE " + predicate, [pending_docs])
            conn.execute("DELETE FROM " + table + " WHERE " + predicate, [pending_docs])
            conn.execute("INSERT INTO " + table + " SELECT * FROM quarantine_baseline." + table
                         + " WHERE " + predicate, [pending_docs])
        for receipt, decision in plans:
            path = _publish(decision, str(data_dir / "pipeline"))
            repository.append_admission_decision(receipt, decision, path, "needs_review", repair,
                                                  in_transaction=True)
        # Exact bidirectional comparison includes every business/provenance
        # column and detects missing or newly exposed canonical rows.
        for first, second in (("main", "quarantine_baseline"), ("quarantine_baseline", "main")):
            if conn.execute(f"""SELECT COUNT(*) FROM (
                SELECT * FROM {first}.canonical_transactions WHERE doc_id IN (SELECT unnest(?))
                EXCEPT ALL SELECT * FROM {second}.canonical_transactions WHERE doc_id IN (SELECT unnest(?)))""",
                [docs, docs]).fetchone()[0]:
                raise ValueError("Quarantine did not restore exact baseline canonical rows")
        conn.execute("COMMIT")
        return {**counts, "applied": True}
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.execute("DETACH quarantine_baseline")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    data_dir = args.data_dir.expanduser().resolve(strict=True)
    db_path = (data_dir / "congress.duckdb").resolve(strict=True)
    if db_path.resolve() == args.baseline.expanduser().resolve():
        raise ValueError("Working database and baseline must differ")
    scope = json.loads(args.scope.read_text())
    with database_owner(db_path, "model admission quarantine maintenance"):
        pointer = publication_pointer(db_path)
        if not pointer.is_file() or "quality_hold" not in json.loads(pointer.read_text()):
            raise ValueError("Quarantine requires an existing publication quality_hold")
        with duckdb.connect(str(db_path), read_only=not args.apply) as conn:
            result = quarantine(conn, scope, args.baseline, data_dir, args.reason, apply=args.apply)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
