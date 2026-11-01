"""One database owner, durable attempts, and independently bounded extraction lanes.

All database calls run on the calling thread. Local children use spawn; provider
threads receive only immutable work envelopes and never inherit a connection.
"""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from contextlib import ExitStack, nullcontext
from dataclasses import replace
from datetime import datetime, timedelta
import hashlib
import json
import multiprocessing
from pathlib import Path
import time
from uuid import uuid4

from analyzer.exceptions import DataSourceError
from analyzer.ocr.contracts import ExtractionLane, ExtractionOutcome, WorkEnvelope, task_key
from analyzer.persistence.database import Database
from analyzer.persistence.ownership import assert_publication_allowed


def provider_code(result) -> str | None:
    """Read only the last durable provider HTTP body, never model/error prose."""
    responses = result.evidence.get("provider_responses") or []
    if not responses:
        return None
    try:
        body = json.loads(responses[-1])
    except (TypeError, ValueError):
        return None
    error = body.get("error") if isinstance(body, dict) else None
    code = error.get("code") if isinstance(error, dict) else None
    return str(code) if isinstance(code, (str, int)) and not isinstance(code, bool) else None


def _scope(db: Database, years: list[int] | None, doc_ids: list[str] | None = None) -> list[dict]:
    """Official latest-generation PTR scope, including missing acquisitions."""
    if years == []:
        raise DataSourceError("House archive scope unknown: no archive years requested")
    headers = db.conn.execute("""
        WITH latest AS (
            SELECT archive_year, generation_id, ptr_count, metadata_count
            FROM house_archive_generations
            QUALIFY row_number() OVER (
                PARTITION BY archive_year ORDER BY promoted_at DESC, generation_id DESC
            ) = 1
        )
        SELECT g.archive_year, g.generation_id, g.ptr_count, g.metadata_count,
               COUNT(m.doc_id) FILTER (WHERE m.filing_type = 'P'), COUNT(m.doc_id)
        FROM latest g LEFT JOIN house_generation_metadata m
          ON m.archive_year = g.archive_year AND m.generation_id = g.generation_id
        GROUP BY ALL ORDER BY g.archive_year
    """).fetchall()
    known_years = {row[0] for row in headers}
    missing = sorted(set(years or []) - known_years)
    if missing or not headers:
        raise DataSourceError(f"House archive scope unknown: no official generation header for {missing or 'any year'}")
    for year, generation, expected_ptr, expected_metadata, actual_ptr, actual_metadata in headers:
        if years is not None and year not in years:
            continue
        if (expected_ptr is None or expected_metadata is None
                or expected_ptr < 0 or expected_metadata < 0
                or expected_ptr != actual_ptr or expected_metadata != actual_metadata):
            raise DataSourceError(
                f"House archive {year} generation {generation} scope counts mismatched or unknown: "
                f"expected PTR={expected_ptr}, metadata={expected_metadata}; "
                f"actual PTR={actual_ptr}, metadata={actual_metadata}"
            )
    rows = db.conn.execute("""
        WITH latest AS (
            SELECT archive_year, generation_id FROM house_archive_generations
            QUALIFY row_number() OVER (
                PARTITION BY archive_year ORDER BY promoted_at DESC, generation_id DESC
            ) = 1
        )
        SELECT m.archive_year, m.generation_id, m.doc_id, m.first_name,
               m.last_name, m.filing_date, a.artifact_sha256
        FROM latest g JOIN house_generation_metadata m
          ON m.archive_year = g.archive_year AND m.generation_id = g.generation_id
        LEFT JOIN house_pdf_artifacts a
          ON a.archive_year = m.archive_year AND a.generation_id = m.generation_id
         AND a.doc_id = m.doc_id
        WHERE m.filing_type = 'P'
        ORDER BY m.archive_year, m.doc_id
    """).fetchall()
    scope = [dict(archive_year=int(y), generation_id=g, doc_id=str(d),
                 first_name=f or "", last_name=l or "", filing_date=str(dt)[:10],
                 artifact_sha256=sha)
            for y, g, d, f, l, dt, sha in rows if years is None or y in years]
    if doc_ids is None:
        return scope
    if not doc_ids:
        raise DataSourceError("House archive scope unknown: no document IDs requested")
    requested = set(doc_ids)
    unknown = requested - {item["doc_id"] for item in scope}
    if unknown:
        raise DataSourceError(f"Document IDs outside official requested-year PTR scope: {sorted(unknown)}")
    return [item for item in scope if item["doc_id"] in requested]


def pipeline_status(data_dir: str | Path, *, years: list[int] | None = None,
                    database: Database | None = None, doc_ids: list[str] | None = None) -> dict:
    """Read one published version, or project the already-owned writer connection."""
    from analyzer.persistence.ownership import publication_state, resolve_read_path

    path = Path(data_dir) / "congress.duckdb"
    with nullcontext(database) if database is not None else Database(resolve_read_path(path), read_only=True) as db:
        scope = _scope(db, years, doc_ids)
        generations = {(s["archive_year"], s["generation_id"]) for s in scope}
        unresolved_docs = {(y, g, d) for y, g in generations
                           for d in db.get_unresolved_house_doc_ids(y, g)}
        if doc_ids is not None:
            unresolved_docs &= {(s["archive_year"], s["generation_id"], s["doc_id"]) for s in scope}
        unresolved = len(unresolved_docs)
        import json
        from collections import Counter

        tasks = {}
        selected_tasks = []
        work_counts = Counter()
        scoped = {(s["archive_year"], s["generation_id"], s["doc_id"]): s for s in scope}
        for task_id, envelope, state in db.conn.execute("SELECT task_id, envelope, status FROM extraction_work").fetchall():
            item = json.loads(envelope)
            key = (item.get("archive_year"), item.get("generation_id"), item.get("doc_id"))
            if doc_ids is not None and (key not in scoped
                    or item.get("artifact_sha256") != scoped[key]["artifact_sha256"]):
                continue
            selected_tasks.append(task_id)
            work_counts[state] += 1
            tasks.setdefault(key, set()).add(state)
        work_status = db.work.status() if doc_ids is None else {
            "total": sum(work_counts.values()), "by_state": dict(work_counts),
            "results": db.conn.execute("SELECT COUNT(*) FROM extraction_results WHERE task_id IN (SELECT UNNEST(?))",
                                       [selected_tasks]).fetchone()[0],
        }
        counts = Counter()
        for item in scope:
            key = (item["archive_year"], item["generation_id"], item["doc_id"])
            if key not in unresolved_docs:
                counts["resolved"] += 1
                continue
            states = tasks.get(key, set())
            state = next((s for s in ("running", "pending", "retry_wait", "needs_review", "blocked", "failed")
                          if s in states), "unstarted")
            counts["blocked" if not item["artifact_sha256"] else state] += 1
        return {"scope": len(scope), "unresolved": unresolved,
                "resolved": len(scope) - unresolved, "work": work_status,
                "documents": {"total": len(scope), "by_state": dict(counts)},
                "required_without_acquisition": sum(not s["artifact_sha256"] for s in scope),
                "publication": publication_state(path)}


def _envelope(scope: dict, data_dir: Path, lane: ExtractionLane, recipe: str) -> WorkEnvelope:
    from analyzer.ocr.common import pdf_page_count
    from analyzer.ocr.ingestion import local_work_artifact

    # The owner binds the acquired bytes once; workers see only this sealed copy.
    # local_work_artifact validates identity; no directory inventory is a scope.
    _, binding = local_work_artifact(None, data_dir, {
        "doc_id": scope["doc_id"], "year": scope["archive_year"],
        "artifact_sha256": scope["artifact_sha256"],
    })
    content = binding.get("pdf_bytes")
    if content is None:
        raise FileNotFoundError("required acquired PDF is missing")
    digest = hashlib.sha256(content).hexdigest()
    artifact = data_dir / "pipeline" / "artifacts" / digest / f'{scope["doc_id"]}.pdf'
    artifact.parent.mkdir(parents=True, exist_ok=True)
    if not artifact.exists():
        import os
        import tempfile
        with tempfile.NamedTemporaryFile(dir=artifact.parent, delete=False) as output:
            temporary = Path(output.name)
            try:
                output.write(content)
                output.flush()
                os.fsync(output.fileno())
                os.replace(temporary, artifact)
            finally:
                temporary.unlink(missing_ok=True)
        artifact.chmod(0o444)
    elif hashlib.sha256(artifact.read_bytes()).hexdigest() != digest:
        raise RuntimeError("sealed artifact digest mismatch")
    key = task_key(scope["doc_id"], scope["archive_year"], scope["generation_id"], digest, recipe, lane)
    return WorkEnvelope(key, str(uuid4()), scope["doc_id"], scope["archive_year"],
                        scope["generation_id"], digest, str(artifact),
                        f'{scope["first_name"]} {scope["last_name"]}'.strip(),
                        scope["filing_date"], pdf_page_count(artifact), recipe, lane)


def _current_binding(db: Database, work: WorkEnvelope) -> bool:
    if db.get_latest_house_generation(work.archive_year) != work.generation_id:
        return False
    row = db.conn.execute("""SELECT artifact_sha256 FROM house_pdf_artifacts
        WHERE archive_year = ? AND generation_id = ? AND doc_id = ?""",
        [work.archive_year, work.generation_id, work.doc_id]).fetchone()
    return row is not None and row[0] == work.artifact_sha256


def _register_receipt(db: Database, result, path: Path) -> tuple[dict, dict | None]:
    """Register the hash-bound delivery once and inspect its current decision."""
    work = result.work
    receipt = dict(task_id=work.task_id, attempt_id=work.attempt_id,
                   result_id=db.work.register_result(result, str(path)), path=str(path))
    receipt["path"] = db.conn.execute("SELECT path FROM extraction_results WHERE result_id = ?",
                                      [receipt["result_id"]]).fetchone()[0]
    finished = db.work.is_finished(receipt)
    predecessor = db.work.effective_admission_decision(receipt) if finished else None
    return receipt, predecessor


def _ingest(db: Database, result, path: Path, data_dir: Path, source) -> str:
    receipt, predecessor = _register_receipt(db, result, path)
    return _ingest_registered(db, result, receipt, predecessor, data_dir, source)


def _ingest_registered(db: Database, result, receipt: dict, predecessor: dict | None,
                       data_dir: Path, source) -> str:
    """Ingest the registered delivery; review authorization is rechecked under lock."""
    from analyzer.ocr.ingestion import insert_transactions_outcome
    from analyzer.ocr.normalization import normalize_candidate
    from analyzer.ocr.workers import admission_guard

    work = result.work
    finished = predecessor is not None
    if finished:
        current = db.conn.execute("SELECT attempt_id, result_id, status FROM extraction_work WHERE task_id = ?",
                                  [work.task_id]).fetchone()
        # Only an explicit historical recipe refusal can be reconsidered. Old
        # generic GLM details can mask the gate reason; do not infer authority.
        recipe_refusal = (predecessor["state"] == "blocked"
                          and predecessor["detail"] == "extraction recipe changed"
                          and result.outcome == ExtractionOutcome.CANDIDATE
                          and result.evidence.get("artifact_verified") is True)
        if ((predecessor["state"] != "needs_review" and not recipe_refusal) or work.lane == ExtractionLane.LOCAL
                or result.outcome not in {ExtractionOutcome.CANDIDATE, ExtractionOutcome.VERIFIED_EMPTY}
                or current != (work.attempt_id, predecessor["result_id"], predecessor["state"])):
            return predecessor["state"]
    db.work.validate_receipt(receipt)
    if not _current_binding(db, work):
        if not finished:
            db.work.finish(receipt, "blocked", "source generation/artifact is no longer current")
        return "needs_review" if finished else "blocked"
    if result.outcome == ExtractionOutcome.RETRY_WAIT:
        delay = result.retry_after if result.retry_after is not None else 60
        db.work.finish(receipt, "retry_wait", result.detail,
                       retry_at=datetime.now() + timedelta(seconds=delay))
        return "retry_wait"
    if not finished and work.lane == ExtractionLane.GLM and result.evidence.get("staging_record"):
        from scripts import glm_stage
        record = result.evidence["staging_record"]
        existing = glm_stage.load_staging_record(data_dir, work.archive_year, work.doc_id)
        if existing is None or glm_stage.audit_staged_record_sha256(existing) != glm_stage.audit_staged_record_sha256(record):
            glm_stage.publish_staging_record(data_dir, record)
    # Publish staging before entering the guard: publication takes the document
    # lock itself. The guard rechecks the exact staged record and holds audit
    # authorization through normalization and the atomic canonical receipt.
    with admission_guard(result, data_dir) as gate:
        if gate["allowed"] and work.lane != ExtractionLane.LOCAL:
            from analyzer.ocr.review import assert_review_supersession

            try:
                # Read the actual current chain under the review lock, not an
                # operator-supplied ID or the pre-lock inspection above.
                assert_review_supersession(db.work, receipt["result_id"], gate["review_binding"])
                if finished:
                    predecessor = db.work.effective_admission_decision(receipt)
            except ValueError as exc:
                gate = {**gate, "allowed": False, "reason": str(exc)}
        local_uncertain = (work.lane == ExtractionLane.LOCAL
                           and result.evidence.get("artifact_verified") is True
                           and result.evidence.get("failure", {}).get("type") == "ParserCascadeError")
        # Deterministic rows already have the local cascade schema, not the OCR
        # schema. _save_parse_results owns their existing complete-rowset
        # consolidation/validation, so do not OCR-normalize them a second time.
        local_candidate = (work.lane == ExtractionLane.LOCAL
                           and result.outcome == ExtractionOutcome.CANDIDATE and bool(result.rows))
        candidate = None if local_candidate else gate.get("candidate") or normalize_candidate(work, result)
        status = result.outcome.value if candidate is None else candidate["status"]
        if not gate["allowed"] or status not in {"candidate", "verified_empty"}:
            permanent = ((status == "blocked" or result.outcome == ExtractionOutcome.BLOCKED)
                         and not local_uncertain) or gate["reason"] in {
                "extraction recipe changed", "artifact verification evidence missing"}
            disposition = "blocked" if permanent else "needs_review"
            if not finished:
                db.work.finish(receipt, disposition, result.detail or gate["reason"] or str(candidate["findings"] if candidate else []))
            return "needs_review" if finished else disposition
        if gate.get("source_proposal") is not None and not finished:
            # Source correction is a new decision, never a rewrite/retag of the
            # original extraction result. Register that immutable refusal first.
            db.work.finish(receipt, "needs_review", "source proposal requires a separate source-reviewed admission decision")
            return "needs_review"
        if finished:
            receipt["admission_decision"] = {
                "predecessor": predecessor, "result": result,
                "spool": str(data_dir / "pipeline" / "admission-decisions"),
            }
        if local_candidate:
            source._save_parse_results(
                work.archive_year,
                [(Path(work.artifact_path),
                  result.rows, result.evidence.get("engines", []))],
                {work.doc_id: {"First": work.member, "Last": "", "FilingDate": work.filing_date}},
                work.generation_id, recipe=work.extraction_recipe, receipt=receipt,
            )
        else:
            from analyzer.ocr.common import MODEL

            model = ("visionb/" + str(result.evidence.get("staging_record", {}).get("model", "glm"))
                     if work.lane == ExtractionLane.GLM else MODEL)
            insert_transactions_outcome(
                work.doc_id, work.archive_year, work.member, candidate["rows"],
                conn=db.conn, receipt=receipt, evidence=result.evidence,
                admission=gate,
                raw_count=len(candidate["rows"]) if gate.get("source_proposal") is not None else len(result.rows),
                parser_version=work.extraction_recipe,
                artifact_sha256=work.artifact_sha256, ingestion_generation=work.generation_id,
                engine_model=model, checkpoint=False,
            )
        if not db.work.is_finished(receipt):
            raise RuntimeError("canonical persistence did not atomically finish its result receipt")
        return db.work.effective_admission_decision(receipt)["state"]


def run_pipeline(
    data_dir: str | Path, years: list[int] | None = None, *,
    local_workers: int = 3, gemini_workers: int = 0, glm_workers: int = 0,
    refresh_recipe: bool = False, publish_every: int = 100,
    lease_seconds: int = 3600, database: Database | None = None,
    doc_ids: list[str] | None = None, extract_doc_ids: list[str] | None = None,
) -> dict:
    """Drain ready work, including scoped GLM rate retries, with explicit API use.

    A provider retry never consumes another provider's slots. Unresolved local
    results may escalate to configured API lanes; a retry is not an escalation.
    """
    assert_publication_allowed(Path(data_dir) / "congress.duckdb")
    from analyzer.download import HouseTransactionSource
    from analyzer.ocr.workers import extract_document, extraction_recipe, load_result
    from analyzer.settings import Settings

    capacities = {ExtractionLane.LOCAL: local_workers, ExtractionLane.GEMINI: gemini_workers,
                  ExtractionLane.GLM: glm_workers}
    if any(type(n) is not int or n < 0 for n in capacities.values()) or not any(capacities.values()):
        raise ValueError("worker slots must be nonnegative, with at least one enabled lane")
    if publish_every < 1 or lease_seconds < 1:
        raise ValueError("publish_every and lease_seconds must be positive")
    lanes = [lane for lane, n in capacities.items() if n]
    recipes = {lane: extraction_recipe(lane) for lane in lanes}
    base = Path(data_dir).resolve()
    spool = base / "pipeline"
    spool.mkdir(parents=True, exist_ok=True)
    settings = Settings()
    settings.data.data_dir = str(base)
    with nullcontext(database) if database is not None else Database(base / "congress.duckdb") as db:
        with ExitStack() as stack:
            # Unproven metadata scope must not replace an earlier publication
            # or start extraction work, including when the row query is empty.
            scope = _scope(db, years, doc_ids)
            extraction_scope = scope
            if extract_doc_ids is not None:
                if not extract_doc_ids:
                    raise DataSourceError("Extraction scope unknown: no document IDs requested")
                requested = set(extract_doc_ids)
                unknown = requested - {item["doc_id"] for item in scope}
                if unknown:
                    raise DataSourceError(
                        f"Extraction document IDs outside official scope: {sorted(unknown)}"
                    )
                extraction_scope = [item for item in scope if item["doc_id"] in requested]
            source = HouseTransactionSource(settings, db=db)
            stack.callback(source.close)
            # Ownership may follow price, metadata, or other maintenance writes.
            # A document-only signature cannot establish whole-database freshness.
            publication = db.publish_snapshot()
            scoped = {(s["archive_year"], s["generation_id"], s["doc_id"]): s for s in scope}
            scoped_doc_ids = sorted({s["doc_id"] for s in scope})
            extraction_doc_ids = sorted({s["doc_id"] for s in extraction_scope})
            extraction_bindings = {(*key, item["artifact_sha256"])
                                   for key, item in scoped.items()
                                   if item["doc_id"] in extraction_doc_ids}
            scoped_bindings = {(*key, item["artifact_sha256"]) for key, item in scoped.items()}
            unresolved = {(y, g): set(db.get_unresolved_house_doc_ids(y, g))
                          for y, g in {(s["archive_year"], s["generation_id"]) for s in scope}}
            missing_years = []

            eligible = set()
            satisfied = set()
            replayed_reviews = []
            replaying = True

            def binding(work):
                return (work.archive_year, work.generation_id, work.doc_id, work.artifact_sha256)

            def escalate(result, state):
                if state != "needs_review" or result.work.lane not in lanes:
                    return
                old = result.work
                if (binding(old) not in eligible
                        or old.extraction_recipe != recipes[old.lane]
                        or old.doc_id not in unresolved[old.archive_year, old.generation_id]):
                    return
                priority = db.conn.execute("SELECT priority FROM extraction_work WHERE task_id = ?",
                                           [old.task_id]).fetchone()[0]
                for lane in lanes[lanes.index(old.lane) + 1:]:
                    recipe = recipes[lane]
                    work = replace(old, task_id=task_key(old.doc_id, old.archive_year, old.generation_id,
                                   old.artifact_sha256, recipe, lane), attempt_id=str(uuid4()),
                                   lane=lane, extraction_recipe=recipe)
                    db.work.register(work, priority=priority)

            def receive(path):
                nonlocal publication
                result = load_result(path)
                key = (result.work.archive_year, result.work.generation_id, result.work.doc_id)
                if key not in scoped:
                    return
                if doc_ids is not None and binding(result.work) not in scoped_bindings:
                    return
                raw_receipt, predecessor = _register_receipt(db, result, Path(path))
                prior = predecessor["state"] if predecessor is not None else "extracted"
                state = _ingest_registered(db, result, raw_receipt, predecessor, base, source)
                generation = (result.work.archive_year, result.work.generation_id)
                if state in {"accepted", "verified_empty"} and (
                        prior not in {"accepted", "verified_empty"}
                        or result.work.doc_id in unresolved[generation]):
                    unresolved[generation] = set(db.get_unresolved_house_doc_ids(*generation))
                    if result.work.doc_id in unresolved[generation]:
                        # Keep the extraction and its historical verdict intact;
                        # record a separate admission refusal, not another parse
                        # or a silently accepted-but-unresolved terminal task.
                        current = db.conn.execute(
                            "SELECT attempt_id, status FROM extraction_work WHERE task_id = ?",
                            [result.work.task_id],
                        ).fetchone()
                        if current == (result.work.attempt_id, state):
                            from analyzer.ocr.workers import _publish

                            detail = "admitted extraction did not resolve its official binding"
                            review = replace(result, outcome=ExtractionOutcome.NEEDS_REVIEW,
                                             evidence={**result.evidence, "admission_contract_error": detail},
                                             detail=detail)
                            # Admission evidence is not a fresh worker delivery.
                            # An orphan from a crash before receipt commit must
                            # not be imported as an inactive extraction attempt.
                            review_path = _publish(review, str(spool / "admission-reviews"))
                            predecessor = db.work.effective_admission_decision(raw_receipt)
                            db.work.append_admission_decision(
                                predecessor, review, review_path, "needs_review", {"reason": detail})
                        state = "needs_review"
                if state in {"accepted", "verified_empty"} and prior not in {"accepted", "verified_empty"}:
                    satisfied.add(binding(result.work))
                    eligible.discard(binding(result.work))
                    db.work.supersede_pending(result.work, "binding satisfied by an admitted extraction")
                if (state in {"accepted", "verified_empty"}
                        and prior not in {"accepted", "verified_empty"}
                        and result.work.lane != ExtractionLane.LOCAL):
                    publication = db.publish_snapshot()
                if replaying and state == "needs_review":
                    replayed_reviews.append(result)
                else:
                    escalate(result, state)

            # Import durable responses before recovering leases or dispatching
            # replacements. A completed response must not trigger another bill.
            for path in sorted((spool / "results").glob("*.json")):
                receive(path)
            # Receipts can reference a durable bundle outside the default spool.
            for receipt in db.work.pending_results(limit=2147483647):
                receive(receipt["path"])
            if doc_ids is None and extract_doc_ids is None:
                db.work.recover_expired()
            else:
                recovery_doc_ids = extraction_doc_ids if extract_doc_ids is not None else scoped_doc_ids
                recovery_bindings = extraction_bindings if extract_doc_ids is not None else scoped_bindings
                recovery_tasks = [task_id for task_id, envelope in db.conn.execute("""
                    SELECT task_id, envelope FROM extraction_work
                    WHERE status = 'running' AND lease_until <= CURRENT_TIMESTAMP
                      AND json_extract_string(envelope, '$.doc_id') IN (SELECT UNNEST(?))
                """, [recovery_doc_ids]).fetchall()
                    if binding(WorkEnvelope.from_dict(json.loads(envelope))) in recovery_bindings]
                db.conn.execute("""UPDATE extraction_work SET status = 'pending',
                    lease_until = NULL, updated_at = CURRENT_TIMESTAMP
                    WHERE task_id IN (SELECT UNNEST(?))
                      AND NOT EXISTS (SELECT 1 FROM extraction_results r
                          WHERE r.task_id = extraction_work.task_id
                            AND r.attempt_id = extraction_work.attempt_id
                            AND r.finished_at IS NULL)""", [recovery_tasks])
            # Resolve canonical scope once, not through the global canonical
            # view for every dispatch. Ledger timestamps distinguish a resumed
            # refresh from an older success on the same immutable binding.
            known = {}
            accepted_at = {}
            for envelope, state, claimed_at, finished_at in db.conn.execute("""
                SELECT w.envelope, w.status, a.claimed_at,
                       (SELECT MAX(r.finished_at) FROM extraction_results r
                        WHERE r.task_id = w.task_id AND r.state IN ('accepted', 'verified_empty'))
                FROM extraction_work w LEFT JOIN extraction_attempts a ON a.attempt_id = w.attempt_id
                WHERE w.lane IS NOT NULL
            """).fetchall():
                work = WorkEnvelope.from_dict(json.loads(envelope))
                if doc_ids is not None and binding(work) not in scoped_bindings:
                    continue
                known[work.task_id] = (work, state, claimed_at)
                if finished_at is not None:
                    key = binding(work)
                    accepted_at[key] = max(accepted_at.get(key, finished_at), finished_at)
            for item in extraction_scope:
                key = (item["archive_year"], item["generation_id"], item["doc_id"], item["artifact_sha256"])
                needed = item["doc_id"] in unresolved[item["archive_year"], item["generation_id"]]
                if needed:
                    eligible.add(key)
                    continue
                if not refresh_recipe or key in satisfied:
                    continue
                root_id = task_key(item["doc_id"], item["archive_year"], item["generation_id"],
                                   item["artifact_sha256"], recipes[lanes[0]], lanes[0])
                root = known.get(root_id)
                if root is None or root[1] in {"pending", "retry_wait", "running"}:
                    eligible.add(key)
                elif (root[1] == "needs_review" and root[2] is not None
                      and (key not in accepted_at or accepted_at[key] < root[2])):
                    eligible.add(key)
            obsolete = []
            obsolete_paid = {}
            for work, state, _ in known.values():
                key = (work.archive_year, work.generation_id, work.doc_id)
                is_selected_for_extraction = binding(work) in extraction_bindings
                is_resolved = (work.doc_id not in unresolved.get(
                    (work.archive_year, work.generation_id), set()
                ))
                if (key in scoped and state in {"pending", "retry_wait"}
                        and ((is_selected_for_extraction and binding(work) not in eligible)
                             or (not is_selected_for_extraction and is_resolved))):
                    obsolete.append(work)
                elif (key in scoped and work.lane != ExtractionLane.LOCAL
                      and work.doc_id not in unresolved[work.archive_year, work.generation_id]
                      and state in {"pending", "retry_wait"}):
                    obsolete_paid.setdefault(work.lane, []).append(work)
            db.work.supersede_pending_many(obsolete, "binding already resolved; no outstanding recipe refresh")
            for lane, works in obsolete_paid.items():
                db.work.supersede_pending_many(works, "resolved recipe refresh does not require paid extraction",
                                               lane=lane.value)
            replaying = False
            for result in replayed_reviews:
                escalate(result, "needs_review")

            def seed(item):
                needed = item["doc_id"] in unresolved[item["archive_year"], item["generation_id"]]
                if not needed and not refresh_recipe:
                    return
                if (item["archive_year"], item["generation_id"], item["doc_id"], item["artifact_sha256"]) not in eligible:
                    return
                if item["artifact_sha256"]:
                    key = task_key(item["doc_id"], item["archive_year"], item["generation_id"],
                                   item["artifact_sha256"], recipes[lanes[0]], lanes[0])
                    if db.conn.execute("SELECT 1 FROM extraction_work WHERE task_id = ?", [key]).fetchone():
                        return
                try:
                    if not item["artifact_sha256"]:
                        raise FileNotFoundError("official PTR has no acquired artifact")
                    work = _envelope(item, base, lanes[0], recipes[lanes[0]])
                except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
                    key = f'blocked:{item["archive_year"]}:{item["generation_id"]}:{item["doc_id"]}'
                    db.work.register_blocked(key, str(exc), scope=item)
                    return
                db.work.register(work, priority=100 if needed else 0)

            # Freeze source bytes only within a small rolling seed window.
            # The official generation remains the durable inventory for items
            # not yet materialized into extraction envelopes.
            seeds = iter(sorted(extraction_scope, key=lambda s:
                s["doc_id"] not in unresolved[s["archive_year"], s["generation_id"]]))
            seeds_done = False

            def ready(lane, limit, retry_delay=0):
                # Scope and recipe filters prevent an older/year-excluded queue
                # head from hiding ready work. Resolved queued bindings were
                # superseded above (and on each admission), without repeatedly
                # evaluating the canonical view or shipping the whole scope.
                return db.conn.execute("""
                    WITH latest AS (
                        SELECT archive_year, generation_id FROM house_archive_generations
                        QUALIFY row_number() OVER (PARTITION BY archive_year
                            ORDER BY promoted_at DESC, generation_id DESC) = 1
                    )
                    SELECT w.task_id FROM extraction_work w JOIN latest g
                      ON json_extract_string(w.envelope, '$.generation_id') = g.generation_id
                     AND CAST(json_extract_string(w.envelope, '$.archive_year') AS INTEGER) = g.archive_year
                    JOIN house_generation_metadata m
                      ON m.archive_year = g.archive_year AND m.generation_id = g.generation_id
                     AND m.doc_id = json_extract_string(w.envelope, '$.doc_id')
                    WHERE w.lane = ? AND json_extract_string(w.envelope, '$.extraction_recipe') = ?
                       AND m.filing_type = 'P' AND g.archive_year IN (SELECT UNNEST(?))
                       AND m.doc_id IN (SELECT UNNEST(?))
                       AND (? OR EXISTS (SELECT 1 FROM house_pdf_artifacts a
                           WHERE a.archive_year = g.archive_year AND a.generation_id = g.generation_id
                             AND a.doc_id = m.doc_id
                             AND a.artifact_sha256 = json_extract_string(w.envelope, '$.artifact_sha256')))
                       AND (w.status = 'pending' OR (w.status = 'retry_wait'
                            AND w.retry_at <= CURRENT_TIMESTAMP + (? * INTERVAL '1 SECOND')))
                      AND NOT EXISTS (
                          SELECT 1 FROM extraction_work active
                          WHERE active.status = 'running'
                            AND json_extract_string(active.envelope, '$.doc_id') = m.doc_id
                            AND CAST(json_extract_string(active.envelope, '$.archive_year') AS INTEGER) = g.archive_year
                            AND json_extract_string(active.envelope, '$.generation_id') = g.generation_id
                            AND json_extract_string(active.envelope, '$.artifact_sha256') = json_extract_string(w.envelope, '$.artifact_sha256')
                      )
                    ORDER BY w.priority DESC,
                        CAST(json_extract_string(w.envelope, '$.page_count') AS INTEGER),
                        w.task_id LIMIT ?
                """, [lane.value, recipes[lane], sorted({s["archive_year"] for s in scope}),
                       extraction_doc_ids, doc_ids is None, retry_delay, limit]).fetchall()

            pools = {}
            for lane in lanes:
                if lane == ExtractionLane.LOCAL:
                    pool = ProcessPoolExecutor(capacities[lane], mp_context=multiprocessing.get_context("spawn"))
                else:
                    pool = ThreadPoolExecutor(capacities[lane], thread_name_prefix=lane.value)
                pools[lane] = stack.enter_context(pool)
            running = {}
            lane_backoff = dict.fromkeys(lanes, 0)
            glm_limit = glm_workers
            glm_successes = 0
            glm_rate_wave_until = 0
            glm_rate_retries = set()
            # Exhaustion stops this run, not a later explicit run after renewal.
            quota_exhausted = set()
            for envelope, delay, result_path in db.conn.execute("""SELECT w.envelope,
                    epoch(w.retry_at - CURRENT_TIMESTAMP), r.path FROM extraction_work w
                    LEFT JOIN extraction_results r ON r.result_id = w.result_id
                    WHERE w.status = 'retry_wait'""").fetchall():
                work = WorkEnvelope.from_dict(json.loads(envelope))
                item = scoped.get((work.archive_year, work.generation_id, work.doc_id))
                if (work.lane not in lanes or work.extraction_recipe != recipes[work.lane]
                        or binding(work) not in eligible or item is None
                        or item["artifact_sha256"] != work.artifact_sha256):
                    continue
                code = provider_code(load_result(result_path)) if result_path and work.lane == ExtractionLane.GLM else None
                if code == "1302":
                    glm_rate_retries.add(work.task_id)
                elif delay is not None and delay > 0:
                    lane_backoff[work.lane] = max(lane_backoff[work.lane], time.monotonic() + delay)

            def rate_retry_wakeup():
                # Only this run's current eligible rate-limited bindings can
                # keep a bulk GLM drain alive. claim() still owns retry_at.
                if ExtractionLane.GLM in quota_exhausted or not glm_rate_retries:
                    return None
                delays = []
                for envelope, delay in db.conn.execute("""SELECT envelope,
                        epoch(retry_at - CURRENT_TIMESTAMP) FROM extraction_work
                        WHERE status = 'retry_wait' AND task_id IN (SELECT UNNEST(?))""",
                        [sorted(glm_rate_retries)]).fetchall():
                    work = WorkEnvelope.from_dict(json.loads(envelope))
                    if (delay is not None and delay > 0 and binding(work) in eligible
                            and work.extraction_recipe == recipes[ExtractionLane.GLM]
                            and _current_binding(db, work)):
                        delays.append(delay)
                return time.monotonic() + min(delays) if delays else None

            review_seen = {}
            review_poll_seconds = 30
            next_review_poll = 0.0

            def poll_review_inbox():
                """Apply changed native review selectors to their registered raw result only."""
                from analyzer.ocr.ingestion import read_work_artifact

                review_dir = spool / "reviews"
                try:
                    paths = sorted(review_dir.glob("*.json"))
                except OSError:
                    return
                present = {path.name for path in paths}
                for missing in review_seen.keys() - present:
                    del review_seen[missing]
                for selector_path in paths:
                    raw_id = selector_path.stem
                    if (len(raw_id) != 64
                            or any(char not in "0123456789abcdef" for char in raw_id)):
                        continue
                    try:
                        stat = selector_path.stat()
                        content = read_work_artifact(base, selector_path)
                    except (OSError, RuntimeError):
                        continue
                    content_sha = hashlib.sha256(content).hexdigest()
                    stat_signature = (stat.st_mtime_ns, stat.st_size, content_sha)
                    previous = review_seen.get(selector_path.name)
                    if previous is not None and previous[:3] == stat_signature:
                        continue
                    try:
                        selection = json.loads(content)
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        signature = (*stat_signature, raw_id, None)
                        review_seen[selector_path.name] = signature
                        continue
                    binding_data = selection.get("binding") if isinstance(selection, dict) else None
                    verdict_sha = selection.get("verdict_sha256") if isinstance(selection, dict) else None
                    signature = (*stat_signature, raw_id, verdict_sha)
                    if (not isinstance(binding_data, dict)
                            or binding_data.get("raw_result_id") != raw_id
                            or not isinstance(verdict_sha, str) or len(verdict_sha) != 64
                            or any(char not in "0123456789abcdef" for char in verdict_sha)):
                        review_seen[selector_path.name] = signature
                        continue
                    row = db.conn.execute("""SELECT path, task_id, attempt_id, finished_at FROM extraction_results
                        WHERE result_id = ?""", [raw_id]).fetchone()
                    if row is None:
                        # A selector may arrive just before its worker result is registered.
                        continue
                    review_seen[selector_path.name] = signature
                    path, task_id, attempt_id, finished_at = row
                    if finished_at is not None:
                        effective = db.work.effective_admission_decision(dict(
                            result_id=raw_id, task_id=task_id, attempt_id=attempt_id, path=path))
                        if effective["state"] not in {"needs_review", "blocked"}:
                            continue
                    receive(path)

            completed = 0
            while True:
                now = time.monotonic()
                if now >= next_review_poll:
                    poll_review_inbox()
                    next_review_poll = time.monotonic() + review_poll_seconds
                if not seeds_done:
                    window = max(1, sum(capacities.values()) * 2)
                    budget = window - len(ready(lanes[0], window))
                    for _ in range(budget):
                        item = next(seeds, None)
                        if item is None:
                            seeds_done = True
                            break
                        seed(item)
                wakeups = []
                for lane in lanes:
                    if lane in quota_exhausted:
                        continue
                    limit = glm_limit if lane == ExtractionLane.GLM else capacities[lane]
                    free = limit - sum(w.lane == lane for w in running.values())
                    if free <= 0:
                        continue
                    delay = lane_backoff[lane] - time.monotonic()
                    if delay > 0:
                        # Wake only for a free lane with work ready at its
                        # deadline. claim() still enforces the durable retry_at.
                        if ready(lane, 1, retry_delay=delay):
                            wakeups.append(lane_backoff[lane])
                        continue
                    # One bounded rolling window across all years, replenished
                    # after every completion rather than after a batch barrier.
                    for (task_id,) in ready(lane, free):
                        work = db.work.claim(task_id, lease_seconds=lease_seconds)
                        running[pools[lane].submit(extract_document, work, str(spool))] = work
                    if (lane == ExtractionLane.GLM
                            and sum(w.lane == lane for w in running.values()) < glm_limit):
                        retry_wakeup = rate_retry_wakeup()
                        if retry_wakeup is not None:
                            wakeups.append(retry_wakeup)
                if not running:
                    if not seeds_done:
                        if (lanes[0] in quota_exhausted
                                or time.monotonic() < lane_backoff[lanes[0]]):
                            break
                        continue
                    retry_wakeup = rate_retry_wakeup()
                    if retry_wakeup is not None:
                        # Unknown quota errors retain their whole-lane backoff.
                        deadline = max(retry_wakeup, lane_backoff[ExtractionLane.GLM])
                        delay = min(deadline, next_review_poll) - time.monotonic()
                        if delay > 0:
                            time.sleep(delay)
                        continue
                    break
                wakeups.append(next_review_poll)
                timeout = max(0, min(wakeups) - time.monotonic())
                done, _ = wait(running, timeout=timeout, return_when=FIRST_COMPLETED)
                for future in done:
                    work = running.pop(future)
                    # Unexpected worker failures remain leased for explicit
                    # recovery; they are not silently converted into success.
                    path = future.result()
                    result = load_result(path)
                    if result.work != work:
                        raise ValueError("worker returned a different work identity")
                    receive(path)
                    code = provider_code(result) if work.lane == ExtractionLane.GLM else None
                    if work.lane == ExtractionLane.GLM and code == "1308":
                        quota_exhausted.add(work.lane)
                    elif result.outcome == ExtractionOutcome.RETRY_WAIT and code == "1302":
                        glm_rate_retries.add(work.task_id)
                        now = time.monotonic()
                        if now >= glm_rate_wave_until:
                            glm_limit = max(1, glm_limit // 2)
                            glm_successes = 0
                            # One reduction per returned retry window, not per
                            # sibling response from the same saturated wave.
                            glm_rate_wave_until = now + (result.retry_after or 60)
                    elif result.outcome == ExtractionOutcome.RETRY_WAIT:
                        lane_backoff[work.lane] = time.monotonic() + (result.retry_after or 60)
                    elif work.lane == ExtractionLane.GLM and result.outcome == ExtractionOutcome.CANDIDATE:
                        glm_rate_retries.discard(work.task_id)
                        glm_successes += 1
                        if glm_successes >= glm_limit:
                            glm_limit = min(glm_workers, glm_limit + 1)
                            glm_successes = 0
                    completed += 1
                    if completed % publish_every == 0:
                        publication = db.publish_snapshot()
            for year, generation in unresolved:
                if not db.get_unresolved_house_doc_ids(year, generation):
                    db.mark_house_generation_parse_complete(year, generation)
            publication = db.publish_snapshot()
            return {**pipeline_status(base, years=years, database=db, doc_ids=doc_ids),
                    "completed_attempts": completed, "missing_years": missing_years,
                    "snapshot": publication,
                    **({"APIquota_exhausted": True,
                        "exhausted_lanes": sorted(lane.value for lane in quota_exhausted)}
                       if quota_exhausted else {})}
