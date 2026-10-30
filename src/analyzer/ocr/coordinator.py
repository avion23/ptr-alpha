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
import multiprocessing
from pathlib import Path
import time
from uuid import uuid4

from analyzer.ocr.contracts import ExtractionLane, ExtractionOutcome, WorkEnvelope, task_key
from analyzer.persistence.database import Database


def _scope(db: Database, years: list[int] | None) -> list[dict]:
    """Official latest-generation PTR scope, including missing acquisitions."""
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
    return [dict(archive_year=int(y), generation_id=g, doc_id=str(d),
                 first_name=f or "", last_name=l or "", filing_date=str(dt)[:10],
                 artifact_sha256=sha)
            for y, g, d, f, l, dt, sha in rows if years is None or y in years]


def pipeline_status(data_dir: str | Path, *, years: list[int] | None = None,
                    database: Database | None = None) -> dict:
    """Read one published version, or project the already-owned writer connection."""
    from analyzer.persistence.ownership import publication_state, resolve_read_path

    path = Path(data_dir) / "congress.duckdb"
    with nullcontext(database) if database is not None else Database(resolve_read_path(path), read_only=True) as db:
        scope = _scope(db, years)
        generations = {(s["archive_year"], s["generation_id"]) for s in scope}
        unresolved_docs = {(y, g, d) for y, g in generations
                           for d in db.get_unresolved_house_doc_ids(y, g)}
        unresolved = len(unresolved_docs)
        import json
        from collections import Counter

        tasks = {}
        for envelope, state in db.conn.execute("SELECT envelope, status FROM extraction_work").fetchall():
            item = json.loads(envelope)
            key = (item.get("archive_year"), item.get("generation_id"), item.get("doc_id"))
            tasks.setdefault(key, set()).add(state)
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
                "resolved": len(scope) - unresolved, "work": db.work.status(),
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


def _ingest(db: Database, result, path: Path, data_dir: Path, source) -> str:
    from analyzer.ocr.ingestion import insert_transactions_outcome
    from analyzer.ocr.normalization import normalize_candidate
    from analyzer.ocr.workers import admission_decision

    work = result.work
    receipt = dict(task_id=work.task_id, attempt_id=work.attempt_id,
                   result_id=db.work.register_result(result, str(path)))
    if db.work.is_finished(receipt):
        row = db.conn.execute("SELECT state FROM extraction_results WHERE result_id = ?",
                              [receipt["result_id"]]).fetchone()
        # Re-admission is a new decision over the SAME extraction attempt,
        # not a new extraction or a rewrite of its historical refusal.
        current = db.conn.execute("SELECT attempt_id, status FROM extraction_work WHERE task_id = ?",
                                  [work.task_id]).fetchone()
        if (row[0] != "needs_review" or work.lane != ExtractionLane.GLM
                or current != (work.attempt_id, "needs_review")
                or not _current_binding(db, work)):
            return row[0]
        gate = admission_decision(result, data_dir)
        if not gate["allowed"]:
            return row[0]
        from analyzer.ocr.workers import _publish

        decision = replace(result, evidence={**result.evidence, "admission_binding": gate["audit_binding"]})
        if decision == result:
            return row[0]
        decision_path = Path(_publish(decision, str(data_dir / "pipeline")))
        db.conn.execute("BEGIN TRANSACTION")
        try:
            db.conn.execute("UPDATE extraction_work SET status = 'running', updated_at = CURRENT_TIMESTAMP WHERE task_id = ?",
                            [work.task_id])
            db.work.register_result(decision, str(decision_path))
            db.conn.execute("COMMIT")
        except Exception:
            db.conn.execute("ROLLBACK")
            raise
        return _ingest(db, decision, decision_path, data_dir, source)
    db.work.validate_receipt(receipt)
    if not _current_binding(db, work):
        db.work.finish(receipt, "blocked", "source generation/artifact is no longer current")
        return "blocked"
    if result.outcome == ExtractionOutcome.RETRY_WAIT:
        delay = result.retry_after if result.retry_after is not None else 60
        db.work.finish(receipt, "retry_wait", result.detail,
                       retry_at=datetime.now() + timedelta(seconds=delay))
        return "retry_wait"
    if work.lane == ExtractionLane.GLM and result.evidence.get("staging_record"):
        from scripts import glm_stage
        record = result.evidence["staging_record"]
        existing = glm_stage.load_staging_record(data_dir, work.archive_year, work.doc_id)
        if existing is None or glm_stage.audit_staged_record_sha256(existing) != glm_stage.audit_staged_record_sha256(record):
            glm_stage.publish_staging_record(data_dir, record)
    gate = admission_decision(result, data_dir)
    local_uncertain = (work.lane == ExtractionLane.LOCAL
                       and result.evidence.get("artifact_verified") is True
                       and result.evidence.get("failure", {}).get("type") == "ParserCascadeError")
    # Deterministic rows already have the local cascade schema, not the OCR
    # schema. _save_parse_results owns their existing complete-rowset
    # consolidation/validation, so do not OCR-normalize them a second time.
    local_candidate = (work.lane == ExtractionLane.LOCAL
                       and result.outcome == ExtractionOutcome.CANDIDATE and bool(result.rows))
    candidate = None if local_candidate else normalize_candidate(work, result)
    status = result.outcome.value if candidate is None else candidate["status"]
    if not gate["allowed"] or status not in {"candidate", "verified_empty"}:
        permanent = (status == "blocked" and not local_uncertain) or gate["reason"] in {
            "extraction recipe changed", "artifact verification evidence missing"}
        disposition = "blocked" if permanent else "needs_review"
        db.work.finish(receipt, disposition, result.detail or gate["reason"] or str(candidate["findings"] if candidate else []))
        return disposition
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
            raw_count=len(result.rows), parser_version=work.extraction_recipe,
            artifact_sha256=work.artifact_sha256, ingestion_generation=work.generation_id,
            engine_model=model, checkpoint=False,
        )
    row = db.conn.execute("SELECT state FROM extraction_results WHERE result_id = ?",
                          [receipt["result_id"]]).fetchone()
    if not db.work.is_finished(receipt):
        raise RuntimeError("canonical persistence did not atomically finish its result receipt")
    return row[0]


def run_pipeline(
    data_dir: str | Path, years: list[int] | None = None, *,
    local_workers: int = 3, gemini_workers: int = 0, glm_workers: int = 0,
    refresh_recipe: bool = False, publish_every: int = 100,
    lease_seconds: int = 3600, database: Database | None = None,
) -> dict:
    """Drain ready work. API use is explicit; deferred retries survive for next run.

    A provider retry never consumes another provider's slots. Unresolved local
    results may escalate to configured API lanes; a retry is not an escalation.
    """
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
        source = HouseTransactionSource(settings, db=db)
        with ExitStack() as stack:
            stack.callback(source.close)
            scope = _scope(db, years)
            scoped = {(s["archive_year"], s["generation_id"], s["doc_id"]): s for s in scope}
            known_years = {int(r[0]) for r in db.conn.execute(
                "SELECT DISTINCT archive_year FROM house_archive_generations").fetchall()}
            if years is None and not known_years:
                raise ValueError("no official House generation scope; acquire archives first or specify required years")
            missing_years = sorted(set(years or []) - known_years)
            for year in missing_years:
                db.work.register_blocked(f"missing-generation:{year}", "no official PTR generation scope",
                                         scope={"archive_year": year})

            def escalate(result, state):
                if state != "needs_review" or result.work.lane not in lanes:
                    return
                old = result.work
                if old.doc_id not in db.get_unresolved_house_doc_ids(old.archive_year, old.generation_id):
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
                result = load_result(path)
                key = (result.work.archive_year, result.work.generation_id, result.work.doc_id)
                if key not in scoped:
                    return
                state = _ingest(db, result, Path(path), base, source)
                escalate(result, state)

            # Import durable responses before recovering leases or dispatching
            # replacements. A completed response must not trigger another bill.
            for path in sorted((spool / "results").glob("*.json")):
                receive(path)
            # Receipts can reference a durable bundle outside the default spool.
            for receipt in db.work.pending_results(limit=2147483647):
                receive(receipt["path"])
            db.work.recover_expired()
            unresolved = { (y, g): set(db.get_unresolved_house_doc_ids(y, g))
                           for y, g in {(s["archive_year"], s["generation_id"]) for s in scope} }
            def seed(item):
                needed = item["doc_id"] in unresolved[item["archive_year"], item["generation_id"]]
                if not needed and not refresh_recipe:
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
            seeds = iter(sorted(scope, key=lambda s:
                s["doc_id"] not in unresolved[s["archive_year"], s["generation_id"]]))
            seeds_done = False

            def ready(lane, limit):
                # Scope and recipe filters prevent an older/year-excluded queue
                # head from hiding ready work. The repository owns claiming.
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
                      AND (w.status = 'pending' OR (w.status = 'retry_wait' AND w.retry_at <= CURRENT_TIMESTAMP))
                    ORDER BY w.priority DESC, w.task_id LIMIT ?
                """, [lane.value, recipes[lane], sorted({s["archive_year"] for s in scope}), limit]).fetchall()

            def publication_signature():
                return (db.conn.execute("SELECT COUNT(*), MAX(updated_at) FROM extraction_work").fetchone(),
                        db.conn.execute("SELECT COUNT(*), MAX(finished_at) FROM extraction_results").fetchone(),
                        db.conn.execute("SELECT archive_year, generation_id, parse_status FROM house_archive_generations ORDER BY 1, 2").fetchall())

            publication = db.publish_snapshot()
            published_signature = publication_signature()
            pools = {}
            for lane in lanes:
                if lane == ExtractionLane.LOCAL:
                    pool = ProcessPoolExecutor(capacities[lane], mp_context=multiprocessing.get_context("spawn"))
                else:
                    pool = ThreadPoolExecutor(capacities[lane], thread_name_prefix=lane.value)
                pools[lane] = stack.enter_context(pool)
            running = {}
            lane_backoff = {}
            for lane in lanes:
                delay = db.conn.execute("""SELECT epoch(MAX(retry_at) - CURRENT_TIMESTAMP)
                    FROM extraction_work WHERE lane = ? AND status = 'retry_wait'""",
                    [lane.value]).fetchone()[0]
                lane_backoff[lane] = time.monotonic() + max(0, delay or 0)
            completed = 0
            while True:
                if not seeds_done:
                    window = max(1, sum(capacities.values()) * 2)
                    budget = window - len(ready(lanes[0], window))
                    for _ in range(budget):
                        item = next(seeds, None)
                        if item is None:
                            seeds_done = True
                            break
                        seed(item)
                for lane in lanes:
                    free = capacities[lane] - sum(w.lane == lane for w in running.values())
                    if not free or time.monotonic() < lane_backoff[lane]:
                        continue
                    # One bounded rolling window across all years, replenished
                    # after every completion rather than after a batch barrier.
                    for (task_id,) in ready(lane, free):
                        work = db.work.claim(task_id, lease_seconds=lease_seconds)
                        running[pools[lane].submit(extract_document, work, str(spool))] = work
                if not running:
                    if time.monotonic() < lane_backoff[lanes[0]]:
                        break
                    if not seeds_done:
                        continue
                    break
                done, _ = wait(running, return_when=FIRST_COMPLETED)
                for future in done:
                    work = running.pop(future)
                    # Unexpected worker failures remain leased for explicit
                    # recovery; they are not silently converted into success.
                    path = future.result()
                    result = load_result(path)
                    if result.work != work:
                        raise ValueError("worker returned a different work identity")
                    receive(path)
                    if result.outcome == ExtractionOutcome.RETRY_WAIT:
                        lane_backoff[work.lane] = time.monotonic() + (result.retry_after or 60)
                    completed += 1
                    if completed % publish_every == 0:
                        publication = db.publish_snapshot()
                        published_signature = publication_signature()
            for year, generation in unresolved:
                if not db.get_unresolved_house_doc_ids(year, generation):
                    db.mark_house_generation_parse_complete(year, generation)
            if publication_signature() != published_signature:
                publication = db.publish_snapshot()
            return {**pipeline_status(base, years=years, database=db),
                    "completed_attempts": completed, "missing_years": missing_years,
                    "snapshot": publication}
