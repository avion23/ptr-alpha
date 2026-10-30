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
        unresolved = sum(len(db.get_unresolved_house_doc_ids(y, g)) for y, g in generations)
        return {"scope": len(scope), "unresolved": unresolved,
                "resolved": len(scope) - unresolved, "work": db.work.status(),
                "publication": publication_state(path)}


def _envelope(scope: dict, data_dir: Path, lane: ExtractionLane) -> WorkEnvelope:
    from analyzer.ocr.common import pdf_page_count
    from analyzer.ocr.ingestion import local_work_artifact
    from analyzer.ocr.workers import extraction_recipe

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
    recipe = extraction_recipe(lane)
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
        return row[0]
    if not _current_binding(db, work):
        db.work.finish(receipt, "blocked", "source generation/artifact is no longer current")
        return "blocked"
    if result.outcome == ExtractionOutcome.RETRY_WAIT:
        delay = result.retry_after if result.retry_after is not None else 60
        db.work.finish(receipt, "retry_wait", result.detail,
                       retry_at=datetime.now() + timedelta(seconds=delay))
        return "retry_wait"
    candidate = normalize_candidate(work, result)
    gate = admission_decision(result, data_dir)
    status = candidate["status"]
    if not gate["allowed"] or status not in {"candidate", "verified_empty"}:
        disposition = "blocked" if status == "blocked" else "needs_review"
        db.work.finish(receipt, disposition, gate["reason"] or result.detail or str(candidate["findings"]))
        return disposition
    if work.lane == ExtractionLane.LOCAL and status == "candidate":
        source._save_parse_results(
            work.archive_year,
            [(Path(work.artifact_path),
              result.rows, result.evidence.get("engines_attempted", []))],
            {work.doc_id: {"First": work.member, "Last": "", "FilingDate": work.filing_date}},
            work.generation_id, recipe=work.extraction_recipe, receipt=receipt,
        )
    else:
        insert_transactions_outcome(
            work.doc_id, work.archive_year, work.member, candidate["rows"],
            conn=db.conn, receipt=receipt, evidence=result.evidence,
            raw_count=len(result.rows), parser_version=work.extraction_recipe,
            artifact_sha256=work.artifact_sha256, ingestion_generation=work.generation_id,
            checkpoint=False,
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
    base = Path(data_dir).resolve()
    spool = base / "pipeline" / "results"
    spool.mkdir(parents=True, exist_ok=True)
    settings = Settings()
    settings.data.data_dir = str(base)
    with nullcontext(database) if database is not None else Database(base / "congress.duckdb") as db:
        source = HouseTransactionSource(settings, db=db)
        with ExitStack() as stack:
            stack.callback(source.close)
            scope = _scope(db, years)
            scoped = {(s["archive_year"], s["generation_id"], s["doc_id"]): s for s in scope}
            missing_years = sorted(set(years or []) - {s["archive_year"] for s in scope})
            for year in missing_years:
                db.work.register_blocked(f"missing-generation:{year}", "no official PTR generation scope",
                                         scope={"archive_year": year})

            def escalate(result, state):
                if state != "needs_review" or result.work.lane not in lanes:
                    return
                index = lanes.index(result.work.lane) + 1
                if index >= len(lanes):
                    return
                old = result.work
                lane = lanes[index]
                recipe = extraction_recipe(lane)
                work = replace(old, task_id=task_key(old.doc_id, old.archive_year, old.generation_id,
                               old.artifact_sha256, recipe, lane), attempt_id=str(uuid4()),
                               lane=lane, extraction_recipe=recipe)
                db.work.register(work, priority=100)

            def receive(path):
                result = load_result(path)
                key = (result.work.archive_year, result.work.generation_id, result.work.doc_id)
                if key not in scoped:
                    return
                state = _ingest(db, result, Path(path), base, source)
                escalate(result, state)

            # Import durable responses before recovering leases or dispatching
            # replacements. A completed response must not trigger another bill.
            for path in sorted(spool.rglob("*.json")):
                receive(path)
            db.work.recover_expired()
            unresolved = { (y, g): set(db.get_unresolved_house_doc_ids(y, g))
                           for y, g in {(s["archive_year"], s["generation_id"]) for s in scope} }
            for item in scope:
                needed = item["doc_id"] in unresolved[item["archive_year"], item["generation_id"]]
                if not needed and not refresh_recipe:
                    continue
                try:
                    if not item["artifact_sha256"]:
                        raise FileNotFoundError("official PTR has no acquired artifact")
                    work = _envelope(item, base, lanes[0])
                except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
                    key = f'blocked:{item["archive_year"]}:{item["generation_id"]}:{item["doc_id"]}'
                    db.work.register_blocked(key, str(exc), scope=item)
                    continue
                db.work.register(work, priority=100 if needed else 0)

            pools = {}
            for lane in lanes:
                if lane == ExtractionLane.LOCAL:
                    pool = ProcessPoolExecutor(capacities[lane], mp_context=multiprocessing.get_context("spawn"))
                else:
                    pool = ThreadPoolExecutor(capacities[lane], thread_name_prefix=lane.value)
                pools[lane] = stack.enter_context(pool)
            running = {}
            lane_backoff = {lane: 0.0 for lane in lanes}
            completed = 0
            publication = None
            while True:
                for lane in lanes:
                    free = capacities[lane] - sum(w.lane == lane for w in running.values())
                    if not free or time.monotonic() < lane_backoff[lane]:
                        continue
                    # One bounded rolling window across all years, replenished
                    # after every completion rather than after a batch barrier.
                    for queued in db.work.pending(lane, limit=free):
                        if (queued.archive_year, queued.generation_id, queued.doc_id) not in scoped:
                            continue
                        work = db.work.claim(queued.task_id, lease_seconds=lease_seconds)
                        running[pools[lane].submit(extract_document, work, str(spool))] = work
                if not running:
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
            for year, generation in unresolved:
                if not db.get_unresolved_house_doc_ids(year, generation):
                    db.mark_house_generation_parse_complete(year, generation)
            publication = db.publish_snapshot()
            return {**pipeline_status(base, years=years, database=db),
                    "completed_attempts": completed, "missing_years": missing_years,
                    "snapshot": publication}
