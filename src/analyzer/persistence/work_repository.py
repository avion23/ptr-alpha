"""Durable dispatch, attempts and atomic admission receipts on the owner connection."""
from __future__ import annotations

import hashlib
import json
from uuid import uuid4

from analyzer.ocr.contracts import ResultBundle, WorkEnvelope


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


class WorkRepository:
    STATES = frozenset({"accepted", "verified_empty", "needs_review", "blocked", "retry_wait", "failed", "superseded"})

    def __init__(self, conn):
        self.conn = conn

    def init_schema(self):
        self.conn.execute("""CREATE TABLE IF NOT EXISTS extraction_work (
            task_id VARCHAR PRIMARY KEY, envelope JSON, lane VARCHAR,
            status VARCHAR NOT NULL, priority INTEGER DEFAULT 0,
            attempt_id VARCHAR, lease_until TIMESTAMP, retry_at TIMESTAMP,
            detail VARCHAR, result_id VARCHAR, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)
        """)
        self.conn.execute("""CREATE TABLE IF NOT EXISTS extraction_attempts (
            attempt_id VARCHAR PRIMARY KEY, task_id VARCHAR NOT NULL,
            envelope JSON NOT NULL, claimed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)
        """)
        self.conn.execute("""CREATE TABLE IF NOT EXISTS extraction_results (
            result_id VARCHAR PRIMARY KEY, task_id VARCHAR NOT NULL,
            attempt_id VARCHAR NOT NULL, path VARCHAR NOT NULL,
            state VARCHAR DEFAULT 'extracted', detail VARCHAR,
            received_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, finished_at TIMESTAMP)
        """)

    def register(self, work: WorkEnvelope, priority=0):
        self.conn.execute("""INSERT INTO extraction_work
            (task_id, envelope, lane, status, priority) VALUES (?, ?, ?, 'pending', ?)
            ON CONFLICT DO NOTHING""", [work.task_id, _json(work.to_dict()), work.lane.value, priority])

    def register_blocked(self, task_id: str, detail: str, *, scope: dict, priority=0):
        """Account for official scope lacking an artifact; not an extraction envelope."""
        if not task_id or not detail:
            raise ValueError("Blocked scope requires identity and reason")
        self.conn.execute("""INSERT INTO extraction_work
            (task_id, envelope, status, priority, detail) VALUES (?, ?, 'blocked', ?, ?)
            ON CONFLICT DO NOTHING""", [task_id, _json(scope), priority, detail])

    def supersede_pending(self, work: WorkEnvelope, detail: str) -> int:
        """Cancel queued producers for an accepted binding; retain all attempts/results."""
        if not isinstance(detail, str) or not detail.strip():
            raise ValueError("Superseding work requires a reason")
        rows = self.conn.execute("""UPDATE extraction_work SET status = 'superseded',
            detail = ?, lease_until = NULL, retry_at = NULL, updated_at = CURRENT_TIMESTAMP
            WHERE status IN ('pending', 'retry_wait')
              AND json_extract_string(envelope, '$.doc_id') = ?
              AND json_extract_string(envelope, '$.archive_year') = ?
              AND json_extract_string(envelope, '$.generation_id') = ?
              AND json_extract_string(envelope, '$.artifact_sha256') = ?
            RETURNING task_id""", [detail, work.doc_id, str(work.archive_year),
                                    work.generation_id, work.artifact_sha256]).fetchall()
        return len(rows)

    def pending(self, lane, limit=100):
        rows = self.conn.execute("""SELECT envelope FROM extraction_work
            WHERE lane = ? AND (status = 'pending' OR
                (status = 'retry_wait' AND retry_at <= CURRENT_TIMESTAMP))
            ORDER BY priority DESC, task_id LIMIT ?""", [getattr(lane, "value", lane), limit]).fetchall()
        return [WorkEnvelope.from_dict(json.loads(row[0])) for row in rows]

    def claim(self, task_id, lease_seconds=300) -> WorkEnvelope:
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        self.conn.execute("BEGIN TRANSACTION")
        try:
            row = self.conn.execute("""SELECT envelope FROM extraction_work WHERE task_id = ?
                AND (status = 'pending' OR (status = 'retry_wait' AND retry_at <= CURRENT_TIMESTAMP))""", [task_id]).fetchone()
            if row is None:
                raise ValueError("Task is not ready to claim")
            data = json.loads(row[0])
            data["attempt_id"] = str(uuid4())
            work = WorkEnvelope.from_dict(data)
            self.conn.execute("""UPDATE extraction_work SET status = 'running', attempt_id = ?,
                envelope = ?, lease_until = CURRENT_TIMESTAMP + (? * INTERVAL '1 SECOND'),
                retry_at = NULL, updated_at = CURRENT_TIMESTAMP
                WHERE task_id = ?""", [work.attempt_id, _json(data), lease_seconds, task_id])
            self.conn.execute("INSERT INTO extraction_attempts (attempt_id, task_id, envelope) VALUES (?, ?, ?)", [work.attempt_id, task_id, _json(data)])
            self.conn.execute("COMMIT")
            return work
        except Exception:
            self.conn.execute("ROLLBACK")
            raise

    def recover_expired(self) -> int:
        rows = self.conn.execute("""UPDATE extraction_work SET status = 'pending',
            lease_until = NULL, updated_at = CURRENT_TIMESTAMP
            WHERE status = 'running' AND lease_until <= CURRENT_TIMESTAMP
              AND NOT EXISTS (SELECT 1 FROM extraction_results r
                  WHERE r.task_id = extraction_work.task_id
                    AND r.attempt_id = extraction_work.attempt_id
                    AND r.finished_at IS NULL)
            RETURNING task_id""").fetchall()
        return len(rows)

    def pending_results(self, limit=100) -> list[dict]:
        """Replay durable deliveries before dispatching fresh extraction work."""
        rows = self.conn.execute("""SELECT r.task_id, r.attempt_id, r.result_id, r.path
            FROM extraction_results r JOIN extraction_work w USING (task_id, attempt_id)
            WHERE r.finished_at IS NULL AND w.status = 'running'
            ORDER BY r.received_at, r.result_id LIMIT ?""", [limit]).fetchall()
        return [dict(zip(("task_id", "attempt_id", "result_id", "path"), row)) for row in rows]

    def register_result(self, result: ResultBundle, path: str) -> str:
        row = self.conn.execute("SELECT envelope FROM extraction_attempts WHERE attempt_id = ? AND task_id = ?", [result.work.attempt_id, result.work.task_id]).fetchone()
        if row is None or json.loads(row[0]) != result.work.to_dict():
            raise ValueError("Result does not match a registered attempt")
        identity = hashlib.sha256(_json(result.to_dict()).encode()).hexdigest()
        current = self.conn.execute("SELECT attempt_id, status FROM extraction_work WHERE task_id = ?",
                                    [result.work.task_id]).fetchone()
        superseded = current is None or current != (result.work.attempt_id, "running")
        self.conn.execute("""INSERT INTO extraction_results
            (result_id, task_id, attempt_id, path, state, detail, finished_at)
            VALUES (?, ?, ?, ?, ?, ?, CASE WHEN ? THEN CURRENT_TIMESTAMP ELSE NULL END)
            ON CONFLICT (result_id) DO UPDATE SET state = excluded.state,
                detail = excluded.detail, finished_at = excluded.finished_at
            WHERE extraction_results.finished_at IS NULL AND excluded.state = 'superseded'""",
            [identity, result.work.task_id, result.work.attempt_id, path,
             "superseded" if superseded else "extracted",
             "Result belongs to an inactive attempt" if superseded else None, superseded])
        return identity

    def is_finished(self, receipt: dict) -> bool:
        row = self.conn.execute("""SELECT finished_at FROM extraction_results
            WHERE result_id = ? AND task_id = ? AND attempt_id = ?""",
            [receipt["result_id"], receipt["task_id"], receipt["attempt_id"]]).fetchone()
        if row is None:
            raise ValueError("Unknown result receipt")
        return row[0] is not None

    def validate_receipt(self, receipt: dict) -> WorkEnvelope:
        finished = self.is_finished(receipt)
        if not finished:
            row = self.conn.execute("SELECT attempt_id, status FROM extraction_work WHERE task_id = ?", [receipt["task_id"]]).fetchone()
            if row is None or row[0] != receipt["attempt_id"] or row[1] != "running":
                raise ValueError("Result belongs to an inactive attempt")
        row = self.conn.execute("SELECT envelope FROM extraction_attempts WHERE task_id = ? AND attempt_id = ?",
                                [receipt["task_id"], receipt["attempt_id"]]).fetchone()
        if row is None:
            raise ValueError("Unknown receipt attempt")
        return WorkEnvelope.from_dict(json.loads(row[0]))

    def finish(self, receipt: dict, state: str, detail=None, retry_at=None, in_transaction=False):
        if state not in self.STATES:
            raise ValueError(f"Unknown disposition: {state}")
        if state == "retry_wait" and retry_at is None:
            raise ValueError("retry_wait requires retry_at")
        if not in_transaction:
            self.conn.execute("BEGIN TRANSACTION")
        try:
            self.validate_receipt(receipt)
            if not self.is_finished(receipt):
                self.conn.execute("""UPDATE extraction_results SET state = ?, detail = ?,
                    finished_at = CURRENT_TIMESTAMP WHERE result_id = ?""", [state, detail, receipt["result_id"]])
                self.conn.execute("""UPDATE extraction_work SET status = ?, detail = ?, retry_at = ?,
                    lease_until = NULL, result_id = ?, updated_at = CURRENT_TIMESTAMP WHERE task_id = ?""",
                    [state, detail, retry_at, receipt["result_id"], receipt["task_id"]])
            if not in_transaction:
                self.conn.execute("COMMIT")
        except Exception:
            if not in_transaction:
                self.conn.execute("ROLLBACK")
            raise

    def status(self) -> dict:
        counts = dict(self.conn.execute("SELECT status, COUNT(*) FROM extraction_work GROUP BY status").fetchall())
        return {"total": sum(counts.values()), "by_state": counts,
                "results": self.conn.execute("SELECT COUNT(*) FROM extraction_results").fetchone()[0]}

    def effective_admission_decision(self, receipt: dict) -> dict:
        """Follow append-only decisions without rewriting the extraction receipt."""
        keys = ("result_id", "task_id", "attempt_id", "path")
        current = {key: receipt[key] for key in keys}
        seen = set()
        while True:
            if current["result_id"] in seen:
                raise ValueError("Admission decision cycle")
            seen.add(current["result_id"])
            row = self.conn.execute("""SELECT state, detail, finished_at FROM extraction_results
                WHERE result_id = ? AND task_id = ? AND attempt_id = ? AND path = ?""",
                [current[key] for key in keys]).fetchone()
            if row is None or row[2] is None:
                raise ValueError("Admission decision requires a finished exact receipt")
            successors = self.conn.execute("""SELECT result_id, task_id, attempt_id, path
                FROM extraction_results WHERE task_id = ? AND attempt_id = ?
                  AND json_valid(detail)
                  AND json_extract_string(TRY_CAST(detail AS JSON), '$.kind') = 'admission_decision'
                  AND json_extract_string(TRY_CAST(detail AS JSON), '$.predecessor.result_id') = ?""",
                [current["task_id"], current["attempt_id"], current["result_id"]]).fetchall()
            if not successors:
                detail = json.loads(row[1]) if row[1] and row[1].startswith("{") else row[1]
                return {**current, "state": row[0], "detail": detail}
            if len(successors) != 1:
                raise ValueError("Conflicting admission decisions")
            current = dict(zip(keys, successors[0]))

    def append_admission_decision(self, predecessor_receipt: dict, decision: ResultBundle,
                                  path: str, state: str, detail: dict,
                                  in_transaction=False) -> dict:
        """Append an immutable decision and advance only the still-current attempt."""
        if state not in {"accepted", "verified_empty", "needs_review", "blocked"}:
            raise ValueError("Unsupported admission decision state")
        if not isinstance(detail, dict):
            raise TypeError("Admission decision detail must be a dictionary")
        expected_outcome = {"accepted": "candidate", "verified_empty": "verified_empty",
                            "needs_review": "needs_review", "blocked": "blocked"}[state]
        if decision.outcome.value != expected_outcome:
            raise ValueError("Decision outcome does not match disposition")
        from analyzer.ocr.workers import load_result

        if load_result(path) != decision:
            raise ValueError("Decision does not match its immutable result bundle")
        predecessor = {key: predecessor_receipt[key]
                       for key in ("result_id", "task_id", "attempt_id", "path")}
        identity = hashlib.sha256(_json(decision.to_dict()).encode()).hexdigest()
        receipt = {"result_id": identity, "task_id": decision.work.task_id,
                   "attempt_id": decision.work.attempt_id, "path": str(path)}
        event = _json({"kind": "admission_decision", "predecessor": predecessor,
                       "decision": decision.to_dict(), "detail": detail})
        if not in_transaction:
            self.conn.execute("BEGIN TRANSACTION")
        try:
            if self.validate_receipt(predecessor) != decision.work:
                raise ValueError("Decision does not match predecessor extraction attempt")
            effective = self.effective_admission_decision(predecessor)
            current = self.conn.execute("""SELECT attempt_id, result_id, status FROM extraction_work
                WHERE task_id = ?""", [decision.work.task_id]).fetchone()
            if current != (decision.work.attempt_id, effective["result_id"], effective["state"]):
                raise ValueError("Admission scope changed: task has newer work")
            existing = self.conn.execute("""SELECT task_id, attempt_id, path, state, detail
                FROM extraction_results WHERE result_id = ?""", [identity]).fetchone()
            if existing is not None:
                if (existing != (receipt["task_id"], receipt["attempt_id"], receipt["path"], state, event)
                        or effective["result_id"] != identity):
                    raise ValueError("Admission scope changed or decision identity reused")
            else:
                if effective["result_id"] != predecessor["result_id"]:
                    raise ValueError("Admission scope changed: predecessor has a newer decision")
                self.conn.execute("""INSERT INTO extraction_results
                    (result_id, task_id, attempt_id, path, state, detail, finished_at)
                    VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)""",
                    [identity, receipt["task_id"], receipt["attempt_id"], receipt["path"], state, event])
                self.conn.execute("""UPDATE extraction_work SET status = ?, detail = ?,
                    result_id = ?, lease_until = NULL, retry_at = NULL,
                    updated_at = CURRENT_TIMESTAMP WHERE task_id = ?""",
                    [state, event, identity, receipt["task_id"]])
            if not in_transaction:
                self.conn.execute("COMMIT")
            return receipt
        except Exception:
            if not in_transaction:
                self.conn.execute("ROLLBACK")
            raise
