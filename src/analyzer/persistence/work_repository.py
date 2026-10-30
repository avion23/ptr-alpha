"""Durable dispatch, attempts and atomic admission receipts on the owner connection."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta
from uuid import uuid4

from analyzer.ocr.contracts import WorkEnvelope, ResultBundle


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


class WorkRepository:
    STATES = {"accepted", "verified_empty", "needs_review", "blocked", "retry_wait", "failed"}

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
                envelope = ?, lease_until = ?, retry_at = NULL, updated_at = CURRENT_TIMESTAMP
                WHERE task_id = ?""", [work.attempt_id, _json(data), datetime.now() + timedelta(seconds=lease_seconds), task_id])
            self.conn.execute("INSERT INTO extraction_attempts (attempt_id, task_id, envelope) VALUES (?, ?, ?)", [work.attempt_id, task_id, _json(data)])
            self.conn.execute("COMMIT")
            return work
        except Exception:
            self.conn.execute("ROLLBACK")
            raise

    def recover_expired(self) -> int:
        rows = self.conn.execute("""UPDATE extraction_work SET status = 'pending',
            lease_until = NULL, updated_at = CURRENT_TIMESTAMP
            WHERE status = 'running' AND lease_until <= CURRENT_TIMESTAMP RETURNING task_id""").fetchall()
        return len(rows)

    def register_result(self, result: ResultBundle, path: str) -> str:
        row = self.conn.execute("SELECT envelope FROM extraction_attempts WHERE attempt_id = ? AND task_id = ?", [result.work.attempt_id, result.work.task_id]).fetchone()
        if row is None or json.loads(row[0]) != result.work.to_dict():
            raise ValueError("Result does not match a registered attempt")
        identity = hashlib.sha256(_json(result.to_dict()).encode()).hexdigest()
        self.conn.execute("""INSERT INTO extraction_results
            (result_id, task_id, attempt_id, path) VALUES (?, ?, ?, ?) ON CONFLICT DO NOTHING""",
            [identity, result.work.task_id, result.work.attempt_id, path])
        return identity

    def is_finished(self, receipt: dict) -> bool:
        row = self.conn.execute("""SELECT finished_at FROM extraction_results
            WHERE result_id = ? AND task_id = ? AND attempt_id = ?""",
            [receipt["result_id"], receipt["task_id"], receipt["attempt_id"]]).fetchone()
        if row is None:
            raise ValueError("Unknown result receipt")
        return row[0] is not None

    def validate_receipt(self, receipt: dict):
        if self.is_finished(receipt):
            return
        row = self.conn.execute("SELECT attempt_id, status FROM extraction_work WHERE task_id = ?", [receipt["task_id"]]).fetchone()
        if row is None or row[0] != receipt["attempt_id"] or row[1] != "running":
            raise ValueError("Result belongs to an inactive attempt")

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
