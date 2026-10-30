"""Single-owner coordinator regressions: temporary scope and DB-free workers."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import socket
import threading
from uuid import uuid4

import pytest

from analyzer.ocr import coordinator, workers
from analyzer.ocr.contracts import ExtractionLane, ExtractionOutcome, ResultBundle, WorkEnvelope, task_key
from analyzer.persistence.database import Database
from analyzer.persistence.ownership import OwnershipError, publication_state


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refused(*args, **kwargs):
        raise AssertionError("pipeline tests must not access the network")

    monkeypatch.setattr(socket.socket, "connect", refused)
    monkeypatch.setattr(socket, "create_connection", refused)


@pytest.fixture
def scope(tmp_path, monkeypatch):
    from analyzer.ocr import common

    monkeypatch.setattr(common, "pdf_page_count", lambda _: 1)
    content = b"%PDF-bound-fixture"
    digest = hashlib.sha256(content).hexdigest()
    with Database(tmp_path / "congress.duckdb") as db:
        db.conn.execute("INSERT INTO house_archive_generations (archive_year, generation_id) VALUES (2025, 'g')")
        for doc_id in ("a", "b", "c", "d"):
            pdf = tmp_path / "2025" / "pdfs" / f"{doc_id}.pdf"
            pdf.parent.mkdir(parents=True, exist_ok=True)
            pdf.write_bytes(content)
            db.conn.execute("INSERT INTO house_generation_metadata (archive_year, generation_id, doc_id, first_name, last_name, filing_date, filing_type) VALUES (2025, 'g', ?, 'Jane', 'Doe', '2025-01-20', 'P')", [doc_id])
            db.conn.execute("INSERT INTO house_pdf_artifacts (archive_year, generation_id, doc_id, artifact_sha256) VALUES (2025, 'g', ?, ?)", [doc_id, digest])
    return tmp_path, digest


def bound_work(root, digest, doc_id="a"):
    lane = ExtractionLane.LOCAL
    recipe = workers.extraction_recipe(lane)
    return WorkEnvelope(
        task_key(doc_id, 2025, "g", digest, recipe, lane), str(uuid4()), doc_id,
        2025, "g", digest, str(root / "2025" / "pdfs" / f"{doc_id}.pdf"),
        "Jane Doe", "2025-01-20", 1, recipe, lane,
    )


def test_late_superseded_bundle_replays_without_poisoning_current_attempt(scope):
    root, digest = scope
    with Database(root / "congress.duckdb") as db:
        db.conn.execute("DELETE FROM house_generation_metadata WHERE doc_id != 'a'")
        original = bound_work(root, digest)
        db.work.register(original)
        attempt_a = db.work.claim(original.task_id)
        db.conn.execute("UPDATE extraction_work SET lease_until = CURRENT_TIMESTAMP - INTERVAL '1 SECOND'")
        assert db.work.recover_expired() == 1
        attempt_b = db.work.claim(original.task_id)
        assert attempt_b.attempt_id != attempt_a.attempt_id
        before = db.conn.execute("SELECT envelope, status, attempt_id, lease_until FROM extraction_work WHERE task_id = ?", [original.task_id]).fetchone()
        workers._publish(ResultBundle(attempt_a, ExtractionOutcome.BLOCKED, [], detail="late A"), str(root / "pipeline"))
        for _ in range(2):
            summary = coordinator.run_pipeline(root, years=[2025], database=db)
            assert summary["completed_attempts"] == 0
            assert db.conn.execute("SELECT envelope, status, attempt_id, lease_until FROM extraction_work WHERE task_id = ?", [original.task_id]).fetchone() == before
            assert db.conn.execute("SELECT state, finished_at IS NOT NULL FROM extraction_results WHERE attempt_id = ?", [attempt_a.attempt_id]).fetchall() == [("superseded", True)]


def test_rolling_local_and_provider_lanes_share_one_owner(scope, monkeypatch):
    root, digest = scope
    with Database(root / "congress.duckdb") as db:
        for priority, doc in enumerate(("d", "c", "b", "a"), 1):
            db.work.register(bound_work(root, digest, doc), priority=priority * 100)
    owner_thread = threading.get_ident()
    writer_opens = []
    real_init = Database.__init__

    def observed_init(self, *args, **kwargs):
        if not kwargs.get("read_only", False):
            writer_opens.append(threading.get_ident())
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(Database, "__init__", observed_init)
    pools = []

    def local_pool(count, *, mp_context):
        assert mp_context.get_start_method() == "spawn"
        pools.append(count)
        return ThreadPoolExecutor(count, thread_name_prefix="local-fixture")

    monkeypatch.setattr(coordinator, "ProcessPoolExecutor", local_pool)
    straggler_started = threading.Event()
    rolling_started = threading.Event()
    api_started = threading.Event()
    calls = []
    call_lock = threading.Lock()

    def extract(work, spool):
        assert threading.get_ident() != owner_thread
        state = publication_state(root / "congress.duckdb")
        assert state["owner_active"] and state["published"]
        assert set(work.to_dict()).isdisjoint({"db", "db_path", "connection"})
        with call_lock:
            calls.append((work.lane, work.doc_id))
        if work.lane == ExtractionLane.LOCAL:
            if work.doc_id == "a":
                straggler_started.set()
                assert rolling_started.wait(10), "rolling queue waited for the straggler"
                assert api_started.wait(10), "API lane did not overlap local work"
            elif work.doc_id == "b":
                assert straggler_started.wait(10)
            elif work.doc_id == "c":
                assert straggler_started.is_set()
                rolling_started.set()
            outcome = ExtractionOutcome.NEEDS_REVIEW if work.doc_id in {"b", "c"} else ExtractionOutcome.BLOCKED
            result = ResultBundle(work, outcome, [], evidence={
                "artifact_verified": True, "page_count": 1,
                "page_outcomes": [{"page_number": 1, "outcome": "unknown"}],
            }, detail="fixture local disposition")
        elif work.lane == ExtractionLane.GEMINI:
            result = ResultBundle(work, ExtractionOutcome.RETRY_WAIT, [], detail="quota", retry_after=3600)
        else:
            api_started.set()
            result = ResultBundle(work, ExtractionOutcome.NEEDS_REVIEW, [], evidence={"artifact_verified": True}, detail="audit required")
        return workers._publish(result, spool)

    monkeypatch.setattr(workers, "extract_document", extract)
    summary = coordinator.run_pipeline(root, years=[2025], local_workers=2, gemini_workers=1, glm_workers=1)
    assert pools == [2]
    assert writer_opens == [owner_thread]
    assert len([call for call in calls if call[0] == ExtractionLane.LOCAL]) == 4
    assert len([call for call in calls if call[0] == ExtractionLane.GEMINI]) == 1
    assert {doc for lane, doc in calls if lane == ExtractionLane.GLM} == {"b", "c"}
    assert summary["work"]["by_state"]["retry_wait"] == 1
    assert summary["work"]["by_state"]["pending"] == 1
    assert summary["publication"]["published"]
    assert not publication_state(root / "congress.duckdb")["owner_active"]
    with Database(root / "congress.duckdb", read_only=True) as reader:
        assert reader.conn.execute("SELECT COUNT(*) FROM transactions").fetchone() == (0,)


def test_pipeline_reuses_existing_owner_and_refuses_competing_writer(scope):
    root, _ = scope
    with Database(root / "congress.duckdb") as owner:
        with pytest.raises(OwnershipError, match="writer owner"):
            coordinator.run_pipeline(root, years=[2025])
        owner.publish_snapshot()
        status = coordinator.pipeline_status(root, years=[2025])
        assert status["scope"] == status["unresolved"] == 4
        assert coordinator.pipeline_status(root, years=[2025], database=owner)["scope"] == 4
