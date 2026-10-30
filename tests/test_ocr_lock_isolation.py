"""Single-owner coordinator regressions: temporary scope and DB-free workers."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
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
        db.conn.execute("INSERT INTO house_archive_generations (archive_year, generation_id, metadata_count, ptr_count) VALUES (2025, 'g', 4, 4)")
        for doc_id in ("a", "b", "c", "d"):
            pdf = tmp_path / "2025" / "pdfs" / f"{doc_id}.pdf"
            pdf.parent.mkdir(parents=True, exist_ok=True)
            pdf.write_bytes(content)
            db.conn.execute("INSERT INTO house_generation_metadata (archive_year, generation_id, doc_id, first_name, last_name, filing_date, filing_type) VALUES (2025, 'g', ?, 'Jane', 'Doe', '2025-01-20', 'P')", [doc_id])
            db.conn.execute("INSERT INTO house_pdf_artifacts (archive_year, generation_id, doc_id, artifact_sha256) VALUES (2025, 'g', ?, ?)", [doc_id, digest])
    return tmp_path, digest


def bound_work(root, digest, doc_id="a", lane=ExtractionLane.LOCAL):
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
        db.conn.execute("DELETE FROM house_pdf_artifacts WHERE doc_id != 'a'")
        db.conn.execute("UPDATE house_archive_generations SET metadata_count=1, ptr_count=1")
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


def test_local_watchdog_publishes_db_free_uncertainty_and_escalates(scope, monkeypatch):
    import duckdb
    import signal
    from analyzer import download
    from analyzer.parsing import cascade

    root, digest = scope
    with Database(root / "congress.duckdb") as owner:
        owner.conn.execute("DELETE FROM house_generation_metadata WHERE doc_id != 'a'")
        owner.conn.execute("DELETE FROM house_pdf_artifacts WHERE doc_id != 'a'")
        owner.conn.execute("UPDATE house_archive_generations SET metadata_count=1, ptr_count=1")
        work = bound_work(root, digest)
        owner.work.register(work)
        claimed = owner.work.claim(work.task_id)
    prior_handler = signal.getsignal(signal.SIGALRM)

    def stalled_backend(path):
        time.sleep(5)
        raise AssertionError("watchdog did not interrupt the backend")

    def no_database(*args, **kwargs):
        raise AssertionError("extraction worker attempted database access")

    with monkeypatch.context() as worker_patch:
        worker_patch.setattr(download, "_PARSE_DOC_BUDGET_SECONDS", 0.02)
        worker_patch.setattr(cascade, "parse_pdf_worker", stalled_backend)
        worker_patch.setattr(duckdb, "connect", no_database)
        worker_patch.setattr(workers.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout="", stderr="", returncode=0))
        bundle_path = workers.extract_document(claimed, str(root / "pipeline"))
    result = workers.load_result(bundle_path)
    assert result.outcome == ExtractionOutcome.BLOCKED
    assert result.rows == []
    assert result.evidence["artifact_verified"] is True
    assert result.evidence["failure"]["type"] == "ParserCascadeError"
    assert "parse budget 0.02s exceeded" in result.evidence["failure"]["message"]
    assert signal.getsignal(signal.SIGALRM) == prior_handler
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)
    calls = []

    def quota(work, spool):
        calls.append(work.lane)
        return workers._publish(ResultBundle(work, ExtractionOutcome.RETRY_WAIT, [], retry_after=3600), spool)

    monkeypatch.setattr(workers, "extract_document", quota)
    summary = coordinator.run_pipeline(root, [2025], local_workers=1, gemini_workers=1)
    assert calls == [ExtractionLane.GEMINI]
    assert summary["completed_attempts"] == 1
    assert summary["work"]["by_state"] == {"needs_review": 1, "retry_wait": 1}


def test_zero_document_jobs_publish_metadata_and_price_changes(tmp_path):
    path = tmp_path / "congress.duckdb"
    with Database(path) as owner:
        owner.conn.execute("INSERT INTO house_archive_generations (archive_year, generation_id, metadata_count, ptr_count) VALUES (2025, 'empty', 0, 0)")
        owner.conn.execute("INSERT INTO metadata (doc_id, first_name, filing_type) VALUES ('non-ptr', 'Before', 'A')")
        owner.conn.execute("INSERT INTO prices (ticker, date, close) VALUES ('AAPL', '2025-01-02', 100)")
        before = owner.publish_snapshot()
        with Database(path, read_only=True) as pinned:
            owner.conn.execute("UPDATE metadata SET first_name = 'After' WHERE doc_id='non-ptr'")
            owner.conn.execute("UPDATE prices SET close = 125 WHERE ticker='AAPL'")
            result = coordinator.run_pipeline(tmp_path, [2025], database=owner)
            assert result["completed_attempts"] == 0
            assert result["scope"] == 0
            assert result["snapshot"]["version"] != before["version"]
            assert pinned.conn.execute("SELECT first_name FROM metadata").fetchall() == [("Before",)]
            assert pinned.conn.execute("SELECT close FROM prices").fetchall() == [(100.0,)]
            with Database(path, read_only=True) as reader:
                assert reader.conn.execute("SELECT first_name FROM metadata").fetchall() == [("After",)]
                assert reader.conn.execute("SELECT close FROM prices").fetchall() == [(125.0,)]


def test_same_binding_has_one_provider_call_and_no_duplicate_on_replay(scope, monkeypatch):
    root, digest = scope
    with Database(root / "congress.duckdb") as owner:
        owner.conn.execute("DELETE FROM house_generation_metadata WHERE doc_id != 'a'")
        owner.conn.execute("DELETE FROM house_pdf_artifacts WHERE doc_id != 'a'")
        owner.conn.execute("UPDATE house_archive_generations SET metadata_count=1, ptr_count=1")
        for lane in ExtractionLane:
            owner.work.register(bound_work(root, digest, lane=lane))
    monkeypatch.setattr(coordinator, "ProcessPoolExecutor", lambda slots, **kwargs: ThreadPoolExecutor(slots))
    calls = []
    row = {"asset": "Apple (AAPL)", "type": "Purchase", "date": "01/15/25",
           "notif_date": None, "amount_raw": "$1,234.56", "owner": "SP", "account": None,
           "instrument_type": "unknown", "raw_cells": {"Amount": "$1,234.56"}, "page_number": 1}

    def extract(work, spool):
        calls.append(work.lane)
        assert work.lane != ExtractionLane.GLM, "a second provider spent quota on an already admitted binding"
        local = work.lane == ExtractionLane.LOCAL
        evidence = {"artifact_verified": True, "page_count": 1, "output_schema_version": 4,
                    "page_outcomes": [{"page_number": 1, "outcome": "unknown" if local else "transactions"}]}
        result = ResultBundle(work, ExtractionOutcome.NEEDS_REVIEW if local else ExtractionOutcome.CANDIDATE,
                              [] if local else [row], raw_response=None if local else json.dumps(row), evidence=evidence)
        return workers._publish(result, spool)

    monkeypatch.setattr(workers, "extract_document", extract)
    first = coordinator.run_pipeline(root, [2025], local_workers=1, gemini_workers=1, glm_workers=1)
    second = coordinator.run_pipeline(root, [2025], local_workers=1, gemini_workers=1, glm_workers=1)
    assert calls == [ExtractionLane.LOCAL, ExtractionLane.GEMINI]
    assert first["completed_attempts"] == 2
    assert second["completed_attempts"] == 0
    assert second["unresolved"] == 0
    assert second["work"]["by_state"] == {"accepted": 1, "needs_review": 1, "superseded": 1}
    with Database(root / "congress.duckdb", read_only=True) as reader:
        assert reader.conn.execute("SELECT COUNT(*) FROM transactions").fetchone() == (1,)


@pytest.mark.parametrize("counts", [(2, 2), (None, None), (0, None)])
@pytest.mark.parametrize("operation", ["run", "status"])
def test_incomplete_official_scope_counts_cannot_report_completion(tmp_path, counts, operation):
    with Database(tmp_path / "congress.duckdb") as owner:
        owner.conn.execute("INSERT INTO house_archive_generations (archive_year, generation_id, metadata_count, ptr_count) VALUES (2025, 'missing-scope', ?, ?)", list(counts))
        owner.publish_snapshot()
    with pytest.raises(ValueError):
        if operation == "status":
            coordinator.pipeline_status(tmp_path, years=[2025])
        else:
            coordinator.run_pipeline(tmp_path, [2025])


def test_cron_module_entrypoint_runs_actual_spawn_worker_without_api(tmp_path):
    from tests.test_glm_stage import make_pdf

    root = Path(__file__).resolve().parents[1]
    data = tmp_path / "data"
    pdf = make_pdf(data / "2025" / "pdfs" / "spawn-probe.pdf")
    digest = hashlib.sha256(pdf.read_bytes()).hexdigest()
    with Database(data / "congress.duckdb") as owner:
        owner.conn.execute("INSERT INTO house_archive_generations (archive_year, generation_id, metadata_count, ptr_count) VALUES (2025, 'spawn', 1, 1)")
        owner.conn.execute("INSERT INTO house_generation_metadata (archive_year, generation_id, doc_id, first_name, last_name, filing_date, filing_type) VALUES (2025, 'spawn', 'spawn-probe', 'Jane', 'Doe', '2025-01-20', 'P')")
        owner.conn.execute("INSERT INTO house_pdf_artifacts (archive_year, generation_id, doc_id, artifact_sha256) VALUES (2025, 'spawn', 'spawn-probe', ?)", [digest])
    env = dict(os.environ, PYTHONPATH=str(root / "src"), PTR_SKIP_DOCS="spawn-probe")
    process = subprocess.run(
        [sys.executable, "-m", "analyzer.cli", "pipeline", "--data-dir", str(data),
         "--year", "2025", "--local-workers", "1", "--gemini-workers", "0", "--glm-workers", "0"],
        cwd=root, env=env, capture_output=True, text=True, timeout=60,
    )
    assert process.returncode == 1, process.stdout + process.stderr
    summary = json.loads(process.stdout)
    assert summary["completed_attempts"] == 1
    assert summary["unresolved"] == 1
    assert summary["work"]["by_state"] == {"needs_review": 1}
    bundle = workers.load_result(next((data / "pipeline" / "results").glob("*.json")))
    assert bundle.work.lane == ExtractionLane.LOCAL
    assert bundle.evidence["failure"]["type"] == "ParserCascadeError"
    assert "skipped via PTR_SKIP_DOCS" in bundle.evidence["failure"]["message"]


@pytest.mark.parametrize("number", ["NaN", "Infinity", "1e999"])
def test_nonfinite_raw_source_is_durable_refusal_without_stopping_other_documents(scope, monkeypatch, number):
    from analyzer.ocr import common

    root, _ = scope
    with Database(root / "congress.duckdb") as owner:
        owner.conn.execute("DELETE FROM house_generation_metadata WHERE doc_id NOT IN ('a', 'b')")
        owner.conn.execute("DELETE FROM house_pdf_artifacts WHERE doc_id NOT IN ('a', 'b')")
        owner.conn.execute("UPDATE house_archive_generations SET metadata_count=2, ptr_count=2")
    row = {"asset": "Apple (AAPL)", "type": "Purchase", "date": "01/15/25",
           "notif_date": None, "amount_raw": "$1,234", "owner": None, "account": None,
           "instrument_type": "unknown", "raw_cells": {"nested": [{"number": "NUMBER_TOKEN"}]}}
    prefix = "SCHEMA: 4\nMEMBER: Jane Doe\nPAGES: 1\nPAGE: 1\nROW: "
    invalid = prefix + json.dumps(row).replace('"NUMBER_TOKEN"', number)
    valid = prefix + json.dumps(row).replace('"NUMBER_TOKEN"', '123')
    run = common.subprocess.run
    calls = []

    def provider(command, *args, **kwargs):
        if command[0] != "llm":
            return run(command, *args, **kwargs)
        doc = Path(command[command.index("-a") + 1]).stem
        calls.append(doc)
        return SimpleNamespace(returncode=0, stdout=invalid if doc == "a" else valid, stderr="")

    monkeypatch.setattr(common.subprocess, "run", provider)
    summary = coordinator.run_pipeline(root, [2025], local_workers=0, gemini_workers=1)
    assert sorted(calls) == ["a", "b"]
    assert summary["completed_attempts"] == 2
    assert summary["work"]["by_state"] == {"accepted": 1, "needs_review": 1}
    bundles = {bundle.work.doc_id: bundle for bundle in (
        workers.load_result(path) for path in (root / "pipeline" / "results").glob("*.json"))}
    assert bundles["a"].raw_response == invalid
    assert bundles["a"].rows == []
    assert "invalid_response" in bundles["a"].evidence["provider_error"]
    assert len(bundles["b"].rows) == 1
    with Database(root / "congress.duckdb", read_only=True) as reader:
        assert reader.conn.execute("SELECT doc_id FROM transactions").fetchall() == [("b",)]
