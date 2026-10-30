"""Grounded local review authority, immutable receipts and scoped withdrawal."""
from copy import deepcopy
from dataclasses import replace
from datetime import date
import hashlib
import json
from pathlib import Path
import socket

import pytest

from analyzer.ocr import common, coordinator, review, workers
from analyzer.ocr.contracts import ExtractionLane, ExtractionOutcome, ResultBundle
from analyzer.persistence.database import Database
from .conftest import make_ocr_output, make_text_pdf


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refused(*args, **kwargs):
        raise AssertionError("review tests cannot access providers or network")
    monkeypatch.setattr(socket.socket, "connect", refused)
    monkeypatch.setattr(socket, "create_connection", refused)


@pytest.fixture
def candidate(tmp_path):
    pdf = make_text_pdf(tmp_path / "2025/pdfs/review.pdf", [
        "HOUSE PERIODIC TRANSACTION REPORT", "Filer: Jane Doe", "Filed: 01/20/2025",
        "Asset | Type | Transaction Date | Notification Date | Amount | Owner",
        "Apple Inc. (AAPL) | Purchase | 01/15/25 | 01/20/25 | $12,345.67 | SP",
    ])
    digest = hashlib.sha256(pdf.read_bytes()).hexdigest()
    with Database(tmp_path / "congress.duckdb") as db:
        db.conn.execute("INSERT INTO house_archive_generations (archive_year,generation_id,metadata_count,ptr_count) VALUES (2025,'g',1,1)")
        db.conn.execute("INSERT INTO house_generation_metadata (archive_year,generation_id,doc_id,first_name,last_name,filing_date,filing_type) VALUES (2025,'g','review','Jane','Doe','2025-01-20','P')")
        db.conn.execute("INSERT INTO house_pdf_artifacts (archive_year,generation_id,doc_id,artifact_sha256) VALUES (2025,'g','review',?)", [digest])
        work = coordinator._envelope({"archive_year": 2025, "generation_id": "g", "doc_id": "review", "artifact_sha256": digest,
                                      "first_name": "Jane", "last_name": "Doe", "filing_date": "2025-01-20"},
                                     tmp_path, ExtractionLane.GEMINI, workers.extraction_recipe(ExtractionLane.GEMINI))
        db.work.register(work)
        work = db.work.claim(work.task_id)
        raw = make_ocr_output(date="01/15/25", notif_date="01/20/25")
        parsed = common.parse_gemini_output(raw, expected_page_count=1)
        result = ResultBundle(work, ExtractionOutcome.CANDIDATE, parsed.transactions, raw, {
            "artifact_verified": True, "output_schema_version": 4, "page_count": 1,
            "page_outcomes": [{"page_number": 1, "outcome": "transactions"}],
        })
        yield db, result, tmp_path


def source_report(result):
    """Unit oracle for the literal one-row source above, not provider approval."""
    material = review.review_material(result)
    row, = material["projection"]
    assert row["member"] == "Jane Doe"
    assert row["ticker"] == "AAPL"
    assert row["transaction_type"] == "Purchase"
    assert row["transaction_date"] == "2025-01-15"
    assert row["notification_date"] == "2025-01-20"
    assert row["amount_raw"] == "$12,345.67"
    assert row["owner_code"] == "SP"
    report = review.review_report_template(result)
    report["evidence"] = "Synthetic source: complete one-page House PTR, Jane Doe, filed January 20."
    report["pages"][0].update(outcome="transactions", evidence="One actual Apple row; no other rows or examples.")
    report["rows"][0]["evidence"] = "Page 1 sole row: AAPL Purchase 01/15/25 notified 01/20/25 $12,345.67 SP; no instrument/option/account terms."
    report["omissions_checked"] = True
    return report


def ingest(db, result, root):
    path = workers._publish(result, str(root / "pipeline"))
    return coordinator._ingest(db, result, Path(path), root, None)


def raw_receipt(db, result):
    row = db.conn.execute("SELECT result_id,task_id,attempt_id,path FROM extraction_results WHERE task_id=? ORDER BY finished_at LIMIT 1",
                          [result.work.task_id]).fetchone()
    return dict(zip(("result_id", "task_id", "attempt_id", "path"), row))


def test_provider_self_assertions_and_silver_observations_never_authorize(candidate):
    db, result, root = candidate
    forged = replace(result, evidence={**result.evidence, "approved": True, "human_reviewed": True,
                                      "reviewer": "human", "audit": "pass", "review": {"verdict": "pass"}})
    assert not workers.admission_decision(forged, root)["allowed"]
    assert ingest(db, result, root) == "needs_review"
    assert db.conn.execute("SELECT COUNT(*) FROM transactions").fetchone() == (0,)
    review.prepare_review(result, root)
    report = source_report(result)
    with pytest.raises(ValueError, match="human review"):
        review.record_review(result, root, report, verdict="pass")
    review.record_review(result, root, report, verdict="fail")
    assert not workers.admission_decision(result, root)["allowed"]
    with pytest.raises(ValueError, match="re-extraction"):
        review.record_review(result, root, report, verdict="pass", human_reviewed=True)


@pytest.mark.parametrize("missing", ["pages", "rows", "critical_fields", "omissions", "evidence", "binding", "policy"])
def test_positive_review_requires_exact_binding_and_complete_source_coverage(candidate, missing):
    _, result, root = candidate
    review.prepare_review(result, root)
    report = source_report(result)
    if missing in {"pages", "rows"}:
        report[missing] = []
    elif missing == "critical_fields":
        report["rows"][0]["critical_fields"] = ["transaction_type"]
    elif missing == "omissions":
        report["omissions_checked"] = False
    elif missing == "evidence":
        report["rows"][0]["evidence"] = ""
    elif missing == "binding":
        report["binding"]["final_projection_sha256"] = "0" * 64
    else:
        report["policy_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        review.record_review(result, root, report, verdict="pass", human_reviewed=True)
    assert not workers.admission_decision(result, root)["allowed"]


def test_reviewed_readmission_appends_proof_without_retagging_raw_receipt(candidate):
    db, result, root = candidate
    assert ingest(db, result, root) == "needs_review"
    receipt = raw_receipt(db, result)
    original = db.conn.execute("SELECT * FROM extraction_results WHERE result_id=?", [receipt["result_id"]]).fetchone()
    frozen_raw = Path(receipt["path"]).read_bytes()
    directory = Path(review.prepare_review(result, root))
    assert list((directory / "pages").glob("*.png"))
    template = json.loads((directory / "report-template.json").read_text())
    assert template["omissions_checked"] is False
    assert template["pages"][0]["outcome"] == "unknown"
    report = source_report(result)
    review.record_review(result, root, report, verdict="pass", human_reviewed=True)
    assert ingest(db, result, root) == "accepted"
    assert db.conn.execute("SELECT * FROM extraction_results WHERE result_id=?", [receipt["result_id"]]).fetchone() == original
    assert Path(receipt["path"]).read_bytes() == frozen_raw
    effective = db.work.effective_admission_decision(receipt)
    proof = effective["detail"]["detail"]
    assert proof["normalization_recipe"] == review.review_material(result)["binding"]["normalization_recipe"]
    assert proof["policy_sha256"] == review.digest(review.POLICY)
    assert proof["final_projection"] == review.review_material(result)["projection"]
    assert proof["final_projection_sha256"] == review.digest(proof["final_projection"])
    assert proof["review_binding"]["raw_result_id"] == receipt["result_id"]
    assert proof["audit_binding"] is None
    assert db.conn.execute("SELECT ticker,transaction_type,notification_date::VARCHAR,amount_raw FROM canonical_transactions").fetchall() == [
        ("AAPL", "Purchase", "2025-01-20", "$12,345.67")]


@pytest.mark.parametrize("component", ["source.pdf", "record.json", "pages", "bundle-seal.json"])
def test_frozen_source_pixels_candidate_and_seal_are_all_required(candidate, component):
    _, result, root = candidate
    directory = Path(review.prepare_review(result, root))
    review.record_review(result, root, source_report(result), verdict="pass", human_reviewed=True)
    target = next((directory / "pages").glob("*.png")) if component == "pages" else directory / component
    target.write_bytes(b"changed")
    assert not workers.admission_decision(result, root)["allowed"]


@pytest.fixture
def quarantinable(candidate):
    from analyzer.ocr.ingestion import record_parse_run

    db, result, root = candidate
    db.conn.execute("UPDATE house_archive_generations SET promoted_at='2026-08-01'")
    prior = {**review.review_material(result)["projection"][0],
             "source_row_id": "previous-accepted-lot", "amount_raw": "$12,000", "amount_midpoint": 12000.0}
    columns = list(prior)
    db.conn.execute(f"INSERT INTO transactions ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})", list(prior.values()))
    db.conn.execute("UPDATE transactions SET created_at='2026-09-01'")
    record_parse_run(db.conn, "review", 2025, "success", 1, 1, parser_version="gemini-baseline",
                     artifact_sha256=result.work.artifact_sha256, ingestion_generation="g", engines_attempted="gemini/old")
    db.conn.execute("UPDATE pdf_parse_runs SET parsed_at='2026-09-02',rowset_created_at='2026-09-01'")
    # Deliberately unrelated data exercises scope isolation, not just row counts.
    db.conn.execute("INSERT INTO transactions (doc_id,source,ticker,asset_description) VALUES ('unrelated','form4','MSFT','untouched')")
    baseline_rows = db.conn.execute("SELECT * FROM canonical_transactions WHERE doc_id='review'").fetchall()
    assert len(baseline_rows) == 1
    baseline = Path(db.publish_snapshot()["path"])
    review.prepare_review(result, root)
    review.record_review(result, root, source_report(result), verdict="pass", human_reviewed=True)
    assert ingest(db, result, root) == "accepted"
    receipt = raw_receipt(db, result)
    scope = [{**receipt, "lane": "gemini"}]
    yield db, result, root, baseline, scope, baseline_rows


def test_quarantine_restores_prior_business_rows_preserves_temporal_history_and_is_idempotent(quarantinable):
    from scripts.quarantine_model_admissions import quarantine

    db, result, root, baseline, scope, historical = quarantinable
    raw = db.conn.execute("SELECT * FROM extraction_results WHERE result_id=?", [scope[0]["result_id"]]).fetchone()
    unrelated = db.conn.execute("SELECT * FROM transactions WHERE doc_id='unrelated'").fetchall()
    before = db.conn.execute("SELECT * FROM canonical_transactions WHERE doc_id='review'").fetchall()
    assert before != historical
    dry = quarantine(db.conn, scope, baseline, root, "operator quality withdrawal")
    assert dry["pending"] == 1 and not dry["applied"]
    assert db.conn.execute("SELECT * FROM canonical_transactions WHERE doc_id='review'").fetchall() == before
    outcome = quarantine(db.conn, scope, baseline, root, "operator quality withdrawal", apply=True)
    assert outcome["baseline_canonical_rows"] == 1
    with Database(baseline, read_only=True) as baseline_db:
        assert db.conn.execute("SELECT * EXCLUDE(created_at) FROM canonical_transactions WHERE doc_id='review'").fetchall() == baseline_db.conn.execute(
            "SELECT * EXCLUDE(created_at) FROM canonical_transactions WHERE doc_id='review'").fetchall()
    # New restoration timestamps prevent duplicated historical UNION ALL observations.
    with db.canonical_as_of(date(2026, 10, 1)):
        assert db.conn.execute("SELECT * FROM canonical_transactions WHERE doc_id='review'").fetchall() == historical
    assert db.conn.execute("SELECT * FROM extraction_results WHERE result_id=?", [scope[0]["result_id"]]).fetchone() == raw
    assert db.conn.execute("SELECT * FROM transactions WHERE doc_id='unrelated'").fetchall() == unrelated
    assert db.work.effective_admission_decision(scope[0])["state"] == "needs_review"
    assert db.conn.execute("SELECT status FROM pdf_parse_runs WHERE parser_version=?", [result.work.extraction_recipe]).fetchall() == []
    restored = db.conn.execute("SELECT * FROM transactions ORDER BY id").fetchall()
    replay = quarantine(db.conn, scope, baseline, root, "operator quality withdrawal", apply=True)
    assert replay["already_quarantined"] == 1 and replay["pending"] == 0
    assert db.conn.execute("SELECT * FROM transactions ORDER BY id").fetchall() == restored


def test_quarantine_failure_rolls_back_rows_runs_and_decisions(quarantinable, monkeypatch):
    from scripts.quarantine_model_admissions import quarantine
    from analyzer.persistence.work_repository import WorkRepository

    db, _, root, baseline, scope, _ = quarantinable
    tables = ("transactions", "transaction_versions", "pdf_parse_runs", "parse_run_versions", "extraction_results", "extraction_work")
    before = {table: db.conn.execute("SELECT * FROM " + table).fetchall() for table in tables}

    def refuse(*args, **kwargs):
        raise ValueError("injected append failure after row replacement")

    monkeypatch.setattr(WorkRepository, "append_admission_decision", refuse)
    with pytest.raises(ValueError, match="injected append"):
        quarantine(db.conn, scope, baseline, root, "operator quality withdrawal", apply=True)
    for table in tables:
        assert db.conn.execute("SELECT * FROM " + table).fetchall() == before[table]


def test_withdrawn_raw_result_cannot_reuse_its_old_review_to_readmit(quarantinable):
    from scripts.quarantine_model_admissions import quarantine

    db, result, root, baseline, scope, _ = quarantinable
    quarantine(db.conn, scope, baseline, root, "operator quality withdrawal", apply=True)
    withdrawn = db.work.effective_admission_decision(scope[0])
    restored = db.conn.execute("SELECT * FROM canonical_transactions WHERE doc_id='review'").fetchall()
    assert ingest(db, result, root) == "needs_review"
    assert db.work.effective_admission_decision(scope[0]) == withdrawn
    assert db.conn.execute("SELECT * FROM canonical_transactions WHERE doc_id='review'").fetchall() == restored
