"""Grounded local review authority, immutable receipts and scoped withdrawal."""
import hashlib
import json
import socket
from dataclasses import replace
from datetime import date
from pathlib import Path

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
def candidate(tmp_path, request):
    occurrences = getattr(request, 'param', 1)
    pdf = make_text_pdf(tmp_path / "2025/pdfs/review.pdf", [
        "HOUSE PERIODIC TRANSACTION REPORT", "Filer: Jane Doe", "Filed: 01/20/2025",
        "Asset | Type | Transaction Date | Notification Date | Amount | Owner",
    ] + ["Apple Inc. (AAPL) | Purchase | 01/15/25 | 01/20/25 | $12,345.67 | SP"] * occurrences)
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
        raw += ('\n' + raw.splitlines()[-1]) * (occurrences - 1)
        parsed = common.parse_gemini_output(raw, expected_page_count=1)
        result = ResultBundle(work, ExtractionOutcome.CANDIDATE, parsed.transactions, raw, {
            "artifact_verified": True, "output_schema_version": 4, "page_count": 1,
            "page_outcomes": [{"page_number": 1, "outcome": "transactions"}],
        })
        yield db, result, tmp_path


def source_report(result):
    """Unit oracle for the literal source above, not provider approval."""
    material = review.review_material(result)
    for row in material["projection"]:
        assert row["member"] == "Jane Doe"
        assert row["ticker"] == "AAPL"
        assert row["transaction_type"] == "Purchase"
        assert row["transaction_date"] == "2025-01-15"
        assert row["notification_date"] == "2025-01-20"
        assert row["amount_raw"] == "$12,345.67"
        assert row["owner_code"] == "SP"
    report = review.review_report_template(result)
    report["evidence"] = "Synthetic source: complete one-page House PTR, Jane Doe, filed January 20."
    report["pages"][0].update(outcome="transactions", evidence=f"{len(result.rows)} actual Apple row occurrences; no other rows or examples.")
    for number, row in enumerate(report['rows'], 1):
        row["evidence"] = f"Page 1 row {number}: AAPL Purchase 01/15/25 notified 01/20/25 $12,345.67 SP; no instrument/option/account terms."
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


def test_coherent_wrong_checkbox_claim_is_not_source_verification(candidate):
    from analyzer.ocr.normalization import normalize_candidate

    db, result, root = candidate  # The literal source says Purchase.
    raw = make_ocr_output(type='Sale', date='01/15/25', notif_date='01/20/25')
    parsed = common.parse_gemini_output(raw, expected_page_count=1)
    wrong = replace(result, rows=parsed.transactions, raw_response=raw)
    normalized = normalize_candidate(wrong.work, wrong)
    assert normalized['status'] == 'candidate' and not normalized['findings']
    assert wrong.rows[0]['type'] == wrong.rows[0]['raw_cells']['type'] == 'Sale'
    # Agreement between two model-produced fields cannot establish pixel fidelity.
    assert ingest(db, wrong, root) == 'needs_review'
    assert db.conn.execute('SELECT COUNT(*) FROM transactions').fetchone() == (0,)


@pytest.mark.parametrize('partial_detection', [False, True])
def test_offline_grid_observations_cannot_authorize_api_admission(candidate, monkeypatch, partial_detection):
    import sys

    from scripts import house_grid_evidence

    _, result, root = candidate
    output = root / 'offline-observations'
    monkeypatch.setattr(sys, 'argv', ['house_grid_evidence', result.work.artifact_path, str(output)])
    monkeypatch.setattr(house_grid_evidence, '_orient_image', lambda image, _: image)
    if partial_detection:
        # Even successful local detector observations are not an all-fields review.
        monkeypatch.setattr(house_grid_evidence, 'analyze_page', lambda *args: {
            'status': 'partial_evidence', 'rows': [{'disposition': 'marked_row_not_certified_transaction',
            'fields': {'direction': {'state': 'observed', 'value': 'PURCHASE'}}}],
        })
    code = house_grid_evidence.main()
    observations = json.loads((output / 'evidence.json').read_text())
    assert code == (0 if partial_detection else 2)
    assert observations['status'] == ('partial_evidence_only' if partial_detection else 'unknown')
    assert observations['approval'] is False
    assert observations['artifact_sha256'] == result.work.artifact_sha256
    assert len(observations['pages']) == 1
    review.prepare_review(result, root)
    with pytest.raises(ValueError, match='exact frozen candidate'):
        review.record_review(result, root, observations, verdict='pass', human_reviewed=True)
    assert not workers.admission_decision(result, root)['allowed']


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
    from analyzer.persistence.work_repository import WorkRepository
    from scripts.quarantine_model_admissions import quarantine

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
    with pytest.raises(ValueError, match="supersedes_admission_decision_id"):
        review.record_review(result, root, source_report(result), verdict="pass", human_reviewed=True, database=db)
    directory = Path(review.prepare_review(result, root, database=db))
    checklist = directory / f"report-template-{withdrawn['result_id']}.json"
    assert json.loads(checklist.read_text())["omissions_checked"] is False
    report = source_report(result)
    report["supersedes_admission_decision_id"] = withdrawn["result_id"]
    review.record_review(result, root, report, verdict="pass", human_reviewed=True, database=db)
    assert ingest(db, result, root) == "accepted"
    assert db.work.effective_admission_decision(scope[0])["result_id"] != withdrawn["result_id"]


def test_actual_review_cli_is_database_free_with_active_writer_and_held_lagging_snapshot(quarantinable, monkeypatch):
    import duckdb
    from typer.testing import CliRunner

    from analyzer import cli
    from analyzer.persistence import database as database_module
    from analyzer.persistence.ownership import (
        publication_pointer,
        publication_state,
        resolve_read_path,
    )
    from scripts.quarantine_model_admissions import quarantine

    db, result, root, baseline, scope, _ = quarantinable
    quarantine(db.conn, scope, baseline, root, 'operator withdrawal', apply=True)
    withdrawn = db.work.effective_admission_decision(scope[0])
    pointer = publication_pointer(db.db_path)
    held = {**json.loads(pointer.read_text()), 'quality_hold': {'reason': 'operator reconciliation required'}}
    pointer.write_text(json.dumps(held))
    pointer_before = pointer.read_bytes()
    assert publication_state(db.db_path)['owner_active']
    assert resolve_read_path(db.db_path) == baseline
    with Database(db.db_path, read_only=True) as pinned:
        assert pinned.conn.execute('SELECT COUNT(*) FROM extraction_results').fetchone() == (0,)
    original_rows = db.conn.execute('SELECT * FROM canonical_transactions').fetchall()
    raw_path = scope[0]['path']
    runner = CliRunner()

    def forbidden(*args, **kwargs):
        raise AssertionError('review CLI opened a database instead of using immutable evidence')

    for acknowledgment in (None, '0' * 64, withdrawn['result_id']):
        prepare_args = ['ocr-review-prepare', raw_path, '--data-dir', str(root)]
        if acknowledgment:
            prepare_args += ['--supersedes-admission-decision', acknowledgment]
        with monkeypatch.context() as guard:
            guard.setattr(database_module.Database, '__init__', forbidden)
            guard.setattr(duckdb, 'connect', forbidden)
            prepared = runner.invoke(cli.app, prepare_args)
        assert prepared.exit_code == 0, prepared.output or repr(prepared.exception)
        directory = Path(prepared.output.strip())
        template_name = f'report-template-{acknowledgment}.json' if acknowledgment else 'report-template.json'
        template = json.loads((directory / template_name).read_text())
        assert template['supersedes_admission_decision_id'] == acknowledgment
        assert template['omissions_checked'] is False
        report = source_report(result)
        report['supersedes_admission_decision_id'] = acknowledgment
        report_path = root / 'operator-report.json'
        report_path.write_text(json.dumps(report))
        record_args = ['ocr-review-record', raw_path, str(report_path), '--data-dir', str(root), '--verdict', 'pass']
        with monkeypatch.context() as guard:
            guard.setattr(database_module.Database, '__init__', forbidden)
            guard.setattr(duckdb, 'connect', forbidden)
            unconfirmed = runner.invoke(cli.app, record_args)
            recorded = runner.invoke(cli.app, [*record_args, '--human-reviewed'])
        assert unconfirmed.exit_code != 0
        assert recorded.exit_code == 0, recorded.output or repr(recorded.exception)
        assert pointer.read_bytes() == pointer_before
        state = ingest(db, result, root)
        if acknowledgment != withdrawn['result_id']:
            assert state == 'needs_review'
            assert db.work.effective_admission_decision(scope[0]) == withdrawn
            assert db.conn.execute('SELECT * FROM canonical_transactions').fetchall() == original_rows
        else:
            assert state == 'accepted'
        assert pointer.read_bytes() == pointer_before  # Review never publishes or releases the hold.


def test_successful_model_empty_claim_requires_grounded_all_page_absence_review(candidate, monkeypatch):
    db, _, root = candidate
    pdf = make_text_pdf(root / "2025/pdfs/empty.pdf", [
        "HOUSE PERIODIC TRANSACTION REPORT", "Filer: Jane Doe", "Filed: 01/20/2025",
        "No transactions reported. No data rows, continuation pages, or examples.",
    ])
    sha = hashlib.sha256(pdf.read_bytes()).hexdigest()
    db.conn.execute("INSERT INTO house_generation_metadata (archive_year,generation_id,doc_id,first_name,last_name,filing_date,filing_type) VALUES (2025,'g','empty','Jane','Doe','2025-01-20','P')")
    db.conn.execute("INSERT INTO house_pdf_artifacts (archive_year,generation_id,doc_id,artifact_sha256) VALUES (2025,'g','empty',?)", [sha])
    work = coordinator._envelope({'archive_year': 2025, 'generation_id': 'g', 'doc_id': 'empty',
                                 'artifact_sha256': sha, 'first_name': 'Jane', 'last_name': 'Doe', 'filing_date': '2025-01-20'},
                                 root, ExtractionLane.GEMINI, workers.extraction_recipe(ExtractionLane.GEMINI))
    db.work.register(work)
    work = db.work.claim(work.task_id)
    raw = "SCHEMA: 4\nMEMBER: Jane Doe\nPAGES: 1\nPAGE: 1\nNO_TRANSACTIONS"
    parsed = common.parse_gemini_output(raw, expected_page_count=1)

    def provider(path, **kwargs):
        return raw, '', common.ArtifactMetadata(sha, 1, len(pdf.read_bytes()), str(path), parsed)

    monkeypatch.setattr(common, 'call_gemini', provider)
    result = workers.load_result(workers.extract_document(work, str(root / 'pipeline')))
    assert result.outcome == ExtractionOutcome.CANDIDATE  # Reviewable claim, not verified absence.
    assert result.rows == []
    assert [(page['page_number'], page['outcome']) for page in result.evidence['page_outcomes']] == [(1, 'unknown')]
    assert ingest(db, result, root) == 'needs_review'
    review.prepare_review(result, root)
    report = review.review_report_template(result)
    report['evidence'] = 'Unit oracle: complete one-page source explicitly says no transactions; no table rows.'
    report['omissions_checked'] = True
    with pytest.raises(ValueError, match='affirmative source-grounded coverage'):
        review.record_review(result, root, report, verdict='pass', human_reviewed=True)
    report['pages'][0].update(outcome='verified_nontransaction', evidence='Whole page checked: filer header and no-transactions statement only.')
    review.record_review(result, root, report, verdict='pass', human_reviewed=True)
    assert ingest(db, result, root) == 'verified_empty'
    assert db.conn.execute("SELECT status,transaction_count FROM pdf_parse_runs WHERE doc_id='empty'").fetchall() == [('no_txs', 0)]


@pytest.mark.parametrize('raw,error', [('', None), ('garbage', None),
    ('SCHEMA: 4\nMEMBER: Jane Doe\nPAGES: 1\nPAGE: 1\nNO_TRANSACTIONS', {'failure': {'type': 'TimeoutError'}})])
def test_failed_or_garbage_empty_results_cannot_be_prepared_for_positive_review(candidate, raw, error):
    _, result, root = candidate
    empty = replace(result, rows=[], raw_response=raw, evidence={**result.evidence, **(error or {}),
                    'page_outcomes': [{'page_number': 1, 'outcome': 'unknown'}]})
    with pytest.raises(ValueError):
        review.prepare_review(empty, root)


def test_sql_projection_failure_rolls_back_canonical_rows_before_receipt_acceptance(candidate, monkeypatch):
    from analyzer.ocr import ingestion

    db, result, root = candidate
    review.prepare_review(result, root)
    review.record_review(result, root, source_report(result), verdict="pass", human_reviewed=True)
    verify = ingestion._verified_final_admission

    def corrupt_before_verification(conn, doc_id, admission):
        conn.execute("UPDATE transactions SET transaction_type='Sale' WHERE doc_id=?", [doc_id])
        return verify(conn, doc_id, admission)

    monkeypatch.setattr(ingestion, "_verified_final_admission", corrupt_before_verification)
    with pytest.raises(ValueError, match="final SQL rows differ"):
        ingest(db, result, root)
    assert db.conn.execute("SELECT COUNT(*) FROM transactions").fetchone() == (0,)
    assert db.conn.execute("SELECT COUNT(*) FROM pdf_parse_runs WHERE status='success'").fetchone() == (0,)
    assert db.work.effective_admission_decision(raw_receipt(db, result))["state"] == "needs_review"


def test_review_authority_expires_at_guard_exit(candidate):
    from analyzer.ocr.ingestion import insert_transactions_outcome

    db, result, root = candidate
    review.prepare_review(result, root)
    review.record_review(result, root, source_report(result), verdict="pass", human_reviewed=True)
    with workers.admission_guard(result, root) as authority:
        assert review.has_review_authority(authority)
    assert not review.has_review_authority(authority)
    outcome = insert_transactions_outcome(
        result.work.doc_id, 2025, result.work.member, authority["candidate"]["rows"],
        conn=db.conn, admission=authority, artifact_sha256=result.work.artifact_sha256,
        ingestion_generation="g", parser_version=result.work.extraction_recipe, checkpoint=False)
    assert outcome.status == "needs_review"
    assert db.conn.execute("SELECT COUNT(*) FROM transactions").fetchone() == (0,)


@pytest.mark.parametrize('candidate', [2], indirect=True)
def test_reviewed_repeated_lots_keep_distinct_occurrences_and_replay_does_not_duplicate(candidate):
    db, result, root = candidate
    review.prepare_review(result, root)
    review.record_review(result, root, source_report(result), verdict='pass', human_reviewed=True)
    assert ingest(db, result, root) == 'accepted'
    accepted = db.conn.execute('SELECT * FROM extraction_results').fetchall()
    rows = db.conn.execute('SELECT source_row_id FROM canonical_transactions ORDER BY source_row_id').fetchall()
    assert len(rows) == 2 and rows[0] != rows[1]
    assert ingest(db, result, root) == 'accepted'
    assert db.conn.execute('SELECT * FROM extraction_results').fetchall() == accepted
    assert db.conn.execute('SELECT source_row_id FROM canonical_transactions ORDER BY source_row_id').fetchall() == rows


@pytest.mark.parametrize('trade,notification,expected_sources,local_status', [
    ('2025-01-21', '2025-01-21', ['gemini_ocr'], 'invalidated'),
    ('2025-01-15', '2025-01-14', ['gemini_ocr'], 'invalidated'),
    ('2025-01-15', '2025-01-15', ['gemini_ocr', 'house_pdf'], 'success'),
])
def test_reviewed_replacement_invalidates_only_semantically_invalid_local_rows(candidate, trade, notification, expected_sources, local_status):
    import pandas as pd

    db, result, root = candidate
    local = {**review.review_material(result)['projection'][0], 'source': 'house_pdf',
             'source_row_id': 'local:r1', 'transaction_date': trade, 'notification_date': notification}
    db.upsert_transactions(pd.DataFrame([local]), source='house_pdf')
    db.parse_runs.upsert(doc_id='review', year=2025, parser_version='pdftotext', status='success',
                         engines_attempted='pdftotext', raw_row_count=1, transaction_count=1,
                         artifact_sha256=result.work.artifact_sha256, ingestion_generation='g')
    review.prepare_review(result, root)
    review.record_review(result, root, source_report(result), verdict='pass', human_reviewed=True)
    assert ingest(db, result, root) == 'accepted'
    assert [row[0] for row in db.conn.execute('SELECT source FROM transactions ORDER BY source').fetchall()] == expected_sources
    assert db.conn.execute("SELECT status FROM pdf_parse_runs WHERE parser_version='pdftotext'").fetchone() == (local_status,)
