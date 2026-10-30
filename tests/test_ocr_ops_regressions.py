import json
import runpy
import subprocess
import sys
from datetime import date
from pathlib import Path

import pytest

from analyzer.member_names import canonical_member_key
from analyzer.ocr import common as common
from analyzer.ocr import ingestion as ops
from analyzer.ocr import runner
from analyzer.persistence.database import Database
from .conftest import make_ocr_output, make_text_pdf


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    import socket

    def refused(*args, **kwargs):
        raise AssertionError("OCR tests must not access the network")

    monkeypatch.setattr(socket.socket, "connect", refused)
    monkeypatch.setattr(socket, "create_connection", refused)


@pytest.fixture
def stored(tmp_path, monkeypatch):
    path = tmp_path / 'congress.duckdb'
    pdf = tmp_path / '2024' / 'pdfs' / '1.pdf'
    make_text_pdf(pdf, ["HOUSE PTR - Jane Doe - Filed 02/01/2024",
                        "Asset | Type | Trade Date | Notification Date | Amount | Owner",
                        "Apple (AAPL) | Purchase | 01/15/24 | 01/20/24 | $12,345.67 | SP"])
    monkeypatch.setattr(common, 'pdf_page_count', lambda _: 1)
    digest = common.pdf_sha256(pdf)
    db = Database(path)
    db.conn.execute("INSERT INTO metadata (doc_id, first_name, last_name, filing_date, filing_type) VALUES ('1', 'Jane', 'Doe', '2024-02-01', 'P')")
    db.conn.execute("INSERT INTO house_archive_generations (archive_year, generation_id, metadata_count, ptr_count) VALUES (2024, 'g', 1, 1)")
    db.conn.execute("INSERT INTO house_generation_metadata (archive_year, generation_id, doc_id, first_name, last_name, filing_date, filing_type) VALUES (2024, 'g', '1', 'Jane', 'Doe', '2024-02-01', 'P')")
    db.conn.execute("INSERT INTO house_pdf_artifacts (archive_year, generation_id, doc_id, artifact_sha256) VALUES (2024, 'g', '1', ?)", [digest])
    ops.record_parse_run(db.conn, '1', 2024, 'zero_rows', 0, 0, parser_version='deterministic', engines_attempted='pdfplumber')
    db.close()
    return path, pdf, digest


def cache_and_insert(stored, notification='01/20/24'):
    from analyzer.ocr import coordinator, review, workers
    from analyzer.ocr.contracts import ExtractionLane, ExtractionOutcome, ResultBundle

    path, pdf, digest = stored
    notification = None if notification == 'N/A' else notification
    make_text_pdf(pdf, ["HOUSE PTR - Jane Doe - Filed 02/01/2024",
                        "Asset | Type | Trade Date | Notification Date | Amount | Owner",
                        f"Apple (AAPL) | Purchase | 01/15/24 | {notification or ''} | $12,345.67 | SP"])
    digest = common.pdf_sha256(pdf)
    output = make_ocr_output(asset="Apple (AAPL)", notif_date=notification)
    common.write_cached_response('1', pdf, output, str(path.parent / 'gemini_cache'))
    parsed = common.parse_gemini_output(output)
    with Database(path) as db:
        db.conn.execute("UPDATE house_pdf_artifacts SET artifact_sha256=? WHERE doc_id='1'", [digest])
        work = coordinator._envelope(dict(archive_year=2024, generation_id='g', doc_id='1',
                                         artifact_sha256=digest, first_name='Jane', last_name='Doe', filing_date='2024-02-01'),
                                     path.parent, ExtractionLane.GEMINI, workers.extraction_recipe(ExtractionLane.GEMINI))
        db.work.register(work)
        work = db.work.claim(work.task_id)
        result = ResultBundle(work, ExtractionOutcome.CANDIDATE, parsed.transactions, output, {
            "artifact_verified": True, "output_schema_version": 4, "page_count": 1,
            "page_outcomes": [{"page_number": 1, "outcome": "transactions"}],
        })
        review.prepare_review(result, path.parent)
        report = review.review_report_template(result)
        projection, = review.review_material(result)["projection"]
        assert (projection["ticker"], projection["transaction_type"], projection["transaction_date"],
                projection["notification_date"], projection["amount_raw"], projection["owner_code"]) == (
                    "AAPL", "Purchase", "2024-01-15", "2024-01-20" if notification else None, "$12,345.67", "SP")
        report["evidence"] = "Unit oracle: synthetic Jane Doe House PTR filed 02/01/24, one Apple row."
        report["pages"][0].update(outcome="transactions", evidence="Entire page has only the declared Apple row; no examples or omissions.")
        report["rows"][0]["evidence"] = f"Apple (AAPL), Purchase, 01/15/24, notification {notification or 'blank'}, $12,345.67, SP; no option or instrument terms."
        report["omissions_checked"] = True
        review.record_review(result, path.parent, report, verdict="pass", human_reviewed=True)
        result_path = workers._publish(result, str(path.parent / 'pipeline'))
        assert coordinator._ingest(db, result, Path(result_path), path.parent, None) == "accepted"


@pytest.mark.parametrize(
    "raw", [None, "not a valid source response"],
)
def test_success_requires_valid_source_response(stored, monkeypatch, raw):
    path, _, _ = stored
    monkeypatch.setattr(common, "call_gemini", lambda *args, **kwargs: (raw, "", None))
    result = runner.run_gemini_ocr_for_year(2024, str(path.parent))
    assert result["unresolved"] == 1
    assert result["work"]["by_state"] in ({"needs_review": 1}, {"blocked": 1})
    from analyzer.ocr.workers import load_result
    from analyzer.ocr.contracts import ExtractionOutcome
    bundle = load_result(next((path.parent / "pipeline" / "results").glob("*.json")))
    assert bundle.outcome == ExtractionOutcome.BLOCKED
    assert bundle.evidence["failure"]["type"] in {"ValueError", "GeminiOutputError"}
    with Database(path) as db:
        assert db.conn.execute("SELECT COUNT(*) FROM transactions").fetchone() == (0,)


def test_missing_notification_is_resolved(stored):
    cache_and_insert(stored, 'N/A')
    path, _, _ = stored
    with Database(path) as db:
        assert db.get_unresolved_house_doc_ids(2024, 'g') == []
        assert db.conn.execute("SELECT notification_date FROM transactions").fetchall() == [(None,)]


@pytest.mark.parametrize(
    ("instrument", "strike"), [("unknown", None), ("call", "100")],
)
def test_lossless_cache_matches_typed_instrument_projection(stored, instrument, strike):
    path, pdf, digest = stored
    row = {
        "asset": "Apple (AAPL)", "type": "Purchase", "date": "01/15/24",
        "notif_date": "01/20/24", "amount_raw": "$12,345.67", "owner": "SP",
        "account": None, "instrument_type": instrument, "strike_price": strike,
        "expiry_date": None, "raw_cells": {"Amount": "$12,345.67", "Owner": "SP"},
    }
    output = "SCHEMA: 4\nMEMBER: Jane Doe\nPAGES: 1\nPAGE: 1\nROW: " + json.dumps(row)
    common.write_cached_response('1', pdf, output, str(path.parent / 'gemini_cache'))
    parsed = common.parse_gemini_output(output)
    assert not parsed.legacy_limited
    from analyzer.ocr.normalization import canonical_row_projection, normalize_rows
    typed, findings = normalize_rows('1', parsed.member, parsed.transactions, '2024-02-01', 'Jane Doe', digest)
    assert not findings
    normalized, errors = ops._normalize_ocr_rows('1', parsed.member, typed, date(2024, 2, 1),
                                                artifact_sha256=digest, ingestion_generation='g')
    assert not errors
    projected, = normalized
    assert (projected['instrument_type'], projected['strike_price'], projected['amount_raw'], projected['raw_owner']) == (
        instrument, None if strike is None else 100.0, "$12,345.67", "SP")
    sql_unknown = {**projected, 'instrument_type': None if instrument == 'unknown' else instrument}
    assert canonical_row_projection(sql_unknown) == canonical_row_projection(projected)
    # A faithful projection alone still does not authorize publication.
    assert ops.insert_transactions('1', 2024, parsed.member, parsed.transactions, db_path=str(path),
                                   artifact_sha256=digest, ingestion_generation='g') == 0


def test_null_provenance_is_stale(stored):
    cache_and_insert(stored)
    path, _, _ = stored
    db = Database(path)
    db.conn.execute("INSERT INTO transactions (doc_id, source, chamber, asset_description) VALUES ('1', 'gemini_ocr', 'House', 'stale')")
    db.close()
    assert len(ops.get_ocr_work_items(db_path=str(path), data_dir=path.parent)) == 1


def test_unreviewed_semantic_zero_preserves_exact_artifact(stored):
    cache_and_insert(stored)
    path, _, digest = stored
    ops.insert_transactions('1', 2024, 'Jane Doe', [], raw_count=1, db_path=str(path), artifact_sha256=digest, ingestion_generation='g')
    db = Database(path)
    try:
        assert db.conn.execute('SELECT COUNT(*) FROM transactions').fetchone()[0] == 1
    finally:
        db.close()


@pytest.mark.parametrize("admission", [None, {"allowed": True, "review_binding": {"human_reviewed": True}}])
def test_unreviewed_insertion_refuses_before_opening_a_writer(stored, monkeypatch, admission):
    path, _, digest = stored

    def forbidden(*args, **kwargs):
        raise AssertionError("unreviewed API input opened a writer")

    monkeypatch.setattr(ops.duckdb, "connect", forbidden)
    parsed = common.parse_gemini_output(
        "MEMBER: Jane Doe\nPAGES: 1\nPAGE: 1\n"
        "Apple (AAPL) | Purchase | 01/15/24 | 01/20/24 | A"
    )
    outcome = ops.insert_transactions_outcome(
        "1", 2024, parsed.member, parsed.transactions,
        db_path=str(path), artifact_sha256=digest, ingestion_generation="g", admission=admission,
    )
    assert outcome.status == 'needs_review'


def _partial_transactions():
    parsed = common.parse_gemini_output(
        "MEMBER: Jane Doe\nPAGES: 1\nPAGE: 1\n"
        "Microsoft (MSFT) | Purchase | 01/15/24 | 01/20/24 | A\n"
        "Bad date row (BAD) | Purchase | 02/05/24 | 02/10/24 | B"
    )
    return parsed.member, parsed.transactions


def test_partial_ocr_is_retained_but_canonical_commit_is_blocked(stored):
    path, pdf, digest = stored
    cache_dir = str(path.parent / "gemini_cache")
    cache_and_insert(stored)  # existing canonical row must not be deleted
    member, transactions = _partial_transactions()
    common.write_cached_response(
        "1",
        pdf,
        "MEMBER: Jane Doe\nPAGES: 1\nPAGE: 1\n"
        "Microsoft (MSFT) | Purchase | 01/15/24 | 01/20/24 | A\n"
        "Bad date row (BAD) | Purchase | 02/05/24 | 02/10/24 | B",
        cache_dir,
    )

    outcome = ops.insert_transactions_outcome(
        "1",
        2024,
        member,
        transactions,
        db_path=str(path),
        raw_count=2,
        artifact_sha256=digest,
        ingestion_generation="g",
    )

    assert outcome.status == "needs_review"
    assert outcome.inserted == 0
    assert outcome.committed is False and outcome.complete is False
    # Deterministic negative validation remains independent of publication authority.
    _, findings = common.validate_transactions('1', member, transactions, date(2024, 2, 1), 'Jane Doe')
    assert findings == {"date_out_of_window": 1}
    db = Database(path)
    try:
        assert db.conn.execute(
            "SELECT ticker FROM transactions WHERE doc_id='1' AND source='gemini_ocr'"
        ).fetchall() == [("AAPL",)]
        run = db.conn.execute(
            "SELECT status, transaction_count, error_message FROM pdf_parse_runs "
            "WHERE doc_id='1' AND parser_version=?",
            [ops.GEMINI_PARSER_VERSION],
        ).fetchone()
        assert run is None  # No unauthorised direct call may retag an accepted parse run.
        assert 'ocr_partial_results' not in {row[0] for row in db.conn.execute('SHOW TABLES').fetchall()}
    finally:
        db.close()

    cached = common.inspect_cached_response("1", str(pdf), cache_dir)
    assert cached is not None and len(cached.parsed.transactions) == 2
    assert ops.get_ocr_work_items(db_path=str(path), data_dir=path.parent) == [
        ("1", 2024, str(pdf))
    ]


def test_partial_canonical_override_cannot_bypass_operator_review(stored, capsys):
    path, _, digest = stored
    member, transactions = _partial_transactions()
    outcome = ops.insert_transactions_outcome(
        "1",
        2024,
        member,
        transactions,
        db_path=str(path),
        raw_count=2,
        artifact_sha256=digest,
        ingestion_generation="g",
        force_partial_canonical=True,
    )
    assert "FORCED" not in capsys.readouterr().out
    assert outcome.status == "needs_review" and not outcome.committed
    assert not outcome.complete
    db = Database(path)
    try:
        assert db.conn.execute(
            "SELECT ticker FROM transactions WHERE doc_id='1' AND source='gemini_ocr'"
        ).fetchall() == []
        assert db.conn.execute(
            "SELECT status, error_message FROM pdf_parse_runs WHERE doc_id='1' "
            "AND parser_version=?",
            [ops.GEMINI_PARSER_VERSION],
        ).fetchone() is None
        assert 'ocr_partial_results' not in {row[0] for row in db.conn.execute('SHOW TABLES').fetchall()}
    finally:
        db.close()


@pytest.mark.parametrize("mutation", ["replacement", "no_txs", "error"])
def test_ocr_mutations_preserve_accepted_historical_rows(stored, mutation):
    cache_and_insert(stored)
    path, _, digest = stored
    db = Database(path)
    db.conn.execute("UPDATE house_archive_generations SET promoted_at='2024-02-01'")
    db.conn.execute("UPDATE transactions SET created_at='2024-02-02'")
    db.conn.execute("UPDATE pdf_parse_runs SET parsed_at='2024-02-03',rowset_created_at='2024-02-02' WHERE status='success'")
    with db.canonical_as_of(date(2024, 3, 1)):
        assert db.get_transactions(2024).ticker.tolist() == ["AAPL"]
    db.close()
    if mutation == "replacement":
        parsed = common.parse_gemini_output(
            'MEMBER: Jane Doe\nPAGES: 1\nPAGE: 1\nMicrosoft (MSFT) | Purchase | 01/15/24 | 01/20/24 | A')
        ops.insert_transactions('1', 2024, parsed.member, parsed.transactions,
                                db_path=str(path), artifact_sha256=digest, ingestion_generation='g')
    else:
        ops.insert_transactions('1', 2024, 'Jane Doe', [], raw_count=int(mutation == "error"),
                                db_path=str(path), artifact_sha256=digest, ingestion_generation='g')
    db = Database(path)
    try:
        with db.canonical_as_of(date(2024, 3, 1)):
            assert db.get_transactions(2024).ticker.tolist() == ["AAPL"]
        # Initialization's NULL-owner normalization is now versioned as well;
        # assert the actual accepted pre-OCR version, not unrelated migrations.
        assert db.conn.execute("SELECT ticker FROM transactions WHERE created_at='2024-02-02'").fetchall() == [('AAPL',)]
        assert db.conn.execute("SELECT COUNT(*) FROM transaction_versions WHERE created_at='2024-02-02'").fetchone()[0] == 0
    finally:
        db.close()


def test_current_generation_not_latest_artifact_acquisition(stored):
    path, _, digest = stored
    db = Database(path)
    try:
        db.conn.execute("INSERT INTO house_archive_generations (archive_year, generation_id, promoted_at) VALUES (2024, 'new', '2099-01-01')")
        with pytest.raises(RuntimeError):
            ops._resolve_ingestion_generation(db.conn, '1', 2024, digest)
    finally:
        db.close()


def test_cached_response_is_parsed_once(stored, monkeypatch):
    cache_and_insert(stored)
    path, pdf, _ = stored
    original = common.parse_gemini_output
    calls = []
    def parse(*args, **kwargs):
        calls.append(kwargs.get('expected_page_count'))
        return original(*args, **kwargs)
    monkeypatch.setattr(common, 'parse_gemini_output', parse)
    assert common.inspect_cached_response('1', pdf, str(path.parent / 'gemini_cache')) is not None
    assert calls == [1]


def test_fresh_response_is_parsed_once(stored, monkeypatch):
    from types import SimpleNamespace
    path, pdf, _ = stored
    output = 'SCHEMA: 4\nMEMBER: Jane Doe\nPAGES: 1\nPAGE: 1\nNO_TRANSACTIONS'
    monkeypatch.setattr(common.subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=output, stderr=''))
    original = common.parse_gemini_output
    calls = []
    def parse(*args, **kwargs):
        calls.append(kwargs.get('expected_page_count'))
        return original(*args, **kwargs)
    monkeypatch.setattr(common, 'parse_gemini_output', parse)
    assert common.call_gemini(str(pdf), doc_id='1', cache_dir=str(path.parent / 'gemini_cache'))[0] == output
    assert calls == [1]


@pytest.mark.parametrize('left,right', [('John A Smith', 'John B Smith'), ('John Adam Smith', 'John Bruce Smith')])
def test_identity_mismatch_rejected(left, right):
    valid, rejected = common.validate_transactions(
        '1', left, [{'asset': 'Apple', 'type': 'Purchase',
        'date': '01/01/24', 'amount_letter': 'A'}], date(2024, 2, 1), right)
    assert valid == []
    assert rejected == {'member_mismatch': 1}


def test_parenthetical_asset_class_is_not_official_ticker():
    assert ops.extract_ticker('Private investment (LLC)') is None


@pytest.mark.parametrize('asset,expected', [
    ('Apple (AAPL)', ('AAPL', None, None, 'asset_description')),
    ('Holding (XYZ)', ('XYZ', None, None, 'asset_description')),
    ('Holding $XYZ', ('XYZ', None, None, 'asset_description')),
    ('Apple', (None, 'AAPL', 'AAPL', 'unverified')),
    ('Private investment (LLC)', (None, None, None, 'not_reported')),
    ('Holding Ticker: XYZ', ('XYZ', 'XYZ', None, 'official')),
    ('Holding (ABC) Ticker: XYZ', ('XYZ', 'XYZ', None, 'official')),
])
def test_parenthetical_ticker_retains_provenance(stored, asset, expected):
    path, _, digest = stored
    parsed = common.parse_gemini_output(
        f'MEMBER: Jane Doe\nPAGES: 1\nPAGE: 1\n{asset} | Purchase | 01/15/24 | N/A | A')
    rows, errors = ops._normalize_ocr_rows('1', parsed.member, parsed.transactions, date(2024, 2, 1),
                                         artifact_sha256=digest, ingestion_generation='g')
    assert not errors
    row, = rows
    assert tuple(row[key] for key in ('ticker', 'raw_ticker', 'ticker_candidate', 'ticker_origin')) == expected


@pytest.mark.parametrize('name,expected', [('John A. Smith', 'JOHN A SMITH'),
    ('John B. Smith', 'JOHN B SMITH'), ('J. D. Vance', 'JD VANCE'),
    ('John Smith CPC', 'JOHN SMITH'), ('John Smith M.D.', 'JOHN SMITH'),
    ('Honorable Neal P. Dunn MD, FACS', 'NEAL P DUNN'),
    ('Neal P. Dunn', 'NEAL P DUNN'), ('JD VANCE', 'JD VANCE')])
def test_initials(name, expected):
    assert canonical_member_key(name) == expected


def test_year_runner_parses_fresh_response_once(stored, monkeypatch):
    from types import SimpleNamespace

    path, pdf, _ = stored
    output = 'SCHEMA: 4\nMEMBER: Jane Doe\nPAGES: 1\nPAGE: 1\nNO_TRANSACTIONS'
    run = common.subprocess.run

    def provider(command, *args, **kwargs):
        if command[0] == 'llm':
            return SimpleNamespace(returncode=0, stdout=output, stderr='')
        return run(command, *args, **kwargs)  # Keep actual COW publication, not a fake cp.

    monkeypatch.setattr(common.subprocess, 'run', provider)
    original = common.parse_gemini_output
    calls = []
    def parse(*args, **kwargs):
        calls.append(kwargs.get('expected_page_count'))
        return original(*args, **kwargs)
    monkeypatch.setattr(common, 'parse_gemini_output', parse)
    monkeypatch.setattr(ops, 'parse_gemini_output', parse)
    summary = runner.run_gemini_ocr_for_year(2024, str(path.parent))
    assert summary['unresolved'] == 1  # Model empty output is not verified absence.
    assert summary['work']['by_state'] == {'needs_review': 1}
    assert calls == [1]


def test_timeout_retains_artifact_metadata(tmp_path, monkeypatch):
    pdf = tmp_path / 'a.pdf'
    pdf.write_bytes(b'pdf')
    monkeypatch.setattr(common, 'pdf_page_count', lambda _: 1)
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired('llm', 180)
    monkeypatch.setattr(common.subprocess, 'run', timeout)
    output, error, metadata = common.call_gemini(str(pdf))
    assert output is None
    assert metadata is not None
    assert metadata.sha256 == common.pdf_sha256(pdf)


def test_quota_retries_with_backoff(tmp_path, monkeypatch):
    from types import SimpleNamespace
    pdf = tmp_path / 'a.pdf'
    pdf.write_bytes(b'pdf')
    monkeypatch.setattr(common, 'pdf_page_count', lambda _: 1)
    calls, sleeps = [], []
    def run(*args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=1, stderr='429 RESOURCE_EXHAUSTED', stdout='')
    monkeypatch.setattr(common.subprocess, 'run', run)
    monkeypatch.setattr(common.time, 'sleep', sleeps.append)
    output, error, metadata = common.call_gemini(str(pdf))
    assert len(calls) == 3
    assert sleeps == [30, 60]
    assert output is None and metadata is not None


def test_quota_stops_year_batch(stored, monkeypatch):
    path, pdf, digest = stored
    calls = []
    def call(*args, **kwargs):
        calls.append(args)
        return None, '429 RESOURCE_EXHAUSTED', common.ArtifactMetadata(digest, 1, 3, str(pdf))
    monkeypatch.setattr(common, 'call_gemini', call)
    summary = runner.run_gemini_ocr_for_year(2024, str(path.parent))
    assert summary['work']['by_state'] == {'retry_wait': 1}
    assert summary['unresolved'] == 1
    assert len(calls) == 1


@pytest.mark.parametrize("reported_pages,metadata_pages", [(1, 1), (2, 1), (1, 2)])
def test_runner_admission_requires_bound_physical_page_coverage(stored, monkeypatch, reported_pages, metadata_pages):
    path, pdf, digest = stored
    output = make_ocr_output().replace('PAGES: 1', f'PAGES: {reported_pages}')
    parsed = common.parse_gemini_output(output, expected_page_count=1)
    assert parsed.page_count == 1  # Existing policy clamps unreported blank trailing pages.
    assert parsed.covered_pages == frozenset({1})
    metadata = common.ArtifactMetadata(digest, metadata_pages, len(pdf.read_bytes()), str(pdf), parsed)
    monkeypatch.setattr(common, 'call_gemini', lambda *args, **kwargs: (output, '', metadata))
    summary = runner.run_gemini_ocr_for_year(2024, str(path.parent))
    assert summary['unresolved'] == 1
    assert summary['work']['by_state'] == {'needs_review' if metadata_pages == 1 else 'blocked': 1}
    from analyzer.ocr.workers import load_result
    bundle = load_result(next((path.parent / 'pipeline' / 'results').glob('*.json')))
    assert bundle.evidence['page_count'] == 1
    if metadata_pages != 1:
        assert bundle.evidence['failure']['message'] == 'provider success lacks bound validated response metadata'


@pytest.mark.parametrize("failure", ["member", "page_binding", "artifact_binding"])
def test_year_runner_validation_failure_preserves_accepted_rows(stored, monkeypatch, failure):
    cache_and_insert(stored)
    from analyzer.ocr import workers
    recipe = workers.extraction_recipe
    monkeypatch.setattr(workers, 'extraction_recipe', lambda lane: recipe(lane) + ':fresh-fixture')
    path, pdf, digest = stored
    member = 'Other Person' if failure == 'member' else 'Jane Doe'
    output = make_ocr_output(member=member)
    metadata = common.ArtifactMetadata(
        'b' * 64 if failure == 'artifact_binding' else digest,
        2 if failure == 'page_binding' else 1, len(pdf.read_bytes()), str(pdf),
        common.parse_gemini_output(output, expected_page_count=1),
    )
    monkeypatch.setattr(common, 'call_gemini', lambda *args, **kwargs: (output, '', metadata))
    summary = runner.run_gemini_ocr_for_year(2024, str(path.parent), refresh=True)
    assert summary['work']['by_state'] == {'accepted': 1, 'needs_review' if failure == 'member' else 'blocked': 1}
    db = Database(path)
    try:
        assert db.conn.execute('SELECT COUNT(*) FROM transactions').fetchone()[0] == 1
    finally:
        db.close()


def test_fresh_legacy_provider_response_requires_review_without_upgrading_published_rows(stored, monkeypatch):
    from analyzer.ocr.normalization import normalize_candidate
    from analyzer.ocr.workers import load_result

    cache_and_insert(stored)  # An old trusted materialization remains valid history.
    from analyzer.ocr import workers
    recipe = workers.extraction_recipe
    monkeypatch.setattr(workers, 'extraction_recipe', lambda lane: recipe(lane) + ':fresh-fixture')
    path, pdf, digest = stored
    with Database(path) as db:
        original = db.conn.execute("SELECT * FROM transactions").fetchall()
    legacy = 'MEMBER: Jane Doe\nPAGES: 1\nPAGE: 1\nApple (AAPL) | Purchase | 01/15/24 | N/A | A'
    parsed = common.parse_gemini_output(legacy, expected_page_count=1)
    calls = []

    def provider(*args, **kwargs):
        calls.append(kwargs["doc_id"])
        return legacy, '', common.ArtifactMetadata(digest, 1, len(pdf.read_bytes()), str(pdf), parsed)

    monkeypatch.setattr(common, 'call_gemini', provider)
    for index in range(2):
        summary = runner.run_gemini_ocr_for_year(2024, str(path.parent), refresh=True)
        assert summary['completed_attempts'] == (1 if index == 0 else 0)
        assert summary['work']['by_state'] == {'accepted': 1, 'needs_review': 1}
        assert summary['unresolved'] == 0  # The earlier accepted materialization is not erased.
    assert calls == ['1']
    bundle, = [item for item in (load_result(p) for p in (path.parent / 'pipeline' / 'results').glob('*.json'))
               if item.raw_response == legacy]
    normalized = normalize_candidate(bundle.work, bundle)
    assert normalized['legacy_limited']
    assert normalized['findings']['obsolete_output_schema'] == 1
    assert normalized['status'] == 'needs_review'
    assert bundle.raw_response == legacy
    assert bundle.rows == parsed.transactions
    with Database(path) as db:
        assert db.conn.execute("SELECT * FROM transactions").fetchall() == original
        assert db.conn.execute("SELECT state FROM extraction_results ORDER BY state").fetchall() == [('accepted',), ('needs_review',)]


@pytest.mark.parametrize(
    ("outcome_status", "inserted", "expected", "exit_status"),
    [
        ("success", 1, 1, 0),
        ("partial", 1, 2, 1),
        ("rejected", 0, 1, 1),
        ("error", 0, 1, 1),
    ],
)
def test_operational_drain_exit_status_tracks_admission_outcome(
    monkeypatch, capsys, outcome_status, inserted, expected, exit_status
):
    monkeypatch.setattr(sys, "argv", ["ocr_zero_rows"])
    monkeypatch.setattr(runner, "_ocr_engine_mode", lambda: "gemini")
    from analyzer.ocr import coordinator
    summary = {"unresolved": 0 if outcome_status == "success" else 1,
               "missing_years": [], "completed_attempts": expected,
               "accepted_rows": inserted}
    calls = []

    def run(*args, **kwargs):
        calls.append(kwargs)
        return summary

    monkeypatch.setattr(coordinator, "run_pipeline", run)

    assert runner.main() == exit_status
    assert json.loads(capsys.readouterr().out) == summary
    assert calls == [{"local_workers": 0, "gemini_workers": 1,
                      "glm_workers": 0, "refresh_recipe": False}]


def test_operational_drain_no_available_work_is_success(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["ocr_zero_rows"])
    monkeypatch.setattr(runner, "_ocr_engine_mode", lambda: "gemini")
    from analyzer.ocr import coordinator
    summary = {"unresolved": 0, "missing_years": [], "completed_attempts": 0}
    monkeypatch.setattr(coordinator, "run_pipeline", lambda *args, **kwargs: summary)

    assert runner.main() == 0
    assert json.loads(capsys.readouterr().out) == summary


@pytest.mark.parametrize("exit_status", [0, 1])
def test_operational_script_maps_main_status_to_process_exit(
    monkeypatch, exit_status
):
    from pathlib import Path

    monkeypatch.setattr(runner, "main", lambda **kwargs: exit_status)
    script = Path(__file__).resolve().parents[1] / "scripts" / "ocr_zero_rows.py"

    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(script), run_name="__main__")

    assert exc.value.code == exit_status


def test_insert_rejects_unbound_generation(stored):
    path, _, digest = stored
    assert ops.insert_transactions('1', 2024, 'Jane Doe', [], db_path=str(path), artifact_sha256=digest, ingestion_generation='invented') == 0
    with Database(path) as db:
        with pytest.raises(RuntimeError, match='generation'):
            ops._resolve_ingestion_generation(db.conn, '1', 2024, 'unbound-artifact')


def test_insertion_uses_bound_metadata(stored):
    path, _, digest = stored
    db = Database(path)
    db.conn.execute("UPDATE metadata SET first_name = 'Other', last_name = 'Person'")
    db.close()
    cache_and_insert(stored)
    db = Database(path)
    try:
        assert db.conn.execute('SELECT member FROM transactions').fetchall() == [('Jane Doe',)]
    finally:
        db.close()


def test_empty_mismatched_report_is_not_no_transactions(stored):
    path, _, digest = stored
    ops.insert_transactions('1', 2024, 'Wrong Person', [], db_path=str(path), artifact_sha256=digest, ingestion_generation='g')
    db = Database(path)
    try:
        assert db.conn.execute("SELECT status FROM pdf_parse_runs WHERE parser_version = ?", [common.GEMINI_PARSER_VERSION]).fetchone() is None
        assert db.conn.execute("SELECT COUNT(*) FROM transactions").fetchone() == (0,)
    finally:
        db.close()
