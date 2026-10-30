import json
import runpy
import subprocess
import sys
from datetime import date

import pytest

from analyzer.member_names import canonical_member_key
from analyzer.ocr import common as common
from analyzer.ocr import ingestion as ops
from analyzer.ocr import runner
from analyzer.persistence.database import Database


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
    pdf.parent.mkdir(parents=True)
    pdf.write_bytes(b'pdf')
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
    path, pdf, digest = stored
    output = f'MEMBER: Jane Doe\nPAGES: 1\nPAGE: 1\nApple (AAPL) | Purchase | 01/15/24 | {notification} | A'
    common.write_cached_response('1', pdf, output, str(path.parent / 'gemini_cache'))
    parsed = common.parse_gemini_output(output)
    ops.insert_transactions('1', 2024, parsed.member, parsed.transactions, db_path=str(path), artifact_sha256=digest, ingestion_generation='g')


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
    assert ops.get_ocr_work_items(db_path=str(path), data_dir=path.parent) == []


@pytest.mark.parametrize(
    ("instrument", "strike"), [("unknown", None), ("call", "100")],
)
def test_lossless_cache_matches_persisted_instrument_projection(stored, instrument, strike):
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
    assert ops.insert_transactions(
        '1', 2024, parsed.member, parsed.transactions, db_path=str(path),
        artifact_sha256=digest, ingestion_generation='g',
    ) == 1
    with Database(path) as db:
        if instrument == "unknown":
            # Existing canonical rows represent unknown instruments as SQL NULL.
            db.conn.execute("UPDATE transactions SET instrument_type = NULL WHERE doc_id='1'")
        persisted = db.conn.execute(
            "SELECT instrument_type, strike_price, amount_raw, raw_owner FROM transactions WHERE doc_id='1'"
        ).fetchone()
    assert persisted == (None if instrument == "unknown" else instrument,
                         None if strike is None else 100.0, "$12,345.67", "SP")
    # SQL's unknown -> NULL and DOUBLE coercion must not trigger OCR redispatch.
    assert ops.get_ocr_work_items(db_path=str(path), data_dir=path.parent) == []


def test_null_provenance_is_stale(stored):
    cache_and_insert(stored)
    path, _, _ = stored
    db = Database(path)
    db.conn.execute("INSERT INTO transactions (doc_id, source, chamber, asset_description) VALUES ('1', 'gemini_ocr', 'House', 'stale')")
    db.close()
    assert len(ops.get_ocr_work_items(db_path=str(path), data_dir=path.parent)) == 1


def test_semantic_zero_retires_exact_artifact(stored):
    cache_and_insert(stored)
    path, _, digest = stored
    ops.insert_transactions('1', 2024, 'Jane Doe', [], raw_count=1, db_path=str(path), artifact_sha256=digest, ingestion_generation='g')
    db = Database(path)
    try:
        assert db.conn.execute('SELECT COUNT(*) FROM transactions').fetchone()[0] == 0
    finally:
        db.close()


@pytest.mark.parametrize("insert_fails", [True, False])
def test_insertion_preserves_primary_failure_and_always_closes(stored, monkeypatch, insert_fails):
    path, _, digest = stored
    connect = ops.duckdb.connect
    closed = []
    drops = []

    class FaultyConnection:
        def __init__(self):
            self.conn = connect(str(path))

        def execute(self, sql, *args):
            if sql.startswith("INSERT INTO staging_gemini_ocr_transactions") and insert_fails:
                raise ValueError("primary insert failure")
            if sql.startswith("DROP TABLE IF EXISTS staging_gemini_ocr_transactions"):
                drops.append(sql)
                if len(drops) > 1:
                    raise RuntimeError("drop failure")
            return self.conn.execute(sql, *args)

        def close(self):
            closed.append(True)
            self.conn.close()
            raise RuntimeError("close failure")

    def failed_record(*args, **kwargs):
        raise RuntimeError("record failure")

    monkeypatch.setattr(ops.duckdb, "connect", lambda _: FaultyConnection())
    monkeypatch.setattr(ops, "record_failed_ocr_attempt", failed_record)
    parsed = common.parse_gemini_output(
        "MEMBER: Jane Doe\nPAGES: 1\nPAGE: 1\n"
        "Apple (AAPL) | Purchase | 01/15/24 | 01/20/24 | A"
    )
    error, message = (ValueError, "primary insert failure") if insert_fails else (RuntimeError, "drop failure")
    with pytest.raises(error, match=message):
        ops.insert_transactions_outcome(
            "1", 2024, parsed.member, parsed.transactions,
            db_path=str(path), artifact_sha256=digest, ingestion_generation="g",
        )
    assert closed == [True]


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

    assert outcome.status == "partial"
    assert outcome.inserted == 0 and outcome.expected == 2
    assert outcome.committed is False and outcome.complete is False
    assert outcome.rejections == {"date_out_of_window": 1}
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
        assert run[0:2] == ("partial", 0)
        assert "canonical commit blocked" in run[2]
        retained = db.conn.execute(
            "SELECT transactions_json::VARCHAR, rejections_json::VARCHAR "
            "FROM ocr_partial_results WHERE doc_id='1'"
        ).fetchone()
        assert "Microsoft (MSFT)" in retained[0] and "Bad date row" in retained[0]
        assert json.loads(retained[1]) == {"date_out_of_window": 1}
    finally:
        db.close()

    cached = common.inspect_cached_response("1", str(pdf), cache_dir)
    assert cached is not None and len(cached.parsed.transactions) == 2
    assert ops.get_ocr_work_items(db_path=str(path), data_dir=path.parent) == [
        ("1", 2024, str(pdf))
    ]


def test_partial_canonical_override_is_loud_and_still_incomplete(stored, capsys):
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
    assert "WARNING: --force-partial-canonical" in capsys.readouterr().out
    assert outcome.status == "partial" and outcome.committed
    assert not outcome.complete
    db = Database(path)
    try:
        assert db.conn.execute(
            "SELECT ticker FROM transactions WHERE doc_id='1' AND source='gemini_ocr'"
        ).fetchall() == [("MSFT",)]
        assert db.conn.execute(
            "SELECT status, error_message FROM pdf_parse_runs WHERE doc_id='1' "
            "AND parser_version=?",
            [ops.GEMINI_PARSER_VERSION],
        ).fetchone()[0] == "success"
        assert "FORCED PARTIAL CANONICAL COMMIT" in db.conn.execute(
            "SELECT error_message FROM pdf_parse_runs WHERE doc_id='1' "
            "AND parser_version=?",
            [ops.GEMINI_PARSER_VERSION],
        ).fetchone()[0]
        assert db.conn.execute(
            "SELECT COUNT(*) FROM ocr_partial_results WHERE doc_id='1'"
        ).fetchone()[0] == 1
    finally:
        db.close()


@pytest.mark.parametrize("mutation", ["replacement", "no_txs", "error"])
def test_ocr_mutations_preserve_accepted_historical_rows(stored, mutation):
    cache_and_insert(stored)
    path, _, digest = stored
    db = Database(path)
    db.conn.execute("UPDATE house_archive_generations SET promoted_at='2024-02-01'")
    db.conn.execute("UPDATE transactions SET created_at='2024-02-02'")
    db.conn.execute("UPDATE pdf_parse_runs SET parsed_at='2024-02-03'")
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
        assert db.conn.execute("SELECT COUNT(*) FROM transaction_versions WHERE created_at='2024-02-02'").fetchone()[0] == 1
        assert db.conn.execute("SELECT COUNT(*) FROM parse_run_versions").fetchone()[0] >= 1
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
    output = 'MEMBER: Jane Doe\nPAGES: 1\nPAGE: 1\nNO_TRANSACTIONS'
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
    assert ops.insert_transactions('1', 2024, parsed.member, parsed.transactions,
        db_path=str(path), artifact_sha256=digest, ingestion_generation='g') == 1
    db = Database(path)
    try:
        assert db.conn.execute('SELECT ticker, raw_ticker, ticker_candidate, ticker_origin FROM transactions').fetchone() == expected
    finally:
        db.close()


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
    output = 'MEMBER: Jane Doe\nPAGES: 1\nPAGE: 1\nNO_TRANSACTIONS'
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
    output = f'MEMBER: Jane Doe\nPAGES: {reported_pages}\nPAGE: 1\nApple (AAPL) | Purchase | 01/15/24 | N/A | A'
    parsed = common.parse_gemini_output(output, expected_page_count=1)
    assert parsed.page_count == 1  # Existing policy clamps unreported blank trailing pages.
    assert parsed.covered_pages == frozenset({1})
    metadata = common.ArtifactMetadata(digest, metadata_pages, len(pdf.read_bytes()), str(pdf), parsed)
    monkeypatch.setattr(common, 'call_gemini', lambda *args, **kwargs: (output, '', metadata))
    summary = runner.run_gemini_ocr_for_year(2024, str(path.parent))
    assert summary['unresolved'] == (0 if metadata_pages == 1 else 1)
    assert summary['work']['by_state'] == {'accepted' if metadata_pages == 1 else 'blocked': 1}
    from analyzer.ocr.workers import load_result
    bundle = load_result(next((path.parent / 'pipeline' / 'results').glob('*.json')))
    assert bundle.evidence['page_count'] == 1
    if metadata_pages != 1:
        assert bundle.evidence['failure']['message'] == 'provider success lacks bound validated response metadata'


@pytest.mark.parametrize("failure", ["member", "page_binding", "artifact_binding"])
def test_year_runner_validation_failure_preserves_accepted_rows(stored, monkeypatch, failure):
    cache_and_insert(stored)
    path, pdf, digest = stored
    member = 'Other Person' if failure == 'member' else 'Jane Doe'
    output = f'MEMBER: {member}\nPAGES: 1\nPAGE: 1\nApple (AAPL) | Purchase | 01/15/24 | N/A | A'
    metadata = common.ArtifactMetadata(
        'b' * 64 if failure == 'artifact_binding' else digest,
        2 if failure == 'page_binding' else 1, len(pdf.read_bytes()), str(pdf),
        common.parse_gemini_output(output, expected_page_count=1),
    )
    monkeypatch.setattr(common, 'call_gemini', lambda *args, **kwargs: (output, '', metadata))
    summary = runner.run_gemini_ocr_for_year(2024, str(path.parent), refresh=True)
    assert summary['work']['by_state'] == {'needs_review' if failure == 'member' else 'blocked': 1}
    db = Database(path)
    try:
        assert db.conn.execute('SELECT COUNT(*) FROM transactions').fetchone()[0] == 1
    finally:
        db.close()


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
    with pytest.raises(RuntimeError, match='generation'):
        ops.insert_transactions('1', 2024, 'Jane Doe', [], db_path=str(path), artifact_sha256=digest, ingestion_generation='invented')


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
        assert db.conn.execute("SELECT status FROM pdf_parse_runs WHERE parser_version = ?", [common.GEMINI_PARSER_VERSION]).fetchone()[0] == 'error'
    finally:
        db.close()
