import subprocess
from datetime import date

import pytest

from analyzer.member_names import canonical_member_key
from scripts import gemini_ocr_common as common
from scripts import ocr_zero_rows as ops
from analyzer.database import Database


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
    db.conn.execute("INSERT INTO house_archive_generations (archive_year, generation_id) VALUES (2024, 'g')")
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


def test_missing_notification_is_resolved(stored):
    cache_and_insert(stored, 'N/A')
    path, _, _ = stored
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
    monkeypatch.setattr(common.subprocess, 'run', lambda *a, **kw: SimpleNamespace(returncode=0, stdout=output, stderr=''))
    monkeypatch.setattr(ops, 'get_ocr_work_items', lambda **kw: [('1', 2024, str(pdf))])
    monkeypatch.setattr(ops.time, 'sleep', lambda _: None)
    original = common.parse_gemini_output
    calls = []
    def parse(*args, **kwargs):
        calls.append(kwargs.get('expected_page_count'))
        return original(*args, **kwargs)
    monkeypatch.setattr(common, 'parse_gemini_output', parse)
    monkeypatch.setattr(ops, 'parse_gemini_output', parse)
    assert ops.run_gemini_ocr_for_year(2024, str(path.parent)) == 0
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
    monkeypatch.setattr(ops.time, 'sleep', sleeps.append)
    output, error, metadata = common.call_gemini(str(pdf))
    assert len(calls) == 3
    assert sleeps == [30, 60]
    assert output is None and metadata is not None


def test_quota_stops_year_batch(stored, monkeypatch):
    path, pdf, digest = stored
    calls = []
    monkeypatch.setattr(ops, 'get_ocr_work_items', lambda **kwargs: [('1', 2024, str(pdf)), ('2', 2024, str(pdf))])
    monkeypatch.setattr(ops.time, 'sleep', lambda _: None)
    def call(*args, **kwargs):
        calls.append(args)
        return None, '429 RESOURCE_EXHAUSTED', common.ArtifactMetadata(digest, 1, 3, str(pdf))
    monkeypatch.setattr(ops, 'call_gemini', call)
    ops.run_gemini_ocr_for_year(2024, str(path.parent))
    assert len(calls) == 1


def test_runner_keeps_physical_page_count(stored, monkeypatch):
    path, pdf, digest = stored
    output = 'MEMBER: Jane Doe\nPAGES: 2\nPAGE: 1\nApple (AAPL) | Purchase | 01/15/24 | N/A | A'
    common.parse_gemini_output(output, expected_page_count=1)
    monkeypatch.setattr(ops, 'get_ocr_work_items', lambda **kwargs: [('1', 2024, str(pdf))])
    monkeypatch.setattr(ops.time, 'sleep', lambda _: None)
    monkeypatch.setattr(ops, 'call_gemini', lambda *args, **kwargs: (output, '', common.ArtifactMetadata(digest, 1, 3, str(pdf), common.parse_gemini_output(output, expected_page_count=1))))
    assert ops.run_gemini_ocr_for_year(2024, str(path.parent)) == 1


def test_year_runner_validation_failure_retires_rows(stored, monkeypatch):
    cache_and_insert(stored)
    path, pdf, digest = stored
    output = 'MEMBER: Other Person\nPAGES: 1\nPAGE: 1\nApple (AAPL) | Purchase | 01/15/24 | N/A | A'
    monkeypatch.setattr(ops, 'get_ocr_work_items', lambda **kwargs: [('1', 2024, str(pdf))])
    monkeypatch.setattr(ops.time, 'sleep', lambda _: None)
    monkeypatch.setattr(ops, 'call_gemini', lambda *args, **kwargs: (output, '', common.ArtifactMetadata(digest, 1, 3, str(pdf), common.parse_gemini_output(output, expected_page_count=1))))
    assert ops.run_gemini_ocr_for_year(2024, str(path.parent)) == 0
    db = Database(path)
    try:
        assert db.conn.execute('SELECT COUNT(*) FROM transactions').fetchone()[0] == 0
    finally:
        db.close()


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
