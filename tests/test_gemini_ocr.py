from datetime import datetime
import json
import queue

import pytest
import duckdb

from analyzer.database import Database
from scripts import gemini_ocr_common
from scripts.ocr_zero_rows import insert_transactions


OCR_SCHEMA_COLUMNS = {
    "chamber": "VARCHAR",
    "source_record_id": "VARCHAR",
    "source_row_id": "VARCHAR",
    "official_filing_date": "DATE",
    "available_date": "DATE",
    "notification_date": "DATE",
    "amends_source_record_id": "VARCHAR",
    "raw_transaction_subtype": "VARCHAR",
    "ticker_origin": "VARCHAR",
    "raw_asset_class": "VARCHAR",
    "raw_asset_description": "VARCHAR",
    "ingestion_generation": "VARCHAR",
    "artifact_sha256": "VARCHAR",
}


def _enable_ocr_schema(connection):
    for column, column_type in OCR_SCHEMA_COLUMNS.items():
        connection.execute(
            f"ALTER TABLE transactions ADD COLUMN IF NOT EXISTS {column} {column_type}"
        )
    connection.execute("DROP INDEX IF EXISTS idx_tx_unique_v2")
    connection.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_tx_unique_source_row "
        "ON transactions(source_record_id, source_row_id)"
    )


def _insert_transactions(*args, **kwargs):
    kwargs.setdefault("artifact_sha256", "test-artifact-sha256")
    return insert_transactions(*args, **kwargs)


@pytest.fixture(autouse=True)
def _single_page_pdfinfo(monkeypatch):
    monkeypatch.setattr(gemini_ocr_common, "pdf_page_count", lambda unused: 1)


def _tx(asset="Apple Inc. (AAPL)", date="01/15/24", tx_type="Purchase", amount="A"):
    return {
        "asset": asset,
        "type": tx_type,
        "date": date,
        "notif_date": "01/20/24",
        "amount_letter": amount,
        "amount_midpoint": 8000,
    }


def test_validation_has_no_fixed_row_cap_for_large_real_filings():
    txs = [_tx(asset=f"Asset {index}") for index in range(722)]

    valid, rejections = gemini_ocr_common.validate_transactions(
        "doc-large", "Jane Doe", txs, datetime(2024, 1, 20), "Jane Doe"
    )

    assert len(valid) == 722
    assert "row_count_exceeds_cap" not in rejections


def test_validation_date_window_drops_bad_rows():
    txs = [_tx(date="01/15/24"), _tx(date="01/01/22"), _tx(date="02/10/24")]

    valid, rejections = gemini_ocr_common.validate_transactions(
        "doc-date", "Jane Doe", txs, datetime(2024, 1, 20), "Jane Doe"
    )

    assert [tx["date"] for tx in valid] == ["01/15/24"]
    assert rejections["date_out_of_window"] == 2


def test_validation_preserves_repeated_lots_without_source_identity():
    txs = [_tx(), _tx(), _tx(asset="Microsoft Corp. (MSFT)")]

    valid, rejections = gemini_ocr_common.validate_transactions(
        "doc-dupe", "Jane Doe", txs, datetime(2024, 1, 20), "Jane Doe"
    )

    assert len(valid) == 3
    assert "duplicate_collapsed" not in rejections


def test_validation_member_mismatch_uses_metadata_name():
    valid, rejections = gemini_ocr_common.validate_transactions(
        "doc-member", "Injected Junk", [_tx()], datetime(2024, 1, 20), "Jane Q. Doe"
    )

    assert valid[0]["member"] == "Jane Q. Doe"
    assert rejections["member_mismatch"] == 1


def test_call_gemini_empty_stdout_not_cached(monkeypatch, tmp_path):
    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"%PDF-canary")

    def fake_run(args, capture_output, text, timeout):
        class Result:
            returncode = 0
            stdout = "  \n\t"
            stderr = ""

        return Result()

    monkeypatch.setattr(gemini_ocr_common.subprocess, "run", fake_run)
    output, error = gemini_ocr_common.call_gemini(
        str(pdf), doc_id="doc-empty", cache_dir=str(tmp_path)
    )

    assert output is None
    assert error == "invalid_response: empty_response"
    assert not (tmp_path / "doc-empty.json").exists()


def test_call_gemini_ignores_partial_cache_file(monkeypatch, tmp_path):
    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"%PDF-canary")
    (tmp_path / "doc-partial.json").write_text('{"cache_envelope_version":')
    calls = []

    def fake_run(args, capture_output, text, timeout):
        calls.append(args)

        class Result:
            returncode = 0
            stdout = "MEMBER: Jane Doe\nPAGES: 1\nPAGE: 1\nApple Inc. (AAPL) | Purchase | 01/15/24 | 01/20/24 | A\n"
            stderr = ""

        return Result()

    monkeypatch.setattr(gemini_ocr_common.subprocess, "run", fake_run)
    output, error = gemini_ocr_common.call_gemini(
        str(pdf), doc_id="doc-partial", cache_dir=str(tmp_path)
    )

    assert output is not None
    assert error == ""
    assert len(calls) == 1
    envelope = json.loads((tmp_path / "doc-partial.json").read_text())
    assert envelope["pdf_sha256"] == gemini_ocr_common.pdf_sha256(pdf)
    assert envelope["prompt_sha256"] == gemini_ocr_common.PROMPT_SHA256
    assert envelope["model"] == gemini_ocr_common.MODEL
    assert envelope["parser_version"] == gemini_ocr_common.GEMINI_PARSER_VERSION


def test_cache_is_invalidated_when_pdf_changes(monkeypatch, tmp_path):
    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"%PDF-first")
    outputs = iter(
        [
            "MEMBER: Jane Doe\nPAGES: 1\nPAGE: 1\nApple Inc. (AAPL) | Purchase | 01/15/24 | 01/20/24 | A\n",
            "MEMBER: Jane Doe\nPAGES: 1\nPAGE: 1\nMicrosoft (MSFT) | Sale | 01/16/24 | 01/21/24 | B\n",
        ]
    )
    calls = []

    def fake_run(args, capture_output, text, timeout):
        calls.append(args)

        class Result:
            returncode = 0
            stdout = next(outputs)
            stderr = ""

        return Result()

    monkeypatch.setattr(gemini_ocr_common.subprocess, "run", fake_run)
    first, _ = gemini_ocr_common.call_gemini(
        str(pdf), doc_id="doc-hash", cache_dir=str(tmp_path)
    )
    cached, _ = gemini_ocr_common.call_gemini(
        str(pdf), doc_id="doc-hash", cache_dir=str(tmp_path)
    )
    pdf.write_bytes(b"%PDF-second")
    changed, _ = gemini_ocr_common.call_gemini(
        str(pdf), doc_id="doc-hash", cache_dir=str(tmp_path)
    )

    assert first == cached
    assert "Microsoft" in changed
    assert len(calls) == 2


def test_call_uses_one_immutable_pdf_snapshot(monkeypatch, tmp_path):
    pdf = tmp_path / "changing.pdf"
    original = b"%PDF-original-bytes"
    pdf.write_bytes(original)
    attached = []

    def fake_run(args, capture_output, text, timeout):
        attachment = args[args.index("-a") + 1]
        attached.append(gemini_ocr_common.Path(attachment).read_bytes())
        pdf.write_bytes(b"%PDF-mutated-after-snapshot")

        class Result:
            returncode = 0
            stdout = (
                "MEMBER: Jane Doe\nPAGES: 1\nPAGE: 1\n"
                "Apple (AAPL) | Purchase | 01/15/24 | 01/20/24 | A"
            )
            stderr = ""

        return Result()

    monkeypatch.setattr(gemini_ocr_common.subprocess, "run", fake_run)
    output, error = gemini_ocr_common.call_gemini(
        str(pdf), doc_id="snapshot", cache_dir=str(tmp_path)
    )
    envelope = json.loads((tmp_path / "snapshot.json").read_text())
    assert output is not None and error == ""
    assert attached == [original]
    assert (
        envelope["pdf_sha256"] == gemini_ocr_common.hashlib.sha256(original).hexdigest()
    )


def test_cache_path_sanitizes_doc_id(tmp_path):
    path = gemini_ocr_common.cache_path("folder/doc\\id", cache_dir=str(tmp_path))
    assert path == tmp_path / "folder_doc_id.json"


@pytest.mark.parametrize(
    "bad_output",
    [
        "MEMBER: Jane Doe\nPAGES: 1\nPAGE: 1\n | Purchase | 01/15/24 | 01/20/24 | A",
        "MEMBER: Jane Doe\nPAGES: 1\nPAGE: 1\nApple | Invalid | 01/15/24 | 01/20/24 | A",
        "MEMBER: Jane Doe\nPAGES: 1\nPAGE: 1\nApple | Purchase | bad | 01/20/24 | A",
        "MEMBER: Jane Doe\nPAGES: 1\nPAGE: 1\nApple | Purchase | 01/15/24 | bad | A",
        "MEMBER: Jane Doe\nPAGES: 1\nPAGE: 1\nApple | Purchase | 01/15/24 | 01/20/24 | ABRACADABRA",
        "MEMBER: \nNO_TRANSACTIONS",
        "MEMBER: Jane Doe\nPAGES: 1\nPAGE: 1\nApple | Purchase | 01/15/24 | 01/20/24",
    ],
)
def test_schema_rejects_malformed_outputs(bad_output):
    with pytest.raises(gemini_ocr_common.GeminiOutputError):
        gemini_ocr_common.parse_gemini_output(bad_output)


def test_schema_skips_prompt_example_consistently():
    parsed = gemini_ocr_common.parse_gemini_output(
        "MEMBER: Jane Doe\nPAGES: 1\nPAGE: 1\n"
        "Example: Mega Corp. Common Stock | Purchase | 01/01/24 | 01/02/24 | A\n"
        "Apple (AAPL) | Purchase | 01/15/24 | 01/20/24 | B"
    )
    assert len(parsed.transactions) == 1
    assert parsed.transactions[0]["asset"] == "Apple (AAPL)"


def test_example_filter_matches_only_exact_prompt_exemplar():
    parsed = gemini_ocr_common.parse_gemini_output(
        "MEMBER: Jane Doe\nPAGES: 1\nPAGE: 1\n"
        "Mega Corp Growth Fund | Purchase | 01/15/24 | 01/20/24 | A"
    )
    assert [tx["asset"] for tx in parsed.transactions] == ["Mega Corp Growth Fund"]


def test_schema_requires_outcome_for_every_pdf_page():
    output = (
        "MEMBER: Jane Doe\nPAGES: 2\nPAGE: 1\n"
        "Apple (AAPL) | Purchase | 01/15/24 | 01/20/24 | A"
    )
    with pytest.raises(
        gemini_ocr_common.GeminiOutputError, match="missing page outcomes"
    ):
        gemini_ocr_common.parse_gemini_output(output, expected_page_count=2)


def test_production_validation_has_no_document_id_gates():
    assert not hasattr(gemini_ocr_common, "KNOWN_DOCUMENT_CANARIES")
    assert not hasattr(gemini_ocr_common, "validate_known_document")


def test_insert_fails_clearly_before_provenance_schema_exists(tmp_path):
    db_path = tmp_path / "legacy.duckdb"
    db = Database(db_path)
    db.conn.execute(
        "INSERT INTO metadata VALUES ('legacy', 'Jane', 'Doe', TIMESTAMP '2024-01-20', 'P', CURRENT_TIMESTAMP)"
    )
    db.close()
    with pytest.raises(RuntimeError, match="requires the provenance schema"):
        _insert_transactions("legacy", 2024, "Jane Doe", [_tx()], db_path=str(db_path))
    from scripts.ocr_zero_rows import get_ocr_work_items

    with pytest.raises(RuntimeError, match="requires the provenance schema"):
        get_ocr_work_items(db_path=str(db_path), data_dir=tmp_path)


def test_insert_transactions_sets_source_and_is_idempotent(tmp_path):
    db_path = tmp_path / "congress.duckdb"
    db = Database(db_path)
    _enable_ocr_schema(db.conn)
    db.conn.execute("""
        INSERT INTO metadata (doc_id, first_name, last_name, filing_date, filing_type, fetched_at)
        VALUES ('doc-insert', 'Jane', 'Doe', TIMESTAMP '2024-01-20', 'P', CURRENT_TIMESTAMP)
    """)
    db.conn.close()

    txs = [_tx()]
    assert (
        _insert_transactions("doc-insert", 2024, "Jane Doe", txs, db_path=str(db_path))
        == 1
    )
    assert (
        _insert_transactions("doc-insert", 2024, "Jane Doe", txs, db_path=str(db_path))
        == 1
    )

    con = Database(db_path).conn
    rows = con.execute("""
        SELECT member, ticker, asset_description, source
        FROM transactions
        WHERE doc_id = 'doc-insert'
    """).fetchall()
    parse_runs = con.execute("""
        SELECT parser_version, status, raw_row_count, transaction_count
        FROM pdf_parse_runs
        WHERE doc_id = 'doc-insert'
        ORDER BY parsed_at
    """).fetchall()
    con.close()

    assert rows == [("Jane Doe", "AAPL", "Apple Inc. (AAPL)", "gemini_ocr")]
    assert len(parse_runs) == 2
    assert parse_runs[-1] == (gemini_ocr_common.GEMINI_PARSER_VERSION, "success", 1, 1)


def test_insert_preserves_repeated_lots_with_distinct_source_rows(tmp_path):
    db_path = tmp_path / "lots.duckdb"
    db = Database(db_path)
    _enable_ocr_schema(db.conn)
    db.conn.execute(
        "INSERT INTO metadata VALUES ('lots', 'Jane', 'Doe', TIMESTAMP '2024-01-20', 'P', CURRENT_TIMESTAMP)"
    )
    db.close()
    assert (
        _insert_transactions(
            "lots", 2024, "Jane Doe", [_tx(), _tx()], db_path=str(db_path)
        )
        == 2
    )
    con = duckdb.connect(str(db_path))
    rows = con.execute(
        "SELECT source_record_id, source_row_id FROM transactions WHERE doc_id='lots' ORDER BY source_row_id"
    ).fetchall()
    con.close()
    assert rows == [("lots", "lots:row:1"), ("lots", "lots:row:2")]


def test_insert_transactions_plumbs_authoritative_provenance_when_schema_supports_it(
    tmp_path,
):
    db_path = tmp_path / "congress.duckdb"
    db = Database(db_path)
    _enable_ocr_schema(db.conn)
    db.conn.execute(
        "INSERT INTO metadata VALUES ('doc-provenance', 'Jane', 'Doe', TIMESTAMP '2024-01-20', 'P', CURRENT_TIMESTAMP)"
    )
    db.close()

    inserted = _insert_transactions(
        "doc-provenance",
        2024,
        "Jane Doe",
        [_tx()],
        db_path=str(db_path),
        artifact_sha256="abc123",
    )
    assert inserted == 1
    connection = Database(db_path).conn
    row = connection.execute(
        "SELECT chamber, source_record_id, source_row_id, official_filing_date, notification_date, raw_asset_description, ingestion_generation, artifact_sha256 FROM transactions WHERE doc_id='doc-provenance'"
    ).fetchone()
    connection.close()
    assert row == (
        "House",
        "doc-provenance",
        "doc-provenance:row:1",
        datetime(2024, 1, 20).date(),
        datetime(2024, 1, 20).date(),
        "Apple Inc. (AAPL)",
        gemini_ocr_common.GEMINI_PARSER_VERSION,
        "abc123",
    )


def test_insert_transactions_empty_list_preserves_existing_rows(tmp_path):
    db_path = tmp_path / "congress.duckdb"
    db = Database(db_path)
    _enable_ocr_schema(db.conn)
    db.conn.execute("""
        INSERT INTO metadata (doc_id, first_name, last_name, filing_date, filing_type, fetched_at)
        VALUES ('doc-preserve', 'Jane', 'Doe', TIMESTAMP '2024-01-20', 'P', CURRENT_TIMESTAMP)
    """)
    db.conn.close()

    assert (
        _insert_transactions(
            "doc-preserve", 2024, "Jane Doe", [_tx()], db_path=str(db_path)
        )
        == 1
    )
    assert (
        _insert_transactions("doc-preserve", 2024, "Jane Doe", [], db_path=str(db_path))
        == 0
    )

    con = Database(db_path).conn
    row_count = con.execute(
        "SELECT COUNT(*) FROM transactions WHERE doc_id = 'doc-preserve'"
    ).fetchone()[0]
    latest_run = con.execute("""
        SELECT status, raw_row_count, transaction_count
        FROM pdf_parse_runs
        WHERE doc_id = 'doc-preserve'
        ORDER BY parsed_at DESC
        LIMIT 1
    """).fetchone()
    con.close()

    assert row_count == 1
    assert latest_run is not None
    assert latest_run == ("no_txs", 0, 0)


def test_insert_transactions_all_bad_rows_preserves_existing_rows(tmp_path):
    db_path = tmp_path / "congress.duckdb"
    db = Database(db_path)
    _enable_ocr_schema(db.conn)
    db.conn.execute("""
        INSERT INTO metadata (doc_id, first_name, last_name, filing_date, filing_type, fetched_at)
        VALUES ('doc-bad', 'Jane', 'Doe', TIMESTAMP '2024-01-20', 'P', CURRENT_TIMESTAMP)
    """)
    db.conn.close()

    assert (
        _insert_transactions("doc-bad", 2024, "Jane Doe", [_tx()], db_path=str(db_path))
        == 1
    )
    bad_txs = [_tx(date="not a date")]
    assert (
        _insert_transactions("doc-bad", 2024, "Jane Doe", bad_txs, db_path=str(db_path))
        == 0
    )

    con = Database(db_path).conn
    rows = con.execute(
        "SELECT ticker, asset_description FROM transactions WHERE doc_id = 'doc-bad'"
    ).fetchall()
    latest_run = con.execute("""
        SELECT status, raw_row_count, transaction_count, error_message
        FROM pdf_parse_runs
        WHERE doc_id = 'doc-bad'
        ORDER BY parsed_at DESC
        LIMIT 1
    """).fetchone()
    con.close()

    assert rows == [("AAPL", "Apple Inc. (AAPL)")]
    assert latest_run == ("error", 1, 0, '{"invalid_transaction_date": 1}')


def test_insert_transactions_mixed_batch_replaces_with_valid_rows(tmp_path):
    db_path = tmp_path / "congress.duckdb"
    db = Database(db_path)
    _enable_ocr_schema(db.conn)
    db.conn.execute("""
        INSERT INTO metadata (doc_id, first_name, last_name, filing_date, filing_type, fetched_at)
        VALUES ('doc-mixed', 'Jane', 'Doe', TIMESTAMP '2024-01-20', 'P', CURRENT_TIMESTAMP)
    """)
    db.conn.close()

    assert (
        _insert_transactions(
            "doc-mixed",
            2024,
            "Jane Doe",
            [_tx(asset="Old Inc. (OLD)")],
            db_path=str(db_path),
        )
        == 1
    )
    txs = [
        _tx(asset="Microsoft Corp. (MSFT)"),
        _tx(asset="Bad Corp. (BAD)", date="bad"),
    ]
    assert (
        _insert_transactions("doc-mixed", 2024, "Jane Doe", txs, db_path=str(db_path))
        == 0
    )

    con = Database(db_path).conn
    rows = con.execute(
        "SELECT ticker, asset_description FROM transactions WHERE doc_id = 'doc-mixed'"
    ).fetchall()
    latest_run = con.execute("""
        SELECT status, raw_row_count, transaction_count, error_message
        FROM pdf_parse_runs
        WHERE doc_id = 'doc-mixed'
        ORDER BY parsed_at DESC
        LIMIT 1
    """).fetchone()
    con.close()

    assert rows == [("OLD", "Old Inc. (OLD)")]
    assert latest_run == ("error", 2, 0, '{"invalid_transaction_date": 1}')


def test_row_construction_failure_aborts_whole_batch_before_delete(
    monkeypatch, tmp_path
):
    from scripts import ocr_zero_rows

    db_path = tmp_path / "congress.duckdb"
    db = Database(db_path)
    _enable_ocr_schema(db.conn)
    db.conn.execute(
        "INSERT INTO metadata VALUES ('doc-construction', 'Jane', 'Doe', TIMESTAMP '2024-01-20', 'P', CURRENT_TIMESTAMP)"
    )
    db.close()
    assert (
        _insert_transactions(
            "doc-construction",
            2024,
            "Jane Doe",
            [_tx(asset="Old Inc. (OLD)")],
            db_path=str(db_path),
        )
        == 1
    )

    def resolve(asset):
        if "Break" in asset:
            raise RuntimeError("alias construction failed")
        return None

    monkeypatch.setattr(ocr_zero_rows, "resolve_ticker", resolve)
    result = _insert_transactions(
        "doc-construction",
        2024,
        "Jane Doe",
        [_tx(asset="New Inc. (NEW)"), _tx(asset="Break Alias")],
        db_path=str(db_path),
    )
    assert result == 0
    con = Database(db_path).conn
    assert con.execute(
        "SELECT asset_description FROM transactions WHERE doc_id='doc-construction'"
    ).fetchall() == [("Old Inc. (OLD)",)]
    status, error = con.execute(
        "SELECT status, error_message FROM pdf_parse_runs WHERE doc_id='doc-construction' ORDER BY parsed_at DESC LIMIT 1"
    ).fetchone()
    con.close()
    assert status == "error"
    assert "alias construction failed" in error


def test_parallel_writer_acknowledges_failure(monkeypatch):
    from scripts import ocr_parallel

    acknowledgement = queue.Queue(maxsize=1)
    item = {"ack": acknowledgement}
    monkeypatch.setattr(
        ocr_parallel,
        "_write_item",
        lambda unused: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    ocr_parallel._flush([item])
    inserted, error = acknowledgement.get_nowait()
    assert inserted is None
    assert isinstance(error, RuntimeError)
    assert str(error) == "boom"


def test_work_selection_uses_current_db_not_progress(tmp_path):
    from scripts.ocr_zero_rows import get_ocr_work_items

    db_path = tmp_path / "congress.duckdb"
    db = Database(db_path)
    _enable_ocr_schema(db.conn)
    for doc_id in ("retry", "done", "rejected"):
        db.conn.execute(
            "INSERT INTO metadata VALUES (?, 'Jane', 'Doe', TIMESTAMP '2024-01-20', 'P', CURRENT_TIMESTAMP)",
            [doc_id],
        )
        db.conn.execute(
            "INSERT INTO pdf_parse_runs (doc_id, year, parser_version, status, engines_attempted, raw_row_count, transaction_count) VALUES (?, 2024, 'v3', 'zero_rows', '', 0, 0)",
            [doc_id],
        )
    db.conn.execute(
        "INSERT INTO pdf_parse_runs (doc_id, year, parser_version, status, engines_attempted, raw_row_count, transaction_count) VALUES ('done', 2024, ?, 'success', '', 1, 1)",
        [gemini_ocr_common.GEMINI_PARSER_VERSION],
    )
    artifact = b"%PDF-terminal-state"
    pdf_dir = tmp_path / "2024" / "pdfs"
    pdf_dir.mkdir(parents=True)
    for doc_id in ("retry", "done", "rejected"):
        (pdf_dir / f"{doc_id}.pdf").write_bytes(artifact + doc_id.encode())
    digest = gemini_ocr_common.pdf_sha256(pdf_dir / "done.pdf")
    db.conn.execute(
        """INSERT INTO transactions (
               doc_id, member, ticker, asset_description, transaction_date,
               disclosure_date, transaction_type, amount_raw, owner_code, source,
               source_record_id, source_row_id, ingestion_generation, artifact_sha256,
               raw_asset_description, raw_transaction_subtype, notification_date
           ) VALUES (
               'done', 'Jane Doe', 'AAPL', 'Apple Inc. (AAPL)', DATE '2024-01-15',
               DATE '2024-01-20', 'Purchase', 'A', '', 'gemini_ocr',
               'done', 'done:page:1:row:1', ?, ?,
               'Apple Inc. (AAPL)', 'Purchase', DATE '2024-01-20'
           )""",
        [gemini_ocr_common.GEMINI_PARSER_VERSION, digest],
    )
    db.conn.execute(
        "INSERT INTO pdf_parse_runs (doc_id, year, parser_version, status, engines_attempted, raw_row_count, transaction_count) VALUES ('rejected', 2024, ?, 'rejected', '', 301, 0)",
        [gemini_ocr_common.GEMINI_PARSER_VERSION],
    )
    db.close()
    gemini_ocr_common.write_cached_response(
        "done",
        pdf_dir / "done.pdf",
        "MEMBER: Jane Doe\nPAGES: 1\nPAGE: 1\n"
        "Apple Inc. (AAPL) | Purchase | 01/15/24 | 01/20/24 | A",
        cache_dir=str(tmp_path / "gemini_cache"),
    )

    work = get_ocr_work_items(db_path=str(db_path), data_dir=tmp_path, year=2024)
    assert [item[0] for item in work] == ["rejected", "retry"]

    connection = duckdb.connect(str(db_path))
    connection.execute(
        "UPDATE transactions SET raw_asset_description='tampered' WHERE doc_id='done'"
    )
    connection.close()
    work = get_ocr_work_items(db_path=str(db_path), data_dir=tmp_path, year=2024)
    assert [item[0] for item in work] == ["done", "rejected", "retry"]
