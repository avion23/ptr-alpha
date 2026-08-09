from datetime import datetime
import json
import queue

import pytest

from analyzer.database import Database
from scripts import gemini_ocr_common
from scripts.ocr_zero_rows import insert_transactions


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


def test_validation_duplicate_collapse():
    txs = [_tx(), _tx(), _tx(asset="Microsoft Corp. (MSFT)")]

    valid, rejections = gemini_ocr_common.validate_transactions(
        "doc-dupe", "Jane Doe", txs, datetime(2024, 1, 20), "Jane Doe"
    )

    assert len(valid) == 2
    assert rejections["duplicate_collapsed"] == 1


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


def test_schema_requires_outcome_for_every_pdf_page():
    output = (
        "MEMBER: Jane Doe\nPAGES: 2\nPAGE: 1\n"
        "Apple (AAPL) | Purchase | 01/15/24 | 01/20/24 | A"
    )
    with pytest.raises(
        gemini_ocr_common.GeminiOutputError, match="missing page outcomes"
    ):
        gemini_ocr_common.parse_gemini_output(output, expected_page_count=2)


def test_known_document_canary_rejects_plausible_partial_output():
    parsed = gemini_ocr_common.parse_gemini_output(
        "MEMBER: Harold Rogers\nPAGES: 1\nPAGE: 1\n"
        "SPDR ETF | Purchase | 03/31/26 | 05/08/26 | A",
        expected_page_count=1,
    )
    gemini_ocr_common.validate_known_document(
        "9115808",
        gemini_ocr_common.KNOWN_DOCUMENT_CANARIES["9115808"]["sha256"],
        parsed,
    )
    with pytest.raises(gemini_ocr_common.GeminiOutputError, match="expected 9 rows"):
        gemini_ocr_common.validate_known_document(
            "9115813",
            gemini_ocr_common.KNOWN_DOCUMENT_CANARIES["9115813"]["sha256"],
            parsed,
        )


def test_56_page_known_document_cannot_be_accepted_as_no_transactions():
    canary = gemini_ocr_common.KNOWN_DOCUMENT_CANARIES["8221322"]
    no_transactions = gemini_ocr_common.ParsedGeminiOutput(
        "Known Member", [], 0, True, 56, frozenset(range(1, 57))
    )
    with pytest.raises(
        gemini_ocr_common.GeminiOutputError,
        match="page 2 expected at least 18 rows",
    ):
        gemini_ocr_common.validate_known_document(
            "8221322", canary["sha256"], no_transactions
        )

    page_two_rows = [
        {
            "asset": f"Page two asset {index}",
            "date": "01/01/26",
            "amount_letter": "A",
            "page_number": 2,
        }
        for index in range(18)
    ]
    covered = gemini_ocr_common.ParsedGeminiOutput(
        "Known Member",
        page_two_rows,
        18,
        False,
        56,
        frozenset(range(1, 57)),
    )
    gemini_ocr_common.validate_known_document("8221322", canary["sha256"], covered)


@pytest.mark.parametrize("doc_id", ["9115808", "9115813", "9116141"])
def test_all_pinned_document_canary_counts_and_rows(doc_id):
    canary = gemini_ocr_common.KNOWN_DOCUMENT_CANARIES[doc_id]
    fragment, tx_date, amount = canary["row"]
    transactions = [
        {
            "asset": f"Filler asset {index}",
            "date": "01/01/26",
            "amount_letter": "A",
        }
        for index in range(canary["row_count"] - 1)
    ]
    transactions.append(
        {"asset": fragment.title(), "date": tx_date, "amount_letter": amount}
    )
    parsed = gemini_ocr_common.ParsedGeminiOutput(
        "Known Member",
        transactions,
        len(transactions),
        False,
        1,
        frozenset({1}),
    )
    gemini_ocr_common.validate_known_document(doc_id, canary["sha256"], parsed)


def test_insert_transactions_sets_source_and_is_idempotent(tmp_path):
    db_path = tmp_path / "congress.duckdb"
    db = Database(db_path)
    db.conn.execute("""
        INSERT INTO metadata (doc_id, first_name, last_name, filing_date, filing_type, fetched_at)
        VALUES ('doc-insert', 'Jane', 'Doe', TIMESTAMP '2024-01-20', 'P', CURRENT_TIMESTAMP)
    """)
    db.conn.close()

    txs = [_tx()]
    assert (
        insert_transactions("doc-insert", 2024, "Jane Doe", txs, db_path=str(db_path))
        == 1
    )
    assert (
        insert_transactions("doc-insert", 2024, "Jane Doe", txs, db_path=str(db_path))
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


def test_insert_transactions_plumbs_authoritative_provenance_when_schema_supports_it(
    tmp_path,
):
    db_path = tmp_path / "congress.duckdb"
    db = Database(db_path)
    db.conn.execute("ALTER TABLE transactions ADD COLUMN chamber VARCHAR")
    db.conn.execute("ALTER TABLE transactions ADD COLUMN source_record_id VARCHAR")
    db.conn.execute("ALTER TABLE transactions ADD COLUMN official_filing_date DATE")
    db.conn.execute("ALTER TABLE transactions ADD COLUMN notification_date DATE")
    db.conn.execute("ALTER TABLE transactions ADD COLUMN raw_asset_description VARCHAR")
    db.conn.execute("ALTER TABLE transactions ADD COLUMN ingestion_generation VARCHAR")
    db.conn.execute("ALTER TABLE transactions ADD COLUMN artifact_sha256 VARCHAR")
    db.conn.execute(
        "INSERT INTO metadata VALUES ('doc-provenance', 'Jane', 'Doe', TIMESTAMP '2024-01-20', 'P', CURRENT_TIMESTAMP)"
    )
    db.close()

    inserted = insert_transactions(
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
        "SELECT chamber, source_record_id, official_filing_date, notification_date, raw_asset_description, ingestion_generation, artifact_sha256 FROM transactions WHERE doc_id='doc-provenance'"
    ).fetchone()
    connection.close()
    assert row == (
        "House",
        "doc-provenance",
        datetime(2024, 1, 20).date(),
        datetime(2024, 1, 20).date(),
        "Apple Inc. (AAPL)",
        gemini_ocr_common.GEMINI_PARSER_VERSION,
        "abc123",
    )


def test_insert_transactions_empty_list_preserves_existing_rows(tmp_path):
    db_path = tmp_path / "congress.duckdb"
    db = Database(db_path)
    db.conn.execute("""
        INSERT INTO metadata (doc_id, first_name, last_name, filing_date, filing_type, fetched_at)
        VALUES ('doc-preserve', 'Jane', 'Doe', TIMESTAMP '2024-01-20', 'P', CURRENT_TIMESTAMP)
    """)
    db.conn.close()

    assert (
        insert_transactions(
            "doc-preserve", 2024, "Jane Doe", [_tx()], db_path=str(db_path)
        )
        == 1
    )
    assert (
        insert_transactions("doc-preserve", 2024, "Jane Doe", [], db_path=str(db_path))
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
    db.conn.execute("""
        INSERT INTO metadata (doc_id, first_name, last_name, filing_date, filing_type, fetched_at)
        VALUES ('doc-bad', 'Jane', 'Doe', TIMESTAMP '2024-01-20', 'P', CURRENT_TIMESTAMP)
    """)
    db.conn.close()

    assert (
        insert_transactions("doc-bad", 2024, "Jane Doe", [_tx()], db_path=str(db_path))
        == 1
    )
    bad_txs = [_tx(date="not a date")]
    assert (
        insert_transactions("doc-bad", 2024, "Jane Doe", bad_txs, db_path=str(db_path))
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
    db.conn.execute("""
        INSERT INTO metadata (doc_id, first_name, last_name, filing_date, filing_type, fetched_at)
        VALUES ('doc-mixed', 'Jane', 'Doe', TIMESTAMP '2024-01-20', 'P', CURRENT_TIMESTAMP)
    """)
    db.conn.close()

    assert (
        insert_transactions(
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
        insert_transactions("doc-mixed", 2024, "Jane Doe", txs, db_path=str(db_path))
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
    db.conn.execute(
        "INSERT INTO transactions (doc_id, member, ticker, transaction_date, disclosure_date, transaction_type, amount_raw, owner_code, source) VALUES ('done', 'Jane Doe', 'AAPL', DATE '2024-01-15', DATE '2024-01-20', 'Purchase', 'A', '', 'gemini_ocr')"
    )
    db.conn.execute(
        "INSERT INTO pdf_parse_runs (doc_id, year, parser_version, status, engines_attempted, raw_row_count, transaction_count) VALUES ('rejected', 2024, ?, 'rejected', '', 301, 0)",
        [gemini_ocr_common.GEMINI_PARSER_VERSION],
    )
    db.close()

    work = get_ocr_work_items(db_path=str(db_path), data_dir=tmp_path, year=2024)
    assert [item[0] for item in work] == ["rejected", "retry"]
