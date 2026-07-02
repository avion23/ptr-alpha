from datetime import datetime
import queue

from analyzer.database import Database
from scripts import gemini_ocr_common
from scripts import ocr_parallel
from scripts.ocr_zero_rows import insert_transactions


def _tx(asset="Apple Inc. (AAPL)", date="01/15/24", tx_type="Purchase", amount="A"):
    return {
        "asset": asset,
        "type": tx_type,
        "date": date,
        "notif_date": "01/20/24",
        "amount_letter": amount,
        "amount_midpoint": 8000,
    }


def test_validation_cap_rejects_whole_doc():
    txs = [_tx(date="01/15/24") for _ in range(301)]

    valid, rejections = gemini_ocr_common.validate_transactions(
        "doc-cap", "Jane Doe", txs, datetime(2024, 1, 20), "Jane Doe"
    )

    assert valid == []
    assert rejections == {"row_count_exceeds_cap": 301}


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


def test_call_gemini_cache_round_trip(monkeypatch, tmp_path):
    calls = []

    def fake_run(args, capture_output, text, timeout):
        calls.append(args)

        class Result:
            returncode = 0
            stdout = "MEMBER: Jane Doe\nApple Inc. (AAPL) | Purchase | 01/15/24 | 01/20/24 | A\n"
            stderr = ""

        return Result()

    monkeypatch.setattr(gemini_ocr_common.subprocess, "run", fake_run)

    first, first_error = gemini_ocr_common.call_gemini(
        "sample.pdf", doc_id="doc-cache", cache_dir=str(tmp_path)
    )
    second, second_error = gemini_ocr_common.call_gemini(
        "sample.pdf", doc_id="doc-cache", cache_dir=str(tmp_path)
    )

    assert first == second
    assert first_error == second_error == ""
    assert len(calls) == 1
    assert calls[0][-4:] == ["-o", "temperature", "0", gemini_ocr_common.PROMPT]


def test_call_gemini_empty_stdout_not_cached(monkeypatch, tmp_path):
    def fake_run(args, capture_output, text, timeout):
        class Result:
            returncode = 0
            stdout = "  \n\t"
            stderr = ""

        return Result()

    monkeypatch.setattr(gemini_ocr_common.subprocess, "run", fake_run)

    output, error = gemini_ocr_common.call_gemini(
        "sample.pdf", doc_id="doc-empty", cache_dir=str(tmp_path)
    )

    assert output == ""
    assert error == "empty_response"
    assert not (tmp_path / "doc-empty.txt").exists()


def test_call_gemini_ignores_empty_cache_file(monkeypatch, tmp_path):
    (tmp_path / "doc-empty-cache.txt").write_text("\n")
    calls = []

    def fake_run(args, capture_output, text, timeout):
        calls.append(args)

        class Result:
            returncode = 0
            stdout = "MEMBER: Jane Doe\nApple Inc. (AAPL) | Purchase | 01/15/24 | 01/20/24 | A\n"
            stderr = ""

        return Result()

    monkeypatch.setattr(gemini_ocr_common.subprocess, "run", fake_run)

    output, error = gemini_ocr_common.call_gemini(
        "sample.pdf", doc_id="doc-empty-cache", cache_dir=str(tmp_path)
    )

    assert output is not None
    assert "MEMBER: Jane Doe" in output
    assert error == ""
    assert len(calls) == 1


def test_cache_path_sanitizes_doc_id(tmp_path):
    path = gemini_ocr_common.cache_path("folder/doc\\id", cache_dir=str(tmp_path))

    assert path == tmp_path / "folder_doc_id.txt"


def test_parallel_empty_response_enqueues_error(monkeypatch):
    test_queue = queue.Queue()
    monkeypatch.setattr(ocr_parallel, "write_q", test_queue)
    monkeypatch.setattr(
        ocr_parallel,
        "call_gemini",
        lambda pdf_path, doc_id=None, refresh=False, timeout=90: ("", "empty_response"),
    )

    doc_id, year, status, count, error = ocr_parallel.process_one(("doc-empty", 2024, "sample.pdf"))
    queued = test_queue.get_nowait()

    assert (doc_id, year, status, count, error) == ("doc-empty", 2024, "error", 0, "empty_response")
    assert queued == {
        "doc_id": "doc-empty",
        "year": 2024,
        "status": "error",
        "raw_count": 0,
        "error": "empty_response",
    }


def test_insert_transactions_sets_source_and_is_idempotent(tmp_path):
    db_path = tmp_path / "congress.duckdb"
    db = Database(db_path)
    db.conn.execute("""
        INSERT INTO metadata (doc_id, first_name, last_name, filing_date, filing_type, fetched_at)
        VALUES ('doc-insert', 'Jane', 'Doe', TIMESTAMP '2024-01-20', 'P', CURRENT_TIMESTAMP)
    """)
    db.conn.close()

    txs = [_tx()]
    assert insert_transactions("doc-insert", 2024, "Jane Doe", txs, db_path=str(db_path)) == 1
    assert insert_transactions("doc-insert", 2024, "Jane Doe", txs, db_path=str(db_path)) == 1

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
    assert parse_runs[-1] == ("v4-gemini-manual", "success", 1, 1)


def test_insert_transactions_empty_list_preserves_existing_rows(tmp_path):
    db_path = tmp_path / "congress.duckdb"
    db = Database(db_path)
    db.conn.execute("""
        INSERT INTO metadata (doc_id, first_name, last_name, filing_date, filing_type, fetched_at)
        VALUES ('doc-preserve', 'Jane', 'Doe', TIMESTAMP '2024-01-20', 'P', CURRENT_TIMESTAMP)
    """)
    db.conn.close()

    assert insert_transactions("doc-preserve", 2024, "Jane Doe", [_tx()], db_path=str(db_path)) == 1
    assert insert_transactions("doc-preserve", 2024, "Jane Doe", [], db_path=str(db_path)) == 0

    con = Database(db_path).conn
    row_count = con.execute("SELECT COUNT(*) FROM transactions WHERE doc_id = 'doc-preserve'").fetchone()[0]
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

    assert insert_transactions("doc-bad", 2024, "Jane Doe", [_tx()], db_path=str(db_path)) == 1
    bad_txs = [_tx(date="not a date")]
    assert insert_transactions("doc-bad", 2024, "Jane Doe", bad_txs, db_path=str(db_path)) == 0

    con = Database(db_path).conn
    rows = con.execute("SELECT ticker, asset_description FROM transactions WHERE doc_id = 'doc-bad'").fetchall()
    latest_run = con.execute("""
        SELECT status, raw_row_count, transaction_count, error_message
        FROM pdf_parse_runs
        WHERE doc_id = 'doc-bad'
        ORDER BY parsed_at DESC
        LIMIT 1
    """).fetchone()
    con.close()

    assert rows == [("AAPL", "Apple Inc. (AAPL)")]
    assert latest_run == ("no_txs", 1, 0, "bad date: not a date")


def test_insert_transactions_mixed_batch_replaces_with_valid_rows(tmp_path):
    db_path = tmp_path / "congress.duckdb"
    db = Database(db_path)
    db.conn.execute("""
        INSERT INTO metadata (doc_id, first_name, last_name, filing_date, filing_type, fetched_at)
        VALUES ('doc-mixed', 'Jane', 'Doe', TIMESTAMP '2024-01-20', 'P', CURRENT_TIMESTAMP)
    """)
    db.conn.close()

    assert insert_transactions("doc-mixed", 2024, "Jane Doe", [_tx(asset="Old Inc. (OLD)")], db_path=str(db_path)) == 1
    txs = [_tx(asset="Microsoft Corp. (MSFT)"), _tx(asset="Bad Corp. (BAD)", date="bad")]
    assert insert_transactions("doc-mixed", 2024, "Jane Doe", txs, db_path=str(db_path)) == 1

    con = Database(db_path).conn
    rows = con.execute("SELECT ticker, asset_description FROM transactions WHERE doc_id = 'doc-mixed'").fetchall()
    latest_run = con.execute("""
        SELECT status, raw_row_count, transaction_count, error_message
        FROM pdf_parse_runs
        WHERE doc_id = 'doc-mixed'
        ORDER BY parsed_at DESC
        LIMIT 1
    """).fetchone()
    con.close()

    assert rows == [("MSFT", "Microsoft Corp. (MSFT)")]
    assert latest_run == ("success", 2, 1, "bad date: bad")
