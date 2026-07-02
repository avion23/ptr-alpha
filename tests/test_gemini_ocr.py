from datetime import datetime

from analyzer.database import Database
from scripts import gemini_ocr_common
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
