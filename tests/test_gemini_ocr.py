from datetime import date, datetime
from pathlib import Path
import json
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace
from uuid import uuid4

import pytest
import duckdb
import pandas as pd

from analyzer.persistence.database import Database
from analyzer.ocr import common as gemini_ocr_common
from analyzer.ocr.ingestion import insert_transactions, normalize_date


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
    "raw_ticker": "VARCHAR",
    "ticker_candidate": "VARCHAR",
    "raw_asset_class": "VARCHAR",
    "raw_owner": "VARCHAR",
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
        "ON transactions(source, chamber, source_record_id, source_row_id, ingestion_generation)"
    )


def _insert_transactions(*args, **kwargs):
    kwargs.setdefault("artifact_sha256", "test-artifact-sha256")
    kwargs.setdefault("ingestion_generation", "test-house-generation")
    # Each insertion fixture represents an acquired, metadata-bound artifact.
    conn = duckdb.connect(str(kwargs["db_path"]))
    doc_id, year = args[:2]
    generation = kwargs["ingestion_generation"]
    conn.execute(
        "INSERT INTO house_archive_generations (archive_year, generation_id) VALUES (?, ?) ON CONFLICT DO NOTHING",
        [year, generation],
    )
    conn.execute(
        "INSERT INTO house_pdf_artifacts (archive_year, generation_id, doc_id, artifact_sha256) VALUES (?, ?, ?, ?) ON CONFLICT DO NOTHING",
        [year, generation, doc_id, kwargs["artifact_sha256"]],
    )
    conn.execute(
        "INSERT INTO house_generation_metadata SELECT ?, ?, doc_id, first_name, last_name, filing_date, filing_type, fetched_at FROM metadata ON CONFLICT DO NOTHING",
        [year, generation],
    )
    conn.close()
    return insert_transactions(*args, **kwargs)


@pytest.fixture(autouse=True)
def _single_page_pdfinfo(monkeypatch):
    monkeypatch.setattr(gemini_ocr_common, "pdf_page_count", lambda unused: 1)


@pytest.fixture(autouse=True)
def _forbid_network(monkeypatch):
    import socket

    def refused(*args, **kwargs):
        raise AssertionError("OCR tests must not access the network")

    monkeypatch.setattr(socket.socket, "connect", refused)
    monkeypatch.setattr(socket, "create_connection", refused)


def _bound_work(tmp_path):
    from analyzer.ocr.contracts import ExtractionLane, WorkEnvelope, task_key
    from analyzer.ocr.workers import extraction_recipe

    lane = ExtractionLane.GEMINI
    recipe = extraction_recipe(lane)
    identity = ("doc", 2025, "generation", "a" * 64, recipe, lane)
    return WorkEnvelope(
        task_key(*identity), str(uuid4()), "doc", 2025, "generation",
        "a" * 64, str(tmp_path / "doc.pdf"), "Jane Doe", "2025-01-20", 1,
        recipe, lane,
    )


def _tx(asset="Apple Inc. (AAPL)", date="01/15/24", tx_type="Purchase", amount="A"):
    return {
        "asset": asset,
        "type": tx_type,
        "date": date,
        "notif_date": "01/20/24",
        "amount_letter": amount,
        "amount_midpoint": 8000,
    }


def _projection(doc_id, transactions, *, member="Jane Doe", filing="2024-01-20", sha="test-artifact-sha256"):
    from analyzer.ocr.normalization import normalize_rows
    from analyzer.ocr.ingestion import _normalize_ocr_rows

    typed, findings = normalize_rows(doc_id, member, transactions, filing, member, sha)
    assert not findings
    rows, errors = _normalize_ocr_rows(doc_id, member, typed, date.fromisoformat(filing),
                                     artifact_sha256=sha, ingestion_generation="test-house-generation")
    assert not errors
    return rows


def _seed_accepted_history(doc_id, transactions, db_path):
    """Historical fixture setup, not a claim that a fresh API result is approved."""
    from analyzer.ocr.ingestion import record_parse_run

    assert _insert_transactions(doc_id, 2024, 'Jane Doe', transactions, db_path=str(db_path)) == 0
    with Database(db_path) as db:
        rows = _projection(doc_id, transactions)
        db.upsert_transactions(pd.DataFrame(rows), source='gemini_ocr')
        record_parse_run(db.conn, doc_id, 2024, 'success', len(rows), len(rows),
                         parser_version='historical-gemini', artifact_sha256='test-artifact-sha256',
                         ingestion_generation='test-house-generation')
    return len(rows)


def test_normalize_date_accepts_slash_and_hyphen_formats():
    assert normalize_date("1/5/24") == "2024-01-05"
    assert normalize_date("01-05-2024") == "2024-01-05"
    assert normalize_date("2024-01-05") is None


def test_validation_has_no_fixed_row_cap_for_large_real_filings():
    txs = [_tx(asset=f"Asset {index}") for index in range(722)]

    valid, rejections = gemini_ocr_common.validate_transactions(
        "doc-large", "Jane Doe", txs, datetime(2024, 1, 20), "Jane Doe"
    )

    assert len(valid) == 722
    assert "row_count_exceeds_cap" not in rejections


def test_validation_accepts_delayed_filings_and_rejects_post_filing_trades():
    txs = [_tx(date="01/15/24"), _tx(date="01/01/22"), _tx(date="02/10/24")]

    valid, rejections = gemini_ocr_common.validate_transactions(
        "doc-date", "Jane Doe", txs, datetime(2024, 1, 20), "Jane Doe"
    )

    assert [tx["date"] for tx in valid] == ["01/15/24", "01/01/22"]
    assert rejections["date_out_of_window"] == 1


def test_validation_rejects_notification_before_transaction():
    tx = _tx(date="01/15/24")
    tx["notif_date"] = "01/14/24"

    valid, rejections = gemini_ocr_common.validate_transactions(
        "doc-notification", "Jane Doe", [tx], datetime(2024, 1, 20), "Jane Doe"
    )

    assert valid == []
    assert rejections == {"notification_before_transaction": 1}


@pytest.mark.parametrize("notification", ["11/01/30", "11/18/23"])
def test_validation_nulls_unverifiable_post_filing_notification(notification):
    tx = _tx(date="10/20/23")
    tx.update(notif_date=notification, notification_date=notification)
    valid, rejections = gemini_ocr_common.validate_transactions(
        "8220049", "Charles J. Fleischmann", [tx],
        datetime(2023, 11, 17), "Charles J. Fleischmann",
    )
    assert rejections == {}
    assert len(valid) == 1
    assert valid[0]["notif_date"] is None
    assert valid[0]["notification_date"] is None
    assert tx["notif_date"] == notification  # retain the input evidence
    repeated, _ = gemini_ocr_common.validate_transactions(
        "8220049", "Charles J. Fleischmann", valid,
        datetime(2023, 11, 17), "Charles J. Fleischmann",
    )
    assert repeated == valid


def test_validation_accepts_spouse_dc_over_1m_column_k():
    txs = [_tx(amount="K")]

    valid, rejections = gemini_ocr_common.validate_transactions(
        "doc-k", "Jane Doe", txs, datetime(2024, 1, 20), "Jane Doe"
    )

    assert [tx["amount_letter"] for tx in valid] == ["K"]
    assert [tx["amount_midpoint"] for tx in valid] == [1000000]
    assert rejections == {}


def test_validation_nulls_future_notification_without_filing_bound(monkeypatch):
    class FixedDate(date):
        @classmethod
        def today(cls):
            return cls(2026, 10, 15)

    monkeypatch.setattr(gemini_ocr_common, "date", FixedDate)
    tx = _tx(date="10/01/26")
    tx["notif_date"] = "10/16/26"
    valid, rejections = gemini_ocr_common.validate_transactions(
        "future", "Jane Doe", [tx], None, "Jane Doe"
    )
    assert rejections == {}
    assert len(valid) == 1
    assert valid[0]["notif_date"] is None
    assert tx["notif_date"] == "10/16/26"


def test_verified_8220049_corrections_survive_raw_and_validated_projection(tmp_path):
    from analyzer.ocr.normalization import normalize_rows
    from analyzer.ocr.ingestion import _normalize_ocr_rows

    output = (Path(__file__).parent / "fixtures/8220049-ocr.txt").read_text()
    parsed = gemini_ocr_common.parse_gemini_output(output, expected_page_count=3)
    original = json.dumps(parsed.transactions, sort_keys=True)
    sha = "d5cc2d68dca7a0e2a2b99ecd632d3af941fe87ac48fbed5b33598e78f692980b"
    db_path = tmp_path / "replay.duckdb"
    db = Database(db_path)
    _enable_ocr_schema(db.conn)
    db.conn.execute("""
        INSERT INTO metadata (doc_id, first_name, last_name, filing_date, filing_type, fetched_at)
        VALUES ('8220049', 'Charles J.', 'Fleischmann', TIMESTAMP '2023-11-17', 'P', CURRENT_TIMESTAMP)
    """)
    db.close()
    validated, _ = gemini_ocr_common.validate_transactions(
        "8220049", parsed.member, parsed.transactions,
        datetime(2023, 11, 17), parsed.member,
    )
    assert validated[6]["notif_date"] is None
    for txs in (parsed.transactions, parsed.transactions, validated):
        assert _insert_transactions(
            "8220049", 2023, parsed.member, txs, db_path=str(db_path),
            artifact_sha256=sha,
        ) == 0  # Historical correction evidence is not new API admission authority.
        with duckdb.connect(str(db_path)) as conn:
            rows = conn.execute("""
                SELECT source_row_id, transaction_type, notification_date
                FROM transactions ORDER BY source_row_id
            """).fetchall()
        assert rows == []
        typed, findings = normalize_rows('8220049', parsed.member, txs,
                                         '2023-11-17', parsed.member, sha)
        expected_policy = [
            dict(occurrence=number, source_occurrence=number,
                 field='notification_date', notification_date='2030-11-01',
                 filing_date='2023-11-17')
            for number in (6, 12)
        ] if txs is parsed.transactions else []
        assert findings == ({'notification_after_filing': expected_policy}
                            if expected_policy else {})
        projected, errors = _normalize_ocr_rows(
            '8220049', parsed.member, typed, date(2023, 11, 17),
            artifact_sha256=sha, ingestion_generation='test-house-generation',
        )
        assert not errors
        rows = [(row['source_row_id'], row['transaction_type'], row['notification_date']) for row in projected]
        assert len(rows) == 13
        by_row = {row[0]: row[1:] for row in rows}
        for number in range(7, 12):
            assert by_row[f"8220049:page:2:row:{number}"] == (
                "Purchase", '2023-11-01'
            )
        assert by_row["8220049:page:2:row:6"] == ("Sale", None)
        assert by_row["8220049:page:3:row:12"] == ("Sale", None)
    assert json.dumps(parsed.transactions, sort_keys=True) == original
    assert gemini_ocr_common.apply_verified_house_corrections(
        "8220049", "different-pdf", parsed.transactions
    ) == parsed.transactions
    assert gemini_ocr_common.apply_verified_house_corrections(
        "different-doc", sha, parsed.transactions
    ) == parsed.transactions
    # A same-name row with different coordinates/economics is not verified.
    for change in ({"page_number": 1}, {"date": "10/21/23"}, {"amount_letter": "B"}):
        changed = dict(parsed.transactions[6], **change)
        assert gemini_ocr_common.apply_verified_house_corrections(
            "8220049", sha, [changed]
        ) == [changed]


def test_validation_preserves_repeated_lots_without_source_identity():
    txs = [_tx(), _tx(), _tx(asset="Microsoft Corp. (MSFT)")]

    valid, rejections = gemini_ocr_common.validate_transactions(
        "doc-dupe", "Jane Doe", txs, datetime(2024, 1, 20), "Jane Doe"
    )

    assert len(valid) == 3
    assert "duplicate_collapsed" not in rejections


def test_validation_member_mismatch_is_rejected():
    valid, rejections = gemini_ocr_common.validate_transactions(
        "doc-member", "Injected Junk", [_tx()], datetime(2024, 1, 20), "Jane Q. Doe"
    )

    assert valid == []
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
    output, error, metadata = gemini_ocr_common.call_gemini(
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
    output, error, metadata = gemini_ocr_common.call_gemini(
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


def test_cache_identity_includes_explicit_model_and_parser(monkeypatch, tmp_path):
    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"%PDF-canary")
    calls = []

    def fake_run(args, capture_output, text, timeout):
        calls.append(args)

        class Result:
            returncode = 0
            stdout = "MEMBER: Jane Doe\nPAGES: 1\nPAGE: 1\nApple Inc. (AAPL) | Purchase | 01/15/24 | 01/20/24 | A\n"
            stderr = ""

        return Result()

    monkeypatch.setattr(gemini_ocr_common.subprocess, "run", fake_run)
    kwargs = {
        "doc_id": "doc-model",
        "cache_dir": str(tmp_path),
        "model": gemini_ocr_common.GEMINI_35_MODEL,
        "parser_version": gemini_ocr_common.GEMINI_35_PARSER_VERSION,
    }
    first, error, _ = gemini_ocr_common.call_gemini(str(pdf), **kwargs)
    cached, cached_error, _ = gemini_ocr_common.call_gemini(str(pdf), **kwargs)

    assert first == cached
    assert error == cached_error == ""
    assert len(calls) == 1
    assert calls[0][calls[0].index("-m") + 1] == gemini_ocr_common.GEMINI_35_MODEL
    envelope = json.loads((tmp_path / "doc-model.json").read_text())
    assert envelope["model"] == gemini_ocr_common.GEMINI_35_MODEL
    assert envelope["parser_version"] == gemini_ocr_common.GEMINI_35_PARSER_VERSION


def test_gemini_call_forwards_thinking_and_output_budget(monkeypatch, tmp_path):
    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"%PDF-canary")
    calls = []

    def fake_run(args, capture_output, text, timeout):
        calls.append(args)

        class Result:
            returncode = 0
            stdout = "MEMBER: Jane Doe\nPAGES: 1\nPAGE: 1\nApple Inc. (AAPL) | Purchase | 01/15/24 | 01/20/24 | A\n"
            stderr = ""

        return Result()

    monkeypatch.setattr(gemini_ocr_common.subprocess, "run", fake_run)
    output, error, _ = gemini_ocr_common.call_gemini(
        str(pdf),
        model="gemini/gemini-3-flash-preview",
        parser_version="v-test-minimal-thinking",
        cache_dir=str(tmp_path),
        thinking_level="minimal",
        max_output_tokens=65536,
    )

    assert output is not None
    assert error == ""
    command = calls[0]
    assert command[command.index("thinking_level") + 1] == "minimal"
    assert command[command.index("max_output_tokens") + 1] == "65536"


def test_non_gemini_model_omits_temperature_option(monkeypatch, tmp_path):
    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"%PDF-canary")
    calls = []

    def fake_run(args, capture_output, text, timeout):
        calls.append(args)

        class Result:
            returncode = 0
            stdout = "MEMBER: Jane Doe\nPAGES: 1\nPAGE: 1\nApple Inc. (AAPL) | Purchase | 01/15/24 | 01/20/24 | A\n"
            stderr = ""

        return Result()

    monkeypatch.setattr(gemini_ocr_common.subprocess, "run", fake_run)
    output, error, _ = gemini_ocr_common.call_gemini(
        str(pdf),
        model="gpt-5.6-sol",
        parser_version="v7-gpt-5.6-sol",
        cache_dir=str(tmp_path),
    )

    assert output is not None
    assert error == ""
    assert "temperature" not in calls[0]


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
    first, _, _ = gemini_ocr_common.call_gemini(
        str(pdf), doc_id="doc-hash", cache_dir=str(tmp_path)
    )
    cached, _, _ = gemini_ocr_common.call_gemini(
        str(pdf), doc_id="doc-hash", cache_dir=str(tmp_path)
    )
    pdf.write_bytes(b"%PDF-second")
    changed, _, _ = gemini_ocr_common.call_gemini(
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
    output, error, metadata = gemini_ocr_common.call_gemini(
        str(pdf), doc_id="snapshot", cache_dir=str(tmp_path)
    )
    envelope = json.loads((tmp_path / "snapshot.json").read_text())
    assert output is not None and error == ""
    assert attached == [original]
    assert (
        envelope["pdf_sha256"] == gemini_ocr_common.hashlib.sha256(original).hexdigest()
    )
    assert metadata is not None
    assert metadata.sha256 == envelope["pdf_sha256"]
    db_path = tmp_path / "snapshot.duckdb"
    db = Database(db_path)
    _enable_ocr_schema(db.conn)
    db.conn.execute(
        "INSERT INTO metadata (doc_id, first_name, last_name, filing_date, filing_type, fetched_at) VALUES ('snapshot', 'Jane', 'Doe', TIMESTAMP '2024-01-20', 'P', CURRENT_TIMESTAMP)"
    )
    db.close()
    parsed = gemini_ocr_common.parse_gemini_output(output)
    assert (
        _insert_transactions(
            "snapshot",
            2024,
            parsed.member,
            parsed.transactions,
            db_path=str(db_path),
            artifact_sha256=metadata.sha256,
        )
        == 0
    )
    connection = duckdb.connect(str(db_path))
    stored = connection.execute(
        "SELECT artifact_sha256 FROM transactions WHERE doc_id='snapshot'"
    ).fetchone()
    connection.close()
    assert stored is None
    projected, = _projection('snapshot', parsed.transactions, sha=metadata.sha256)
    assert projected['artifact_sha256'] == gemini_ocr_common.hashlib.sha256(original).hexdigest()


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


@pytest.mark.parametrize("sale_type", ["partial sale", "Sale (partial)", "Sale (Partial)"])
def test_schema_accepts_explicit_partial_sale_direction(sale_type):
    parsed = gemini_ocr_common.parse_gemini_output(
        "MEMBER: Jane Doe\nPAGES: 1\nPAGE: 1\n"
        f"Apple (AAPL) | {sale_type} | 01/15/24 | 01/20/24 | A"
    )
    assert parsed.transactions == [
        {
            "asset": "Apple (AAPL)",
            "type": "Sale",
            "date": "01/15/24",
            "notif_date": "01/20/24",
            "amount_letter": "A",
            "amount_midpoint": gemini_ocr_common.AMOUNT_MIDPOINTS["A"],
            "page_number": 1,
        }
    ]


@pytest.mark.parametrize(
    "row,reason",
    [
        ("Apple | Sale (unknown) | 01/15/24 | 01/20/24 | A", "transaction type"),
        ("Apple | Sale (partial) |  | 01/20/24 | A", "transaction date"),
        ("Apple | Sale (partial) | 01/15/24 | 01/20/24 | $348.94", "amount range"),
    ],
)
def test_partial_sale_alias_keeps_critical_fields_fail_closed(row, reason):
    with pytest.raises(gemini_ocr_common.GeminiOutputError, match=reason):
        gemini_ocr_common.parse_gemini_output(
            f"MEMBER: Jane Doe\nPAGES: 1\nPAGE: 1\n{row}"
        )


def test_schema_accepts_unpadded_but_unambiguous_dates():
    parsed = gemini_ocr_common.parse_gemini_output(
        "MEMBER: Jane Doe\nPAGES: 1\nPAGE: 1\n"
        "Apple (AAPL) | Purchase | 1/5/24 | 1/20/24 | A"
    )
    assert parsed.transactions[0]["date"] == "1/5/24"
    assert parsed.transactions[0]["notif_date"] == "1/20/24"


def test_schema_uses_final_member_block_after_reasoning():
    parsed = gemini_ocr_common.parse_gemini_output(
        "Thinking Process:\n"
        "MEMBER: Draft Wrong\nPAGES: 1\nPAGE: 1\nNO_TRANSACTIONS\n"
        "More reasoning after the draft.\n"
        "MEMBER: Jane Doe\nPAGES: 1\nPAGE: 1\n"
        "Apple (AAPL) | Purchase | 10/9/24 | 10/9/24 | A"
    )
    assert parsed.member == "Jane Doe"
    assert len(parsed.transactions) == 1


def test_schema_accepts_explicit_missing_notification_date():
    parsed = gemini_ocr_common.parse_gemini_output(
        "MEMBER: Jane Doe\nPAGES: 1\nPAGE: 1\n"
        "Apple (AAPL) | Purchase | 01/15/24 | N/A | A"
    )

    assert parsed.transactions[0]["notif_date"] is None
    valid, rejections = gemini_ocr_common.validate_transactions(
        "doc-na", "Jane Doe", parsed.transactions, datetime(2024, 1, 20), "Jane Doe"
    )
    assert len(valid) == 1
    assert rejections == {}


def test_schema_accepts_trailing_hallucinated_empty_page_only():
    parsed = gemini_ocr_common.parse_gemini_output(
        "MEMBER: Jane Doe\nPAGES: 2\nPAGE: 1\n"
        "Apple (AAPL) | Purchase | 01/15/24 | 01/20/24 | A\n"
        "PAGE: 2\nNO_TRANSACTIONS",
        expected_page_count=1,
    )

    assert parsed.page_count == 1
    assert parsed.covered_pages == frozenset({1})
    assert len(parsed.transactions) == 1


def test_schema_accepts_inflated_page_header_when_real_pages_are_complete():
    parsed = gemini_ocr_common.parse_gemini_output(
        "MEMBER: Jane Doe\nPAGES: 3\nPAGE: 1\n"
        "Apple (AAPL) | Purchase | 01/15/24 | 01/20/24 | A",
        expected_page_count=1,
    )

    assert parsed.page_count == 1
    assert parsed.covered_pages == frozenset({1})
    assert len(parsed.transactions) == 1


def test_schema_rejects_transaction_on_hallucinated_extra_page():
    output = (
        "MEMBER: Jane Doe\nPAGES: 2\nPAGE: 1\n"
        "Apple (AAPL) | Purchase | 01/15/24 | 01/20/24 | A\n"
        "PAGE: 2\nMicrosoft (MSFT) | Purchase | 01/16/24 | 01/20/24 | A"
    )
    with pytest.raises(gemini_ocr_common.GeminiOutputError, match="page count mismatch"):
        gemini_ocr_common.parse_gemini_output(output, expected_page_count=1)


def test_schema_accepts_spouse_dc_over_1m_column_k():
    parsed = gemini_ocr_common.parse_gemini_output(
        "MEMBER: Jane Doe\nPAGES: 1\nPAGE: 1\n"
        "Northern Fund | Purchase | 01/15/24 | 01/20/24 | K"
    )
    assert [tx["amount_letter"] for tx in parsed.transactions] == ["K"]
    assert [tx["amount_midpoint"] for tx in parsed.transactions] == [1000000]


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
        "INSERT INTO metadata (doc_id, first_name, last_name, filing_date, filing_type, fetched_at) VALUES ('legacy', 'Jane', 'Doe', TIMESTAMP '2024-01-20', 'P', CURRENT_TIMESTAMP)"
    )
    existing = {
        row[1]
        for row in db.conn.execute("PRAGMA table_info('transactions')").fetchall()
    }
    db.close()
    if set(OCR_SCHEMA_COLUMNS) <= existing:
        pytest.skip("production provenance schema is present")
    with pytest.raises(RuntimeError, match="requires the provenance schema"):
        _insert_transactions("legacy", 2024, "Jane Doe", [_tx()], db_path=str(db_path))
    from analyzer.ocr.ingestion import get_ocr_work_items

    with pytest.raises(RuntimeError, match="requires the provenance schema"):
        get_ocr_work_items(db_path=str(db_path), data_dir=tmp_path)


def test_unreviewed_insert_is_idempotent_and_projection_retains_source(tmp_path):
    db_path = tmp_path / "congress.duckdb"
    db = Database(db_path)
    _enable_ocr_schema(db.conn)
    db.conn.execute("""
        INSERT INTO metadata (doc_id, first_name, last_name, filing_date, filing_type, fetched_at)
        VALUES ('doc-insert', 'Jane', 'Doe', TIMESTAMP '2024-01-20', 'P', CURRENT_TIMESTAMP)
    """)
    db.close()

    txs = [_tx()]
    assert (
        _insert_transactions("doc-insert", 2024, "Jane Doe", txs, db_path=str(db_path))
        == 0
    )
    assert (
        _insert_transactions("doc-insert", 2024, "Jane Doe", txs, db_path=str(db_path))
        == 0
    )

    con = duckdb.connect(str(db_path), read_only=True)
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

    assert rows == [] and parse_runs == []
    projected, = _projection('doc-insert', txs)
    assert tuple(projected[key] for key in ('member', 'ticker', 'asset_description', 'source')) == (
        "Jane Doe", "AAPL", "Apple Inc. (AAPL)", "gemini_ocr")


def test_unreviewed_ocr_cannot_replace_semantically_invalid_deterministic_rows(tmp_path):
    db_path = tmp_path / "replace-invalid.duckdb"
    db = Database(db_path)
    _enable_ocr_schema(db.conn)
    db.conn.execute(
        "INSERT INTO metadata (doc_id, first_name, last_name, filing_date, filing_type, fetched_at) "
        "VALUES ('replace-invalid', 'Jane', 'Doe', TIMESTAMP '2024-01-20', 'P', CURRENT_TIMESTAMP)"
    )
    db.upsert_transactions(
        pd.DataFrame([{
            "doc_id": "replace-invalid",
            "member": "Jane Doe",
            "ticker": "AAPL",
            "transaction_date": datetime(2024, 1, 21),
            "disclosure_date": datetime(2024, 1, 20),
            "transaction_type": "Purchase",
            "chamber": "house",
            "source_record_id": "replace-invalid",
            "source_row_id": "pdftotext:r1",
            "official_filing_date": datetime(2024, 1, 20),
            "ingestion_generation": "test-house-generation",
            "artifact_sha256": "test-artifact-sha256",
        }]),
        source="house_pdf",
    )
    db.upsert_parse_run(
        doc_id="replace-invalid",
        year=2024,
        parser_version="v5-deterministic",
        status="success",
        engines_attempted="pdftotext",
        raw_row_count=1,
        transaction_count=1,
        artifact_sha256="test-artifact-sha256",
        ingestion_generation="test-house-generation",
    )
    db.close()

    assert (
        _insert_transactions(
            "replace-invalid",
            2024,
            "Jane Doe",
            [_tx()],
            db_path=str(db_path),
        )
        == 0
    )

    con = duckdb.connect(str(db_path), read_only=True)
    rows = con.execute(
        "SELECT source, transaction_date FROM transactions "
        "WHERE doc_id='replace-invalid' ORDER BY source"
    ).fetchall()
    runs = con.execute(
        "SELECT parser_version, status, transaction_count FROM pdf_parse_runs "
        "WHERE doc_id='replace-invalid' ORDER BY parser_version"
    ).fetchall()
    con.close()

    assert rows == [("house_pdf", datetime(2024, 1, 21).date())]
    assert runs == [("v5-deterministic", "success", 1)]


@pytest.mark.parametrize(
    ("notification_date", "expected_sources", "expected_deterministic_status"),
    [
        (datetime(2024, 1, 14), ["house_pdf"], "success"),
        (datetime(2024, 1, 15), ["house_pdf"], "success"),
    ],
)
def test_unreviewed_supersession_cannot_change_house_notification_chronology(
    notification_date,
    expected_sources,
    expected_deterministic_status,
    tmp_path,
):
    db_path = tmp_path / "notification-chronology.duckdb"
    db = Database(db_path)
    _enable_ocr_schema(db.conn)
    db.conn.execute(
        "INSERT INTO metadata (doc_id, first_name, last_name, filing_date, filing_type, fetched_at) "
        "VALUES ('notification-chronology', 'Jane', 'Doe', TIMESTAMP '2024-01-20', 'P', CURRENT_TIMESTAMP)"
    )
    db.upsert_transactions(
        pd.DataFrame(
            [
                {
                    "doc_id": "notification-chronology",
                    "member": "Jane Doe",
                    "ticker": "AAPL",
                    "transaction_date": datetime(2024, 1, 15),
                    "disclosure_date": datetime(2024, 1, 20),
                    "notification_date": notification_date,
                    "transaction_type": "Purchase",
                    "chamber": "house",
                    "source_record_id": "notification-chronology",
                    "source_row_id": "pdftotext:r1",
                    "official_filing_date": datetime(2024, 1, 20),
                    "ingestion_generation": "test-house-generation",
                    "artifact_sha256": "test-artifact-sha256",
                }
            ]
        ),
        source="house_pdf",
    )
    db.upsert_parse_run(
        doc_id="notification-chronology",
        year=2024,
        parser_version="v5-deterministic",
        status="success",
        engines_attempted="pdftotext",
        raw_row_count=1,
        transaction_count=1,
        artifact_sha256="test-artifact-sha256",
        ingestion_generation="test-house-generation",
    )
    db.close()

    assert (
        _insert_transactions(
            "notification-chronology",
            2024,
            "Jane Doe",
            [_tx()],
            db_path=str(db_path),
        )
        == 0
    )

    con = duckdb.connect(str(db_path), read_only=True)
    rows = con.execute(
        "SELECT source FROM transactions "
        "WHERE doc_id='notification-chronology' ORDER BY source"
    ).fetchall()
    deterministic_run = con.execute(
        "SELECT status, transaction_count FROM pdf_parse_runs "
        "WHERE doc_id='notification-chronology' AND parser_version='v5-deterministic'"
    ).fetchone()
    con.close()

    assert [source for (source,) in rows] == expected_sources
    assert deterministic_run == (
        expected_deterministic_status,
        0 if expected_deterministic_status == "invalidated" else 1,
    )


def test_projection_preserves_repeated_lots_with_distinct_source_rows(tmp_path):
    db_path = tmp_path / "lots.duckdb"
    db = Database(db_path)
    _enable_ocr_schema(db.conn)
    db.conn.execute(
        "INSERT INTO metadata (doc_id, first_name, last_name, filing_date, filing_type, fetched_at) VALUES ('lots', 'Jane', 'Doe', TIMESTAMP '2024-01-20', 'P', CURRENT_TIMESTAMP)"
    )
    db.close()
    assert (
        _insert_transactions(
            "lots", 2024, "Jane Doe", [_tx(), _tx()], db_path=str(db_path)
        )
        == 0
    )
    con = duckdb.connect(str(db_path))
    rows = con.execute(
        "SELECT source_record_id, source_row_id FROM transactions WHERE doc_id='lots' ORDER BY source_row_id"
    ).fetchall()
    con.close()
    assert rows == []
    projected = _projection('lots', [_tx(), _tx()])
    assert [(row['source_record_id'], row['source_row_id']) for row in projected] == [
        ("lots", "lots:row:1"), ("lots", "lots:row:2")]


def test_projection_retains_authoritative_provenance_without_authorizing_insertion(
    tmp_path,
):
    db_path = tmp_path / "congress.duckdb"
    db = Database(db_path)
    _enable_ocr_schema(db.conn)
    db.conn.execute(
        "INSERT INTO metadata (doc_id, first_name, last_name, filing_date, filing_type, fetched_at) VALUES ('doc-provenance', 'Jane', 'Doe', TIMESTAMP '2024-01-20', 'P', CURRENT_TIMESTAMP)"
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
    assert inserted == 0
    connection = duckdb.connect(str(db_path), read_only=True)
    row = connection.execute(
        "SELECT chamber, source_record_id, source_row_id, official_filing_date, notification_date, raw_asset_description, ingestion_generation, artifact_sha256 FROM transactions WHERE doc_id='doc-provenance'"
    ).fetchone()
    connection.close()
    assert row is None
    projected, = _projection('doc-provenance', [_tx()], sha='abc123')
    assert tuple(projected[key] for key in ('chamber', 'source_record_id', 'source_row_id',
        'official_filing_date', 'notification_date', 'raw_asset_description', 'ingestion_generation', 'artifact_sha256')) == (
        "House",
        "doc-provenance",
        "doc-provenance:row:1",
        date(2024, 1, 20),
        '2024-01-20',
        "Apple Inc. (AAPL)",
        "test-house-generation",
        "abc123",
    )


def test_tickerless_resolver_is_unverified_candidate_not_canonical(
    monkeypatch, tmp_path
):
    from analyzer.ocr import ingestion as ocr_zero_rows

    db_path = tmp_path / "candidate.duckdb"
    db = Database(db_path)
    _enable_ocr_schema(db.conn)
    db.conn.execute(
        "INSERT INTO metadata (doc_id, first_name, last_name, filing_date, filing_type, fetched_at) VALUES ('candidate', 'Jane', 'Doe', TIMESTAMP '2024-01-20', 'P', CURRENT_TIMESTAMP)"
    )
    db.close()
    monkeypatch.setattr(ocr_zero_rows, "resolve_ticker", lambda asset: "ACME")
    assert (
        _insert_transactions(
            "candidate",
            2024,
            "Jane Doe",
            [_tx(asset="Acme Holdings")],
            db_path=str(db_path),
        )
        == 0
    )
    connection = duckdb.connect(str(db_path))
    row = connection.execute(
        "SELECT ticker, raw_ticker, ticker_candidate, ticker_origin, available_date, raw_asset_class "
        "FROM transactions WHERE doc_id='candidate'"
    ).fetchone()
    connection.close()
    assert row is None
    projected, = _projection('candidate', [_tx(asset='Acme Holdings')])
    assert tuple(projected.get(key) for key in ('ticker', 'raw_ticker', 'ticker_candidate',
                                               'ticker_origin', 'available_date', 'raw_asset_class')) == (
        None,
        "ACME",
        "ACME",
        "unverified",
        datetime(2024, 1, 20).date(),
        "Not separately reported",
    )


def test_mixed_ticker_provenance_rows_use_fixed_full_staging_schema(
    monkeypatch, tmp_path
):
    from analyzer.ocr import ingestion as ocr_zero_rows

    db_path = tmp_path / "mixed-schema.duckdb"
    db = Database(db_path)
    _enable_ocr_schema(db.conn)
    db.conn.execute(
        "INSERT INTO metadata (doc_id, first_name, last_name, filing_date, filing_type, fetched_at) VALUES ('mixed-schema', 'Jane', 'Doe', TIMESTAMP '2024-01-20', 'P', CURRENT_TIMESTAMP)"
    )
    db.close()
    monkeypatch.setattr(ocr_zero_rows, "resolve_ticker", lambda asset: "PRIVATE")
    rows = [_tx(asset="Apple Ticker: AAPL"), _tx(asset="Private Holdings")]
    assert (
        _insert_transactions(
            "mixed-schema", 2024, "Jane Doe", rows, db_path=str(db_path)
        )
        == 0
    )
    connection = duckdb.connect(str(db_path))
    stored = connection.execute(
        "SELECT ticker, ticker_candidate, ticker_origin FROM transactions "
        "WHERE doc_id='mixed-schema' ORDER BY source_row_id"
    ).fetchall()
    connection.close()
    assert stored == []
    projected = _projection('mixed-schema', rows)
    assert [(row.get('ticker'), row.get('ticker_candidate'), row['ticker_origin']) for row in projected] == [
        ("AAPL", None, "official"), (None, "PRIVATE", "unverified")]


def test_ticker_provenance_matrix_rejects_inconsistent_rows():
    from analyzer.ocr.ingestion import validate_ticker_provenance

    with pytest.raises(ValueError, match="unverified"):
        validate_ticker_provenance(
            {
                "ticker_origin": "unverified",
                "ticker": "AAPL",
                "raw_ticker": "AAPL",
                "ticker_candidate": "AAPL",
            }
        )


def test_unreviewed_empty_list_preserves_existing_ocr_rows(tmp_path):
    db_path = tmp_path / "congress.duckdb"
    db = Database(db_path)
    _enable_ocr_schema(db.conn)
    db.conn.execute("""
        INSERT INTO metadata (doc_id, first_name, last_name, filing_date, filing_type, fetched_at)
        VALUES ('doc-preserve', 'Jane', 'Doe', TIMESTAMP '2024-01-20', 'P', CURRENT_TIMESTAMP)
    """)
    db.close()

    assert (
        _seed_accepted_history("doc-preserve", [_tx()], db_path)
        == 1
    )
    assert (
        _insert_transactions("doc-preserve", 2024, "Jane Doe", [], db_path=str(db_path))
        == 0
    )

    con = duckdb.connect(str(db_path), read_only=True)
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
    assert latest_run == ("success", 1, 1)


def test_unreviewed_bad_rows_preserve_current_artifact_rows(tmp_path):
    db_path = tmp_path / "congress.duckdb"
    db = Database(db_path)
    _enable_ocr_schema(db.conn)
    db.conn.execute("""
        INSERT INTO metadata (doc_id, first_name, last_name, filing_date, filing_type, fetched_at)
        VALUES ('doc-bad', 'Jane', 'Doe', TIMESTAMP '2024-01-20', 'P', CURRENT_TIMESTAMP)
    """)
    db.close()

    assert (
        _seed_accepted_history("doc-bad", [_tx()], db_path)
        == 1
    )
    bad_txs = [_tx(date="not a date")]
    assert (
        _insert_transactions("doc-bad", 2024, "Jane Doe", bad_txs, db_path=str(db_path))
        == 0
    )

    con = duckdb.connect(str(db_path), read_only=True)
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

    assert rows == [('AAPL', 'Apple Inc. (AAPL)')]
    assert latest_run[:3] == ('success', 1, 1)
    valid, findings = gemini_ocr_common.validate_transactions('doc-bad', 'Jane Doe', bad_txs, date(2024,1,20), 'Jane Doe')
    assert valid == [] and findings == {'invalid_transaction_date': 1}


def test_unreviewed_mixed_and_later_valid_batches_preserve_current_artifact_rows(tmp_path):
    db_path = tmp_path / "congress.duckdb"
    db = Database(db_path)
    _enable_ocr_schema(db.conn)
    db.conn.execute("""
        INSERT INTO metadata (doc_id, first_name, last_name, filing_date, filing_type, fetched_at)
        VALUES ('doc-mixed', 'Jane', 'Doe', TIMESTAMP '2024-01-20', 'P', CURRENT_TIMESTAMP)
    """)
    db.close()

    assert (
        _seed_accepted_history('doc-mixed', [_tx(asset='Old Inc. (OLD)')], db_path)
        == 1
    )
    txs = [
        _tx(asset="Microsoft Corp. (MSFT)"),
        _tx(asset="Bad Corp. (BAD)", date="bad"),
    ]
    # Neither malformed nor syntactically valid API results can replace
    # accepted history without independent, bound source review.
    assert (
        _insert_transactions("doc-mixed", 2024, "Jane Doe", txs, db_path=str(db_path))
        == 0
    )

    con = duckdb.connect(str(db_path), read_only=True)
    blocked_run = con.execute("""
        SELECT status, raw_row_count, transaction_count, error_message
        FROM pdf_parse_runs
        WHERE doc_id = 'doc-mixed' AND status = 'partial'
        ORDER BY parsed_at DESC
        LIMIT 1
    """).fetchone()
    assert 'ocr_partial_results' not in {row[0] for row in con.execute('SHOW TABLES').fetchall()}
    kept_rows = con.execute(
        "SELECT ticker FROM transactions WHERE doc_id = 'doc-mixed'"
    ).fetchall()
    con.close()

    assert blocked_run is None
    valid, findings = gemini_ocr_common.validate_transactions('doc-mixed', 'Jane Doe', txs, date(2024,1,20), 'Jane Doe')
    assert len(valid) == 1 and findings == {'invalid_transaction_date': 1}
    assert [row[0] for row in kept_rows] == ["OLD"], "existing artifact row must survive"

    # A later syntactically valid batch still cannot replace without review.
    assert (
        _insert_transactions(
            "doc-mixed", 2024, "Jane Doe", [_tx(asset="Microsoft Corp. (MSFT)")],
            db_path=str(db_path),
        )
        == 0
    )

    con = duckdb.connect(str(db_path), read_only=True)
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

    # Neither fresh result may retag the historical acceptance.
    assert latest_run[0] == "success"
    assert latest_run[1:3] == (1, 1)
    assert [row[0] for row in rows] == ["OLD"]


def test_unreviewed_row_construction_failure_cannot_retire_current_artifact_rows(
    monkeypatch, tmp_path
):
    from analyzer.ocr import ingestion as ocr_zero_rows

    db_path = tmp_path / "congress.duckdb"
    db = Database(db_path)
    _enable_ocr_schema(db.conn)
    db.conn.execute(
        "INSERT INTO metadata (doc_id, first_name, last_name, filing_date, filing_type, fetched_at) VALUES ('doc-construction', 'Jane', 'Doe', TIMESTAMP '2024-01-20', 'P', CURRENT_TIMESTAMP)"
    )
    db.close()
    assert (
        _seed_accepted_history('doc-construction', [_tx(asset='Old Inc. (OLD)')], db_path)
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
    con = duckdb.connect(str(db_path), read_only=True)
    assert con.execute(
        "SELECT asset_description FROM transactions WHERE doc_id='doc-construction'"
    ).fetchall() == [('Old Inc. (OLD)',)]
    status, _error = con.execute(
        "SELECT status, error_message FROM pdf_parse_runs WHERE doc_id='doc-construction' ORDER BY parsed_at DESC LIMIT 1"
    ).fetchone()
    con.close()
    assert status == "success"
    with pytest.raises(AssertionError, match="alias construction failed"):
        _projection('doc-construction', [_tx(asset='New Inc. (NEW)'), _tx(asset='Break Alias')])


def test_bound_worker_reports_quota_without_retry_or_database(monkeypatch, tmp_path):
    from analyzer.ocr import workers
    from analyzer.ocr.contracts import ExtractionOutcome

    work = _bound_work(tmp_path)
    calls = []

    @contextmanager
    def snapshot(path):
        assert str(path) == work.artifact_path
        yield SimpleNamespace(path=Path(path), sha256=work.artifact_sha256, page_count=1)

    def quota(*args, **kwargs):
        calls.append(kwargs)
        kwargs["response_sink"]("raw quota response")
        return None, "429 RESOURCE_EXHAUSTED", None

    def forbidden(*args, **kwargs):
        raise AssertionError("worker opened a database")

    monkeypatch.setattr(gemini_ocr_common, "snapshot_pdf", snapshot)
    monkeypatch.setattr(gemini_ocr_common, "call_gemini", quota)
    monkeypatch.setattr(duckdb, "connect", forbidden)
    result = workers.load_result(workers.extract_document(work, str(tmp_path / "spool")))
    assert len(calls) == 1
    assert result.outcome == ExtractionOutcome.RETRY_WAIT
    assert result.retry_after > 0
    assert result.raw_response == "raw quota response"
    assert not workers.admission_decision(result, tmp_path)["allowed"]
    assert not list(tmp_path.rglob("*.duckdb"))


@pytest.mark.parametrize("flag", ["--timeout", "--doc-ids", "--retry-failed"])
def test_parallel_rejects_unsupported_legacy_flags(monkeypatch, flag):
    from scripts import ocr_parallel
    import sys

    monkeypatch.setattr(sys, "argv", ["ocr_parallel", flag, "123"])
    with pytest.raises(SystemExit) as exc:
        ocr_parallel.main()
    assert exc.value.code == 2


def test_result_cannot_be_normalized_for_another_document(tmp_path):
    from analyzer.ocr.contracts import ExtractionOutcome, ResultBundle
    from analyzer.ocr.normalization import normalize_candidate

    work = _bound_work(tmp_path)
    result = ResultBundle(work, ExtractionOutcome.CANDIDATE, [_tx()])
    for changed in (replace(work, doc_id="other"), replace(work, archive_year=2024),
                    replace(work, generation_id="other"), replace(work, artifact_sha256="b" * 64)):
        with pytest.raises(ValueError, match="identity"):
            normalize_candidate(changed, result)


def test_failed_receipt_finish_remains_replayable(tmp_path):
    from analyzer.ocr.contracts import ExtractionOutcome, ResultBundle

    with Database(tmp_path / "ledger.duckdb") as db:
        work = _bound_work(tmp_path)
        db.work.register(work)
        claimed = db.work.claim(work.task_id)
        result = ResultBundle(claimed, ExtractionOutcome.CANDIDATE, [_tx()])
        result_id = db.work.register_result(result, str(tmp_path / "result.json"))
        receipt = db.work.pending_results()[0]
        assert receipt["result_id"] == result_id
        with pytest.raises(ValueError, match="retry_at"):
            db.work.finish(receipt, "retry_wait")
        assert not db.work.is_finished(receipt)
        assert db.work.pending_results() == [receipt]
        assert db.work.status()["by_state"] == {"running": 1}


def test_schema4_preserves_repeated_lots_exact_amount_and_raw_columns(tmp_path):
    from analyzer.ocr.contracts import ExtractionOutcome, ResultBundle
    from analyzer.ocr.normalization import normalize_candidate

    row = {
        "asset": "Apple Inc. (AAPL)", "type": "Purchase", "date": "01/15/25",
        "notif_date": None, "amount_raw": "$12,345.67", "owner": "SP",
        "account": "Joint account", "instrument_type": "unknown",
        "raw_cells": {"Amount": "$12,345.67", "Owner": "SP", "Unmapped": "verbatim"},
        "extra_source_field": {"terms": ["keep", "all"]},
    }
    raw = "SCHEMA: 4\nMEMBER: Jane Doe\nPAGES: 1\nPAGE: 1\n" + "\n".join(
        "ROW: " + json.dumps(row) for _ in range(2)
    )
    parsed = gemini_ocr_common.parse_gemini_output(raw, expected_page_count=1)
    assert parsed.output_schema_version == 4
    assert not parsed.legacy_limited
    assert parsed.raw_row_count == 2
    assert parsed.transactions == [{**row, "page_number": 1}] * 2
    work = _bound_work(tmp_path)
    result = ResultBundle(work, ExtractionOutcome.CANDIDATE, parsed.transactions, raw,
                          evidence={"output_schema_version": 4, "page_count": 1,
                                    "page_outcomes": [{"page_number": 1, "outcome": "transactions"}]})
    before = result.to_dict()
    normalized = normalize_candidate(work, result)
    assert normalized["status"] == "candidate"
    assert not normalized["legacy_limited"]
    assert [item["occurrence"] for item in normalized["rows"]] == [1, 2]
    assert [item["owner_code"] for item in normalized["rows"]] == ["SP", "SP"]
    assert [item["amount_exact"] for item in normalized["rows"]] == ["12345.67"] * 2
    assert normalized["raw_rows"] == parsed.transactions
    assert result.to_dict() == before
    legacy = gemini_ocr_common.parse_gemini_output(
        "MEMBER: Jane Doe\nPAGES: 1\nPAGE: 1\nApple (AAPL) | Purchase | 01/15/25 | N/A | A"
    )
    assert legacy.output_schema_version == 3
    assert legacy.legacy_limited


@pytest.mark.parametrize("raw_heading,date", [
    ("Date of Notified Transaction", "01/20/25"),
    ("DATE NOTIFIED OF TRANSACTION", "1/20/25"),
    ("Date of Notified of Transaction", "01/20/25"),
])
def test_explicit_raw_notification_date_with_typed_null_requires_review(tmp_path, raw_heading, date):
    from analyzer.ocr.contracts import ExtractionOutcome, ResultBundle
    from analyzer.ocr.normalization import normalize_candidate

    # Synthetic equivalents of the short-filings/9106236 spotcheck defects.
    row = {**_tx(date="01/15/25"), "notif_date": None, "amount_raw": "$1,000-$15,000",
           "raw_cells": {raw_heading: date}, "page_number": 1}
    work = _bound_work(tmp_path)
    result = ResultBundle(work, ExtractionOutcome.CANDIDATE, [row], evidence={
        "output_schema_version": 4, "page_count": 1,
        "page_outcomes": [{"page_number": 1, "outcome": "transactions"}],
    })
    before = result.to_dict()
    normalized = normalize_candidate(work, result)
    assert normalized["status"] == "needs_review"
    assert normalized["findings"]["raw_cell_consistency"] == [{
        "occurrence": 1, "field": "notification_date", "source_key": raw_heading,
        "reason": "typed_omission",
    }]
    assert normalized["raw_rows"] == [row]
    # A contradiction finding must not silently repair either evidence layer.
    assert normalized["rows"][0]["notification_date"] is None
    assert result.to_dict() == before


def test_typed_sale_conflicting_with_raw_purchase_checkbox_requires_review(tmp_path):
    from analyzer.ocr.contracts import ExtractionOutcome, ResultBundle
    from analyzer.ocr.normalization import normalize_candidate

    row = {**_tx(date="01/15/25", tx_type="Sale"), "notif_date": "01/20/25", "amount_raw": "$50,001-$100,000",
           "raw_cells": {"PURCHASE": "X", "SALE": "", "EXCHANGE": ""}, "page_number": 1}
    work = _bound_work(tmp_path)
    result = ResultBundle(work, ExtractionOutcome.CANDIDATE, [row], evidence={
        "output_schema_version": 4, "page_count": 1,
        "page_outcomes": [{"page_number": 1, "outcome": "transactions"}],
    })
    before = result.to_dict()
    normalized = normalize_candidate(work, result)
    assert normalized["status"] == "needs_review"
    assert normalized["findings"]["raw_cell_consistency"] == [{
        "occurrence": 1, "field": "type", "source_key": "PURCHASE",
        "reason": "typed_source_mismatch",
    }]
    assert normalized["raw_rows"] == [row]
    assert result.to_dict() == before


@pytest.mark.parametrize("typed,cells,field,reason", [
    ({"owner": "JT"}, {"Owner": ""}, "owner", "source_blank_typed_populated"),
    ({"amount_raw": "$1,000-$15,000"},
     {"$1,000-$15,000": "", "$15,001-$50,000": "X"}, "amount_raw", "typed_source_mismatch"),
    ({"type": "Sale"}, {"Purchase": "X", "Sale": "X"}, "type", "multiple_source_selections"),
])
def test_raw_cell_claim_conflicts_are_findings_not_silent_corrections(tmp_path, typed, cells, field, reason):
    from analyzer.ocr.contracts import ExtractionOutcome, ResultBundle
    from analyzer.ocr.normalization import normalize_candidate

    row = {**_tx(date="01/15/25"), "notif_date": "01/20/25", "amount_raw": "$1,000-$15,000",
           **typed, "raw_cells": cells, "page_number": 1}
    work = _bound_work(tmp_path)
    result = ResultBundle(work, ExtractionOutcome.CANDIDATE, [row], evidence={
        "output_schema_version": 4, "page_count": 1,
        "page_outcomes": [{"page_number": 1, "outcome": "transactions"}],
    })
    before = result.to_dict()
    normalized = normalize_candidate(work, result)
    assert normalized["status"] == "needs_review"
    conflicts = normalized["findings"]["raw_cell_consistency"]
    assert all(item["occurrence"] == 1 and item["field"] == field and item["reason"] == reason
               for item in conflicts)
    assert conflicts
    for key, value in typed.items():
        assert normalized["rows"][0][key] == value
    assert result.to_dict() == before


@pytest.mark.parametrize("number", ["NaN", "Infinity", "1e999"])
def test_schema4_rejects_nonfinite_numbers_nested_in_raw_source(number):
    row = {"asset": "Apple (AAPL)", "type": "Purchase", "date": "01/15/25",
           "notif_date": None, "amount_raw": "$1,234", "owner": None, "account": None,
           "instrument_type": "unknown", "raw_cells": {"nested": [{"number": "NUMBER_TOKEN"}]}}
    raw = "SCHEMA: 4\nMEMBER: Jane Doe\nPAGES: 1\nPAGE: 1\nROW: " + json.dumps(row).replace('"NUMBER_TOKEN"', number)
    with pytest.raises(gemini_ocr_common.GeminiOutputError, match="invalid ROW JSON"):
        gemini_ocr_common.parse_gemini_output(raw, expected_page_count=1)


def test_durable_delivery_blocks_redispatch_and_finish_replay_is_idempotent(tmp_path):
    from analyzer.ocr.contracts import ExtractionOutcome, ResultBundle

    path = tmp_path / "replay.duckdb"
    with Database(path) as db:
        work = _bound_work(tmp_path)
        db.work.register(work)
        claimed = db.work.claim(work.task_id)
        result = ResultBundle(claimed, ExtractionOutcome.CANDIDATE, [_tx()])
        result_id = db.work.register_result(result, "durable-result.json")
        assert db.work.register_result(result, "durable-result.json") == result_id
        db.conn.execute("UPDATE extraction_work SET lease_until = CURRENT_TIMESTAMP - INTERVAL '1 SECOND'")
        assert db.work.recover_expired() == 0
        receipt = db.work.pending_results()[0]
    with Database(path) as recovered:
        assert recovered.work.pending_results() == [receipt]
        assert recovered.work.pending(claimed.lane) == []
        assert recovered.work.validate_receipt(receipt) == claimed
        recovered.work.finish(receipt, "needs_review", "validation refused")
        recovered.work.finish(receipt, "accepted", "must not overwrite first disposition")
        assert recovered.work.is_finished(receipt)
        assert recovered.work.pending_results() == []
        assert recovered.work.status() == {
            "total": 1, "by_state": {"needs_review": 1}, "results": 1,
        }
        assert recovered.conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0] == 0


def test_semantic_zero_error_preserves_prior_ocr_rows(tmp_path):
    db_path = tmp_path / "error-preserves.duckdb"
    db = Database(db_path)
    _enable_ocr_schema(db.conn)
    db.conn.execute(
        "INSERT INTO metadata (doc_id, first_name, last_name, filing_date, filing_type, fetched_at) VALUES ('error-preserves', 'Jane', 'Doe', TIMESTAMP '2024-01-20', 'P', CURRENT_TIMESTAMP)"
    )
    db.conn.execute(
        """INSERT INTO transactions (
               doc_id, member, ticker, transaction_date, disclosure_date,
               transaction_type, amount_raw, source, chamber, source_record_id,
               source_row_id, ingestion_generation, artifact_sha256
           ) VALUES (
               'error-preserves', 'Jane Doe', 'OLD', DATE '2024-01-01', DATE '2024-01-20',
               'Purchase', 'A', 'gemini_ocr', NULL, NULL, 'legacy', NULL, 'old-sha'
           )"""
    )
    db.close()

    for _ in range(2):
        assert (
            _insert_transactions(
                "error-preserves",
                2024,
                "Jane Doe",
                [],
                db_path=str(db_path),
                raw_count=3,
            )
            == 0
        )
    connection = duckdb.connect(str(db_path))
    rows = connection.execute(
        "SELECT ticker FROM transactions WHERE doc_id='error-preserves'"
    ).fetchall()
    latest = connection.execute(
        "SELECT status, raw_row_count, transaction_count, error_message "
        "FROM pdf_parse_runs WHERE doc_id='error-preserves' "
        "ORDER BY parsed_at DESC LIMIT 1"
    ).fetchone()
    connection.close()
    assert rows == [("OLD",)]
    assert latest is None  # Unauthorized empty claims cannot retag historical rows.


def test_unreviewed_failed_exact_artifact_revalidation_preserves_current_ocr_rows(tmp_path):
    db_path = tmp_path / "failed-revalidation.duckdb"
    db = Database(db_path)
    _enable_ocr_schema(db.conn)
    db.conn.execute(
        "INSERT INTO metadata (doc_id, first_name, last_name, filing_date, filing_type, fetched_at) "
        "VALUES ('failed-revalidation', 'Jane', 'Doe', TIMESTAMP '2024-01-20', 'P', CURRENT_TIMESTAMP)"
    )
    db.conn.execute(
        """INSERT INTO transactions (
               doc_id, member, ticker, transaction_date, disclosure_date,
               transaction_type, amount_raw, source, chamber, source_record_id,
               source_row_id, official_filing_date, available_date,
               notification_date, ingestion_generation, artifact_sha256
           ) VALUES (
               'failed-revalidation', 'Jane Doe', 'AAPL', DATE '2024-01-15', DATE '2024-01-20',
               'Purchase', 'A', 'gemini_ocr', 'House', 'failed-revalidation',
               'failed-revalidation:old', DATE '2024-01-20', DATE '2024-01-20',
               DATE '2024-01-20', 'test-house-generation', 'test-artifact-sha256'
           )"""
    )
    db.close()

    rejected = _tx()
    rejected["notif_date"] = "01/14/24"
    assert (
        _insert_transactions(
            "failed-revalidation",
            2024,
            "Jane Doe",
            [rejected],
            db_path=str(db_path),
        )
        == 0
    )

    connection = duckdb.connect(str(db_path), read_only=True)
    try:
        assert connection.execute(
            "SELECT COUNT(*) FROM transactions WHERE doc_id='failed-revalidation'"
        ).fetchone()[0] == 1
        assert connection.execute(
            "SELECT status, transaction_count, error_message FROM pdf_parse_runs "
            "WHERE doc_id='failed-revalidation' ORDER BY parsed_at DESC LIMIT 1"
        ).fetchone() is None
    finally:
        connection.close()


def test_unreviewed_no_transactions_preserves_legacy_null_and_unrelated_rows(tmp_path):
    from analyzer.ocr.ingestion import get_ocr_work_items

    db_path = tmp_path / "no-txs.duckdb"
    db = Database(db_path)
    _enable_ocr_schema(db.conn)
    db.conn.execute(
        "INSERT INTO metadata (doc_id, first_name, last_name, filing_date, filing_type, fetched_at) VALUES ('no-txs', 'Jane', 'Doe', TIMESTAMP '2024-01-20', 'P', CURRENT_TIMESTAMP)"
    )
    db.conn.execute(
        "INSERT INTO pdf_parse_runs (doc_id, year, parser_version, status, engines_attempted, raw_row_count, transaction_count) "
        "VALUES ('no-txs', 2024, 'v3', 'zero_rows', '', 0, 0)"
    )
    db.conn.execute(
        """INSERT INTO transactions (
               doc_id, member, ticker, transaction_date, disclosure_date,
               transaction_type, amount_raw, source, chamber, source_record_id,
               source_row_id, ingestion_generation, artifact_sha256
           ) VALUES
               ('no-txs', 'Jane Doe', 'OLD', DATE '2024-01-01', DATE '2024-01-20',
                'Purchase', 'A', 'gemini_ocr', NULL, NULL, 'legacy-null', NULL, 'old-sha'),
               ('no-txs', 'Jane Doe', 'CAP', DATE '2024-01-02', DATE '2024-01-20',
                'Purchase', 'A', 'capitol_trades', 'House', 'no-txs', 'cap', 'v1', 'cap-sha'),
               ('no-txs', 'Jane Doe', 'SEN', DATE '2024-01-03', DATE '2024-01-20',
                'Purchase', 'A', 'gemini_ocr', 'Senate', 'no-txs', 'sen', 'old', 'sen-sha')
        """
    )
    db.close()
    pdf_dir = tmp_path / "2024" / "pdfs"
    pdf_dir.mkdir(parents=True)
    pdf = pdf_dir / "no-txs.pdf"
    pdf.write_bytes(b"%PDF-no-transactions")
    digest = gemini_ocr_common.pdf_sha256(pdf)
    connection = duckdb.connect(str(db_path))
    connection.execute(
        """INSERT INTO house_archive_generations (
               archive_year, generation_id, metadata_sha256,
               metadata_http_status, metadata_count, ptr_count, parse_status
           ) VALUES (2024, 'test-house-generation', 'metadata-sha', 200, 1, 1, 'incomplete')"""
    )
    connection.execute(
        """INSERT INTO house_pdf_artifacts (
               archive_year, doc_id, generation_id, artifact_sha256, http_status
           ) VALUES (2024, 'no-txs', 'test-house-generation', ?, 200)""",
        [digest],
    )
    connection.close()
    gemini_ocr_common.write_cached_response(
        "no-txs",
        pdf,
        "MEMBER: Jane Doe\nPAGES: 1\nPAGE: 1\nNO_TRANSACTIONS",
        cache_dir=str(tmp_path / "gemini_cache"),
    )
    assert [
        item[0]
        for item in get_ocr_work_items(
            db_path=str(db_path), data_dir=tmp_path, year=2024
        )
    ] == ["no-txs"]

    assert (
        _insert_transactions(
            "no-txs",
            2024,
            "Jane Doe",
            [],
            db_path=str(db_path),
            raw_count=0,
            artifact_sha256=digest,
        )
        == 0
    )
    connection = duckdb.connect(str(db_path))
    rows = connection.execute(
        "SELECT source, COUNT(*) FROM transactions WHERE doc_id='no-txs' "
        "GROUP BY source ORDER BY source"
    ).fetchall()
    latest = connection.execute(
        "SELECT status, raw_row_count, transaction_count FROM pdf_parse_runs "
        "WHERE doc_id='no-txs' ORDER BY parsed_at DESC LIMIT 1"
    ).fetchone()
    connection.close()
    assert rows == [("capitol_trades", 1), ("gemini_ocr", 2)]
    assert latest == ("zero_rows", 0, 0)
    assert [item[0] for item in get_ocr_work_items(db_path=str(db_path), data_dir=tmp_path, year=2024)] == ['no-txs']

    assert (
        _insert_transactions(
            "no-txs", 2024, "Jane Doe", [], db_path=str(db_path), raw_count=0, artifact_sha256=digest
        )
        == 0
    )
    assert [item[0] for item in get_ocr_work_items(db_path=str(db_path), data_dir=tmp_path, year=2024)] == ['no-txs']


def test_work_selection_uses_current_db_not_progress(tmp_path):
    from analyzer.ocr.ingestion import get_ocr_work_items

    db_path = tmp_path / "congress.duckdb"
    db = Database(db_path)
    _enable_ocr_schema(db.conn)
    for doc_id in ("retry", "done", "rejected"):
        db.conn.execute(
            "INSERT INTO metadata (doc_id, first_name, last_name, filing_date, filing_type, fetched_at) VALUES (?, 'Jane', 'Doe', TIMESTAMP '2024-01-20', 'P', CURRENT_TIMESTAMP)",
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
        """INSERT INTO house_archive_generations (
               archive_year, generation_id, metadata_sha256,
               metadata_http_status, metadata_count, ptr_count, parse_status
           ) VALUES (2024, 'test-house-generation', 'metadata-sha', 200, 3, 3, 'incomplete')"""
    )
    db.conn.execute(
        """INSERT INTO house_pdf_artifacts (
               archive_year, doc_id, generation_id, artifact_sha256, http_status
           ) VALUES (2024, 'done', 'test-house-generation', ?, 200)""",
        [digest],
    )
    db.conn.execute(
        """UPDATE pdf_parse_runs
           SET artifact_sha256 = ?, ingestion_generation = 'test-house-generation'
           WHERE doc_id = 'done' AND parser_version = ?""",
        [digest, gemini_ocr_common.GEMINI_PARSER_VERSION],
    )
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
        ["test-house-generation", digest],
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

    connection = duckdb.connect(str(db_path))
    connection.execute(
        """INSERT INTO transactions (
               doc_id, member, ticker, transaction_date, disclosure_date,
               transaction_type, amount_raw, source, chamber, source_record_id,
               source_row_id, ingestion_generation, artifact_sha256
           ) VALUES (
               'done', 'Jane Doe', 'OLD', DATE '2024-01-01', DATE '2024-01-20',
               'Purchase', 'A', 'gemini_ocr', 'House', 'done',
               'old-v4-row', 'v4-gemini', 'old-sha'
           )"""
    )
    connection.execute(
        """INSERT INTO transactions (
               doc_id, member, ticker, transaction_date, disclosure_date,
               transaction_type, amount_raw, source, chamber, source_record_id,
               source_row_id, ingestion_generation, artifact_sha256
           ) VALUES
               ('done', 'Jane Doe', 'NULLGEN', DATE '2024-01-02', DATE '2024-01-20',
                'Purchase', 'A', 'gemini_ocr', NULL, NULL, 'old-null-row', NULL, 'old-sha'),
               ('done', 'Jane Doe', 'CAP', DATE '2024-01-03', DATE '2024-01-20',
                'Purchase', 'A', 'capitol_trades', 'House', 'done', 'cap-row', 'v1', 'cap-sha')
        """
    )
    connection.close()
    work = get_ocr_work_items(db_path=str(db_path), data_dir=tmp_path, year=2024)
    assert [item[0] for item in work] == ["done", "rejected", "retry"]

    parsed = gemini_ocr_common.parse_gemini_output(
        "MEMBER: Jane Doe\nPAGES: 1\nPAGE: 1\n"
        "Apple Inc. (AAPL) | Purchase | 01/15/24 | 01/20/24 | A"
    )
    assert (
        _insert_transactions(
            "done",
            2024,
            parsed.member,
            parsed.transactions,
            db_path=str(db_path),
            artifact_sha256=digest,
        )
        == 0
    )
    connection = duckdb.connect(str(db_path))
    counts = connection.execute(
        "SELECT source, ingestion_generation, COUNT(*) FROM transactions "
        "WHERE source_record_id='done' GROUP BY source, ingestion_generation "
        "ORDER BY source, ingestion_generation"
    ).fetchall()
    connection.close()
    assert counts == [
        ("capitol_trades", "v1", 1),
        ("gemini_ocr", "test-house-generation", 1),
        ("gemini_ocr", "v4-gemini", 1),
    ]
    work = get_ocr_work_items(db_path=str(db_path), data_dir=tmp_path, year=2024)
    assert [item[0] for item in work] == ["done", "rejected", "retry"]

    connection = duckdb.connect(str(db_path))
    connection.execute(
        "UPDATE transactions SET raw_asset_description='tampered' WHERE doc_id='done'"
    )
    connection.close()
    work = get_ocr_work_items(db_path=str(db_path), data_dir=tmp_path, year=2024)
    assert [item[0] for item in work] == ["done", "rejected", "retry"]
