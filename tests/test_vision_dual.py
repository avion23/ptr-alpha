"""Dual vision read: engine-B client, agreement gate, and persistence.

Every case runs against a local ephemeral-port stub HTTP server and a
temporary DuckDB; engine A is a stub callable. No real model calls and
``data/congress.duckdb`` is never opened.
"""

from __future__ import annotations

import datetime
import json
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import duckdb
import pytest

from analyzer.database import Database, _HOUSE_PARSE_SOURCE
from analyzer.vision_agreement import (
    AGREE,
    DISAGREE,
    disagreement_notes,
    gate_rows,
    normalize_date_value,
)
from analyzer.vision_engines import (
    DUAL_ENGINES_ATTEMPTED,
    DUAL_PARSER_VERSION,
    A_ERROR,
    GoogleVisionBRateLimitError,
    GoogleVisionBServiceUnavailableError,
    UNRESOLVED,
    DualReadDoc,
    TransientVisionBError,
    TruncatedCompletion,
    VisionBAuthError,
    VisionBConfig,
    VisionBPermanentError,
    VisionBQuotaError,
    VisionBUnavailable,
    dual_read,
    dual_read_enabled,
    read_document_with_vision_b,
    run_dual_read_for_doc,
)
from scripts import gemini_ocr_common as gem
from scripts import ocr_zero_rows
from scripts.ocr_zero_rows import get_ocr_work_items

A_OUTPUT = (
    "MEMBER: Jane Doe\nPAGES: 1\nPAGE: 1\n"
    "Apple Inc. (AAPL) | Purchase | 01/15/24 | 01/20/24 | A\n"
)
# Same reading, different surface form: casing, whitespace, date style.
B_OUTPUT_REFORMAT = (
    "MEMBER: Jane Doe\nPAGES: 1\nPAGE: 1\n"
    "  apple inc. (aapl)  | purchase | 1/15/24 | 01/20/24 | a\n"
)
B_OUTPUT_DISAGREE = A_OUTPUT.replace("| A\n", "| B\n")
NO_TXS_OUTPUT = "MEMBER: Jane Doe\nPAGES: 1\nPAGE: 1\nNO_TRANSACTIONS\n"
_TINY_PNG = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
    "AAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)
DATA_URL = "data:image/png;base64," + _TINY_PNG

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
        "ON transactions(source, chamber, source_record_id, source_row_id, "
        "ingestion_generation)"
    )


def _seed_db(db_path, *, sha="dual-sha-1", doc_id="dual-doc"):
    db_path = Path(db_path) / "congress.duckdb"
    db = Database(db_path)
    _enable_ocr_schema(db.conn)
    db.conn.execute(
        """
        INSERT INTO metadata (doc_id, first_name, last_name, filing_date,
                              filing_type, fetched_at)
        VALUES (?, 'Jane', 'Doe', TIMESTAMP '2024-01-20', 'P', CURRENT_TIMESTAMP)
        """,
        [doc_id],
    )
    db.conn.execute(
        "INSERT INTO house_archive_generations (archive_year, generation_id) "
        "VALUES (2024, 'gen-dual') ON CONFLICT DO NOTHING"
    )
    db.conn.execute(
        "INSERT INTO house_pdf_artifacts (archive_year, generation_id, doc_id, "
        "artifact_sha256) VALUES (2024, 'gen-dual', ?, ?) ON CONFLICT DO NOTHING",
        [doc_id, sha],
    )
    db.conn.execute(
        "INSERT INTO house_generation_metadata SELECT 2024, 'gen-dual', doc_id, "
        "first_name, last_name, filing_date, filing_type, fetched_at FROM metadata"
    )
    db.close()


def _read_a(output=A_OUTPUT, sha="dual-sha-1", error=None):
    """Stub engine A with the call_gemini contract (output, error, metadata)."""

    def read_a(doc, *, refresh=False):
        del doc, refresh
        if error is not None:
            return None, error, None
        parsed = gem.parse_gemini_output(output, expected_page_count=1)
        return output, "", gem.ArtifactMetadata(sha, 1, len(output), "stub.pdf", parsed)

    return read_a


def _pages(path, count):
    del path
    return [DATA_URL] * count


def _doc(tmp_path, *, doc_id="dual-doc", sha="dual-sha-1"):
    return DualReadDoc(
        doc_id,
        2024,
        "stub.pdf",
        data_dir=str(tmp_path),
        db_path=str(Path(tmp_path) / "congress.duckdb"),
    )


class _StubVisionServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, content, status=200, sleep_seconds=0.0):
        self.content = content
        self.status = status
        self.sleep_seconds = sleep_seconds
        self.requests = []
        super().__init__(("127.0.0.1", 0), _StubHandler)


class _StubHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        server = self.server
        length = int(self.headers.get("Content-Length") or 0)
        server.requests.append(json.loads(self.rfile.read(length) or b"{}"))
        if server.sleep_seconds:
            time.sleep(server.sleep_seconds)
        body = json.dumps(
            {"choices": [{"message": {"content": server.content}}]}
        ).encode()
        self.send_response(server.status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def vision_b_server():
    started = []

    def start(content, **kwargs):
        server = _StubVisionServer(content, **kwargs)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        started.append(server)
        return server

    yield start
    for server in started:
        server.shutdown()
        server.server_close()


def _env(server, timeout=None):
    env = {
        "VISION_B_BASE_URL": f"http://127.0.0.1:{server.server_address[1]}",
        "VISION_B_MODEL": "stub-qwen-vl",
    }
    if timeout is not None:
        env["VISION_B_TIMEOUT"] = str(timeout)
    return env


def _mock_http_response(monkeypatch, status, body):
    from analyzer import vision_engines

    class _Response:
        status_code = status
        text = body

    monkeypatch.setattr(
        vision_engines.requests, "post", lambda *args, **kwargs: _Response()
    )


def _dead_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


# ---------------------------------------------------------------------------
# Unit: normalization and gate
# ---------------------------------------------------------------------------


def test_normalize_date_value_handles_iso_slash_and_missing_tokens():
    assert normalize_date_value("1/5/24") == "2024-01-05"
    assert normalize_date_value("01-05-2024") == "2024-01-05"
    assert normalize_date_value("2024-01-05") == "2024-01-05"
    assert normalize_date_value(None) is None
    assert normalize_date_value("N/A") is None
    # Unparsable garbage keeps its raw text so two wrong readings never match.
    assert normalize_date_value("13/45/24") == "13/45/24"


def test_gate_agrees_after_normalization():
    rows_a = [
        {
            "asset": "Apple  Inc. (AAPL)",
            "type": "Purchase",
            "date": "01/15/24",
            "notif_date": "01/20/24",
            "amount_letter": "a",
        }
    ]
    rows_b = [
        {
            "asset": " apple inc. (aapl) ",
            "type": "PURCHASE",
            "date": "1/15/2024",
            "notif_date": "1/20/2024",
            "amount_letter": "A",
        }
    ]
    assert gate_rows(rows_a, rows_b).status == AGREE


def test_gate_page_attribution_does_not_block_agreement():
    row_a = {"asset": "Apple Inc.", "type": "Sale", "date": "01/15/24",
             "notif_date": None, "amount_letter": "A", "page_number": 2}
    row_b = {"asset": "apple inc.", "type": "sale", "date": "2024-01-15",
             "notif_date": None, "amount_letter": "A", "page_number": 3}
    assert gate_rows([row_a], [row_b]).status == AGREE


def test_gate_detects_amount_and_multiset_differences():
    row = {"asset": "Apple Inc.", "type": "Purchase", "date": "01/15/24",
           "notif_date": None, "amount_letter": "A"}
    bumped = dict(row, amount_letter="B")
    result = gate_rows([row], [bumped])
    assert result.status == DISAGREE
    assert "only_in_a=1" in result.diff_summary
    assert "only_in_b=1" in result.diff_summary
    # A duplicated row is a multiset difference, not a set difference.
    assert gate_rows([row, row], [row]).status == DISAGREE


def test_disagreement_notes_fit_error_column_and_stay_valid_json():
    rows = [
        {
            "asset": f"Intermediate Fund {index} Institutional Shares",
            "type": "Purchase",
            "date": "01/15/24",
            "notif_date": "01/20/24",
            "amount_letter": "A",
            "page_number": 1,
        }
        for index in range(120)
    ]
    notes = disagreement_notes(
        rows,
        rows[:119],
        reason="row_mismatch",
        engines_attempted=DUAL_ENGINES_ATTEMPTED,
        diff_summary="only_in_a=1[fund|purchase|...]; only_in_b=0[]",
    )
    assert len(notes) <= 1000  # record_parse_run truncates error_message at 1000
    parsed = json.loads(notes)
    assert parsed["truncated"] >= 1
    assert parsed["reading_a"] and parsed["reading_b"]


# ---------------------------------------------------------------------------
# Unit: engine-B client against the local stub endpoint
# ---------------------------------------------------------------------------


def test_vision_b_client_posts_pages_and_the_shared_prompt(vision_b_server):
    server = vision_b_server(B_OUTPUT_REFORMAT)
    config = VisionBConfig.from_env(_env(server))

    text = read_document_with_vision_b([DATA_URL], config)

    assert text == B_OUTPUT_REFORMAT
    payload = server.requests[0]
    assert payload["model"] == "stub-qwen-vl"
    assert payload["temperature"] == 0
    content = payload["messages"][0]["content"]
    assert content[0]["image_url"]["url"] == DATA_URL
    # The SAME transaction-extraction prompt as the Gemini path, never forked.
    assert content[-1]["text"] == gem.PROMPT


def test_vision_b_unavailable_on_connection_error():
    config = VisionBConfig(
        base_url=f"http://127.0.0.1:{_dead_port()}", model="stub", timeout=2
    )
    with pytest.raises(TransientVisionBError):
        read_document_with_vision_b([DATA_URL], config)


def test_vision_b_http_500_is_transient_and_keeps_body_snippet(vision_b_server):
    body = "upstream overload: " + "x" * 500
    server = vision_b_server(body, status=500)
    with pytest.raises(TransientVisionBError, match="500") as excinfo:
        read_document_with_vision_b([DATA_URL], VisionBConfig.from_env(_env(server)))
    # The non-200 body is retained (bounded) so classification survives.
    message = str(excinfo.value)
    assert "upstream overload" in message
    assert len(message) < 700  # bounded snippet, not the raw 500-char body


def test_vision_b_http_429_and_zai_1302_are_quota_stops(vision_b_server):
    server = vision_b_server("ignored", status=429)
    with pytest.raises(VisionBQuotaError, match="429"):
        read_document_with_vision_b([DATA_URL], VisionBConfig.from_env(_env(server)))

    # Z.AI's 1302 "Rate limit reached for requests" body is quota even when
    # a gateway surfaces it under a different status code.
    server1302 = vision_b_server(
        '{"error":{"code":"1302","message":"Rate limit reached for requests"}}',
        status=503,
    )
    with pytest.raises(VisionBQuotaError, match="1302"):
        read_document_with_vision_b(
            [DATA_URL], VisionBConfig.from_env(_env(server1302))
        )


def test_google_rate_shaped_429_is_transient_not_a_quota_stop(monkeypatch):
    from analyzer import vision_engines as ve

    _mock_http_response(
        monkeypatch,
        429,
        '{"error":{"status":"RESOURCE_EXHAUSTED","message":"Quota exceeded '
        'for quota metric GenerateContentRequestsPerDay; per minute limit"}}',
    )
    config = VisionBConfig(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        model="gemini-3-flash-preview",
    )

    with pytest.raises(GoogleVisionBRateLimitError) as excinfo:
        read_document_with_vision_b([DATA_URL], config)

    assert isinstance(excinfo.value, TransientVisionBError)
    assert not isinstance(excinfo.value, VisionBQuotaError)
    assert ve.classify_visionb_error(excinfo.value) == "transient"


def test_google_503_is_transient_endpoint_saturation(monkeypatch):
    from analyzer import vision_engines as ve

    _mock_http_response(
        monkeypatch, 503, "UNAVAILABLE: model currently experiencing high demand"
    )
    config = VisionBConfig(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        model="gemini-3.1-flash-lite",
    )

    with pytest.raises(GoogleVisionBServiceUnavailableError) as excinfo:
        read_document_with_vision_b([DATA_URL], config)

    assert isinstance(excinfo.value, TransientVisionBError)
    assert ve.classify_visionb_error(excinfo.value) == "transient"


def test_zai_host_429_remains_a_quota_stop(monkeypatch):
    _mock_http_response(monkeypatch, 429, "Rate limit reached for requests")
    config = VisionBConfig(base_url="https://api.z.ai/v1", model="glm-5.3-flash")

    with pytest.raises(VisionBQuotaError):
        read_document_with_vision_b([DATA_URL], config)


def test_zai_host_503_keeps_generic_transient_classification(monkeypatch):
    _mock_http_response(monkeypatch, 503, "UNAVAILABLE: upstream busy")
    config = VisionBConfig(base_url="https://api.z.ai/v1", model="glm-5.3-flash")

    with pytest.raises(TransientVisionBError) as excinfo:
        read_document_with_vision_b([DATA_URL], config)

    assert type(excinfo.value) is TransientVisionBError


@pytest.mark.parametrize(
    ("status", "body", "exception"),
    [
        (401, "invalid API key", VisionBAuthError),
        (403, "permission denied", VisionBAuthError),
        (500, "Internal Server Error", TransientVisionBError),
    ],
)
def test_google_non_429_http_classification_is_unchanged(
    monkeypatch, status, body, exception
):
    from analyzer import vision_engines as ve

    _mock_http_response(monkeypatch, status, body)
    config = VisionBConfig(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        model="gemini-3-flash-preview",
    )

    with pytest.raises(exception) as excinfo:
        read_document_with_vision_b([DATA_URL], config)

    assert ve.classify_visionb_error(excinfo.value) in {
        "auth",
        "transient",
    }


def test_vision_b_http_401_is_auth_and_400_is_permanent(vision_b_server):
    server = vision_b_server("invalid api key", status=401)
    with pytest.raises(VisionBAuthError, match="401"):
        read_document_with_vision_b([DATA_URL], VisionBConfig.from_env(_env(server)))

    server400 = vision_b_server("bad image payload", status=400)
    with pytest.raises(VisionBPermanentError, match="400") as excinfo:
        read_document_with_vision_b(
            [DATA_URL], VisionBConfig.from_env(_env(server400))
        )
    assert "bad image payload" in str(excinfo.value)


def test_vision_b_truncated_completion_is_rejected():
    # finish_reason="length" means the completion was cut off mid-answer:
    # reject it as TruncatedCompletion instead of parsing a partial reading.
    class _LengthHandler(_StubHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            json.loads(self.rfile.read(length) or b"{}")
            body = json.dumps(
                {
                    "choices": [
                        {
                            "message": {"content": "MEMBER: Jane Doe\nPAGES: 1\nPA"},
                            "finish_reason": "length",
                        }
                    ]
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    class _LengthServer(ThreadingHTTPServer):
        daemon_threads = True

        def __init__(self):
            super().__init__(("127.0.0.1", 0), _LengthHandler)

    length_server = _LengthServer()
    threading.Thread(target=length_server.serve_forever, daemon=True).start()
    try:
        with pytest.raises(TruncatedCompletion, match="finish_reason=length"):
            read_document_with_vision_b(
                [DATA_URL], VisionBConfig.from_env(_env(length_server))
            )
    finally:
        length_server.shutdown()
        length_server.server_close()


def test_vision_b_unavailable_on_timeout(vision_b_server):
    server = vision_b_server(B_OUTPUT_REFORMAT, sleep_seconds=2)
    env = _env(server, timeout=0.2)
    with pytest.raises(TransientVisionBError):
        read_document_with_vision_b([DATA_URL], VisionBConfig.from_env(env))


def test_vision_b_unavailable_on_malformed_envelope(monkeypatch):
    from analyzer import vision_engines

    class _BrokenResponse:
        status_code = 200

        @staticmethod
        def json():
            return {"choices": []}

    monkeypatch.setattr(
        vision_engines.requests, "post", lambda *args, **kwargs: _BrokenResponse()
    )
    with pytest.raises(TransientVisionBError, match="envelope"):
        read_document_with_vision_b(
            [DATA_URL], VisionBConfig(base_url="http://127.0.0.1:1", model="stub")
        )


# ---------------------------------------------------------------------------
# Interface hardening (2026-10-18 audit): output budget, per-page timeout,
# documented Z.AI error taxonomy, finish-reason gate, request shape
# ---------------------------------------------------------------------------


def test_request_max_tokens_scales_with_pages_within_documented_ceiling():
    from analyzer import vision_engines as ve

    # Documented request-schema maximum (docs.z.ai chat-completion):
    # glm-5.3-flash vision max_tokens min 1 / max 131072.
    assert ve.ZAI_MAX_TOKENS_CEILING == 131072
    # The floor covers the owner's flat-8192 bound with always-on reasoning
    # headroom (reasoning tokens share the output budget on glm-5.3-flash).
    assert ve.request_max_tokens(1) == 16384
    assert ve.request_max_tokens(None) == ve.request_max_tokens(1)
    # ...and scales past the floor for heavy filings (68-row docs are real:
    # 68 rows x 60 tokens ~ 4K answer tokens; 1500/page bounds 20 rows/page).
    assert ve.request_max_tokens(9) == 4096 + 1500 * 9
    assert (
        ve.request_max_tokens(9)
        > ve.request_max_tokens(3)
        >= ve.request_max_tokens(1)
    )
    assert ve.request_max_tokens(1000) == 131072  # capped at the schema max


def test_request_timeout_math_is_per_page():
    from analyzer import vision_engines as ve

    assert ve.request_timeout(1) == 180.0  # equals the old flat budget
    assert ve.request_timeout(2) == 240.0
    assert ve.request_timeout(3) == 300.0  # tonight's heavy-doc failure case
    assert ve.request_timeout(5) == 420.0
    assert ve.request_timeout(None) == ve.request_timeout(1)
    # An operator-pinned VISION_B_TIMEOUT always wins verbatim (auto == None).
    assert ve.request_timeout(3, configured=600.0) == 600.0
    assert ve.request_timeout(3, configured=0.2) == 0.2


def test_client_sends_max_tokens_and_per_page_timeout(monkeypatch):
    from analyzer import vision_engines as ve

    captured = {}

    class _OKResponse:
        status_code = 200
        text = "ok"

        @staticmethod
        def json():
            return {
                "choices": [
                    {
                        "message": {
                            "content": "MEMBER: Jane Doe\nPAGES: 1\nPAGE: 1\nNO_TRANSACTIONS\n"
                        }
                    }
                ]
            }

    def fake_post(url, *, json=None, headers=None, timeout=None):
        captured["payload"] = json
        captured["timeout"] = timeout
        return _OKResponse()

    monkeypatch.setattr(ve.requests, "post", fake_post)
    pages = [DATA_URL] * 4
    read_document_with_vision_b(
        pages, VisionBConfig(base_url="http://visionb.invalid", model="stub")
    )
    assert captured["payload"]["max_tokens"] == ve.request_max_tokens(4)
    assert captured["timeout"] == ve.request_timeout(4)  # auto: 120s + 60s/page
    read_document_with_vision_b(
        pages,
        VisionBConfig(
            base_url="http://visionb.invalid", model="stub", timeout=0.5
        ),
    )
    assert captured["timeout"] == 0.5  # an operator pin is honored verbatim


def test_finish_reason_fail_closed_taxonomy(monkeypatch):
    from analyzer import vision_engines as ve

    payload = (
        "MEMBER: Jane Doe\nPAGES: 1\nPAGE: 1\n"
        "Apple Inc. (AAPL) | Purchase | 01/15/24 | 01/20/24 | A\n"
    )

    def respond_with(finish_reason):
        class _Response:
            status_code = 200
            text = "ok"

            @staticmethod
            def json():
                return {
                    "choices": [
                        {
                            "message": {"content": payload},
                            "finish_reason": finish_reason,
                        }
                    ]
                }

        monkeypatch.setattr(
            ve.requests, "post", lambda *args, **kwargs: _Response()
        )

    config = VisionBConfig(base_url="http://visionb.invalid", model="stub")
    # A documented abnormal termination with a plausible body is NOT a
    # reading: parsing it would let a filtered/window-exceeded cut masquerade
    # as a (possibly partial) extraction.
    respond_with("sensitive")
    with pytest.raises(VisionBPermanentError, match="finish_reason=sensitive"):
        read_document_with_vision_b([DATA_URL], config)
    respond_with("model_context_window_exceeded")
    with pytest.raises(VisionBPermanentError, match="context_window"):
        read_document_with_vision_b([DATA_URL], config)
    # Transport-shaped abnormal terminations stay retriable.
    respond_with("network_error")
    with pytest.raises(
        TransientVisionBError, match="finish_reason=network_error"
    ):
        read_document_with_vision_b([DATA_URL], config)
    # "stop" — and no finish_reason at all — remain readings.
    respond_with("stop")
    assert read_document_with_vision_b([DATA_URL], config) == payload
    respond_with(None)
    assert read_document_with_vision_b([DATA_URL], config) == payload


@pytest.mark.parametrize(
    ("status", "body", "kind"),
    [
        # "-"-code / 500 "Internal Error" (seen live tonight): transient.
        (500, "Internal Error", "transient"),
        (500, '{"error":{"code":"1200","message":"API Call Error"}}', "transient"),
        (500, '{"error":{"code":"1230","message":"API call process error"}}', "transient"),
        # 1302 quota, under its documented status and under a foreign status.
        (429, "Rate limit reached for requests", "quota"),
        (503, '{"error":{"code":"1302","message":"Rate limit reached for requests"}}', "quota"),
        # 1113 "Insufficient balance or no resource package" is quota/balance.
        (429, '{"error":{"code":"1113","message":"Insufficient balance or no resource package. Please recharge."}}', "quota"),
        (402, '{"error":{"code":"1113","message":"Insufficient balance or no resource package. Please recharge."}}', "quota"),
        # 1305 is a TRANSIENT overload, not a quota stop.
        (429, '{"error":{"code":"1305","message":"The service may be temporarily overloaded, please try again later"}}', "transient"),
        # 1309/1311 never self-heal: fatal, alert-shaped (auth bucket).
        (429, '{"error":{"code":"1309","message":"Your GLM Coding Plan package has expired and is temporarily unavailable."}}', "auth"),
        (429, '{"error":{"code":"1311","message":"Your current subscription plan does not yet include access to glm-5.3-flash"}}', "auth"),
        # 1000-1005 auth failures, 1220 permission.
        (401, '{"error":{"code":"1002","message":"Invalid API key"}}', "auth"),
        (401, '{"error":{"code":"1000","message":"Authentication Failed"}}', "auth"),
        (403, '{"error":{"code":"1220","message":"You do not have permission to access this API"}}', "auth"),
        # 1211/1261/1301 are permanent request defects (400).
        (400, '{"error":{"code":"1261","message":"Prompt too long"}}', "permanent"),
        (400, '{"error":{"code":"1211","message":"Unknown Model"}}', "permanent"),
        (400, '{"error":{"code":"1301","message":"sensitive content detected"}}', "permanent"),
    ],
)
def test_documented_zai_error_codes_classify(status, body, kind):
    from analyzer import vision_engines as ve

    exc = ve._classify_http_error(status, body)
    assert ve.classify_visionb_error(exc) == kind


def test_multi_page_images_travel_in_order_in_the_documented_shape(
    vision_b_server,
):
    server = vision_b_server(A_OUTPUT)
    config = VisionBConfig.from_env(_env(server))
    page_urls = [
        "data:image/png;base64,UEEx",
        "data:image/png;base64,UEEy",
        "data:image/png;base64,UEEz",
    ]
    read_document_with_vision_b(page_urls, config)
    content = server.requests[0]["messages"][0]["content"]
    images = [part for part in content if part.get("type") == "image_url"]
    # Pages stay in order, images first, prompt last — Z.AI's documented
    # multi-image form is repeated image_url content blocks.
    assert [part["image_url"]["url"] for part in images] == page_urls
    assert content[-1]["type"] == "text"


def test_render_dpis_stay_inside_documented_zai_image_caps():
    # Documented per-image caps (vision schema): <5MB and <= 6000x6000 px.
    # 300 DPI on letter is 2550x3300 — inside; 600 DPI would NOT be.
    from analyzer import vision_engines
    from scripts import glm_stage

    for dpi in (glm_stage.RENDER_DPI, vision_engines.PAGE_RENDER_DPI, 300):
        assert 8.5 * dpi <= 6000
        assert 11.0 * dpi <= 6000


# ---------------------------------------------------------------------------
# Orchestrator: dual_read decisions
# ---------------------------------------------------------------------------


def test_dual_read_agree_records_v6_decision():
    decision, rows = dual_read(
        DualReadDoc("dual-doc", 2024, "stub.pdf"),
        read_a=_read_a(),
        read_b=lambda images: B_OUTPUT_REFORMAT,
        page_images=_pages,
        env={},
    )
    assert decision.status == AGREE
    assert decision.parser_version == DUAL_PARSER_VERSION
    assert decision.engines_attempted == DUAL_ENGINES_ATTEMPTED
    assert decision.raw_row_count == 1
    assert [row["asset"] for row in rows] == ["Apple Inc. (AAPL)"]


def test_dual_read_unconfigured_engine_b_is_unresolved_not_a_only():
    def fail_images(path, count):
        raise AssertionError("page rasterization must not run without engine B")

    decision, rows = dual_read(
        DualReadDoc("dual-doc", 2024, "stub.pdf"),
        read_a=_read_a(),
        page_images=fail_images,
        env={},
    )
    # "Both readers must agree" holds during outages too: B unconfigured is
    # a config defect leaving the doc UNRESOLVED, never an A-only reading.
    assert decision.status == UNRESOLVED
    assert "not configured" in decision.diff_summary
    assert decision.output is None  # A's reading is not handed out
    assert rows == []


def test_dual_read_engine_b_down_is_unresolved_no_rows():
    env = {
        "VISION_B_BASE_URL": f"http://127.0.0.1:{_dead_port()}",
        "VISION_B_MODEL": "stub-qwen-vl",
    }
    decision, rows = dual_read(
        DualReadDoc("dual-doc", 2024, "stub.pdf"),
        read_a=_read_a(),
        page_images=_pages,
        env=env,
    )
    assert decision.status == UNRESOLVED  # engine absent is unresolved, not agreement
    assert decision.error is None
    assert decision.output is None  # nothing A-only may persist from this
    assert rows == []


def test_dual_read_engine_a_error_is_passthrough():
    decision, rows = dual_read(
        DualReadDoc("dual-doc", 2024, "stub.pdf"),
        read_a=_read_a(error="llm exited 1"),
        page_images=_pages,
        env={},
    )
    assert decision.status == A_ERROR
    assert decision.error == "llm exited 1"
    assert rows == []


def test_dual_read_opt_in_flag_default_off():
    assert dual_read_enabled({}) is False
    assert dual_read_enabled({"DUAL_READ": "0"}) is False
    assert dual_read_enabled({"DUAL_READ": "1"}) is True


# ---------------------------------------------------------------------------
# Persistence through run_dual_read_for_doc (temporary DuckDB)
# ---------------------------------------------------------------------------


def test_dual_read_agreement_persists_rows_as_v6(tmp_path, vision_b_server):
    _seed_db(tmp_path)
    server = vision_b_server(B_OUTPUT_REFORMAT)

    outcome = run_dual_read_for_doc(
        _doc(tmp_path), env=_env(server), read_a=_read_a(), page_images=_pages
    )

    assert outcome.handled
    assert outcome.inserted == 1
    assert outcome.progress_key == "completed"
    assert outcome.member == "Jane Doe"
    with duckdb.connect(str(tmp_path / "congress.duckdb")) as conn:
        rows = conn.execute(
            "SELECT transaction_type, amount_raw, source FROM transactions "
            "WHERE doc_id = 'dual-doc'"
        ).fetchall()
        run_row = conn.execute(
            "SELECT parser_version, status, engines_attempted, transaction_count "
            "FROM pdf_parse_runs WHERE doc_id = 'dual-doc'"
        ).fetchone()
    assert rows == [("Purchase", "A", "gemini_ocr")]
    assert run_row == (DUAL_PARSER_VERSION, "success", DUAL_ENGINES_ATTEMPTED, 1)


def test_dual_read_zero_row_agreement_records_no_txs(tmp_path, vision_b_server):
    _seed_db(tmp_path)
    server = vision_b_server(NO_TXS_OUTPUT)

    outcome = run_dual_read_for_doc(
        _doc(tmp_path),
        env=_env(server),
        read_a=_read_a(output=NO_TXS_OUTPUT),
        page_images=_pages,
    )

    assert outcome.handled
    assert outcome.inserted == 0
    assert outcome.progress_key == "no_txs"
    with duckdb.connect(str(tmp_path / "congress.duckdb")) as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM transactions WHERE doc_id = 'dual-doc'"
        ).fetchone()[0]
        status = conn.execute(
            "SELECT parser_version, status FROM pdf_parse_runs WHERE doc_id = 'dual-doc'"
        ).fetchone()
    assert count == 0
    assert status == (DUAL_PARSER_VERSION, "no_txs")


def test_dual_read_partial_insert_is_never_completed(
    tmp_path, monkeypatch, vision_b_server
):
    _seed_db(tmp_path)
    server = vision_b_server(B_OUTPUT_REFORMAT)  # both readers agree...

    def partial_insert(*args, **kwargs):
        # ...but validation kept out one of two staged rows: explicit
        # partial outcome from insert_transactions.
        report = kwargs.get("report")
        if report is not None:
            report["outcome"] = ocr_zero_rows.InsertOutcome(
                status="partial",
                inserted=1,
                expected=2,
                committed=True,
                rejections={"impossible_chronology": 1},
            )
        return 1

    monkeypatch.setattr(ocr_zero_rows, "insert_transactions", partial_insert)

    outcome = run_dual_read_for_doc(
        _doc(tmp_path), env=_env(server), read_a=_read_a(), page_images=_pages
    )

    assert outcome.handled
    assert outcome.inserted == 1
    # Never call partial output complete: the doc is marked for attention.
    assert outcome.progress_key == "errors"


def test_run_gemini_ocr_for_year_engine_b_quota_stop_ends_batch(
    tmp_path, monkeypatch, vision_b_server, capsys
):
    _seed_db_docs(tmp_path, list(_TWO_SHAS.items()))
    server = vision_b_server("ignored", status=429)
    _set_engine_b(monkeypatch, server.server_address[1])
    monkeypatch.setenv("DUAL_READ", "1")
    calls: list = []
    _stub_loop_engines(
        monkeypatch, outputs=_TWO_OUTPUTS, shas=_TWO_SHAS, calls=calls
    )
    monkeypatch.setattr(
        ocr_zero_rows, "get_ocr_work_items", lambda **kwargs: list(_TWO_DOCS)
    )
    monkeypatch.setattr(ocr_zero_rows, "COOLDOWN", 0)

    total = ocr_zero_rows.run_gemini_ocr_for_year(2024, data_dir=str(tmp_path))

    # The year runner honors the B-quota batch-stop exactly like main().
    assert total == 0
    assert calls == ["dual-doc-a"]
    assert "Engine B quota exhausted; stopping batch" in capsys.readouterr().out
    with duckdb.connect(str(tmp_path / "congress.duckdb")) as conn:
        assert conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM pdf_parse_runs").fetchone()[0] == 0


def test_run_gemini_ocr_for_year_dual_without_b_config_stops(
    tmp_path, monkeypatch, capsys
):
    _seed_db_docs(tmp_path, list(_TWO_SHAS.items()))
    monkeypatch.setenv("DUAL_READ", "1")
    monkeypatch.delenv("VISION_B_BASE_URL", raising=False)
    monkeypatch.delenv("VISION_B_MODEL", raising=False)
    calls: list = []
    _stub_loop_engines(
        monkeypatch, outputs=_TWO_OUTPUTS, shas=_TWO_SHAS, calls=calls
    )
    monkeypatch.setattr(
        ocr_zero_rows, "get_ocr_work_items", lambda **kwargs: list(_TWO_DOCS)
    )
    monkeypatch.setattr(ocr_zero_rows, "COOLDOWN", 0)

    total = ocr_zero_rows.run_gemini_ocr_for_year(2024, data_dir=str(tmp_path))

    assert total == 0
    assert calls == []  # engine A never spent on ungated readings
    out = capsys.readouterr().out
    assert "ALERT" in out and "not configured" in out


def test_dual_read_disagreement_persists_nothing_and_records_both_readings(
    tmp_path, vision_b_server
):
    _seed_db(tmp_path)
    server = vision_b_server(B_OUTPUT_DISAGREE)

    outcome = run_dual_read_for_doc(
        _doc(tmp_path), env=_env(server), read_a=_read_a(), page_images=_pages
    )

    assert outcome.handled
    assert outcome.inserted == 0
    assert outcome.progress_key == "errors"
    with duckdb.connect(str(tmp_path / "congress.duckdb")) as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM transactions WHERE doc_id = 'dual-doc'"
        ).fetchone()[0]
        run_row = conn.execute(
            "SELECT parser_version, status, engines_attempted, raw_row_count, "
            "transaction_count, error_message FROM pdf_parse_runs "
            "WHERE doc_id = 'dual-doc'"
        ).fetchone()
    assert count == 0  # nothing persists from either side
    parser_version, status, engines, raw_count, tx_count, message = run_row
    assert status == "invalid" and tx_count == 0 and raw_count == 1
    assert parser_version == DUAL_PARSER_VERSION
    assert engines == DUAL_ENGINES_ATTEMPTED
    assert len(message) <= 1000
    notes = json.loads(message)
    assert notes["reason"] == "row_mismatch"
    assert notes["member_a"] == "Jane Doe"
    assert notes["reading_a"] == [
        ["Apple Inc. (AAPL)", "Purchase", "01/15/24", "01/20/24", "A", 1]
    ]
    assert notes["reading_b"] == [
        ["Apple Inc. (AAPL)", "Purchase", "01/15/24", "01/20/24", "B", 1]
    ]
    assert "only_in_a=1" in notes["diff"] and "only_in_b=1" in notes["diff"]


def test_dual_read_schema_invalid_engine_b_lands_in_review_not_canonical(
    tmp_path, vision_b_server
):
    _seed_db(tmp_path)
    server = vision_b_server("hello world, no extraction schema here")

    outcome = run_dual_read_for_doc(
        _doc(tmp_path), env=_env(server), read_a=_read_a(), page_images=_pages
    )

    # B answered but broke the schema: fail closed into review, not A-only.
    assert outcome.handled and outcome.decision.status == DISAGREE
    assert outcome.inserted == 0
    with duckdb.connect(str(tmp_path / "congress.duckdb")) as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM transactions WHERE doc_id = 'dual-doc'"
        ).fetchone()[0]
        message = conn.execute(
            "SELECT error_message FROM pdf_parse_runs WHERE doc_id = 'dual-doc'"
        ).fetchone()[0]
    assert count == 0
    notes = json.loads(message)
    assert notes["reason"] == "visionb_schema_invalid"
    assert "no extraction schema" in notes["reading_b_raw"]


def test_dual_read_engine_down_lands_retry_state_persisting_nothing(tmp_path):
    _seed_db(tmp_path)
    _add_deterministic_zero_rows(tmp_path, ["dual-doc"])
    env = {
        "VISION_B_BASE_URL": f"http://127.0.0.1:{_dead_port()}",
        "VISION_B_MODEL": "stub-qwen-vl",
    }

    outcome = run_dual_read_for_doc(
        _doc(tmp_path), env=env, read_a=_read_a(), page_images=_pages
    )

    # Engine B absent is UNRESOLVED: the orchestrator persists NOTHING from
    # either side — no transactions, no v5 success run — and lands an error
    # run that keeps the doc on the OCR work list for the next run.
    assert outcome.handled
    assert outcome.decision.status == UNRESOLVED
    assert outcome.inserted == 0
    assert outcome.progress_key == "errors"
    db_file = str(Path(tmp_path) / "congress.duckdb")
    with duckdb.connect(db_file) as conn:
        tx_count = conn.execute(
            "SELECT COUNT(*) FROM transactions WHERE doc_id = 'dual-doc'"
        ).fetchone()[0]
        run_row = conn.execute(
            "SELECT parser_version, status, engines_attempted, error_message "
            "FROM pdf_parse_runs WHERE doc_id = 'dual-doc' "
            "AND parser_version = ? ORDER BY parsed_at DESC LIMIT 1",
            [DUAL_PARSER_VERSION],
        ).fetchone()
    assert tx_count == 0
    parser_version, status, engines, message = run_row
    assert status == "error"  # retry state, not a terminal review state
    assert engines == DUAL_ENGINES_ATTEMPTED
    assert "connection" in message or "refused" in message
    # The doc stays on the OCR work list: it must be retried, not dropped.
    work = get_ocr_work_items(db_path=db_file, data_dir=str(tmp_path), year=2024)
    assert [doc_id for doc_id, _, _ in work] == ["dual-doc"]


def test_dual_read_engine_b_quota_stops_without_persisting(tmp_path):
    _seed_db(tmp_path)

    def quota_read_b(images):
        raise VisionBQuotaError("visionb quota/rate-limit exhausted (http 429)")

    outcome = run_dual_read_for_doc(
        _doc(tmp_path), env={}, read_a=_read_a(), read_b=quota_read_b,
        page_images=_pages,
    )
    # Quota is returned UNHANDLED and nothing is written: the caller stops
    # the batch; the doc stays untouched for the next run.
    assert not outcome.handled
    assert outcome.decision.status == UNRESOLVED
    db_file = str(Path(tmp_path) / "congress.duckdb")
    with duckdb.connect(db_file) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM transactions WHERE doc_id = 'dual-doc'"
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM pdf_parse_runs WHERE doc_id = 'dual-doc'"
        ).fetchone()[0] == 0


def test_dual_read_google_rate_limit_lands_retry_state(tmp_path):
    _seed_db(tmp_path)

    def rate_limited_read_b(images):
        raise GoogleVisionBRateLimitError(
            "visionb Google rate limit transient (http 429): "
            "RESOURCE_EXHAUSTED quota exceeded for quota metric"
        )

    outcome = run_dual_read_for_doc(
        _doc(tmp_path),
        env={},
        read_a=_read_a(),
        read_b=rate_limited_read_b,
        page_images=_pages,
    )

    assert outcome.handled
    assert outcome.decision.error_kind == "transient"
    with duckdb.connect(str(Path(tmp_path) / "congress.duckdb")) as conn:
        row = conn.execute(
            "SELECT status, error_message FROM pdf_parse_runs WHERE doc_id = 'dual-doc'"
        ).fetchone()
    assert row[0] == "error"
    assert "RESOURCE_EXHAUSTED" in row[1]


def test_dual_run_classifies_as_gemini_ocr_source_family(tmp_path):
    db = Database(tmp_path / "classify.duckdb")
    db.conn.execute(
        """
        INSERT INTO pdf_parse_runs (doc_id, year, parser_version, status,
                                    engines_attempted, raw_row_count,
                                    transaction_count)
        VALUES ('d1', 2024, 'v6-dual-validated', 'success', 'gemini+visionb', 3, 3)
        """
    )
    source = db.conn.execute(
        f"SELECT {_HOUSE_PARSE_SOURCE} FROM pdf_parse_runs AS parse_run "
        "WHERE doc_id = 'd1'"
    ).fetchone()[0]
    db.close()
    assert source == "gemini_ocr"


# ---------------------------------------------------------------------------
# Nightly backlog cron: main() loop dual routing (stubbed engines, tmp DuckDB)
# ---------------------------------------------------------------------------


def _seed_db_docs(db_path, docs):
    """Seed one DB with several OCR-ready (doc_id, sha) docs for loop tests."""
    db_path = Path(db_path) / "congress.duckdb"
    db = Database(db_path)
    _enable_ocr_schema(db.conn)
    db.conn.execute(
        "INSERT INTO house_archive_generations (archive_year, generation_id) "
        "VALUES (2024, 'gen-dual') ON CONFLICT DO NOTHING"
    )
    for doc_id, sha in docs:
        db.conn.execute(
            """
            INSERT INTO metadata (doc_id, first_name, last_name, filing_date,
                                  filing_type, fetched_at)
            VALUES (?, 'Jane', 'Doe', TIMESTAMP '2024-01-20', 'P', CURRENT_TIMESTAMP)
            """,
            [doc_id],
        )
        db.conn.execute(
            "INSERT INTO house_pdf_artifacts (archive_year, generation_id, doc_id, "
            "artifact_sha256) VALUES (2024, 'gen-dual', ?, ?) ON CONFLICT DO NOTHING",
            [doc_id, sha],
        )
        db.conn.execute(
            "INSERT INTO house_generation_metadata SELECT 2024, 'gen-dual', doc_id, "
            "first_name, last_name, filing_date, filing_type, fetched_at "
            "FROM metadata WHERE metadata.doc_id = ?",
            [doc_id],
        )
    db.close()


def _add_deterministic_zero_rows(db_path, doc_ids):
    """Give docs a deterministic zero_rows run so they sit on the OCR work list."""
    db_path = Path(db_path) / "congress.duckdb"
    db = Database(db_path)
    for index, doc_id in enumerate(doc_ids):
        db.conn.execute(
            "INSERT INTO pdf_parse_runs (doc_id, year, parser_version, status, "
            "engines_attempted, raw_row_count, transaction_count, parsed_at) "
            "VALUES (?, 2024, 'v4-house-deterministic', 'zero_rows', "
            "'house_pdf', 0, 0, ?)",
            [doc_id, datetime.datetime(2024, 2, 1, 12, 0, 0)
             + datetime.timedelta(minutes=index)],
        )
    db.close()


def _stub_loop_engines(monkeypatch, *, outputs, shas, calls, errors=None):
    """Stub engine A (per-doc output/error) and page rasterization."""
    from analyzer import vision_engines

    def read_a(doc, *, refresh=False):
        del refresh
        calls.append(doc.doc_id)
        error = (errors or {}).get(doc.doc_id)
        if error is not None:
            return None, error, None
        output = outputs[doc.doc_id]
        parsed = gem.parse_gemini_output(output, expected_page_count=1)
        return (
            output,
            "",
            gem.ArtifactMetadata(shas[doc.doc_id], 1, len(output), "stub.pdf", parsed),
        )

    monkeypatch.setattr(vision_engines, "_default_read_a", read_a)
    monkeypatch.setattr(vision_engines, "_default_page_data_urls", _pages)


def _wire_main(
    monkeypatch,
    tmp_path,
    *,
    docs,
    shas,
    outputs,
    errors=None,
    dual=True,
):
    """Point main() at a tmp DuckDB/progress file with stubbed engines.

    Returns (db_file, progress_path, calls). Engine A is stubbed at both call
    sites (vision_engines._default_read_a for the dual orchestrator and
    ocr_zero_rows.call_gemini for the single-engine path) so no test here can
    reach the network through either route.
    """
    calls: list = []
    _stub_loop_engines(
        monkeypatch, outputs=outputs, shas=shas, calls=calls, errors=errors
    )

    def fake_call_gemini(pdf_path, doc_id=None, refresh=False):
        del pdf_path, refresh
        calls.append(doc_id)
        error = (errors or {}).get(doc_id)
        if error is not None:
            return None, error, None
        output = outputs[doc_id]
        parsed = gem.parse_gemini_output(output, expected_page_count=1)
        return (
            output,
            "",
            gem.ArtifactMetadata(shas[doc_id], 1, len(output), "stub.pdf", parsed),
        )

    monkeypatch.setattr(ocr_zero_rows, "call_gemini", fake_call_gemini)
    db_file = Path(tmp_path) / "congress.duckdb"
    progress_path = tmp_path / "progress.json"
    monkeypatch.setattr(sys, "argv", ["ocr_zero_rows"])
    monkeypatch.setattr(ocr_zero_rows, "DB_PATH", str(db_file))
    monkeypatch.setattr(ocr_zero_rows, "COOLDOWN", 0)
    monkeypatch.setattr(ocr_zero_rows, "get_zero_row_pdfs", lambda: list(docs))
    original_load = ocr_zero_rows.load_progress
    original_save = ocr_zero_rows.save_progress
    monkeypatch.setattr(
        ocr_zero_rows, "load_progress", lambda: original_load(progress_path)
    )
    monkeypatch.setattr(
        ocr_zero_rows, "save_progress", lambda p: original_save(p, progress_path)
    )
    if dual:
        monkeypatch.setenv("DUAL_READ", "1")
    else:
        monkeypatch.delenv("DUAL_READ", raising=False)
    return db_file, progress_path, calls


def _set_engine_b(monkeypatch, port):
    monkeypatch.setenv("VISION_B_BASE_URL", f"http://127.0.0.1:{port}")
    monkeypatch.setenv("VISION_B_MODEL", "stub-qwen-vl")


_TWO_DOCS = [("dual-doc-a", 2024, "stub.pdf"), ("dual-doc-b", 2024, "stub.pdf")]
_TWO_SHAS = {"dual-doc-a": "dual-sha-a", "dual-doc-b": "dual-sha-b"}
_TWO_OUTPUTS = {"dual-doc-a": A_OUTPUT, "dual-doc-b": A_OUTPUT}


def test_main_loop_dual_agree_persists_v6_and_marks_processed(
    tmp_path, monkeypatch, vision_b_server, capsys
):
    _seed_db(tmp_path)
    server = vision_b_server(B_OUTPUT_REFORMAT)
    _set_engine_b(monkeypatch, server.server_address[1])
    db_file, progress_path, _ = _wire_main(
        monkeypatch,
        tmp_path,
        docs=[("dual-doc", 2024, "stub.pdf")],
        shas={"dual-doc": "dual-sha-1"},
        outputs={"dual-doc": A_OUTPUT},
    )

    ocr_zero_rows.main()

    with duckdb.connect(str(db_file)) as conn:
        run_row = conn.execute(
            "SELECT parser_version, status, engines_attempted FROM pdf_parse_runs "
            "WHERE doc_id = 'dual-doc'"
        ).fetchone()
        tx_count = conn.execute(
            "SELECT COUNT(*) FROM transactions WHERE doc_id = 'dual-doc'"
        ).fetchone()[0]
    assert run_row == (DUAL_PARSER_VERSION, "success", DUAL_ENGINES_ATTEMPTED)
    assert tx_count == 1
    assert json.loads(progress_path.read_text())["completed"] == ["dual-doc"]
    # The validated dual run counts as deterministic recency: the doc leaves
    # the nightly work list instead of being re-read.
    assert (
        get_ocr_work_items(db_path=str(db_file), data_dir=str(tmp_path), year=2024)
        == []
    )
    assert "DUAL AGREE: inserted=1" in capsys.readouterr().out


def test_main_loop_dual_disagree_marks_processed_without_persisting(
    tmp_path, monkeypatch, vision_b_server
):
    _seed_db(tmp_path)
    server = vision_b_server(B_OUTPUT_DISAGREE)
    _set_engine_b(monkeypatch, server.server_address[1])
    db_file, progress_path, _ = _wire_main(
        monkeypatch,
        tmp_path,
        docs=[("dual-doc", 2024, "stub.pdf")],
        shas={"dual-doc": "dual-sha-1"},
        outputs={"dual-doc": A_OUTPUT},
    )

    ocr_zero_rows.main()

    with duckdb.connect(str(db_file)) as conn:
        tx_count = conn.execute(
            "SELECT COUNT(*) FROM transactions WHERE doc_id = 'dual-doc'"
        ).fetchone()[0]
        run_row = conn.execute(
            "SELECT status, transaction_count FROM pdf_parse_runs "
            "WHERE doc_id = 'dual-doc'"
        ).fetchone()
    assert tx_count == 0  # nothing persists from either side
    assert run_row == ("invalid", 0)
    progress = json.loads(progress_path.read_text())
    assert progress["errors"] == ["dual-doc"]  # marked processed, not lost
    assert progress["completed"] == []
    # The invalid dual run leaves the work list: no nightly retry loop.
    assert (
        get_ocr_work_items(db_path=str(db_file), data_dir=str(tmp_path), year=2024)
        == []
    )


def test_main_loop_engine_b_quota_stop_ends_batch_cleanly(
    tmp_path, monkeypatch, vision_b_server, capsys
):
    _seed_db_docs(tmp_path, list(_TWO_SHAS.items()))
    server = vision_b_server("ignored", status=429)
    _set_engine_b(monkeypatch, server.server_address[1])
    db_file, progress_path, calls = _wire_main(
        monkeypatch, tmp_path, docs=_TWO_DOCS, shas=_TWO_SHAS, outputs=_TWO_OUTPUTS
    )

    ocr_zero_rows.main()  # returns cleanly: a quota stop is not a crash

    assert calls == ["dual-doc-a"]  # batch ended at the first doc
    out = capsys.readouterr().out
    assert "Engine B quota exhausted; stopping batch" in out
    assert "=== DONE ===" in out
    with duckdb.connect(str(db_file)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM pdf_parse_runs").fetchone()[0] == 0
    # Nothing was processed, so nothing was marked: no progress file at all.
    assert not progress_path.exists()


def test_main_loop_engine_a_quota_stop_still_ends_batch(
    tmp_path, monkeypatch, vision_b_server, capsys
):
    _seed_db_docs(tmp_path, list(_TWO_SHAS.items()))
    server = vision_b_server(B_OUTPUT_REFORMAT)  # B would agree, but is never asked
    _set_engine_b(monkeypatch, server.server_address[1])
    db_file, progress_path, calls = _wire_main(
        monkeypatch,
        tmp_path,
        docs=_TWO_DOCS,
        shas=_TWO_SHAS,
        outputs=_TWO_OUTPUTS,
        errors={"dual-doc-a": "429 RESOURCE_EXHAUSTED: quota exceeded"},
    )

    ocr_zero_rows.main()

    assert calls == ["dual-doc-a"]
    assert server.requests == []  # engine B is never called after A's quota stop
    assert "Quota exhausted; stopping batch" in capsys.readouterr().out
    with duckdb.connect(str(db_file)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM pdf_parse_runs").fetchone()[0] == 0
    assert not progress_path.exists()  # nothing processed, nothing marked


def test_main_loop_engine_b_transient_outage_lands_retry_state_not_a_only(
    tmp_path, monkeypatch
):
    _seed_db_docs(tmp_path, list(_TWO_SHAS.items()))
    _add_deterministic_zero_rows(tmp_path, [doc_id for doc_id, _ in _TWO_SHAS.items()])
    _set_engine_b(monkeypatch, _dead_port())  # connection refused, not a quota stop
    db_file, progress_path, calls = _wire_main(
        monkeypatch, tmp_path, docs=_TWO_DOCS, shas=_TWO_SHAS, outputs=_TWO_OUTPUTS
    )

    ocr_zero_rows.main()

    # The batch continues (an outage is not a crash) but NOTHING persists
    # A-only: each doc lands an error run that retries it next night.
    assert calls == ["dual-doc-a", "dual-doc-b"]
    with duckdb.connect(str(db_file)) as conn:
        tx_count = conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
        runs = conn.execute(
            "SELECT doc_id, parser_version, engines_attempted, status "
            "FROM pdf_parse_runs WHERE parser_version = ? ORDER BY doc_id",
            [DUAL_PARSER_VERSION],
        ).fetchall()
    assert tx_count == 0  # never an A-only persistence
    assert runs == [
        ("dual-doc-a", DUAL_PARSER_VERSION, DUAL_ENGINES_ATTEMPTED, "error"),
        ("dual-doc-b", DUAL_PARSER_VERSION, DUAL_ENGINES_ATTEMPTED, "error"),
    ]
    assert json.loads(progress_path.read_text())["errors"] == [
        "dual-doc-a",
        "dual-doc-b",
    ]
    # Both docs stay on the OCR work list for the retry.
    work = get_ocr_work_items(db_path=str(db_file), data_dir=str(tmp_path), year=2024)
    assert [doc_id for doc_id, _, _ in work] == ["dual-doc-a", "dual-doc-b"]


def test_main_loop_dual_without_engine_b_config_alerts_and_stops(
    tmp_path, monkeypatch, capsys
):
    _seed_db_docs(tmp_path, list(_TWO_SHAS.items()))
    monkeypatch.delenv("VISION_B_BASE_URL", raising=False)
    monkeypatch.delenv("VISION_B_MODEL", raising=False)
    db_file, progress_path, calls = _wire_main(
        monkeypatch, tmp_path, docs=_TWO_DOCS, shas=_TWO_SHAS, outputs=_TWO_OUTPUTS
    )

    ocr_zero_rows.main()

    # A dual run without engine B is a config defect: alert and stop the
    # batch before spending engine A on readings nothing may persist.
    assert calls == []  # engine A never contacted
    out = capsys.readouterr().out
    assert "ALERT" in out and "not configured" in out
    with duckdb.connect(str(db_file)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM pdf_parse_runs").fetchone()[0] == 0
    assert not progress_path.exists()


def test_main_loop_without_dual_read_stays_single_engine(
    tmp_path, monkeypatch, vision_b_server
):
    _seed_db(tmp_path)
    server = vision_b_server(B_OUTPUT_REFORMAT)
    _set_engine_b(monkeypatch, server.server_address[1])  # B up, but DUAL_READ off
    db_file, _, calls = _wire_main(
        monkeypatch,
        tmp_path,
        docs=[("dual-doc", 2024, "stub.pdf")],
        shas={"dual-doc": "dual-sha-1"},
        outputs={"dual-doc": A_OUTPUT},
        dual=False,
    )

    ocr_zero_rows.main()

    assert calls == ["dual-doc"]  # engine A read once, no gating round trip
    assert server.requests == []  # engine B never contacted
    with duckdb.connect(str(db_file)) as conn:
        run_row = conn.execute(
            "SELECT parser_version, engines_attempted, status FROM pdf_parse_runs "
            "WHERE doc_id = 'dual-doc'"
        ).fetchone()
    assert run_row == (gem.GEMINI_PARSER_VERSION, gem.MODEL, "success")


# ---------------------------------------------------------------------------
# GLM-primary single-engine mode (OCR_ENGINE=visionb)
# ---------------------------------------------------------------------------


def _write_tiny_pdf(path):
    """Minimal 1-page PDF: snapshot_pdf must hash and pdfinfo page-count it."""
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 72 72] >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_at = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_at}\n%%EOF\n"
    ).encode()
    Path(path).write_bytes(bytes(out))


def test_ocr_engine_mode_resolution_and_precedence():
    resolve = ocr_zero_rows._ocr_engine_mode
    assert resolve({}) == "gemini"
    assert resolve({"OCR_ENGINE": "gemini"}) == "gemini"
    assert resolve({"OCR_ENGINE": " visionb "}) == "visionb"
    assert resolve({"OCR_ENGINE": "dual"}) == "dual"
    # DUAL_READ=1 forces dual regardless of OCR_ENGINE.
    assert resolve({"OCR_ENGINE": "visionb", "DUAL_READ": "1"}) == "dual"
    assert resolve({"OCR_ENGINE": "gemini", "DUAL_READ": "1"}) == "dual"
    # A typo must fail loudly, never silently drain with the wrong quota.
    with pytest.raises(ValueError):
        resolve({"OCR_ENGINE": "glm"})


def test_is_quota_error_recognizes_zai_glm_quota_phrasing():
    # Z.AI maps quota exhaustion to HTTP 429 with body error code 1302
    # ("Rate limit reached for requests", docs.z.ai/api-reference/api-code).
    assert gem.is_quota_error("visionb http 429")
    assert gem.is_quota_error(
        '{"error":{"code":"1302","message":"Rate limit reached for requests"}}'
    )
    assert gem.is_quota_error("insufficient quota balance")
    assert gem.is_quota_error("requests throttled, retry later")
    # Transient and schema failures must NOT stop the batch.
    assert not gem.is_quota_error("visionb transport failure: connection refused")
    assert not gem.is_quota_error("invalid_response: line 3: unexpected text")


def test_visionb_run_classifies_as_gemini_ocr_source_family(tmp_path):
    db = Database(tmp_path / "classify-glm.duckdb")
    db.conn.execute(
        """
        INSERT INTO pdf_parse_runs (doc_id, year, parser_version, status,
                                    engines_attempted, raw_row_count,
                                    transaction_count)
        VALUES ('d1', 2024, ?, 'success', ?, 3, 3)
        """,
        [ocr_zero_rows.GLM_PARSER_VERSION, ocr_zero_rows.GLM_ENGINES_ATTEMPTED],
    )
    source = db.conn.execute(
        f"SELECT {_HOUSE_PARSE_SOURCE} FROM pdf_parse_runs AS parse_run "
        "WHERE doc_id = 'd1'"
    ).fetchone()[0]
    db.close()
    assert source == "gemini_ocr"


def test_visionb_runs_leave_the_ocr_work_list(tmp_path):
    db_path = str(Path(tmp_path) / "worklist.duckdb")
    db = Database(db_path)
    _enable_ocr_schema(db.conn)
    db.conn.execute(
        """
        INSERT INTO metadata (doc_id, first_name, last_name, filing_date,
                              filing_type, fetched_at)
        VALUES ('wl-doc', 'Jane', 'Doe', TIMESTAMP '2024-01-20', 'P',
                CURRENT_TIMESTAMP)
        """
    )
    base = datetime.datetime(2024, 2, 1, 12, 0, 0)
    db.conn.execute(
        "INSERT INTO pdf_parse_runs (doc_id, year, parser_version, status, "
        "engines_attempted, raw_row_count, transaction_count, parsed_at) "
        "VALUES ('wl-doc', 2024, 'v4-house-deterministic', 'zero_rows', "
        "'house_pdf', 0, 0, ?)",
        [base],
    )
    db.close()
    work = get_ocr_work_items(db_path=db_path, data_dir=str(tmp_path), year=2024)
    assert [doc_id for doc_id, _, _ in work] == ["wl-doc"]

    # A GLM no_txs run is done: the doc must leave the work list.
    db = Database(db_path)
    db.conn.execute(
        "INSERT INTO pdf_parse_runs (doc_id, year, parser_version, status, "
        "engines_attempted, raw_row_count, transaction_count, parsed_at) "
        "VALUES ('wl-doc', 2024, ?, 'no_txs', ?, 0, 0, ?)",
        [ocr_zero_rows.GLM_PARSER_VERSION, ocr_zero_rows.GLM_ENGINES_ATTEMPTED,
         base + datetime.timedelta(hours=1)],
    )
    db.close()
    assert get_ocr_work_items(db_path=db_path, data_dir=str(tmp_path), year=2024) == []

    # An error GLM run is not terminal: the doc stays in work for the retry.
    db = Database(db_path)
    db.conn.execute(
        "INSERT INTO pdf_parse_runs (doc_id, year, parser_version, status, "
        "engines_attempted, raw_row_count, transaction_count, parsed_at) "
        "VALUES ('wl-doc', 2024, ?, 'error', ?, 0, 0, ?)",
        [ocr_zero_rows.GLM_PARSER_VERSION, ocr_zero_rows.GLM_ENGINES_ATTEMPTED,
         base + datetime.timedelta(hours=2)],
    )
    db.close()
    work = get_ocr_work_items(db_path=db_path, data_dir=str(tmp_path), year=2024)
    assert [doc_id for doc_id, _, _ in work] == ["wl-doc"]


def test_main_loop_visionb_mode_stages_glm_read_and_never_touches_canonical(
    tmp_path, monkeypatch, vision_b_server, capsys
):
    tiny_pdf = tmp_path / "stub.pdf"
    _write_tiny_pdf(tiny_pdf)
    tiny_sha = gem.pdf_sha256(str(tiny_pdf))
    _seed_db(tmp_path, sha=tiny_sha)  # artifact binding uses the real hash
    server = vision_b_server(A_OUTPUT)
    _set_engine_b(monkeypatch, server.server_address[1])
    monkeypatch.setenv("OCR_ENGINE", "visionb")
    monkeypatch.setattr(
        "scripts.glm_stage.rasterize_page_data_urls",
        lambda pdf_path, dpi=150: [DATA_URL],
    )
    db_file, progress_path, calls = _wire_main(
        monkeypatch,
        tmp_path,
        docs=[("dual-doc", 2024, str(tiny_pdf))],
        shas={"dual-doc": tiny_sha},
        outputs={"dual-doc": A_OUTPUT},
        dual=False,
    )

    ocr_zero_rows.main()

    assert calls == []  # engine A (Gemini) never contacted — the whole point
    assert len(server.requests) == 1  # engine B read the document
    payload = server.requests[0]
    assert payload["model"] == "stub-qwen-vl"
    content = payload["messages"][0]["content"]
    # The SAME transaction-extraction prompt as the Gemini path, never forked.
    assert content[-1]["text"] == gem.PROMPT
    # The reading landed as a STAGING record, not canonical state.
    staging_record = None
    import scripts.glm_stage as glm_stage

    staging_record = glm_stage.load_staging_record(tmp_path, 2024, "dual-doc")
    assert staging_record is not None
    assert staging_record["parse_status"] == "success"
    assert staging_record["rows"] == [
        ["Apple Inc. (AAPL)", "Purchase", "01/15/24", "01/20/24", "A"]
    ]
    assert staging_record["member"] == "Jane Doe"
    with duckdb.connect(str(db_file)) as conn:
        run_row = conn.execute(
            "SELECT COUNT(*) FROM pdf_parse_runs WHERE doc_id = 'dual-doc'"
        ).fetchone()[0]
        tx_count = conn.execute(
            "SELECT COUNT(*) FROM transactions WHERE doc_id = 'dual-doc'"
        ).fetchone()[0]
    assert run_row == 0  # canonical ingestion of GLM rows ONLY via promote
    assert tx_count == 0
    assert not progress_path.exists()  # staging records are the resume state
    out = capsys.readouterr().out
    assert "DUAL" not in out
    assert "STAGED success" in out and "glm_stage promote" in out


def test_main_loop_visionb_mode_skips_already_staged_docs(
    tmp_path, monkeypatch, vision_b_server
):
    tiny_pdf = tmp_path / "stub.pdf"
    _write_tiny_pdf(tiny_pdf)
    tiny_sha = gem.pdf_sha256(str(tiny_pdf))
    _seed_db(tmp_path, sha=tiny_sha)
    server = vision_b_server(A_OUTPUT)
    _set_engine_b(monkeypatch, server.server_address[1])
    monkeypatch.setenv("OCR_ENGINE", "visionb")
    monkeypatch.setattr(
        "scripts.glm_stage.rasterize_page_data_urls",
        lambda pdf_path, dpi=150: [DATA_URL],
    )
    _, _, calls = _wire_main(
        monkeypatch,
        tmp_path,
        docs=[("dual-doc", 2024, str(tiny_pdf))],
        shas={"dual-doc": tiny_sha},
        outputs={"dual-doc": A_OUTPUT},
        dual=False,
    )

    import scripts.glm_stage as glm_stage

    # First run stages the doc; second run must not re-read engine B for it.
    ocr_zero_rows.main()
    assert len(server.requests) == 1
    ocr_zero_rows.main()
    assert len(server.requests) == 1  # no second read: already staged
    assert glm_stage.load_staging_record(tmp_path, 2024, "dual-doc") is not None
    assert calls == []


def test_main_loop_ocr_engine_unset_stays_gemini_and_never_contacts_b(
    tmp_path, monkeypatch, vision_b_server
):
    _seed_db(tmp_path)
    server = vision_b_server(A_OUTPUT)  # B would answer, but is never asked
    _set_engine_b(monkeypatch, server.server_address[1])
    monkeypatch.delenv("OCR_ENGINE", raising=False)
    db_file, _, calls = _wire_main(
        monkeypatch,
        tmp_path,
        docs=[("dual-doc", 2024, "stub.pdf")],
        shas={"dual-doc": "dual-sha-1"},
        outputs={"dual-doc": A_OUTPUT},
        dual=False,
    )

    ocr_zero_rows.main()

    assert calls == ["dual-doc"]  # engine A read once, exactly as before
    assert server.requests == []  # engine B never contacted
    with duckdb.connect(str(db_file)) as conn:
        run_row = conn.execute(
            "SELECT parser_version, engines_attempted, status FROM pdf_parse_runs "
            "WHERE doc_id = 'dual-doc'"
        ).fetchone()
    assert run_row == (gem.GEMINI_PARSER_VERSION, gem.MODEL, "success")


def test_main_loop_dual_read_overrides_ocr_engine(
    tmp_path, monkeypatch, vision_b_server, capsys
):
    _seed_db(tmp_path)
    server = vision_b_server(B_OUTPUT_REFORMAT)
    _set_engine_b(monkeypatch, server.server_address[1])
    monkeypatch.setenv("OCR_ENGINE", "visionb")  # would be single-engine B...
    db_file, _, _ = _wire_main(
        monkeypatch,
        tmp_path,
        docs=[("dual-doc", 2024, "stub.pdf")],
        shas={"dual-doc": "dual-sha-1"},
        outputs={"dual-doc": A_OUTPUT},
        dual=True,  # ...but DUAL_READ=1 wins precedence
    )

    ocr_zero_rows.main()

    assert "DUAL AGREE: inserted=1" in capsys.readouterr().out
    with duckdb.connect(str(db_file)) as conn:
        run_row = conn.execute(
            "SELECT parser_version, status, engines_attempted FROM pdf_parse_runs "
            "WHERE doc_id = 'dual-doc'"
        ).fetchone()
    assert run_row == (DUAL_PARSER_VERSION, "success", DUAL_ENGINES_ATTEMPTED)


def test_main_loop_ocr_engine_dual_alias_runs_dual(
    tmp_path, monkeypatch, vision_b_server, capsys
):
    _seed_db(tmp_path)
    server = vision_b_server(B_OUTPUT_REFORMAT)
    _set_engine_b(monkeypatch, server.server_address[1])
    monkeypatch.setenv("OCR_ENGINE", "dual")
    db_file, _, _ = _wire_main(
        monkeypatch,
        tmp_path,
        docs=[("dual-doc", 2024, "stub.pdf")],
        shas={"dual-doc": "dual-sha-1"},
        outputs={"dual-doc": A_OUTPUT},
        dual=False,  # DUAL_READ unset: the OCR_ENGINE alias selects dual
    )

    ocr_zero_rows.main()

    assert "DUAL AGREE: inserted=1" in capsys.readouterr().out
    with duckdb.connect(str(db_file)) as conn:
        run_row = conn.execute(
            "SELECT parser_version, status, engines_attempted FROM pdf_parse_runs "
            "WHERE doc_id = 'dual-doc'"
        ).fetchone()
    assert run_row == (DUAL_PARSER_VERSION, "success", DUAL_ENGINES_ATTEMPTED)


def test_main_loop_visionb_quota_stop_ends_batch_cleanly(
    tmp_path, monkeypatch, vision_b_server, capsys
):
    tiny_pdf = tmp_path / "stub.pdf"
    _write_tiny_pdf(tiny_pdf)
    _seed_db_docs(tmp_path, list(_TWO_SHAS.items()))
    server = vision_b_server("ignored", status=429)
    _set_engine_b(monkeypatch, server.server_address[1])
    monkeypatch.setenv("OCR_ENGINE", "visionb")
    db_file, progress_path, calls = _wire_main(
        monkeypatch,
        tmp_path,
        docs=[(doc_id, 2024, str(tiny_pdf)) for doc_id, _ in _TWO_SHAS.items()],
        shas=_TWO_SHAS,
        outputs=_TWO_OUTPUTS,
        dual=False,
    )

    ocr_zero_rows.main()  # returns cleanly: a quota stop is not a crash

    assert calls == []  # engine A never contacted
    assert len(server.requests) == 1  # first doc hit engine B and got the 429
    out = capsys.readouterr().out
    assert "Quota exhausted; stopping batch" in out
    assert "=== DONE ===" in out
    with duckdb.connect(str(db_file)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM pdf_parse_runs").fetchone()[0] == 0
    assert not progress_path.exists()  # nothing processed, nothing marked


def test_main_loop_visionb_without_engine_b_config_fails_closed(
    tmp_path, monkeypatch, capsys
):
    _seed_db(tmp_path)
    monkeypatch.setenv("OCR_ENGINE", "visionb")
    monkeypatch.delenv("DUAL_READ", raising=False)
    monkeypatch.delenv("VISION_B_BASE_URL", raising=False)
    monkeypatch.delenv("VISION_B_MODEL", raising=False)
    db_file, progress_path, calls = _wire_main(
        monkeypatch,
        tmp_path,
        docs=[("dual-doc", 2024, "stub.pdf")],
        shas={"dual-doc": "dual-sha-1"},
        outputs={"dual-doc": A_OUTPUT},
        dual=False,
    )

    ocr_zero_rows.main()

    assert calls == []  # engine A is not silently spent instead
    assert "stopping batch" in capsys.readouterr().out
    with duckdb.connect(str(db_file)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM pdf_parse_runs").fetchone()[0] == 0
    assert not progress_path.exists()
