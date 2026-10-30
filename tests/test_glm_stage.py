"""Staged GLM OCR ingestion + pixel audit (glm_stage.py / audit_sample.py).

Every case runs in temporary directories: ``data/congress.duckdb`` is never
opened, and the canonical insert path (``ocr_zero_rows.insert_transactions``)
is monkeypatched for promotion tests. No model calls are made.
"""

from __future__ import annotations

import base64
import builtins
import hashlib
import io
import json
import os
import shutil
import threading
from uuid import uuid4
from pathlib import Path
from types import SimpleNamespace

import pytest

from analyzer.vision_engines import (
    GoogleVisionBRateLimitError,
    GoogleVisionBServiceUnavailableError,
    VisionBConfig,
    VisionBUnavailable,
)
from scripts import audit_sample, glm_stage, ocr_zero_rows as operational_ocr_zero_rows
from analyzer.ocr import ingestion as ocr_zero_rows
from analyzer.ocr.common import PROMPT_SHA256
from .conftest import make_ocr_output

pytestmark = pytest.mark.slow


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    import socket

    def refused(*args, **kwargs):
        raise AssertionError("GLM tests must not access the network")

    monkeypatch.setattr(socket.socket, "connect", refused)
    monkeypatch.setattr(socket, "create_connection", refused)

GLM_OUTPUT_ONE_ROW = (
    "MEMBER: Jane Doe\nPAGES: 1\nPAGE: 1\n"
    "Apple Inc. (AAPL) | Purchase | 01/15/24 | 01/20/24 | A\n"
)
GLM_OUTPUT_NO_TXS = "MEMBER: Jane Doe\nPAGES: 1\nPAGE: 1\nNO_TRANSACTIONS\n"
GLM_OUTPUT_PAGE2 = (
    "MEMBER: Jane Doe\nPAGES: 2\nPAGE: 1\nNO_TRANSACTIONS\nPAGE: 2\n"
    "Microsoft Corp (MSFT) | Sale | 02/02/24 | 02/05/24 | B\n"
)


def make_pdf(path: Path, pages: int = 1) -> Path:
    """Minimal blank-page PDF that poppler (pdfinfo/pdftoppm) can read."""
    objs = {}
    objs[1] = "<< /Type /Catalog /Pages 2 0 R >>"
    kids = " ".join(f"{3 + i} 0 R" for i in range(pages))
    objs[2] = f"<< /Type /Pages /Kids [{kids}] /Count {pages} >>"
    for i in range(pages):
        objs[3 + i] = "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>"
    body = ["%PDF-1.4"]
    offsets = {}
    for number in sorted(objs):
        offsets[number] = sum(len(segment) + 1 for segment in body)
        body.append(f"{number} 0 obj {objs[number]} endobj")
    xref_at = sum(len(segment) + 1 for segment in body)
    size = len(objs) + 1
    body += ["xref", f"0 {size}", "0000000000 65535 f "]
    for number in range(1, size):
        body.append(f"{offsets[number]:010d} 00000 n ")
    body += [
        "trailer",
        f"<< /Size {size} /Root 1 0 R >>",
        "startxref",
        str(xref_at),
        "%%EOF",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(body), encoding="ascii")
    return path


def seed_year(data_dir: Path, year: int, doc_ids: list[str], pages: int = 1) -> None:
    for doc_id in doc_ids:
        make_pdf(data_dir / str(year) / "pdfs" / f"{doc_id}.pdf", pages=pages)


def seed_house_queue(data_dir, doc_ids, *, archive_year=2024, filing_year=2024):
    """Synthetic current House acquisitions; never opens the live database."""
    from analyzer.persistence.database import Database

    seed_year(data_dir, archive_year, doc_ids)
    db_path = data_dir / "congress.duckdb"
    db = Database(db_path)
    digests = {}
    db.conn.execute(
        "INSERT INTO house_archive_generations (archive_year, generation_id, metadata_count, ptr_count) VALUES (?, 'current', ?, ?)",
        [archive_year, len(doc_ids), len(doc_ids)],
    )
    for doc_id in doc_ids:
        digest = glm_stage.pdf_sha256(glm_stage.default_pdf_path(data_dir, archive_year, doc_id))
        digests[doc_id] = digest
        date = f"{filing_year}-02-01"
        db.conn.execute(
            "INSERT INTO metadata (doc_id, archive_year, filing_date, filing_type) VALUES (?, ?, ?, 'P')",
            [doc_id, archive_year, date],
        )
        db.conn.execute(
            "INSERT INTO house_generation_metadata (doc_id, archive_year, generation_id, filing_date, filing_type) VALUES (?, ?, 'current', ?, 'P')",
            [doc_id, archive_year, date],
        )
        db.conn.execute(
            "INSERT INTO house_pdf_artifacts (doc_id, archive_year, generation_id, artifact_sha256) VALUES (?, ?, 'current', ?)",
            [doc_id, archive_year, digest],
        )
        ocr_zero_rows.record_parse_run(
            db.conn, doc_id, archive_year, "error", 0, 0,
            parser_version="deterministic", engines_attempted="pdfplumber",
            artifact_sha256=digest, ingestion_generation="current",
        )
    db.close()
    return db_path, digests


def read_fn_from_outputs(outputs: dict[str, str]):
    def read_fn(pdf_path, page_count):
        return outputs[Path(pdf_path).stem]

    return read_fn


@pytest.fixture
def data_dir(tmp_path) -> Path:
    return tmp_path / "data"


@pytest.fixture
def private_read_spy(monkeypatch):
    """Observe both path opens and fd-backed reads before asserting rejection."""
    def install(private_root):
        reads = []
        private_root = private_root.resolve()
        for owner in (builtins, io):
            original = owner.open

            def tracked(file, mode="r", *args, _open=original, **kwargs):
                path = Path(os.readlink(f"/proc/self/fd/{file}")) if isinstance(file, int) else Path(file)
                if "r" in mode and path.resolve().is_relative_to(private_root):
                    reads.append(str(path))
                return _open(file, mode, *args, **kwargs)

            monkeypatch.setattr(owner, "open", tracked)
        return reads
    return install


@pytest.fixture
def fake_insert(monkeypatch):
    """Monkeypatch the canonical insert path + generation resolution.

    The fake returns the canonical structured outcome directly.
    """
    calls = []

    def _insert(
        doc_id,
        year,
        member,
        transactions,
        *,
        db_path,
        parser_version,
        raw_count,
        artifact_sha256,
        ingestion_generation,
        engine_model,
        checkpoint=True,
    ):
        calls.append(
            {
                "doc_id": doc_id,
                "year": year,
                "member": member,
                "transactions": transactions,
                "db_path": db_path,
                "parser_version": parser_version,
                "raw_count": raw_count,
                "artifact_sha256": artifact_sha256,
                "ingestion_generation": ingestion_generation,
                "engine_model": engine_model,
            }
        )
        return ocr_zero_rows.InsertOutcome(
            status="no_txs" if not transactions else "success",
            inserted=len(transactions),
            expected=raw_count if raw_count is not None else len(transactions),
            committed=True,
        )

    monkeypatch.setattr(ocr_zero_rows, "insert_transactions_outcome", _insert)
    monkeypatch.setattr(
        ocr_zero_rows, "insert_transactions",
        lambda *a, **kw: pytest.fail("promotion called the integer compatibility adapter"),
    )
    monkeypatch.setattr(
        ocr_zero_rows,
        "resolve_ingestion_generation",
        lambda db_path, doc_id, year, sha: "gen-1",
    )
    return calls


def test_artifact_request_normalizes_only_identity_without_changing_evidence():
    legacy = {
        "doc_id": "legacy", "year": 2024, "pdf_sha256": "digest",
        "model": "legacy-model", "prompt_sha256": PROMPT_SHA256,
        "parsed_at_utc": "2026-10-18T00:00:00Z",
        "parser_version": "v1-glm53-vision", "pages": 1,
        "member": "Jane Doe",
        "rows": [["original evidence", "Purchase", "01/15/24", None, "A"]],
        "raw_output": None, "parse_status": "success", "error": None,
        "audit": "pass",
    }
    raw = json.dumps(legacy)
    record = glm_stage.decode_staging_record(raw)
    request = glm_stage.ArtifactRequest.from_record(record)
    assert request.resolver_fields() == {
        "doc_id": "legacy", "year": 2024, "archive_year": None,
        "artifact_sha256": "digest", "ingestion_generation": None,
    }
    assert json.dumps(record) == raw  # no defaults added to content-bound JSON
    record["artifact_sha256"] = "different"
    with pytest.raises(RuntimeError, match="conflicting"):
        glm_stage.ArtifactRequest.from_record(record)
    with pytest.raises(RuntimeError, match="conflicting"):
        glm_stage.ArtifactRequest.from_manifest(record)


# --------------------------------------------------------------------------
# stage: record format + round trip
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# finding 6: provenance (render_dpi, row_pages, raw_output evidence, snapshot)
# --------------------------------------------------------------------------


def test_stage_record_schema_round_trip(data_dir):
    seed_year(data_dir, 2024, ["1001", "1002"])
    outputs = {"1001": GLM_OUTPUT_ONE_ROW, "1002": GLM_OUTPUT_NO_TXS}
    result = glm_stage.stage_batch(
        2024, 10, data_dir=data_dir, read_fn=read_fn_from_outputs(outputs)
    )
    assert result["staged"] == 2
    assert result["errors"] == 0
    assert result["quota_stop"] is False

    record = glm_stage.load_staging_record(data_dir, 2024, "1001")
    assert record is not None
    # Documented schema fields:
    assert record["doc_id"] == "1001"
    assert record["year"] == 2024
    assert record["pdf_sha256"] == glm_stage.pdf_sha256(
        data_dir / "2024" / "pdfs" / "1001.pdf"
    )
    assert record["prompt_sha256"] == PROMPT_SHA256
    assert record["parser_version"] == "v1-glm53-vision"
    assert record["parsed_at_utc"].endswith("Z")
    assert record["pages"] == 1
    assert record["rows"] == [
        ["Apple Inc. (AAPL)", "Purchase", "01/15/24", "01/20/24", "A"]
    ]
    assert record["parse_status"] == "success"
    assert record["error"] is None
    assert record["member"] == "Jane Doe"
    assert record["raw_output"] == GLM_OUTPUT_ONE_ROW

    no_txs = glm_stage.load_staging_record(data_dir, 2024, "1002")
    assert no_txs["parse_status"] == "no_txs"
    assert no_txs["rows"] == []

    # Re-running stages nothing new: existing records are skipped.
    again = glm_stage.stage_batch(
        2024, 10, data_dir=data_dir, read_fn=read_fn_from_outputs(outputs)
    )
    assert again["candidates"] == 0


def test_stage_limit_and_error_records(data_dir):
    seed_year(data_dir, 2023, ["a", "b", "c", "d"])
    outputs = {
        "a": GLM_OUTPUT_ONE_ROW,
        "b": GLM_OUTPUT_NO_TXS,
        "c": GLM_OUTPUT_ONE_ROW,
        "d": GLM_OUTPUT_ONE_ROW,
    }

    limited = glm_stage.stage_batch(
        2023, 2, data_dir=data_dir, read_fn=read_fn_from_outputs(outputs)
    )
    assert limited["candidates"] == 2
    assert glm_stage.load_staging_record(data_dir, 2023, "a") is not None
    assert glm_stage.load_staging_record(data_dir, 2023, "c") is None

    # One bad doc never kills the batch: 'c' errors while 'd' still stages.
    def broken_read(pdf_path, page_count):
        if Path(pdf_path).stem == "c":
            raise ValueError("malformed pixels")
        return outputs[Path(pdf_path).stem]

    bad = glm_stage.stage_batch(2023, 5, data_dir=data_dir, read_fn=broken_read)
    assert bad["candidates"] == 2
    assert bad["staged"] == 1 and bad["errors"] == 1
    error_record = glm_stage.load_staging_record(data_dir, 2023, "c")
    assert error_record["parse_status"] == "error"
    assert error_record["error"] == "malformed pixels"
    good_record = glm_stage.load_staging_record(data_dir, 2023, "d")
    assert good_record["parse_status"] == "success"


def test_quota_stop_is_clean_and_reports_remaining(data_dir):
    seed_year(data_dir, 2024, ["d1", "d2", "d3"])
    attempted = []

    def quota_read(pdf_path, page_count):
        attempted.append(Path(pdf_path).stem)
        raise VisionBUnavailable("visionb http 429")

    result = glm_stage.stage_batch(2024, 10, data_dir=data_dir, read_fn=quota_read)
    assert result["quota_stop"] is True
    assert result["remaining"] == 3
    assert attempted == ["d1"]  # batch stopped; later docs untouched
    assert list((data_dir / "staging" / "glm-backlog" / "2024").glob("*.json")) == []


# --------------------------------------------------------------------------
# stage: bounded transient retries + error classification (FP finding 7)
# --------------------------------------------------------------------------


def test_transient_failures_are_retried_with_jittered_backoff(data_dir, monkeypatch):
    from analyzer.vision_engines import TransientVisionBError

    seed_year(data_dir, 2024, ["r1"])
    attempts = []
    sleeps = []

    def flaky_read(pdf_path, page_count):
        attempts.append(1)
        if len(attempts) < 3:
            raise TransientVisionBError("visionb http 503: upstream overload")
        return GLM_OUTPUT_ONE_ROW

    monkeypatch.setattr(
        glm_stage, "time", SimpleNamespace(sleep=sleeps.append)
    )
    monkeypatch.setattr(glm_stage.random, "uniform", lambda a, b: 0.5)
    result = glm_stage.stage_batch(
        2024, 10, data_dir=data_dir, read_fn=flaky_read, cooldown=0
    )
    assert result["staged"] == 1 and result["errors"] == 0
    assert len(attempts) == 3  # initial attempt + 2 bounded retries
    # Jittered exponential backoff: base * 2**attempt + jitter.
    assert sleeps == [2.0 + 0.5, 4.0 + 0.5]
    record = glm_stage.load_staging_record(data_dir, 2024, "r1")
    assert record["parse_status"] == "success"


def test_google_429_retry_waits_at_least_one_minute(data_dir, monkeypatch):
    from analyzer import vision_engines as ve

    seed_year(data_dir, 2024, ["rate-limited"])
    monkeypatch.setattr(glm_stage, "rasterize_page_data_urls", lambda *a: ["page"])
    monkeypatch.setattr(glm_stage, "time", SimpleNamespace(sleep=lambda _: None))

    class _RateLimitedResponse:
        status_code = 429
        text = "RESOURCE_EXHAUSTED: quota exceeded for quota metric per minute"

    monkeypatch.setattr(
        ve.requests, "post", lambda *args, **kwargs: _RateLimitedResponse()
    )
    sleeps = []
    with pytest.raises(GoogleVisionBRateLimitError):
        glm_stage.read_document(
            data_dir / "2024" / "pdfs" / "rate-limited.pdf",
            VisionBConfig(
                base_url="https://generativelanguage.googleapis.com/v1beta/openai",
                model="gemini-3-flash-preview",
            ),
            sleep_fn=sleeps.append,
            jitter_fn=lambda low, high: high,
        )

    assert len(sleeps) == 2
    assert all(60 <= delay <= 90 for delay in sleeps)


def test_google_503_uses_long_backoff_without_expanding_retry_budget(
    data_dir, monkeypatch
):
    from analyzer import vision_engines as ve

    seed_year(data_dir, 2024, ["busy"])
    monkeypatch.setattr(glm_stage, "rasterize_page_data_urls", lambda *a: ["page"])

    class _UnavailableResponse:
        status_code = 503
        text = "UNAVAILABLE: model currently experiencing high demand"

    attempts = []
    monkeypatch.setattr(
        ve.requests,
        "post",
        lambda *args, **kwargs: (attempts.append(1) or _UnavailableResponse()),
    )
    sleeps = []
    with pytest.raises(GoogleVisionBServiceUnavailableError):
        glm_stage.read_document(
            data_dir / "2024" / "pdfs" / "busy.pdf",
            VisionBConfig(
                base_url="https://generativelanguage.googleapis.com/v1beta/openai",
                model="gemini-3.1-flash-lite",
            ),
            sleep_fn=sleeps.append,
            jitter_fn=lambda low, high: high,
        )

    assert len(attempts) == glm_stage.MAX_READ_RETRIES + 1 == 3
    assert sleeps == [90.0, 90.0]


def test_zai_503_keeps_short_transient_backoff(data_dir, monkeypatch):
    from analyzer import vision_engines as ve
    from analyzer.vision_engines import TransientVisionBError

    seed_year(data_dir, 2024, ["zai-busy"])
    monkeypatch.setattr(glm_stage, "rasterize_page_data_urls", lambda *a: ["page"])

    class _UnavailableResponse:
        status_code = 503
        text = "UNAVAILABLE: upstream busy"

    attempts = []
    monkeypatch.setattr(
        ve.requests,
        "post",
        lambda *args, **kwargs: (attempts.append(1) or _UnavailableResponse()),
    )
    sleeps = []
    with pytest.raises(TransientVisionBError) as excinfo:
        glm_stage.read_document(
            data_dir / "2024" / "pdfs" / "zai-busy.pdf",
            VisionBConfig(base_url="https://api.z.ai/v1", model="glm-5.3-flash"),
            sleep_fn=sleeps.append,
            jitter_fn=lambda low, high: 0.5,
        )

    assert type(excinfo.value) is TransientVisionBError
    assert len(attempts) == glm_stage.MAX_READ_RETRIES + 1 == 3
    assert sleeps == [2.5, 4.5]


def test_google_429_streak_degrades_to_batch_stop(data_dir, monkeypatch, capsys):
    from analyzer import vision_engines as ve

    doc_ids = [f"rate-{index}" for index in range(7)]
    seed_year(data_dir, 2024, doc_ids)
    monkeypatch.setattr(glm_stage, "rasterize_page_data_urls", lambda *a: ["page"])
    monkeypatch.setattr(glm_stage, "time", SimpleNamespace(sleep=lambda _: None))

    class _RateLimitedResponse:
        status_code = 429
        text = "RESOURCE_EXHAUSTED: quota exceeded for quota metric per minute"

    requests = []

    def rate_limited_post(*args, **kwargs):
        requests.append(1)
        return _RateLimitedResponse()

    monkeypatch.setattr(ve.requests, "post", rate_limited_post)
    config = VisionBConfig(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        model="gemini-3-flash-preview",
    )

    result = glm_stage.stage_batch(2024, 10, data_dir=data_dir, config=config)

    assert glm_stage.GOOGLE_429_CONSECUTIVE_STOP_THRESHOLD == 5
    assert result["quota_stop"] is True
    assert result["google_quota_dead"] is True
    assert result["remaining"] == 3
    assert result["errors"] == 4
    assert len(requests) == 5 * (glm_stage.MAX_READ_RETRIES + 1)
    staged_dir = data_dir / "staging" / "glm-backlog" / "2024"
    records = sorted(staged_dir.glob("*.json"))
    assert [record.stem for record in records] == doc_ids[:4]
    assert all(
        glm_stage.load_staging_record(data_dir, 2024, doc_id)["error_kind"]
        == "transient"
        for doc_id in doc_ids[:4]
    )

    monkeypatch.setattr(glm_stage, "stage_batch", lambda *a, **k: result)
    assert glm_stage.main(["--data-dir", str(data_dir), "stage", "--year", "2024"]) == 0
    assert "Google quota dead for the session" in capsys.readouterr().out


def test_google_503_streak_stops_with_saturation_message(
    data_dir, monkeypatch, capsys
):
    from analyzer import vision_engines as ve

    doc_ids = [f"busy-{index}" for index in range(7)]
    seed_year(data_dir, 2024, doc_ids)
    monkeypatch.setattr(glm_stage, "rasterize_page_data_urls", lambda *a: ["page"])
    monkeypatch.setattr(glm_stage, "time", SimpleNamespace(sleep=lambda _: None))

    class _UnavailableResponse:
        status_code = 503
        text = "UNAVAILABLE: model currently experiencing high demand"

    requests = []

    def unavailable_post(*args, **kwargs):
        requests.append(1)
        return _UnavailableResponse()

    monkeypatch.setattr(ve.requests, "post", unavailable_post)
    config = VisionBConfig(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        model="gemini-3.1-flash-lite",
    )

    result = glm_stage.stage_batch(2024, 10, data_dir=data_dir, config=config)

    assert glm_stage.GOOGLE_503_CONSECUTIVE_STOP_THRESHOLD == 5
    assert result["quota_stop"] is True
    assert result["google_endpoint_saturated"] is True
    assert result["google_quota_dead"] is False
    assert result["remaining"] == 3
    assert result["errors"] == 4
    assert len(requests) == 5 * (glm_stage.MAX_READ_RETRIES + 1)
    staged_dir = data_dir / "staging" / "glm-backlog" / "2024"
    records = sorted(staged_dir.glob("*.json"))
    assert [record.stem for record in records] == doc_ids[:4]
    assert all(
        glm_stage.load_staging_record(data_dir, 2024, doc_id)["error_kind"]
        == "transient"
        for doc_id in doc_ids[:4]
    )

    monkeypatch.setattr(glm_stage, "stage_batch", lambda *a, **k: result)
    assert glm_stage.main(["--data-dir", str(data_dir), "stage", "--year", "2024"]) == 0
    assert (
        "google endpoint saturated — stopping cleanly; error records retry nightly"
        in capsys.readouterr().out.lower()
    )


def test_transient_retries_exhaust_into_error_record_and_batch_continues(
    data_dir, monkeypatch
):
    from analyzer.vision_engines import TransientVisionBError

    seed_year(data_dir, 2024, ["x1", "x2"])
    attempts = []
    monkeypatch.setattr(glm_stage, "time", SimpleNamespace(sleep=lambda *_: None))

    def down_read(pdf_path, page_count):
        attempts.append(Path(pdf_path).stem)
        raise TransientVisionBError("visionb transport failure: refused")

    result = glm_stage.stage_batch(2024, 10, data_dir=data_dir, read_fn=down_read)
    assert attempts == ["x1"] * 3 + ["x2"] * 3  # each doc retried, batch lived
    assert result["staged"] == 0 and result["errors"] == 2
    assert result["quota_stop"] is False and result["fatal_stop"] is False
    record = glm_stage.load_staging_record(data_dir, 2024, "x1")
    assert record["parse_status"] == "error"
    assert record["error_kind"] == "transient"


def test_quota_is_never_retried_and_stops_cleanly(data_dir, monkeypatch):
    from analyzer.vision_engines import VisionBQuotaError

    seed_year(data_dir, 2024, ["q1", "q2"])
    attempts = []

    def quota_read(pdf_path, page_count):
        attempts.append(1)
        raise VisionBQuotaError("visionb quota/rate-limit exhausted (http 429)")

    result = glm_stage.stage_batch(2024, 10, data_dir=data_dir, read_fn=quota_read)
    assert attempts == [1]  # no retry: a quota stop is not transient
    assert result["quota_stop"] is True and result["fatal_stop"] is False
    assert list((data_dir / "staging" / "glm-backlog" / "2024").glob("*.json")) == []


def test_auth_defect_alerts_and_stops_batch_without_records(data_dir, monkeypatch, capsys):
    from analyzer.vision_engines import VisionBAuthError

    seed_year(data_dir, 2024, ["a1", "a2"])
    attempts = []

    def auth_read(pdf_path, page_count):
        attempts.append(Path(pdf_path).stem)
        raise VisionBAuthError(
            "visionb auth rejected (http 401): {\"error\":{\"code\":1002,"
            '"message":"invalid api key"}}'
        )

    result = glm_stage.stage_batch(2024, 10, data_dir=data_dir, read_fn=auth_read)
    assert attempts == ["a1"]  # no retry: credentials are broken, not busy
    assert result["fatal_stop"] is True and result["quota_stop"] is False
    assert list((data_dir / "staging" / "glm-backlog" / "2024").glob("*.json")) == []
    out = capsys.readouterr().out
    assert "ALERT" in out and "auth" in out

    # The CLI surfaces the fatal stop as a failure exit, not a silent 0.
    monkeypatch.setattr(
        glm_stage,
        "stage_batch",
        lambda *a, **k: {
            "year": 2024,
            "candidates": 2,
            "staged": 0,
            "errors": 0,
            "quota_stop": False,
            "fatal_stop": True,
            "remaining": 2,
        },
    )
    rc = glm_stage.main(["--data-dir", str(data_dir), "stage", "--year", "2024"])
    assert rc == 1
    assert "auth/config defect stopped the batch" in capsys.readouterr().out


def test_error_records_carry_the_classified_kind(data_dir):
    from analyzer.vision_engines import (
        TransientVisionBError,
        TruncatedCompletion,
        VisionBPermanentError,
        VisionBQuotaError,
    )

    seed_year(data_dir, 2024, ["k1", "k2", "k3", "k4"])
    cases = {
        "k1": (TransientVisionBError("visionb http 503"), "transient"),
        "k2": (TruncatedCompletion("visionb truncated completion"), "truncated"),
        "k3": (VisionBQuotaError("visionb quota (http 429)"), "quota"),
        "k4": (VisionBPermanentError("visionb http 400: bad image"), "permanent"),
    }

    def classified_read(pdf_path, page_count):
        exc, _ = cases[Path(pdf_path).stem]
        raise exc

    # build_staging_record never raises: it classifies into error_kind so
    # the batch loop can tell per-doc defects from batch-wide stops.
    for doc_id, (_, kind) in cases.items():
        record = glm_stage.build_staging_record(
            doc_id, 2024, glm_stage.default_pdf_path(data_dir, 2024, doc_id),
            None, read_fn=classified_read,
        )
        assert record["parse_status"] == "error"
        assert record["error_kind"] == kind
    # A schema-level parse failure (not an engine class) is its own kind.
    record = glm_stage.build_staging_record(
        "k5", 2024, glm_stage.default_pdf_path(data_dir, 2024, "k5"),
        None, read_fn=lambda *a: "not the schema",
    )
    assert record["error_kind"] == "parse"


def test_renders_are_cached_across_transient_retries(data_dir, monkeypatch):
    from analyzer.vision_engines import TransientVisionBError

    seed_year(data_dir, 2024, ["rc1"])
    renders = []

    def fake_rasterize(pdf_path, dpi=glm_stage.RENDER_DPI):
        renders.append(1)
        return ["data:image/png;base64,QUJD"]

    def transient_then_ok(page_urls, config):
        assert page_urls == ["data:image/png;base64,QUJD"]
        if len(renders) == 1 and not getattr(transient_then_ok, "ok", False):
            transient_then_ok.ok = True
            raise TransientVisionBError("visionb http 502")
        return GLM_OUTPUT_ONE_ROW

    monkeypatch.setattr(glm_stage, "rasterize_page_data_urls", fake_rasterize)
    monkeypatch.setattr(
        glm_stage, "read_document_with_vision_b", transient_then_ok
    )
    monkeypatch.setattr(glm_stage, "time", SimpleNamespace(sleep=lambda *_: None))
    config = VisionBConfig(base_url="http://127.0.0.1:1", model="stub")
    record = glm_stage.build_staging_record(
        "rc1", 2024, glm_stage.default_pdf_path(data_dir, 2024, "rc1"), config
    )
    assert record["parse_status"] == "success"
    assert len(renders) == 1  # one render, reused by the retry


def test_check_canonical_skips_successfully_parsed_docs(data_dir, monkeypatch):
    seed_year(data_dir, 2024, ["keep", "parsed"])
    monkeypatch.setattr(
        glm_stage, "canonical_parsed_doc_ids", lambda db_path: {"parsed"}
    )
    candidates = glm_stage.select_candidates(
        2024, 10, data_dir, check_canonical=True, db_path="unused.duckdb"
    )
    assert candidates == ["keep"]


# --------------------------------------------------------------------------
# promote: consumes staging via the existing insert path
# --------------------------------------------------------------------------


def stage_one(data_dir, year, doc_id, output):
    seed_year(data_dir, year, [doc_id])
    return glm_stage.stage_batch(
        year, 5, data_dir=data_dir, read_fn=read_fn_from_outputs({doc_id: output})
    )


def write_report_state(data_dir, *, state="complete", budget_met=True, schema="audit-report-state-v2", raw=None):
    """Write the audit-sample report state file (the promote gate input)."""
    root = glm_stage.staging_root(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    if raw is not None:
        (root / "audit-report-state.json").write_text(raw, encoding="utf-8")
        return
    lot_size = 299 if state == "complete" and budget_met is True else 30
    audited = lot_size if state == "complete" else lot_size - 1
    sample_file = "audit-sample-promotion-fixture.json"
    envelope = {
        "schema": audit_sample.SAMPLE_SCHEMA,
        "seed": 8675309,
        "per_year": lot_size,
        "created_at_utc": "2026-10-24T00:00:00Z",
        "frame_counts_by_year": {"2024": lot_size},
        "frame_counts_by_stratum": {
            "2024": {"flagged": 0, "unflagged": lot_size}
        },
        "lot": [
            {
                "doc_id": f"fixture-{index}",
                "year": 2024,
                "staged_record_sha256": "0" * 64,
                "model": "synthetic",
                "prompt_sha256": "0" * 64,
                "parser_version": "synthetic",
                "asset_review": False,
            }
            for index in range(lot_size)
        ],
    }
    sample_bytes = json.dumps(envelope, indent=2, sort_keys=True).encode("utf-8")
    (root / sample_file).write_bytes(sample_bytes)
    payload = {
        "schema": schema,
        "state": state,
        "reason": None if state == "complete" else "unresolved sampled docs",
        "completion_reasons": [] if state == "complete" else ["fixture: no verdict recorded"],
        "completion_guidance": [] if state == "complete" else ["record an explicit verdict"],
        "budget_met": budget_met,
        "lot_size": lot_size,
        "audited": audited,
        "critical_failures": [],
        "critical_failure_count": 0,
        "cosmetic_failures": [],
        "cosmetic_failure_count": 0,
        "unresolved_count": 0 if state == "complete" else 1,
        "unresolved": [] if state == "complete" else [["fixture", "no verdict recorded"]],
        "rate": 0.0,
        "upper_bound": audit_sample.binomial_upper_bound(0, audited),
        "confidence": 0.95,
        "max_rate": 0.01,
        "population_critical_failure_count": 0,
        "audit_history_sha256": audit_sample.audit_history_sha256(
            audit_sample.load_audit_results(data_dir)
        ),
        "sample_file": sample_file,
        "sample_file_sha256": hashlib.sha256(sample_bytes).hexdigest(),
        "omission_check_status": "not_run",
    }
    (root / "audit-report-state.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def stage_one_audited(data_dir, year, doc_id, output):
    """Stage one doc, stamp the mandatory audit: pass, and land a complete
    audit report with the budget met (the fully-gated-open world)."""
    stage_one(data_dir, year, doc_id, output)
    audit_sample.record_verdict(data_dir, doc_id, "pass", "matches pixels")
    write_report_state(data_dir)
    return glm_stage.load_staging_record(data_dir, year, doc_id)


def test_audit_verdict_refuses_stale_record_after_restage(data_dir, monkeypatch):
    stage_one(data_dir, 2024, "audit-race", GLM_OUTPUT_ONE_ROW)
    live = glm_stage.staging_record_path(data_dir, 2024, "audit-race")
    original_bytes = live.read_bytes()
    replacement = glm_stage.build_staging_record(
        "audit-race",
        2024,
        glm_stage.default_pdf_path(data_dir, 2024, "audit-race"),
        None,
        read_fn=lambda *_: GLM_OUTPUT_NO_TXS,
        data_dir=data_dir,
    )
    history_path = glm_stage.staging_root(data_dir) / "audit-results.json"
    history_bytes = json.dumps([{"doc_id": "older", "verdict": "fail"}]).encode()
    history_path.write_bytes(history_bytes)
    journal_path = glm_stage.staging_root(data_dir) / glm_stage.PROMOTED_JOURNAL_NAME
    journal_prefix = b'{"doc_id":"already-promoted"}\n'
    journal_path.write_bytes(journal_prefix)

    read_artifact = ocr_zero_rows.read_work_artifact
    restaged = False

    def read_then_restage(root, path):
        nonlocal restaged
        contents = read_artifact(root, path)
        if Path(path).name == live.name and not restaged:
            restaged = True
            glm_stage.publish_staging_record(data_dir, replacement)
        return contents

    monkeypatch.setattr(ocr_zero_rows, "read_work_artifact", read_then_restage)
    with pytest.raises(ValueError, match="changed while recording verdict"):
        audit_sample.record_verdict(data_dir, "audit-race", "pass", "reviewed A")

    assert restaged
    (attempt_number, attempt_path), = glm_stage.archived_attempts(
        data_dir, 2024, "audit-race"
    )
    assert attempt_number == 0 and attempt_path.read_bytes() == original_bytes
    assert glm_stage.load_staging_record(data_dir, 2024, "audit-race")["parse_status"] == "no_txs"
    assert history_path.read_bytes() == history_bytes
    assert journal_path.read_bytes() == journal_prefix


def test_operational_legacy_writer_archives_concurrent_record(data_dir, monkeypatch):
    seed_year(data_dir, 2024, ["legacy-writer"])
    pdf = glm_stage.default_pdf_path(data_dir, 2024, "legacy-writer")
    original = glm_stage.build_staging_record(
        "legacy-writer", 2024, pdf, None, read_fn=lambda *_: GLM_OUTPUT_ONE_ROW,
        data_dir=data_dir,
    )
    replacement = glm_stage.build_staging_record(
        "legacy-writer", 2024, pdf, None, read_fn=lambda *_: GLM_OUTPUT_NO_TXS,
        data_dir=data_dir,
    )
    original_bytes = []

    def publish_while_legacy_reads(*_args, **_kwargs):
        glm_stage.write_staging_record(data_dir, original)
        original_bytes.append(
            glm_stage.staging_record_path(data_dir, 2024, "legacy-writer").read_bytes()
        )
        return replacement

    monkeypatch.setattr(
        operational_ocr_zero_rows.time, "sleep", lambda *_args: None
    )
    monkeypatch.setattr(glm_stage, "build_staging_record", publish_while_legacy_reads)
    operational_ocr_zero_rows._run_visionb_staging(
        [("legacy-writer", 2024, pdf)], None, data_dir=data_dir
    )

    (attempt_number, archived_path), = glm_stage.archived_attempts(
        data_dir, 2024, "legacy-writer"
    )
    assert attempt_number == 0 and archived_path.read_bytes() == original_bytes[0]
    assert glm_stage.load_staging_record(data_dir, 2024, "legacy-writer")[
        "parse_status"
    ] == "no_txs"


def test_promote_consumes_staging_record(data_dir, fake_insert):
    stage_one_audited(data_dir, 2024, "2001", GLM_OUTPUT_ONE_ROW)
    sha = glm_stage.pdf_sha256(data_dir / "2024" / "pdfs" / "2001.pdf")

    result = glm_stage.promote_docs(2024, ["2001"], data_dir=data_dir)
    assert result == {"year": 2024, "promoted": 1, "refused": 0, "failed": 0}

    call = fake_insert[0]
    assert call["doc_id"] == "2001"
    assert call["year"] == 2024
    assert call["member"] == "Jane Doe"
    assert call["parser_version"] == glm_stage.GLM_PARSER_VERSION == "v1-glm53-vision"
    assert call["engine_model"] == glm_stage.GLM_ENGINES_ATTEMPTED == "visionb"
    assert call["artifact_sha256"] == sha
    assert call["ingestion_generation"] == "gen-1"
    assert call["raw_count"] == 1
    (tx,) = call["transactions"]
    assert tx["asset"] == "Apple Inc. (AAPL)"
    assert tx["type"] == "Purchase"
    assert tx["date"] == "01/15/24"
    assert tx["notif_date"] == "01/20/24"
    assert tx["amount_letter"] == "A"
    assert tx["amount_midpoint"] == 8000

    # Promotion consumed the staging record and journaled the run.
    assert glm_stage.load_staging_record(data_dir, 2024, "2001") is None
    journal = glm_stage.read_promoted_journal(glm_stage.staging_root(data_dir))
    assert [entry["doc_id"] for entry in journal] == ["2001"]
    assert journal[0]["inserted"] == 1
    assert journal[0]["staged_rows_sha256"] == glm_stage.staged_rows_sha256(
        [["Apple Inc. (AAPL)", "Purchase", "01/15/24", "01/20/24", "A"]]
    )
    archived = glm_stage.archived_attempts(data_dir, 2024, "2001")
    assert len(archived) == 1 and archived[0][1].exists()
    assert journal[0]["staging_attempt"].endswith(archived[0][1].name)
    # main()-style progress bookkeeping happened.
    progress = json.loads(
        (data_dir / "ocr_progress_gemini_manual.json").read_text()
    )
    assert progress["completed"] == ["2001"]


def test_promote_no_txs_records_and_consumes(data_dir, fake_insert):
    stage_one_audited(data_dir, 2024, "2002", GLM_OUTPUT_NO_TXS)
    result = glm_stage.promote_docs(2024, ["2002"], data_dir=data_dir)
    assert result["promoted"] == 1
    assert fake_insert[0]["transactions"] == []
    assert fake_insert[0]["raw_count"] == 0
    progress = json.loads(
        (data_dir / "ocr_progress_gemini_manual.json").read_text()
    )
    assert progress["no_txs"] == ["2002"]
    assert glm_stage.load_staging_record(data_dir, 2024, "2002") is None


def test_promote_audit_approval_is_mandatory(data_dir, fake_insert, capsys):
    # Per-record gate: with a passing report but no audit stamp on the
    # record, promotion is refused, the insert path never runs, and the
    # record stays for the audit.
    stage_one(data_dir, 2024, "2003", GLM_OUTPUT_ONE_ROW)
    write_report_state(data_dir)
    refused = glm_stage.promote_docs(2024, ["2003"], data_dir=data_dir)
    assert refused["refused"] == 1 and refused["promoted"] == 0
    assert fake_insert == []
    assert glm_stage.load_staging_record(data_dir, 2024, "2003") is not None
    assert "audit approval is mandatory" in capsys.readouterr().out

    # The legacy permissive flag is refused loudly, not silently honored.
    with pytest.raises(ValueError, match="mandatory"):
        glm_stage.promote_docs(
            2024, ["2003"], data_dir=data_dir, require_audit_pass=False
        )

    # A passing pixel audit stamps the staging record... but promotion
    # still needs the report-level gate: a missing audit report state
    # refuses (fail-closed), as do incomplete reports and a blown budget.
    audit_sample.record_verdict(data_dir, "2003", "pass", "matches pixels")
    record = glm_stage.load_staging_record(data_dir, 2024, "2003")
    assert record["audit"] == "pass"
    (glm_stage.staging_root(data_dir) / "audit-report-state.json").unlink()
    missing = glm_stage.promote_docs(2024, ["2003"], data_dir=data_dir)
    assert missing["refused"] == 1 and missing["promoted"] == 0
    assert "audit report state missing" in capsys.readouterr().out
    write_report_state(data_dir, state="incomplete", budget_met=None)
    incomplete = glm_stage.promote_docs(2024, ["2003"], data_dir=data_dir)
    assert incomplete["refused"] == 1 and incomplete["promoted"] == 0
    incomplete_out = capsys.readouterr().out
    assert "state incomplete because fixture: no verdict recorded" in incomplete_out
    assert "how to fix: record an explicit verdict" in incomplete_out
    assert "resolved-only diagnostics: 0/29 resolved (lot n=30)" in incomplete_out
    assert "95% upper bound" in incomplete_out and "max_rate 1%" in incomplete_out
    write_report_state(data_dir, state="complete", budget_met=False)
    budget_blown = glm_stage.promote_docs(2024, ["2003"], data_dir=data_dir)
    assert budget_blown["refused"] == 1 and budget_blown["promoted"] == 0
    # Complete report + met budget opens the gate.
    write_report_state(data_dir)  # complete + budget_met True
    promoted = glm_stage.promote_docs(2024, ["2003"], data_dir=data_dir)
    assert promoted["promoted"] == 1
    assert len(fake_insert) == 1
    assert glm_stage.load_staging_record(data_dir, 2024, "2003") is None


def test_promote_force_unaudited_is_explicit_and_loud(data_dir, fake_insert, capsys):
    stage_one(data_dir, 2024, "2004", GLM_OUTPUT_ONE_ROW)
    result = glm_stage.promote_docs(
        2024, ["2004"], data_dir=data_dir, force_unaudited=True
    )
    assert result["promoted"] == 1
    out = capsys.readouterr().out
    assert "WARNING" in out and "WITHOUT a passing pixel audit" in out
    journal = glm_stage.read_promoted_journal(glm_stage.staging_root(data_dir))
    assert journal[0]["unaudited"] is True
    assert glm_stage.load_staging_record(data_dir, 2024, "2004") is None
    archived = glm_stage.archived_attempts(data_dir, 2024, "2004")[0][1]
    assert journal[0]["staging_attempt"] == str(
        archived.relative_to(glm_stage.staging_root(data_dir))
    )
    assert journal[0]["staged_record_sha256"] == audit_sample.staged_record_sha256(
        json.loads(archived.read_bytes())
    )


@pytest.mark.parametrize("force_unaudited", [False, True])
def test_legacy_success_without_optional_row_hash_still_promotes(
    data_dir, fake_insert, force_unaudited
):
    stage_one(data_dir, 2024, "legacy-row-hash", GLM_OUTPUT_ONE_ROW)
    live = glm_stage.staging_record_path(data_dir, 2024, "legacy-row-hash")
    legacy = json.loads(live.read_text(encoding="utf-8"))
    del legacy["staged_rows_sha256"]
    live.write_text(json.dumps(legacy), encoding="utf-8")

    if not force_unaudited:
        audit_sample.record_verdict(
            data_dir, "legacy-row-hash", "pass", "legacy content reviewed"
        )
        write_report_state(data_dir)
    result = glm_stage.promote_docs(
        2024,
        ["legacy-row-hash"],
        data_dir=data_dir,
        force_unaudited=force_unaudited,
    )

    assert result["promoted"] == 1 and len(fake_insert) == 1
    (attempt_number, archived_path), = glm_stage.archived_attempts(
        data_dir, 2024, "legacy-row-hash"
    )
    archived = json.loads(archived_path.read_text(encoding="utf-8"))
    assert "staged_rows_sha256" not in archived  # never backfilled from live rows
    journal = glm_stage.read_promoted_journal(glm_stage.staging_root(data_dir))
    assert journal[0]["unaudited"] is force_unaudited
    assert journal[0]["staged_rows_sha256"] == glm_stage.staged_rows_sha256(
        archived["rows"]
    )
    assert attempt_number == 0
    if not force_unaudited:
        history = json.loads(
            (glm_stage.staging_root(data_dir) / "audit-results.json").read_text()
        )
        assert history[0]["staged_record_sha256"] == audit_sample.staged_record_sha256(
            legacy
        )


def test_promote_refuses_malformed_or_foreign_report_state(data_dir, fake_insert, capsys):
    stage_one_audited(data_dir, 2024, "2007", GLM_OUTPUT_ONE_ROW)
    # Malformed JSON is treated like a missing report: refuse, never guess.
    write_report_state(data_dir, raw="{not json")
    result = glm_stage.promote_docs(2024, ["2007"], data_dir=data_dir)
    assert result["refused"] == 1 and result["promoted"] == 0
    assert fake_insert == []
    assert "audit report state missing" in capsys.readouterr().out
    # A complete/true stamp without supporting lot/bound metrics is ambiguous.
    write_report_state(
        data_dir,
        raw=json.dumps(
            {
                "schema": "audit-report-state-v2",
                "state": "complete",
                "budget_met": True,
            }
        ),
    )
    result = glm_stage.promote_docs(2024, ["2007"], data_dir=data_dir)
    assert result["refused"] == 1 and result["promoted"] == 0
    out = capsys.readouterr().out
    assert "required gate metrics are missing or invalid" in out
    assert "lot_size" in out and "upper_bound" in out
    # A foreign schema version is equally untrusted.
    write_report_state(data_dir, schema="audit-report-state-v1")
    result = glm_stage.promote_docs(2024, ["2007"], data_dir=data_dir)
    assert result["refused"] == 1 and result["promoted"] == 0
    kept = glm_stage.load_staging_record(data_dir, 2024, "2007")
    assert kept is not None
    # force_unaudited is the only bypass, and it stays loud.
    result = glm_stage.promote_docs(
        2024, ["2007"], data_dir=data_dir, force_unaudited=True
    )
    assert result["promoted"] == 1
    out = capsys.readouterr().out
    assert "WARNING" in out and "WITHOUT a passing pixel audit" in out


def test_promote_refuses_sample_lot_replaced_after_report(
    data_dir, fake_insert, capsys
):
    stage_one_audited(data_dir, 2024, "lot-binding", GLM_OUTPUT_ONE_ROW)
    state_path = glm_stage.staging_root(data_dir) / "audit-report-state.json"
    state_bytes = state_path.read_bytes()
    state = json.loads(state_bytes)
    lot_path = glm_stage.staging_root(data_dir) / state["sample_file"]
    original_lot = lot_path.read_bytes()
    history_path = glm_stage.staging_root(data_dir) / "audit-results.json"
    history_bytes = history_path.read_bytes()

    replacement = json.loads(original_lot)
    replacement["frame_counts_by_year"]["2024"] += 1
    lot_path.write_text(json.dumps(replacement), encoding="utf-8")
    result = glm_stage.promote_docs(2024, ["lot-binding"], data_dir=data_dir)

    assert result["refused"] == 1 and result["promoted"] == 0
    assert fake_insert == []
    assert state_path.read_bytes() == state_bytes
    assert history_path.read_bytes() == history_bytes
    assert "frozen sample lot changed or is missing since report" in capsys.readouterr().out
    assert state["sample_file_sha256"] == hashlib.sha256(original_lot).hexdigest()


def test_promote_report_state_gate_refuses_whole_batch(data_dir, fake_insert, capsys):
    stage_one_audited(data_dir, 2024, "2008", GLM_OUTPUT_ONE_ROW)
    stage_one_audited(data_dir, 2024, "2009", GLM_OUTPUT_NO_TXS)
    (glm_stage.staging_root(data_dir) / "audit-report-state.json").unlink()
    result = glm_stage.promote_docs(2024, None, data_dir=data_dir)
    assert result == {"year": 2024, "promoted": 0, "refused": 2, "failed": 0}
    assert fake_insert == []
    assert "audit report state missing — run audit_sample.py report" in (
        capsys.readouterr().out
    )
    # Both records survive for the audit.
    assert glm_stage.load_staging_record(data_dir, 2024, "2008") is not None
    assert glm_stage.load_staging_record(data_dir, 2024, "2009") is not None


def test_promote_partial_insert_keeps_and_marks_record(data_dir, monkeypatch, capsys):
    stage_one_audited(data_dir, 2024, "2005", GLM_OUTPUT_ONE_ROW)
    # Simulate a partial insert: 1 of 2 staged rows validated and committed.
    record = glm_stage.load_staging_record(data_dir, 2024, "2005")
    record["rows"] = [
        ["Apple Inc. (AAPL)", "Purchase", "01/15/24", "01/20/24", "A"],
        ["Microsoft Corp (MSFT)", "Sale", "02/02/24", "02/05/24", "B"],
    ]
    record["row_pages"] = [1, 1]
    record["raw_count"] = 2
    record["audit"] = "pass"
    record["staged_rows_sha256"] = glm_stage.staged_rows_sha256(record["rows"])
    glm_stage.write_staging_record(data_dir, record)
    audit_sample.record_verdict(data_dir, "2005", "pass", "updated test fixture")
    write_report_state(data_dir)

    def partial_insert(*args, **kwargs):
        assert "report" not in kwargs
        return ocr_zero_rows.InsertOutcome(
            status="partial",
            inserted=1,
            expected=2,
            committed=True,
            rejections={"impossible_chronology": 1},
        )

    monkeypatch.setattr(ocr_zero_rows, "insert_transactions_outcome", partial_insert)
    monkeypatch.setattr(
        ocr_zero_rows,
        "resolve_ingestion_generation",
        lambda *a: "gen-1",
    )
    result = glm_stage.promote_docs(2024, ["2005"], data_dir=data_dir)
    assert result["failed"] == 1 and result["promoted"] == 0
    kept = glm_stage.load_staging_record(data_dir, 2024, "2005")
    assert kept is not None  # partial output is never consumed as complete
    assert kept["promotion"]["outcome"]["status"] == "partial"
    assert kept["promotion"]["outcome"]["inserted"] == 1
    assert kept["promotion"]["outcome"]["expected"] == 2
    assert kept["promotion"]["outcome"]["rejections"] == {
        "impossible_chronology": 1
    }
    assert glm_stage.verdict_binding(data_dir, "2005", kept) == (True, None)
    progress = json.loads(
        (data_dir / "ocr_progress_gemini_manual.json").read_text()
    )
    assert progress["errors"] == ["2005"]
    assert "partial" in capsys.readouterr().out


def test_promote_member_rejection_is_not_no_txs(data_dir, monkeypatch, capsys):
    stage_one_audited(data_dir, 2024, "2006", GLM_OUTPUT_ONE_ROW)
    record = glm_stage.load_staging_record(data_dir, 2024, "2006")
    record["audit"] = "pass"
    record["rows"] = []
    record["parse_status"] = "no_txs"
    record["row_pages"] = []
    record["staged_rows_sha256"] = glm_stage.staged_rows_sha256(record["rows"])
    glm_stage.write_staging_record(data_dir, record)
    audit_sample.record_verdict(data_dir, "2006", "pass", "updated test fixture")
    write_report_state(data_dir)

    def identity_reject(*args, **kwargs):
        return ocr_zero_rows.InsertOutcome(
            status="rejected",
            inserted=0,
            expected=0,
            rejections={"member_mismatch": 1},
        )

    monkeypatch.setattr(ocr_zero_rows, "insert_transactions_outcome", identity_reject)
    monkeypatch.setattr(
        ocr_zero_rows,
        "resolve_ingestion_generation",
        lambda *a: "gen-1",
    )
    result = glm_stage.promote_docs(2024, ["2006"], data_dir=data_dir)
    assert result["failed"] == 1 and result["promoted"] == 0
    kept = glm_stage.load_staging_record(data_dir, 2024, "2006")
    assert kept["promotion"]["outcome"]["status"] == "rejected"
    assert kept["promotion"]["outcome"]["rejections"] == {"member_mismatch": 1}
    progress = json.loads(
        (data_dir / "ocr_progress_gemini_manual.json").read_text()
    )
    # The rejection must land in errors, never in the no_txs bucket.
    assert progress["errors"] == ["2006"]
    assert progress["no_txs"] == []


def test_promote_refuses_error_records_and_sha_drift(data_dir, fake_insert):
    seed_year(data_dir, 2024, ["3001", "3002"])
    glm_stage.write_staging_record(
        data_dir,
        {
            "doc_id": "3001",
            "year": 2024,
            "pdf_sha256": "deadbeef",
            "model": None,
            "prompt_sha256": PROMPT_SHA256,
            "parser_version": "v1-glm53-vision",
            "parsed_at_utc": "2026-10-18T00:00:00Z",
            "pages": 1,
            "member": "Jane Doe",
            "rows": [],
            "raw_output": None,
            "parse_status": "error",
            "error": "boom",
            "audit": None,
        },
    )
    outputs = {"3002": GLM_OUTPUT_ONE_ROW}
    glm_stage.stage_batch(
        2024, 5, data_dir=data_dir, read_fn=read_fn_from_outputs(outputs)
    )
    # SHA drift: rewrite the artifact after staging.
    (data_dir / "2024" / "pdfs" / "3002.pdf").write_bytes(b"%PDF-1.4 tampered")

    result = glm_stage.promote_docs(2024, ["3001", "3002"], data_dir=data_dir)
    assert result["refused"] == 2 and result["promoted"] == 0
    assert fake_insert == []
    # Refused records are kept for a later attempt.
    assert glm_stage.load_staging_record(data_dir, 2024, "3001") is not None
    assert glm_stage.load_staging_record(data_dir, 2024, "3002") is not None


def test_promote_keeps_record_when_insert_rejects(data_dir, monkeypatch):
    stage_one_audited(data_dir, 2024, "4001", GLM_OUTPUT_ONE_ROW)

    def zero_insert(*args, **kwargs):
        return ocr_zero_rows.InsertOutcome(status="rejected", inserted=0, expected=1)

    monkeypatch.setattr(ocr_zero_rows, "insert_transactions_outcome", zero_insert)
    monkeypatch.setattr(
        ocr_zero_rows,
        "resolve_ingestion_generation",
        lambda *a: "gen-1",
    )
    result = glm_stage.promote_docs(2024, ["4001"], data_dir=data_dir)
    assert result["failed"] == 1 and result["promoted"] == 0
    assert glm_stage.load_staging_record(data_dir, 2024, "4001") is not None
    progress = json.loads(
        (data_dir / "ocr_progress_gemini_manual.json").read_text()
    )
    assert progress["errors"] == ["4001"]


def test_staging_status_counts(data_dir, fake_insert):
    seed_year(data_dir, 2024, ["5001", "5002", "5003"])
    outputs = {
        "5001": GLM_OUTPUT_ONE_ROW,
        "5002": GLM_OUTPUT_NO_TXS,
        "5003": GLM_OUTPUT_ONE_ROW,
    }

    def half_broken(pdf_path, page_count):
        if Path(pdf_path).stem == "5003":
            raise ValueError("nope")
        return outputs[Path(pdf_path).stem]

    glm_stage.stage_batch(2024, 10, data_dir=data_dir, read_fn=half_broken)
    audit_sample.record_verdict(data_dir, "5001", "pass", "matches pixels")
    write_report_state(data_dir)
    glm_stage.promote_docs(2024, ["5001"], data_dir=data_dir)
    rows = glm_stage.staging_status(data_dir)
    assert rows == [
        {
            "year": 2024,
            "staged": 2,
            "success": 0,
            "no_txs": 1,
            "errors": 1,
            "promoted": 1,
        }
    ]


def test_malformed_promotion_journal_blocks_explicit_promotion(data_dir, fake_insert):
    stage_one(data_dir, 2024, "journal", GLM_OUTPUT_ONE_ROW)
    audit_sample.record_verdict(data_dir, "journal", "pass", "matches pixels")
    write_report_state(data_dir)
    live = glm_stage.staging_record_path(data_dir, 2024, "journal")
    live_bytes = live.read_bytes()
    journal = glm_stage.staging_root(data_dir) / glm_stage.PROMOTED_JOURNAL_NAME
    journal.write_text('{"doc_id":"previous"}\n{broken\n', encoding="utf-8")
    journal_bytes = journal.read_bytes()

    with pytest.raises(ValueError, match="malformed promotion journal .*:2"):
        glm_stage.promote_docs(2024, ["journal"], data_dir=data_dir)

    assert fake_insert == []
    assert live.read_bytes() == live_bytes
    assert journal.read_bytes() == journal_bytes
    assert glm_stage.archived_attempts(data_dir, 2024, "journal") == []


# --------------------------------------------------------------------------
# audit: sampler determinism, bundle, verdicts, report math
# --------------------------------------------------------------------------


def write_staged_records(data_dir, spec: dict[int, list[str]]):
    for year, doc_ids in spec.items():
        for doc_id in doc_ids:
            make_pdf(data_dir / str(year) / "pdfs" / f"{doc_id}.pdf")
            glm_stage.write_staging_record(
                data_dir,
                {
                    "doc_id": doc_id,
                    "year": year,
                    "pdf_sha256": f"sha-{doc_id}",
                    "model": "glm-5.3-flash",
                    "prompt_sha256": PROMPT_SHA256,
                    "parser_version": "v1-glm53-vision",
                    "parsed_at_utc": "2026-10-18T00:00:00Z",
                    "pages": 1,
                    "member": "Jane Doe",
                    "rows": [["X", "Purchase", "01/15/24", None, "A"]],
                    "raw_output": None,
                    "parse_status": "success",
                    "error": None,
                    "audit": None,
                },
            )


def test_sampler_is_deterministic_and_stratified(data_dir):
    write_staged_records(
        data_dir, {2023: [f"a{i}" for i in range(6)], 2024: [f"b{i}" for i in range(4)]}
    )
    # An error record must never enter the sampling frame.
    glm_stage.write_staging_record(
        data_dir,
        {
            "doc_id": "bad",
            "year": 2023,
            "pdf_sha256": "sha-bad",
            "model": None,
            "prompt_sha256": PROMPT_SHA256,
            "parser_version": "v1-glm53-vision",
            "parsed_at_utc": "2026-10-18T00:00:00Z",
            "pages": 1,
            "member": None,
            "rows": [],
            "raw_output": None,
            "parse_status": "error",
            "error": "boom",
            "audit": None,
        },
    )
    records = audit_sample.load_staging_records(data_dir)
    sample_one = audit_sample.build_sample(records, per_year=2, seed=7)
    sample_again = audit_sample.build_sample(records, per_year=2, seed=7)
    assert sample_one == sample_again
    assert len(sample_one) == 4
    by_year = {2023: 0, 2024: 0}
    for item in sample_one:
        by_year[item["year"]] += 1
        assert item["doc_id"] != "bad"
    assert by_year == {2023: 2, 2024: 2}

    # Same seed -> same file on disk, byte-identical.
    path_one = audit_sample.staging_root(data_dir) / "audit-sample-7.json"
    audit_sample._atomic_write_json(path_one, sample_one)
    first_bytes = path_one.read_bytes()
    audit_sample._atomic_write_json(path_one, sample_again)
    assert path_one.read_bytes() == first_bytes


def test_review_bundle_copies_record_and_renders_pages(data_dir):
    write_staged_records(data_dir, {2024: ["6001"]})
    record = glm_stage.load_staging_record(data_dir, 2024, "6001")
    record["pdf_sha256"] = glm_stage.pdf_sha256(glm_stage.default_pdf_path(data_dir, 2024, "6001"))
    glm_stage.write_staging_record(data_dir, record)
    sample = [{"doc_id": "6001", "year": 2024}]
    problems = audit_sample.emit_review_bundle(sample, data_dir)
    assert problems == []
    doc_dir = audit_sample.staging_root(data_dir) / "audit-review" / "2024" / "6001"
    copied = json.loads((doc_dir / "record.json").read_text())
    assert copied["doc_id"] == "6001"
    assert (doc_dir / "pages").is_dir()


@pytest.mark.skipif(shutil.which("pdftoppm") is None, reason="poppler missing")
def test_poppler_glue_renders_pages(data_dir):
    pdf = make_pdf(data_dir / "2024" / "pdfs" / "7001.pdf", pages=3)
    urls = glm_stage.rasterize_page_data_urls(pdf)
    assert len(urls) == 3
    for url in urls:
        assert url.startswith("data:image/png;base64,")
        base64.b64decode(url.split(",", 1)[1])
        # Z.AI vision request schema: every image must stay under 5MB.
        assert len(url) < 5_000_000

    bundle_dir = data_dir / "bundle" / "7001"
    pages = audit_sample.render_pdf_pages(pdf, bundle_dir)
    assert [p.name for p in pages] == ["page-1.png", "page-2.png", "page-3.png"]


def test_record_verdict_appends_and_stamps(data_dir):
    write_staged_records(data_dir, {2024: ["8001"]})
    result = audit_sample.record_verdict(data_dir, "8001", "fail", "invented row")
    assert result["staging_year"] == 2024
    results = json.loads(
        (audit_sample.staging_root(data_dir) / "audit-results.json").read_text()
    )
    assert len(results) == 1
    assert results[0]["doc_id"] == "8001"
    assert results[0]["verdict"] == "fail"
    assert results[0]["notes"] == "invented row"
    assert results[0]["audited_at_utc"].endswith("Z")
    record = glm_stage.load_staging_record(data_dir, 2024, "8001")
    assert record["audit"] == "fail"
    with pytest.raises(ValueError):
        audit_sample.record_verdict(data_dir, "8001", "maybe", "")


def test_binomial_bound_zero_of_299_within_one_percent():
    # The FP-review case: 0 material errors in 299 docs -> one-sided exact
    # 95% upper bound 0.997% (< 1%).
    assert audit_sample.binomial_upper_bound(0, 299) <= 0.01
    exact = audit_sample._exact_upper_bound(0, 299, 0.95)
    assert exact == pytest.approx(1 - 0.05 ** (1 / 299))
    assert exact < 0.01
    # Both paths agree where scipy is available.
    try:
        from scipy.stats import beta

        scipy_value = float(beta.ppf(0.95, 1, 299))
        assert exact == pytest.approx(scipy_value, rel=1e-9)
    except ImportError:  # pragma: no cover
        pass
    # ...and any material error blows the illustrative budget at n=299.
    assert audit_sample.binomial_upper_bound(1, 299) > 0.01


def test_report_counts_omissions_and_budget():
    staged_rows = {
        "9001": [["A", "Purchase", "01/15/24", None, "A"]],
        "9002": [["B", "Sale", "02/02/24", None, "B"]],
        "9003": [["C", "Sale", "03/03/24", None, "C"]],
    }
    results = [
        {"doc_id": "9001", "verdict": "pass", "notes": "", "audited_at_utc": "t1"},
        {"doc_id": "9002", "verdict": "pass", "notes": "", "audited_at_utc": "t1"},
        {"doc_id": "9003", "verdict": "fail", "notes": "wrong date", "audited_at_utc": "t1"},
    ]
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        gt_dir = Path(td)
        # 9001: omission — ground truth has 2 rows, staged has 1.
        (gt_dir / "9001.json").write_text(
            json.dumps(
                {
                    "rows": [
                        ["A", "Purchase", "01/15/24", None, "A"],
                        ["A2", "Sale", "01/16/24", None, "A"],
                    ]
                }
            )
        )
        (gt_dir / "9002.json").write_text(json.dumps({"rows": staged_rows["9002"]}))
        report = audit_sample.build_report(
            results,
            staged_rows,
            promoted_doc_ids={"9001", "9002"},
            ground_truth_dir=gt_dir,
            max_rate=0.01,
        )
    assert report["audited"] == 3
    assert report["failure_count"] == 2
    assert ("9001", "omission: ground-truth rows 2 > staged rows 1") in report["failures"]
    assert ("9003", "verdict:fail") in report["failures"]
    assert report["clean"] == ["9002"]
    assert report["budget_met"] is False
    assert report["pooled_binomial_upper_bound"] == pytest.approx(
        audit_sample._exact_upper_bound(2, 3, 0.95)
    )
    # No frozen sampling design was supplied, so the acceptance bound fails
    # closed even though the pooled binomial diagnostic remains available.
    assert report["upper_bound"] == 1.0
    assert report["upper_bound_method"].startswith("fail-closed:")


def test_report_budget_met_for_clean_299_sample(data_dir):
    doc_ids = [f"d{i}" for i in range(299)]
    write_staged_records(data_dir, {2024: doc_ids})
    records = audit_sample.load_staging_records(data_dir)
    staged_by_doc = {str(record["doc_id"]): record for _, record in records}
    results = [
        {"doc_id": f"d{i}", "verdict": "pass", "notes": "", "audited_at_utc": "t"}
        for i in range(299)
    ]
    report = audit_sample.build_report(
        results,
        {
            doc_id: list(record.get("rows") or [])
            for doc_id, record in staged_by_doc.items()
        },
        promoted_doc_ids=set(doc_ids),
        ground_truth_dir=None,
        max_rate=0.01,
        lot=[
            {
                "doc_id": f"d{i}",
                "year": 2024,
                "asset_review": False,
                "staged_record_sha256": audit_sample.staged_record_sha256(
                    staged_by_doc[f"d{i}"]
                ),
            }
            for i in range(299)
        ],
        staged_records_by_doc=staged_by_doc,
        frame_counts_by_year={"2024": 299},
        frame_counts_by_stratum={"2024": {"flagged": 0, "unflagged": 299}},
    )
    assert report["failure_count"] == 0
    assert len(report["clean"]) == 299
    assert report["budget_met"] is True
    assert report["upper_bound"] <= 0.01


def test_live_30_doc_report_refusal_names_budget_math(
    data_dir, fake_insert, capsys
):
    doc_ids = [
        "20005631", "20006505", "20006545", "20006739", "20006747",
        "20006751", "20006839", "20006944", "20006996", "20007206",
        "20007208", "20007359", "20007420", "20007551", "20007588",
        "20007592", "20007596", "20007702", "20007716", "20007829",
        "20007899", "20007964", "20007971", "20008047", "20008135",
        "20008194", "20008277", "20008431", "20008488", "20008556",
    ]
    write_staged_records(data_dir, {2017: doc_ids})
    records = audit_sample.load_staging_records(data_dir)
    sample = [
        {"doc_id": record["doc_id"], "year": year}
        for year, record in records
    ]
    root = glm_stage.staging_root(data_dir)
    sample_path = root / "audit-sample-20261019.json"
    audit_sample.write_sample_file(
        records,
        sample,
        seed=20261019,
        per_year=30,
        out_path=sample_path,
    )
    # Model a larger frozen population than the 30-document audit sample so
    # the finite-population upper bound remains above the 1% acceptance cap.
    envelope = json.loads(sample_path.read_text())
    envelope["frame_counts_by_year"] = {"2017": 300}
    envelope["frame_counts_by_stratum"] = {
        "2017": {"flagged": 0, "unflagged": 300}
    }
    sample_path.write_text(json.dumps(envelope), encoding="utf-8")
    cosmetic_doc = "20006747"
    for doc_id in doc_ids:
        if doc_id == cosmetic_doc:
            continue
        audit_sample.record_verdict(data_dir, doc_id, "pass", "pixel match")
    # Re-recorded cosmetic failure remains resolved but does not count as critical.
    audit_sample.record_verdict(
        data_dir, cosmetic_doc, "fail", "asset text only", severity="cosmetic"
    )
    audit_sample.record_verdict(
        data_dir, cosmetic_doc, "fail", "reconfirmed text-only", severity="cosmetic"
    )
    # The real eight prior-lot verdicts must remain in the population tally.
    for doc_id in ("20006438", "20006516"):
        audit_sample.record_verdict(
            data_dir, doc_id, "fail", "reclassified text-only", severity="cosmetic"
        )
    audit_sample.record_verdict(
        data_dir,
        "20006575",
        "fail",
        "transaction_type Sale-vs-Purchase",
        severity="critical",
    )
    audit_sample.record_verdict(
        data_dir, "20006575", "pass", "later re-check; sticky fail remains"
    )
    for doc_id in ("20006470", "20006657", "20006682", "20006694", "20006832"):
        audit_sample.record_verdict(data_dir, doc_id, "pass", "historical pass")

    report_rc = audit_sample.main(
        ["--data-dir", str(data_dir), "report", "--max-rate", "0.01"]
    )
    report_state = glm_stage.audit_report_state(data_dir)
    assert report_rc == 1
    assert report_state["state"] == "complete"
    assert report_state["budget_met"] is False
    assert report_state["critical_failure_count"] == 0
    assert report_state["audited"] == 30
    assert report_state["upper_bound"] == pytest.approx(
        audit_sample._hypergeometric_upper_failures(0, 30, 300, 0.05) / 300
    )
    assert report_state["outside_lot"] == sorted(
        [
            "20006438", "20006470", "20006516", "20006575", "20006657",
            "20006682", "20006694", "20006832",
        ]
    )
    assert report_state["population_critical_failure_count"] == 1
    capsys.readouterr()

    result = glm_stage.promote_docs(2017, [doc_ids[0]], data_dir=data_dir)
    out = capsys.readouterr().out
    assert result == {"year": 2017, "promoted": 0, "refused": 1, "failed": 0}
    assert fake_insert == []
    assert "complete but the critical error budget is not met" in out
    assert "critical 0/30 resolved (lot n=30)" in out
    assert "rate 0%" in out
    assert "95% one-sided upper bound" in out
    assert "max_rate 1%" in out
    assert "budget_met=False" in out
    assert "population sticky critical-failure tally 1 across audit history" in out
    assert "outside-lot failures remain visible, excluded from the current-lot bound" in out
    assert "omission check not run" in out


# --------------------------------------------------------------------------
# CLI contract
# --------------------------------------------------------------------------


def test_cli_help_clean(capsys):
    with pytest.raises(SystemExit) as exc:
        glm_stage.main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "stage" in out and "promote" in out and "status" in out

    with pytest.raises(SystemExit) as exc:
        glm_stage.main(["stage", "--help"])
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert "--check-canonical" in help_text
    assert "--limit" in help_text


def test_cli_stage_without_endpoint_config_fails_cleanly(
    data_dir, monkeypatch, capsys
):
    for var in (
        "VISION_B_BASE_URL",
        "VISION_B_MODEL",
        "VISION_B_API_KEY",
        "VISION_B_TIMEOUT",
    ):
        monkeypatch.delenv(var, raising=False)
    seed_year(data_dir, 2024, ["x1"])
    rc = glm_stage.main(["--data-dir", str(data_dir), "stage", "--year", "2024"])
    assert rc == 1
    assert "not configured" in capsys.readouterr().err
    assert list((data_dir / "staging").rglob("*.json")) == []


# --------------------------------------------------------------------------
# finding 6: staging provenance + verdict binding
# --------------------------------------------------------------------------


def test_staging_records_carry_render_dpi_and_row_pages(data_dir):
    make_pdf(data_dir / "2024" / "pdfs" / "pg1.pdf", pages=2)
    record = glm_stage.build_staging_record(
        "pg1",
        2024,
        glm_stage.default_pdf_path(data_dir, 2024, "pg1"),
        None,
        read_fn=lambda *a: GLM_OUTPUT_PAGE2,
    )
    assert record["parse_status"] == "success"
    assert record["render_dpi"] == glm_stage.RENDER_DPI == 150
    # Per-row page attribution, aligned with rows: the row was read on page 2.
    assert record["row_pages"] == [2]
    assert len(record["rows"]) == len(record["row_pages"]) == 1
    assert record["rows"][0][0] == "Microsoft Corp (MSFT)"


def test_schema4_staging_roundtrip_preserves_every_raw_field(data_dir):
    seed_year(data_dir, 2024, ["lossless"])
    row = {
        "asset": "Apple Inc. (AAPL)", "type": "Purchase", "date": "01/15/24",
        "notif_date": None, "amount_raw": "$12,345.67", "owner": "SP",
        "account": "Joint account", "instrument_type": "call", "strike_price": "100",
        "expiry_date": "2025-01-17", "option_terms": {"contracts": "3", "verbatim": "100 calls"},
        "raw_cells": {"Amount": "$12,345.67", "Owner": "SP", "Extra": "untouched"},
    }
    raw = "SCHEMA: 4\nMEMBER: Jane Doe\nPAGES: 1\nPAGE: 1\nROW: " + json.dumps(row)
    record = glm_stage.build_staging_record(
        "lossless", 2024, glm_stage.default_pdf_path(data_dir, 2024, "lossless"),
        None, read_fn=lambda *args: raw,
    )
    assert record["parse_status"] == "success", record["error"]
    assert record["output_schema_version"] == 4
    assert record["legacy_limited"] is False
    assert record["rows"] == [{**row, "page_number": 1}]
    assert record["raw_output"] == raw
    assert record["row_pages"] == [1]
    glm_stage.publish_staging_record(data_dir, record)
    restored = glm_stage.load_staging_record(data_dir, 2024, "lossless")
    assert restored == record
    assert glm_stage.staged_rows_sha256(restored["rows"]) == record["staged_rows_sha256"]
    changed = json.loads(json.dumps(restored["rows"]))
    changed[0]["owner"] = "JT"
    assert glm_stage.staged_rows_sha256(changed) != record["staged_rows_sha256"]


def test_error_records_retain_raw_output_as_evidence(data_dir):
    seed_year(data_dir, 2024, ["evid1"])
    invalid = "MEMBER: Jane Doe\nPAGES: 1\nPAGE: 1\ngarbage line\n"
    record = glm_stage.build_staging_record(
        "evid1",
        2024,
        glm_stage.default_pdf_path(data_dir, 2024, "evid1"),
        None,
        read_fn=lambda *a: invalid,
    )
    assert record["parse_status"] == "error"
    assert record["error_kind"] == "parse"
    # The invalid response text is retained: it is evidence for the audit.
    assert record["raw_output"] == invalid


def test_record_binds_sha256_to_the_exact_bytes_the_render_saw(
    data_dir, monkeypatch
):
    seed_year(data_dir, 2024, ["snap1"])
    rendered = {}

    def fake_rasterize(pdf_path, dpi=glm_stage.RENDER_DPI):
        data = Path(pdf_path).read_bytes()
        rendered["sha"] = hashlib.sha256(data).hexdigest()
        rendered["dpi"] = dpi
        return ["data:image/png;base64,QUJD"]

    monkeypatch.setattr(glm_stage, "rasterize_page_data_urls", fake_rasterize)
    monkeypatch.setattr(
        glm_stage, "read_document_with_vision_b", lambda *a, **k: GLM_OUTPUT_ONE_ROW
    )
    config = VisionBConfig(base_url="http://127.0.0.1:1", model="stub")
    record = glm_stage.build_staging_record(
        "snap1", 2024, glm_stage.default_pdf_path(data_dir, 2024, "snap1"), config
    )
    assert record["parse_status"] == "success"
    # The hashed artifact IS the artifact the model rendered from — one
    # snapshot, not a hash of the live path read separately from the render.
    assert rendered["sha"] == record["pdf_sha256"]
    assert rendered["dpi"] == record["render_dpi"] == 150
    assert record["pdf_sha256"] == glm_stage.pdf_sha256(
        data_dir / "2024" / "pdfs" / "snap1.pdf"
    )


def test_archived_attempts_are_not_live_records(data_dir):
    write_staged_records(data_dir, {2024: ["iso1"]})
    glm_stage.archive_current_attempt(data_dir, 2024, "iso1")
    write_staged_records(data_dir, {2024: ["iso1"]})  # recreated live record
    glm_stage.archive_current_attempt(data_dir, 2024, "iso1")
    assert glm_stage.load_staging_record(data_dir, 2024, "iso1") is None
    assert [n for n, _ in glm_stage.archived_attempts(data_dir, 2024, "iso1")] == [0, 1]
    # Historical attempts must never leak into live-record readers (status,
    # promote-by-year, the audit sampling frame).
    assert glm_stage.iter_staging_records(data_dir) == []
    assert audit_sample.load_staging_records(data_dir) == []
    assert glm_stage.staging_status(data_dir) == []
    archived_bytes = [
        path.read_bytes()
        for _, path in glm_stage.archived_attempts(data_dir, 2024, "iso1")
    ]
    assert glm_stage.archive_current_attempt(data_dir, 2024, "iso1") is None
    assert [
        path.read_bytes()
        for _, path in glm_stage.archived_attempts(data_dir, 2024, "iso1")
    ] == archived_bytes


def test_archive_publication_never_replaces_a_concurrent_attempt(data_dir, monkeypatch):
    write_staged_records(data_dir, {2024: ["exclusive"]})
    live = glm_stage.staging_record_path(data_dir, 2024, "exclusive")
    live_bytes = live.read_bytes()
    first_target = glm_stage.attempts_dir(data_dir, 2024) / "exclusive.attempt-0.json"
    original_link = glm_stage.os.link
    injected = False

    def occupy_first_number(source, destination):
        nonlocal injected
        if Path(destination) == first_target and not injected:
            first_target.write_bytes(b"concurrent-attempt")
            injected = True
        return original_link(source, destination)

    monkeypatch.setattr(glm_stage.os, "link", occupy_first_number)
    archived = glm_stage.archive_current_attempt(data_dir, 2024, "exclusive")
    assert injected
    assert archived == glm_stage.attempts_dir(data_dir, 2024) / "exclusive.attempt-1.json"
    assert first_target.read_bytes() == b"concurrent-attempt"
    assert archived.read_bytes() == live_bytes
    assert not live.exists()


def test_concurrent_restage_publications_preserve_every_attempt(data_dir):
    seed_year(data_dir, 2024, ["restage-race"])
    pdf = glm_stage.default_pdf_path(data_dir, 2024, "restage-race")

    def initial_failure(*_):
        raise ValueError("initial failure")

    initial = glm_stage.build_staging_record(
        "restage-race", 2024, pdf, None, read_fn=initial_failure
    )
    glm_stage.write_staging_record(data_dir, initial)
    initial_bytes = glm_stage.staging_record_path(data_dir, 2024, "restage-race").read_bytes()
    inference_barrier = threading.Barrier(2)
    outputs = {
        "restage-a": GLM_OUTPUT_ONE_ROW,
        "restage-b": (
            "MEMBER: Jane Doe\nPAGES: 1\nPAGE: 1\n"
            "Microsoft Corp (MSFT) | Sale | 02/02/24 | 02/05/24 | B\n"
        ),
    }

    def read_fn(*_):
        inference_barrier.wait(timeout=5)
        return outputs[threading.current_thread().name]

    results = []
    errors = []

    def restage(name):
        threading.current_thread().name = name
        try:
            results.append(
                glm_stage.stage_batch(
                    2024,
                    1,
                    data_dir=data_dir,
                    manifest=[{"doc_id": "restage-race", "year": 2024}],
                    read_fn=read_fn,
                )
            )
        except BaseException as exc:
            errors.append(exc)

    workers = [
        threading.Thread(target=restage, args=(name,))
        for name in outputs
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=15)
    assert not any(worker.is_alive() for worker in workers)
    assert errors == []
    assert len(results) == 2 and all(result["staged"] == 1 for result in results)

    attempts = glm_stage.archived_attempts(data_dir, 2024, "restage-race")
    assert [number for number, _ in attempts] == [0, 1]
    assert attempts[0][1].read_bytes() == initial_bytes
    published = {attempts[1][1].read_bytes(), glm_stage.staging_record_path(
        data_dir, 2024, "restage-race"
    ).read_bytes()}
    assert len(published) == 2
    assert {json.loads(raw)["raw_output"] for raw in published} == set(outputs.values())


def test_unreadable_live_record_blocks_population_consumption(data_dir, fake_insert):
    write_staged_records(data_dir, {2024: ["population"]})
    archived = glm_stage.archive_current_attempt(data_dir, 2024, "population")
    assert archived is not None
    archived_bytes = archived.read_bytes()
    write_staged_records(data_dir, {2024: ["population"]})
    live = glm_stage.staging_record_path(data_dir, 2024, "population")
    live_bytes = live.read_bytes()
    bad = glm_stage.staging_root(data_dir) / "2024" / "unreadable.json"
    bad_bytes = b"{broken"
    bad.write_bytes(bad_bytes)

    with pytest.raises(ValueError, match="unreadable staging record .*unreadable.json"):
        glm_stage.iter_staging_records(data_dir)
    with pytest.raises(ValueError, match="unreadable staging record .*unreadable.json"):
        audit_sample.load_staging_records(data_dir)
    with pytest.raises(ValueError, match="unreadable staging record .*unreadable.json"):
        glm_stage.promote_docs(2024, ["population"], data_dir=data_dir)

    assert fake_insert == []
    assert live.read_bytes() == live_bytes
    assert bad.read_bytes() == bad_bytes
    assert archived.read_bytes() == archived_bytes


def test_staging_record_contract_rejects_semantic_corruption(data_dir):
    error_record = {
        "doc_id": "legacy-error", "year": 2024, "pdf_sha256": None,
        "model": None, "prompt_sha256": PROMPT_SHA256,
        "parsed_at_utc": "2026-10-18T00:00:00Z",
        "parser_version": "v1-glm53-vision", "pages": None,
        "member": None, "rows": [], "raw_output": "invalid response",
        "parse_status": "error", "error": "parse failed", "audit": None,
    }
    # Legacy error records need neither the additive attempt metadata nor
    # promotion-only artifact bindings; failed work remains in the population.
    assert glm_stage.decode_staging_record(json.dumps(error_record)) == error_record
    error_path = glm_stage.staging_record_path(data_dir, 2024, "legacy-error")
    error_path.parent.mkdir(parents=True, exist_ok=True)
    error_path.write_text(json.dumps(error_record), encoding="utf-8")
    assert glm_stage.iter_staging_records(data_dir) == [(2024, error_record)]

    for payload in (
        {},
        {**error_record, "year": True},
        {**error_record, "parse_status": "unknown"},
        {**error_record, "rows": {"not": "a row list"}},
        {**error_record, "error": None},
    ):
        with pytest.raises(ValueError, match="staging record"):
            glm_stage.decode_staging_record(json.dumps(payload))

    # A syntactically valid JSON object is still a failed live record, not an
    # ignored file or an empty population entry.
    malformed_path = glm_stage.staging_root(data_dir) / "2024" / "empty.json"
    malformed_path.parent.mkdir(parents=True, exist_ok=True)
    malformed_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="unreadable staging record .*empty.json"):
        glm_stage.iter_staging_records(data_dir)
    malformed_path.write_text(json.dumps(error_record), encoding="utf-8")
    with pytest.raises(ValueError, match="identity does not match its live path"):
        glm_stage.iter_staging_records(data_dir)


def test_semantically_invalid_peer_blocks_explicit_promotion(data_dir, fake_insert):
    write_staged_records(data_dir, {2024: ["explicit-peer"]})
    record = glm_stage.staging_record_path(data_dir, 2024, "explicit-peer")
    record_bytes = record.read_bytes()
    peer = glm_stage.staging_root(data_dir) / "2024" / "broken-peer.json"
    peer.write_text("{}", encoding="utf-8")
    peer_bytes = peer.read_bytes()

    with pytest.raises(ValueError, match="missing required fields"):
        glm_stage.promote_docs(2024, ["explicit-peer"], data_dir=data_dir)

    assert fake_insert == []
    assert record.read_bytes() == record_bytes
    assert peer.read_bytes() == peer_bytes
    assert glm_stage.archived_attempts(data_dir, 2024, "explicit-peer") == []


def test_force_unaudited_does_not_skip_staged_rows_hash(data_dir, fake_insert):
    stage_one(data_dir, 2024, "content-proof", GLM_OUTPUT_ONE_ROW)
    record = glm_stage.load_staging_record(data_dir, 2024, "content-proof")
    record["rows"][0][0] = "tampered after staging"
    glm_stage.write_staging_record(data_dir, record)
    live = glm_stage.staging_record_path(data_dir, 2024, "content-proof")
    live_bytes = live.read_bytes()

    with pytest.raises(ValueError, match="do not match staged_rows_sha256"):
        glm_stage.promote_docs(
            2024, ["content-proof"], data_dir=data_dir, force_unaudited=True
        )

    assert fake_insert == []
    assert live.read_bytes() == live_bytes
    live.unlink()  # the deliberately corrupt peer blocks population-wide admission

    stage_one(data_dir, 2024, "identity-proof", GLM_OUTPUT_ONE_ROW)
    pdf = glm_stage.default_pdf_path(data_dir, 2024, "identity-proof")
    make_pdf(pdf, pages=2)
    refused = glm_stage.promote_docs(
        2024, ["identity-proof"], data_dir=data_dir, force_unaudited=True
    )
    assert refused["refused"] == 1 and refused["promoted"] == 0
    assert fake_insert == []


def test_report_replaces_stale_complete_state_after_invalid_population(data_dir):
    write_staged_records(data_dir, {2024: ["report-peer"]})
    assert audit_sample.main(
        [
            "--data-dir", str(data_dir), "sample", "--per-year", "1",
            "--seed", "91", "--no-bundle",
        ]
    ) == 0
    audit_sample.record_verdict(data_dir, "report-peer", "pass", "verified")
    assert audit_sample.main(["--data-dir", str(data_dir), "report"]) == 0
    report_path = glm_stage.staging_root(data_dir) / "audit-report-state.json"
    assert json.loads(report_path.read_text(encoding="utf-8"))["state"] == "complete"

    glm_stage.staging_record_path(data_dir, 2024, "report-peer").write_text(
        "{}", encoding="utf-8"
    )
    assert audit_sample.main(["--data-dir", str(data_dir), "report"]) == 2
    state = json.loads(report_path.read_text(encoding="utf-8"))
    assert state["state"] == "incomplete"
    assert state["budget_met"] is None
    assert "missing required fields" in state["reason"]


def test_restage_preserves_all_archived_attempt_bytes(data_dir):
    seed_year(data_dir, 2024, ["cap1"])
    manifest = [{"doc_id": "cap1", "year": 2024}]
    counter = {"n": 0}
    expected_archives = {}

    def failing_read(pdf_path, page_count):
        counter["n"] += 1
        raise ValueError(f"boom {counter['n']}")

    # Six manifest-mode re-stages of the still-unresolved doc: every write
    # archives the previous attempt instead of overwriting it.
    for _ in range(6):
        live = glm_stage.staging_record_path(data_dir, 2024, "cap1")
        if live.exists():
            expected_archives[len(expected_archives)] = live.read_bytes()
        result = glm_stage.stage_batch(
            2024,
            5,
            data_dir=data_dir,
            read_fn=failing_read,
            manifest=manifest,
            cooldown=0,
        )
        assert result["errors"] == 1
    live = glm_stage.load_staging_record(data_dir, 2024, "cap1")
    assert live["parse_status"] == "error"
    assert live["error"] == "boom 6"  # newest attempt keeps the canonical name
    archives = glm_stage.archived_attempts(data_dir, 2024, "cap1")
    assert [n for n, _ in archives] == list(range(5))
    assert [path.read_bytes() for _, path in archives] == [
        expected_archives[number] for number, _ in archives
    ]
    assert [hashlib.sha256(path.read_bytes()).hexdigest() for _, path in archives] == [
        hashlib.sha256(expected_archives[number]).hexdigest() for number, _ in archives
    ]
    contents = [json.loads(path.read_text()) for _, path in archives]
    assert [record["error"] for record in contents] == [
        f"boom {attempt}" for attempt in range(1, 6)
    ]


def test_archive_keeps_bytes_when_audit_reference_publishes_mid_archive(
    data_dir, monkeypatch
):
    write_staged_records(data_dir, {2024: ["race1"]})
    live = glm_stage.staging_record_path(data_dir, 2024, "race1")
    for _ in range(4):
        glm_stage.archive_current_attempt(data_dir, 2024, "race1")
        write_staged_records(data_dir, {2024: ["race1"]})
    old_archives = glm_stage.archived_attempts(data_dir, 2024, "race1")
    old_bytes = {number: path.read_bytes() for number, path in old_archives}

    audit_path = glm_stage.staging_root(data_dir) / glm_stage.AUDIT_RESULTS_NAME
    staged_bytes = live.read_bytes()
    target = glm_stage.attempts_dir(data_dir, 2024) / "race1.attempt-4.json"
    original_link = glm_stage.os.link

    def publish_reference_after_archive(source, destination):
        result = original_link(source, destination)
        if Path(source) == live and Path(destination) == target:
            glm_stage.atomic_write_json(
                audit_path,
                [
                    {
                        "doc_id": "race1",
                        "staged_record_sha256": audit_sample.staged_record_sha256(
                            json.loads(old_bytes[0])
                        ),
                    }
                ],
            )
        return result

    # Deterministically interleave reference publication after the archive
    # rename, while archive_current_attempt is still executing.
    monkeypatch.setattr(glm_stage.os, "link", publish_reference_after_archive)
    assert glm_stage.archive_current_attempt(data_dir, 2024, "race1") == target
    assert target.read_bytes() == staged_bytes
    assert {
        number: path.read_bytes()
        for number, path in glm_stage.archived_attempts(data_dir, 2024, "race1")
        if number < 4
    } == old_bytes
    verdict = json.loads(audit_path.read_text(encoding="utf-8"))[0]
    assert verdict["staged_record_sha256"] == audit_sample.staged_record_sha256(
        json.loads(old_bytes[0])
    )


def test_archive_never_needs_a_lossy_restore_after_reference_race(
    data_dir, monkeypatch
):
    write_staged_records(data_dir, {2024: ["race2"]})
    for _ in range(4):
        glm_stage.archive_current_attempt(data_dir, 2024, "race2")
        write_staged_records(data_dir, {2024: ["race2"]})
    archives = glm_stage.archived_attempts(data_dir, 2024, "race2")
    original_bytes = {number: path.read_bytes() for number, path in archives}
    oldest = archives[0][1]
    record = json.loads(oldest.read_bytes())
    live = glm_stage.staging_record_path(data_dir, 2024, "race2")
    live_bytes = live.read_bytes()
    audit_path = glm_stage.staging_root(data_dir) / glm_stage.AUDIT_RESULTS_NAME
    original_unlink = Path.unlink

    def publish_after_unlink(path, *args, **kwargs):
        result = original_unlink(path, *args, **kwargs)
        if path == oldest:
            glm_stage.atomic_write_json(
                audit_path,
                [
                    {
                        "doc_id": "race2",
                        "staged_record_sha256": audit_sample.staged_record_sha256(
                            record
                        ),
                    }
                ],
            )
        return result

    def fail_restore(path, data):
        if path == oldest:
            raise OSError("simulated crash during attempt restoration")
        return original_write_bytes(path, data)

    original_write_bytes = Path.write_bytes
    monkeypatch.setattr(Path, "unlink", publish_after_unlink)
    monkeypatch.setattr(Path, "write_bytes", fail_restore)

    target = glm_stage.archive_current_attempt(data_dir, 2024, "race2")
    assert target is not None and target.exists()
    assert target.read_bytes() == live_bytes
    assert {
        number: path.read_bytes()
        for number, path in glm_stage.archived_attempts(data_dir, 2024, "race2")
        if number < 4
    } == original_bytes
    assert oldest.read_bytes() == original_bytes[0]


def test_dir_scan_mode_does_not_retry_error_records(data_dir):
    seed_year(data_dir, 2024, ["scanerr"])
    manifest = [{"doc_id": "scanerr", "year": 2024}]

    def failing_read(pdf_path, page_count):
        raise ValueError("boom")

    glm_stage.stage_batch(
        2024, 5, data_dir=data_dir, read_fn=failing_read,
        manifest=manifest, cooldown=0,
    )
    # Manual dir-scan mode keeps legacy semantics: staged docs (errors
    # included) are skipped. Only the authoritative manifest queue retries.
    again = glm_stage.stage_batch(
        2024, 5, data_dir=data_dir, read_fn=failing_read, cooldown=0
    )
    assert again["candidates"] == 0


def test_manifest_mode_drains_authoritative_queue_and_notes_missing_pdfs(
    data_dir, capsys
):
    seed_year(data_dir, 2024, ["1001", "1002"])
    outputs = {"1001": GLM_OUTPUT_ONE_ROW, "1002": GLM_OUTPUT_NO_TXS}
    read_fn = read_fn_from_outputs(outputs)
    # 1002 is already staged (staged docs await the audit; re-reading them
    # would burn quota), yet a manifest generated before that staging still
    # lists it: the live-record skip wins.
    record = glm_stage.build_staging_record(
        "1002", 2024, glm_stage.default_pdf_path(data_dir, 2024, "1002"),
        None, read_fn=read_fn,
    )
    glm_stage.write_staging_record(data_dir, record)

    manifest = [
        {"doc_id": "1001", "year": 2024},
        {"doc_id": "1002", "year": 2024},  # already staged -> skipped
        {"doc_id": "zz", "year": 2024},  # no local PDF -> note, never silent
        {"doc_id": "2019-doc", "year": 2019},  # other year -> ignored
    ]
    result = glm_stage.stage_batch(
        2024, 10, data_dir=data_dir, read_fn=read_fn,
        manifest=manifest, cooldown=0,
    )
    assert result["candidates"] == 1
    assert result["staged"] == 1
    assert result["missing_pdfs"] == 1
    staged_record = glm_stage.load_staging_record(data_dir, 2024, "1001")
    assert staged_record["parse_status"] == "success"
    assert glm_stage.load_staging_record(data_dir, 2024, "zz") is None
    out = capsys.readouterr().out
    assert "[zz] manifest entry has no local PDF" in out


def test_load_work_manifest_items_fail_closed(data_dir):
    root = glm_stage.staging_root(data_dir)
    items, error = glm_stage.load_work_manifest_items(root / "work-manifest.json")
    assert items == [] and error and "unreadable" in error

    root.mkdir(parents=True, exist_ok=True)
    foreign = root / "wm.json"
    foreign.write_text(json.dumps({"schema": "other-v1", "items": []}))
    items, error = glm_stage.load_work_manifest_items(foreign)
    assert items == [] and "not a glm-work-manifest-v1" in error

    foreign.write_text(
        json.dumps({"schema": glm_stage.WORK_MANIFEST_SCHEMA, "items": [{"doc_id": "a"}]})
    )
    items, error = glm_stage.load_work_manifest_items(foreign)
    assert items == [] and "malformed manifest item" in error

    foreign.write_text(
        json.dumps(
            {"schema": glm_stage.WORK_MANIFEST_SCHEMA, "items": [{"doc_id": "a", "year": 2024}]}
        )
    )
    items, error = glm_stage.load_work_manifest_items(foreign)
    assert error is None
    assert items == [{"doc_id": "a", "year": 2024}]


def test_cli_stage_from_manifest_refuses_unusable_manifest(data_dir, capsys):
    seed_year(data_dir, 2024, ["x1"])
    rc = glm_stage.main(
        [
            "--data-dir", str(data_dir), "stage", "--year", "2024",
            "--from-manifest", str(data_dir / "nope.json"),
        ]
    )
    assert rc == 2
    assert "work manifest" in capsys.readouterr().err


def test_manifest_years_are_data_derived_and_validated():
    years, error = glm_stage.manifest_years(
        [
            {"doc_id": "a", "year": 2014},
            {"doc_id": "b", "year": 2025},
            {"doc_id": "c", "year": 2014},
        ]
    )
    assert error is None and years == [2025, 2014]
    years, error = glm_stage.manifest_years([{"doc_id": "bad", "year": "2014"}])
    assert years == [] and "malformed manifest year" in error


def test_cli_manifest_dry_run_checks_canonical_and_reports_missing_directory(
    data_dir, monkeypatch, capsys
):
    seed_year(data_dir, 2024, ["queued", "staged", "canonical", "directory-only"])
    staged = glm_stage.build_staging_record(
        "staged",
        2024,
        glm_stage.default_pdf_path(data_dir, 2024, "staged"),
        None,
        read_fn=lambda *_: GLM_OUTPUT_ONE_ROW,
    )
    staged.update(archive_year=2024, ingestion_generation="current")
    glm_stage.write_staging_record(data_dir, staged)
    db_path, digests = seed_house_queue(data_dir, ["canonical", "staged"])
    from analyzer.persistence.database import Database

    db = Database(db_path)
    ocr_zero_rows.record_parse_run(
        db.conn, "canonical", 2024, "no_txs", 0, 0,
        parser_version="v1-glm53-vision", engines_attempted="visionb",
        artifact_sha256=digests["canonical"], ingestion_generation="current",
    )
    db.close()
    manifest_path = data_dir / "work-manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": glm_stage.WORK_MANIFEST_SCHEMA,
                "items": [
                    {"doc_id": "queued", "year": 2024},
                    {"doc_id": "staged", "year": 2024},
                    {"doc_id": "canonical", "year": 2024},
                    {"doc_id": "missing-pdf", "year": 2024},
                ],
            }
        )
    )
    rc = glm_stage.main(
        [
            "--data-dir",
            str(data_dir),
            "stage",
            "--year",
            "2024",
            "--from-manifest",
            str(manifest_path),
            "--check-canonical",
            "--dry-run",
        ]
    )
    assert rc == 0
    output = capsys.readouterr().out
    assert "DRY RUN year=2024: manifest_candidates=1" in output
    assert "missing_pdfs=1" in output
    assert glm_stage.load_staging_record(data_dir, 2024, "queued") is None
    assert glm_stage.load_staging_record(data_dir, 2024, "directory-only") is None

    # A missing year directory is an explicit counted skip, not empty success.
    empty_manifest = data_dir / "empty-manifest.json"
    empty_manifest.write_text(
        json.dumps({"schema": glm_stage.WORK_MANIFEST_SCHEMA, "items": []})
    )
    rc = glm_stage.main(
        [
            "--data-dir",
            str(data_dir),
            "stage",
            "--year",
            "2012",
            "--from-manifest",
            str(empty_manifest),
            "--dry-run",
        ]
    )
    assert rc == 2
    output = capsys.readouterr().out
    assert "missing PDF directory" in output
    assert "missing_dirs=1" in output


def test_manifest_terminal_skip_requires_current_identity_source_count_and_provenance(data_dir):
    from analyzer.persistence.database import Database

    complete = [f"empty-{i}" for i in range(11)] + ["success", "invalid"]
    outstanding = ["old-generation", "wrong-hash", "unbound", "wrong-source", "wrong-count", "bad-provenance", "empty-with-rows"]
    db_path, digests = seed_house_queue(data_dir, complete + outstanding)
    db = Database(db_path)
    for doc_id in complete + outstanding:
        status = "no_txs" if doc_id.startswith("empty") else "invalid" if doc_id == "invalid" else "success"
        digest = "other" if doc_id == "wrong-hash" else None if doc_id == "unbound" else digests[doc_id]
        generation = "old" if doc_id == "old-generation" else None if doc_id == "unbound" else "current"
        count = 0 if status != "success" else 2 if doc_id == "wrong-count" else 1
        ocr_zero_rows.record_parse_run(
            db.conn, doc_id, 2024, status, count, count,
            parser_version="v1-glm53-vision", engines_attempted="visionb",
            artifact_sha256=digest, ingestion_generation=generation,
        )
        if status == "success" or doc_id == "empty-with-rows":
            db.conn.execute(
                """INSERT INTO transactions (
                    doc_id, source, chamber, source_record_id, source_row_id,
                    transaction_date, disclosure_date, official_filing_date,
                    artifact_sha256, ingestion_generation
                ) VALUES (?, ?, 'House', ?, ?, '2024-01-15', '2024-02-01', '2024-02-01', ?, ?)""",
                [doc_id, "house_pdf" if doc_id == "wrong-source" else "gemini_ocr",
                 doc_id, None if doc_id == "bad-provenance" else f"{doc_id}:row:1", digest, generation],
            )
    db.close()
    items = [dict(doc_id=d, year=2024, artifact_sha256=digests[d], ingestion_generation="current") for d in complete + outstanding]
    candidates, notes = glm_stage.select_candidates_from_manifest(items, 2024, 100, data_dir, check_canonical=True)
    assert candidates == outstanding
    assert notes == []
    assert len(items) - len(candidates) == 13
    # The legacy manual directory selector retains its documented semantics.
    assert "old-generation" in glm_stage.canonical_parsed_doc_ids(db_path)


def test_archive_year_queue_export_deduplicates_scoped_and_default_work(data_dir):
    db_path, digests = seed_house_queue(data_dir, ["8216835"], archive_year=2018, filing_year=2020)
    expected = [("8216835", 2018, str(data_dir / "2018/pdfs/8216835.pdf"))]
    assert ocr_zero_rows.get_ocr_work_items(db_path=str(db_path), data_dir=data_dir) == expected
    assert ocr_zero_rows.get_ocr_work_items(db_path=str(db_path), data_dir=data_dir, year=2018) == expected
    assert ocr_zero_rows.get_ocr_work_items(db_path=str(db_path), data_dir=data_dir, year=2020) == []
    manifest = ocr_zero_rows.build_work_manifest(db_path=str(db_path), data_dir=data_dir)
    assert manifest["items"] == [dict(doc_id="8216835", year=2018, archive_year=2018, artifact_sha256=digests["8216835"], ingestion_generation="current")]


def test_legacy_cross_year_manifest_stage_audit_and_promote_use_one_bound_artifact(data_dir, fake_insert, monkeypatch):
    _, digests = seed_house_queue(data_dir, ["8216835"], archive_year=2018, filing_year=2020)
    pdf = data_dir / "2018/pdfs/8216835.pdf"
    snapshots = []

    def read_fn(path, page_count):
        snapshots.append((Path(path).read_bytes(), page_count))
        return GLM_OUTPUT_ONE_ROW

    # Frozen old manifest: missing filing-year artifact, null hash/generation.
    item = dict(doc_id="8216835", year=2020, artifact_sha256=None, ingestion_generation=None)
    result = glm_stage.stage_batch(2020, 10, data_dir=data_dir, manifest=[item], read_fn=read_fn)
    assert result["staged"] == 1 and result["missing_pdfs"] == result["missing_dirs"] == 0
    assert snapshots == [(pdf.read_bytes(), 1)]
    record = glm_stage.load_staging_record(data_dir, 2020, "8216835")
    assert record["year"] == 2020 and record["archive_year"] == 2018
    assert record["pdf_sha256"] == digests["8216835"] and record["ingestion_generation"] == "current"
    assert glm_stage.load_staging_record(data_dir, 2018, "8216835") is None
    repeat = glm_stage.stage_batch(2020, 10, data_dir=data_dir, manifest=[item], dry_run=True)
    assert repeat["candidates"] == repeat["missing_dirs"] == repeat["missing_pdfs"] == 0
    # A new archive-year export must reuse the frozen filing-year reading.
    regrouped = dict(item, year=2018, archive_year=2018, artifact_sha256=digests["8216835"], ingestion_generation="current")
    assert glm_stage.select_candidates_from_manifest([regrouped], 2018, 10, data_dir) == ([], [])
    assert audit_sample.emit_review_bundle([item], data_dir) == []
    audit_sample.record_verdict(data_dir, "8216835", "pass", "verified")
    write_report_state(data_dir)
    monkeypatch.setattr(ocr_zero_rows, "resolve_ingestion_generation", lambda *args: "current")
    promoted = glm_stage.promote_docs(2020, ["8216835"], data_dir=data_dir)
    assert promoted["promoted"] == 1
    assert fake_insert[0]["year"] == 2018
    assert fake_insert[0]["artifact_sha256"] == digests["8216835"]
    assert fake_insert[0]["ingestion_generation"] == "current"


def test_manifest_refuses_wrong_hash_generation_and_unrecorded_cross_year(data_dir):
    from analyzer.persistence.database import Database

    db_path, digests = seed_house_queue(data_dir, ["bound"], archive_year=2018, filing_year=2020)
    base = dict(doc_id="bound", year=2020, artifact_sha256=digests["bound"], ingestion_generation="current")
    for field, value in [("artifact_sha256", "wrong"), ("ingestion_generation", "old"), ("archive_year", 2017)]:
        candidates, notes = glm_stage.select_candidates_from_manifest([dict(base, **{field: value})], 2020, 10, data_dir)
        assert candidates == [] and len(notes) == 1
        assert "artifact binding refused" in notes[0] and "has no local PDF" not in notes[0]
    db = Database(db_path)
    db.conn.execute("DELETE FROM house_pdf_artifacts")
    db.close()
    # An unrelated same-ID PDF elsewhere must never satisfy the queue.
    make_pdf(data_dir / "2017/pdfs/bound.pdf", pages=2)
    candidates, notes = glm_stage.select_candidates_from_manifest([dict(doc_id="bound", year=2020)], 2020, 10, data_dir)
    assert candidates == [] and "recorded House acquisition" in notes[0]
    result = glm_stage.stage_batch(2020, 10, data_dir=data_dir, manifest=[dict(doc_id="bound", year=2020)], dry_run=True)
    assert result["blocked_artifacts"] == 1 and result["missing_pdfs"] == 0
    (data_dir / "2018/pdfs/bound.pdf").unlink()
    candidates, notes = glm_stage.select_candidates_from_manifest([dict(doc_id="bound", year=2020)], 2020, 10, data_dir)
    assert candidates == [] and "has no local PDF" in notes[0]


def test_manifest_snapshot_hash_is_checked_before_model_read(data_dir):
    pdf = make_pdf(data_dir / "2024/pdfs/changed.pdf")
    calls = []
    record = glm_stage.build_staging_record(
        "changed", 2024, pdf, None, read_fn=lambda *args: calls.append(args),
        artifact_identity=dict(archive_year=2024, artifact_sha256="not-the-snapshot", ingestion_generation="current"),
    )
    assert record["parse_status"] == "error" and "changed after manifest selection" in record["error"]
    assert calls == []


@pytest.mark.parametrize("binding", ["old-generation", "unbound", "wrong-archive"])
def test_manifest_blocks_incompatible_staged_identity_without_mutating_audit(data_dir, binding):
    _, digests = seed_house_queue(data_dir, ["stale"])
    pdf = glm_stage.default_pdf_path(data_dir, 2024, "stale")
    record = glm_stage.build_staging_record("stale", 2024, pdf, None, read_fn=lambda *_: GLM_OUTPUT_ONE_ROW)
    if binding != "unbound":
        record.update(archive_year=2023 if binding == "wrong-archive" else 2024,
                      ingestion_generation="current" if binding == "wrong-archive" else "old")
    record["audit"] = "pass"
    glm_stage.write_staging_record(data_dir, record)
    staged_path = glm_stage.staging_record_path(data_dir, 2024, "stale")
    before = staged_path.read_bytes()
    item = dict(doc_id="stale", year=2024, archive_year=2024,
                artifact_sha256=digests["stale"], ingestion_generation="current")
    candidates, notes = glm_stage.select_candidates_from_manifest([item], 2024, 10, data_dir)
    assert candidates == [] and "evidence reconciliation required" in notes[0]
    assert staged_path.read_bytes() == before
    assert not list((glm_stage.staging_root(data_dir) / "attempts").glob("*"))


def test_manifest_does_not_treat_two_unbound_same_hash_records_as_complete(data_dir):
    seed_year(data_dir, 2024, ["legacy"])
    record = glm_stage.build_staging_record("legacy", 2024,
        glm_stage.default_pdf_path(data_dir, 2024, "legacy"), None,
        read_fn=lambda *_: GLM_OUTPUT_ONE_ROW)
    glm_stage.write_staging_record(data_dir, record)
    candidates, notes = glm_stage.select_candidates_from_manifest(
        [dict(doc_id="legacy", year=2024)], 2024, 10, data_dir)
    assert candidates == [] and "evidence reconciliation required" in notes[0]


def test_audit_renders_verified_snapshot_even_if_original_is_replaced(data_dir, monkeypatch):
    seed_year(data_dir, 2024, ["race"])
    pdf = glm_stage.default_pdf_path(data_dir, 2024, "race")
    record = glm_stage.build_staging_record("race", 2024, pdf, None, read_fn=lambda *_: GLM_OUTPUT_ONE_ROW)
    glm_stage.write_staging_record(data_dir, record)
    render = audit_sample.render_pdf_pages
    seen = []

    def replace_original_then_render(snapshot, out_dir):
        assert Path(snapshot) != pdf and Path(snapshot).name == pdf.name
        assert glm_stage.pdf_sha256(snapshot) == record["pdf_sha256"]
        make_pdf(pdf, pages=2)
        pages = render(snapshot, out_dir)
        seen.extend(pages)
        return pages

    monkeypatch.setattr(audit_sample, "render_pdf_pages", replace_original_then_render)
    assert audit_sample.emit_review_bundle([dict(doc_id="race", year=2024)], data_dir) == []
    assert [p.name for p in seen] == ["page-1.png"]
    assert glm_stage.pdf_page_count(pdf) == 2


def test_audit_reuses_resolution_snapshot_when_original_changes(data_dir, monkeypatch):
    seed_year(data_dir, 2024, ["race"])
    pdf = glm_stage.default_pdf_path(data_dir, 2024, "race")
    record = glm_stage.build_staging_record("race", 2024, pdf, None, read_fn=lambda *_: GLM_OUTPUT_ONE_ROW)
    glm_stage.write_staging_record(data_dir, record)
    resolve = glm_stage.manifest_artifact

    def resolve_then_replace(*args, **kwargs):
        result = resolve(*args, **kwargs)
        make_pdf(pdf, pages=2)
        return result

    monkeypatch.setattr(glm_stage, "manifest_artifact", resolve_then_replace)
    problems = audit_sample.emit_review_bundle([dict(doc_id="race", year=2024)], data_dir)
    assert problems == []
    pages = glm_stage.staging_root(data_dir) / "audit-review/2024/race/pages"
    assert [p.name for p in pages.glob("*.png")] == ["page-1.png"]


@pytest.mark.parametrize("workers", [1, 2])
def test_cli_non_dry_run_reports_blocked_artifacts_for_every_worker_path(data_dir, monkeypatch, capsys, workers):
    seed_year(data_dir, 2024, ["mismatch"])
    path = data_dir / "work-manifest.json"
    path.write_text(json.dumps(dict(schema=glm_stage.WORK_MANIFEST_SCHEMA, items=[
        dict(doc_id="mismatch", year=2024, artifact_sha256="wrong")
    ])))
    monkeypatch.setattr(glm_stage.VisionBConfig, "from_env", lambda: SimpleNamespace(model="synthetic"))
    monkeypatch.setattr(glm_stage, "build_staging_record", lambda *a, **k: pytest.fail("blocked artifact read"))
    result = glm_stage.main(["--data-dir", str(data_dir), "stage", "--year", "2024",
                             "--from-manifest", str(path), "--workers", str(workers)])
    assert result == 2
    output = capsys.readouterr().out
    assert "blocked_artifacts=1" in output and "errors=0" in output


@pytest.mark.parametrize("link_kind", ["file", "directory", "internal-file"])
def test_manifest_symlinks_never_reach_model(data_dir, tmp_path, link_kind, private_read_spy):
    seed_year(data_dir, 2024, ["private"])
    pdf = glm_stage.default_pdf_path(data_dir, 2024, "private")
    private = make_pdf(tmp_path / "outside/private.pdf", pages=2)
    if link_kind == "directory":
        pdf.unlink()
        pdf.parent.rmdir()
        pdf.parent.symlink_to(private.parent, target_is_directory=True)
    else:
        if link_kind == "internal-file":
            private = make_pdf(data_dir / "internal/private.pdf", pages=2)
        pdf.unlink()
        pdf.symlink_to(private)
    calls = []
    private_reads = private_read_spy(private.parent)
    result = glm_stage.stage_batch(2024, 10, data_dir=data_dir,
        manifest=[dict(doc_id="private", year=2024)], read_fn=lambda *a: calls.append(a))
    assert private_reads == []
    assert result["blocked_artifacts"] == 1 and result["staged"] == 0
    assert calls == []


def test_snapshot_read_is_root_anchored_during_directory_retarget(data_dir, tmp_path, monkeypatch):
    seed_year(data_dir, 2024, ["race"])
    pdf = glm_stage.default_pdf_path(data_dir, 2024, "race")
    safe_bytes = pdf.read_bytes()
    private = make_pdf(tmp_path / "outside/race.pdf", pages=2)
    real_open = ocr_zero_rows.os.open
    retargeted = []

    def open_then_retarget(path, flags, *args, **kwargs):
        fd = real_open(path, flags, *args, **kwargs)
        if path == "pdfs" and not retargeted:
            pdf.parent.rename(data_dir / "pinned-pdfs")
            pdf.parent.symlink_to(private.parent, target_is_directory=True)
            retargeted.append(True)
        return fd

    monkeypatch.setattr(ocr_zero_rows.os, "open", open_then_retarget)
    assert ocr_zero_rows.read_work_artifact(data_dir, pdf) == safe_bytes
    assert retargeted
    with pytest.raises(OSError):
        ocr_zero_rows.read_work_artifact(data_dir, pdf)


def test_snapshot_rejects_file_symlink_swapped_at_open(data_dir, tmp_path, monkeypatch):
    seed_year(data_dir, 2024, ["race"])
    pdf = glm_stage.default_pdf_path(data_dir, 2024, "race")
    private = make_pdf(tmp_path / "outside/race.pdf", pages=2)
    real_open = ocr_zero_rows.os.open

    def retarget_then_open(path, flags, *args, **kwargs):
        if path == "race.pdf":
            pdf.unlink()
            pdf.symlink_to(private)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(ocr_zero_rows.os, "open", retarget_then_open)
    calls = []
    record = glm_stage.build_staging_record("race", 2024, pdf, None,
        data_dir=data_dir, read_fn=lambda *a: calls.append(a))
    assert record["parse_status"] == "error" and calls == []


def test_approved_data_root_symlink_is_pinned_before_manifest_read(data_dir, tmp_path, monkeypatch):
    seed_year(data_dir, 2024, ["safe"])
    safe_bytes = glm_stage.default_pdf_path(data_dir, 2024, "safe").read_bytes()
    outside = tmp_path / "outside"
    make_pdf(outside / "2024/pdfs/safe.pdf", pages=2)
    alias = tmp_path / "approved-root"
    alias.symlink_to(data_dir, target_is_directory=True)
    assert ocr_zero_rows.read_work_artifact(alias, alias / "2024/pdfs/safe.pdf") == safe_bytes
    resolve = glm_stage.manifest_artifact

    def retarget_root_then_resolve(*args, **kwargs):
        alias.unlink()
        alias.symlink_to(outside, target_is_directory=True)
        return resolve(*args, **kwargs)

    monkeypatch.setattr(glm_stage, "manifest_artifact", retarget_root_then_resolve)
    seen = []

    def read_fn(snapshot, pages):
        seen.append((Path(snapshot).read_bytes(), pages))
        return GLM_OUTPUT_ONE_ROW

    result = glm_stage.stage_batch(2024, 10, data_dir=alias,
        manifest=[dict(doc_id="safe", year=2024)], read_fn=read_fn)
    assert result["staged"] == 1 and seen == [(safe_bytes, 1)]


@pytest.mark.parametrize("workers", [1, 2])
def test_operation_root_survives_physical_rename_and_external_replacement(
    data_dir, tmp_path, monkeypatch, private_read_spy, workers
):
    seed_year(data_dir, 2024, ["safe"])
    safe_bytes = glm_stage.default_pdf_path(data_dir, 2024, "safe").read_bytes()
    outside = tmp_path / "private"
    make_pdf(outside / "2024/pdfs/safe.pdf", pages=2)
    private_reads = private_read_spy(outside)
    moved = tmp_path / "original-root"
    select = glm_stage.select_candidates_from_manifest
    root_fds = []

    def select_then_replace(*args, **kwargs):
        result = select(*args, **kwargs)
        root_fds.append(args[3].fd)
        data_dir.rename(moved)
        data_dir.symlink_to(outside, target_is_directory=True)
        return result

    monkeypatch.setattr(glm_stage, "select_candidates_from_manifest", select_then_replace)
    seen = []

    def read_fn(snapshot, pages):
        seen.append((Path(snapshot).read_bytes(), pages))
        return GLM_OUTPUT_ONE_ROW

    result = glm_stage.stage_batch(2024, 10, data_dir=data_dir, workers=workers,
        manifest=[dict(doc_id="safe", year=2024)], read_fn=read_fn)
    assert private_reads == []
    assert result["staged"] == 1 and seen == [(safe_bytes, 1)]
    assert glm_stage.load_staging_record(moved, 2024, "safe")["pages"] == 1
    assert not (outside / "staging").exists()
    for fd in root_fds:
        with pytest.raises(OSError):
            os.fstat(fd)


def test_operation_root_fd_closes_when_selection_raises(data_dir, monkeypatch):
    seed_year(data_dir, 2024, ["safe"])
    fds = []

    def fail(*args, **kwargs):
        fds.append(args[3].fd)
        raise RuntimeError("synthetic selection failure")

    monkeypatch.setattr(glm_stage, "select_candidates_from_manifest", fail)
    with pytest.raises(RuntimeError, match="synthetic selection failure"):
        glm_stage.stage_batch(2024, 10, data_dir=data_dir,
            manifest=[dict(doc_id="safe", year=2024)])
    for fd in fds:
        with pytest.raises(OSError):
            os.fstat(fd)


@pytest.mark.parametrize("replace_pdf", [False, True])
def test_audit_bundle_owns_one_record_attempt_during_restage(data_dir, tmp_path, monkeypatch, replace_pdf):
    seed_year(data_dir, 2024, ["attempt"])
    pdf = glm_stage.default_pdf_path(data_dir, 2024, "attempt")
    old = glm_stage.build_staging_record("attempt", 2024, pdf, None, read_fn=lambda *_: GLM_OUTPUT_ONE_ROW)
    glm_stage.write_staging_record(data_dir, old)
    staged_path = glm_stage.staging_record_path(data_dir, 2024, "attempt")
    old_bytes = staged_path.read_bytes()
    replacement = make_pdf(tmp_path / "replacement/attempt.pdf", pages=2)
    new = glm_stage.build_staging_record("attempt", 2024, replacement, None, read_fn=lambda *_: GLM_OUTPUT_PAGE2)
    new_bytes = json.dumps(new).encode()
    write_bytes = Path.write_bytes
    restaged = []

    def copy_then_restage(path, payload):
        result = write_bytes(path, payload)
        if path.name == "record.json" and "audit-review" in path.parts:
            restaged.append(True)
            write_bytes(staged_path, new_bytes)
            if replace_pdf:
                write_bytes(pdf, replacement.read_bytes())
        return result

    monkeypatch.setattr(Path, "write_bytes", copy_then_restage)
    problems = audit_sample.emit_review_bundle([dict(doc_id="attempt", year=2024)], data_dir)
    bundle = glm_stage.staging_root(data_dir) / "audit-review/2024/attempt"
    assert restaged and (bundle / "record.json").read_bytes() == old_bytes
    assert staged_path.read_bytes() == new_bytes
    if replace_pdf:
        assert len(problems) == 1 and "expected SHA-256" in problems[0]
        assert not list(bundle.glob("pages/*.png"))
    else:
        assert problems == []
        assert [p.name for p in bundle.glob("pages/*.png")] == ["page-1.png"]
    with pytest.raises(ValueError, match="staged attempt changed since review bundle"):
        audit_sample.record_verdict(data_dir, "attempt", "pass", "reviewed old pixels")
    assert staged_path.read_bytes() == new_bytes


def test_promotion_never_reopens_verified_pdf_path(data_dir, tmp_path, monkeypatch, fake_insert, private_read_spy):
    stage_one(data_dir, 2024, "promote-race", GLM_OUTPUT_ONE_ROW)
    pdf = glm_stage.default_pdf_path(data_dir, 2024, "promote-race")
    original_hash = glm_stage.pdf_sha256(pdf)
    audit_sample.record_verdict(data_dir, "promote-race", "pass", "verified")
    write_report_state(data_dir)
    private = make_pdf(tmp_path / "private/promote-race.pdf", pages=2)
    private_reads = private_read_spy(private.parent)
    resolve = glm_stage.manifest_artifact

    def resolve_then_replace(*args, **kwargs):
        result = resolve(*args, **kwargs)
        pdf.unlink()
        pdf.symlink_to(private)
        return result

    monkeypatch.setattr(glm_stage, "manifest_artifact", resolve_then_replace)
    monkeypatch.setattr(glm_stage, "pdf_sha256", lambda *_: pytest.fail("mutable PDF path reopened"))
    result = glm_stage.promote_docs(2024, ["promote-race"], data_dir=data_dir)
    assert private_reads == []
    assert result["promoted"] == 1  # synthetic insert only, using the verified old snapshot
    assert len(fake_insert) == 1 and fake_insert[0]["artifact_sha256"] == original_hash


def test_promotion_journals_the_exact_locked_attempt(data_dir, monkeypatch, fake_insert):
    record = stage_one_audited(data_dir, 2024, "attempt-bound", GLM_OUTPUT_ONE_ROW)
    live = glm_stage.staging_record_path(data_dir, 2024, "attempt-bound")
    validated_bytes = live.read_bytes()
    replacement = dict(record)
    replacement["parsed_at_utc"] = "2026-10-24T00:00:00Z"
    lock_requested = threading.Event()
    worker_errors = []
    original_lock = glm_stage.document_publication_lock
    original_insert = ocr_zero_rows.insert_transactions_outcome

    def observed_lock(data_dir, year, doc_id):
        if threading.current_thread().name == "promotion-restager":
            lock_requested.set()
        return original_lock(data_dir, year, doc_id)

    def restage_after_promotion():
        threading.current_thread().name = "promotion-restager"
        try:
            with glm_stage.document_publication_lock(data_dir, 2024, "attempt-bound"):
                glm_stage.archive_current_attempt(data_dir, 2024, "attempt-bound")
                glm_stage.write_staging_record(data_dir, replacement)
        except BaseException as exc:
            worker_errors.append(exc)

    worker = threading.Thread(target=restage_after_promotion)

    def insert_then_race(*args, **kwargs):
        outcome = original_insert(*args, **kwargs)
        worker.start()
        assert lock_requested.wait(timeout=5)
        return outcome

    monkeypatch.setattr(glm_stage, "document_publication_lock", observed_lock)
    monkeypatch.setattr(ocr_zero_rows, "insert_transactions_outcome", insert_then_race)
    result = glm_stage.promote_docs(2024, ["attempt-bound"], data_dir=data_dir)
    worker.join(timeout=10)

    assert not worker.is_alive()
    assert worker_errors == []
    assert result["promoted"] == 1
    (entry,) = glm_stage.read_promoted_journal(glm_stage.staging_root(data_dir))
    archived = glm_stage.staging_root(data_dir) / entry["staging_attempt"]
    assert archived.read_bytes() == validated_bytes
    assert entry["staged_record_sha256"] == glm_stage.audit_staged_record_sha256(
        json.loads(validated_bytes)
    )
    assert glm_stage.load_staging_record(data_dir, 2024, "attempt-bound")[
        "parsed_at_utc"
    ] == replacement["parsed_at_utc"]


def test_staged_row_to_tx_page_number_optional():
    tx = glm_stage.staged_row_to_tx(["X", "Purchase", "01/15/24", None, "A"])
    assert tx["page_number"] is None  # records without row_pages stay valid
    assert (
        glm_stage.staged_row_to_tx(
            ["X", "Purchase", "01/15/24", None, "A"], page_number=3
        )["page_number"]
        == 3
    )


def test_promote_threads_stage_time_page_numbers(data_dir, fake_insert):
    make_pdf(data_dir / "2024" / "pdfs" / "pgn.pdf", pages=2)
    record = glm_stage.build_staging_record(
        "pgn", 2024, glm_stage.default_pdf_path(data_dir, 2024, "pgn"),
        None, read_fn=lambda *a: GLM_OUTPUT_PAGE2,
    )
    glm_stage.write_staging_record(data_dir, record)
    audit_sample.record_verdict(data_dir, "pgn", "pass", "ok")
    write_report_state(data_dir)
    result = glm_stage.promote_docs(2024, ["pgn"], data_dir=data_dir)
    assert result["promoted"] == 1
    (tx,) = fake_insert[0]["transactions"]
    assert tx["page_number"] == 2  # page-bound source_row_id, Gemini-path parity


def test_promote_requires_verdict_bound_to_record_identity(
    data_dir, fake_insert, capsys
):
    # An audit stamp with NO bound pass entry in the audit history is
    # refused: the stamp alone says nothing (fail-closed).
    stage_one(data_dir, 2024, "bnd1", GLM_OUTPUT_ONE_ROW)
    record = glm_stage.load_staging_record(data_dir, 2024, "bnd1")
    record["audit"] = "pass"
    glm_stage.write_staging_record(data_dir, record)
    write_report_state(data_dir)
    result = glm_stage.promote_docs(2024, ["bnd1"], data_dir=data_dir)
    assert result["refused"] == 1 and result["promoted"] == 0
    assert fake_insert == []
    assert "no pass verdict in the audit results history" in capsys.readouterr().out

    # record_verdict stores the record identity; the bound pass promotes.
    audit_sample.record_verdict(data_dir, "bnd1", "pass", "matches pixels")
    write_report_state(data_dir)
    result = glm_stage.promote_docs(2024, ["bnd1"], data_dir=data_dir)
    assert result["promoted"] == 1
    assert len(fake_insert) == 1


def test_promote_refuses_legacy_and_restaged_verdicts(data_dir, fake_insert, capsys):
    stage_one(data_dir, 2024, "bnd2", GLM_OUTPUT_ONE_ROW)
    root = glm_stage.staging_root(data_dir)
    record = glm_stage.load_staging_record(data_dir, 2024, "bnd2")
    record["audit"] = "pass"
    glm_stage.write_staging_record(data_dir, record)
    write_report_state(data_dir)

    # Legacy entry (pre-identity-binding): a pass with no identity fields
    # can never satisfy the binding check.
    (root / "audit-results.json").write_text(
        json.dumps(
            [{"doc_id": "bnd2", "verdict": "pass", "notes": "", "audited_at_utc": "t"}]
        )
    )
    write_report_state(data_dir)
    result = glm_stage.promote_docs(2024, ["bnd2"], data_dir=data_dir)
    assert result["refused"] == 1 and fake_insert == []
    assert "not bound to this record's identity" in capsys.readouterr().out

    # A record re-staged after its audit (identity triple changed while the
    # stamp was carried over) no longer matches the audited identity.
    audit_sample.record_verdict(data_dir, "bnd2", "pass", "matches")
    write_report_state(data_dir)
    record = glm_stage.load_staging_record(data_dir, 2024, "bnd2")
    record["parsed_at_utc"] = "2026-10-19T00:00:00Z"
    glm_stage.write_staging_record(data_dir, record)
    result = glm_stage.promote_docs(2024, ["bnd2"], data_dir=data_dir)
    assert result["refused"] == 1 and fake_insert == []
    out = capsys.readouterr().out
    assert "re-staged after the audit" in out and "re-audit" in out
    assert glm_stage.load_staging_record(data_dir, 2024, "bnd2") is not None


def test_verdict_binding_rejects_row_mutation_but_accepts_content_bound_record(
    data_dir,
):
    stage_one(data_dir, 2024, "rowhash", GLM_OUTPUT_ONE_ROW)
    audit_sample.record_verdict(data_dir, "rowhash", "pass", "pixel match")
    results_path = glm_stage.staging_root(data_dir) / glm_stage.AUDIT_RESULTS_NAME
    entry = json.loads(results_path.read_text())[0]
    record = glm_stage.load_staging_record(data_dir, 2024, "rowhash")
    assert entry["staged_record_sha256"] == glm_stage.audit_staged_record_sha256(record)
    assert glm_stage.verdict_binding(data_dir, "rowhash", record) == (True, None)

    # Exact old behavior: the metadata triple alone still matches after row
    # mutation. A pre-content-hash verdict must not authorize changed rows.
    metadata_only = {
        key: value for key, value in entry.items() if key != "staged_record_sha256"
    }
    results_path.write_text(json.dumps([metadata_only]))
    record["rows"][0][0] = "Mutated after audit"
    record["staged_rows_sha256"] = glm_stage.staged_rows_sha256(record["rows"])
    bound, reason = glm_stage.verdict_binding(data_dir, "rowhash", record)
    assert not bound
    assert "pre-content-hash" in reason

    # Restore the reviewed content and its frozen hash: new binding verifies.
    results_path.write_text(json.dumps([entry]))
    record["rows"] = [["Apple Inc. (AAPL)", "Purchase", "01/15/24", "01/20/24", "A"]]
    record["staged_rows_sha256"] = glm_stage.staged_rows_sha256(record["rows"])
    bound, reason = glm_stage.verdict_binding(data_dir, "rowhash", record)
    assert bound and reason is None


@pytest.mark.parametrize("invalidate", ["before", "during"])
def test_glm_audit_guard_holds_real_verdict_authorization_through_commit(data_dir, monkeypatch, invalidate):
    from analyzer.ocr import common, coordinator, ingestion, workers
    from analyzer.ocr.contracts import ExtractionLane, ExtractionOutcome, ResultBundle, WorkEnvelope, task_key
    from analyzer.persistence.database import Database

    path, digests = seed_house_queue(data_dir, ["guard"])
    record = stage_one_audited(data_dir, 2024, "guard", make_ocr_output())
    recipe = workers.extraction_recipe(ExtractionLane.GLM)
    work = WorkEnvelope(
        task_key("guard", 2024, "current", digests["guard"], recipe, ExtractionLane.GLM),
        str(uuid4()), "guard", 2024, "current", digests["guard"],
        str(glm_stage.default_pdf_path(data_dir, 2024, "guard")), "Jane Doe", "2024-02-01", 1,
        recipe, ExtractionLane.GLM,
    )
    attempted = threading.Event()
    invalidated = threading.Event()
    errors = []

    def revoke():
        attempted.set()
        try:
            audit_sample.record_verdict(data_dir, "guard", "fail", "new pixel defect")
        except Exception as error:  # noqa: BLE001 -- propagate invalidator thread failures to the test
            errors.append(error)
        finally:
            invalidated.set()

    with Database(path) as owner:
        owner.work.register(work)
        claimed = owner.work.claim(work.task_id)
        parsed = common.parse_gemini_output(record["raw_output"], expected_page_count=1)
        result = ResultBundle(claimed, ExtractionOutcome.CANDIDATE, parsed.transactions,
                              record["raw_output"], evidence={"artifact_verified": True,
                              "member": "Jane Doe", "staging_record": record,
                              "page_count": 1, "output_schema_version": 4,
                              "page_outcomes": [{"page_number": 1, "outcome": "transactions"}]})
        bundle_path = workers._publish(result, str(data_dir / "pipeline"))
        real_commit = ingestion.insert_transactions_outcome
        invalidator = threading.Thread(target=revoke)
        committed = []

        def commit(*args, **kwargs):
            assert invalidate == "during", "revoked authorization reached canonical persistence"
            invalidator.start()
            assert attempted.wait(5)
            assert not invalidated.wait(0.1), "verdict changed after admission but before commit"
            outcome = real_commit(*args, **kwargs)
            assert owner.work.is_finished(kwargs["receipt"])
            assert owner.conn.execute("SELECT COUNT(*) FROM transactions WHERE doc_id='guard'").fetchone() == (1,)
            assert not invalidated.is_set(), "audit guard released before canonical receipt committed"
            committed.append(outcome.status)
            return outcome

        monkeypatch.setattr(ingestion, "insert_transactions_outcome", commit)
        if invalidate == "before":
            revoke()
        try:
            state = coordinator._ingest(owner, result, Path(bundle_path), data_dir, None)
        finally:
            if invalidator.ident is not None:
                invalidator.join(5)
        assert not invalidator.is_alive()
        assert not errors
        assert invalidated.is_set()
        assert committed == (["success"] if invalidate == "during" else [])
        assert state == ("accepted" if invalidate == "during" else "needs_review")
        assert owner.conn.execute("SELECT COUNT(*) FROM transactions WHERE doc_id='guard'").fetchone() == ((1,) if invalidate == "during" else (0,))
    assert glm_stage.load_staging_record(data_dir, 2024, "guard")["audit"] == "fail"


# --------------------------------------------------------------------------
# finding 11: workers cap (Z.AI allows 2 streams; overrides are refused)
# --------------------------------------------------------------------------


def test_workers_above_two_stream_budget_refused(data_dir):
    seed_year(data_dir, 2024, ["w1"])
    outputs = {"w1": GLM_OUTPUT_ONE_ROW}
    with pytest.raises(ValueError, match="2 concurrent vision streams"):
        glm_stage.stage_batch(
            2024, 5, data_dir=data_dir,
            read_fn=read_fn_from_outputs(outputs), workers=3,
        )
    with pytest.raises(ValueError):
        glm_stage.stage_batch(
            2024, 5, data_dir=data_dir,
            read_fn=read_fn_from_outputs(outputs), workers=0,
        )
    # workers=2 is exactly the budget and must be accepted.
    result = glm_stage.stage_batch(
        2024, 5, data_dir=data_dir,
        read_fn=read_fn_from_outputs(outputs), workers=2, cooldown=0,
    )
    assert result["staged"] == 1


def test_cli_stage_workers_override_refused(data_dir, capsys):
    seed_year(data_dir, 2024, ["w2"])
    rc = glm_stage.main(
        ["--data-dir", str(data_dir), "stage", "--year", "2024", "--workers", "8"]
    )
    assert rc == 2
    assert "workers=8 refused" in capsys.readouterr().err


# --------------------------------------------------------------------------
# interface audit item 3: asset_review flag (GLM asset-normalization damage)
# --------------------------------------------------------------------------

ASSET_DAMAGE_OUTPUT = (
    "MEMBER: Jane Doe\nPAGES: 1\nPAGE: 1\n"
    + "Apple Inc. (AAPL) | Purchase | 01/15/24 | 01/20/24 | A\n" * 8
    + "Apple Inc. | Purchase | 01/15/24 | 01/20/24 | A\n"
    + "Apple Inc. Apple Inc. | Purchase | 01/15/24 | 01/20/24 | A\n"
)


def test_asset_review_flags_missing_ticker_context():
    # The live failure mode: one row's ticker mangled/dropped ("PARNTERS")
    # while >80% of the doc's other rows carry a parenthesized ticker.
    reasons = glm_stage.detect_asset_review_reasons(
        [
            {"asset": "Apple Inc. (AAPL)"},
            {"asset": "Apple Inc. (AAPL)"},
            {"asset": "Apple Inc. (AAPL)"},
            {"asset": "Apple Inc. PARNTERS"},
        ]
    )
    assert reasons == ["missing_ticker_context:row4"]


def test_asset_review_flags_dropped_repeated_issuer_wording():
    # The live failure mode: "Energy Transfer Equity, L.P. Energy Transfer
    # Equity, L.P." in the PDF, GLM emitted it once on one row while the
    # doubled form survived on a sibling row -> strict substring.
    reasons = glm_stage.detect_asset_review_reasons(
        [
            {"asset": "Energy Transfer Equity, L.P."},
            {"asset": "Energy Transfer Equity, L.P. Energy Transfer Equity, L.P."},
        ]
    )
    assert reasons == ["asset_substring:row1<row2"]


def test_asset_review_ticker_rule_threshold_is_strictly_over_80_percent():
    rows = [{"asset": f"Thing {i} (T{i})"} for i in range(4)] + [
        {"asset": "Untickered Energy Partners LP"},
        {"asset": "Utica Municipal Bond Fund"},
    ]
    # Exactly 4/5 = 80% of each untickered row's others tickered: at the
    # threshold, NOT over it.
    assert glm_stage.detect_asset_review_reasons(rows) == []
    rows.insert(0, {"asset": "Another Fund (AF)"})  # 5/6 > 80% -> flag both
    reasons = glm_stage.detect_asset_review_reasons(rows)
    assert reasons == [
        "missing_ticker_context:row6",
        "missing_ticker_context:row7",
    ]


def test_asset_review_stays_silent_on_clean_and_untickered_docs():
    clean = [{"asset": f"Fund {i} Institutional Shares (F{i})"} for i in range(5)]
    assert glm_stage.detect_asset_review_reasons(clean) == []
    # A doc where NO row carries a ticker (legit untickered assets) is clean.
    untickered = [{"asset": "First Republic Residential Mortgage Trust"}] * 3
    assert glm_stage.detect_asset_review_reasons(untickered) == []
    # Single-row docs can never establish ticker context.
    assert glm_stage.detect_asset_review_reasons([{"asset": "Apple Inc."}]) == []


def test_staged_records_carry_the_asset_review_flag(data_dir):
    seed_year(data_dir, 2024, ["flag1", "clean1"])
    outputs = {
        "flag1": ASSET_DAMAGE_OUTPUT,
        "clean1": GLM_OUTPUT_ONE_ROW,  # one fully tickered row
    }
    result = glm_stage.stage_batch(
        2024, 5, data_dir=data_dir, read_fn=read_fn_from_outputs(outputs)
    )
    assert result["staged"] == 2
    flagged = glm_stage.load_staging_record(data_dir, 2024, "flag1")
    assert flagged["parse_status"] == "success"
    assert flagged["asset_review"] is True
    assert set(flagged["asset_review_reasons"]) >= {
        "missing_ticker_context:row9",
        "asset_substring:row9<row1",
    }
    clean = glm_stage.load_staging_record(data_dir, 2024, "clean1")
    assert clean["asset_review"] is False
    assert clean["asset_review_reasons"] == []


def test_error_records_are_never_asset_review_flagged(data_dir):
    seed_year(data_dir, 2024, ["err1"])

    def broken(pdf_path, page_count):
        raise ValueError("boom")

    record = glm_stage.build_staging_record(
        "err1",
        2024,
        glm_stage.default_pdf_path(data_dir, 2024, "err1"),
        None,
        read_fn=broken,
    )
    assert record["parse_status"] == "error"
    assert record["asset_review"] is False
    assert record["asset_review_reasons"] == []
