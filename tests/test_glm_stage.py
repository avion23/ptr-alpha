"""Staged GLM OCR ingestion + pixel audit (glm_stage.py / audit_sample.py).

Every case runs in temporary directories: ``data/congress.duckdb`` is never
opened, and the canonical insert path (``ocr_zero_rows.insert_transactions``)
is monkeypatched for promotion tests. No model calls are made.
"""

from __future__ import annotations

import base64
import json
import shutil
from pathlib import Path

import pytest

from analyzer.vision_engines import VisionBUnavailable
from scripts import audit_sample, glm_stage
from scripts import ocr_zero_rows
from scripts.gemini_ocr_common import PROMPT_SHA256

GLM_OUTPUT_ONE_ROW = (
    "MEMBER: Jane Doe\nPAGES: 1\nPAGE: 1\n"
    "Apple Inc. (AAPL) | Purchase | 01/15/24 | 01/20/24 | A\n"
)
GLM_OUTPUT_NO_TXS = "MEMBER: Jane Doe\nPAGES: 1\nPAGE: 1\nNO_TRANSACTIONS\n"


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


def read_fn_from_outputs(outputs: dict[str, str]):
    def read_fn(pdf_path, page_count):
        return outputs[Path(pdf_path).stem]

    return read_fn


@pytest.fixture
def data_dir(tmp_path) -> Path:
    return tmp_path / "data"


@pytest.fixture
def fake_insert(monkeypatch):
    """Monkeypatch the canonical insert path + generation resolution."""
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
        return len(transactions)

    monkeypatch.setattr(ocr_zero_rows, "insert_transactions", _insert)
    monkeypatch.setattr(
        ocr_zero_rows,
        "resolve_ingestion_generation",
        lambda db_path, doc_id, year, sha: "gen-1",
    )
    return calls


# --------------------------------------------------------------------------
# stage: record format + round trip
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


def test_promote_consumes_staging_record(data_dir, fake_insert):
    stage_one(data_dir, 2024, "2001", GLM_OUTPUT_ONE_ROW)
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
    # main()-style progress bookkeeping happened.
    progress = json.loads(
        (data_dir / "ocr_progress_gemini_manual.json").read_text()
    )
    assert progress["completed"] == ["2001"]


def test_promote_no_txs_records_and_consumes(data_dir, fake_insert):
    stage_one(data_dir, 2024, "2002", GLM_OUTPUT_NO_TXS)
    result = glm_stage.promote_docs(2024, ["2002"], data_dir=data_dir)
    assert result["promoted"] == 1
    assert fake_insert[0]["transactions"] == []
    assert fake_insert[0]["raw_count"] == 0
    progress = json.loads(
        (data_dir / "ocr_progress_gemini_manual.json").read_text()
    )
    assert progress["no_txs"] == ["2002"]
    assert glm_stage.load_staging_record(data_dir, 2024, "2002") is None


def test_promote_require_audit_pass_gate(data_dir, fake_insert):
    stage_one(data_dir, 2024, "2003", GLM_OUTPUT_ONE_ROW)
    refused = glm_stage.promote_docs(
        2024, ["2003"], data_dir=data_dir, require_audit_pass=True
    )
    assert refused["refused"] == 1 and refused["promoted"] == 0
    assert fake_insert == []
    assert glm_stage.load_staging_record(data_dir, 2024, "2003") is not None

    # A passing pixel audit stamps the staging record and opens the gate.
    audit_sample.record_verdict(data_dir, "2003", "pass", "matches pixels")
    record = glm_stage.load_staging_record(data_dir, 2024, "2003")
    assert record["audit"] == "pass"
    promoted = glm_stage.promote_docs(
        2024, ["2003"], data_dir=data_dir, require_audit_pass=True
    )
    assert promoted["promoted"] == 1
    assert len(fake_insert) == 1
    assert glm_stage.load_staging_record(data_dir, 2024, "2003") is None


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
            "rows": [["X", "Purchase", "01/15/24", None, "A"]],
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
    stage_one(data_dir, 2024, "4001", GLM_OUTPUT_ONE_ROW)

    def zero_insert(*args, **kwargs):
        return 0

    monkeypatch.setattr(ocr_zero_rows, "insert_transactions", zero_insert)
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
    assert report["upper_bound"] == pytest.approx(
        audit_sample._exact_upper_bound(2, 3, 0.95)
    )


def test_report_budget_met_for_clean_299_sample():
    results = [
        {"doc_id": f"d{i}", "verdict": "pass", "notes": "", "audited_at_utc": "t"}
        for i in range(299)
    ]
    report = audit_sample.build_report(
        results,
        {},
        promoted_doc_ids={f"d{i}" for i in range(299)},
        ground_truth_dir=None,
        max_rate=0.01,
    )
    assert report["failure_count"] == 0
    assert len(report["clean"]) == 299
    assert report["budget_met"] is True
    assert report["upper_bound"] <= 0.01


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
