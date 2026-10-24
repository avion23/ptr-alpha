"""Fail-closed audit report semantics (FP review 2026-10-18, finding 9).

Reproduces the Astra probes as regression tests:

* 299 null verdicts must NOT yield ``budget_met=True`` -- the report
  refuses (state ``incomplete``, ``budget_met None``).
* fail-then-pass must NOT reduce the failure count (sticky failures;
  repaired tracked separately).
* staged-record sha drift after sampling -> UNRESOLVED.
* omission detection compares row MULTISETS, not counts.
* population-weighted pooling math on a known small case.
* determinism of sampling, freezing, and reporting.
* owner error budget: cosmetic fails (asset-description text) are
  tracked but never block promotion; ``budget_met`` counts critical
  fails only; legacy entries without severity count critical.

Every case runs in temporary directories; ``data/congress.duckdb`` is
never opened and no model calls are made.
"""

from __future__ import annotations

import json
import hashlib
import io
from pathlib import Path

import pytest

from scripts import audit_sample


def tiny_png(color="white"):
    from PIL import Image

    output = io.BytesIO()
    Image.new("RGB", (2, 2), color).save(output, format="PNG")
    return output.getvalue()


def test_attempt_bundles_preserve_old_evidence(data_dir, monkeypatch):
    from scripts import glm_stage
    from analyzer.ocr import common as gemini_ocr_common

    pdf = b"frozen PDF snapshot"
    monkeypatch.setattr(glm_stage, "manifest_artifact", lambda *a: (Path("source.pdf"), {"pdf_bytes": pdf}))
    monkeypatch.setattr(gemini_ocr_common, "pdf_page_count", lambda *a: 1)

    def render(snapshot, out_dir):
        assert snapshot.read_bytes() == pdf
        out_dir.mkdir()
        page = out_dir / "page-1.png"
        page.write_bytes(tiny_png())
        return [page]

    monkeypatch.setattr(audit_sample, "render_pdf_pages", render)
    record = make_record("attempt")
    record["pdf_sha256"] = hashlib.sha256(pdf).hexdigest()
    path = write_record(data_dir, record)
    root = audit_sample.staging_root(data_dir)
    sample = [{"year": 2024, "doc_id": "attempt"}]
    envelope = write_envelope(data_dir)
    lot_path = root / "audit-sample-1.json"
    lot_bytes = lot_path.read_bytes()
    assert audit_sample.emit_review_bundle(sample, data_dir) == []
    legacy = root / "audit-review/2024/attempt"
    legacy_bytes = {p.relative_to(legacy): p.read_bytes() for p in legacy.rglob("*") if p.is_file()}
    assert audit_sample.emit_review_bundle(sample, data_dir)
    assert legacy_bytes == {
        p.relative_to(legacy): p.read_bytes() for p in legacy.rglob("*") if p.is_file()
    }
    audit_sample.record_verdict(data_dir, "attempt", "fail", "old pixels failed")
    audit_sample.record_verdict(data_dir, "attempt", "pass", "old attempt rechecked")
    old_history = audit_sample.load_audit_results(data_dir)

    assert audit_sample.emit_review_bundle(sample, data_dir, attempt_scoped=True) == []
    first = audit_sample._attempt_bundle_dir(data_dir, 2024, "attempt", record)
    first_bytes = (first / "record.json").read_bytes()
    record["parsed_at_utc"] = "2026-10-21T01:00:00Z"
    record["rows"] = [["new", "P", "2024-01-01", "2024-01-02", "A"]]
    write_record(data_dir, record)
    assert glm_stage.verdict_binding(data_dir, "attempt", record)[0] is False
    before = path.read_bytes()
    history_bytes = (root / audit_sample.AUDIT_RESULTS_NAME).read_bytes()
    with pytest.raises(ValueError, match="changed since review"):
        audit_sample.record_verdict(data_dir, "attempt", "pass")
    with pytest.raises(ValueError, match="unreadable attempt bundle"):
        audit_sample.record_verdict(data_dir, "attempt", "pass", attempt_scoped=True)
    assert path.read_bytes() == before
    assert (root / audit_sample.AUDIT_RESULTS_NAME).read_bytes() == history_bytes

    assert audit_sample.emit_review_bundle(sample, data_dir, attempt_scoped=True) == []
    second = audit_sample._attempt_bundle_dir(data_dir, 2024, "attempt", record)
    second_record = second / "record.json"
    page = second / "pages/page-1.png"
    original = second_record.read_bytes()
    second_record.write_bytes(first_bytes)
    with pytest.raises(ValueError, match="changed since review"):
        audit_sample.record_verdict(data_dir, "attempt", "pass", attempt_scoped=True)
    assert audit_sample.emit_review_bundle(sample, data_dir, attempt_scoped=True)
    assert second_record.read_bytes() == first_bytes  # never repaired/replaced
    second_record.write_bytes(original)
    page.unlink()
    assert audit_sample.emit_review_bundle(sample, data_dir, attempt_scoped=True)
    with pytest.raises(ValueError, match="incomplete"):
        audit_sample.record_verdict(data_dir, "attempt", "pass", attempt_scoped=True)
    assert path.read_bytes() == before
    assert (root / audit_sample.AUDIT_RESULTS_NAME).read_bytes() == history_bytes
    page.write_bytes(tiny_png())
    inode = second_record.stat().st_ino
    monkeypatch.setattr(audit_sample, "render_pdf_pages", lambda *a: pytest.fail("must reuse, not rerender"))
    assert audit_sample.emit_review_bundle(sample, data_dir, attempt_scoped=True) == []
    assert second_record.stat().st_ino == inode
    entry = audit_sample.record_verdict(data_dir, "attempt", "pass", attempt_scoped=True)["entry"]
    assert entry["staged_record_sha256"] == audit_sample.staged_record_sha256(record)
    assert entry["parsed_at_utc"] == record["parsed_at_utc"]
    assert entry["review_bundle_record"] == str(second_record.relative_to(data_dir))
    assert glm_stage.verdict_binding(data_dir, "attempt", json.loads(path.read_text())) == (True, None)
    assert audit_sample.emit_review_bundle(sample, data_dir, attempt_scoped=True) == []
    assert second_record.read_bytes() == original  # live audit stamp is excluded, frozen bytes are not
    assert audit_sample.load_audit_results(data_dir) == old_history + [entry]
    assert (first / "record.json").read_bytes() == first_bytes
    assert lot_path.read_bytes() == lot_bytes
    assert {p.relative_to(legacy): p.read_bytes() for p in legacy.rglob("*") if p.is_file()} == legacy_bytes
    report = audit_sample.build_report(
        audit_sample.load_audit_results(data_dir), {"attempt": record["rows"]}, set(),
        lot=envelope["lot"], staged_records_by_doc={"attempt": record},
        frame_counts_by_year=envelope["frame_counts_by_year"],
        frame_counts_by_stratum=envelope["frame_counts_by_stratum"],
    )
    assert report["critical_failure_count"] == 1
    assert report["repaired_count"] == 1
    assert report["budget_met"] is False
    assert report["caveat"] == audit_sample.CAVEAT


@pytest.mark.parametrize("payload", [b"{broken", b"{}", b"null", b"[1]", b"\xff"])
def test_corrupt_history_refuses_before_stamping(data_dir, payload):
    path = write_record(data_dir, make_record("history"))
    before = path.read_bytes()
    history = audit_sample.staging_root(data_dir) / audit_sample.AUDIT_RESULTS_NAME
    history.write_bytes(payload)
    with pytest.raises(ValueError):
        audit_sample.record_verdict(data_dir, "history", "pass")
    assert path.read_bytes() == before
    assert history.read_bytes() == payload


def test_unreadable_history_refuses_before_stamping(data_dir, monkeypatch):
    from analyzer.ocr import ingestion as ocr_zero_rows

    path = write_record(data_dir, make_record("history"))
    before = path.read_bytes()
    history = audit_sample.staging_root(data_dir) / audit_sample.AUDIT_RESULTS_NAME
    history.write_text('[{"legacy": true}]')
    history_bytes = history.read_bytes()
    read = ocr_zero_rows.read_work_artifact

    def unreadable(root, candidate):
        if candidate.name == audit_sample.AUDIT_RESULTS_NAME:
            raise PermissionError("unreadable history")
        return read(root, candidate)

    monkeypatch.setattr(ocr_zero_rows, "read_work_artifact", unreadable)
    with pytest.raises(ValueError, match="unreadable audit history"):
        audit_sample.record_verdict(data_dir, "history", "pass")
    assert path.read_bytes() == before
    assert history.read_bytes() == history_bytes


def test_append_preserves_legacy_dictionaries(data_dir):
    path = write_record(data_dir, make_record("history"))
    history = audit_sample.staging_root(data_dir) / audit_sample.AUDIT_RESULTS_NAME
    legacy = [{"unknown": 1}, {"doc_id": "old", "verdict": "fail"}, {}]
    history.write_text(json.dumps(legacy))
    audit_sample.record_verdict(data_dir, "history", "pass")
    assert json.loads(history.read_text())[:-1] == legacy
    assert json.loads(path.read_text())["audit"] == "pass"


def test_attempt_bundle_refuses_symlink_parent_and_missing_live(data_dir, tmp_path):
    record = make_record("bound")
    write_record(data_dir, record)
    root = audit_sample.staging_root(data_dir)
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "audit-review-attempts").symlink_to(outside, target_is_directory=True)
    assert audit_sample.emit_review_bundle(
        [{"year": 2024, "doc_id": "bound"}], data_dir, attempt_scoped=True,
    )
    assert list(outside.iterdir()) == []
    with pytest.raises(ValueError):
        audit_sample.record_verdict(data_dir, "bound", "pass", attempt_scoped=True)
    with pytest.raises(ValueError, match="current staging record"):
        audit_sample.record_verdict(data_dir, "absent", "pass", attempt_scoped=True)
    assert not (root / audit_sample.AUDIT_RESULTS_NAME).exists()


def test_attempt_bundle_failed_render_does_not_publish(data_dir, monkeypatch):
    from scripts import glm_stage
    from analyzer.ocr import common as gemini_ocr_common

    record = make_record("failed")
    record["pdf_sha256"] = hashlib.sha256(b"PDF").hexdigest()
    write_record(data_dir, record)
    monkeypatch.setattr(glm_stage, "manifest_artifact", lambda *a: (Path("source.pdf"), {"pdf_bytes": b"PDF"}))
    monkeypatch.setattr(gemini_ocr_common, "pdf_page_count", lambda *a: 1)

    def fail(*args):
        raise RuntimeError("render failed")

    monkeypatch.setattr(audit_sample, "render_pdf_pages", fail)
    assert audit_sample.emit_review_bundle(
        [{"year": 2024, "doc_id": "failed"}], data_dir, attempt_scoped=True,
    ) == ["failed: render failed"]
    directory = audit_sample._attempt_bundle_dir(data_dir, 2024, "failed", record)
    assert not directory.exists()
    assert list(directory.parent.iterdir()) == []


@pytest.mark.parametrize("mutation", [
    "different-png", "header-only", "truncated-png", "missing-seal", "corrupt-seal",
    "schema", "pdf_sha256", "staged_record_sha256", "record_file_sha256",
    "prompt_sha256", "parsed_at_utc", "doc_id", "year", "page_count", "page-order",
    "page-size", "page-hash", "page-filename", "extra-page", "missing-page",
    "frozen-record-bytes",
])
def test_attempt_bundle_seal_rejects_changed_payload_before_writes(data_dir, monkeypatch, mutation):
    from scripts import glm_stage
    from analyzer.ocr import common as gemini_ocr_common

    record = make_record("sealed")
    record["pages"] = 2
    record["pdf_sha256"] = hashlib.sha256(b"PDF").hexdigest()
    live = write_record(data_dir, record)
    monkeypatch.setattr(glm_stage, "manifest_artifact", lambda *a: (Path("source.pdf"), {"pdf_bytes": b"PDF"}))
    monkeypatch.setattr(gemini_ocr_common, "pdf_page_count", lambda *a: 2)

    def render(snapshot, out_dir):
        out_dir.mkdir()
        for number, color in enumerate(("white", "black"), 1):
            (out_dir / f"page-{number}.png").write_bytes(tiny_png(color))

    monkeypatch.setattr(audit_sample, "render_pdf_pages", render)
    sample = [{"year": 2024, "doc_id": "sealed"}]
    assert audit_sample.emit_review_bundle(sample, data_dir, attempt_scoped=True) == []
    directory = audit_sample._attempt_bundle_dir(data_dir, 2024, "sealed", record)
    seal_path = directory / "bundle-seal.json"
    seal = json.loads(seal_path.read_text())
    assert seal["record_file_sha256"] == hashlib.sha256((directory / "record.json").read_bytes()).hexdigest()
    assert [p["filename"] for p in seal["pages"]] == ["page-1.png", "page-2.png"]
    monkeypatch.setattr(audit_sample, "render_pdf_pages", lambda *a: pytest.fail("must not rerender existing bundle"))
    assert audit_sample.emit_review_bundle(sample, data_dir, attempt_scoped=True) == []
    page = directory / "pages/page-1.png"
    if mutation == "different-png":
        page.write_bytes(tiny_png("red"))
    elif mutation == "header-only":
        page.write_bytes(b"\x89PNG\r\n\x1a\n")
    elif mutation == "truncated-png":
        page.write_bytes(tiny_png()[:-15])
    elif mutation == "missing-seal":
        seal_path.unlink()
    elif mutation == "corrupt-seal":
        seal_path.write_text("{broken")
    elif mutation == "extra-page":
        (directory / "pages/page-3.png").write_bytes(tiny_png())
    elif mutation == "missing-page":
        page.unlink()
    elif mutation == "frozen-record-bytes":
        with (directory / "record.json").open("ab") as handle:
            handle.write(b"\n")  # same stable content, different sealed file bytes
    else:
        if mutation == "page-order":
            seal["pages"].reverse()
        elif mutation.startswith("page-"):
            field = {"page-size": "size", "page-hash": "sha256", "page-filename": "filename"}[mutation]
            seal["pages"][0][field] = "../outside.png" if field == "filename" else "wrong"
        else:
            seal[mutation] = "wrong"
        seal_path.write_text(json.dumps(seal))
    history = audit_sample.staging_root(data_dir) / audit_sample.AUDIT_RESULTS_NAME
    history.write_text('[{"doc_id": "legacy", "verdict": "fail"}]')
    before_live, before_history = live.read_bytes(), history.read_bytes()
    before_bundle = {p.relative_to(directory): p.read_bytes() for p in directory.rglob("*") if p.is_file()}
    assert audit_sample.emit_review_bundle(sample, data_dir, attempt_scoped=True)
    with pytest.raises(ValueError):
        audit_sample.record_verdict(data_dir, "sealed", "pass", attempt_scoped=True)
    assert live.read_bytes() == before_live
    assert history.read_bytes() == before_history
    assert {p.relative_to(directory): p.read_bytes() for p in directory.rglob("*") if p.is_file()} == before_bundle
    assert glm_stage.verdict_binding(data_dir, "sealed", record)[0] is False


def test_attempt_bundle_initial_truncated_png_never_published(data_dir, monkeypatch):
    from scripts import glm_stage
    from analyzer.ocr import common as gemini_ocr_common

    record = make_record("truncated")
    record["pdf_sha256"] = hashlib.sha256(b"PDF").hexdigest()
    write_record(data_dir, record)
    monkeypatch.setattr(glm_stage, "manifest_artifact", lambda *a: (Path("source.pdf"), {"pdf_bytes": b"PDF"}))
    monkeypatch.setattr(gemini_ocr_common, "pdf_page_count", lambda *a: 1)

    def render(snapshot, out_dir):
        out_dir.mkdir()
        (out_dir / "page-1.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    monkeypatch.setattr(audit_sample, "render_pdf_pages", render)
    assert audit_sample.emit_review_bundle(
        [{"year": 2024, "doc_id": "truncated"}], data_dir, attempt_scoped=True,
    )
    directory = audit_sample._attempt_bundle_dir(data_dir, 2024, "truncated", record)
    assert not directory.exists()
    assert list(directory.parent.iterdir()) == []


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def make_record(doc_id: str, year: int = 2024, rows: list | None = None) -> dict:
    rows = rows if rows is not None else [["X", "Purchase", "01/15/24", None, "A"]]
    return {
        "doc_id": doc_id,
        "year": year,
        "pdf_sha256": f"sha-{doc_id}",
        "model": "glm-5.3-flash",
        "prompt_sha256": "prompt-abc",
        "parser_version": "v1-glm53-vision",
        "parsed_at_utc": "2026-10-18T00:00:00Z",
        "pages": 1,
        "member": "Jane Doe",
        "rows": rows,
        "raw_output": None,
        "parse_status": "success" if rows else "no_txs",
        "error": None,
        "audit": None,
    }


def write_record(data_dir: Path, record: dict) -> Path:
    path = (
        audit_sample.staging_root(data_dir)
        / str(record["year"])
        / f"{record['doc_id']}.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    return path


def write_envelope(
    data_dir: Path,
    seed: int = 1,
    frame_counts: dict | None = None,
    frame_counts_by_stratum: dict | None = None,
) -> dict:
    records = audit_sample.load_staging_records(data_dir)
    sample = [
        {"doc_id": str(record["doc_id"]), "year": int(record["year"])}
        for _, record in records
    ]
    actual_frame, actual_strata = audit_sample._sampling_design_frame(records)
    envelope = {
        "schema": audit_sample.SAMPLE_SCHEMA,
        "seed": seed,
        "per_year": len(sample),
        "created_at_utc": "2026-10-18T00:00:00Z",
        "frame_counts_by_year": frame_counts if frame_counts is not None else actual_frame,
        "frame_counts_by_stratum": (
            frame_counts_by_stratum
            if frame_counts_by_stratum is not None
            else actual_strata
        ),
        "lot": audit_sample.freeze_lot(records, sample),
    }
    out = audit_sample.staging_root(data_dir) / f"audit-sample-{seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(envelope, indent=2, sort_keys=True), encoding="utf-8")
    return envelope


def pass_entry(doc_id: str, at: str = "t1") -> dict:
    return {"doc_id": doc_id, "verdict": "pass", "notes": "", "audited_at_utc": at}


def fail_entry(doc_id: str, at: str = "t1", notes: str = "", severity: str | None = None) -> dict:
    """A fail entry; ``severity=None`` simulates a legacy (pre---severity) entry."""
    entry = {"doc_id": doc_id, "verdict": "fail", "notes": notes, "audited_at_utc": at}
    if severity is not None:
        entry["severity"] = severity
    return entry


@pytest.fixture
def data_dir(tmp_path) -> Path:
    return tmp_path / "data"


def _complete_99_doc_census(data_dir: Path) -> list[str]:
    doc_ids = [f"census-{i:02d}" for i in range(99)]
    for doc_id in doc_ids:
        write_record(data_dir, make_record(doc_id))
    write_envelope(data_dir)
    for doc_id in doc_ids:
        audit_sample.record_verdict(data_dir, doc_id, "pass", "matches pixels")
    assert audit_sample.main(["--data-dir", str(data_dir), "report"]) == 0
    return doc_ids


# --------------------------------------------------------------------------
# Astra probe 1: null/missing verdicts must refuse, never count as clean
# --------------------------------------------------------------------------


def test_299_null_verdicts_refuse_report():
    results = [
        {"doc_id": f"d{i}", "verdict": None, "notes": "", "audited_at_utc": "t"}
        for i in range(299)
    ]
    report = audit_sample.build_report(results, {}, set(), ground_truth_dir=None)
    assert report["state"] == "incomplete"
    assert report["unresolved_count"] == 299
    assert report["budget_met"] is None          # can never default to clean
    assert report["failure_count"] == 0
    assert report["clean"] == []
    assert all(
        reason == "invalid/missing verdict in audit history"
        for _, reason in report["unresolved"]
    )


def test_lot_without_verdicts_refuses_report():
    lot = [{"doc_id": f"d{i}", "year": 2024, "staged_record_sha256": "x"} for i in range(299)]
    report = audit_sample.build_report(
        [],
        {},
        set(),
        lot=lot,
        staged_records_by_doc={f"d{i}": make_record(f"d{i}") for i in range(299)},
    )
    assert report["state"] == "incomplete"
    assert report["unresolved_count"] == 299
    assert report["budget_met"] is None
    assert {reason for _, reason in report["unresolved"]} == {"no verdict recorded"}
    assert report["completion_reasons"][0] == "d0: no verdict recorded"


def test_invalid_verdict_string_refuses():
    report = audit_sample.build_report(
        [{"doc_id": "a", "verdict": "seems-fine", "audited_at_utc": "t"}],
        {},
        set(),
    )
    assert report["state"] == "incomplete"
    assert report["unresolved_count"] == 1
    assert report["budget_met"] is None


# --------------------------------------------------------------------------
# Astra probe 2: fail-then-pass stays a failure (repaired tracked separately)
# --------------------------------------------------------------------------


def test_fail_then_pass_is_sticky_failure_and_tracked_repaired():
    results = [fail_entry("x", "t1"), pass_entry("x", "t2")]
    report = audit_sample.build_report(results, {}, promoted_doc_ids={"x"})
    assert report["failure_count"] == 1
    assert ("x", "verdict:fail") in report["failures"]
    assert report["repaired"] == ["x"]
    assert report["repaired_count"] == 1
    assert report["audited"] == 1  # repaired doc stays in the denominator
    assert report["clean"] == []   # a repaired doc is never "clean"
    assert report["budget_met"] is False  # 1 failure in 1 -> bound 1.0 > 1%


def test_pass_then_fail_is_also_a_failure():
    results = [pass_entry("x", "t1"), fail_entry("x", "t2")]
    report = audit_sample.build_report(results, {}, promoted_doc_ids={"x"})
    assert report["failure_count"] == 1
    assert ("x", "verdict:fail") in report["failures"]
    assert report["repaired_count"] == 0


def test_record_command_appends_history_and_report_uses_worst_verdict(data_dir):
    write_record(data_dir, make_record("9001", rows=[["X", "Purchase", "01/15/24", None, "A"]]))
    audit_sample.record_verdict(data_dir, "9001", "fail", "invented row")
    audit_sample.record_verdict(data_dir, "9001", "pass", "re-checked, matches")
    results = json.loads(
        (audit_sample.staging_root(data_dir) / "audit-results.json").read_text()
    )
    assert [entry["verdict"] for entry in results] == ["fail", "pass"]  # append-only

    records = audit_sample.load_staging_records(data_dir)
    lot = audit_sample.freeze_lot(records, [{"doc_id": "9001", "year": 2024}])
    current = dict(records[0][1])
    report = audit_sample.build_report(
        results, {"9001": current.get("rows", [])}, {"9001"}, lot=lot,
        staged_records_by_doc={"9001": current},
    )
    assert report["failure_count"] == 1
    assert report["repaired_count"] == 1
    assert report["budget_met"] is False


# --------------------------------------------------------------------------
# frozen lot: sha drift -> unresolved
# --------------------------------------------------------------------------


def test_sha_drift_after_sampling_is_unresolved():
    record = make_record("s1", rows=[["X", "Purchase", "01/15/24", None, "A"]])
    records = [(2024, record)]
    lot = audit_sample.freeze_lot(records, [{"doc_id": "s1", "year": 2024}])
    assert lot[0]["staged_record_sha256"] == audit_sample.staged_record_sha256(record)

    drifted = dict(record, model="some-other-model")  # recipe change after sampling
    report = audit_sample.build_report(
        [pass_entry("s1")],
        {"s1": drifted["rows"]},
        {"s1"},
        lot=lot,
        staged_records_by_doc={"s1": drifted},
    )
    assert report["state"] == "incomplete"
    assert report["unresolved"] == [("s1", "recipe/staged-record changed after sampling")]
    assert report["budget_met"] is None


def test_row_change_after_sampling_is_unresolved():
    record = make_record("s2", rows=[["X", "Purchase", "01/15/24", None, "A"]])
    lot = audit_sample.freeze_lot([(2024, record)], [{"doc_id": "s2", "year": 2024}])
    restaged = dict(record, rows=[["X", "Purchase", "01/16/24", None, "A"]])
    report = audit_sample.build_report(
        [pass_entry("s2")], {"s2": restaged["rows"]}, {"s2"},
        lot=lot, staged_records_by_doc={"s2": restaged},
    )
    assert ("s2", "recipe/staged-record changed after sampling") in report["unresolved"]


def test_verdict_stamping_does_not_look_like_drift():
    """Stamping audit fields (record workflow) must keep the doc resolvable."""
    record = make_record("s3", rows=[])
    lot = audit_sample.freeze_lot([(2024, record)], [{"doc_id": "s3", "year": 2024}])
    stamped = dict(record, audit="pass", audit_notes="ok", audited_at_utc="t9")
    report = audit_sample.build_report(
        [pass_entry("s3")], {"s3": []}, {"s3"},
        lot=lot, staged_records_by_doc={"s3": stamped},
    )
    assert report["state"] == "complete"
    assert report["clean"] == ["s3"]
    assert report["unresolved_count"] == 0


def test_sticky_failure_survives_sha_drift():
    """A recorded fail is never demoted to unresolved (count can't shrink)."""
    record = make_record("s4", rows=[])
    lot = audit_sample.freeze_lot([(2024, record)], [{"doc_id": "s4", "year": 2024}])
    drifted = dict(record, rows=[["Y", "Sale", "02/02/24", None, "B"]])
    report = audit_sample.build_report(
        [fail_entry("s4")], {"s4": drifted["rows"]}, set(),
        lot=lot, staged_records_by_doc={"s4": drifted},
    )
    assert report["failure_count"] == 1
    assert ("s4", "verdict:fail") in report["failures"]


def test_promoted_before_omission_check_is_unresolved():
    """Promotion consumes staged rows: no record left -> no clean pass."""
    lot = [{"doc_id": "gone", "year": 2024, "staged_record_sha256": "x"}]
    report = audit_sample.build_report(
        [pass_entry("gone")], {}, {"gone"}, lot=lot, staged_records_by_doc={}
    )
    assert report["state"] == "incomplete"
    assert report["unresolved_count"] == 1
    assert "staging record missing" in report["unresolved"][0][1]
    assert report["budget_met"] is None


def test_lot_entry_freezes_recipe_fields():
    record = make_record("s5")
    lot = audit_sample.freeze_lot([(2024, record)], [{"doc_id": "s5", "year": 2024}])
    assert lot[0]["model"] == "glm-5.3-flash"
    assert lot[0]["prompt_sha256"] == "prompt-abc"
    assert lot[0]["parser_version"] == "v1-glm53-vision"
    assert lot[0]["staged_record_sha256"] == audit_sample.staged_record_sha256(record)


# --------------------------------------------------------------------------
# omission detection: row multisets, not counts
# --------------------------------------------------------------------------


def test_multiset_equal_count_content_diff_is_omission():
    gt_rows = [
        ["A", "Purchase", "01/15/24", None, "A"],
        ["B", "Sale", "02/02/24", None, "B"],
    ]
    staged_rows = [
        ["A", "Purchase", "01/15/24", None, "A"],
        ["B", "Purchase", "02/02/24", None, "B"],  # type flipped; counts equal
    ]
    # Ground truth must come from a file; write it.
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        (Path(td) / "m1.json").write_text(json.dumps({"rows": gt_rows}))
        report = audit_sample.build_report(
            [pass_entry("m1")], {"m1": staged_rows}, {"m1"},
            ground_truth_dir=td,
        )
    assert report["failure_count"] == 1
    reason = report["failures"][0][1]
    assert reason.startswith("omission: staged rows missing ground-truth rows")
    assert report["omission_details"]["m1"]["missing"] == [["b", "sale", "02/02/24", "", "b"]]
    assert report["budget_met"] is False


def test_multiset_order_insensitive_match_is_clean():
    staged = [
        ["A", "Purchase", "01/15/24", None, "A"],
        ["B", "Sale", "02/02/24", None, "B"],
    ]
    gt = list(reversed(staged))  # same multiset, different order
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        (Path(td) / "m2.json").write_text(json.dumps({"rows": gt}))
        report = audit_sample.build_report(
            [pass_entry("m2")], {"m2": staged}, {"m2"}, ground_truth_dir=td
        )
    assert report["state"] == "complete"
    assert report["clean"] == ["m2"]
    assert report["budget_met"] is False  # 0 failures but n=1: bound 95% > 1%


def test_invented_rows_are_material_failures():
    staged = [["A", "Purchase", "01/15/24", None, "A"], ["Z", "Sale", "03/03/24", None, "C"]]
    gt = [["A", "Purchase", "01/15/24", None, "A"]]
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        (Path(td) / "m3.json").write_text(json.dumps({"rows": gt}))
        report = audit_sample.build_report(
            [pass_entry("m3")], {"m3": staged}, {"m3"}, ground_truth_dir=td
        )
    assert report["failure_count"] == 1
    assert report["failures"][0][1].startswith("invention: staged rows absent from ground truth")


def test_count_omission_keeps_legacy_reason_string():
    """gt 2 rows vs staged 1 row keeps the human-readable count message."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        (Path(td) / "m4.json").write_text(
            json.dumps(
                {
                    "rows": [
                        ["A", "Purchase", "01/15/24", None, "A"],
                        ["A2", "Sale", "01/16/24", None, "A"],
                    ]
                }
            )
        )
        report = audit_sample.build_report(
            [pass_entry("m4")],
            {"m4": [["A", "Purchase", "01/15/24", None, "A"]]},
            {"m4"},
            ground_truth_dir=td,
        )
    assert ("m4", "omission: ground-truth rows 2 > staged rows 1") in report["failures"]


def test_pass_without_ground_truth_is_unresolved_not_clean():
    import tempfile

    with tempfile.TemporaryDirectory() as td:  # empty ground-truth dir
        report = audit_sample.build_report(
            [pass_entry("m5")], {"m5": []}, {"m5"}, ground_truth_dir=td
        )
    assert report["state"] == "incomplete"
    assert report["unresolved"] == [("m5", "no ground-truth file for omission check")]
    assert report["budget_met"] is None
    assert report["completion_reasons"] == [
        "m5: no ground-truth file for omission check"
    ]


def test_fail_verdict_with_no_ground_truth_stays_failure():
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        report = audit_sample.build_report(
            [fail_entry("m6")], {"m6": []}, set(), ground_truth_dir=td
        )
    assert report["failure_count"] == 1
    assert ("m6", "verdict:fail") in report["failures"]


# --------------------------------------------------------------------------
# stratification honesty: population-weighted pooling
# --------------------------------------------------------------------------


def _pooling_fixture():
    records = []
    for i in range(10):
        records.append((2023, make_record(f"t{i}", year=2023)))
    for i in range(2):
        records.append((2024, make_record(f"u{i}", year=2024)))
    results = [pass_entry(f"t{i}") for i in range(10)]
    results += [fail_entry("u0"), pass_entry("u1")]
    promoted = {f"t{i}" for i in range(10)} | {"u1"}
    return records, results, promoted


def test_population_weighted_pooling_known_case():
    records, results, promoted = _pooling_fixture()
    lot = audit_sample.freeze_lot(records, [
        {"doc_id": record["doc_id"], "year": year} for year, record in records
    ])
    staged_by_doc = {record["doc_id"]: record["rows"] for _, record in records}
    report = audit_sample.build_report(
        results, staged_by_doc, promoted,
        lot=lot,
        staged_records_by_doc={record["doc_id"]: record for _, record in records},
        frame_counts_by_year={"2023": 90, "2024": 10},
    )
    assert report["state"] == "complete"
    strata = {s["year"]: s for s in report["strata"]}
    assert strata[2023]["docs"] == 10 and strata[2023]["failures"] == 0
    assert strata[2024]["docs"] == 2 and strata[2024]["failures"] == 1
    assert strata[2023]["rate"] == 0.0
    assert strata[2024]["rate"] == pytest.approx(0.5)
    # (90 * 0.0 + 10 * 0.5) / 100 = 0.05 -- NOT the unweighted mean 0.25.
    assert report["pooled_rate"] == pytest.approx(0.05)
    assert report["pooling_method"].startswith("population-weighted")
    assert "staged counts per year" in report["pooling_method"]


def test_stratified_bound_weights_flagged_and_unflagged_cells():
    """Worked 2-cell example: census flagged docs, random sample of plain docs."""
    records = []
    results = []
    sample = []
    for i in range(5):
        record = make_record(f"flag{i}")
        record["asset_review"] = True
        records.append((2024, record))
        sample.append({"doc_id": record["doc_id"], "year": 2024})
        results.append(
            fail_entry(record["doc_id"], severity="critical")
            if i == 0
            else pass_entry(record["doc_id"])
        )
    for i in range(5):
        record = make_record(f"plain{i}")
        records.append((2024, record))
        sample.append({"doc_id": record["doc_id"], "year": 2024})
        results.append(pass_entry(record["doc_id"]))
    lot = audit_sample.freeze_lot(records, sample)
    staged = {record["doc_id"]: record for _, record in records}
    report = audit_sample.build_report(
        results,
        {doc_id: [] for doc_id in staged},
        set(staged),
        lot=lot,
        staged_records_by_doc=staged,
        frame_counts_by_year={"2024": 100},
        frame_counts_by_stratum={
            "2024": {"flagged": 5, "unflagged": 95}
        },
    )
    # K_flagged=1 exactly (5/5 census); no plain docs were observed because
    # all five remaining quota slots were not available in this drawn lot.
    # In this worked sample, 5 plain docs are observed; the other 90 are not,
    # so they conservatively contribute 90 possible critical failures.
    assert report["upper_bound_method"].startswith("simultaneous exact hypergeometric")
    assert report["upper_bound_strata"] == [
        {
            "year": 2024,
            "asset_review": "flagged",
            "population": 5,
            "audited": 5,
            "critical_failures": 1,
            "upper_critical_failures": 1,
            "upper_rate": 0.2,
        },
        {
            "year": 2024,
            "asset_review": "unflagged",
            "population": 95,
            "audited": 5,
            "critical_failures": 0,
            "upper_critical_failures": report["upper_bound_strata"][1]["upper_critical_failures"],
            "upper_rate": report["upper_bound_strata"][1]["upper_rate"],
        },
    ]
    assert report["upper_bound"] == pytest.approx(
        sum(cell["upper_critical_failures"] for cell in report["upper_bound_strata"]) / 100
    )
    assert report["upper_bound"] >= report["pooled_binomial_upper_bound"]
    assert report["critical_failure_count"] == 1


def test_unweighted_fallback_is_labelled():
    records, results, promoted = _pooling_fixture()
    lot = audit_sample.freeze_lot(records, [
        {"doc_id": record["doc_id"], "year": year} for year, record in records
    ])
    report = audit_sample.build_report(
        results,
        {record["doc_id"]: record["rows"] for _, record in records},
        promoted,
        lot=lot,
        staged_records_by_doc={record["doc_id"]: record for _, record in records},
        frame_counts_by_year=None,
    )
    assert report["pooled_rate"] == pytest.approx(0.25)  # mean(0.0, 0.5)
    assert "unweighted-stratum mean" in report["pooling_method"]


@pytest.mark.parametrize(
    ("annual_counts", "reason"),
    [
        ({"2024": float("nan")}, "invalid annual frame weight"),
        ({"2024": 0}, "disagrees with year-by-stratum population"),
    ],
)
def test_invalid_annual_frame_weights_make_report_incomplete(
    data_dir, capsys, annual_counts, reason
):
    record = make_record("weighted")
    write_record(data_dir, record)
    write_envelope(data_dir, frame_counts=annual_counts)
    audit_sample.record_verdict(data_dir, "weighted", "pass")

    rc = audit_sample.main(["--data-dir", str(data_dir), "report"])
    out = capsys.readouterr().out
    state = json.loads(
        (audit_sample.staging_root(data_dir) / audit_sample.REPORT_STATE_NAME).read_text()
    )
    assert rc == 2
    assert reason in out
    assert state["state"] == "incomplete" and state["budget_met"] is None
    assert reason in state["reason"]


def test_zero_weight_is_valid_for_an_empty_year(data_dir):
    record = make_record("nonempty", year=2024)
    write_record(data_dir, record)
    envelope = write_envelope(
        data_dir,
        frame_counts={"2023": 0, "2024": 1},
        frame_counts_by_stratum={"2024": {"flagged": 0, "unflagged": 1}},
    )
    audit_sample.record_verdict(data_dir, "nonempty", "pass")
    report = audit_sample.build_report(
        audit_sample.load_audit_results(data_dir),
        {"nonempty": record["rows"]},
        set(),
        lot=envelope["lot"],
        staged_records_by_doc={"nonempty": record},
        frame_counts_by_year=envelope["frame_counts_by_year"],
        frame_counts_by_stratum=envelope["frame_counts_by_stratum"],
    )
    assert report["state"] == "complete"
    assert report["pooled_rate"] == 0.0
    assert report["pooling_method"].startswith("population-weighted")


# --------------------------------------------------------------------------
# outside-lot verdicts and sample-file contract
# --------------------------------------------------------------------------


def test_verdict_outside_frozen_lot_does_not_block_current_lot():
    record = make_record("a")
    lot = audit_sample.freeze_lot([(2024, record)], [{"doc_id": "a", "year": 2024}])
    report = audit_sample.build_report(
        [pass_entry("a"), fail_entry("z")],
        {}, set(),
        lot=lot,
        staged_records_by_doc={"a": record},
    )
    assert report["state"] == "complete"
    assert report["outside_lot"] == ["z"]
    assert report["completion_reasons"] == []
    assert report["budget_met"] is False


def test_load_sample_lot_refuses_legacy_bare_list(data_dir):
    root = audit_sample.staging_root(data_dir)
    root.mkdir(parents=True)
    (root / "audit-sample-7.json").write_text(json.dumps([{"doc_id": "a", "year": 2024}]))
    envelope, error = audit_sample.load_sample_lot(data_dir)
    assert envelope is None
    assert "frozen-lot format" in error


def test_load_sample_lot_picks_newest(data_dir):
    write_record(data_dir, make_record("a"))
    write_envelope(data_dir, seed=1)
    envelope, error = audit_sample.load_sample_lot(data_dir)
    assert error is None
    assert envelope["schema"] == audit_sample.SAMPLE_SCHEMA
    assert [entry["doc_id"] for entry in envelope["lot"]] == ["a"]


# --------------------------------------------------------------------------
# determinism
# --------------------------------------------------------------------------


def test_report_is_deterministic():
    records, results, promoted = _pooling_fixture()
    lot = audit_sample.freeze_lot(records, [
        {"doc_id": record["doc_id"], "year": year} for year, record in records
    ])
    kwargs = dict(
        lot=lot,
        staged_records_by_doc={record["doc_id"]: record for _, record in records},
        frame_counts_by_year={"2023": 90, "2024": 10},
    )
    one = audit_sample.build_report(results, {r["doc_id"]: r["rows"] for _, r in records}, promoted, **kwargs)
    two = audit_sample.build_report(results, {r["doc_id"]: r["rows"] for _, r in records}, promoted, **kwargs)
    assert one == two


def test_freeze_lot_is_deterministic(data_dir):
    write_record(data_dir, make_record("d1", year=2023))
    write_record(data_dir, make_record("d2", year=2024))
    records = audit_sample.load_staging_records(data_dir)
    sample = audit_sample.build_sample(records, per_year=1, seed=7)
    assert audit_sample.freeze_lot(records, sample) == audit_sample.freeze_lot(records, sample)
    assert audit_sample.build_sample(records, per_year=1, seed=7) == audit_sample.build_sample(
        records, per_year=1, seed=7
    )


def test_sample_draws_asset_review_docs_first(data_dir):
    """Flagged docs get guaranteed representation; plain fill the rest."""
    for i in range(6):
        rec = make_record(f"r{i}", year=2017)
        rec["asset_review"] = i in (2, 5)  # exactly two flagged
        write_record(data_dir, rec)
    records = audit_sample.load_staging_records(data_dir)
    sample = audit_sample.build_sample(records, per_year=3, seed=11)
    chosen = {item["doc_id"] for item in sample}
    assert {"r2", "r5"} <= chosen          # both flagged drawn first
    assert len(chosen & {"r0", "r1", "r3", "r4"}) == 1  # one plain slot
    # Same seed reproduces exactly (stays seed-deterministic).
    again = audit_sample.build_sample(records, per_year=3, seed=11)
    assert [i["doc_id"] for i in again] == [i["doc_id"] for i in sample]


def test_sample_flagged_overflow_fills_quota_from_flagged(data_dir):
    """More flagged than slots: the whole quota comes from flagged docs."""
    for i in range(5):
        rec = make_record(f"f{i}", year=2017)
        rec["asset_review"] = i >= 1  # four flagged of five
        write_record(data_dir, rec)
    records = audit_sample.load_staging_records(data_dir)
    sample = audit_sample.build_sample(records, per_year=2, seed=3)
    chosen = {item["doc_id"] for item in sample}
    assert len(chosen) == 2
    assert chosen <= {"f1", "f2", "f3", "f4"}  # never the unflagged doc


def test_freeze_lot_carries_asset_review_flag(data_dir):
    flagged = make_record("flg", year=2017)
    flagged["asset_review"] = True
    write_record(data_dir, flagged)
    write_record(data_dir, make_record("ok", year=2017))
    records = audit_sample.load_staging_records(data_dir)
    sample = audit_sample.build_sample(records, per_year=2, seed=5)
    lot = audit_sample.freeze_lot(records, sample)
    flags = {entry["doc_id"]: entry["asset_review"] for entry in lot}
    assert flags == {"flg": True, "ok": False}


# --------------------------------------------------------------------------
# CLI: refusal exit codes + audit-report-state.json contract
# --------------------------------------------------------------------------


def test_cli_report_first_line_is_caveat_and_budget_not_met_exits_1(data_dir, capsys):
    record = make_record("p1", rows=[["X", "Purchase", "01/15/24", None, "A"]])
    write_record(data_dir, record)
    # Explicit population frame: the lot is 1 doc of a 100-doc population, so
    # the design-based bound cannot certify <1% (census would trivially pass).
    write_envelope(
        data_dir,
        seed=3,
        frame_counts={"2024": 100},
        frame_counts_by_stratum={"2024": {"flagged": 0, "unflagged": 100}},
    )
    root = audit_sample.staging_root(data_dir)
    (root / "promoted.jsonl").write_text('{"doc_id": "p1", "inserted": 1}\n')
    gt = data_dir / "gt"
    gt.mkdir()
    (gt / "p1.json").write_text(json.dumps({"rows": record["rows"]}))
    # The reviewer records an explicit pass (stamps the staging record;
    # frozen sha is unaffected by audit-mutable fields).
    audit_sample.record_verdict(data_dir, "p1", "pass", "matches pixels")

    rc = audit_sample.main(
        ["--data-dir", str(data_dir), "report", "--ground-truth-dir", str(gt)]
    )
    out = capsys.readouterr().out
    assert rc == 1  # complete, but 0/1 failures -> 95% upper bound 95% > 1%
    assert out.splitlines()[0] == audit_sample.CAVEAT
    assert "Promotion consumes staged rows" in out or "BEFORE `promote`" in out
    assert "NOT MET" in out
    state = json.loads((root / "audit-report-state.json").read_text())
    assert state["state"] == "complete"
    assert state["budget_met"] is False


def test_cli_report_null_verdicts_refuse_exit_2(data_dir, capsys):
    for i in range(3):
        write_record(data_dir, make_record(f"n{i}"))
    write_envelope(data_dir, seed=4)
    root = audit_sample.staging_root(data_dir)
    # Astra's probe: null verdicts in the results file.
    (root / "audit-results.json").write_text(
        json.dumps(
            [
                {"doc_id": f"n{i}", "verdict": None, "notes": "", "audited_at_utc": "t"}
                for i in range(3)
            ]
        )
    )
    rc = audit_sample.main(["--data-dir", str(data_dir), "report"])
    out = capsys.readouterr().out
    assert rc == 2
    assert "INCOMPLETE AUDIT: 3 unresolved" in out
    assert "budget_met cannot be computed" in out
    state = json.loads((root / "audit-report-state.json").read_text())
    assert state["state"] == "incomplete"
    assert state["budget_met"] is None


def test_cli_report_without_sample_lot_refuses(data_dir, capsys):
    rc = audit_sample.main(["--data-dir", str(data_dir), "report"])
    out = capsys.readouterr().out
    assert rc == 2
    assert "no frozen sample lot" in out
    state = json.loads((audit_sample.staging_root(data_dir) / "audit-report-state.json").read_text())
    assert state["state"] == "incomplete"
    assert state["budget_met"] is None


def test_unreadable_live_record_blocks_sampling_and_complete_report(data_dir, capsys):
    from scripts import glm_stage

    record = make_record("population")
    write_record(data_dir, record)
    envelope = write_envelope(data_dir)
    assert audit_sample.load_staging_records(data_dir) == [(2024, record)]
    entry = audit_sample.record_verdict(data_dir, "population", "pass")["entry"]
    assert entry["staged_record_sha256"] == audit_sample.staged_record_sha256(record)
    assert glm_stage.verdict_binding(data_dir, "population", record) == (True, None)
    assert audit_sample.main(["--data-dir", str(data_dir), "report"]) == 0
    capsys.readouterr()

    archived = (
        audit_sample.staging_root(data_dir)
        / "2024"
        / "attempts"
        / "population.attempt-0.json"
    )
    archived.parent.mkdir(parents=True)
    archived_bytes = b'{"prior_attempt":"retained byte-for-byte"}\n'
    archived.write_bytes(archived_bytes)
    bad = audit_sample.staging_root(data_dir) / "2024" / "unreadable.json"
    bad_bytes = b"{broken"
    bad.write_bytes(bad_bytes)

    sample = audit_sample.staging_root(data_dir) / "audit-sample-999.json"
    with pytest.raises(ValueError, match="unreadable staging record .*unreadable.json"):
        audit_sample.main(
            ["--data-dir", str(data_dir), "sample", "--seed", "999", "--no-bundle"]
        )
    assert not sample.exists()

    rc = audit_sample.main(["--data-dir", str(data_dir), "report"])
    out = capsys.readouterr().out
    state = json.loads(
        (
            audit_sample.staging_root(data_dir) / audit_sample.REPORT_STATE_NAME
        ).read_text()
    )
    assert rc == 2
    assert "INCOMPLETE AUDIT: unreadable staging record" in out
    assert state["state"] == "incomplete" and state["budget_met"] is None
    assert "unreadable.json" in state["reason"]
    assert bad.read_bytes() == bad_bytes
    assert archived.read_bytes() == archived_bytes
    assert glm_stage.verdict_binding(data_dir, "population", record) == (True, None)
    assert len(envelope["lot"]) == 1


def test_malformed_promotion_journal_cannot_complete_report(data_dir, capsys):
    record = make_record("ledger")
    write_record(data_dir, record)
    write_envelope(data_dir)
    audit_sample.record_verdict(data_dir, "ledger", "pass")
    root = audit_sample.staging_root(data_dir)
    journal = root / "promoted.jsonl"
    journal.write_text('{"doc_id":"ledger"}\n{broken\n', encoding="utf-8")

    rc = audit_sample.main(["--data-dir", str(data_dir), "report"])
    out = capsys.readouterr().out
    state = json.loads((root / audit_sample.REPORT_STATE_NAME).read_text())
    assert rc == 2
    assert "malformed promotion journal" in out and "promoted.jsonl:2" in out
    assert state["state"] == "incomplete" and state["budget_met"] is None
    assert "promoted.jsonl:2" in state["reason"]


def test_malformed_audit_history_cannot_complete_report(data_dir, capsys):
    record = make_record("history-population")
    write_record(data_dir, record)
    write_envelope(data_dir)
    root = audit_sample.staging_root(data_dir)
    history = root / audit_sample.AUDIT_RESULTS_NAME
    history.write_text(
        json.dumps([pass_entry("history-population"), None]), encoding="utf-8"
    )

    rc = audit_sample.main(["--data-dir", str(data_dir), "report"])
    out = capsys.readouterr().out
    state = json.loads((root / audit_sample.REPORT_STATE_NAME).read_text())
    assert rc == 2
    assert "malformed audit history" in out
    assert state["state"] == "incomplete" and state["budget_met"] is None


def test_cli_report_sha_drift_refuses(data_dir, capsys):
    record = make_record("p9", rows=[["X", "Purchase", "01/15/24", None, "A"]])
    write_record(data_dir, record)
    write_envelope(data_dir, seed=5)
    # Recipe change after sampling: re-stage with different rows.
    write_record(data_dir, dict(record, rows=[["X", "Purchase", "01/16/24", None, "A"]]))
    root = audit_sample.staging_root(data_dir)
    (root / "audit-results.json").write_text(json.dumps([pass_entry("p9")]))
    (root / "promoted.jsonl").write_text('{"doc_id": "p9"}\n')

    rc = audit_sample.main(["--data-dir", str(data_dir), "report"])
    out = capsys.readouterr().out
    assert rc == 2
    assert "recipe/staged-record changed after sampling" in out
    state = json.loads((root / "audit-report-state.json").read_text())
    assert state["state"] == "incomplete"
    assert state["budget_met"] is None
    assert state["unresolved_count"] == 1
    assert state["completion_reasons"] == [
        "p9: recipe/staged-record changed after sampling"
    ]


# --------------------------------------------------------------------------
# finding 6: verdicts bind to the audited record's extraction identity
# --------------------------------------------------------------------------


def test_record_verdict_binds_record_identity(data_dir):
    record = make_record("id1", rows=[["X", "Purchase", "01/15/24", None, "A"]])
    write_record(data_dir, record)
    audit_sample.record_verdict(data_dir, "id1", "pass", "matches pixels")
    (entry,) = json.loads(
        (audit_sample.staging_root(data_dir) / "audit-results.json").read_text()
    )
    # glm_stage.promote requires this triple to still match the live record.
    assert entry["pdf_sha256"] == record["pdf_sha256"]
    assert entry["prompt_sha256"] == record["prompt_sha256"]
    assert entry["parsed_at_utc"] == record["parsed_at_utc"]


def test_verdict_without_record_leaves_identity_unbound(data_dir):
    entry = audit_sample.record_verdict(data_dir, "ghost", "fail", "no record")[
        "entry"
    ]
    assert entry["staged_record_sha256"] is None
    assert entry["pdf_sha256"] is None
    assert entry["prompt_sha256"] is None
    assert entry["parsed_at_utc"] is None


def test_identity_bound_entries_keep_report_state_semantics(data_dir):
    """The finding-9 contract must keep passing with identity-bound entries."""
    record = make_record("id2", rows=[["X", "Purchase", "01/15/24", None, "A"]])
    write_record(data_dir, record)
    records = audit_sample.load_staging_records(data_dir)
    lot = audit_sample.freeze_lot(records, [{"doc_id": "id2", "year": 2024}])
    audit_sample.record_verdict(data_dir, "id2", "pass", "ok")
    records_after = audit_sample.load_staging_records(data_dir)
    report = audit_sample.build_report(
        audit_sample.load_audit_results(data_dir),
        {r["doc_id"]: r.get("rows", []) for _, r in records_after},
        {"id2"},
        lot=lot,
        staged_records_by_doc={r["doc_id"]: r for _, r in records_after},
    )
    assert report["state"] == "complete"
    assert report["clean"] == ["id2"]
    assert report["unresolved_count"] == 0


# --------------------------------------------------------------------------
# owner error budget: transaction-critical vs cosmetic fails
# --------------------------------------------------------------------------


def test_record_severity_defaults_critical_and_validates(data_dir):
    write_record(data_dir, make_record("r1"))
    write_record(data_dir, make_record("r2"))
    audit_sample.record_verdict(data_dir, "r1", "fail", "wrong tx_date")  # default
    audit_sample.record_verdict(
        data_dir, "r2", "fail", "issuer wording dropped", severity="cosmetic"
    )
    entries = json.loads(
        (audit_sample.staging_root(data_dir) / "audit-results.json").read_text()
    )
    assert entries[0]["severity"] == "critical"  # backward-compatible default
    assert entries[1]["severity"] == "cosmetic"
    with pytest.raises(ValueError):
        audit_sample.record_verdict(data_dir, "r1", "fail", severity="minor")

    # The severity stamp is audit-mutable: it must never read as recipe drift.
    stamped = json.loads(
        (audit_sample.staging_root(data_dir) / "2024" / "r2.json").read_text()
    )
    assert stamped["audit_severity"] == "cosmetic"


def test_severity_stamp_does_not_break_frozen_sha():
    record = make_record("r3")
    lot = audit_sample.freeze_lot([(2024, record)], [{"doc_id": "r3", "year": 2024}])
    stamped = dict(
        record, audit="fail", audit_severity="cosmetic", audit_notes="x", audited_at_utc="t"
    )
    report = audit_sample.build_report(
        [fail_entry("r3", severity="cosmetic")],
        {"r3": []},
        set(),
        lot=lot,
        staged_records_by_doc={"r3": stamped},
    )
    assert report["state"] == "complete"  # stamping is not drift
    assert report["cosmetic_failure_count"] == 1


def test_cosmetic_only_fail_meets_budget_with_zero_critical_rate():
    """299 docs, 1 cosmetic fail: critical rate 0 -> bound ~0.997% <= 1%."""
    results = [fail_entry("c1", severity="cosmetic")]
    results += [pass_entry(f"p{i}") for i in range(298)]
    promoted = {f"p{i}" for i in range(298)}
    ids = ["c1"] + [f"p{i}" for i in range(298)]
    records = [(2024, make_record(doc_id)) for doc_id in ids]
    lot = audit_sample.freeze_lot(
        records,
        [{"doc_id": record["doc_id"], "year": year} for year, record in records],
    )
    staged = {record["doc_id"]: record for _, record in records}
    report = audit_sample.build_report(
        results,
        {},
        promoted,
        lot=lot,
        staged_records_by_doc=staged,
        frame_counts_by_year={"2024": 100000},
        frame_counts_by_stratum={"2024": {"flagged": 0, "unflagged": 100000}},
    )
    assert report["state"] == "complete"
    assert report["failure_count"] == 1  # still a recorded (sticky) failure
    assert ("c1", "verdict:fail") in report["failures"]
    assert report["critical_failure_count"] == 0
    assert report["cosmetic_failures"] == [("c1", "verdict:fail")]
    assert report["cosmetic_failure_count"] == 1
    assert report["rate"] == 0.0  # budget math is critical-only
    assert report["upper_bound"] <= 0.01
    assert report["budget_met"] is True  # cosmetic fails do NOT block


def test_mixed_severities_budget_reflects_critical_only():
    """Same n, same total fail count; only the severity flips the budget."""

    def cohort(severity):
        results = [fail_entry("x1", severity=severity)]
        results += [pass_entry(f"p{i}") for i in range(399)]
        promoted = {f"p{i}" for i in range(399)}
        ids = ["x1"] + [f"p{i}" for i in range(399)]
        records = [(2024, make_record(doc_id)) for doc_id in ids]
        lot = audit_sample.freeze_lot(
            records,
            [{"doc_id": record["doc_id"], "year": year} for year, record in records],
        )
        staged = {record["doc_id"]: record for _, record in records}
        return results, promoted, lot, staged

    frame_counts = {"2024": 100000}
    frame_strata = {"2024": {"flagged": 0, "unflagged": 100000}}
    results, promoted, lot, staged = cohort("critical")
    kwargs = dict(
        lot=lot,
        staged_records_by_doc=staged,
        frame_counts_by_year=frame_counts,
        frame_counts_by_stratum=frame_strata,
    )
    critical_report = audit_sample.build_report(results, {}, promoted, **kwargs)
    assert critical_report["failure_count"] == 1
    assert critical_report["critical_failure_count"] == 1
    assert critical_report["cosmetic_failure_count"] == 0
    assert critical_report["rate"] == pytest.approx(1 / 400)
    assert critical_report["budget_met"] is False  # bound(1, 400) ~ 1.3% > 1%

    results, promoted, lot, staged = cohort("cosmetic")
    kwargs["lot"] = lot
    kwargs["staged_records_by_doc"] = staged
    cosmetic_report = audit_sample.build_report(results, {}, promoted, **kwargs)
    assert cosmetic_report["failure_count"] == 1  # total unchanged
    assert cosmetic_report["critical_failure_count"] == 0
    assert cosmetic_report["cosmetic_failure_count"] == 1
    assert cosmetic_report["rate"] == 0.0
    assert cosmetic_report["budget_met"] is True  # cosmetic fails do NOT block


def test_legacy_fail_without_severity_counts_critical_conservatively():
    report = audit_sample.build_report([fail_entry("L1")], {}, set())
    assert report["critical_failure_count"] == 1
    assert report["cosmetic_failure_count"] == 0
    assert report["legacy_unclassified_failures"] == ["L1"]
    assert report["budget_met"] is False


def test_reclassification_via_rerecord_with_severity_cosmetic():
    """Operator re-records a legacy fail as cosmetic: the explicit call governs."""
    results = [fail_entry("L2"), fail_entry("L2", at="t2", severity="cosmetic")]
    report = audit_sample.build_report(results, {}, set())
    assert report["failures"] == [("L2", "verdict:fail")]  # still sticky
    assert report["critical_failure_count"] == 0
    assert report["cosmetic_failure_count"] == 1
    assert report["legacy_unclassified_failures"] == []
    assert report["repaired_count"] == 0  # fail -> fail is not a repair


def test_critical_fail_is_never_downgraded_by_later_cosmetic_rerecord():
    results = [
        fail_entry("L3", severity="critical"),
        fail_entry("L3", at="t2", severity="cosmetic"),
    ]
    report = audit_sample.build_report(results, {}, set())
    assert report["critical_failure_count"] == 1
    assert report["cosmetic_failure_count"] == 0
    assert report["budget_met"] is False


def test_omission_failures_count_critical():
    """Row-multiset mismatches are transaction-critical; no auto-downgrade."""
    import tempfile

    staged = [["A", "Purchase", "01/15/24", None, "A"]]
    with tempfile.TemporaryDirectory() as td:
        (Path(td) / "o1.json").write_text(
            json.dumps(
                {
                    "rows": [
                        ["A", "Purchase", "01/15/24", None, "A"],
                        ["A2", "Sale", "01/16/24", None, "A"],
                    ]
                }
            )
        )
        report = audit_sample.build_report(
            [pass_entry("o1")], {"o1": staged}, {"o1"}, ground_truth_dir=td
        )
    assert report["failures"] == [("o1", "omission: ground-truth rows 2 > staged rows 1")]
    assert report["critical_failure_count"] == 1
    assert report["cosmetic_failure_count"] == 0


def test_cli_report_prints_critical_cosmetic_split(data_dir, capsys):
    for i in range(3):
        write_record(data_dir, make_record(f"m{i}"))
    write_envelope(data_dir, seed=13)
    root = audit_sample.staging_root(data_dir)
    (root / "audit-results.json").write_text(
        json.dumps(
            [
                {
                    "doc_id": "m0",
                    "verdict": "fail",
                    "severity": "critical",
                    "notes": "wrong tx type",
                    "audited_at_utc": "t",
                },
                {
                    "doc_id": "m1",
                    "verdict": "fail",
                    "severity": "cosmetic",
                    "notes": "issuer wording dropped",
                    "audited_at_utc": "t",
                },
                {"doc_id": "m2", "verdict": "pass", "notes": "", "audited_at_utc": "t"},
            ]
        )
    )
    (root / "promoted.jsonl").write_text('{"doc_id": "m2"}\n')

    rc = audit_sample.main(["--data-dir", str(data_dir), "report"])
    out = capsys.readouterr().out
    assert rc == 1  # 1 critical fail in 3 -> bound > 1%
    assert "Critical fails: 1 (rate" in out
    assert "| cosmetic fails: 1" in out
    assert "FAIL m0: verdict:fail" in out
    assert "COSMETIC-FAIL m1" in out
    assert "tracked; not blocking" in out
    assert "NOT MET" in out


def test_cli_report_legacy_fail_prints_reclassify_note_and_counts_critical(data_dir, capsys):
    write_record(data_dir, make_record("L9"))
    write_envelope(data_dir, seed=9)
    root = audit_sample.staging_root(data_dir)
    # Legacy entry: written before --severity existed (no severity field).
    (root / "audit-results.json").write_text(json.dumps([fail_entry("L9")]))

    rc = audit_sample.main(["--data-dir", str(data_dir), "report"])
    out = capsys.readouterr().out
    assert rc == 1  # conservative: 1 critical fail in 1 -> NOT MET
    assert "counted CRITICAL" in out
    assert "--severity cosmetic" in out
    assert "LEGACY-UNCLASSIFIED L9" in out
    state = json.loads((root / "audit-report-state.json").read_text())
    assert state["critical_failure_count"] == 1
    assert state["cosmetic_failure_count"] == 0


def test_state_file_gains_critical_failures_and_keeps_v2_contract(data_dir, capsys):
    write_record(data_dir, make_record("S1"))
    # Explicit population frame (1 audited of 100): 0 critical in 1 must not
    # certify a <1% bound for the population.
    write_envelope(
        data_dir,
        seed=11,
        frame_counts={"2024": 100},
        frame_counts_by_stratum={"2024": {"flagged": 0, "unflagged": 100}},
    )
    audit_sample.record_verdict(
        data_dir, "S1", "fail", "asset description spacing", severity="cosmetic"
    )

    rc = audit_sample.main(["--data-dir", str(data_dir), "report"])
    assert rc == 1  # complete; 0 critical in 1 -> bound 95% > 1%
    state = json.loads(
        (audit_sample.staging_root(data_dir) / "audit-report-state.json").read_text()
    )
    # Existing audit-report-state-v2 keys remain alongside the exact lot hash:
    for key in (
        "schema", "state", "reason", "budget_met", "lot_size", "audited",
        "failure_count", "unresolved_count", "unresolved", "repaired_count",
        "rate", "upper_bound", "confidence", "max_rate", "pooled_rate",
        "pooling_method", "sample_file", "caveat", "generated_at_utc",
    ):
        assert key in state, key
    assert state["schema"] == "audit-report-state-v2"
    assert state["sample_file_sha256"] == hashlib.sha256(
        (audit_sample.staging_root(data_dir) / state["sample_file"]).read_bytes()
    ).hexdigest()
    assert state["state"] == "complete"
    assert state["budget_met"] is False
    assert state["rate"] == 0.0          # budget math is critical-only
    assert state["upper_bound"] > 0.01
    assert state["failure_count"] == 1   # total failures keep their meaning
    # New keys alongside the existing fields:
    assert state["critical_failures"] == []
    assert state["critical_failure_count"] == 0
    assert state["cosmetic_failures"] == [["S1", "verdict:fail"]]
    assert state["cosmetic_failure_count"] == 1
    assert state["completion_reasons"] == []
    assert state["omission_check_status"] == "not_run"


def test_new_critical_verdict_invalidates_population_report(data_dir, capsys):
    from scripts import glm_stage

    doc_ids = _complete_99_doc_census(data_dir)
    report_path = audit_sample.staging_root(data_dir) / audit_sample.REPORT_STATE_NAME
    fresh_state = report_path.read_bytes()
    initial = json.loads(fresh_state)
    assert initial["budget_met"] is True
    assert initial["audit_history_sha256"] == audit_sample.audit_history_sha256(
        audit_sample.load_audit_results(data_dir)
    )

    audit_sample.record_verdict(data_dir, doc_ids[0], "fail", "critical field mismatch")
    invalidated = json.loads(report_path.read_text())
    assert invalidated["state"] == "incomplete" and invalidated["budget_met"] is None
    blocked = glm_stage.promote_docs(2024, [doc_ids[1]], data_dir=data_dir)
    assert blocked["refused"] == 1 and blocked["promoted"] == 0
    assert "verdict history changed" in capsys.readouterr().out

    # Even if a stale complete file survives a concurrent/legacy publication,
    # its explicit binding cannot authorize promotion against changed history.
    report_path.write_bytes(fresh_state)
    blocked = glm_stage.promote_docs(2024, [doc_ids[1]], data_dir=data_dir)
    assert blocked["refused"] == 1 and blocked["promoted"] == 0
    assert "audit history changed since report" in capsys.readouterr().out

    assert audit_sample.main(["--data-dir", str(data_dir), "report"]) == 1
    fresh = json.loads(report_path.read_text())
    assert fresh["state"] == "complete" and fresh["budget_met"] is False
    assert fresh["critical_failure_count"] == 1
    assert fresh["rate"] == pytest.approx(1 / 99)
    assert fresh["upper_bound"] > 0.01
    assert fresh["audit_history_sha256"] == audit_sample.audit_history_sha256(
        audit_sample.load_audit_results(data_dir)
    )


def test_same_seed_sample_is_immutable_and_stale_report_must_be_rerun(
    data_dir, capsys
):
    from scripts import glm_stage

    doc_ids = _complete_99_doc_census(data_dir)
    root = audit_sample.staging_root(data_dir)
    sample_path = root / "audit-sample-1.json"
    original_lot = sample_path.read_bytes()

    # An identical retry reuses the original envelope bytes.
    assert audit_sample.main(
        [
            "--data-dir", str(data_dir), "sample", "--seed", "1",
            "--per-year", "99", "--no-bundle",
        ]
    ) == 0
    assert sample_path.read_bytes() == original_lot

    write_record(data_dir, make_record("new-unresolved"))
    history_path = root / audit_sample.AUDIT_RESULTS_NAME
    history_before = history_path.read_bytes()
    assert audit_sample.main(
        [
            "--data-dir", str(data_dir), "sample", "--seed", "1",
            "--per-year", "100", "--no-bundle",
        ]
    ) == 2
    assert "immutable" in capsys.readouterr().err
    assert sample_path.read_bytes() == original_lot
    assert history_path.read_bytes() == history_before

    # A conflicting resample invalidates the old completion; only a fresh
    # report over the still-frozen original lot can reopen promotion.
    blocked = glm_stage.promote_docs(2024, [doc_ids[0]], data_dir=data_dir)
    assert blocked["refused"] == 1 and blocked["promoted"] == 0
    assert audit_sample.main(["--data-dir", str(data_dir), "report"]) == 0
    state = glm_stage.audit_report_state(data_dir)
    assert state["sample_file_sha256"] == hashlib.sha256(original_lot).hexdigest()
    assert glm_stage.audit_report_gate_open(
        state,
        audit_sample.audit_history_sha256(audit_sample.load_audit_results(data_dir)),
        audit_sample.sample_lot_file_sha256(data_dir, state["sample_file"]),
    )


def test_report_invalidation_and_history_write_failures_fail_closed(
    data_dir, monkeypatch, capsys
):
    from scripts import glm_stage

    doc_ids = _complete_99_doc_census(data_dir)
    root = audit_sample.staging_root(data_dir)
    report_path = root / audit_sample.REPORT_STATE_NAME
    history_path = root / audit_sample.AUDIT_RESULTS_NAME
    record_path = root / "2024" / f"{doc_ids[0]}.json"
    original_state = report_path.read_bytes()
    original_history = history_path.read_bytes()
    original_record = record_path.read_bytes()
    real_atomic_write = audit_sample._atomic_write_json

    def fail_invalidation(path, payload):
        if Path(path).name == report_path.name:
            raise OSError("injected report-state write failure")
        return real_atomic_write(path, payload)

    monkeypatch.setattr(audit_sample, "_atomic_write_json", fail_invalidation)
    with pytest.raises(OSError, match="report-state write failure"):
        audit_sample.record_verdict(data_dir, doc_ids[0], "fail", "critical mismatch")
    assert report_path.read_bytes() == original_state
    assert history_path.read_bytes() == original_history
    assert record_path.read_bytes() == original_record

    def fail_history_append(path, payload):
        if Path(path).name == history_path.name:
            state = json.loads(report_path.read_text())
            assert state["state"] == "incomplete" and state["budget_met"] is None
            raise OSError("injected history append failure")
        return real_atomic_write(path, payload)

    monkeypatch.setattr(audit_sample, "_atomic_write_json", fail_history_append)
    with pytest.raises(OSError, match="history append failure"):
        audit_sample.record_verdict(data_dir, doc_ids[0], "fail", "critical mismatch")
    assert history_path.read_bytes() == original_history
    assert json.loads(record_path.read_text())["audit"] == "fail"
    invalidated = json.loads(report_path.read_text())
    assert invalidated["state"] == "incomplete" and invalidated["budget_met"] is None
    blocked = glm_stage.promote_docs(2024, [doc_ids[1]], data_dir=data_dir)
    assert blocked["refused"] == 1 and blocked["promoted"] == 0
    assert "verdict history changed" in capsys.readouterr().out


def test_live_30_doc_lot_completes_despite_historical_outside_lot_verdicts(
    data_dir, capsys
):
    """Seed 20261019: resolved 2017 lot, one cosmetic fail, older extras."""
    doc_ids = [
        "20005631", "20006505", "20006545", "20006739", "20006747",
        "20006751", "20006839", "20006944", "20006996", "20007206",
        "20007208", "20007359", "20007420", "20007551", "20007588",
        "20007592", "20007596", "20007702", "20007716", "20007829",
        "20007899", "20007964", "20007971", "20008047", "20008135",
        "20008194", "20008277", "20008431", "20008488", "20008556",
    ]
    for doc_id in doc_ids:
        write_record(data_dir, make_record(doc_id, year=2017))
    write_envelope(
        data_dir,
        seed=20261019,
        frame_counts={"2017": 584},
        frame_counts_by_stratum={
            "2017": {"flagged": 0, "unflagged": 584}
        },
    )
    root = audit_sample.staging_root(data_dir)
    cosmetic_doc = "20006747"
    results = [pass_entry(doc_id) for doc_id in doc_ids if doc_id != cosmetic_doc]
    # Re-recording a cosmetic fail stays one resolved cosmetic failure.
    results.extend(
        [
            fail_entry(cosmetic_doc, severity="cosmetic"),
            fail_entry(cosmetic_doc, at="t2", severity="cosmetic"),
        ]
    )
    cosmetic_outside = ["20006438", "20006516"]
    critical_outside = "20006575"
    pass_outside = ["20006470", "20006657", "20006682", "20006694", "20006832"]
    results.extend(fail_entry(doc_id, severity="cosmetic") for doc_id in cosmetic_outside)
    results.extend(
        [
            fail_entry(critical_outside, severity="critical"),
            pass_entry(critical_outside, at="later-pass"),
        ]
    )  # later pass cannot launder a sticky outside-lot critical failure
    results.extend(pass_entry(doc_id) for doc_id in pass_outside)
    (root / "audit-results.json").write_text(json.dumps(results))

    rc = audit_sample.main(
        ["--data-dir", str(data_dir), "report", "--max-rate", "0.01"]
    )
    out = capsys.readouterr().out
    state = json.loads((root / "audit-report-state.json").read_text())

    assert rc == 1  # complete measurement; the 30-doc bound misses 1%
    assert state["state"] == "complete"
    assert state["budget_met"] is False
    assert state["lot_size"] == state["audited"] == 30
    assert state["critical_failure_count"] == 0
    assert state["cosmetic_failure_count"] == 1
    assert state["rate"] == 0.0
    assert state["upper_bound"] < state["pooled_binomial_upper_bound"]
    assert state["upper_bound_method"].startswith("simultaneous exact hypergeometric")
    assert state["outside_lot"] == sorted(
        cosmetic_outside + [critical_outside] + pass_outside
    )
    assert state["outside_lot_critical_failure_count"] == 1
    assert state["outside_lot_cosmetic_failure_count"] == 2
    assert state["outside_lot_passes"] == sorted(pass_outside)
    assert state["population_critical_failure_count"] == 1
    assert state["population_cosmetic_failure_count"] == 3
    assert state["completion_reasons"] == []
    assert "Omission check not run" in out and "pixel-only certification" in out
    assert "One-sided exact 95% design-based upper bound (critical):" in out
    assert "pooled unweighted-binomial reference (not acceptance bound): 9.503%" in out
    assert "Owner budget (critical-only design-based upper bound <= 1%" in out
    assert "NOT MET" in out
    assert "8 OUTSIDE-LOT doc(s)" in out
    assert "outside-lot contribution of 1 critical and 2 cosmetic" in out
    assert "OUTSIDE-LOT 20006575: fail (critical; remains in population tally)" in out
    assert "OUTSIDE-LOT 20006470: pass-only" in out


def test_incomplete_report_state_names_completion_reason(data_dir, capsys):
    write_record(data_dir, make_record("needs-verdict"))
    write_envelope(data_dir, seed=19)
    rc = audit_sample.main(["--data-dir", str(data_dir), "report"])
    out = capsys.readouterr().out
    state = json.loads(
        (audit_sample.staging_root(data_dir) / "audit-report-state.json").read_text()
    )
    assert rc == 2
    assert "INCOMPLETE-REASON needs-verdict: no verdict recorded" in out
    assert "HOW TO FIX: record an explicit verdict" in out
    assert state["state"] == "incomplete"
    assert state["budget_met"] is None
    assert state["reason"] == "needs-verdict: no verdict recorded"
    assert state["completion_reasons"] == ["needs-verdict: no verdict recorded"]
    assert state["completion_guidance"] == [
        "record an explicit verdict with audit_sample.py record --doc-id "
        "<id> --verdict pass|fail"
    ]


def test_incomplete_state_payload_carries_severity_keys(data_dir, capsys):
    rc = audit_sample.main(["--data-dir", str(data_dir), "report"])  # no lot
    assert rc == 2
    state = json.loads(
        (audit_sample.staging_root(data_dir) / "audit-report-state.json").read_text()
    )
    assert state["schema"] == "audit-report-state-v2"
    assert state["critical_failures"] == []
    assert state["critical_failure_count"] == 0
    assert state["cosmetic_failures"] == []
    assert state["cosmetic_failure_count"] == 0
    assert state["budget_met"] is None
    assert "no frozen sample lot" in state["reason"]
    assert state["completion_reasons"] == [state["reason"]]
    assert state["sample_file_sha256"] is None


@pytest.mark.parametrize(
    ("lot", "reason"),
    [
        ([], "frozen sample lot contains no valid documents"),
        ([{}], "malformed frozen lot entry at index 0 (missing doc_id)"),
        ([{"doc_id": "bad-year", "year": "unknown"}], "invalid year"),
    ],
)
def test_empty_or_malformed_frozen_lot_is_named_incomplete(lot, reason):
    report = audit_sample.build_report(
        [], {}, set(), lot=lot, staged_records_by_doc={}
    )
    assert report["state"] == "incomplete"
    assert report["budget_met"] is None
    assert any(reason in item for item in report["completion_reasons"])


def test_cli_record_accepts_severity_flag(data_dir, capsys):
    write_record(data_dir, make_record("c9"))
    rc = audit_sample.main(
        [
            "--data-dir", str(data_dir), "record",
            "--doc-id", "c9", "--verdict", "fail",
            "--severity", "cosmetic", "--notes", "text-only",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "severity: cosmetic" in out
    assert "does NOT block promotion" in out
    (entry,) = json.loads(
        (audit_sample.staging_root(data_dir) / "audit-results.json").read_text()
    )
    assert entry["severity"] == "cosmetic"
