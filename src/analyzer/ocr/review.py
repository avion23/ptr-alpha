"""Producer-neutral, operator-recorded source review of immutable API results.

The local filesystem/recording command is the operator trust boundary, not a
provider's evidence dictionary. LLM findings can refuse, never authorize.
All writers and admission hold the existing audit-root lock before doc locks.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from analyzer.ocr.contracts import ExtractionLane, ExtractionOutcome


POLICY = {
    "identity": "house-api-grounded-review-v1",
    "authority": "local-operator-record-v1",
    "critical_fields": [
        "member", "asset_description", "ticker", "transaction_type", "owner_code",
        "transaction_date", "notification_date", "disclosure_date", "amount_raw",
        "amount_midpoint", "instrument_type", "strike_price", "expiry_date",
    ],
    "coverage": "all source pages, all row occurrences, every critical field; omissions included",
}


class ReviewAuthorization(dict):
    """Borrowed authority: valid only while admission_guard holds the root lock."""

    def __init__(self, decision):
        super().__init__(decision)
        self.active = True


def has_review_authority(admission):
    return (isinstance(admission, ReviewAuthorization) and admission.active
            and admission.get("allowed") is True and bool(admission.get("review_binding")))


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(encode(value)).hexdigest()


def review_material(result):
    """One normalization and one typed final persistence projection; no I/O."""
    from analyzer.ocr.normalization import normalize_candidate
    from analyzer.ocr.ingestion import final_ocr_projection

    if result.work.lane == ExtractionLane.LOCAL:
        raise ValueError("source review is for API candidates")
    if result.outcome not in (ExtractionOutcome.CANDIDATE, ExtractionOutcome.VERIFIED_EMPTY):
        raise ValueError("only an unchanged candidate or empty claim can be reviewed")
    candidate = normalize_candidate(result.work, result)
    if candidate["findings"] or (result.rows and candidate["status"] != "candidate"):
        raise ValueError("candidate fails deterministic validation; re-extract before review")
    projection = final_ocr_projection(result.work, candidate["rows"])
    binding = {
        "raw_result_id": digest(result.to_dict()),
        "work": result.work.to_dict(),
        "normalization_recipe": candidate["normalization_recipe"],
        "final_projection_sha256": digest(projection),
        "final_projection_count": len(projection),
        "policy_sha256": digest(POLICY),
    }
    return {"candidate": candidate, "projection": projection, "binding": binding}


def bundle_record(result, material):
    return {
        "schema": "house-api-review-bundle-v1", "doc_id": result.work.doc_id,
        "year": result.work.archive_year, "pages": result.work.page_count,
        "pdf_sha256": result.work.artifact_sha256,
        "binding": material["binding"], "policy": POLICY,
        "raw_result": result.to_dict(), "candidate": material["candidate"],
        "final_projection": material["projection"],
    }


def review_report_template(result, material=None):
    """Unapproved checklist for the operator; blanks/unknowns cannot pass."""
    material = review_material(result) if material is None else material
    return {
        "binding": material["binding"], "policy_sha256": digest(POLICY), "evidence": "",
        "pages": [{"page_number": number, "outcome": "unknown", "evidence": ""}
                  for number in range(1, result.work.page_count + 1)],
        "rows": [{"source_row_id": row["source_row_id"],
                  "page_number": candidate.get("page_number"),
                  "critical_fields": POLICY["critical_fields"], "evidence": ""}
                 for row, candidate in zip(material["projection"], material["candidate"]["rows"])],
        "omissions_checked": False,
    }


def bundle_directory(data_dir, material):
    binding = material["binding"]
    return (Path(data_dir) / "pipeline" / "reviews" / binding["raw_result_id"]
            / digest(binding))


def _validate_bundle(data_dir, directory, record):
    from scripts.audit_sample import _validate_attempt_bundle
    from analyzer.ocr.ingestion import read_work_artifact

    _validate_attempt_bundle(data_dir, directory, record)
    actual = json.loads(read_work_artifact(data_dir, directory / "record.json"))
    if actual != record:
        raise ValueError("frozen candidate, policy or normalization changed")
    pdf = read_work_artifact(data_dir, directory / "source.pdf")
    if hashlib.sha256(pdf).hexdigest() != record["pdf_sha256"]:
        raise ValueError("frozen source PDF changed")


def prepare_review(result, data_dir):
    """Local command: freeze actual PDF, rendered pixels, raw and final rows."""
    from scripts.audit_sample import audit_operation_lock, emit_bound_review_bundle
    from analyzer.ocr.ingestion import approved_data_root, read_work_artifact

    with approved_data_root(data_dir) as root, audit_operation_lock(root):
        material = review_material(result)
        directory = bundle_directory(root, material)
        record = bundle_record(result, material)
        content = read_work_artifact(root, result.work.artifact_path)
        emit_bound_review_bundle(root, directory, encode(record), content)
        _validate_bundle(root, directory, record)
        try:
            with (directory / "report-template.json").open("x") as template:
                json.dump(review_report_template(result, material), template, indent=2, sort_keys=True)
        except FileExistsError:
            pass  # Never overwrite an operator's draft on repeated preparation.
        return str(Path(data_dir) / directory.relative_to(Path(root)))


def _validate_report(report, result, material, verdict):
    if not isinstance(report, dict) or report.get("binding") != material["binding"]:
        raise ValueError("review report must bind the exact frozen candidate")
    if report.get("policy_sha256") != digest(POLICY):
        raise ValueError("review policy identity missing or changed")
    if not isinstance(report.get("evidence"), str) or not report["evidence"].strip():
        raise ValueError("source-grounded evidence report required")
    if verdict != "pass":
        return
    pages = report.get("pages")
    if not isinstance(pages, list) or len(pages) != result.work.page_count:
        raise ValueError("review must cover every source page")
    for number, page in enumerate(pages, 1):
        if (not isinstance(page, dict) or type(page.get("page_number")) is not int
                or page["page_number"] != number
                or page.get("outcome") not in {"transactions", "verified_nontransaction"}
                or not isinstance(page.get("evidence"), str) or not page["evidence"].strip()):
            raise ValueError("each page needs affirmative source-grounded coverage")
    rows = report.get("rows")
    projection = material["projection"]
    if not isinstance(rows, list) or len(rows) != len(projection):
        raise ValueError("review must cover all final row occurrences")
    for row, expected, candidate in zip(rows, projection, material["candidate"]["rows"]):
        if (not isinstance(row, dict) or row.get("source_row_id") != expected["source_row_id"]
                or row.get("critical_fields") != POLICY["critical_fields"]
                or type(row.get("page_number")) is not int
                or not 1 <= row["page_number"] <= result.work.page_count
                or (candidate.get("page_number") is not None
                    and row["page_number"] != candidate["page_number"])
                or pages[row["page_number"] - 1]["outcome"] != "transactions"
                or not isinstance(row.get("evidence"), str) or not row["evidence"].strip()):
            raise ValueError("every row needs page-grounded critical-field verification")
    if {page["page_number"] for page in pages if page["outcome"] == "transactions"} != {
            row["page_number"] for row in rows}:
        raise ValueError("every transaction page must have reviewed rows; omissions require re-extraction")
    if report.get("omissions_checked") is not True:
        raise ValueError("affirmative all-pages omission check required")
    if not projection and any(p["outcome"] != "verified_nontransaction" for p in pages):
        raise ValueError("empty claims require affirmative absence proof on every page")


def record_review(result, data_dir, report, *, verdict, human_reviewed=False):
    """Append a local verdict, never accepting positive silver/model assertions."""
    from scripts.audit_sample import audit_operation_lock
    from analyzer.ocr.ingestion import approved_data_root, read_work_artifact

    if verdict not in {"pass", "fail"}:
        raise ValueError("review verdict must be pass or fail")
    if verdict == "pass" and not human_reviewed:
        raise ValueError("positive authorization requires explicit local human review")
    with approved_data_root(data_dir) as root, audit_operation_lock(root):
        material = review_material(result)
        directory = bundle_directory(root, material)
        record = bundle_record(result, material)
        _validate_bundle(root, directory, record)
        _validate_report(report, result, material, verdict)
        history_path = directory / "verdicts.json"
        history = json.loads(read_work_artifact(root, history_path)) if history_path.exists() else []
        entry = {"verdict": verdict, "binding": material["binding"], "report": report,
                 "authority": POLICY["authority"] if human_reviewed else "silver-failure-only",
                 "bundle_seal_sha256": hashlib.sha256(read_work_artifact(root, directory / "bundle-seal.json")).hexdigest()}
        # A failed unchanged candidate is not repaired by another assertion.
        if verdict == "pass" and any(item["verdict"] == "fail" for item in history):
            raise ValueError("failed candidate requires re-extraction, not a later pass")
        from scripts.glm_stage import atomic_write_json
        atomic_write_json(history_path, [*history, entry])
        return entry


def review_decision_locked(result, data_dir):
    """Caller owns the audit root through final SQL comparison and commit."""
    from analyzer.ocr.ingestion import read_work_artifact

    material = review_material(result)
    directory = bundle_directory(data_dir, material)
    record = bundle_record(result, material)
    try:
        _validate_bundle(data_dir, directory, record)
        history = json.loads(read_work_artifact(data_dir, directory / "verdicts.json"))
        if not history or any(item["verdict"] == "fail" for item in history):
            raise ValueError("missing or failed source review")
        entry = history[-1]
        if entry["authority"] != POLICY["authority"] or entry["binding"] != material["binding"]:
            raise ValueError("review is not locally authorized for this exact candidate")
        seal = hashlib.sha256(read_work_artifact(data_dir, directory / "bundle-seal.json")).hexdigest()
        if entry["bundle_seal_sha256"] != seal or entry["verdict"] != "pass":
            raise ValueError("review seal changed or verdict is not pass")
        _validate_report(entry["report"], result, material, "pass")
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        return {"allowed": False, "reason": f"grounded API review required: {exc}", **material}
    if not result.rows:
        # Authorization derives from the operator's affirmative page report,
        # not from a provider's empty claim. The raw bundle remains unchanged.
        material["candidate"] = {**material["candidate"], "status": "verified_empty"}
    return {"allowed": True, "reason": "locally recorded all-pages source review passed",
            "review_binding": {**material["binding"], "verdict_sha256": digest(entry),
                               "bundle_seal_sha256": seal}, **material}
