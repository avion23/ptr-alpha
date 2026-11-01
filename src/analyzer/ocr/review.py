"""Producer-neutral, operator-recorded source review of immutable API results.

The local filesystem/recording command is the operator trust boundary, not a
provider's evidence dictionary. Independent AI source review is explicitly labelled.
All writers and admission hold the existing audit-root lock before doc locks.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from analyzer.ocr.contracts import ExtractionLane, ExtractionOutcome


POLICY = {
    "identity": "house-api-grounded-review-v2",
    "authority": "local-operator-record-v1",
    "critical_fields": [
        "member", "asset_description", "ticker", "transaction_type", "owner_code",
        "transaction_date", "notification_date", "disclosure_date", "amount_raw",
        "amount_midpoint", "instrument_type", "strike_price", "expiry_date",
    ],
    "coverage": "all source pages, all row occurrences, every critical field; omissions included",
    "ai_authority": "independent-ai-source-review-v1",
    "ai_requirements": "two independent sessions; pixel citations; all field observations; original UTF-8 ledgers with SHA-256; local source-checking and faithful comparison attestations; blind ledger binding is post-review",
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
    if not result.rows:
        from analyzer.ocr.common import parse_gemini_output

        if (not isinstance(result.raw_response, str) or not result.raw_response.strip()
                or result.evidence.get("failure") or result.evidence.get("provider_error")):
            raise ValueError("empty review requires a successfully parsed source claim, not extraction failure")
        parsed = parse_gemini_output(result.raw_response, expected_page_count=result.work.page_count)
        if (parsed.transactions or parsed.raw_row_count or not parsed.no_transactions
                or parsed.covered_pages != frozenset(range(1, result.work.page_count + 1))):
            raise ValueError("empty review requires an explicit parsed claim covering every source page")
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
        "policy_identity": POLICY["identity"],
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


def required_review_supersession(repository, raw_result_id):
    """An appended refusal/retraction is not an ordinary pending raw review."""
    row = repository.conn.execute(
        "SELECT task_id, attempt_id, path, finished_at FROM extraction_results WHERE result_id = ?",
        [raw_result_id],
    ).fetchone()
    if row is None or row[3] is None:
        return None
    receipt = dict(result_id=raw_result_id, task_id=row[0], attempt_id=row[1], path=row[2])
    effective = repository.effective_admission_decision(receipt)
    event = effective["detail"]
    if (effective["state"] == "blocked"
            or (effective["state"] == "needs_review" and isinstance(event, dict)
                and event.get("kind") == "admission_decision")):
        return effective["result_id"]
    return None


def assert_review_supersession(repository, raw_result_id, binding):
    required = required_review_supersession(repository, raw_result_id)
    if binding.get("supersedes_admission_decision_id") != required:
        raise ValueError(f"fresh local source review must acknowledge supersedes_admission_decision_id={required}")


def review_report_template(result, material=None, *, supersedes_admission_decision_id=None):
    """Unapproved checklist for the operator; blanks/unknowns cannot pass."""
    material = review_material(result) if material is None else material
    return {
        "binding": material["binding"], "policy_sha256": digest(POLICY), "evidence": "",
        "supersedes_admission_decision_id": supersedes_admission_decision_id,
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


def _has_failed_projection(data_dir, material, result):
    """Failures follow identical interpreted rows, not a normalization directory.

    Validate every sibling before using its history. Seals detect corruption at
    the existing private-filesystem boundary, not hostile replacement of both.
    """
    from analyzer.ocr.ingestion import read_work_artifact

    binding = material["binding"]
    failed = False
    for directory in bundle_directory(data_dir, material).parent.iterdir():
        record = json.loads(read_work_artifact(data_dir, directory / "record.json"))
        old = record["binding"]
        if (record["schema"] != "house-api-review-bundle-v1"
                or directory.name != digest(old) or old["raw_result_id"] != binding["raw_result_id"]
                or record["raw_result"] != result.to_dict()
                or old["work"] != result.work.to_dict()
                or record["pdf_sha256"] != result.work.artifact_sha256
                or record["candidate"]["normalization_recipe"] != old["normalization_recipe"]
                or old["policy_sha256"] != digest(record["policy"])
                or old["policy_identity"] != record["policy"]["identity"]
                or old["final_projection_sha256"] != digest(record["final_projection"])
                or old["final_projection_count"] != len(record["final_projection"])):
            raise ValueError("invalid sibling source-review binding")
        _validate_bundle(data_dir, directory, record)
        history_path = directory / "verdicts.json"
        # Prepared but unreviewed bundles have no history yet.
        if not history_path.exists() and not history_path.is_symlink():
            continue
        history = json.loads(read_work_artifact(data_dir, history_path))
        if not isinstance(history, list):
            raise TypeError("invalid source-review history")
        seal = hashlib.sha256(read_work_artifact(data_dir, directory / "bundle-seal.json")).hexdigest()
        for entry in history:
            report = entry["report"]
            if (entry["binding"] != old or entry["bundle_seal_sha256"] != seal
                    or entry["verdict"] not in {"pass", "fail"}
                    or entry["authority"] not in {POLICY["authority"], POLICY["ai_authority"], "silver-failure-only"}
                    or (entry["verdict"] == "pass" and entry["authority"] == "silver-failure-only")
                    or report["binding"] != old or report["policy_sha256"] != old["policy_sha256"]
                    or not isinstance(report["evidence"], str) or not report["evidence"].strip()):
                raise ValueError("invalid sealed source-review history")
            if (entry["verdict"] == "fail"
                    and old["final_projection_sha256"] == binding["final_projection_sha256"]):
                failed = True
    return failed


def prepare_review(result, data_dir, *, database=None, supersedes_admission_decision_id=None):
    """Local command: freeze actual PDF, rendered pixels, raw and final rows."""
    from scripts.audit_sample import audit_operation_lock, emit_bound_review_bundle
    from analyzer.ocr.ingestion import approved_data_root, read_work_artifact

    _validate_supersession_id(supersedes_admission_decision_id)
    with approved_data_root(data_dir) as root, audit_operation_lock(root):
        material = review_material(result)
        directory = bundle_directory(root, material)
        record = bundle_record(result, material)
        content = read_work_artifact(root, result.work.artifact_path)
        emit_bound_review_bundle(root, directory, encode(record), content)
        _validate_bundle(root, directory, record)
        required = supersedes_admission_decision_id
        if database is not None:
            current = required_review_supersession(database.work, material["binding"]["raw_result_id"])
            if required is not None and required != current:
                raise ValueError("explicit supersession does not match the borrowed owner's current decision")
            required = current
        # Retractions get a new unapproved checklist, never overwrite an old
        # draft and never reset sticky failures on the immutable candidate.
        name = f"report-template-{required}.json" if required else "report-template.json"
        try:
            with (directory / name).open("x") as template:
                json.dump(review_report_template(result, material,
                          supersedes_admission_decision_id=required), template, indent=2, sort_keys=True)
        except FileExistsError:
            pass  # Never overwrite an operator's draft on repeated preparation.
        ai_name = f"ai-report-template-{required}.json" if required else "ai-report-template.json"
        try:
            with (directory / ai_name).open("x") as template:
                json.dump(ai_review_report_template(result, root,
                          supersedes_admission_decision_id=required), template, indent=2, sort_keys=True)
        except FileExistsError:
            pass
        return str(Path(data_dir) / directory.relative_to(Path(root)))


def _validate_supersession_id(supersedes):
    if supersedes is not None and (not isinstance(supersedes, str) or len(supersedes) != 64
                                   or any(char not in "0123456789abcdef" for char in supersedes)):
        raise ValueError("supersedes_admission_decision_id must be a lowercase SHA-256 or null")


def _validate_report(report, result, material, verdict):
    if not isinstance(report, dict) or report.get("binding") != material["binding"]:
        raise ValueError("review report must bind the exact frozen candidate")
    if report.get("policy_sha256") != digest(POLICY):
        raise ValueError("review policy identity missing or changed")
    _validate_supersession_id(report.get("supersedes_admission_decision_id"))
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


def ai_review_report_template(result, data_dir, *, supersedes_admission_decision_id=None):
    """Unapproved two-session checklist. Call after prepare_review; never copies proposed values."""
    from copy import deepcopy
    from analyzer.ocr.ingestion import read_work_artifact

    material = review_material(result)
    directory = bundle_directory(data_dir, material)
    _validate_bundle(data_dir, directory, bundle_record(result, material))
    seal = json.loads(read_work_artifact(data_dir, directory / "bundle-seal.json"))
    report = review_report_template(result, material,
                                   supersedes_admission_decision_id=supersedes_admission_decision_id)
    report.update(reviewer_id="", session_id="", review_mode="",
                  original_source_ledger="", original_source_ledger_sha256="",
                  ledger_comparison_attested=False)
    for page, image in zip(report["pages"], seal["pages"]):
        page["image_sha256"] = image["sha256"]
    for row in report["rows"]:
        row["fields"] = {field: {"status": "unknown", "value": None, "literal": "",
                                 "page_number": row["page_number"], "image_sha256": "", "evidence": ""}
                         for field in POLICY["critical_fields"]}
    return {"binding": material["binding"], "policy_sha256": digest(POLICY), "evidence": "",
            "supersedes_admission_decision_id": supersedes_admission_decision_id,
            "source_checking_attested": False, "reports": [deepcopy(report), deepcopy(report)]}


def _validate_ai_report(report, result, material, directory, data_dir):
    from analyzer.ocr.ingestion import read_work_artifact
    from analyzer.ocr.common import _strict_date, _normalize_tx_type, validate_transactions
    from analyzer.ocr.normalization import normalize_amount
    from analyzer.parsing.cells import extract_owner_code

    _validate_report(report, result, material, "fail")  # Envelope binding and evidence.
    reports = report.get("reports")
    if report.get("source_checking_attested") is not True or not isinstance(reports, list) or len(reports) != 2:
        raise ValueError("two locally attested independent source reports required")
    seal = json.loads(read_work_artifact(data_dir, directory / "bundle-seal.json"))
    identities = [set(), set()]
    observations = []
    for source in reports:
        _validate_report(source, result, material, "pass")
        # Binding is the local comparison, not a claim a blind reviewer saw the
        # proposal. Preserve the exact original UTF-8 ledger, including its scope.
        ledger = source.get("original_source_ledger")
        if (source.get("review_mode") not in {"blind_source_ledger", "candidate_comparison"}
                or source.get("ledger_comparison_attested") is not True
                or not isinstance(ledger, str) or not ledger.strip()
                or hashlib.sha256(ledger.encode("utf-8")).hexdigest() != source.get("original_source_ledger_sha256")
                or not isinstance(json.loads(ledger), dict)):
            raise ValueError("original immutable source ledger and local comparison attestation required")
        if source.get("supersedes_admission_decision_id") != report.get("supersedes_admission_decision_id"):
            raise ValueError("source report supersession differs")
        for index, key in enumerate(("reviewer_id", "session_id")):
            identity = source.get(key)
            if not isinstance(identity, str) or not identity.strip() or identity in identities[index]:
                raise ValueError("distinct reviewer and session IDs required")
            identities[index].add(identity)
        for page, image in zip(source["pages"], seal["pages"]):
            if page.get("image_sha256") != image["sha256"]:
                raise ValueError("source page image citation missing or changed")
        values = []
        for row, expected in zip(source["rows"], material["projection"]):
            fields = row.get("fields")
            if not isinstance(fields, dict) or set(fields) != set(POLICY["critical_fields"]):
                raise ValueError("every critical field needs an observation")
            for field, observation in fields.items():
                if not isinstance(observation, dict):
                    raise ValueError("field observation must be an object")
                status = observation.get("status")
                value, literal = observation.get("value"), observation.get("literal")
                page = observation.get("page_number")
                if (type(page) is not int or not 1 <= page <= len(seal["pages"])
                        or observation.get("image_sha256") != seal["pages"][page - 1]["sha256"]
                        or not isinstance(observation.get("evidence"), str) or not observation["evidence"].strip()
                        or not isinstance(literal, str) or not literal.strip()):
                    raise ValueError("every field needs literal source evidence and frozen page citation")
                if value != expected[field]:
                    raise ValueError(f"source observation differs from current projection: {field}")
                if status == "not_reported":
                    if literal != "NotReported" or value not in (None, "unknown"):
                        raise ValueError("not-reported field must remain null/unknown")
                elif status == "official_metadata":
                    if field != "disclosure_date" or literal != result.work.filing_date or value != literal:
                        raise ValueError("only bound official filing metadata can supply disclosure date")
                elif status == "derived":
                    amount = fields["amount_raw"]
                    parsed = normalize_amount(amount.get("literal"))
                    if (field != "amount_midpoint" or literal != amount.get("literal")
                            or amount.get("status") != "observed" or parsed["amount_kind"] not in {"range", "exact"}
                            or parsed["amount_midpoint"] != value):
                        raise ValueError("midpoint must be computed from literal source numeric bounds")
                elif status == "observed":
                    if value is None or value == "unknown" or isinstance(value, bool):
                        raise ValueError("unreported values require explicit not_reported status")
                    observed = literal
                    if field in {"transaction_date", "notification_date", "expiry_date", "disclosure_date"}:
                        parsed = _strict_date(literal)
                        observed = parsed.isoformat() if parsed else None
                    elif field == "strike_price":
                        parsed = normalize_amount(literal)
                        observed = parsed["amount_midpoint"] if parsed["amount_kind"] == "exact" else None
                    elif field == "amount_midpoint":
                        raise ValueError("midpoint requires code derivation")
                    elif field == "owner_code":
                        observed = extract_owner_code(literal)
                        observed = "J" if observed == "JT" else observed
                    elif field == "transaction_type":
                        observed = _normalize_tx_type(literal)
                    elif field == "member":
                        _, rejected = validate_transactions(result.work.doc_id, literal, [],
                                                            result.work.filing_date, result.work.member)
                        observed = None if rejected else value
                    elif field == "asset_description":
                        observed = " ".join(literal.split())
                        value = " ".join(value.split())
                    if observed != value:
                        raise ValueError(f"literal source differs from canonical value: {field}")
                else:
                    raise ValueError(f"unknown/conflicting field status: {field}")
                values.append((field, status, value))
        observations.append(values)
    if observations[0] != observations[1]:
        raise ValueError("independent source observations disagree")


def record_review(result, data_dir, report, *, verdict, human_reviewed=False, ai_source_reviewed=False, database=None):
    """Append a local verdict, never accepting positive silver/model assertions."""
    from scripts.audit_sample import audit_operation_lock
    from analyzer.ocr.ingestion import approved_data_root, read_work_artifact

    if verdict not in {"pass", "fail"}:
        raise ValueError("review verdict must be pass or fail")
    if human_reviewed and ai_source_reviewed:
        raise ValueError("human and AI source review modes are mutually exclusive")
    if verdict == "pass" and not (human_reviewed or ai_source_reviewed):
        raise ValueError("positive authorization requires explicit local human review")
    with approved_data_root(data_dir) as root, audit_operation_lock(root):
        material = review_material(result)
        directory = bundle_directory(root, material)
        record = bundle_record(result, material)
        _validate_bundle(root, directory, record)
        _validate_report(report, result, material, "fail" if ai_source_reviewed else verdict)
        if ai_source_reviewed and verdict == "pass":
            _validate_ai_report(report, result, material, directory, root)
        # Recording acknowledges an operator-supplied subject, not its currentness.
        # DB-free commands may run alongside the writer. Only the owner can
        # authorize admission against the actual latest chain under its lock/txn.
        if verdict == "pass" and database is not None:
            assert_review_supersession(database.work, material["binding"]["raw_result_id"], report)
        history_path = directory / "verdicts.json"
        history = json.loads(read_work_artifact(root, history_path)) if history_path.exists() else []
        entry = {"verdict": verdict, "binding": material["binding"], "report": report,
                 "authority": (POLICY["ai_authority"] if ai_source_reviewed else
                               POLICY["authority"] if human_reviewed else "silver-failure-only"),
                 "bundle_seal_sha256": hashlib.sha256(read_work_artifact(root, directory / "bundle-seal.json")).hexdigest()}
        # A failed unchanged candidate is not repaired by another assertion.
        if verdict == "pass" and _has_failed_projection(root, material, result):
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
        if entry["authority"] not in {POLICY["authority"], POLICY["ai_authority"]} or entry["binding"] != material["binding"]:
            raise ValueError("review is not locally authorized for this exact candidate")
        seal = hashlib.sha256(read_work_artifact(data_dir, directory / "bundle-seal.json")).hexdigest()
        if entry["bundle_seal_sha256"] != seal or entry["verdict"] != "pass":
            raise ValueError("review seal changed or verdict is not pass")
        if entry["authority"] == POLICY["ai_authority"]:
            _validate_ai_report(entry["report"], result, material, directory, data_dir)
        else:
            _validate_report(entry["report"], result, material, "pass")
        if _has_failed_projection(data_dir, material, result):
            raise ValueError("failed candidate requires re-extraction, not a later pass")
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        return {"allowed": False, "reason": f"grounded API review required: {exc}", **material}
    if not result.rows:
        # Authorization derives from the operator's affirmative page report,
        # not from a provider's empty claim. The raw bundle remains unchanged.
        material["candidate"] = {**material["candidate"], "status": "verified_empty"}
    return {"allowed": True, "reason": f"locally recorded all-pages source review passed ({entry['authority']})",
             "review_binding": {**material["binding"], "verdict_sha256": digest(entry),
                                "authority": entry["authority"],
                               "supersedes_admission_decision_id": entry["report"].get("supersedes_admission_decision_id"),
                               "bundle_seal_sha256": seal}, **material}
