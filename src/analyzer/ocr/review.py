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
    "render_dpi": 180,
    "ai_requirements": "two independent sessions, one blind source ledger and one candidate comparison; pixel citations; all field observations; original UTF-8 ledgers with SHA-256; local source-checking and faithful comparison attestations; blind ledger binding is post-review",
    "source_proposals": "AI-only schema-4 source observations; original raw result retained; both original ledger hashes bound; all current normalization gates; corrections publish as separate derived decisions",
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


def _ledger_at(ledger, pointer, *, within=None):
    """Resolve a local JSON pointer; no evaluation, guessing or cross-document lookup."""
    if pointer == "" and within is None:
        return ledger
    if (not isinstance(pointer, str) or not pointer.startswith("/")
            or (within is not None and not pointer.startswith(within + "/"))):
        raise ValueError("exact original-ledger JSON pointer required")
    try:
        for part in pointer[1:].split("/"):
            key = part.replace("~1", "/").replace("~0", "~")
            ledger = ledger[int(key)] if isinstance(ledger, list) else ledger[key]
        return ledger
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ValueError("original-ledger JSON pointer does not resolve") from exc


def _original_observation(ledger, document_pointer, observation):
    import re

    original = _ledger_at(ledger, observation.get("ledger_pointer"), within=document_pointer)
    if (not isinstance(original, dict) or original.get("source_page") != observation.get("page_number")
            or not isinstance(original.get("evidence_location"), str) or not original["evidence_location"].strip()):
        raise ValueError("field must reference an actual page-located original source observation")
    if observation.get("status") == "not_reported":
        if original.get("status") != "not_reported" or original.get("normalized_value") not in (None, "unknown"):
            raise ValueError("unreported status contradicts original source ledger")
        return original
    literal = original.get("literal")
    if (original.get("status") != "observed" or not isinstance(literal, str) or not literal.strip()
            or not isinstance(observation.get("literal"), str)
            or not re.search(r"(?<![\d.,])" + re.escape("".join(observation["literal"].split())) + r"(?![\d.,])",
                             "".join(literal.split()))):
        raise ValueError("observed literal is absent from original source ledger")
    return original


def _complete_source_amount(literal):
    """A selected monetary observation must contain one complete numeric claim."""
    import re
    from analyzer.ocr.normalization import normalize_amount

    number = r"\$?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{1,2})?"
    pattern = (r"(?<![\w\d.,])(?:" + r"(?:over|greater than|>)\s*" + number
               + "|" + number + r"\s*(?:-|–|—|to)\s*" + number
               + "|" + number + r")(?![\w\d.,])")
    claims = list(re.finditer(pattern, literal, re.I))
    if len(claims) != 1:
        raise ValueError("source amount observation must preserve one complete numeric claim")
    amount = normalize_amount(claims[0].group())
    if amount["amount_kind"] not in {"range", "exact", "open"}:
        raise ValueError("source amount numeric bounds are unknown")
    return amount


def review_material(result, *, source_proposal=None):
    """One normalization and one typed final persistence projection; no I/O."""
    from analyzer.ocr.normalization import normalize_candidate
    from analyzer.ocr.ingestion import final_ocr_projection

    if result.work.lane == ExtractionLane.LOCAL:
        raise ValueError("source review is for API candidates")
    if result.outcome not in (ExtractionOutcome.CANDIDATE, ExtractionOutcome.VERIFIED_EMPTY):
        raise ValueError("only an unchanged candidate or empty claim can be reviewed")
    subject = result
    if source_proposal is not None:
        from dataclasses import replace
        from copy import deepcopy

        if (not isinstance(source_proposal, dict)
                or source_proposal.get("schema") != "independent-ai-source-proposal-v1"
                or type(result.evidence.get("output_schema_version")) is not int
                or result.evidence["output_schema_version"] != 4
                or any(not isinstance(row.get("raw_cells"), dict) for row in result.rows)):
            raise ValueError("source proposals require an explicit source schema and original schema-4 result")
        rows = source_proposal.get("rows")
        required = {"asset", "type", "date", "notif_date", "amount_raw", "owner", "account", "instrument_type", "raw_cells", "page_number"}
        ledgers = source_proposal.get("source_ledger_sha256s")
        if (not isinstance(rows, list) or any(not isinstance(row, dict) or not required <= row.keys()
                or not isinstance(row["raw_cells"], dict) for row in rows)
                or not isinstance(source_proposal.get("member"), str)
                or not isinstance(ledgers, list) or len(ledgers) != 2):
            raise ValueError("complete source rows, observed member and two original ledger hashes required")
        for ledger in ledgers:
            _validate_supersession_id(ledger)
            if ledger is None:
                raise ValueError("source ledger hash cannot be null")
        subject = replace(result, rows=deepcopy(rows), evidence={
            "member": source_proposal["member"], "output_schema_version": 4,
            "page_count": result.work.page_count,
            "page_outcomes": deepcopy(source_proposal.get("page_outcomes")),
        })
    if source_proposal is None and not result.rows:
        from analyzer.ocr.common import parse_gemini_output

        if (not isinstance(result.raw_response, str) or not result.raw_response.strip()
                or result.evidence.get("failure") or result.evidence.get("provider_error")):
            raise ValueError("empty review requires a successfully parsed source claim, not extraction failure")
        parsed = parse_gemini_output(result.raw_response, expected_page_count=result.work.page_count)
        if (parsed.transactions or parsed.raw_row_count or not parsed.no_transactions
                or parsed.covered_pages != frozenset(range(1, result.work.page_count + 1))):
            raise ValueError("empty review requires an explicit parsed claim covering every source page")
    candidate = normalize_candidate(result.work, subject)
    if candidate["findings"] or (subject.rows and candidate["status"] != "candidate"):
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
    material = {"candidate": candidate, "projection": projection, "binding": binding}
    if source_proposal is not None:
        binding["source_proposal_sha256"] = digest(source_proposal)
        material["source_proposal"] = deepcopy(source_proposal)
        material["source_review_lineage"] = {
            "source_origin": "independent-ai-source-proposal-v1",
            "original_raw_result_id": binding["raw_result_id"],
            "extracted_raw_count": len(result.rows), "verified_source_count": len(subject.rows),
            "source_proposal_sha256": digest(source_proposal),
            "source_ledger_sha256s": ledgers,
            "derived_outcome": "candidate" if subject.rows else "verified_empty",
        }
    return material


def bundle_record(result, material):
    record = {
        "schema": "house-api-review-bundle-v1", "doc_id": result.work.doc_id,
        "year": result.work.archive_year, "pages": result.work.page_count,
        "pdf_sha256": result.work.artifact_sha256,
        "binding": material["binding"], "policy": POLICY,
        "raw_result": result.to_dict(), "candidate": material["candidate"],
        "final_projection": material["projection"],
    }
    if "source_proposal" in material:
        record["source_proposal"] = material["source_proposal"]
        record["source_review_lineage"] = material["source_review_lineage"]
    return record


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
        if (("source_proposal_sha256" in old) != ("source_proposal" in record)
                or ("source_proposal" in record and old["source_proposal_sha256"] != digest(record["source_proposal"]))):
            raise ValueError("invalid sibling source proposal binding")
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


def prepare_review(result, data_dir, *, database=None, supersedes_admission_decision_id=None, source_proposal=None):
    """Local command: freeze actual PDF, rendered pixels, raw and final rows."""
    from scripts.audit_sample import audit_operation_lock, emit_bound_review_bundle
    from analyzer.ocr.ingestion import approved_data_root, read_work_artifact

    _validate_supersession_id(supersedes_admission_decision_id)
    with approved_data_root(data_dir) as root, audit_operation_lock(root):
        material = review_material(result, source_proposal=source_proposal)
        directory = bundle_directory(root, material)
        record = bundle_record(result, material)
        content = read_work_artifact(root, result.work.artifact_path)
        emit_bound_review_bundle(root, directory, encode(record), content, render_dpi=POLICY["render_dpi"])
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
                          supersedes_admission_decision_id=required, source_proposal=source_proposal), template, indent=2, sort_keys=True)
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


def ai_review_report_template(result, data_dir, *, supersedes_admission_decision_id=None, source_proposal=None):
    """Unapproved two-session checklist. Call after prepare_review; never copies proposed values."""
    from copy import deepcopy
    from analyzer.ocr.ingestion import read_work_artifact

    material = review_material(result, source_proposal=source_proposal)
    directory = bundle_directory(data_dir, material)
    _validate_bundle(data_dir, directory, bundle_record(result, material))
    seal = json.loads(read_work_artifact(data_dir, directory / "bundle-seal.json"))
    report = review_report_template(result, material,
                                   supersedes_admission_decision_id=supersedes_admission_decision_id)
    report.update(reviewer_id="", session_id="", review_mode="",
                  original_source_ledger="", original_source_ledger_sha256="",
                  ledger_comparison_attested=False, ledger_document_pointer="",
                  ledger_pdf_sha256_pointer="", ledger_row_count_pointer="", ledger_rows_pointer="", ledger_absence_pointer="")
    if not material["projection"]:
        report["source_header"] = {"member": {
            "status": "unknown", "value": None, "literal": "", "page_number": None,
            "image_sha256": "", "evidence": "", "ledger_pointer": ""}}
    for page, image in zip(report["pages"], seal["pages"]):
        page["image_sha256"] = image["sha256"]
        page["ledger_image_sha256_pointer"] = ""
    for row in report["rows"]:
        row["fields"] = {field: {"status": "unknown", "value": None, "literal": "",
                                 "page_number": row["page_number"], "image_sha256": "", "evidence": "", "ledger_pointer": ""}
                         for field in POLICY["critical_fields"]}
    envelope = {"binding": material["binding"], "policy_sha256": digest(POLICY), "evidence": "",
            "supersedes_admission_decision_id": supersedes_admission_decision_id,
            "source_checking_attested": False, "reports": [deepcopy(report), deepcopy(report)]}
    if source_proposal is not None:
        envelope["source_proposal"] = material["source_proposal"]
    return envelope


def _validate_ai_report(report, result, material, directory, data_dir):
    from analyzer.ocr.ingestion import read_work_artifact
    from analyzer.ocr.common import _strict_date, _normalize_tx_type, validate_transactions
    from analyzer.ocr.normalization import normalize_amount
    from analyzer.parsing.cells import extract_owner_code
    from decimal import Decimal

    def same_numeric_amount(left, right):
        left, right = normalize_amount(left), normalize_amount(right)
        if left["amount_kind"] not in {"range", "exact", "open"} or left["amount_kind"] != right["amount_kind"]:
            return False
        return all((Decimal(left[key]) if left[key] is not None else None) ==
                   (Decimal(right[key]) if right[key] is not None else None)
                   for key in ("amount_lower", "amount_upper", "amount_exact"))

    def same_value(field, left, right):
        # Permit only the witnessed category-label casing; unknowns stay exact.
        return (left == right or (field == "instrument_type"
                and left in ("Stock", "stock") and right in ("Stock", "stock")))

    _validate_report(report, result, material, "fail")  # Envelope binding and evidence.
    reports = report.get("reports")
    if report.get("source_checking_attested") is not True or not isinstance(reports, list) or len(reports) != 2:
        raise ValueError("two locally attested independent source reports required")
    seal = json.loads(read_work_artifact(data_dir, directory / "bundle-seal.json"))
    identities = [set(), set()]
    modes = set()
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
        ledger = json.loads(ledger)
        modes.add(source["review_mode"])
        document_pointer = source.get("ledger_document_pointer")
        document = _ledger_at(ledger, document_pointer)
        if (not isinstance(document, dict) or str(document.get("doc_id")) != result.work.doc_id
                or _ledger_at(ledger, source.get("ledger_pdf_sha256_pointer"), within=document_pointer) != result.work.artifact_sha256
                or type(_ledger_at(ledger, source.get("ledger_row_count_pointer"), within=document_pointer)) is not int
                or _ledger_at(ledger, source["ledger_row_count_pointer"], within=document_pointer) != len(material["projection"])):
            raise ValueError("original source document, artifact or actual occurrence count differs")
        original_rows = _ledger_at(ledger, source.get("ledger_rows_pointer"), within=document_pointer)
        if not isinstance(original_rows, list) or len(original_rows) != len(material["projection"]):
            raise ValueError("original source ledger must cover every actual row occurrence")
        if source.get("supersedes_admission_decision_id") != report.get("supersedes_admission_decision_id"):
            raise ValueError("source report supersession differs")
        for index, key in enumerate(("reviewer_id", "session_id")):
            identity = source.get(key)
            if not isinstance(identity, str) or not identity.strip() or identity in identities[index]:
                raise ValueError("distinct reviewer and session IDs required")
            identities[index].add(identity)
        for page, image in zip(source["pages"], seal["pages"]):
            if (page.get("image_sha256") != image["sha256"]
                    or _ledger_at(ledger, page.get("ledger_image_sha256_pointer"), within=document_pointer) != image["sha256"]):
                raise ValueError("source page image citation missing or changed")
        if "source_proposal" in material:
            if [{"page_number": page["page_number"], "outcome": page["outcome"]}
                    for page in source["pages"]] != material["candidate"]["page_outcomes"]:
                raise ValueError("source proposal and observed page outcomes differ")
        if not material["projection"]:
            header = source.get("source_header", {}).get("member")
            if not isinstance(header, dict):
                raise ValueError("empty source review requires observed filer header")
            page, literal = header.get("page_number"), header.get("literal")
            if (header.get("status") != "observed" or header.get("value") != result.work.member
                    or not isinstance(literal, str) or not literal.strip()
                    or type(page) is not int or not 1 <= page <= len(seal["pages"])
                    or header.get("image_sha256") != seal["pages"][page - 1]["sha256"]
                    or not isinstance(header.get("evidence"), str) or not header["evidence"].strip()):
                raise ValueError("empty source review requires pixel-cited filer identity")
            _, rejected = validate_transactions(result.work.doc_id, literal, [],
                                                result.work.filing_date, result.work.member)
            if rejected:
                raise ValueError("empty source filer differs from official identity")
            _original_observation(ledger, document_pointer, header)
            absence = _ledger_at(ledger, source.get("ledger_absence_pointer"), within=document_pointer)
            if (not isinstance(absence, dict) or absence.get("status") != "observed"
                    or not isinstance(absence.get("literal"), str) or not absence["literal"].strip()
                    or type(absence.get("source_page")) is not int
                    or not 1 <= absence["source_page"] <= len(seal["pages"])
                    or not isinstance(absence.get("evidence_location"), str) or not absence["evidence_location"].strip()):
                raise ValueError("empty source review requires original page-grounded absence observation")
        values = []
        for index, (row, expected) in enumerate(zip(source["rows"], material["projection"])):
            original_row_pointer = source["ledger_rows_pointer"] + f"/{index}"
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
                if not same_value(field, value, expected[field]):
                    raise ValueError(f"source observation differs from current projection: {field}")
                original = (None if status == "official_metadata" else
                            _original_observation(ledger, document_pointer, observation))
                if field not in {"member", "disclosure_date"} and (
                        not observation.get("ledger_pointer", "").startswith(original_row_pointer + "/")
                        or page != row["page_number"]):
                    raise ValueError("row field must reference its own original source occurrence and page")
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
                            or amount.get("status") != "observed" or parsed["amount_kind"] not in {"range", "exact", "open"}
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
                    elif field == "amount_raw":
                        observed = value if same_numeric_amount(literal, value) else None
                        complete = _complete_source_amount(original["literal"])
                        if not same_numeric_amount(complete["amount_raw"], literal):
                            raise ValueError("source amount observation truncates or changes original numeric bounds")
                    elif field == "transaction_type":
                        observed = _normalize_tx_type(literal)
                    elif field == "member":
                        if " ".join(original["literal"].split()) != " ".join(literal.split()):
                            raise ValueError("member literal must preserve original source observation")
                        _, rejected = validate_transactions(result.work.doc_id, literal, [],
                                                            result.work.filing_date, result.work.member)
                        observed = None if rejected else value
                    elif field == "asset_description":
                        if " ".join(original["literal"].split()) != " ".join(literal.split()):
                            raise ValueError("asset literal must preserve complete original source text")
                        observed = " ".join(literal.split())
                        value = " ".join(value.split())
                    if not same_value(field, observed,
                                      expected[field] if field == "instrument_type" else value):
                        raise ValueError(f"literal source differs from canonical value: {field}")
                    original_value = original.get("normalized_value")
                    if field == "transaction_type":
                        original_value = _normalize_tx_type(original_value)
                    elif field == "owner_code":
                        original_value = extract_owner_code(original_value)
                    elif field == "amount_raw" and not isinstance(original_value, dict):
                        original_value = value if same_numeric_amount(original_value, value) else None
                    elif field in {"transaction_date", "notification_date", "expiry_date", "disclosure_date"}:
                        parsed = _strict_date(original_value)
                        original_value = parsed.isoformat() if parsed else None
                    comparison_value = expected[field] if field == "instrument_type" else value
                    if (not isinstance(original_value, dict) and field not in {"member", "asset_description"}
                            and not same_value(field, original_value, comparison_value)):
                        raise ValueError(f"original source value differs from canonical value: {field}")
                else:
                    raise ValueError(f"unknown/conflicting field status: {field}")
                values.append((field, status,
                               expected[field] if field == "instrument_type" else value))
        observations.append(values)
    if observations[0] != observations[1]:
        raise ValueError("independent source observations disagree")
    if modes != {"blind_source_ledger", "candidate_comparison"}:
        raise ValueError("one blind source ledger and one candidate comparison required")
    if "source_proposal" in material:
        if (report.get("source_proposal") != material["source_proposal"]
                or sorted(material["source_proposal"]["source_ledger_sha256s"]) != sorted(
                    source["original_source_ledger_sha256"] for source in reports)):
            raise ValueError("source proposal must bind both exact original source ledgers")


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
        if report.get("source_proposal") is not None and not ai_source_reviewed:
            raise ValueError("source proposals require explicit independent AI source review mode")
        material = review_material(result, source_proposal=report.get("source_proposal"))
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
        if verdict == "pass":
            atomic_write_json(directory.parent.with_suffix(".json"), {
                "binding": material["binding"], "verdict_sha256": digest(entry)})
        return entry


def review_decision_locked(result, data_dir):
    """Caller owns the audit root through final SQL comparison and commit."""
    from analyzer.ocr.ingestion import read_work_artifact

    raw_directory = Path(data_dir) / "pipeline" / "reviews" / digest(result.to_dict())
    selection_path = raw_directory.with_suffix(".json")
    selection = None
    if selection_path.exists() or selection_path.is_symlink():
        selection = json.loads(read_work_artifact(data_dir, selection_path))
        selected = raw_directory / digest(selection["binding"])
        frozen = json.loads(read_work_artifact(data_dir, selected / "record.json"))
        material = review_material(result, source_proposal=frozen.get("source_proposal"))
        if selection["binding"] != material["binding"]:
            raise ValueError("selected source review no longer matches current normalization, policy or proposal")
    else:
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
        if selection is not None and selection["verdict_sha256"] != digest(entry):
            raise ValueError("selected source review verdict changed")
        if _has_failed_projection(data_dir, material, result):
            raise ValueError("failed candidate requires re-extraction, not a later pass")
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        return {"allowed": False, "reason": f"grounded API review required: {exc}", **material}
    if not material["candidate"]["rows"]:
        # Authorization derives from the operator's affirmative page report,
        # not from a provider's empty claim. The raw bundle remains unchanged.
        material["candidate"] = {**material["candidate"], "status": "verified_empty"}
    return {"allowed": True, "reason": f"locally recorded all-pages source review passed ({entry['authority']})",
             "review_binding": {**material["binding"], "verdict_sha256": digest(entry),
                                "authority": entry["authority"],
                               "supersedes_admission_decision_id": entry["report"].get("supersedes_admission_decision_id"),
                               "bundle_seal_sha256": seal}, **material}
