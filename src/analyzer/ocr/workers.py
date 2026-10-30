"""Bound-artifact extraction. No queue selection or canonical database I/O.

``spool_dir`` is a durable data root: bundles live in ``results`` and GLM
readings additionally use the existing ``staging/glm-backlog`` audit layout.
The coordinator may publish evidence['staging_record'] into another audit
data root with ingestion.publish_glm_staging_record; never rewrite the bundle.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from analyzer.ocr import common
from analyzer.ocr.contracts import (
    ExtractionLane,
    ExtractionOutcome,
    ResultBundle,
    WorkEnvelope,
)


def extraction_recipe(lane: ExtractionLane) -> str:
    """Hash code, prompts, models and extraction options, never credentials."""
    return _extraction_recipe(lane)


def _extraction_recipe(lane: ExtractionLane, *, model: str | None = None) -> str:
    from analyzer import vision_engines as vision

    if not isinstance(lane, ExtractionLane):
        raise TypeError("lane must be an ExtractionLane")
    root = Path(__file__).resolve().parents[1]
    paths = [Path(__file__), Path(common.__file__), Path(__file__).with_name("contracts.py")]
    options = {"lane": lane.value}
    if lane == ExtractionLane.LOCAL:
        paths.extend(sorted((root / "parsing").glob("*.py")))
        paths.append(root / "models.py")
        paths.append(root / "download.py")  # tolerant worker/watchdog policy is extraction code
        options["skip_docling"] = os.environ.get("PTR_SKIP_DOCLING") == "1"
        versions = {}
        for package in ("pdfplumber", "camelot-py", "pdf2image", "pytesseract", "docling"):
            try:
                versions[package] = version(package)
            except PackageNotFoundError:
                versions[package] = None
        options["packages"] = versions
    else:
        options.update(prompt_sha256=common.PROMPT_SHA256, temperature=0)
        if lane == ExtractionLane.GEMINI:
            options.update(model=common.MODEL, parser=common.GEMINI_PARSER_VERSION)
        else:
            from scripts import glm_stage

            paths.extend([Path(vision.__file__), Path(glm_stage.__file__)])
            # The model is public recipe data; URL/key are NOT recipe material.
            options.update(
                model=model if model is not None else os.environ.get(vision.VISION_B_MODEL_ENV, ""),
                render_dpi=glm_stage.RENDER_DPI,
                max_tokens_floor=vision.MAX_TOKENS_FLOOR,
                max_tokens_base=vision.MAX_TOKENS_BASE,
                max_tokens_per_page=vision.MAX_TOKENS_PER_PAGE,
                max_tokens_ceiling=vision.ZAI_MAX_TOKENS_CEILING,
            )
    options["code"] = {
        str(path.relative_to(root)) if path.is_relative_to(root) else path.name:
        hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }
    digest = hashlib.sha256(_encode(options)).hexdigest()
    return f"bound-extraction-v1:{lane.value}:{digest}"


def _encode(value) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False).encode()


def _publish(result: ResultBundle, spool_dir: str) -> str:
    """Same fsynced-temp/hard-link protocol as immutable decision receipts."""
    content = _encode(result.to_dict())
    digest = hashlib.sha256(content).hexdigest()
    directory = Path(spool_dir) / "results"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{digest}.json"
    fd, temporary = tempfile.mkstemp(dir=directory, prefix=".result-")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != content:
                raise RuntimeError("immutable result collision") from None
        directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        os.unlink(temporary)
    return str(path)


def load_result(path) -> ResultBundle:
    content = Path(path).read_bytes()
    if Path(path).name != f"{hashlib.sha256(content).hexdigest()}.json":
        raise ValueError("result content does not match immutable bundle identity")
    return ResultBundle.from_dict(json.loads(content))


def _pages(work, classification):
    return [{"page_number": n, "outcome": "unknown", "basis": classification}
            for n in range(1, work.page_count + 1)]


def _model_pages(work, parsed):
    if parsed.page_count != work.page_count or parsed.covered_pages != frozenset(range(1, work.page_count + 1)):
        raise ValueError("parsed page coverage does not match bound artifact")
    row_pages = {row.get("page_number") for row in parsed.transactions}
    return [
        {"page_number": n,
         "outcome": "transactions" if n in row_pages else "unknown",
         "basis": "model_reported_rows" if n in row_pages else "model_reported_no_transactions_unverified"}
        for n in range(1, work.page_count + 1)
    ]


def extract_document(work: WorkEnvelope, spool_dir: str) -> str:
    """Validate one immutable PDF, extract once, durably publish one result.

    No canonical file is opened; all engine caches and staging are spool-local.
    Provider failures retain full responses in artifacts, not log messages.
    ``retry_after`` is a delay in seconds, scoped to this work's lane.
    """
    from analyzer import vision_engines as vision

    evidence = {"page_count": work.page_count,
                "page_outcomes": _pages(work, "not_processed"),
                "output_schema_version": None, "legacy_limited": None,
                "provider_responses": []}
    rows = []
    raw = None
    outcome = ExtractionOutcome.BLOCKED
    detail = None
    retry_after = None
    try:
        if work.extraction_recipe != extraction_recipe(work.lane):
            raise ValueError("extraction recipe changed")
        # Snapshot before subprocesses: hash, page count, rendering and parsing
        # all consume these same bytes, never a mutable original pathname.
        with common.snapshot_pdf(work.artifact_path) as snapshot:
            if snapshot.sha256 != work.artifact_sha256:
                raise ValueError("bound artifact SHA-256 mismatch")
            if snapshot.page_count != work.page_count:
                raise ValueError("bound artifact page count mismatch")
            evidence["artifact_verified"] = True
            if work.lane == ExtractionLane.LOCAL:
                from analyzer.download import engine_error_detail, tolerant_parse_pdf_worker
                from analyzer.parsing.cascade import ParserCascadeError, parse_pdf_worker

                source = subprocess.run(
                    ["pdftotext", "-layout", str(snapshot.path), "-"],
                    capture_output=True, text=True, timeout=60, check=False,
                )
                evidence["text_source"] = {"stdout": source.stdout, "stderr": source.stderr, "returncode": source.returncode}
                # An image-only PDF has no raw text source. Do not substitute
                # normalized parser rows and pretend they were source text.
                raw = source.stdout if source.stdout.strip("\f\n\r\t ") else None
                _, rows, engines = tolerant_parse_pdf_worker(snapshot.path, parse_worker=parse_pdf_worker)
                # Preserve the exact parser row objects (including raw extras).
                evidence["engines"] = engines
                failure = engine_error_detail(engines)
                if failure is not None:
                    raise ParserCascadeError(failure)
                evidence["page_outcomes"] = _pages(work, "local_parser_coverage_unattributed")
                outcome = ExtractionOutcome.CANDIDATE if rows else ExtractionOutcome.NEEDS_REVIEW
                detail = None if rows else "local empty result has no source-policy proof"
            elif work.lane == ExtractionLane.GEMINI:
                raw, error, _metadata = common.call_gemini(
                    str(snapshot.path), doc_id=work.doc_id,
                    cache_dir=str(Path(spool_dir) / "gemini_cache"),
                    parser_version=work.extraction_recipe,
                    response_sink=evidence["provider_responses"].append,
                )
                if error:
                    # Error text is evidence, not an operational log/detail.
                    evidence["provider_error"] = error
                    if common.is_quota_error(error):
                        outcome, retry_after = ExtractionOutcome.RETRY_WAIT, 60.0
                        detail = "provider quota/rate limit"
                    else:
                        outcome, detail = ExtractionOutcome.NEEDS_REVIEW, "provider extraction failed"
                else:
                    if raw is None:
                        raise ValueError("provider returned no source response")
                    if (_metadata is None or _metadata.parsed is None
                            or _metadata.sha256 != work.artifact_sha256
                            or _metadata.page_count != work.page_count):
                        raise ValueError("provider success lacks bound validated response metadata")
                    parsed = _metadata.parsed
                    rows = parsed.transactions
                    evidence["member"] = parsed.member
                    evidence["page_outcomes"] = _model_pages(work, parsed)
                    evidence["output_schema_version"] = parsed.output_schema_version
                    evidence["legacy_limited"] = parsed.legacy_limited
                    outcome = ExtractionOutcome.CANDIDATE if rows else ExtractionOutcome.NEEDS_REVIEW
                    detail = None if rows else "model no_txs has no source-policy proof"
            else:
                from scripts import glm_stage

                config = vision.VisionBConfig.from_env()
                doc = vision.DualReadDoc(work.doc_id, work.archive_year, str(snapshot.path))

                def read_bound(path, page_count):
                    return vision.read_document_with_engine_b(
                        doc, config=config, page_count=page_count,
                        page_images=lambda pdf, count: glm_stage.rasterize_page_data_urls(pdf),
                        response_sink=evidence["provider_responses"].append,
                    )

                record = glm_stage.build_staging_record(
                    work.doc_id, work.archive_year, snapshot.path, config,
                    read_fn=read_bound,
                    artifact_identity={
                        "archive_year": work.archive_year,
                        "ingestion_generation": work.generation_id,
                        "artifact_sha256": work.artifact_sha256,
                        "pdf_bytes": snapshot.path.read_bytes(),
                    },
                )
                record["extraction_recipe"] = work.extraction_recipe
                record["attempt_id"] = work.attempt_id
                evidence["staging_record"] = record
                stage_path, _ = glm_stage.publish_staging_record(spool_dir, record)
                evidence["staging_path"] = str(stage_path)
                raw = record["raw_output"]
                if record["parse_status"] == "error":
                    kind = record.get("error_kind")
                    error = record.get("error") or ""
                    if kind == "quota" or common.is_quota_error(error):
                        outcome, retry_after = ExtractionOutcome.RETRY_WAIT, 60.0
                        detail = "provider quota/rate limit"
                    else:
                        outcome, detail = ExtractionOutcome.NEEDS_REVIEW, f"provider extraction failed ({kind})"
                else:
                    if raw is None:
                        raise ValueError("provider returned no source response")
                    parsed = common.parse_gemini_output(raw, expected_page_count=work.page_count)
                    rows = parsed.transactions
                    evidence["member"] = parsed.member
                    evidence["page_outcomes"] = _model_pages(work, parsed)
                    evidence["output_schema_version"] = parsed.output_schema_version
                    evidence["legacy_limited"] = parsed.legacy_limited
                    outcome = ExtractionOutcome.CANDIDATE if rows else ExtractionOutcome.NEEDS_REVIEW
                    detail = "GLM pixel audit required" if rows else "model no_txs has no source-policy proof"
    except Exception as exc:  # noqa: BLE001 -- failure evidence is the worker's durable output
        # Exceptions can contain complete model bodies; store them only here.
        evidence["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        detail = "bound extraction failed; see result evidence"
    if raw is None and evidence["provider_responses"]:
        raw = evidence["provider_responses"][-1]
    try:
        result = ResultBundle(work, outcome, rows, raw, evidence, detail, retry_after)
    except ValueError as exc:
        # Backend data is not trusted JSON. Preserve the original source and
        # an opaque diagnostic representation; never coerce bad row values
        # into admissible numbers, or let one document abort sibling work.
        rejected = repr({"rows": rows, "raw_response": raw, "evidence": evidence})
        result = ResultBundle(
            work, ExtractionOutcome.NEEDS_REVIEW, [],
            raw if isinstance(raw, str) else None,
            {"page_count": work.page_count,
             "page_outcomes": _pages(work, "invalid_backend_result"),
             "artifact_verified": evidence.get("artifact_verified") is True,
             "contract_failure": {"type": type(exc).__name__, "message": str(exc)},
             "rejected_candidate_repr": rejected},
            "backend result violates the result contract; see retained source evidence",
        )
    return _publish(result, spool_dir)


def admission_decision(result: ResultBundle, data_dir) -> dict:
    """Inspection only. Canonical publication requires ``admission_guard``."""
    with admission_guard(result, data_dir) as decision:
        return decision


@contextmanager
def admission_guard(result: ResultBundle, data_dir):
    """Yield a decision while retaining GLM audit and document authorization.

    The coordinator must keep this scope around normalization AND canonical
    commit, including readmission. Do not call admission_decision or acquire
    the audit-root lock again inside the scope: that lock is not reentrant.
    """
    if result.work.lane == ExtractionLane.LOCAL:
        yield _admission_decision_locked(result, data_dir)
        return
    from scripts import glm_stage
    from contextlib import nullcontext
    from analyzer.ocr.review import ReviewAuthorization, review_decision_locked
    from analyzer.ocr.ingestion import approved_data_root

    with (
        approved_data_root(data_dir) as root,
        glm_stage.audit_operation_lock(root),
        (glm_stage.document_publication_lock(root, result.work.archive_year, result.work.doc_id)
         if result.work.lane == ExtractionLane.GLM else nullcontext()),
    ):
        decision = _admission_decision_locked(result, root)
        if decision["allowed"]:
            try:
                decision = {**decision, **review_decision_locked(result, root)}
            except ValueError as exc:
                decision = {**decision, "allowed": False, "reason": str(exc)}
        authorization = ReviewAuthorization(decision)
        try:
            yield authorization
        finally:
            authorization.active = False


def _admission_decision_locked(result: ResultBundle, data_dir) -> dict:
    refused = {"allowed": False, "reason": result.detail or result.outcome.value, "audit_binding": None}
    if result.outcome not in (ExtractionOutcome.CANDIDATE, ExtractionOutcome.VERIFIED_EMPTY):
        return refused
    if result.evidence.get("artifact_verified") is not True:
        return {**refused, "reason": "artifact verification evidence missing"}
    if result.work.extraction_recipe != extraction_recipe(result.work.lane):
        return {**refused, "reason": "extraction recipe changed"}
    if result.work.lane != ExtractionLane.GLM:
        return {"allowed": True, "reason": "candidate; writer validation required", "audit_binding": None}
    from scripts import glm_stage

    record = glm_stage.load_staging_record(data_dir, result.work.archive_year, result.work.doc_id)
    expected = result.evidence.get("staging_record")
    if record is None or expected is None or glm_stage.audit_staged_record_sha256(record) != glm_stage.audit_staged_record_sha256(expected):
        return {**refused, "reason": "staging evidence missing or changed"}
    state = glm_stage.audit_report_state(data_dir)
    history = glm_stage.audit_history_sha256(glm_stage.load_audit_results(data_dir))
    sample = glm_stage.sample_lot_file_sha256(data_dir, state.get("sample_file") if state else None)
    if not glm_stage.audit_report_gate_open(state, history, sample):
        return {**refused, "reason": glm_stage._audit_report_gate_issue(state, history, sample)}
    bound, reason = glm_stage.verdict_binding(data_dir, result.work.doc_id, record)
    if record.get("audit") != "pass" or not bound:
        return {**refused, "reason": reason or "staging record lacks audit pass"}
    return {
        "allowed": True, "reason": "bound pixel audit passed; writer validation required",
        "audit_binding": {"audit_history_sha256": history, "sample_file_sha256": sample,
                          "staged_record_sha256": glm_stage.audit_staged_record_sha256(record)},
    }
