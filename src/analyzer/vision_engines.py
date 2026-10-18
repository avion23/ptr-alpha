"""Dual vision read: engine B client, agreement orchestration, and persistence.

Engine A is the existing Gemini path (``scripts.gemini_ocr_common``, litellm
routing). Engine B is any OpenAI-compatible chat-completions vision endpoint —
built for the operator's forthcoming vLLM server (Qwen2.5-VL-class) — configured
entirely through the environment:

* ``VISION_B_BASE_URL``   — e.g. ``http://gpu-host:8000/v1`` (required)
* ``VISION_B_MODEL``      — served model id (required)
* ``VISION_B_API_KEY``    — optional bearer token
* ``VISION_B_TIMEOUT``    — optional seconds (default 180)

Per repo policy the endpoint must not serve OpenAI-operated GPT models; the
point of engine B is an *independent model family*, and no default points at
any hosted endpoint.

Decisions, per document (``docs/reviews/extraction-pipeline-fp-20261018.md``):

* ``AGREE``  — normalized row multisets identical: rows persist exactly as
  today (Gemini reading, same validation/insert path) under parser_version
  ``v6-dual-validated`` and ``engines_attempted = gemini+visionb``.
* ``DISAGREE`` — nothing persists from either side; the doc lands in the
  existing manual-review inventory (a ``pdf_parse_runs`` row with status
  ``invalid``, which the generation-acceptance predicates already treat as a
  reviewed/no-rows terminal state) and both readings are stored as JSON in
  the run's ``error_message`` notes column.
* ``A_ONLY`` — engine B unconfigured or unavailable (transport failure,
  timeout, malformed envelope): fall back to A alone under
  ``v5-gemini-validated`` — byte-for-byte today's behavior, never an error.
* ``A_ERROR`` — engine A itself failed: the caller's existing failed-attempt
  handling records it.

An engine-B response that arrives but violates the extraction schema is NOT
treated as "engine absent" (that would silently reintroduce single-model
hallucination risk): it is fail-closed into the DISAGREE review inventory.

Wiring is opt-in only: ``DUAL_READ=1`` (plus engine-B configuration) in the
OCR workers — ``scripts/ocr_zero_rows.main`` (the nightly backlog cron, whose
wrapper sources the operator's engine-B env when present) and
``run_gemini_ocr_for_year`` (``refresh --gemini-ocr``). Unset (the default)
keeps every path single-engine.
"""

from __future__ import annotations

import base64
import io
import os
from dataclasses import dataclass
from typing import Callable, Mapping

import requests

from analyzer.vision_agreement import (
    AGREE,
    DISAGREE,
    GateResult,
    disagreement_notes,
    gate_rows,
)

DUAL_PARSER_VERSION = "v6-dual-validated"
DUAL_ENGINES_ATTEMPTED = "gemini+visionb"
A_ONLY = "A_ONLY"
A_ERROR = "A_ERROR"
DUAL_READ_ENV = "DUAL_READ"

VISION_B_BASE_URL_ENV = "VISION_B_BASE_URL"
VISION_B_MODEL_ENV = "VISION_B_MODEL"
VISION_B_API_KEY_ENV = "VISION_B_API_KEY"
VISION_B_TIMEOUT_ENV = "VISION_B_TIMEOUT"
DEFAULT_VISION_B_TIMEOUT = 180.0
PAGE_RENDER_DPI = 200

ReasonSchemaInvalid = "visionb_schema_invalid"
ReasonRowMismatch = "row_mismatch"


class VisionBUnavailable(Exception):
    """Engine B is absent for this run: transport failure, timeout, or a

    malformed response envelope. Callers fall back to engine A alone — this is
    never a parse failure and never blocks the pipeline.
    """


@dataclass(frozen=True)
class VisionBConfig:
    """Connection settings for the OpenAI-compatible vision endpoint."""

    base_url: str
    model: str
    api_key: str | None = None
    timeout: float = DEFAULT_VISION_B_TIMEOUT

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> VisionBConfig | None:
        """Return None when engine B is unconfigured (not an error)."""
        source = os.environ if env is None else env
        base_url = str(source.get(VISION_B_BASE_URL_ENV) or "").strip().rstrip("/")
        model = str(source.get(VISION_B_MODEL_ENV) or "").strip()
        if not base_url or not model:
            return None
        timeout = DEFAULT_VISION_B_TIMEOUT
        raw_timeout = str(source.get(VISION_B_TIMEOUT_ENV) or "").strip()
        if raw_timeout:
            try:
                timeout = float(raw_timeout)
            except ValueError as exc:
                raise VisionBUnavailable(
                    f"invalid {VISION_B_TIMEOUT_ENV}: {raw_timeout!r}"
                ) from exc
            if timeout <= 0:
                raise VisionBUnavailable(
                    f"invalid {VISION_B_TIMEOUT_ENV}: {raw_timeout!r}"
                )
        api_key = str(source.get(VISION_B_API_KEY_ENV) or "").strip() or None
        return cls(base_url=base_url, model=model, api_key=api_key, timeout=timeout)


def _gemini_common():
    from scripts import gemini_ocr_common

    return gemini_ocr_common


def _ocr_zero_rows():
    from scripts import ocr_zero_rows

    return ocr_zero_rows


def read_document_with_vision_b(
    page_data_urls: list[str],
    config: VisionBConfig,
    *,
    prompt: str | None = None,
) -> str:
    """Read all page images with engine B and return the raw completion.

    The prompt is the SAME transaction-extraction prompt the Gemini path uses
    (imported, never forked). All pages travel in one chat-completions request
    so engine B answers for the whole document in the shared schema. Any
    transport/HTTP/envelope problem raises ``VisionBUnavailable``.
    """
    if not page_data_urls:
        raise VisionBUnavailable("visionb requires at least one page image")
    extraction_prompt = (
        prompt if prompt is not None else _gemini_common().PROMPT
    )
    payload = {
        "model": config.model,
        "temperature": 0,
        "messages": [
            {
                "role": "user",
                "content": [
                    *[
                        {"type": "image_url", "image_url": {"url": url}}
                        for url in page_data_urls
                    ],
                    {"type": "text", "text": extraction_prompt},
                ],
            }
        ],
    }
    headers = {"Content-Type": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"
    url = f"{config.base_url.rstrip('/')}/chat/completions"
    try:
        response = requests.post(
            url, json=payload, headers=headers, timeout=config.timeout
        )
    except requests.RequestException as exc:
        raise VisionBUnavailable(f"visionb transport failure: {exc}") from exc
    if response.status_code != 200:
        raise VisionBUnavailable(f"visionb http {response.status_code}")
    try:
        body = response.json()
        text = body["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise VisionBUnavailable(
            f"visionb malformed response envelope: {exc}"
        ) from exc
    if not isinstance(text, str) or not text.strip():
        raise VisionBUnavailable("visionb empty completion")
    return text


def read_document_with_engine_b(
    doc: DualReadDoc,
    *,
    config: VisionBConfig | None = None,
    env: Mapping[str, str] | None = None,
    page_images: Callable[[str, int], list[str]] | None = None,
    page_count: int | None = None,
) -> str:
    """Read one whole document with engine B alone and return the raw output.

    Engine-B-primary single-engine mode (OCR_ENGINE=visionb): rasterize the
    document and ask engine B in one request with the shared extraction
    prompt. Unlike the dual-read gate there is NO engine-A fallback here —
    any transport/configuration problem raises ``VisionBUnavailable`` so the
    caller records a failed attempt instead of silently spending engine A
    (the scarce Gemini quota this mode exists to conserve). When
    ``page_count`` is not supplied it is derived from the PDF itself.
    """
    if config is None:
        try:
            config = VisionBConfig.from_env(env)
        except VisionBUnavailable:
            config = None
    if config is None:
        raise VisionBUnavailable(
            "visionb is not configured: set VISION_B_BASE_URL and VISION_B_MODEL"
        )
    if page_count is None:
        try:
            page_count = _gemini_common().pdf_page_count(doc.pdf_path)
        except OSError as exc:
            raise VisionBUnavailable(f"visionb page count failed: {exc}") from exc
    images = (page_images or _default_page_data_urls)(doc.pdf_path, page_count)
    return read_document_with_vision_b(images, config)


def _default_page_data_urls(pdf_path: str, page_count: int) -> list[str]:
    """Rasterize every PDF page to a base64 PNG data URL (engine-B input)."""
    try:
        from pdf2image import convert_from_path

        pages = convert_from_path(str(pdf_path), dpi=PAGE_RENDER_DPI, fmt="png")
    except Exception as exc:  # poppler/render failures == engine B absent
        raise VisionBUnavailable(f"visionb page rasterization failed: {exc}") from exc
    urls = []
    for page in pages:
        buffer = io.BytesIO()
        page.save(buffer, format="PNG")
        urls.append(
            "data:image/png;base64,"
            + base64.b64encode(buffer.getvalue()).decode("ascii")
        )
    if not urls:
        raise VisionBUnavailable("visionb page rasterization produced no pages")
    return urls


@dataclass(frozen=True)
class DualReadDoc:
    """One document to dual-read."""

    doc_id: str
    year: int
    pdf_path: str
    data_dir: str = "data"
    db_path: str | None = None

    @property
    def resolved_db_path(self) -> str:
        return self.db_path or os.path.join(self.data_dir, "congress.duckdb")


@dataclass(frozen=True)
class DualDecision:
    """Gate decision plus everything a caller needs to persist or fall back."""

    status: str  # AGREE | DISAGREE | A_ONLY | A_ERROR
    parser_version: str
    engines_attempted: str
    raw_row_count: int = 0
    diff_summary: str | None = None
    notes_json: str | None = None
    # Engine-A passthrough for the unchanged single-engine loop path:
    output: str | None = None
    error: str | None = None
    metadata: object | None = None


def dual_read(
    doc: DualReadDoc,
    *,
    refresh: bool = False,
    env: Mapping[str, str] | None = None,
    read_a: Callable[..., tuple] | None = None,
    read_b: Callable[[list[str]], str] | None = None,
    page_images: Callable[[str, int], list[str]] | None = None,
    config: VisionBConfig | None = None,
) -> tuple[DualDecision, list[dict]]:
    """Read one document with engine A, gate against engine B, decide.

    Returns ``(decision, rows)``. ``rows`` carries the canonical candidate
    (engine A's reading) for AGREE and the A-only fallback, and is empty for
    DISAGREE and A errors — nothing persists from either side there. This
    function performs no writes; persistence lives in
    ``run_dual_read_for_doc`` / the caller.
    """
    gemini = _gemini_common()
    if read_a is None:
        read_a = _default_read_a
    output, error, metadata = read_a(doc, refresh=refresh)
    parsed = getattr(metadata, "parsed", None)
    if output is None or error or metadata is None or parsed is None:
        return (
            DualDecision(
                A_ERROR,
                gemini.GEMINI_PARSER_VERSION,
                gemini.MODEL,
                error=error or "gemini read produced no validated output",
                metadata=metadata,
            ),
            [],
        )
    rows_a = list(parsed.transactions)
    raw_row_count = parsed.raw_row_count

    if config is None:
        try:
            config = VisionBConfig.from_env(env)
        except VisionBUnavailable:
            config = None
    if config is None and read_b is None:
        # Engine B unconfigured: identical to current single-engine behavior.
        return (
            DualDecision(
                A_ONLY,
                gemini.GEMINI_PARSER_VERSION,
                gemini.MODEL,
                raw_row_count=raw_row_count,
                output=output,
                metadata=metadata,
            ),
            rows_a,
        )

    try:
        images = (page_images or _default_page_data_urls)(
            doc.pdf_path, parsed.page_count
        )
        if read_b is not None:
            text_b = read_b(images)
        else:
            text_b = read_document_with_vision_b(images, config)
    except VisionBUnavailable as exc:
        # Engine B absent: fallback to A-only, never an exception. The reason
        # rides along in diff_summary so callers can tell a quota/rate-limit
        # stop (is_quota_error) from a transient outage.
        return (
            DualDecision(
                A_ONLY,
                gemini.GEMINI_PARSER_VERSION,
                gemini.MODEL,
                raw_row_count=raw_row_count,
                diff_summary=str(exc),
                output=output,
                metadata=metadata,
            ),
            rows_a,
        )

    try:
        parsed_b = gemini.parse_gemini_output(
            text_b, expected_page_count=parsed.page_count
        )
    except gemini.GeminiOutputError as exc:
        # B answered but not in the extraction schema: fail closed into the
        # review inventory instead of silently trusting A alone.
        summary = f"engine B output failed schema validation: {exc}"
        notes = disagreement_notes(
            rows_a,
            None,
            reason=ReasonSchemaInvalid,
            engines_attempted=DUAL_ENGINES_ATTEMPTED,
            diff_summary=summary,
            member_a=parsed.member,
            raw_b=text_b,
        )
        return (
            DualDecision(
                DISAGREE,
                DUAL_PARSER_VERSION,
                DUAL_ENGINES_ATTEMPTED,
                raw_row_count=raw_row_count,
                diff_summary=summary,
                notes_json=notes,
                metadata=metadata,
            ),
            [],
        )

    gate: GateResult = gate_rows(rows_a, parsed_b.transactions)
    if gate.status == AGREE:
        return (
            DualDecision(
                AGREE,
                DUAL_PARSER_VERSION,
                DUAL_ENGINES_ATTEMPTED,
                raw_row_count=raw_row_count,
                metadata=metadata,
            ),
            rows_a,
        )
    notes = disagreement_notes(
        rows_a,
        parsed_b.transactions,
        reason=ReasonRowMismatch,
        engines_attempted=DUAL_ENGINES_ATTEMPTED,
        diff_summary=gate.diff_summary,
        member_a=parsed.member,
        member_b=parsed_b.member,
    )
    return (
        DualDecision(
            DISAGREE,
            DUAL_PARSER_VERSION,
            DUAL_ENGINES_ATTEMPTED,
            raw_row_count=raw_row_count,
            diff_summary=gate.diff_summary,
            notes_json=notes,
            metadata=metadata,
        ),
        [],
    )


def _default_read_a(doc: DualReadDoc, *, refresh: bool = False) -> tuple:
    """Engine A: the existing Gemini call, including its response cache."""
    return _gemini_common().call_gemini(
        doc.pdf_path,
        doc_id=doc.doc_id,
        refresh=refresh,
        cache_dir=os.path.join(doc.data_dir, "gemini_cache"),
        parser_version=_gemini_common().GEMINI_PARSER_VERSION,
    )


@dataclass(frozen=True)
class DualLoopOutcome:
    """What the OCR worker loop needs after one dual-read attempt."""

    handled: bool  # True: orchestrator already persisted/quarantined the doc
    decision: DualDecision
    inserted: int = 0
    progress_key: str = "errors"
    member: str | None = None
    output: str | None = None
    error: str | None = None
    metadata: object | None = None


def run_dual_read_for_doc(
    doc: DualReadDoc,
    *,
    refresh: bool = False,
    env: Mapping[str, str] | None = None,
    read_a: Callable[..., tuple] | None = None,
    read_b: Callable[[list[str]], str] | None = None,
    page_images: Callable[[str, int], list[str]] | None = None,
) -> DualLoopOutcome:
    """Dual-read one doc and persist the outcome (opt-in DUAL_READ=1 path).

    AGREE persists through the existing ``insert_transactions`` (same row
    source, validation, and per-doc commit) with the dual parser version and
    ``gemini+visionb`` engines. DISAGREE records the manual-review inventory
    run. A_ONLY/A_ERROR are returned unhandled with engine A's result so the
    caller's existing single-engine code continues unchanged.
    """
    decision, rows = dual_read(
        doc,
        refresh=refresh,
        env=env,
        read_a=read_a,
        read_b=read_b,
        page_images=page_images,
    )
    if decision.status in (A_ONLY, A_ERROR):
        return DualLoopOutcome(
            False,
            decision,
            output=decision.output,
            error=decision.error,
            metadata=decision.metadata,
        )

    ocr = _ocr_zero_rows()
    artifact_sha256 = getattr(decision.metadata, "sha256", None)
    try:
        ingestion_generation = ocr.resolve_ingestion_generation(
            doc.resolved_db_path, doc.doc_id, doc.year, artifact_sha256
        )
    except RuntimeError as exc:
        fallback = DualDecision(
            A_ERROR,
            _gemini_common().GEMINI_PARSER_VERSION,
            _gemini_common().MODEL,
            error=str(exc),
            metadata=decision.metadata,
        )
        return DualLoopOutcome(
            False, fallback, output=None, error=str(exc), metadata=decision.metadata
        )

    if decision.status == DISAGREE:
        record_disagreement(
            doc,
            decision,
            artifact_sha256=artifact_sha256,
            ingestion_generation=ingestion_generation,
        )
        return DualLoopOutcome(
            True, decision, 0, "errors", metadata=decision.metadata
        )

    member = getattr(decision.metadata.parsed, "member", None)
    inserted = ocr.insert_transactions(
        doc.doc_id,
        doc.year,
        member,
        rows,
        db_path=doc.resolved_db_path,
        raw_count=len(rows),
        artifact_sha256=artifact_sha256,
        ingestion_generation=ingestion_generation,
        parser_version=DUAL_PARSER_VERSION,
        engine_model=DUAL_ENGINES_ATTEMPTED,
    )
    if inserted > 0:
        progress_key = "completed"
    elif not rows:
        progress_key = "no_txs"
    else:
        progress_key = "errors"
    return DualLoopOutcome(
        True,
        decision,
        inserted,
        progress_key,
        member=member,
        metadata=decision.metadata,
    )


def record_disagreement(
    doc: DualReadDoc,
    decision: DualDecision,
    *,
    artifact_sha256: str | None,
    ingestion_generation: str | None,
) -> None:
    """Land a DISAGREE in the existing manual-review inventory.

    Mirrors ``_record_failed_ocr_attempt``: retire any gemini_ocr rows bound
    to this exact artifact/generation (both readings are now suspect), then
    record a parse run with status ``invalid`` — the terminal reviewed/no-rows
    state the generation-acceptance predicates already recognize — carrying
    both readings as JSON in ``error_message``.
    """
    import duckdb

    ocr = _ocr_zero_rows()
    conn = duckdb.connect(doc.resolved_db_path)
    try:
        conn.execute("BEGIN TRANSACTION")
        try:
            if artifact_sha256 and ingestion_generation:
                predicate = (
                    "source = 'gemini_ocr' AND doc_id = ? "
                    "AND (chamber = 'House' OR chamber IS NULL) "
                    "AND ingestion_generation = ? AND artifact_sha256 = ?"
                )
                parameters = [
                    str(doc.doc_id),
                    ingestion_generation,
                    artifact_sha256,
                ]
                ocr._archive_rows(conn, "transactions", predicate, parameters)
                conn.execute(
                    """
                    DELETE FROM transactions
                    WHERE source = 'gemini_ocr' AND doc_id = ?
                      AND (chamber = 'House' OR chamber IS NULL)
                      AND ingestion_generation = ? AND artifact_sha256 = ?
                    """,
                    parameters,
                )
            ocr.record_parse_run(
                conn,
                doc.doc_id,
                doc.year,
                "invalid",
                decision.raw_row_count,
                0,
                decision.notes_json
                or decision.diff_summary
                or "dual-read disagreement",
                parser_version=DUAL_PARSER_VERSION,
                artifact_sha256=artifact_sha256,
                ingestion_generation=ingestion_generation,
                engines_attempted=DUAL_ENGINES_ATTEMPTED,
            )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
    finally:
        conn.close()


def dual_read_enabled(env: Mapping[str, str] | None = None) -> bool:
    """Opt-in flag for the dual-read pipeline (default: off)."""
    source = os.environ if env is None else env
    return str(source.get(DUAL_READ_ENV) or "").strip() == "1"
