"""Dual vision read: engine B client, agreement orchestration, and persistence.

Engine A is the existing Gemini path (``scripts.gemini_ocr_common``, litellm
routing). Engine B is any OpenAI-compatible chat-completions vision endpoint —
built for the operator's forthcoming vLLM server (Qwen2.5-VL-class) — configured
entirely through the environment:

* ``VISION_B_BASE_URL``   — e.g. ``http://gpu-host:8000/v1`` (required)
* ``VISION_B_MODEL``      — served model id (required)
* ``VISION_B_API_KEY``    — optional bearer token
* ``VISION_B_TIMEOUT``    — optional seconds; when unset the read timeout
                            auto-scales 120s + 60s/page (1 page = 180s)

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
* ``UNRESOLVED`` — engine B unavailable (unconfigured, transport failure,
  timeout, HTTP error, malformed envelope, quota): the doc goes to the
  retry/review state (an ``error`` parse run that keeps it on the OCR work
  list) and NOTHING persists from either side. "Both readers must agree"
  means B absent is unresolved, never A-only agreement. The only A-only
  persistence allowed anywhere is the explicit legacy Gemini-only mode
  (``OCR_ENGINE`` unset/``gemini``). Quota-shaped B failures additionally
  stop the caller's batch.
* ``A_ERROR`` — engine A itself failed: the caller's existing failed-attempt
  handling records it.

An engine-B response that arrives but violates the extraction schema is NOT
treated as "engine absent" (that would silently reintroduce single-model
hallucination risk): it is fail-closed into the DISAGREE review inventory.

Engine-B client failures are classified (FP review 2026-10-18 finding 7):

* ``TransientVisionBError`` — 5xx, timeout/transport, malformed envelope,
  and truncated completions (``finish_reason="length"``, rejected as
  ``TruncatedCompletion``): retriable; ``scripts/glm_stage`` bounds the
  retries (2, jittered backoff, cached page renders).
* ``VisionBQuotaError`` — HTTP 429 / Z.AI error code 1302 / quota wording:
  never retried per-doc; callers stop the batch.
* ``VisionBAuthError`` (401/403) and other ``VisionBPermanentError`` (400…)
  — no retry, alert.
* ``VisionBConfigError`` — missing/invalid configuration: alert, no retry.
Non-200 response bodies are retained (bounded to 300 chars) inside the
exception so error classification survives logging.

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
UNRESOLVED = "UNRESOLVED"
A_ERROR = "A_ERROR"
DUAL_READ_ENV = "DUAL_READ"

VISION_B_BASE_URL_ENV = "VISION_B_BASE_URL"
VISION_B_MODEL_ENV = "VISION_B_MODEL"
VISION_B_API_KEY_ENV = "VISION_B_API_KEY"
VISION_B_TIMEOUT_ENV = "VISION_B_TIMEOUT"
DEFAULT_VISION_B_TIMEOUT = 180.0  # legacy flat budget == the 1-page auto budget
PAGE_RENDER_DPI = 200
SNIPPET_LIMIT = 300  # bounded non-200/partial body retained in exceptions

# Output-budget sizing (interface audit 2026-10-18, verified against
# docs.z.ai/api-reference/llm/chat-completion: max_tokens min 1 / max 131072
# for the vision request schema; largest real filing tonight was 68 rows,
# ~30-60 output tokens per row; glm-5.3-flash reasoning is always on and
# reasoning tokens draw on the same output budget, so the floor keeps 2x
# headroom over the largest observed answer rather than the bare 8192).
MAX_TOKENS_FLOOR = 16384  # flat budget for small docs (owner bound: >= 8192)
MAX_TOKENS_BASE = 4096  # MEMBER/PAGES header + reasoning headroom
MAX_TOKENS_PER_PAGE = 1500  # ~20 rows/page x ~60 tokens/row + PAGE headers
ZAI_MAX_TOKENS_CEILING = 131072  # documented request schema maximum

# Read-timeout budget (interface audit 2026-10-18): tonight's flat 180s was
# exhausted by heavy multi-page docs (long always-on reasoning before the
# first body byte — requests' read timeout is an idle timeout). The 1-page
# budget equals the old flat 180s; heavier docs scale.
TIMEOUT_BASE_SECONDS = 120.0
TIMEOUT_PER_PAGE_SECONDS = 60.0

# Documented finish_reason values (docs.z.ai chat-completion response
# schema): stop | tool_calls | length | sensitive |
# model_context_window_exceeded | network_error. Only a complete "stop"
# answer is a reading; everything else is fail-closed.
_FINISH_REASON_OK = ("", "stop")
_FINISH_REASON_PERMANENT = (
    "model_context_window_exceeded",
    "sensitive",
    "content_filter",
)

ReasonSchemaInvalid = "visionb_schema_invalid"
ReasonRowMismatch = "row_mismatch"


class VisionBUnavailable(Exception):
    """Engine B could not produce a usable reading.

    Base class for every engine-B client failure. On the dual path an absent
    engine B is UNRESOLVED — the doc is retried/reviewed, never persisted as
    an A-only agreement.
    """


class TransientVisionBError(VisionBUnavailable):
    """Retriable engine-B failure: 5xx, timeout/transport, malformed
    envelope, or a truncated completion. ``scripts/glm_stage`` bounds the
    retries (2, jittered backoff, cached page renders) before recording an
    error; a later run may safely retry the document."""


class TruncatedCompletion(TransientVisionBError):
    """The endpoint returned ``finish_reason="length"``: the completion was
    cut off mid-answer. Accepted payloads must be complete — this is a
    rejection, not a reading."""


class VisionBQuotaError(VisionBUnavailable):
    """Quota/rate-limit exhausted (HTTP 429, Z.AI error code 1302, quota
    wording). Never retried per-doc; callers stop the batch."""


class VisionBPermanentError(VisionBUnavailable):
    """Non-retriable request defect (HTTP 400/404/413/422...). No retry."""


class VisionBAuthError(VisionBPermanentError):
    """Credentials rejected (HTTP 401/403). No retry; alert the operator."""


class VisionBConfigError(VisionBUnavailable):
    """Missing or invalid engine-B configuration. No retry; alert."""


def classify_visionb_error(exc: BaseException) -> str:
    """One-word machine-readable kind for staging records and alerts."""
    if isinstance(exc, VisionBQuotaError):
        return "quota"
    if isinstance(exc, VisionBAuthError):
        return "auth"
    if isinstance(exc, VisionBConfigError):
        return "config"
    if isinstance(exc, TruncatedCompletion):
        return "truncated"
    if isinstance(exc, TransientVisionBError):
        return "transient"
    if isinstance(exc, VisionBPermanentError):
        return "permanent"
    return "parse"


def _snippet(text: str, limit: int = SNIPPET_LIMIT) -> str:
    """Collapse whitespace and bound an error body so it stays loggable."""
    collapsed = " ".join(str(text or "").split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 3] + "..."


def request_max_tokens(page_count: int | None) -> int:
    """Explicit output budget for one document read (interface audit item 1).

    The client used to omit ``max_tokens`` and ride the server default; the
    vision request schema documents no default (only ``example: 1024``, which
    WOULD truncate a 68-row filing), so the budget is now always sent:
    ``max(16384, 4096 + 1500 x pages)`` capped at the documented 131072
    ceiling. 1500/page covers ~20 rows x ~60 tokens plus PAGE headers — far
    above anything a PTR page holds — and the floor leaves headroom for
    always-on reasoning tokens, which share the output budget.
    """
    pages = max(int(page_count or 1), 1)
    budget = MAX_TOKENS_BASE + MAX_TOKENS_PER_PAGE * pages
    return min(max(budget, MAX_TOKENS_FLOOR), ZAI_MAX_TOKENS_CEILING)


def request_timeout(
    page_count: int | None, configured: float | None = None
) -> float:
    """Per-page-count read-timeout budget (interface audit item 4).

    ``120s + 60s/page``: the 1-page budget equals the old flat 180s, and
    heavy multi-page docs (which exhausted 180s tonight mid-generation) get
    proportionally more idle-read slack. An operator-pinned
    ``VISION_B_TIMEOUT`` (``VisionBConfig.timeout`` not None) always wins
    unchanged; the scaling applies only when the timeout is on auto (None).
    """
    pages = max(int(page_count or 1), 1)
    computed = TIMEOUT_BASE_SECONDS + TIMEOUT_PER_PAGE_SECONDS * pages
    return float(computed) if configured is None else float(configured)


@dataclass(frozen=True)
class VisionBConfig:
    """Connection settings for the OpenAI-compatible vision endpoint.

    ``timeout=None`` means AUTO: the client budgets ``120s + 60s/page`` per
    read (``request_timeout``). An explicit value (``VISION_B_TIMEOUT``)
    pins the budget exactly as given.
    """

    base_url: str
    model: str
    api_key: str | None = None
    timeout: float | None = None

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> VisionBConfig | None:
        """Return None when engine B is unconfigured (not an error)."""
        source = os.environ if env is None else env
        base_url = str(source.get(VISION_B_BASE_URL_ENV) or "").strip().rstrip("/")
        model = str(source.get(VISION_B_MODEL_ENV) or "").strip()
        if not base_url or not model:
            return None
        timeout: float | None = None  # auto: per-page budget at request time
        raw_timeout = str(source.get(VISION_B_TIMEOUT_ENV) or "").strip()
        if raw_timeout:
            try:
                timeout = float(raw_timeout)
            except ValueError as exc:
                raise VisionBConfigError(
                    f"invalid {VISION_B_TIMEOUT_ENV}: {raw_timeout!r}"
                ) from exc
            if timeout <= 0:
                raise VisionBConfigError(
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
    in the documented Z.AI vision shape (content array of ``image_url`` parts
    in page order, text prompt last) with an explicit page-scaled
    ``max_tokens`` budget and a ``120s + 60s/page`` read-timeout (auto unless
    ``VISION_B_TIMEOUT`` pins it). Failures
    raise a classified ``VisionBUnavailable`` subclass: transport/timeout and
    5xx are ``TransientVisionBError``; 429/1302/quota is
    ``VisionBQuotaError``; 401/403 is ``VisionBAuthError``; other 4xx is
    ``VisionBPermanentError``; non-``stop`` finish reasons are rejected
    (``length`` as ``TruncatedCompletion``, ``sensitive`` /
    ``model_context_window_exceeded`` as permanent, others transient).
    Non-200 bodies travel (bounded to
    ``SNIPPET_LIMIT`` chars) inside the exception so classification survives.
    """
    if not page_data_urls:
        raise VisionBUnavailable("visionb requires at least one page image")
    extraction_prompt = (
        prompt if prompt is not None else _gemini_common().PROMPT
    )
    payload = {
        "model": config.model,
        "temperature": 0,
        # Explicit per-document output budget (interface audit item 1): the
        # documented vision-schema default is unspecified, and an
        # finish_reason="length" truncation is a rejected reading — size for
        # the heaviest real filing instead of riding an undocumented default.
        "max_tokens": request_max_tokens(len(page_data_urls)),
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
    # One image per page: the page count IS the content-array length, so the
    # read-timeout budget scales with the document (audit item 4). Pages stay
    # in order in the documented vision shape (content array of image_url
    # parts, text last) — Z.AI documents multi-image via repeated image_url
    # blocks, capped at 5MB and 6000x6000 pixels per image (150/300-DPI
    # letter renders are far inside both).
    timeout = request_timeout(len(page_data_urls), configured=config.timeout)
    try:
        response = requests.post(
            url, json=payload, headers=headers, timeout=timeout
        )
    except requests.RequestException as exc:
        raise TransientVisionBError(
            f"visionb transport failure: {exc}"
        ) from exc
    if response.status_code != 200:
        raise _classify_http_error(
            response.status_code, _snippet(_response_body(response))
        )
    try:
        body = response.json()
        choice = body["choices"][0]
        text = choice["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise TransientVisionBError(
            f"visionb malformed response envelope: {exc}"
        ) from exc
    if not isinstance(text, str) or not text.strip():
        raise TransientVisionBError("visionb empty completion")
    finish_reason = (
        choice.get("finish_reason") if isinstance(choice, dict) else None
    )
    rejection = _finish_reason_rejection(finish_reason, text)
    if rejection is not None:
        raise rejection
    return text


def _finish_reason_rejection(finish_reason: object, text: str):
    """Reject any completion that is not a complete ``stop`` answer.

    Z.AI documents finish_reason stop | tool_calls | length | sensitive |
    model_context_window_exceeded | network_error. A non-stop reason with a
    non-empty body is a PARTIAL answer — accepting it would let a truncated
    row set masquerade as a reading (audit item 5/6, fail-closed).
    """
    reason = str(finish_reason or "").strip().lower()
    if reason in _FINISH_REASON_OK:
        return None
    if reason == "length":
        return TruncatedCompletion(
            "visionb truncated completion (finish_reason=length; rejecting "
            f"incomplete output): {_snippet(text)}"
        )
    if reason in _FINISH_REASON_PERMANENT:
        return VisionBPermanentError(
            f"visionb rejected completion (finish_reason={reason}): "
            f"{_snippet(text)}"
        )
    return TransientVisionBError(
        f"visionb abnormal completion (finish_reason={reason or 'missing'}): "
        f"{_snippet(text)}"
    )


def _response_body(response) -> str:
    try:
        return response.text
    except Exception:  # body unreadable: keep the classification, drop text
        return ""


def _classify_http_error(status_code: int, snippet: str) -> VisionBUnavailable:
    """Map an HTTP failure onto the transient/stop/alert classification.

    Against docs.z.ai/api-reference/api-code (verified 2026-10-18):

    * 1302 (429, "Rate limit reached for requests"), 1308/1310/1313/1316-1321
      (429, usage/fair-use limits) and 1113 (429, "Insufficient balance or no
      resource package") are quota — HTTP 429 or the body wording classifies
      there regardless of status.
    * 1305 (429, "The service may be temporarily overloaded, please try again
      later") is TRANSIENT, not quota: retrying per-doc is correct; stopping
      the whole batch on an overload blip would stall the drain.
    * 1309/1311/1314 (429, "package has expired" / plan does not include the
      model) never self-heal — classified as auth-shaped fatal defects so
      callers ALERT and stop instead of silently quota-stopping every night.
    * 1000-1005 (401) and 1220 (403) are credential defects; 1211/1261/1301
      (400) are request defects; 500/1200/1230/1234 (5xx) are transient.
    """
    lowered = snippet.lower()
    if "1305" in lowered or "temporarily overloaded" in lowered:
        return TransientVisionBError(
            f"visionb transient overload (http {status_code}): {snippet}"
        )
    if (
        status_code == 429
        and (
            "package has expired" in lowered
            or "does not yet include access" in lowered
        )
    ):
        # Subscription defect: alert + stop the batch (same handling as auth).
        return VisionBAuthError(
            f"visionb subscription defect (http {status_code}): {snippet}"
        )
    quota_hit = status_code == 429 or "1302" in lowered or (
        "quota" in lowered
        or "rate limit" in lowered
        or "throttl" in lowered
        or "insufficient balance" in lowered
        or "arrears" in lowered
    )
    if quota_hit:
        return VisionBQuotaError(
            f"visionb quota/rate-limit exhausted (http {status_code}): {snippet}"
        )
    if status_code in (401, 403):
        return VisionBAuthError(
            f"visionb auth rejected (http {status_code}): {snippet}"
        )
    if 400 <= status_code < 500:
        return VisionBPermanentError(f"visionb http {status_code}: {snippet}")
    return TransientVisionBError(f"visionb http {status_code}: {snippet}")


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
        raise VisionBConfigError(
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
    """Gate decision plus everything a caller needs to persist or retry."""

    status: str  # AGREE | DISAGREE | UNRESOLVED | A_ERROR
    parser_version: str
    engines_attempted: str
    raw_row_count: int = 0
    diff_summary: str | None = None
    notes_json: str | None = None
    # Engine-A passthrough for the unchanged single-engine loop path (A_ERROR
    # only — UNRESOLVED never carries A's output, so no caller can persist an
    # A-only reading by accident):
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
    (engine A's reading) for AGREE only, and is empty for DISAGREE,
    UNRESOLVED, and A errors — nothing persists from either side there. This
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
        # Engine B unconfigured is a config defect, not an agreement: the doc
        # is UNRESOLVED (retry/review state). A's reading is deliberately not
        # carried on the decision so no caller can persist it as A-only.
        return (
            DualDecision(
                UNRESOLVED,
                gemini.GEMINI_PARSER_VERSION,
                gemini.MODEL,
                raw_row_count=raw_row_count,
                diff_summary=(
                    "visionb is not configured: set VISION_B_BASE_URL and "
                    "VISION_B_MODEL"
                ),
                metadata=metadata,
            ),
            [],
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
        # Engine B absent (transport failure, timeout, HTTP error, quota,
        # malformed envelope): UNRESOLVED — never an A-only fallback. The
        # reason rides along in diff_summary so callers can tell a
        # quota/rate-limit batch stop (is_quota_error) from a transient
        # outage or a config/auth defect needing an alert.
        return (
            DualDecision(
                UNRESOLVED,
                gemini.GEMINI_PARSER_VERSION,
                gemini.MODEL,
                raw_row_count=raw_row_count,
                diff_summary=str(exc),
                metadata=metadata,
            ),
            [],
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
    ``gemini+visionb`` engines; progress is ``completed`` only when the full
    validated row set committed (partial/rejected outcomes mark ``errors``).
    DISAGREE records the manual-review inventory run. UNRESOLVED (engine B
    absent — outage, misconfiguration, quota) persists NOTHING from either
    side: quota-shaped failures return unhandled so the caller stops the
    batch, every other absence lands an ``error`` parse run that keeps the
    doc on the OCR work list for the next run. A_ERROR is returned unhandled
    with engine A's error so the caller's existing single-engine failure
    bookkeeping records it.
    """
    decision, rows = dual_read(
        doc,
        refresh=refresh,
        env=env,
        read_a=read_a,
        read_b=read_b,
        page_images=page_images,
    )
    if decision.status == UNRESOLVED:
        if _gemini_common().is_quota_error(decision.diff_summary or ""):
            # Engine-B quota stop must end the caller's batch cleanly and
            # persist nothing: continuing would either drain ungated or
            # hammer a dead quota for every remaining doc.
            return DualLoopOutcome(
                False, decision, 0, "errors", metadata=decision.metadata
            )
        artifact_sha256 = getattr(decision.metadata, "sha256", None)
        try:
            ingestion_generation = _ocr_zero_rows().resolve_ingestion_generation(
                doc.resolved_db_path, doc.doc_id, doc.year, artifact_sha256
            )
        except RuntimeError:
            ingestion_generation = None
        record_unresolved(
            doc,
            decision,
            artifact_sha256=artifact_sha256,
            ingestion_generation=ingestion_generation,
        )
        return DualLoopOutcome(
            True, decision, 0, "errors", metadata=decision.metadata
        )
    if decision.status == A_ERROR:
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
    report: dict = {}
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
        report=report,
    )
    outcome = ocr.insert_outcome_from_report(
        report, inserted=inserted, expected=len(rows)
    )
    # Never call a partial or rejected output complete: only a full committed
    # row set counts as completed progress, and a committed no_txs run is
    # reported as no_txs — never folded into "completed".
    if outcome.status == "no_txs":
        progress_key = "no_txs"
    elif outcome.complete:
        progress_key = "completed"
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


def record_unresolved(
    doc: DualReadDoc,
    decision: DualDecision,
    *,
    artifact_sha256: str | None,
    ingestion_generation: str | None,
) -> None:
    """Land an UNRESOLVED dual read in the retry state.

    Mirrors ``record_disagreement``: retire any gemini_ocr rows bound to this
    exact artifact/generation (a prior ungated reading is now suspect), then
    record an ``error`` parse run carrying the engine-B absence reason — the
    non-terminal state that keeps the doc on the OCR work list so the next
    run retries it once engine B is back.
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
                "error",
                decision.raw_row_count,
                0,
                f"dual-read unresolved: {decision.diff_summary}"
                or "dual-read unresolved",
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
