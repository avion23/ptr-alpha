#!/usr/bin/env python3
"""Staged GLM OCR ingestion for the backlog drain.

GLM-only backlog candidates are STAGED: written as one JSON record per document under
``data/staging/glm-backlog/<year>/<doc_id>.json`` and explicitly excluded
from canonical/backtests until an owner-approved pixel audit passes
(``audit_sample.py``). Audit approval is MANDATORY at the promotion
boundary: ``promote`` refuses any record without ``audit: "pass"`` unless
``--force-unaudited`` is passed (loud warning). This is also the only route
``OCR_ENGINE=visionb`` readings take to canonical — the nightly drain stages
and never inserts.

Subcommands:

* ``stage   --year Y [--limit N] [--check-canonical] [--from-manifest PATH]``
    Rasterize unread PDFs (pdftoppm -r 150 -png), read them with the GLM
    vision endpoint through the existing ``analyzer.vision_engines`` client
    (shared PROMPT, never forked), parse with ``parse_gemini_output``, and
    write staging records. Transient endpoint failures (5xx, timeout,
    transport, truncation, Google rate-shaped 429s) are retried in-process
    with cached page renders; Google 429s wait 60–90s between attempts.
    Five consecutive rate-limited Google docs with no success degrade to a
    clean batch stop so a dead daily quota cannot fill the manifest with
    errors. Other quota exhaustion stops cleanly too; auth/config defects
    alert and stop the batch with exit 1; per-doc failures never kill it.

    Population (FP review finding 8): ``--from-manifest PATH`` drains the
    AUTHORITATIVE unresolved-work selection exported by
    ``ocr_zero_rows.py --export-work-manifest`` (``get_ocr_work_items``)
    instead of scanning PDF directories — text-born docs cannot leak in and
    promoted docs cannot re-admit. Entries without a local PDF are skipped
    with a note; entries whose live record is an error record are retried
    (the failed record is archived, never overwritten).
* ``promote --year Y [--doc-id ID ...] [--force-unaudited]``
    Insert AUDITED staged rows into canonical through the EXISTING
    ``ocr_zero_rows.insert_transactions`` path (parser_version
    ``v1-glm53-vision``, engines ``visionb``), mirror ``main()``'s parse-run
    bookkeeping, and archive the staging JSON (promotion = consumed) only
    when the FULL staged row set is confirmed committed — a partial or
    rejected insert keeps the live record and stamps the rejection details
    on it. A promoted doc lands a success/no_txs ``v1-glm53-vision`` parse
    run, which is exactly what removes it from ``get_ocr_work_items`` —
    the disposition record IS the queue exclusion.
* ``status``
    Per-year staged / promoted / error counts (staged counts need no DB).

Canonical (``data/congress.duckdb``) is only opened by ``promote`` (which
needs the write lock) and by ``stage --check-canonical`` (read-only, opt-in
because a refresh holds the DB).

Provenance and verdict binding (FP review finding 6): each record stores
``render_dpi`` and per-row page attribution (``row_pages``, aligned with
``rows``); ``raw_output`` is retained even for error records (invalid
responses are evidence); the sha256/page-count/render all read ONE
snapshot of the artifact bytes, not the live file three times. Re-staging
a doc that already has a live record archives the previous record as
``<year>/attempts/<doc_id>.attempt-<n>.json`` (n = prior attempt count,
newest live record keeps the canonical name. Superseded attempts are never
automatically pruned: audit and promotion references can be published
concurrently, so a check-then-delete/restore sequence cannot protect their
original bytes from a crash. Storage grows with the number and size of
restages; any future garbage collection must be explicit and operator-approved
after checking audit, promotion, and receipt dependencies. The attempts
subdirectory keeps history out of record-counting globs in doctor.py and
audit_sample.py. Each record carries a canonical hash of its extracted rows;
verdict binding additionally checks the audit workflow's stable staged-record
content hash and the record identity
(``pdf_sha256 + prompt_sha256 + parsed_at_utc``). Metadata-only legacy
verdicts remain historical but cannot authorize promotion.

Concurrency (FP review finding 11): ``--workers`` is capped at
MAX_WORKERS (2 — the Z.AI subscription allows two concurrent vision
streams; larger values raise ValueError instead of being clamped).
Per-document publication locks serialize live/attempt changes across direct
stage and promote invocations; they are held only for publication and
promotion binding, never during vision inference. The cron wrapper retains its
separate single-instance flock.
"""

from __future__ import annotations

import argparse
import base64
import datetime
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import NotRequired, TypedDict, cast

REPO_ROOT = Path(__file__).resolve().parents[1]
if __name__ == "__main__":
    for _path in (str(REPO_ROOT), str(REPO_ROOT / "src")):
        if _path not in sys.path:
            sys.path.insert(0, _path)

from analyzer.ocr.common import (  # noqa: E402
    AMOUNT_MIDPOINTS,
    PROMPT_SHA256,
    GeminiOutputError,  # noqa: F401  (re-exported for callers/tests)
    is_quota_error,
    parse_gemini_output,
    pdf_page_count,
    pdf_sha256,  # noqa: F401 (re-exported for callers/tests)
)
from analyzer.vision_engines import (  # noqa: E402
    GoogleVisionBRateLimitError,
    GoogleVisionBServiceUnavailableError,
    TransientVisionBError,
    VisionBConfig,
    classify_visionb_error,
    is_google_api_host,
    read_document_with_vision_b,
)
from scripts.audit_sample import (  # noqa: E402
    REPORT_STATE_NAME as AUDIT_REPORT_STATE_NAME,
    REPORT_STATE_SCHEMA as AUDIT_REPORT_STATE_SCHEMA,
    audit_history_sha256,
    audit_operation_lock,
    load_audit_results as load_audit_results,
    sample_lot_file_sha256,
    staged_record_sha256 as audit_staged_record_sha256,
)

GLM_PARSER_VERSION = "v1-glm53-vision"
GLM_ENGINES_ATTEMPTED = "visionb"
AUDIT_RESULTS_NAME = "audit-results.json"
PROMOTED_JOURNAL_NAME = "promoted.jsonl"
RENDER_DPI = 150
MAX_READ_RETRIES = 2  # bounded in-process retries for transient failures
BACKOFF_BASE_SECONDS = 2.0  # jittered exponential backoff base
GOOGLE_RETRY_MIN_SECONDS = 60.0
GOOGLE_RETRY_MAX_SECONDS = 90.0
GOOGLE_429_CONSECUTIVE_STOP_THRESHOLD = 5
# Match the 429 stop threshold, but track a distinct saturation streak.
GOOGLE_503_CONSECUTIVE_STOP_THRESHOLD = 5
# Five failed docs tolerate short bursts while stopping before likely daily
# exhaustion can write transient error records across the rest of a manifest.
MAX_WORKERS = 2  # Z.AI subscription allows 2 concurrent vision streams
CLI_COOLDOWN = 3.0  # seconds between endpoint requests (mirrors ocr_zero_rows)
WORK_MANIFEST_SCHEMA = "glm-work-manifest-v1"
# Asset-normalization review flag (interface audit item 3): GLM sometimes
# damages asset text during normalization — drops repeated issuer wording
# ("Energy Transfer Equity, L.P. Energy Transfer Equity, L.P." extracted
# once) or mangles a ticker ("PARNTERS") — observed at ~3-4% doc-level with
# transaction-critical fields 100% clean. The flag is the MITIGATION, not a
# fix: flagged staging records get ``asset_review: true`` so the audit
# sampler oversamples them.
ASSET_TICKER_PATTERN = re.compile(r"\([A-Za-z0-9][A-Za-z0-9.\-]{0,15}\)")
ASSET_TICKER_CONTEXT_SHARE = 0.8  # >80% of the OTHER rows carry a ticker
# The record identity a verdict is bound to (finding 6): if any of these
# no longer match the live record, the audited attempt is not the live one.
AUDIT_IDENTITY_FIELDS = ("pdf_sha256", "prompt_sha256", "parsed_at_utc")


def _ocr_ingestion():
    # Single import boundary for the production OCR ownership migration.
    from analyzer.ocr import ingestion

    return ingestion


class StagingRecord(TypedDict):
    """JSON evidence, retained verbatim so historical content hashes stay valid."""

    doc_id: str
    year: int
    pdf_sha256: str | None
    prompt_sha256: str
    parsed_at_utc: str
    rows: list[list]
    member: str | None
    parse_status: str
    model: str | None
    parser_version: str
    pages: int | None
    raw_output: str | None
    error: str | None
    audit: str | None
    archive_year: NotRequired[int]
    artifact_sha256: NotRequired[str | None]  # legacy alias, never overrides pdf_sha256
    ingestion_generation: NotRequired[str | None]
    render_dpi: NotRequired[int]
    row_pages: NotRequired[list[int | None]]
    staged_rows_sha256: NotRequired[str | None]
    error_kind: NotRequired[str | None]
    asset_review: NotRequired[bool]
    asset_review_reasons: NotRequired[list[str]]
    promotion: NotRequired[dict]


def decode_staging_record(raw: str | bytes) -> StagingRecord:
    record = json.loads(raw)
    if not isinstance(record, dict):
        raise ValueError("staging record must be a JSON object")

    # TypedDict is only a static description; every live JSON record is a
    # trust boundary. These fields are shared by the original and current
    # staging writers. Additive provenance/promotion fields stay optional so
    # valid legacy and failed attempts remain readable without fabricated
    # defaults.
    required = (
        "doc_id", "year", "pdf_sha256", "model", "prompt_sha256",
        "parsed_at_utc", "rows", "member", "parse_status", "parser_version",
        "pages", "raw_output", "error", "audit",
    )
    missing = [field for field in required if field not in record]
    if missing:
        raise ValueError(
            "staging record is missing required fields: " + ", ".join(missing)
        )

    doc_id = record["doc_id"]
    if (
        not isinstance(doc_id, str)
        or not doc_id.strip()
        or doc_id in (".", "..")
        or Path(doc_id).name != doc_id
    ):
        raise ValueError("staging record doc_id must be a non-empty filename")
    year = record["year"]
    if type(year) is not int or not 1000 <= year <= 9999:
        raise ValueError("staging record year must be an integer from 1000 to 9999")
    status = record["parse_status"]
    if status not in ("success", "no_txs", "error"):
        raise ValueError(
            "staging record parse_status must be success, no_txs, or error"
        )

    for field in ("prompt_sha256", "parsed_at_utc", "parser_version"):
        if not isinstance(record[field], str) or not record[field].strip():
            raise ValueError(f"staging record {field} must be a non-empty string")
    for field in ("pdf_sha256", "model", "member", "raw_output", "error"):
        value = record[field]
        if value is not None and not isinstance(value, str):
            raise ValueError(f"staging record {field} must be a string or null")
    if record["pdf_sha256"] is not None and not record["pdf_sha256"].strip():
        raise ValueError("staging record pdf_sha256 cannot be empty")
    pages = record["pages"]
    if pages is not None and (type(pages) is not int or pages <= 0):
        raise ValueError("staging record pages must be a positive integer or null")
    if record["audit"] not in (None, "pass", "fail"):
        raise ValueError("staging record audit must be pass, fail, or null")

    rows = record["rows"]
    if not isinstance(rows, list) or any(
        not isinstance(row, list)
        or len(row) != 5
        or any(
            not isinstance(row[index], str) or not row[index].strip()
            for index in (0, 1, 2, 4)
        )
        or (row[3] is not None and not isinstance(row[3], str))
        for row in rows
    ):
        raise ValueError("staging record rows must be a list of five-field string rows")
    if status in ("success", "no_txs"):
        if bool(rows) != (status == "success"):
            raise ValueError(f"{status} staging record has inconsistent rows")
        if (
            not isinstance(record["pdf_sha256"], str)
            or not record["pdf_sha256"].strip()
        ):
            raise ValueError(f"{status} staging record must have pdf_sha256")
        if not isinstance(record["member"], str) or not record["member"].strip():
            raise ValueError(f"{status} staging record must have member")
        if pages is None:
            raise ValueError(f"{status} staging record must have a page count")
        if record["error"] is not None:
            raise ValueError(f"{status} staging record cannot carry an error")
    elif rows or not isinstance(record["error"], str) or not record["error"].strip():
        raise ValueError(
            "error staging record must have no rows and a non-empty error"
        )

    if "row_pages" in record:
        row_pages = record["row_pages"]
        if (
            not isinstance(row_pages, list)
            or len(row_pages) != len(rows)
            or any(
                page is not None and (type(page) is not int or page <= 0)
                for page in row_pages
            )
        ):
            raise ValueError("staging record row_pages must align with rows")
    for field in ("artifact_sha256", "ingestion_generation", "error_kind"):
        if (
            field in record
            and record[field] is not None
            and not isinstance(record[field], str)
        ):
            raise ValueError(f"staging record {field} must be a string or null")
    if "archive_year" in record and record["archive_year"] is not None:
        archive_year = record["archive_year"]
        if type(archive_year) is not int or not 1000 <= archive_year <= 9999:
            raise ValueError("staging record archive_year must be an integer year or null")
    if "asset_review" in record and not isinstance(record["asset_review"], bool):
        raise ValueError("staging record asset_review must be boolean")
    if "staged_rows_sha256" in record:
        row_hash = record["staged_rows_sha256"]
        if row_hash is not None and not isinstance(row_hash, str):
            raise ValueError("staging record staged_rows_sha256 must be a string or null")
        if row_hash is not None and row_hash != staged_rows_sha256(rows):
            raise ValueError("staging record rows do not match staged_rows_sha256")

    # Do not inject legacy defaults into content-bound evidence.
    return cast(StagingRecord, record)


@dataclass(frozen=True)
class ArtifactRequest:
    """Only the identity constraints understood by the artifact resolver."""

    doc_id: str
    year: int
    archive_year: int | None = None
    artifact_sha256: str | None = None
    ingestion_generation: str | None = None

    @classmethod
    def from_manifest(cls, item: dict) -> ArtifactRequest:
        artifact_hash = item.get("artifact_sha256")
        pdf_hash = item.get("pdf_sha256")
        if artifact_hash is not None and pdf_hash is not None and artifact_hash != pdf_hash:
            raise RuntimeError("conflicting artifact_sha256 and pdf_sha256")
        return cls(
            doc_id=str(item["doc_id"]), year=int(item["year"]),
            archive_year=item.get("archive_year"),
            artifact_sha256=artifact_hash if artifact_hash is not None else pdf_hash,
            ingestion_generation=item.get("ingestion_generation"),
        )

    @classmethod
    def from_record(cls, record: StagingRecord) -> ArtifactRequest:
        legacy_hash = record.get("artifact_sha256")
        if legacy_hash is not None and legacy_hash != record["pdf_sha256"]:
            raise RuntimeError("conflicting artifact_sha256 and pdf_sha256")
        return cls(
            doc_id=record["doc_id"], year=record["year"],
            archive_year=record.get("archive_year"),
            artifact_sha256=record["pdf_sha256"],
            ingestion_generation=record.get("ingestion_generation"),
        )

    def resolver_fields(self) -> dict:
        # Narrow adapter to the current OCR API; never forward record metadata.
        return {
            "doc_id": self.doc_id, "year": self.year,
            "archive_year": self.archive_year,
            "artifact_sha256": self.artifact_sha256,
            "ingestion_generation": self.ingestion_generation,
        }


def staging_root(data_dir: str | Path = "data") -> Path:
    return _ocr_ingestion().glm_staging_root(data_dir)


@contextmanager
def document_publication_lock(data_dir: str | Path, year: int, doc_id: str):
    """Compatibility alias for the OCR-owned reentrant document lock."""
    with _ocr_ingestion().staging_document_lock(data_dir, year, doc_id):
        yield


def default_db_path(data_dir: str | Path = "data") -> Path:
    return Path(data_dir) / "congress.duckdb"


def default_pdf_path(data_dir: str | Path, year: int, doc_id: str) -> Path:
    return Path(data_dir) / str(year) / "pdfs" / f"{doc_id}.pdf"


def utc_now_iso() -> str:
    return (
        datetime.datetime.now(datetime.timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def atomic_write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def staging_record_path(data_dir: str | Path, year: int, doc_id: str) -> Path:
    return _ocr_ingestion().glm_staging_record_path(data_dir, year, doc_id)


def attempts_dir(data_dir: str | Path, year: int) -> Path:
    """Archived (superseded) staging attempts for a year.

    A subdirectory rather than the year directory itself: doctor.py and
    audit_sample.py count ``<year>/*.json`` as LIVE records; archived
    attempts are historical evidence and must not inflate those counts.
    """
    return _ocr_ingestion().glm_staging_attempts_dir(data_dir, year)


def archived_attempts(data_dir: str | Path, year: int, doc_id: str) -> list[tuple[int, Path]]:
    """(attempt_number, path) for a doc's archived attempts, oldest first."""
    return _ocr_ingestion().archived_glm_staging_attempts(data_dir, year, doc_id)


def staged_rows_sha256(rows) -> str:
    """Hash extracted rows using canonical JSON serialization."""
    canonical = json.dumps(
        rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def archive_current_attempt(
    data_dir: str | Path, year: int, doc_id: str
) -> Path | None:
    """Versioned re-stage (finding 6): archive the live record, never overwrite.

    Moves the existing ``<doc_id>.json`` to
    ``attempts/<doc_id>.attempt-<n>.json`` so the newly written record keeps
    the canonical name and the superseded reading remains inspectable
    evidence. n is one greater than the largest existing archive number. A
    per-document lock and exclusive hard-link publication prevent a racing
    writer from replacing an archive or moving a different live attempt.
    Attempts are never automatically pruned: reference publication can race
    with a check-and-delete, and a failed restore could destroy evidence.
    Returns the archive path, or None when no live record existed.
    """
    return _ocr_ingestion().archive_glm_staging_attempt(data_dir, year, doc_id)


def write_staging_record(data_dir: str | Path, record: StagingRecord) -> Path:
    return _ocr_ingestion().replace_glm_staging_record(data_dir, record)


def publish_staging_record(
    data_dir: str | Path, record: StagingRecord
) -> tuple[Path, Path | None]:
    """Publish a producer attempt and archive any existing live record."""
    return _ocr_ingestion().publish_glm_staging_record(data_dir, record)


def load_staging_record(data_dir: str | Path, year: int, doc_id: str) -> StagingRecord | None:
    path = staging_record_path(data_dir, year, doc_id)
    if not path.exists():
        return None
    return decode_staging_record(path.read_text(encoding="utf-8"))


def find_staging_record(data_dir: str | Path, doc_id: str) -> tuple[int, StagingRecord] | None:
    """Locate a staging record across year directories (audit tooling)."""
    root = staging_root(data_dir)
    for year_dir in sorted(root.glob("[0-9]" * 4)):
        path = year_dir / f"{doc_id}.json"
        if path.exists():
            return int(year_dir.name), decode_staging_record(path.read_text(encoding="utf-8"))
    return None


def iter_staging_records(data_dir: str | Path) -> list[tuple[int, StagingRecord]]:
    """Read live ``<year>/*.json`` records, ordered (year, doc_id).

    Root-level audit metadata and ``<year>/attempts/`` history are outside
    this population; an absent root/year is an empty population. Every JSON
    file directly in a year directory is a live document record and must be
    readable before callers can use population counts or promotion gates.
    """
    records = []
    root = staging_root(data_dir)
    for year_dir in sorted(root.glob("[0-9]" * 4), key=lambda p: p.name):
        for path in sorted(year_dir.glob("*.json")):
            try:
                record = decode_staging_record(path.read_text(encoding="utf-8"))
                if (
                    record["year"] != int(year_dir.name)
                    or record["doc_id"] != path.stem
                ):
                    raise ValueError(
                        "staging record identity does not match its live path"
                    )
            except (OSError, ValueError) as exc:
                raise ValueError(f"unreadable staging record {path}: {exc}") from exc
            records.append((int(year_dir.name), record))
    return records


# --------------------------------------------------------------------------
# stage
# --------------------------------------------------------------------------


def rasterize_page_data_urls(pdf_path: str | Path, dpi: int = RENDER_DPI) -> list[str]:
    """Rasterize every page to a base64 PNG data URL with pdftoppm -r 150 -png."""
    with tempfile.TemporaryDirectory(prefix="glm_stage_render_") as directory:
        prefix = Path(directory) / "page"
        result = subprocess.run(
            [
                "pdftoppm",
                "-r",
                str(dpi),
                "-png",
                str(pdf_path),
                str(prefix),
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "pdftoppm failed: " + (result.stderr.strip() or f"exit {result.returncode}")
            )
        pages = []
        for png in Path(directory).glob("page-*.png"):
            suffix = png.stem.rsplit("-", 1)[-1]
            pages.append((int(suffix), png))
        pages.sort()
        if not pages:
            raise RuntimeError("pdftoppm produced no page images")
        return [
            "data:image/png;base64,"
            + base64.b64encode(png.read_bytes()).decode("ascii")
            for _, png in pages
        ]


def read_document(
    pdf_path: str | Path,
    config: VisionBConfig,
    *,
    read_fn=None,
    page_count: int | None = None,
    max_retries: int = MAX_READ_RETRIES,
    backoff_seconds: float = BACKOFF_BASE_SECONDS,
    sleep_fn=None,
    jitter_fn=None,
) -> str:
    """Render pages once and read the whole document with retries.

    Uses the shared PROMPT via ``read_document_with_vision_b`` (never
    forked). Transient engine failures (``TransientVisionBError``: 5xx,
    timeout/transport, malformed envelope, truncated completion) are
    retried up to ``max_retries`` times with jittered exponential backoff,
    reusing the SAME rendered page images. Google's rate-shaped 429s use a
    60–90s jittered delay for Google-host 429/503 responses; other transient
    failures keep the shorter backoff.
    Quota errors stop immediately; auth/config/permanent failures are raised
    untouched (no retry — the same request would fail again).

    ``read_fn(pdf_path, page_count)`` replaces the render+read pair (tests,
    alternate engines); it gets the same bounded retries.
    """
    sleep_fn = time.sleep if sleep_fn is None else sleep_fn
    jitter_fn = random.uniform if jitter_fn is None else jitter_fn
    renders: list[str] | None = None

    def _attempt() -> str:
        nonlocal renders
        if read_fn is not None:
            pages = (
                page_count
                if page_count is not None
                else pdf_page_count(pdf_path)
            )
            return read_fn(pdf_path, pages)
        if renders is None:
            renders = rasterize_page_data_urls(pdf_path)
        return read_document_with_vision_b(renders, config)

    for attempt in range(max_retries + 1):
        try:
            return _attempt()
        except TransientVisionBError as exc:
            if attempt >= max_retries:
                raise
            if isinstance(
                exc,
                (GoogleVisionBRateLimitError, GoogleVisionBServiceUnavailableError),
            ):
                delay = GOOGLE_RETRY_MIN_SECONDS + jitter_fn(
                    0.0,
                    GOOGLE_RETRY_MAX_SECONDS - GOOGLE_RETRY_MIN_SECONDS,
                )
                sleep_fn(
                    min(
                        max(delay, GOOGLE_RETRY_MIN_SECONDS),
                        GOOGLE_RETRY_MAX_SECONDS,
                    )
                )
            else:
                sleep_fn(backoff_seconds * (2 ** attempt) + jitter_fn(0.0, 1.0))
    raise RuntimeError("unreachable")  # pragma: no cover


def detect_asset_review_reasons(transactions: list[dict]) -> list[str]:
    """Cheap doc-level heuristics for likely GLM asset-text damage.

    Flags (does not reject) a staging record when, per the audit owner's
    contract:

    * a row's asset lacks a parenthesized ticker while >80% of the doc's
      OTHER rows have one (the "PARNTERS"/dropped-ticker failure mode), or
    * a row's asset text is a strict substring of another row's asset
      (the dropped-repeated-issuer-wording failure mode: the true cell is
      the doubled issuer name and GLM emitted it once — the doubled form
      often appears on a sibling row extracted correctly).

    Returns machine-readable reason strings; empty means not flagged. Both
    checks are cheap (O(rows^2) string ops, rows <= a few hundred) and
    deliberately conservative: a single-row doc can never trip the ticker
    rule, and a doc where NO row carries tickers (legit untickered assets)
    cannot either.
    """
    assets = [
        str(tx.get("asset") or "").strip()
        for tx in transactions
        if isinstance(tx, dict)
    ]
    count = len(assets)
    if count < 2:
        return []
    reasons: list[str] = []
    has_ticker = [bool(ASSET_TICKER_PATTERN.search(asset)) for asset in assets]
    tickered = sum(has_ticker)
    for index, asset in enumerate(assets):
        if has_ticker[index] or not asset:
            continue
        others = count - 1
        if tickered / others > ASSET_TICKER_CONTEXT_SHARE:
            reasons.append(f"missing_ticker_context:row{index + 1}")
    folded = [" ".join(asset.split()).casefold() for asset in assets]
    for i in range(count):
        if not folded[i]:
            continue
        for j in range(count):
            if i != j and folded[i] != folded[j] and folded[i] in folded[j]:
                reasons.append(f"asset_substring:row{i + 1}<row{j + 1}")
                break  # one witness per row keeps the reason list bounded
    return reasons


def build_staging_record(
    doc_id: str,
    year: int,
    pdf_path: str | Path,
    config: VisionBConfig | None,
    *,
    read_fn=None,
    artifact_identity: dict | None = None,
    data_dir: str | Path | None = None,
) -> StagingRecord:
    """Read one document and build its staging record. Never raises.

    Provenance (finding 6): the artifact bytes are snapshotted ONCE —
    ``pdf_sha256``, the page count, and the render all read the same bytes,
    so the record can never bind rows to a hash of a file the render never
    saw. ``render_dpi`` and per-row page attribution (``row_pages``,
    aligned with ``rows``) are recorded at stage time, and ``raw_output``
    is retained even for error records: an invalid response is evidence.

    Parse/validation failures become ``parse_status: "error"`` records with
    ``error_kind`` carrying the classified failure (quota/auth/config/
    truncated/transient/permanent/parse) so the batch loop can distinguish
    a per-doc defect from a batch-wide stop. The W3 correction to finding 6
    stands: historical "not configured" records were real config-missing
    (``model: null``) — an unconfigured endpoint is an error record, never
    a silent skip.

    Successful records additionally carry the additive
    ``asset_review``/``asset_review_reasons`` flag (interface audit item 3):
    a cheap post-parse heuristic for GLM's known asset-text normalization
    damage (dropped repeated issuer wording, mangled tickers). The flag
    never rejects a row; it marks the record so the audit sampler
    oversamples it.
    """
    record: StagingRecord = {
        "doc_id": str(doc_id),
        "year": int(year),
        "pdf_sha256": None,
        "model": getattr(config, "model", None),
        "prompt_sha256": PROMPT_SHA256,
        "parser_version": GLM_PARSER_VERSION,
        "parsed_at_utc": utc_now_iso(),
        "render_dpi": RENDER_DPI,
        "pages": None,
        "member": None,
        "rows": [],
        "staged_rows_sha256": None,
        "row_pages": [],
        "raw_output": None,
        "parse_status": "error",
        "error": None,
        "error_kind": None,
        "asset_review": False,
        "asset_review_reasons": [],
        "audit": None,
    }
    raw_output = None
    try:
        # One artifact snapshot feeds sha256, page count, and the render.
        ocr = _ocr_ingestion()

        if artifact_identity is not None and "pdf_bytes" in artifact_identity:
            pdf_bytes = artifact_identity["pdf_bytes"]
        else:
            pdf_bytes = ocr.read_work_artifact(data_dir if data_dir is not None else Path(pdf_path).parent, pdf_path)
        record["pdf_sha256"] = hashlib.sha256(pdf_bytes).hexdigest()
        if artifact_identity is not None:
            if record["pdf_sha256"] != artifact_identity["artifact_sha256"]:
                raise RuntimeError("artifact changed after manifest selection")
            record["archive_year"] = artifact_identity["archive_year"]
            record["ingestion_generation"] = artifact_identity["ingestion_generation"]
        with tempfile.TemporaryDirectory(prefix="glm_stage_snapshot_") as snapshot_dir:
            snapshot = Path(snapshot_dir) / f"{Path(pdf_path).stem}.pdf"
            snapshot.write_bytes(pdf_bytes)
            page_count = pdf_page_count(snapshot)
            record["pages"] = page_count
            if config is None and read_fn is None:
                raise RuntimeError("GLM vision endpoint is not configured")
            raw_output = read_document(
                snapshot, config, read_fn=read_fn, page_count=page_count
            )
            parsed = parse_gemini_output(raw_output, expected_page_count=page_count)
        review_reasons = detect_asset_review_reasons(parsed.transactions)
        record.update(
            member=parsed.member,
            rows=[
                [
                    tx["asset"],
                    tx["type"],
                    tx["date"],
                    tx["notif_date"],
                    tx["amount_letter"],
                ]
                for tx in parsed.transactions
            ],
            row_pages=[tx.get("page_number") for tx in parsed.transactions],
            raw_output=raw_output,
            parse_status="success" if parsed.transactions else "no_txs",
            error=None,
            error_kind=None,
            asset_review=bool(review_reasons),
            asset_review_reasons=review_reasons,
        )
    except Exception as exc:  # one bad doc must never kill the batch
        record["parse_status"] = "error"
        record["error"] = str(exc) or type(exc).__name__
        record["error_kind"] = classify_visionb_error(exc)
        if raw_output is not None:
            # The response text exists but failed the schema/transport gate:
            # retain it. Invalid responses are evidence, not noise.
            record["raw_output"] = raw_output
    record["staged_rows_sha256"] = staged_rows_sha256(record["rows"])
    return record


def canonical_parsed_doc_ids(db_path: str | Path) -> set[str]:
    """Doc ids with ANY successful/no_txs canonical parse run (best effort)."""
    import duckdb

    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        rows = conn.execute(
            """
            SELECT DISTINCT CAST(doc_id AS VARCHAR)
            FROM pdf_parse_runs
            WHERE status IN ('success', 'no_txs')
            """
        ).fetchall()
    finally:
        conn.close()
    return {row[0] for row in rows}


def canonical_terminal_artifacts(db_path: str | Path) -> set[tuple[str, str, str]]:
    """Current acquired PTR identities passing the existing terminal contract."""
    from analyzer.persistence.database import Database

    db = Database(db_path, read_only=True)
    try:
        complete = set()
        years = db.conn.execute("SELECT DISTINCT archive_year FROM house_archive_generations").fetchall()
        for (year,) in years:
            generation = db.get_latest_house_generation(year)
            unresolved = set(db.get_unresolved_house_doc_ids(year, generation))
            rows = db.conn.execute(
                """
                SELECT a.doc_id, a.artifact_sha256, a.generation_id
                FROM house_pdf_artifacts a
                JOIN house_generation_metadata m
                  USING (archive_year, generation_id, doc_id)
                WHERE a.archive_year = ? AND a.generation_id = ?
                  AND m.filing_type = 'P' AND a.artifact_sha256 IS NOT NULL
                """, [year, generation],
            ).fetchall()
            complete.update(tuple(row) for row in rows if row[0] not in unresolved)
        return complete
    finally:
        db.close()


def manifest_artifact(request: ArtifactRequest, data_dir: str | Path, db_path: str | Path | None = None):
    """Return a verified byte snapshot and identity; path is naming only.

    Consumers use identity['pdf_bytes']/['artifact_sha256'], never reopen the
    returned display path after verification.
    """
    import duckdb
    ocr = _ocr_ingestion()

    if not isinstance(data_dir, ocr.ApprovedDataRoot):
        with ocr.approved_data_root(data_dir) as root:
            return manifest_artifact(request, root, db_path)

    path = Path(db_path or default_db_path(data_dir))
    with duckdb.connect(str(path), read_only=True) if path.exists() else nullcontext(None) as conn:
        return ocr.local_work_artifact(conn, data_dir, request.resolver_fields())


def select_candidates(
    year: int,
    limit: int,
    data_dir: str | Path = "data",
    *,
    check_canonical: bool = False,
    db_path: str | Path | None = None,
) -> list[str]:
    """PDFs in data/<year>/pdfs with no staging record yet (sorted).

    Legacy manual-mode selection (``stage --year``). The nightly drain
    uses ``--from-manifest`` instead: this directory scan includes
    text-born docs and cannot see canonical dispositions (finding 8).
    """
    root = staging_root(data_dir)
    already_staged = {
        path.stem for path in (root / str(year)).glob("*.json")
    } if (root / str(year)).exists() else set()
    skip = set(already_staged)
    if check_canonical:
        skip |= canonical_parsed_doc_ids(db_path or default_db_path(data_dir))
    pdfs_dir = Path(data_dir) / str(year) / "pdfs"
    candidates = [
        pdf.stem for pdf in sorted(pdfs_dir.glob("*.pdf")) if pdf.stem not in skip
    ]
    return candidates[: max(limit, 0)]


def load_work_manifest_items(path: str | Path) -> tuple[list[dict], str | None]:
    """(items, error) from an ``ocr_zero_rows --export-work-manifest`` file.

    Fail-closed: an absent, unreadable, malformed, or foreign-schema
    manifest is an error, never an empty queue (draining nothing must be a
    loud operator decision, not a silent no-op).
    """
    manifest_path = Path(path)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [], f"unreadable work manifest {manifest_path}: {exc}"
    if not (
        isinstance(payload, dict)
        and payload.get("schema") == WORK_MANIFEST_SCHEMA
        and isinstance(payload.get("items"), list)
    ):
        return [], (
            f"{manifest_path} is not a {WORK_MANIFEST_SCHEMA} work manifest "
            "(regenerate it: ocr_zero_rows.py --export-work-manifest)"
        )
    for index, item in enumerate(payload["items"]):
        if not isinstance(item, dict) or "doc_id" not in item or "year" not in item:
            return [], f"{manifest_path}: malformed manifest item at index {index}"
    return payload["items"], None


def manifest_years(items: list[dict]) -> tuple[list[int], str | None]:
    """Distinct backlog years from the authoritative manifest, newest first."""
    years = set()
    for index, item in enumerate(items):
        year = item.get("year") if isinstance(item, dict) else None
        if (
            not isinstance(year, int)
            or isinstance(year, bool)
            or not 1000 <= year <= 9999
        ):
            return [], f"malformed manifest year at item {index}: {year!r}"
        years.add(year)
    return sorted(years, reverse=True), None


def select_candidates_from_manifest(
    items: list[dict],
    year: int,
    limit: int,
    data_dir: str | Path = "data",
    *,
    check_canonical: bool = False,
    db_path: str | Path | None = None,
) -> tuple[list[str], list[str]]:
    """Authoritative-queue candidate selection (finding 8).

    The manifest IS the unresolved queue; PDF directories are not scanned,
    so text-born docs cannot leak into the pixel pipeline. Returns
    ``(doc_ids, notes)``: entries for other years are ignored; an entry
    without a local PDF is skipped WITH A NOTE (the artifact vanished after
    export — never silently); an entry whose live staging record is
    compatible success/no_txs is skipped (it is staged and awaiting the audit; re-
    reading it nightly would burn quota); an entry whose live record is an
    ERROR record IS retried — the authoritative queue still lists the doc
    as unresolved, and the re-stage archives the failed attempt as
    versioned evidence instead of overwriting it. Canonical skips require the
    current artifact/generation's verified terminal disposition, not a historic
    doc-ID success. Old filing-year groups resolve through House archive
    metadata and recorded acquisitions; explicit hash/generation mismatches
    are refused, never replaced with another same-ID artifact.
    """
    import duckdb
    ocr = _ocr_ingestion()

    if not isinstance(data_dir, ocr.ApprovedDataRoot):
        with ocr.approved_data_root(data_dir) as root:
            return select_candidates_from_manifest(items, year, limit, root,
                check_canonical=check_canonical, db_path=db_path)

    # A frozen old manifest may stage under filing year, then a new export
    # groups the same artifact under archive year. Reuse that reading, not a
    # second staging/canonical identity for the same document and bytes.
    staged: dict[str, list[tuple[int, dict]]] = {}
    for group_year, record in iter_staging_records(data_dir):
        if isinstance(record, dict) and record.get("doc_id") and record.get("parse_status") in ("success", "no_txs"):
            staged.setdefault(str(record["doc_id"]), []).append((group_year, record))
    canonical = canonical_terminal_artifacts(db_path or default_db_path(data_dir)) if check_canonical else set()
    candidates: list[str] = []
    notes: list[str] = []
    seen: set[str] = set()
    path = Path(db_path or default_db_path(data_dir))
    with duckdb.connect(str(path), read_only=True) if path.exists() else nullcontext(None) as conn:
        for item in items:
            try:
                item_year = int(item["year"])
            except (KeyError, TypeError, ValueError):
                continue
            if item_year != int(year):
                continue
            doc_id = str(item["doc_id"])
            if doc_id in seen:
                continue
            seen.add(doc_id)
            try:
                request = ArtifactRequest.from_manifest(item)
                pdf_path, identity = ocr.local_work_artifact(conn, data_dir, request.resolver_fields())
            except (OSError, RuntimeError) as exc:
                notes.append(f"[{doc_id}] artifact binding refused ({exc}); skipped")
                continue
            if identity["artifact_sha256"] is None:
                notes.append(f"[{doc_id}] manifest entry has no local PDF ({pdf_path}); skipped")
                continue
            if (doc_id, identity["artifact_sha256"], identity["ingestion_generation"]) in canonical:
                continue
            records = staged.get(doc_id, [])
            if any(
                identity["ingestion_generation"] is not None
                and record.get("pdf_sha256") == identity["artifact_sha256"]
                and record.get("archive_year", group_year) == identity["archive_year"]
                and record.get("ingestion_generation") == identity["ingestion_generation"]
                for group_year, record in records
            ):
                continue
            if records:
                notes.append(f"[{doc_id}] artifact binding refused (staged identity is incompatible; evidence reconciliation required); skipped")
                continue
            candidates.append(doc_id)
    return candidates[: max(limit, 0)], notes


def stage_batch(
    year: int,
    limit: int,
    *,
    data_dir: str | Path = "data",
    check_canonical: bool = False,
    db_path: str | Path | None = None,
    config: VisionBConfig | None = None,
    read_fn=None,
    cooldown: float = 0.0,
    workers: int = 1,
    manifest: list[dict] | None = None,
    dry_run: bool = False,
) -> dict:
    """Stage up to ``limit`` docs for ``year``. Never raises for endpoint states.

    Batch-wide stops: non-Google quota exhaustion, five consecutive Google
    rate-shaped 429 docs, or five consecutive Google 503 docs (``quota_stop``,
    clean, exit 0 — the current doc stays unstaged), and auth/config defects
    (``fatal_stop``, alerted — every remaining doc would fail the same way).
    The 429 and 503 streaks are separate; successes reset both, and any
    non-503 doc resets only the 503 streak.

    ``workers`` reads docs concurrently (the GLM call is I/O-bound, so
    threads overlap network latency); a batch-wide stop halts new
    submissions and lets in-flight docs finish. Per-document publication is
    serialized after inference, so the vision calls remain concurrent.

    Concurrency caps (finding 11): ``workers`` above MAX_WORKERS (2) raises
    ValueError — the Z.AI subscription allows two concurrent vision
    streams, and an operator override above that is a quota-budget
    violation, not a tuning knob. Cross-process exclusion is the cron
    wrapper's single-instance flock (``cron/backlog_ocr.sh`` holds
    ``data/logs/ocr-backlog-cron/drain.lock``). A per-document publication
    lock also protects direct invocations and serializes archive-plus-live
    replacement without holding a lock during inference.

    ``manifest`` (``ocr_zero_rows.export_work_manifest`` items) replaces
    the PDF-directory scan with the authoritative unresolved selection
    (finding 8): see ``select_candidates_from_manifest``. A doc whose live
    record is an error record is re-staged; the superseded record is
    archived (``archive_current_attempt``) before the new one is written.
    """
    ocr = _ocr_ingestion()

    if not isinstance(data_dir, ocr.ApprovedDataRoot):
        with ocr.approved_data_root(data_dir) as root:
            return stage_batch(year, limit, data_dir=root, check_canonical=check_canonical,
                db_path=db_path, config=config, read_fn=read_fn, cooldown=cooldown,
                workers=workers, manifest=manifest, dry_run=dry_run)
    if not 1 <= int(workers) <= MAX_WORKERS:
        raise ValueError(
            f"workers={workers} refused: the Z.AI subscription allows "
            f"{MAX_WORKERS} concurrent vision streams; values above "
            f"{MAX_WORKERS} violate the quota budget and are not accepted"
        )
    if manifest is None:
        candidates = select_candidates(
            year, limit, data_dir, check_canonical=check_canonical, db_path=db_path
        )
        notes = []
    else:
        candidates, notes = select_candidates_from_manifest(
            manifest,
            year,
            limit,
            data_dir,
            check_canonical=check_canonical,
            db_path=db_path,
        )
    for note in notes:
        print(f"  {note}", flush=True)
    missing_pdfs = sum("manifest entry has no local PDF" in note for note in notes)
    blocked_artifacts = len(notes) - missing_pdfs
    pdfs_dir = Path(data_dir) / str(year) / "pdfs"
    has_year_items = any(str(item.get("year")) == str(year) for item in manifest or [])
    missing_dirs = int(not pdfs_dir.is_dir() and (not has_year_items or missing_pdfs > 0))
    if missing_dirs:
        print(
            f"SKIP: missing PDF directory {pdfs_dir}; "
            f"year {year} selection is not an empty-success (missing_dirs=1)",
            flush=True,
        )
    if dry_run:
        print(
            f"DRY RUN year={year}: manifest_candidates={len(candidates)} "
            f"missing_pdfs={missing_pdfs} missing_dirs={missing_dirs} "
            f"blocked_artifacts={blocked_artifacts}",
            flush=True,
        )
        return {
            "year": year,
            "candidates": len(candidates),
            "staged": 0,
            "errors": 0,
            "quota_stop": False,
            "google_quota_dead": False,
            "google_endpoint_saturated": False,
            "fatal_stop": False,
            "remaining": 0,
            "missing_pdfs": missing_pdfs,
            "missing_dirs": missing_dirs,
            "blocked_artifacts": blocked_artifacts,
            "dry_run": True,
        }
    if config is None and read_fn is None:
        config = VisionBConfig.from_env()
        if config is None:
            raise RuntimeError(
                "GLM vision endpoint is not configured: set VISION_B_BASE_URL, "
                "VISION_B_MODEL (and VISION_B_API_KEY / VISION_B_TIMEOUT)"
            )
    staged = errors = 0
    quota_stop = False
    google_quota_dead = False
    google_endpoint_saturated = False
    fatal_stop = False
    remaining = 0
    consecutive_google_429s = 0
    consecutive_google_503s = 0
    batch_state_lock = threading.Lock()
    stop = threading.Event()

    def _stage_one(doc_id: str) -> str:
        nonlocal consecutive_google_429s, consecutive_google_503s
        nonlocal google_quota_dead, google_endpoint_saturated
        if stop.is_set():
            return "skipped"
        pdf_path = default_pdf_path(data_dir, year, doc_id)
        identity = None
        if manifest is not None:
            item = next(item for item in manifest if str(item["doc_id"]) == doc_id and int(item["year"]) == year)
            try:
                pdf_path, identity = manifest_artifact(ArtifactRequest.from_manifest(item), data_dir, db_path)
            except (OSError, RuntimeError) as exc:
                print(f"  [{doc_id}] artifact binding refused ({exc})", flush=True)
                return "error"
        record = build_staging_record(
            doc_id,
            year,
            pdf_path,
            config,
            read_fn=read_fn,
            artifact_identity=identity,
            data_dir=data_dir,
        )
        error = record.get("error") or ""
        google_rate_limited = bool(
            config is not None
            and is_google_api_host(config.base_url)
            and record.get("error_kind") == "transient"
            and "google rate limit transient (http 429)" in error.lower()
        )
        google_service_unavailable = bool(
            config is not None
            and is_google_api_host(config.base_url)
            and record.get("error_kind") == "transient"
            and "google endpoint unavailable transient (http 503)" in error.lower()
        )
        if record["parse_status"] == "error" and (
            record.get("error_kind") == "quota"
            or (is_quota_error(error) and not google_rate_limited)
        ):
            # Quota exhaustion: not a doc defect. Write nothing, stop the
            # batch cleanly; the doc stays unstaged for the next run.
            return "quota"
        if record["parse_status"] == "error" and record.get("error_kind") in (
            "auth",
            "config",
        ):
            # Credentials/configuration are batch-wide defects: alert loudly
            # and stop. Write nothing — the operator must fix the setup.
            print(
                f"ALERT: GLM endpoint {record.get('error_kind')} defect: "
                f"{error}; fix the configuration and re-run",
                flush=True,
            )
            return "fatal"
        with batch_state_lock:
            if stop.is_set():
                return "skipped"
            if google_rate_limited:
                consecutive_google_503s = 0
                consecutive_google_429s += 1
                if consecutive_google_429s >= GOOGLE_429_CONSECUTIVE_STOP_THRESHOLD:
                    # Five consecutive 429 docs exceed a short burst; stop
                    # before daily-quota errors can fill the manifest.
                    google_quota_dead = True
                    stop.set()
                    return "google_quota_dead"
            elif google_service_unavailable:
                consecutive_google_503s += 1
                if consecutive_google_503s >= GOOGLE_503_CONSECUTIVE_STOP_THRESHOLD:
                    google_endpoint_saturated = True
                    stop.set()
                    return "google_endpoint_saturated"
            elif record["parse_status"] in ("success", "no_txs"):
                consecutive_google_429s = 0
                consecutive_google_503s = 0
            else:
                consecutive_google_503s = 0
        _, archived = publish_staging_record(data_dir, record)
        if archived is not None:
            print(
                f"  [{doc_id}] prior staging record archived: "
                f"attempts/{archived.name}",
                flush=True,
            )
        if record["parse_status"] == "error":
            print(f"  [{doc_id}] ERROR: {error}", flush=True)
            return "error"
        print(
            f"  [{doc_id}] {record['parse_status']} rows={len(record['rows'])}",
            flush=True,
        )
        return "staged"

    if workers <= 1:
        for index, doc_id in enumerate(candidates):
            if cooldown and index:
                time.sleep(cooldown)
            outcome = _stage_one(doc_id)
            if outcome in (
                "quota",
                "google_quota_dead",
                "google_endpoint_saturated",
            ):
                quota_stop = True
                remaining = len(candidates) - index
                break
            if outcome == "fatal":
                fatal_stop = True
                remaining = len(candidates) - index
                break
            if outcome == "error":
                errors += 1
            elif outcome == "staged":
                staged += 1
        return {
            "year": year,
            "candidates": len(candidates),
            "staged": staged,
            "errors": errors,
            "quota_stop": quota_stop,
            "google_quota_dead": google_quota_dead,
            "google_endpoint_saturated": google_endpoint_saturated,
            "fatal_stop": fatal_stop,
            "remaining": remaining,
            "missing_pdfs": missing_pdfs,
            "missing_dirs": missing_dirs,
            "blocked_artifacts": blocked_artifacts,
        }

    from concurrent.futures import (
        FIRST_COMPLETED,
        ThreadPoolExecutor,
        wait as futures_wait,
    )

    lock = threading.Lock()
    processed = 0

    def _worker(doc_id: str, index: int) -> str:
        if stop.is_set():
            return "skipped"
        if cooldown and index:
            time.sleep(cooldown)
        outcome = _stage_one(doc_id)
        if outcome in (
            "quota",
            "google_quota_dead",
            "google_endpoint_saturated",
            "fatal",
        ):
            stop.set()
        return outcome

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_worker, doc_id, index): index
            for index, doc_id in enumerate(candidates)
        }
        pending = set(futures)
        while pending:
            done, pending = futures_wait(pending, return_when=FIRST_COMPLETED)
            for fut in done:
                index = futures[fut]
                try:
                    outcome = fut.result()
                except Exception as exc:  # a worker crash must not kill the batch
                    outcome = "error"
                    print(f"  worker crash at index {index}: {exc}", flush=True)
                with lock:
                    processed += 1
                    if outcome == "error":
                        errors += 1
                    elif outcome == "staged":
                        staged += 1
                if outcome in (
                    "quota",
                    "google_quota_dead",
                    "google_endpoint_saturated",
                    "fatal",
                ):
                    with lock:
                        remaining = len(candidates) - processed
                        if outcome in (
                            "quota",
                            "google_quota_dead",
                            "google_endpoint_saturated",
                        ):
                            quota_stop = True
                            if outcome == "google_quota_dead":
                                google_quota_dead = True
                            if outcome == "google_endpoint_saturated":
                                google_endpoint_saturated = True
                        else:
                            fatal_stop = True
                    for other in pending:
                        other.cancel()
                    pending = {f for f in pending if not f.cancelled()}
        if quota_stop or fatal_stop:
            with lock:
                remaining = max(remaining, len(candidates) - processed)
    return {
        "year": year,
        "candidates": len(candidates),
        "staged": staged,
        "errors": errors,
        "quota_stop": quota_stop,
        "google_quota_dead": google_quota_dead,
        "google_endpoint_saturated": google_endpoint_saturated,
        "fatal_stop": fatal_stop,
        "remaining": remaining,
        "missing_pdfs": missing_pdfs,
        "missing_dirs": missing_dirs,
        "blocked_artifacts": blocked_artifacts,
    }


# --------------------------------------------------------------------------
# promote
# --------------------------------------------------------------------------


def staged_row_to_tx(row, page_number=None) -> dict:
    """Convert a staged row [asset, type, tx_date, notif_date, amount_letter]
    into the transaction dict ``insert_transactions``/validation expects.

    ``page_number`` comes from the record's ``row_pages`` (per-row page
    attribution recorded at stage time, finding 6) so the canonical
    ``source_row_id`` identity is page-bound exactly like the Gemini path.
    """
    values = list(row)
    if len(values) != 5:
        raise ValueError(f"staged row must have 5 fields, got {len(values)}")
    asset, tx_type, tx_date, notif_date, amount_letter = values
    letter = str(amount_letter or "").strip().upper()
    if letter not in AMOUNT_MIDPOINTS:
        raise ValueError(f"invalid staged amount letter: {amount_letter!r}")
    return {
        "asset": str(asset),
        "type": str(tx_type),
        "date": tx_date,
        "notif_date": notif_date,
        "amount_letter": letter,
        "amount_midpoint": AMOUNT_MIDPOINTS[letter],
        "page_number": page_number,
    }


def audit_report_state(data_dir: str | Path = "data") -> dict | None:
    """The landed audit report state (audit_sample.py ``report``), fail-closed.

    Returns the parsed ``audit-report-state.json`` payload, or None when the
    file is absent, unreadable, malformed, or carries a foreign schema —
    every one of those must refuse promotion exactly like a missing report,
    never guess.
    """
    path = staging_root(data_dir) / AUDIT_REPORT_STATE_NAME
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("schema") != AUDIT_REPORT_STATE_SCHEMA:
        return None
    return payload


def _audit_report_gate_issue(
    state: dict | None,
    current_history_sha256: str | None,
    current_sample_file_sha256: str | None = None,
) -> str | None:
    """Reason the landed report does not prove the promotion preconditions."""
    if state is None:
        return "audit report state is missing or untrusted"
    if state.get("state") != "complete":
        return f"state is {state.get('state')!r}, not 'complete'"
    if state.get("budget_met") is not True:
        return f"budget_met is {state.get('budget_met')!r}, not boolean true"

    required_counts = ("lot_size", "audited", "critical_failure_count", "unresolved_count")
    required_rates = ("rate", "upper_bound", "confidence", "max_rate")
    invalid = [
        key
        for key in required_counts
        if not isinstance(state.get(key), int) or isinstance(state.get(key), bool)
    ]
    invalid.extend(
        key
        for key in required_rates
        if not isinstance(state.get(key), (int, float))
        or isinstance(state.get(key), bool)
        or not math.isfinite(state[key])
    )
    critical_failures = state.get("critical_failures")
    if not isinstance(critical_failures, list) or (
        "critical_failure_count" not in invalid
        and len(critical_failures) != state["critical_failure_count"]
    ):
        invalid.append("critical_failures")
    if invalid:
        return "required gate metrics are missing or invalid: " + ", ".join(invalid)

    lot_size = state["lot_size"]
    audited = state["audited"]
    critical_count = state["critical_failure_count"]
    unresolved = state.get("unresolved")
    if (
        lot_size <= 0
        or audited != lot_size
        or critical_count < 0
        or critical_count > audited
        or state["unresolved_count"] != 0
        or unresolved != []
    ):
        return (
            "complete-state counts are inconsistent (require non-empty lot, "
            "audited == lot_size, no unresolved docs, and valid critical count)"
        )
    if state.get("reason") is not None or (
        "completion_reasons" in state and state.get("completion_reasons") != []
    ):
        return "complete report carries an incomplete-state reason"
    if not math.isclose(state["rate"], critical_count / audited, rel_tol=1e-9, abs_tol=1e-12):
        return "reported critical rate does not match critical_failure_count / audited"
    if not (0 < state["confidence"] < 1 and 0 <= state["max_rate"] <= 1):
        return "confidence or max_rate is outside its valid probability range"
    if not (0 <= state["upper_bound"] <= 1) or state["upper_bound"] > state["max_rate"]:
        return "upper_bound is invalid or exceeds max_rate despite budget_met=true"
    report_history_sha256 = state.get("audit_history_sha256")
    if not isinstance(report_history_sha256, str) or re.fullmatch(
        r"[0-9a-f]{64}", report_history_sha256
    ) is None:
        return "complete report has no trusted audit history binding"
    if report_history_sha256 != current_history_sha256:
        return "audit history changed since report; rerun audit_sample.py report"
    sample_file = state.get("sample_file")
    if (
        not isinstance(sample_file, str)
        or not sample_file
        or Path(sample_file).name != sample_file
    ):
        return "complete report has no trusted frozen sample-lot identity"
    report_sample_sha256 = state.get("sample_file_sha256")
    if not isinstance(report_sample_sha256, str) or re.fullmatch(
        r"[0-9a-f]{64}", report_sample_sha256
    ) is None:
        return "complete report has no trusted frozen sample-lot binding"
    if (
        not isinstance(current_sample_file_sha256, str)
        or report_sample_sha256 != current_sample_file_sha256
    ):
        return "frozen sample lot changed or is missing since report; rerun audit_sample.py report"
    return None


def audit_report_gate_open(
    state: dict | None,
    current_history_sha256: str | None,
    current_sample_file_sha256: str | None = None,
) -> bool:
    """True only when a complete report proves a met budget with valid metrics."""
    return (
        _audit_report_gate_issue(
            state, current_history_sha256, current_sample_file_sha256
        )
        is None
    )


def _audit_report_refusal(
    state: dict | None,
    data_dir: str | Path,
    current_history_sha256: str | None,
    current_sample_file_sha256: str | None,
) -> str:
    """Explain the report-level promotion refusal, including available math."""
    if state is None:
        return (
            "REFUSED: audit report state missing — run audit_sample.py report "
            f"({staging_root(data_dir) / AUDIT_REPORT_STATE_NAME}); an unreadable, "
            "malformed, or foreign-schema report is treated as missing; state and "
            "critical rate/bound/n vs budget are unavailable; canonical "
            "promotion must not proceed"
        )

    report_state = state.get("state")
    if report_state == "incomplete":
        reasons = state.get("completion_reasons")
        reason = (
            "; ".join(str(value) for value in reasons)
            if isinstance(reasons, list) and reasons
            else state.get("reason") or "the report did not record a completion reason"
        )
        guidance = state.get("completion_guidance")
        fix = (
            "; how to fix: " + "; ".join(str(value) for value in guidance)
            if isinstance(guidance, list) and guidance
            else ""
        )
        prefix = f"audit report state incomplete because {reason}{fix}"
    elif report_state == "complete":
        if state.get("budget_met") is False:
            prefix = "audit report is complete but the critical error budget is not met"
        else:
            prefix = "audit report state is complete but " + str(
                _audit_report_gate_issue(
                    state, current_history_sha256, current_sample_file_sha256
                )
            )
    else:
        prefix = f"audit report state {report_state!r} is not recognized"

    def number(value) -> float | None:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        return parsed if math.isfinite(parsed) else None

    def percent(value) -> str:
        parsed = number(value)
        return f"{parsed * 100:.4g}%" if parsed is not None else "unknown"

    audited = number(state.get("audited"))
    lot_size = number(state.get("lot_size"))
    critical_count = number(state.get("critical_failure_count"))
    if critical_count is None:
        failures = state.get("critical_failures")
        critical_count = float(len(failures)) if isinstance(failures, list) else None
    rate = number(state.get("rate"))
    if rate is None and critical_count is not None and audited:
        rate = critical_count / audited
    upper = number(state.get("upper_bound"))
    max_rate = number(state.get("max_rate"))
    confidence = number(state.get("confidence"))
    population_critical = number(state.get("population_critical_failure_count"))
    population_text = (
        f"population sticky critical-failure tally {population_critical:g} "
        "across audit history (outside-lot failures remain visible, excluded "
        "from the current-lot bound)"
        if population_critical is not None
        else "population sticky critical-failure tally unavailable"
    )
    confidence_text = (
        f"{confidence * 100:.4g}%" if confidence is not None else "unknown"
    )
    count_text = (
        f"{critical_count:g}/{audited:g} resolved"
        if critical_count is not None and audited is not None
        else "unknown critical failures / n"
    )
    lot_text = f"lot n={lot_size:g}" if lot_size is not None else "lot n=unknown"
    if report_state == "incomplete":
        metric_text = (
            f"resolved-only diagnostics: {count_text} ({lot_text}), "
            f"critical rate {percent(rate)}, {confidence_text} upper bound "
            f"{percent(upper)}, max_rate {percent(max_rate)}; full-lot "
            "budget is not certified"
        )
    else:
        metric_text = (
            f"critical {count_text} ({lot_text}), rate {percent(rate)}, "
            f"{confidence_text} one-sided upper bound {percent(upper)}, "
            f"max_rate {percent(max_rate)}"
        )
    omission = (
        "; omission check not run — this is a pixel-only certification"
        if state.get("omission_check_status") == "not_run"
        else ""
    )
    return (
        f"REFUSED: {prefix}; {metric_text}; {population_text}; budget_met="
        f"{state.get('budget_met')!r}{omission}; canonical promotion must not proceed"
    )


def record_identity(record: StagingRecord) -> dict[str, str | None]:
    """The (pdf_sha256, prompt_sha256, parsed_at_utc) triple a verdict is
    bound to (finding 6): it changes iff the artifact, extraction recipe, or
    staging attempt changed."""
    return {field: record.get(field) for field in AUDIT_IDENTITY_FIELDS}


def verdict_binding(
    data_dir: str | Path, doc_id: str, record: StagingRecord
) -> tuple[bool, str | None]:
    """(bound, refusal_reason) between the audit history and THIS record.

    ``audit_sample.record_verdict`` stores a canonical stable-record content
    hash plus the extraction identity in its audit-results entry. The stable
    hash includes the staged rows and their dedicated rows hash; identity is
    also checked for recipe/artifact/attempt drift. Metadata-only legacy
    entries remain historical but cannot authorize promotion.
    """
    passes = [
        entry
        for entry in load_audit_results(data_dir)
        if isinstance(entry, dict)
        and str(entry.get("doc_id")) == str(doc_id)
        and entry.get("verdict") == "pass"
    ]
    if not passes:
        return False, (
            "no pass verdict in the audit results history; run "
            "audit_sample.py record"
        )
    identity = record_identity(record)
    rows_hash = staged_rows_sha256(record.get("rows", []))
    recorded_rows_hash = record.get("staged_rows_sha256")
    if recorded_rows_hash is not None and recorded_rows_hash != rows_hash:
        return False, (
            "refused: staged row content does not match staged_rows_sha256; "
            "the staged evidence changed or its content hash is corrupt"
        )
    # Failed promotion details are mutable bookkeeping, not OCR evidence.
    stable_hashes = {
        audit_staged_record_sha256(record),
        audit_staged_record_sha256(
            {key: value for key, value in record.items() if key != "promotion"}
        ),
    }
    for entry in passes:
        entry_identity = {field: entry.get(field) for field in AUDIT_IDENTITY_FIELDS}
        # audit_sample's stable-record hash is a canonical content hash that
        # includes rows (and, for newly staged records, staged_rows_sha256).
        # Prefer it over the legacy metadata-only triple.
        entry_content_hash = entry.get("staged_record_sha256")
        entry_rows_hash = entry.get("staged_rows_sha256")
        if entry_rows_hash is not None and entry_rows_hash != rows_hash:
            continue
        if (
            entry_content_hash in stable_hashes
            and (entry_rows_hash is None or entry_rows_hash == rows_hash)
        ):
            return True, None
        # Old audit records can carry a frozen full-record hash but predate
        # the dedicated rows field. Their original full-record binding is
        # still meaningful and is checked above; metadata-only entries are
        # preserved as pre-content-hash history but cannot authorize a new
        # promotion because they cannot prove which rows were reviewed.
        if (
            entry_identity == identity
            and all(value is not None for value in entry_identity.values())
            and entry_content_hash is None
        ):
            continue
    return False, (
        "refused: the audit pass verdict is not bound to this record's identity "
        "and staged row content (canonical rows sha256); stable record content "
        "also requires pdf_sha256/prompt_sha256/parsed_at_utc; metadata-only/"
        "pre-content-hash verdicts are historical only and cannot authorize "
        "promotion (the record may have been re-staged after the audit) — "
        "re-audit the current attempt"
    )


def append_promoted_journal(root: Path, entry: dict) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with (root / PROMOTED_JOURNAL_NAME).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")


def read_promoted_journal(root: Path) -> list[dict]:
    path = root / PROMOTED_JOURNAL_NAME
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"unreadable promotion journal {path}: {exc}") from exc
    entries = []
    for line_number, line in enumerate(lines, 1):
        line = line.strip()
        if line:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"malformed promotion journal {path}:{line_number}: {exc}"
                ) from exc
            if (
                not isinstance(entry, dict)
                or not isinstance(entry.get("doc_id"), str)
                or not entry["doc_id"].strip()
            ):
                raise ValueError(
                    f"malformed promotion journal {path}:{line_number}: "
                    "expected an object with a non-empty doc_id"
                )
            entries.append(entry)
    return entries


def promote_docs(
    year: int,
    doc_ids: list[str] | None = None,
    *,
    data_dir: str | Path = "data",
    db_path: str | Path | None = None,
    require_audit_pass: bool = True,
    force_unaudited: bool = False,
    cooldown_note: str | None = None,
) -> dict:
    """Promote under the shared audit-root lock and require a fresh report."""
    if not require_audit_pass:
        raise ValueError(
            "audit approval is mandatory: require_audit_pass=False is no "
            "longer accepted (pass audit:'pass' records or force_unaudited)"
        )
    ocr = _ocr_ingestion()
    if not isinstance(data_dir, ocr.ApprovedDataRoot):
        with ocr.approved_data_root(data_dir) as root:
            return promote_docs(
                year, doc_ids, data_dir=root, db_path=db_path,
                require_audit_pass=require_audit_pass,
                force_unaudited=force_unaudited, cooldown_note=cooldown_note,
            )
    # Root audit lock precedes every per-document publication lock.
    with audit_operation_lock(data_dir):
        return _promote_docs_locked(
            year, doc_ids, data_dir=data_dir, db_path=db_path,
            require_audit_pass=require_audit_pass,
            force_unaudited=force_unaudited, cooldown_note=cooldown_note,
        )


def _promote_docs_locked(
    year: int,
    doc_ids: list[str] | None = None,
    *,
    data_dir: str | Path = "data",
    db_path: str | Path | None = None,
    require_audit_pass: bool = True,
    force_unaudited: bool = False,
    cooldown_note: str | None = None,
) -> dict:
    """Promote AUDITED staged records into canonical via the insert path.

    Audit approval is MANDATORY at this boundary (FP review finding 1): a
    record without ``audit == "pass"`` is refused unless
    ``force_unaudited=True`` (the operator's explicit override, printed
    with a loud warning). ``require_audit_pass=False`` is refused loudly —
    there is no permissive default anymore.

    ``insert_transactions_outcome`` (from ``analyzer.ocr.ingestion``) writes
    the transactions AND the pdf_parse_runs record transactionally, exactly
    as the OCR path does. The staging JSON is archived — promotion consumes
    the live record — ONLY when the FULL staged row set is confirmed committed
    (outcome complete: committed, no rejections, count matches). A partial
    or rejected insert keeps the record and stamps the rejection details on
    it; a member rejection is never mistaken for a no_txs.

    Verdict binding (finding 6): an ``audit: pass`` stamp is only accepted
    when the audit-results history contains a pass entry bound to THIS
    record's staged content hash and identity (pdf_sha256 + prompt_sha256 +
    parsed_at_utc). A metadata-only legacy verdict cannot authorize
    promotion; a re-staged attempt must be re-audited.
    """
    if not require_audit_pass:
        raise ValueError(
            "audit approval is mandatory: require_audit_pass=False is no "
            "longer accepted (pass audit:'pass' records or force_unaudited)"
        )
    ocr = _ocr_ingestion()

    if not isinstance(data_dir, ocr.ApprovedDataRoot):
        with ocr.approved_data_root(data_dir) as root:
            return promote_docs(year, doc_ids, data_dir=root, db_path=db_path,
                require_audit_pass=require_audit_pass, force_unaudited=force_unaudited,
                cooldown_note=cooldown_note)

    # Promotion consumes a row from the audited population. Do not let an
    # explicit doc-id or force override bypass unreadable population/history.
    staged_records = iter_staging_records(data_dir)
    audit_results = load_audit_results(data_dir)
    current_history_sha256 = audit_history_sha256(audit_results)
    read_promoted_journal(staging_root(data_dir))
    db_path = str(db_path or default_db_path(data_dir))
    root = staging_root(data_dir)
    progress_path = Path(data_dir) / Path(ocr.PROGRESS_PATH).name
    progress = ocr.load_progress(progress_path)

    if not doc_ids:
        doc_ids = [
            record["doc_id"]
            for yr, record in staged_records
            if yr == int(year)
        ]

    if force_unaudited:
        print(
            "WARNING: --force-unaudited: promoting records WITHOUT a "
            "passing pixel audit. This bypasses the only verified quality "
            "gate on GLM-only readings; output must not be treated as "
            "audited.",
            flush=True,
        )
    else:
        # Report-level gate (finding 9 contract): the pixel audit report
        # must be COMPLETE with the material-error budget MET before any
        # audited record promotes. A missing/incomplete/failing report
        # refuses the whole batch — the per-record audit: pass stamp alone
        # says nothing about the population.
        state = audit_report_state(data_dir)
        current_sample_sha256 = sample_lot_file_sha256(
            data_dir, state.get("sample_file") if state is not None else None
        )
        if not audit_report_gate_open(
            state, current_history_sha256, current_sample_sha256
        ):
            if doc_ids:
                print(
                    _audit_report_refusal(
                        state,
                        data_dir,
                        current_history_sha256,
                        current_sample_sha256,
                    ),
                    flush=True,
                )
                return {
                    "year": int(year),
                    "promoted": 0,
                    "refused": len(doc_ids),
                    "failed": 0,
                }

    promoted = refused = failed = 0

    def _promote_one(doc_id: str) -> None:
        nonlocal promoted, refused, failed
        try:
            record_bytes = ocr.read_work_artifact(data_dir, staging_record_path(data_dir, year, doc_id))
            record = decode_staging_record(record_bytes)
        except FileNotFoundError:
            print(f"  [{doc_id}] no staging record; skipped", flush=True)
            refused += 1
            return
        if record.get("doc_id") != str(doc_id) or record.get("year") != int(year):
            print(f"  [{doc_id}] refused: staging record identity does not match selection", flush=True)
            refused += 1
            return
        if record.get("parse_status") == "error":
            print(f"  [{doc_id}] refused: staging record is an error record", flush=True)
            refused += 1
            return
        if record.get("audit") != "pass" and not force_unaudited:
            print(
                f"  [{doc_id}] refused: audit approval is mandatory and the "
                "staging record lacks audit: pass (run the pixel audit or "
                "pass --force-unaudited explicitly)",
                flush=True,
            )
            refused += 1
            return
        if not force_unaudited:
            bound, reason = verdict_binding(data_dir, doc_id, record)
            if not bound:
                print(f"  [{doc_id}] {reason}", flush=True)
                refused += 1
                return
        try:
            _, identity = manifest_artifact(ArtifactRequest.from_record(record), data_dir, db_path)
            artifact_year = identity["archive_year"]
            current_sha = identity["artifact_sha256"]
        except (OSError, RuntimeError) as exc:
            print(f"  [{doc_id}] refused: PDF unreadable ({exc})", flush=True)
            refused += 1
            return
        if current_sha != record.get("pdf_sha256"):
            print(
                f"  [{doc_id}] refused: staged pdf_sha256 no longer matches "
                "the artifact on disk",
                flush=True,
            )
            refused += 1
            return
        try:
            rows = record["rows"]
            row_pages = record.get("row_pages") or []
            transactions = [
                staged_row_to_tx(
                    row,
                    page_number=(
                        row_pages[index] if index < len(row_pages) else None
                    ),
                )
                for index, row in enumerate(rows)
            ]
            ingestion_generation = ocr.resolve_ingestion_generation(
                db_path, doc_id, artifact_year, record["pdf_sha256"]
            )
            if record.get("ingestion_generation") is not None and record["ingestion_generation"] != ingestion_generation:
                raise RuntimeError("staged artifact generation is no longer current")
            staged_count = len(transactions)
            outcome = ocr.insert_transactions_outcome(
                doc_id,
                artifact_year,
                record["member"],
                transactions,
                db_path=db_path,
                parser_version=GLM_PARSER_VERSION,
                raw_count=staged_count,
                artifact_sha256=record["pdf_sha256"],
                ingestion_generation=ingestion_generation,
                engine_model=GLM_ENGINES_ATTEMPTED,
            )
        except Exception as exc:
            # duckdb write-lock and generation-binding problems surface
            # naturally; the staging record is kept for a later attempt.
            print(f"  [{doc_id}] ERROR: {exc}", flush=True)
            failed += 1
            return

        if outcome.complete:
            ok, progress_key = True, (
                "no_txs" if outcome.status == "no_txs" else "completed"
            )
        else:
            ok, progress_key = False, "errors"
        # Mirror main()'s run bookkeeping (the parse-run row itself was
        # already written transactionally inside insert_transactions).
        ocr.mark_progress(progress, doc_id, progress_key)
        ocr.save_progress(progress, progress_path)
        if not ok:
            # Partial or rejected output is never complete: keep the staging
            # record and stamp the explicit rejection details on it.
            record["promotion"] = {
                "outcome": outcome.as_dict(),
                "at_utc": utc_now_iso(),
            }
            write_staging_record(data_dir, record)
            print(
                f"  [{doc_id}] ERROR: insert {outcome.status} "
                f"(inserted={outcome.inserted}/{outcome.expected} staged rows; "
                f"rejections={outcome.rejections or outcome.detail}); "
                "staging record kept and marked",
                flush=True,
            )
            failed += 1
            return
        staged_hash = audit_staged_record_sha256(record)
        rows_hash = staged_rows_sha256(rows)
        archived = archive_current_attempt(data_dir, year, doc_id)
        append_promoted_journal(
            root,
            {
                "doc_id": str(doc_id),
                "year": int(year),
                "archive_year": int(artifact_year),
                "inserted": outcome.inserted,
                "outcome_status": outcome.status,
                "unaudited": bool(force_unaudited),
                "parser_version": GLM_PARSER_VERSION,
                "promoted_at_utc": utc_now_iso(),
                "staged_record_sha256": staged_hash,
                "staged_rows_sha256": rows_hash,
                "staging_attempt": (
                    str(archived.relative_to(root)) if archived is not None else None
                ),
            },
        )
        promoted += 1
        print(f"  [{doc_id}] promoted: inserted={outcome.inserted}", flush=True)

    for doc_id in doc_ids:
        with document_publication_lock(data_dir, year, doc_id):
            _promote_one(doc_id)

    return {
        "year": int(year),
        "promoted": promoted,
        "refused": refused,
        "failed": failed,
    }


# --------------------------------------------------------------------------
# status
# --------------------------------------------------------------------------


def staging_status(data_dir: str | Path = "data") -> list[dict]:
    """Per-year staged / promoted / error counts. No DB access."""
    promoted_by_year: dict[int, int] = {}
    for entry in read_promoted_journal(staging_root(data_dir)):
        try:
            promoted_by_year[int(entry["year"])] = (
                promoted_by_year.get(int(entry["year"]), 0) + 1
            )
        except (KeyError, TypeError, ValueError):
            continue
    years: dict[int, dict] = {}
    for year, record in iter_staging_records(data_dir):
        row = years.setdefault(
            year, {"year": year, "staged": 0, "success": 0, "no_txs": 0, "errors": 0}
        )
        row["staged"] += 1
        status = record.get("parse_status")
        key = "errors" if status == "error" else status
        if key in ("success", "no_txs", "errors"):
            row[key] += 1
    rows = []
    for year in sorted(set(years) | set(promoted_by_year)):
        row = years.get(
            year, {"year": year, "staged": 0, "success": 0, "no_txs": 0, "errors": 0}
        )
        row["promoted"] = promoted_by_year.get(year, 0)
        rows.append(row)
    return rows


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="glm_stage.py",
        description=(
            "Staged GLM OCR ingestion: GLM-only backlog candidates stay in "
            "data/staging/glm-backlog, excluded from canonical until an "
            "owner-approved pixel audit passes (FP review 2026-10-18)."
        ),
    )
    parser.add_argument("--data-dir", default="data", help="data directory (default: data)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_stage = subparsers.add_parser(
        "stage", help="GLM-read unread PDFs into staging records"
    )
    p_stage.add_argument("--year", type=int, required=True)
    p_stage.add_argument("--limit", type=int, default=25, help="max docs this batch")
    p_stage.add_argument(
        "--from-manifest",
        default=None,
        metavar="PATH",
        help=(
            "drain the AUTHORITATIVE unresolved-work manifest exported by "
            "ocr_zero_rows.py --export-work-manifest (finding 8) instead of "
            "scanning data/<year>/pdfs; entries without a local PDF are "
            "skipped with a note. --year filters the manifest"
        ),
    )
    p_stage.add_argument(
        "--check-canonical",
        action="store_true",
        help=(
            "also skip docs with a successful canonical parse run (opens "
            "data/congress.duckdb read-only; off by default because a "
            "refresh may hold the DB)"
        ),
    )
    p_stage.add_argument(
        "--dry-run",
        action="store_true",
        help="show manifest/canonical selection and skip counts without GLM calls",
    )
    p_stage.add_argument(
        "--cooldown", type=float, default=CLI_COOLDOWN, help="seconds between docs"
    )
    p_stage.add_argument(
        "--workers",
        type=int,
        default=MAX_WORKERS,
        help=(
            "concurrent vision reads; the Z.AI subscription allows "
            f"{MAX_WORKERS} streams and larger values are REFUSED "
            "(ValueError), not clamped"
        ),
    )

    p_promote = subparsers.add_parser(
        "promote",
        help=(
            "insert AUDITED staged rows into canonical (consumes the "
            "record; audit: pass is mandatory)"
        ),
    )
    p_promote.add_argument("--year", type=int, required=True)
    p_promote.add_argument(
        "--doc-id",
        nargs="*",
        default=None,
        help="specific docs (default: every staged doc for the year)",
    )
    p_promote.add_argument(
        "--require-audit-pass",
        action="store_true",
        default=True,
        help=(
            "audit approval is mandatory (default): requires audit: pass on "
            "each record AND a complete audit report with the budget met "
            "(audit-report-state.json); accepted for compatibility"
        ),
    )
    p_promote.add_argument(
        "--force-unaudited",
        action="store_true",
        help=(
            "operator override: promote records WITHOUT audit: pass and "
            "without a passing audit report (loud warning; output is not "
            "audited)"
        ),
    )
    p_promote.add_argument(
        "--db-path", default=None, help="canonical duckdb (default: <data-dir>/congress.duckdb)"
    )

    subparsers.add_parser("status", help="per-year staged/promoted/error counts")

    args = parser.parse_args(argv)
    data_dir = args.data_dir

    if args.command == "stage":
        manifest_items = None
        if args.from_manifest:
            manifest_items, manifest_error = load_work_manifest_items(
                args.from_manifest
            )
            if manifest_error:
                print(f"ERROR: {manifest_error}", file=sys.stderr)
                return 2
        try:
            result = stage_batch(
                args.year,
                args.limit,
                data_dir=data_dir,
                check_canonical=args.check_canonical,
                cooldown=args.cooldown,
                workers=args.workers,
                manifest=manifest_items,
                dry_run=args.dry_run,
            )
        except ValueError as exc:
            # Finding 11: a workers override above the 2-stream budget is a
            # hard refusal, not a clamp.
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        print(
            f"\n=== {'STAGE DRY RUN' if result.get('dry_run') else 'STAGE DONE'} "
            f"year={result['year']} "
            f"candidates={result['candidates']} staged={result['staged']} "
            f"errors={result['errors']} missing_pdfs={result.get('missing_pdfs', 0)} "
            f"missing_dirs={result.get('missing_dirs', 0)} "
            f"blocked_artifacts={result.get('blocked_artifacts', 0)} ==="
        )
        if result.get("missing_dirs") or result.get("blocked_artifacts"):
            return 2
        if result["quota_stop"]:
            if result.get("google_endpoint_saturated"):
                print(
                    "google endpoint saturated — stopping cleanly; error records "
                    "retry nightly. "
                    f"Remaining unstaged candidates: {result['remaining']}"
                )
            elif result.get("google_quota_dead"):
                print(
                    "Google quota dead for the session; stopping batch cleanly. "
                    f"Remaining unstaged candidates: {result['remaining']}"
                )
            else:
                print(
                    f"Quota exhausted; stopping batch cleanly. "
                    f"Remaining unstaged candidates: {result['remaining']}"
                )
        if result.get("fatal_stop"):
            print(
                "ALERT: auth/config defect stopped the batch; fix the GLM "
                f"endpoint configuration. Remaining candidates: "
                f"{result['remaining']}"
            )
            return 1
        return 0

    if args.command == "promote":
        result = promote_docs(
            args.year,
            args.doc_id,
            data_dir=data_dir,
            db_path=args.db_path,
            force_unaudited=args.force_unaudited,
        )
        print(
            f"\n=== PROMOTE DONE year={result['year']} promoted={result['promoted']} "
            f"refused={result['refused']} failed={result['failed']} ==="
        )
        return 0 if result["failed"] == 0 else 1

    if args.command == "status":
        rows = staging_status(data_dir)
        print(
            f"{'year':>6}  {'staged':>6}  {'success':>7}  {'no_txs':>6}  "
            f"{'errors':>6}  {'promoted':>8}"
        )
        for row in rows:
            print(
                f"{row['year']:>6}  {row['staged']:>6}  {row['success']:>7}  "
                f"{row['no_txs']:>6}  {row['errors']:>6}  {row['promoted']:>8}"
            )
        if not rows:
            print("(no staging records)")
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
