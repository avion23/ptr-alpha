#!/usr/bin/env python3
"""Staged GLM OCR ingestion for the backlog drain (FP review 2026-10-18).

Per ``docs/reviews/glm-engine-fp-review-20261018.md`` §5, GLM-only backlog
candidates are STAGED: written as one JSON record per document under
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
    bookkeeping, and delete the staging JSON (promotion = consumed) only
    when the FULL staged row set is confirmed committed — a partial or
    rejected insert keeps the record and stamps the rejection details on
    it. A promoted doc lands a success/no_txs ``v1-glm53-vision`` parse
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
newest live record keeps the canonical name, at most MAX_RETAINED_ATTEMPTS
retained — the attempted archive subdirectory keeps it out of the
record-counting globs in doctor.py/audit_sample.py). ``audit record``
stores the record identity (``pdf_sha256 + prompt_sha256 +
parsed_at_utc``) in its audit-results entry, and ``promote`` refuses any
audit-pass record whose live identity no longer matches the audited
identity (recipe changed after audit → re-audit).

Concurrency (FP review finding 11): ``--workers`` is capped at
MAX_WORKERS (2 — the Z.AI subscription allows two concurrent vision
streams; larger values raise ValueError instead of being clamped).
Cross-process exclusion is the cron wrapper's single-instance flock
(``cron/backlog_ocr.sh`` holds ``data/logs/ocr-backlog-cron/drain.lock``
via ``flock -n``); no in-process lock is taken.
"""

from __future__ import annotations

import argparse
import base64
import datetime
import glob as glob_module
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
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(REPO_ROOT / "src"), str(REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from scripts.gemini_ocr_common import (  # noqa: E402
    AMOUNT_MIDPOINTS,
    PROMPT,
    PROMPT_SHA256,
    GeminiOutputError,  # noqa: F401  (re-exported for callers/tests)
    is_quota_error,
    parse_gemini_output,
    pdf_page_count,
    pdf_sha256,
)
from analyzer.vision_engines import (  # noqa: E402
    GoogleVisionBRateLimitError,
    GoogleVisionBServiceUnavailableError,
    TransientVisionBError,
    VisionBConfig,
    VisionBUnavailable,
    classify_visionb_error,
    is_google_api_host,
    read_document_with_vision_b,
)
from scripts.audit_sample import (  # noqa: E402
    REPORT_STATE_NAME as AUDIT_REPORT_STATE_NAME,
    REPORT_STATE_SCHEMA as AUDIT_REPORT_STATE_SCHEMA,
    load_audit_results as load_audit_results,
)

GLM_PARSER_VERSION = "v1-glm53-vision"
GLM_ENGINES_ATTEMPTED = "visionb"
STAGING_SUBDIR = Path("staging") / "glm-backlog"
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
MAX_RETAINED_ATTEMPTS = 3  # archived re-stage attempts kept per doc
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


def staging_root(data_dir: str | Path = "data") -> Path:
    return Path(data_dir) / STAGING_SUBDIR


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


def _atomic_write_json(path: Path, payload) -> None:
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
    return staging_root(data_dir) / str(year) / f"{doc_id}.json"


def attempts_dir(data_dir: str | Path, year: int) -> Path:
    """Archived (superseded) staging attempts for a year.

    A subdirectory rather than the year directory itself: doctor.py and
    audit_sample.py count ``<year>/*.json`` as LIVE records; archived
    attempts are historical evidence and must not inflate those counts.
    """
    return staging_root(data_dir) / str(year) / "attempts"


def archived_attempts(data_dir: str | Path, year: int, doc_id: str) -> list[tuple[int, Path]]:
    """(attempt_number, path) for a doc's archived attempts, oldest first."""
    directory = attempts_dir(data_dir, year)
    if not directory.exists():
        return []
    pairs = []
    prefix = f"{glob_module.escape(doc_id)}.attempt-"
    for path in directory.glob(f"{prefix}*.json"):
        suffix = path.name[len(doc_id) + len(".attempt-") : -len(".json")]
        if suffix.isdigit():
            pairs.append((int(suffix), path))
    return sorted(pairs)


def archive_current_attempt(data_dir: str | Path, year: int, doc_id: str) -> Path | None:
    """Versioned re-stage (finding 6): archive the live record, never overwrite.

    Renames the existing ``<doc_id>.json`` to
    ``attempts/<doc_id>.attempt-<n>.json`` so the newly written record keeps
    the canonical name and the superseded reading remains inspectable
    evidence. n is the prior attempt count (max existing archive number + 1,
    which equals the prior attempt count until the retention cap prunes the
    oldest — after pruning the numbering keeps growing instead of colliding
    with a retained archive). At most MAX_RETAINED_ATTEMPTS archives are
    retained (oldest deleted). Returns the archive path, or None when no
    live record existed.
    """
    live = staging_record_path(data_dir, year, doc_id)
    if not live.exists():
        return None
    prior = archived_attempts(data_dir, year, doc_id)
    next_number = (prior[-1][0] + 1) if prior else 0
    target = attempts_dir(data_dir, year) / f"{doc_id}.attempt-{next_number}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(live, target)
    retained = archived_attempts(data_dir, year, doc_id)
    for _, stale in retained[: max(len(retained) - MAX_RETAINED_ATTEMPTS, 0)]:
        stale.unlink(missing_ok=True)
    return target


def write_staging_record(data_dir: str | Path, record: dict) -> Path:
    path = staging_record_path(data_dir, record["year"], record["doc_id"])
    _atomic_write_json(path, record)
    return path


def load_staging_record(data_dir: str | Path, year: int, doc_id: str) -> dict | None:
    path = staging_record_path(data_dir, year, doc_id)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def find_staging_record(data_dir: str | Path, doc_id: str) -> tuple[int, dict] | None:
    """Locate a staging record across year directories (audit tooling)."""
    root = staging_root(data_dir)
    for year_dir in sorted(root.glob("[0-9]" * 4)):
        path = year_dir / f"{doc_id}.json"
        if path.exists():
            return int(year_dir.name), json.loads(path.read_text(encoding="utf-8"))
    return None


def iter_staging_records(data_dir: str | Path) -> list[tuple[int, dict]]:
    """All staging records, ordered (year, doc_id)."""
    records = []
    root = staging_root(data_dir)
    for year_dir in sorted(root.glob("[0-9]" * 4), key=lambda p: p.name):
        for path in sorted(year_dir.glob("*.json")):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
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
) -> dict:
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
    record = {
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
        pdf_bytes = Path(pdf_path).read_bytes()
        record["pdf_sha256"] = hashlib.sha256(pdf_bytes).hexdigest()
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
    success/no_txs is skipped (it is staged and awaiting the audit; re-
    reading it nightly would burn quota); an entry whose live record is an
    ERROR record IS retried — the authoritative queue still lists the doc
    as unresolved, and the re-stage archives the failed attempt as
    versioned evidence instead of overwriting it.
    """
    skip: set[str] = set()
    year_dir = staging_root(data_dir) / str(year)
    if year_dir.exists():
        for path in year_dir.glob("*.json"):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(record, dict):
                continue
            doc_id = str(record.get("doc_id") or path.stem)
            if record.get("parse_status") == "error":
                continue  # retryable: the queue still wants this doc
            skip.add(doc_id)
    if check_canonical:
        skip |= canonical_parsed_doc_ids(db_path or default_db_path(data_dir))
    candidates: list[str] = []
    notes: list[str] = []
    seen: set[str] = set()
    for item in items:
        try:
            item_year = int(item["year"])
        except (KeyError, TypeError, ValueError):
            continue
        if item_year != int(year):
            continue
        doc_id = str(item["doc_id"])
        if doc_id in seen or doc_id in skip:
            continue
        seen.add(doc_id)
        pdf_path = default_pdf_path(data_dir, year, doc_id)
        if not pdf_path.exists():
            notes.append(
                f"[{doc_id}] manifest entry has no local PDF "
                f"({pdf_path}); skipped"
            )
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
    submissions and lets in-flight docs finish. Records are per-doc files,
    so no ordering or locking beyond the counters is needed.

    Concurrency caps (finding 11): ``workers`` above MAX_WORKERS (2) raises
    ValueError — the Z.AI subscription allows two concurrent vision
    streams, and an operator override above that is a quota-budget
    violation, not a tuning knob. Cross-process exclusion is the cron
    wrapper's single-instance flock (``cron/backlog_ocr.sh`` holds
    ``data/logs/ocr-backlog-cron/drain.lock``); no in-process lock is
    taken because a single process is bounded by ``workers`` and overlaps
    across processes are excluded by the wrapper.

    ``manifest`` (``ocr_zero_rows.export_work_manifest`` items) replaces
    the PDF-directory scan with the authoritative unresolved selection
    (finding 8): see ``select_candidates_from_manifest``. A doc whose live
    record is an error record is re-staged; the superseded record is
    archived (``archive_current_attempt``) before the new one is written.
    """
    if not 1 <= int(workers) <= MAX_WORKERS:
        raise ValueError(
            f"workers={workers} refused: the Z.AI subscription allows "
            f"{MAX_WORKERS} concurrent vision streams; values above "
            f"{MAX_WORKERS} violate the quota budget and are not accepted"
        )
    if config is None and read_fn is None:
        config = VisionBConfig.from_env()
        if config is None:
            raise RuntimeError(
                "GLM vision endpoint is not configured: set VISION_B_BASE_URL, "
                "VISION_B_MODEL (and VISION_B_API_KEY / VISION_B_TIMEOUT)"
            )
    if manifest is None:
        candidates = select_candidates(
            year, limit, data_dir, check_canonical=check_canonical, db_path=db_path
        )
        missing_pdfs = 0
    else:
        candidates, notes = select_candidates_from_manifest(
            manifest,
            year,
            limit,
            data_dir,
            check_canonical=check_canonical,
            db_path=db_path,
        )
        missing_pdfs = len(notes)
        for note in notes:
            print(f"  {note}", flush=True)
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
        record = build_staging_record(
            doc_id,
            year,
            default_pdf_path(data_dir, year, doc_id),
            config,
            read_fn=read_fn,
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
        archived = archive_current_attempt(data_dir, year, doc_id)
        if archived is not None:
            print(
                f"  [{doc_id}] prior staging record archived: "
                f"attempts/{archived.name}",
                flush=True,
            )
        write_staging_record(data_dir, record)
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
            else:
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


def _audit_report_gate_issue(state: dict | None) -> str | None:
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
    return None


def audit_report_gate_open(state: dict | None) -> bool:
    """True only when a complete report proves a met budget with valid metrics."""
    return _audit_report_gate_issue(state) is None


def _audit_report_refusal(state: dict | None, data_dir: str | Path) -> str:
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
                _audit_report_gate_issue(state)
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


def record_identity(record: dict) -> dict:
    """The (pdf_sha256, prompt_sha256, parsed_at_utc) triple a verdict is
    bound to (finding 6): it changes iff the artifact, extraction recipe, or
    staging attempt changed."""
    return {field: record.get(field) for field in AUDIT_IDENTITY_FIELDS}


def verdict_binding(
    data_dir: str | Path, doc_id: str, record: dict
) -> tuple[bool, str | None]:
    """(bound, refusal_reason) between the audit history and THIS record.

    ``audit_sample.record_verdict`` stores the audited record's identity in
    its audit-results entry; promotion requires a pass entry whose identity
    equals the live record's. A mismatch means the record was re-staged (or
    the recipe/artifact changed) after the audit — the pass verdict says
    nothing about the live attempt, so promotion is refused (re-audit).
    Entries written before identity binding existed carry no identity and
    can never satisfy the check: fail-closed.
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
    for entry in passes:
        entry_identity = {field: entry.get(field) for field in AUDIT_IDENTITY_FIELDS}
        if entry_identity == identity and all(
            value is not None for value in entry_identity.values()
        ):
            return True, None
    return False, (
        "refused: the audit pass verdict is not bound to this record's "
        "identity (pdf_sha256/prompt_sha256/parsed_at_utc) — the record "
        "was re-staged after the audit or the verdict predates identity "
        "binding; re-audit the current attempt"
    )


def append_promoted_journal(root: Path, entry: dict) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with (root / PROMOTED_JOURNAL_NAME).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")


def read_promoted_journal(root: Path) -> list[dict]:
    path = root / PROMOTED_JOURNAL_NAME
    if not path.exists():
        return []
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
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
    """Promote AUDITED staged records into canonical via the insert path.

    Audit approval is MANDATORY at this boundary (FP review finding 1): a
    record without ``audit == "pass"`` is refused unless
    ``force_unaudited=True`` (the operator's explicit override, printed
    with a loud warning). ``require_audit_pass=False`` is refused loudly —
    there is no permissive default anymore.

    ``insert_transactions`` (imported from ``scripts.ocr_zero_rows``) writes
    the transactions AND the pdf_parse_runs record transactionally, exactly
    as the OCR path does. The staging JSON is deleted — promotion consumes
    the record — ONLY when the FULL staged row set is confirmed committed
    (outcome complete: committed, no rejections, count matches). A partial
    or rejected insert keeps the record and stamps the rejection details on
    it; a member rejection is never mistaken for a no_txs.

    Verdict binding (finding 6): an ``audit: pass`` stamp is only accepted
    when the audit-results history contains a pass entry bound to THIS
    record's identity (pdf_sha256 + prompt_sha256 + parsed_at_utc). A
    record re-staged after its audit (or a verdict predating identity
    binding) is refused — the audited attempt is not the live one.
    """
    if not require_audit_pass:
        raise ValueError(
            "audit approval is mandatory: require_audit_pass=False is no "
            "longer accepted (pass audit:'pass' records or force_unaudited)"
        )
    from scripts import ocr_zero_rows as ocr  # attribute lookups stay monkeypatchable

    db_path = str(db_path or default_db_path(data_dir))
    root = staging_root(data_dir)
    progress_path = Path(data_dir) / Path(ocr.PROGRESS_PATH).name
    progress = ocr.load_progress(progress_path)

    if not doc_ids:
        doc_ids = [
            record["doc_id"]
            for yr, record in iter_staging_records(data_dir)
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
        if not audit_report_gate_open(state):
            if doc_ids:
                print(_audit_report_refusal(state, data_dir), flush=True)
                return {
                    "year": int(year),
                    "promoted": 0,
                    "refused": len(doc_ids),
                    "failed": 0,
                }

    promoted = refused = failed = 0
    for doc_id in doc_ids:
        record = load_staging_record(data_dir, year, doc_id)
        if record is None:
            print(f"  [{doc_id}] no staging record; skipped", flush=True)
            refused += 1
            continue
        if record.get("parse_status") == "error":
            print(f"  [{doc_id}] refused: staging record is an error record", flush=True)
            refused += 1
            continue
        if record.get("audit") != "pass" and not force_unaudited:
            print(
                f"  [{doc_id}] refused: audit approval is mandatory and the "
                "staging record lacks audit: pass (run the pixel audit or "
                "pass --force-unaudited explicitly)",
                flush=True,
            )
            refused += 1
            continue
        if not force_unaudited:
            bound, reason = verdict_binding(data_dir, doc_id, record)
            if not bound:
                print(f"  [{doc_id}] {reason}", flush=True)
                refused += 1
                continue
        pdf_path = default_pdf_path(data_dir, year, doc_id)
        try:
            current_sha = pdf_sha256(pdf_path)
        except OSError as exc:
            print(f"  [{doc_id}] refused: PDF unreadable ({exc})", flush=True)
            refused += 1
            continue
        if current_sha != record.get("pdf_sha256"):
            print(
                f"  [{doc_id}] refused: staged pdf_sha256 no longer matches "
                "the artifact on disk",
                flush=True,
            )
            refused += 1
            continue
        try:
            rows = record.get("rows", [])
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
                db_path, doc_id, year, record["pdf_sha256"]
            )
            staged_count = len(transactions)
            report: dict = {}
            ocr.insert_transactions(
                doc_id,
                year,
                record.get("member"),
                transactions,
                db_path=db_path,
                parser_version=GLM_PARSER_VERSION,
                raw_count=staged_count,
                artifact_sha256=record["pdf_sha256"],
                ingestion_generation=ingestion_generation,
                engine_model=GLM_ENGINES_ATTEMPTED,
                report=report,
            )
            outcome = ocr.insert_outcome_from_report(
                report,
                inserted=(
                    report["outcome"].inserted
                    if isinstance(report.get("outcome"), ocr.InsertOutcome)
                    else 0
                ),
                expected=staged_count,
            )
        except Exception as exc:
            # duckdb write-lock and generation-binding problems surface
            # naturally; the staging record is kept for a later attempt.
            print(f"  [{doc_id}] ERROR: {exc}", flush=True)
            failed += 1
            continue

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
            continue
        staging_record_path(data_dir, year, doc_id).unlink(missing_ok=True)
        append_promoted_journal(
            root,
            {
                "doc_id": str(doc_id),
                "year": int(year),
                "inserted": outcome.inserted,
                "outcome_status": outcome.status,
                "unaudited": bool(force_unaudited),
                "parser_version": GLM_PARSER_VERSION,
                "promoted_at_utc": utc_now_iso(),
            },
        )
        promoted += 1
        print(f"  [{doc_id}] promoted: inserted={outcome.inserted}", flush=True)

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
            f"\n=== STAGE DONE year={result['year']} "
            f"candidates={result['candidates']} staged={result['staged']} "
            f"errors={result['errors']} ==="
        )
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
