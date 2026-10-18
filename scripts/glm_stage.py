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

* ``stage   --year Y [--limit N] [--check-canonical]``
    Rasterize unread PDFs (pdftoppm -r 150 -png), read them with the GLM
    vision endpoint through the existing ``analyzer.vision_engines`` client
    (shared PROMPT, never forked), parse with ``parse_gemini_output``, and
    write staging records. Transient endpoint failures (5xx, timeout,
    transport, truncation) are retried in-process (2 retries, jittered
    backoff, cached page renders) before an error record is written; quota
    exhaustion stops the batch cleanly with exit 0; auth/config defects
    alert and stop the batch with exit 1; per-doc failures never kill the
    batch.
* ``promote --year Y [--doc-id ID ...] [--force-unaudited]``
    Insert AUDITED staged rows into canonical through the EXISTING
    ``ocr_zero_rows.insert_transactions`` path (parser_version
    ``v1-glm53-vision``, engines ``visionb``), mirror ``main()``'s parse-run
    bookkeeping, and delete the staging JSON (promotion = consumed) only
    when the FULL staged row set is confirmed committed — a partial or
    rejected insert keeps the record and stamps the rejection details on
    it.
* ``status``
    Per-year staged / promoted / error counts (staged counts need no DB).

Canonical (``data/congress.duckdb``) is only opened by ``promote`` (which
needs the write lock) and by ``stage --check-canonical`` (read-only, opt-in
because a refresh holds the DB).
"""

from __future__ import annotations

import argparse
import base64
import datetime
import json
import os
import random
import subprocess
import sys
import tempfile
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
    TransientVisionBError,
    VisionBConfig,
    VisionBUnavailable,
    classify_visionb_error,
    read_document_with_vision_b,
)
from scripts.audit_sample import (  # noqa: E402
    REPORT_STATE_NAME as AUDIT_REPORT_STATE_NAME,
    REPORT_STATE_SCHEMA as AUDIT_REPORT_STATE_SCHEMA,
)

GLM_PARSER_VERSION = "v1-glm53-vision"
GLM_ENGINES_ATTEMPTED = "visionb"
STAGING_SUBDIR = Path("staging") / "glm-backlog"
AUDIT_RESULTS_NAME = "audit-results.json"
PROMOTED_JOURNAL_NAME = "promoted.jsonl"
RENDER_DPI = 150
CLI_COOLDOWN = 3.0  # seconds between endpoint requests (mirrors ocr_zero_rows)
MAX_READ_RETRIES = 2  # bounded in-process retries for transient failures
BACKOFF_BASE_SECONDS = 2.0  # jittered exponential backoff base


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
    reusing the SAME rendered page images. Quota errors stop immediately
    (the batch stops); auth/config/permanent failures are raised untouched
    (no retry — the same request would fail again).

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
        except TransientVisionBError:
            if attempt >= max_retries:
                raise
            sleep_fn(backoff_seconds * (2 ** attempt) + jitter_fn(0.0, 1.0))
    raise RuntimeError("unreachable")  # pragma: no cover


def build_staging_record(
    doc_id: str,
    year: int,
    pdf_path: str | Path,
    config: VisionBConfig | None,
    *,
    read_fn=None,
) -> dict:
    """Read one document and build its staging record. Never raises.

    Parse/validation failures become ``parse_status: "error"`` records with
    ``error_kind`` carrying the classified failure (quota/auth/config/
    truncated/transient/permanent/parse) so the batch loop can distinguish
    a per-doc defect from a batch-wide stop.
    """
    record = {
        "doc_id": str(doc_id),
        "year": int(year),
        "pdf_sha256": None,
        "model": getattr(config, "model", None),
        "prompt_sha256": PROMPT_SHA256,
        "parser_version": GLM_PARSER_VERSION,
        "parsed_at_utc": utc_now_iso(),
        "pages": None,
        "member": None,
        "rows": [],
        "raw_output": None,
        "parse_status": "error",
        "error": None,
        "error_kind": None,
        "audit": None,
    }
    try:
        sha = pdf_sha256(pdf_path)
        page_count = pdf_page_count(pdf_path)
        record["pdf_sha256"] = sha
        record["pages"] = page_count
        if config is None and read_fn is None:
            raise RuntimeError("GLM vision endpoint is not configured")
        raw_output = read_document(
            pdf_path, config, read_fn=read_fn, page_count=page_count
        )
        parsed = parse_gemini_output(raw_output, expected_page_count=page_count)
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
            raw_output=raw_output,
            parse_status="success" if parsed.transactions else "no_txs",
            error=None,
            error_kind=None,
        )
    except Exception as exc:  # one bad doc must never kill the batch
        record["parse_status"] = "error"
        record["error"] = str(exc) or type(exc).__name__
        record["error_kind"] = classify_visionb_error(exc)
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
    """PDFs in data/<year>/pdfs with no staging record yet (sorted)."""
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
) -> dict:
    """Stage up to ``limit`` docs for ``year``. Never raises for endpoint states.

    Batch-wide stops: quota exhaustion (``quota_stop``, clean, exit 0 — the
    doc stays unstaged for the next run) and auth/config defects
    (``fatal_stop``, alerted — every remaining doc would fail the same way).
    Per-doc failures never kill the batch.

    ``workers > 1`` reads docs concurrently (the GLM call is I/O-bound, so
    threads overlap network latency); a batch-wide stop halts new
    submissions and lets in-flight docs finish. Records are per-doc files,
    so no ordering or locking beyond the counters is needed.
    """
    if config is None and read_fn is None:
        config = VisionBConfig.from_env()
        if config is None:
            raise RuntimeError(
                "GLM vision endpoint is not configured: set VISION_B_BASE_URL, "
                "VISION_B_MODEL (and VISION_B_API_KEY / VISION_B_TIMEOUT)"
            )
    candidates = select_candidates(
        year, limit, data_dir, check_canonical=check_canonical, db_path=db_path
    )
    staged = errors = 0
    quota_stop = False
    fatal_stop = False
    remaining = 0

    def _stage_one(doc_id: str) -> str:
        record = build_staging_record(
            doc_id,
            year,
            default_pdf_path(data_dir, year, doc_id),
            config,
            read_fn=read_fn,
        )
        error = record.get("error") or ""
        if record["parse_status"] == "error" and (
            record.get("error_kind") == "quota" or is_quota_error(error)
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
            if outcome == "quota":
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
            "fatal_stop": fatal_stop,
            "remaining": remaining,
        }

    import threading
    from concurrent.futures import (
        FIRST_COMPLETED,
        ThreadPoolExecutor,
        wait as futures_wait,
    )

    lock = threading.Lock()
    stop = threading.Event()
    processed = 0

    def _worker(doc_id: str, index: int) -> str:
        if stop.is_set():
            return "skipped"
        if cooldown and index:
            time.sleep(cooldown)
        outcome = _stage_one(doc_id)
        if outcome in ("quota", "fatal"):
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
                if outcome in ("quota", "fatal"):
                    with lock:
                        remaining = len(candidates) - processed
                        if outcome == "quota":
                            quota_stop = True
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
        "fatal_stop": fatal_stop,
        "remaining": remaining,
    }


# --------------------------------------------------------------------------
# promote
# --------------------------------------------------------------------------


def staged_row_to_tx(row) -> dict:
    """Convert a staged row [asset, type, tx_date, notif_date, amount_letter]
    into the transaction dict ``insert_transactions``/validation expects."""
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
        "page_number": None,
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


def audit_report_gate_open(state: dict | None) -> bool:
    """True only for a COMPLETE report whose material-error budget is met."""
    return (
        state is not None
        and state.get("state") == "complete"
        and state.get("budget_met") is True
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
                if state is None:
                    print(
                        "REFUSED: audit report state missing — run "
                        "audit_sample.py report "
                        f"({staging_root(data_dir) / AUDIT_REPORT_STATE_NAME})",
                        flush=True,
                    )
                else:
                    print(
                        "REFUSED: audit report state is "
                        f"{state.get('state')!r} with budget_met="
                        f"{state.get('budget_met')!r}; canonical promotion "
                        "must not proceed",
                        flush=True,
                    )
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
            transactions = [staged_row_to_tx(row) for row in record.get("rows", [])]
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
        default=2,
        help="concurrent vision reads (Z.AI subscription allows 2 streams)",
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
        try:
            result = stage_batch(
                args.year,
                args.limit,
                data_dir=data_dir,
                check_canonical=args.check_canonical,
                cooldown=args.cooldown,
                workers=max(1, args.workers),
            )
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        print(
            f"\n=== STAGE DONE year={result['year']} "
            f"candidates={result['candidates']} staged={result['staged']} "
            f"errors={result['errors']} ==="
        )
        if result["quota_stop"]:
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
