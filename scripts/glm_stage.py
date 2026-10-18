#!/usr/bin/env python3
"""Staged GLM OCR ingestion for the backlog drain (FP review 2026-10-18).

Per ``docs/reviews/glm-engine-fp-review-20261018.md`` §5, GLM-only backlog
candidates are STAGED: written as one JSON record per document under
``data/staging/glm-backlog/<year>/<doc_id>.json`` and explicitly excluded
from canonical/backtests until an owner-approved pixel audit passes
(``audit_sample.py``; promotion gate ``--require-audit-pass``).

Subcommands:

* ``stage   --year Y [--limit N] [--check-canonical]``
    Rasterize unread PDFs (pdftoppm -r 150 -png), read them with the GLM
    vision endpoint through the existing ``analyzer.vision_engines`` client
    (shared PROMPT, never forked), parse with ``parse_gemini_output``, and
    write staging records. Quota exhaustion stops the batch cleanly with
    exit 0; per-doc failures never kill the batch.
* ``promote --year Y [--doc-id ID ...] [--require-audit-pass]``
    Insert staged rows into canonical through the EXISTING
    ``ocr_zero_rows.insert_transactions`` path (parser_version
    ``v1-glm53-vision``, engines ``visionb``), mirror ``main()``'s parse-run
    bookkeeping, and delete the staging JSON (promotion = consumed).
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
    VisionBConfig,
    read_document_with_vision_b,
)

GLM_PARSER_VERSION = "v1-glm53-vision"
GLM_ENGINES_ATTEMPTED = "visionb"
STAGING_SUBDIR = Path("staging") / "glm-backlog"
AUDIT_RESULTS_NAME = "audit-results.json"
PROMOTED_JOURNAL_NAME = "promoted.jsonl"
RENDER_DPI = 150
CLI_COOLDOWN = 3.0  # seconds between endpoint requests (mirrors ocr_zero_rows)


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


def read_document(pdf_path: str | Path, config: VisionBConfig) -> str:
    """Render pages and read the whole document with the GLM vision endpoint.

    Uses the shared PROMPT via ``read_document_with_vision_b`` (never forked).
    Transport/HTTP/envelope failures raise ``VisionBUnavailable``.
    """
    images = rasterize_page_data_urls(pdf_path)
    return read_document_with_vision_b(images, config)


def build_staging_record(
    doc_id: str,
    year: int,
    pdf_path: str | Path,
    config: VisionBConfig | None,
    *,
    read_fn=None,
) -> dict:
    """Read one document and build its staging record. Never raises.

    Parse/validation failures become ``parse_status: "error"`` records.
    Quota exhaustion (429 / quota text) raises nothing either but is
    reported via ``parse_status: "error"`` plus a quota-shaped ``error`` —
    the batch loop checks it BEFORE persisting, so quota stops never write
    a per-doc error record (the doc simply stays unstaged for the next run).
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
        "audit": None,
    }
    try:
        sha = pdf_sha256(pdf_path)
        page_count = pdf_page_count(pdf_path)
        record["pdf_sha256"] = sha
        record["pages"] = page_count
        if read_fn is not None:
            raw_output = read_fn(pdf_path, page_count)
        else:
            if config is None:
                raise RuntimeError("GLM vision endpoint is not configured")
            raw_output = read_document(pdf_path, config)
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
        )
    except Exception as exc:  # one bad doc must never kill the batch
        record["parse_status"] = "error"
        record["error"] = str(exc) or type(exc).__name__
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
    """Stage up to ``limit`` docs for ``year``. Quota stop is clean (no raise).

    ``workers > 1`` reads docs concurrently (the GLM call is I/O-bound, so
    threads overlap network latency); quota stop halts new submissions and
    lets in-flight docs finish. Records are per-doc files, so no ordering or
    locking beyond the counters is needed.
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
    remaining = 0

    def _stage_one(doc_id: str) -> str:
        record = build_staging_record(
            doc_id,
            year,
            default_pdf_path(data_dir, year, doc_id),
            config,
            read_fn=read_fn,
        )
        if record["parse_status"] == "error" and is_quota_error(record["error"] or ""):
            # Quota exhaustion: not a doc defect. Write nothing, stop the
            # batch cleanly; the doc stays unstaged for the next run.
            return "quota"
        write_staging_record(data_dir, record)
        if record["parse_status"] == "error":
            print(f"  [{doc_id}] ERROR: {record['error']}", flush=True)
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
        if outcome == "quota":
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
                if outcome == "quota":
                    quota_stop = True
                    with lock:
                        remaining = len(candidates) - processed
                    for other in pending:
                        other.cancel()
                    pending = {f for f in pending if not f.cancelled()}
        if quota_stop:
            with lock:
                remaining = max(remaining, len(candidates) - processed)
    return {
        "year": year,
        "candidates": len(candidates),
        "staged": staged,
        "errors": errors,
        "quota_stop": quota_stop,
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
    require_audit_pass: bool = False,
    cooldown_note: str | None = None,
) -> dict:
    """Promote staged records into canonical via the EXISTING insert path.

    ``insert_transactions`` (imported from ``scripts.ocr_zero_rows``) writes
    the transactions AND the pdf_parse_runs record transactionally, exactly
    as the OCR path does; afterwards this mirrors ``main()``'s remaining
    bookkeeping (progress file) and deletes the staging JSON — promotion
    consumes the staging record. Refusals leave the record untouched.
    """
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
        if require_audit_pass and record.get("audit") != "pass":
            print(
                f"  [{doc_id}] refused: --require-audit-pass and staging "
                "record lacks audit: pass",
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
            inserted = ocr.insert_transactions(
                doc_id,
                year,
                record.get("member"),
                transactions,
                db_path=db_path,
                parser_version=GLM_PARSER_VERSION,
                raw_count=len(transactions),
                artifact_sha256=record["pdf_sha256"],
                ingestion_generation=ingestion_generation,
                engine_model=GLM_ENGINES_ATTEMPTED,
            )
        except Exception as exc:
            # duckdb write-lock and generation-binding problems surface
            # naturally; the staging record is kept for a later attempt.
            print(f"  [{doc_id}] ERROR: {exc}", flush=True)
            failed += 1
            continue

        if record.get("parse_status") == "no_txs":
            ok, progress_key = True, "no_txs"
        elif inserted > 0:
            ok, progress_key = True, "completed"
        else:
            ok, progress_key = False, "errors"
        # Mirror main()'s run bookkeeping (the parse-run row itself was
        # already written transactionally inside insert_transactions).
        ocr.mark_progress(progress, doc_id, progress_key)
        ocr.save_progress(progress, progress_path)
        if not ok:
            print(
                f"  [{doc_id}] ERROR: validated rows were not inserted "
                "(staging record kept)",
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
                "inserted": inserted,
                "parser_version": GLM_PARSER_VERSION,
                "promoted_at_utc": utc_now_iso(),
            },
        )
        promoted += 1
        print(f"  [{doc_id}] promoted: inserted={inserted}", flush=True)

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
        "promote", help="insert staged rows into canonical (consumes the record)"
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
        help="refuse docs whose staging record lacks audit: pass",
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
        return 0

    if args.command == "promote":
        result = promote_docs(
            args.year,
            args.doc_id,
            data_dir=data_dir,
            db_path=args.db_path,
            require_audit_pass=args.require_audit_pass,
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
