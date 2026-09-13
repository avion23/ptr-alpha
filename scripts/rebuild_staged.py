"""Staged authoritative congressional database rebuild driver.

Builds a fresh, authoritative congressional disclosure database under
``data/.staging/rebuild/<generation>/`` using only the accepted production
components (HouseTransactionSource archive fetch/parse, SenateEFDSource eFD
sweep + Database.persist_source_refresh, YFinancePriceSource acquisition,
PriceSnapshot).  The canonical ``data/congress.duckdb`` is never opened.

Fail-closed rules enforced here:
  * House: a generation is activated (parse_status='complete') only when every
    acquired PDF has a terminal parse run; unresolved PDFs (OCR/parser
    failures) keep the generation incomplete and are listed exactly.
  * The merged cascade raises ParserCascadeError per unresolved PDF, so this
    driver runs the cascade per PDF with exception capture and records the
    failure as an ``error`` parse run instead of aborting the whole year.
  * Senate: a sweep that reports any failed/unavailable filing is quarantined
    (inventoried with exact counts) and nothing is persisted.
  * Local OCR only (PTR_SKIP_DOCLING=1, Tesseract). No paid API calls.

Every stage writes the generation manifest (every artifact SHA, per-source
outcome counts, unresolved lists, invariant results, verdict).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import date
from multiprocessing import Pool
from pathlib import Path

import duckdb

# Docling is disabled exactly as in the accepted production reparse flow; the
# cascade still performs full text-engine comparison with Tesseract OCR as the
# local fallback. Must be set before analyzer imports.
os.environ["PTR_SKIP_DOCLING"] = "1"

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT))

from analyzer.database import Database  # noqa: E402
from analyzer.download import (  # noqa: E402
    _PARSE_FAILURE_PREFIX,
    _PARSE_VERSION,
    _build_member_lookup,
    _engine_error_detail,
    _filter_existing_pdfs,
    _invalid_consolidated_house_docs,
    _tolerant_parse_pdf_worker as _production_tolerant_parse_worker,
    _validated_pdf_sha256,
    preserve_existing_fields,
)
from analyzer.models import FilingType  # noqa: E402
from analyzer import parser_cascade as _parser_cascade  # noqa: E402
from analyzer.parsing import consolidate_transactions  # noqa: E402
from analyzer.price_repository import previous_nyse_session  # noqa: E402
from analyzer.price_snapshot import create_snapshot, save_snapshot  # noqa: E402
from analyzer.settings import DataSettings, Settings  # noqa: E402
from analyzer.source_report_repository import (  # noqa: E402
    SOURCE_REPORT_INPUT_COLUMNS,
    SourceReportOutcome,
    _is_official_paper_url,
)

_parse_pdf_worker = _parser_cascade._parse_pdf_worker
ParserCascadeError = _parser_cascade.ParserCascadeError

HOUSE_YEARS = list(range(2015, date.today().year + 1))
# Production requires a complete current House generation. Historical House
# completeness is required only for the years consumed by the fixed validation
# window, plus the immediately prior year needed by a cross-year 28-day live
# disclosure window. Older archives may remain staged/incomplete without
# blocking promotion because they are not production inputs.
VALIDATION_HOUSE_YEARS = tuple(range(2021, 2026))
REQUIRED_HOUSE_YEARS = tuple(
    sorted(
        set(VALIDATION_HOUSE_YEARS)
        | {max(2015, date.today().year - 1), date.today().year}
    )
)
SENATE_START = date(2024, 1, 1)
PRICE_START = date(2014, 1, 1)


def _sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _database_fingerprint(path: Path) -> dict[str, int | str]:
    stat = path.stat()
    return {
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": _sha256_file(path),
    }


def _checkpoint_database(path: Path) -> None:
    conn = duckdb.connect(str(path))
    try:
        conn.execute("CHECKPOINT")
    finally:
        conn.close()


def _staging_root() -> Path:
    return _REPO_ROOT / "data" / ".staging" / "rebuild"


def _latest_generation() -> str | None:
    root = _staging_root()
    if not root.exists():
        return None
    candidates = sorted(
        (entry.name for entry in root.iterdir() if entry.is_dir()),
        reverse=True,
    )
    return candidates[0] if candidates else None


def _load_manifest(staging: Path) -> dict:
    path = staging / "manifest.json"
    if path.exists():
        return json.loads(path.read_text())
    return {}


def _save_manifest(staging: Path, manifest: dict) -> None:
    path = staging / "manifest.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str))
    tmp.replace(path)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _git_sha() -> str:
    try:
        import subprocess  # noqa: PLC0415

        result = subprocess.run(  # noqa: S603
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=_REPO_ROOT,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def _fresh_database(staging: Path) -> Database:
    db_path = staging / "congress.duckdb"
    if db_path.exists():
        db_path.unlink()
    return Database(db_path)


def _settings_for(staging: Path) -> Settings:
    return Settings(data=DataSettings(data_dir=str(staging)))


# --------------------------------------------------------------------------
# House
# --------------------------------------------------------------------------

def _house_fetch_year(
    src, year: int, manifest: dict, staging: Path
) -> dict:
    summary = src.fetch_and_cache_pdfs(year, refresh_metadata=True)
    gen = src.db.get_latest_house_generation(year)
    if gen is None:
        raise RuntimeError(f"house fetch {year}: no generation recorded")
    artifacts = src.db.conn.execute(
        """
        SELECT doc_id, artifact_sha256, http_status, etag, last_modified,
               content_length
        FROM house_pdf_artifacts
        WHERE archive_year = ? AND generation_id = ?
        ORDER BY doc_id
        """,
        [year, gen],
    ).fetchall()
    generation_row = src.db.conn.execute(
        """
        SELECT metadata_sha256, metadata_http_status, metadata_etag,
               metadata_last_modified, metadata_count, ptr_count
        FROM house_archive_generations
        WHERE archive_year = ? AND generation_id = ?
        """,
        [year, gen],
    ).fetchone()
    entry = {
        "archive_year": year,
        "generation_id": gen,
        "metadata_sha256": generation_row[0],
        "metadata_http_status": generation_row[1],
        "metadata_etag": generation_row[2],
        "metadata_last_modified": generation_row[3],
        "metadata_count": int(generation_row[4]),
        "ptr_count": int(generation_row[5]),
        "downloaded_count": summary.downloaded_count,
        "skipped_count": summary.skipped_count,
        "orphan_pdf_count": summary.orphan_pdf_count,
        "removed_doc_count": summary.removed_doc_count,
        "quarantined_pdf_count": summary.quarantined_pdf_count,
        "parse_status": "incomplete",
        "resolved_doc_count": 0,
        "unresolved_doc_ids": [],
        "artifact_count": len(artifacts),
        "artifacts": {
            str(doc_id): {
                "artifact_sha256": artifact_sha256,
                "http_status": http_status,
                "etag": etag,
                "last_modified": last_modified,
                "content_length": content_length,
            }
            for doc_id, artifact_sha256, http_status, etag, last_modified, content_length in artifacts
        },
    }
    manifest.setdefault("house", {})[str(year)] = entry
    _save_manifest(staging, manifest)
    return entry


def house_fetch(args) -> None:
    staging = Path(args.staging)
    manifest = _load_manifest(staging)
    years = [int(y) for y in args.years] if args.years else HOUSE_YEARS
    src = None
    try:
        src = _house_source(staging)
        for year in years:
            existing = manifest.get("house", {}).get(str(year))
            if existing and not args.force:
                print(f"house-fetch {year}: skipped (already fetched {existing.get('generation_id')})")
                continue
            entry = _house_fetch_year(src, year, manifest, staging)
            print(
                f"house-fetch {year}: generation={entry['generation_id']} "
                f"metadata={entry['metadata_count']} ptr={entry['ptr_count']} "
                f"artifacts={entry['artifact_count']}"
            )
    finally:
        if src is not None:
            src.close()


def _house_source(staging: Path):
    from analyzer.download import HouseTransactionSource  # noqa: PLC0415

    return HouseTransactionSource(_settings_for(staging))


def _tolerant_parse_worker(pdf_path: Path):
    """Use the production worker while retaining the script's test hook."""
    return _production_tolerant_parse_worker(pdf_path, _parse_pdf_worker)


_TEXT_PASS_BUDGET_SECONDS = 30


def _pdf_page_count_hint(pdf_path: Path) -> int:
    """Return a cheap page-count scheduling hint; unreadable PDFs sort last."""
    try:
        probe = subprocess.run(  # noqa: S603
            ["pdfinfo", str(pdf_path)],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return sys.maxsize
    if probe.returncode != 0:
        return sys.maxsize
    for line in probe.stdout.splitlines():
        if not line.startswith("Pages:"):
            continue
        try:
            pages = int(line.split(":", 1)[1].strip())
        except ValueError:
            return sys.maxsize
        return pages if pages > 0 else sys.maxsize
    return sys.maxsize


def _tolerant_text_parse_worker(pdf_path: Path):
    """Bound the trusted-text pass and defer every uncertain result to OCR."""
    return _production_tolerant_parse_worker(
        pdf_path,
        _parser_cascade._parse_text_only_worker,
        budget_seconds=_TEXT_PASS_BUDGET_SECONDS,
    )


def _persist_house_parse_batch(
    db: Database,
    *,
    year: int,
    ingestion_generation: str,
    member_lookup: dict,
    artifact_hashes: dict[str, str],
    results: list,
) -> dict:
    """Persist one bounded House parse batch and return aggregate telemetry."""
    pdf_transactions: dict = {}
    raw_counts: dict[str, int] = {}
    parse_attempts: list[tuple[str, list[str], str | None]] = []
    for pdf_path, transactions, engines_attempted in results:
        doc_id = pdf_path.stem
        pdf_transactions[pdf_path] = transactions
        raw_counts[doc_id] = len(transactions)
        error_message = _engine_error_detail(engines_attempted)
        parse_attempts.append(
            (
                doc_id,
                [
                    engine
                    for engine in engines_attempted
                    if not engine.startswith(_PARSE_FAILURE_PREFIX)
                ],
                error_message,
            )
        )

    df = consolidate_transactions(pdf_transactions, member_lookup)
    invalid_docs = _invalid_consolidated_house_docs(df)
    batch_failure = invalid_docs.pop("<batch>", None)
    if batch_failure:
        invalid_docs = {doc_id: batch_failure for doc_id, _, _ in parse_attempts}
    if invalid_docs and not df.empty:
        df = df[~df["doc_id"].astype(str).isin(invalid_docs)].copy()
    transaction_counts = (
        df["doc_id"].astype(str).value_counts().to_dict() if not df.empty else {}
    )
    if not df.empty:
        df["chamber"] = "house"
        df["ingestion_generation"] = ingestion_generation
        df["source_record_id"] = df["doc_id"].astype(str)
        df["official_filing_date"] = df["disclosure_date"]
        df["artifact_sha256"] = df["doc_id"].astype(str).map(artifact_hashes)
        if "asset_description" in df.columns:
            df["raw_asset_description"] = df["asset_description"]
    df = preserve_existing_fields(df, db)

    parse_runs = []
    for doc_id, engines_attempted, error_message in parse_attempts:
        count = transaction_counts.get(doc_id, 0)
        if error_message is not None:
            status = "error"
        elif doc_id in invalid_docs:
            status = "error"
            error_message = "; ".join(invalid_docs[doc_id])
        elif count:
            status = "success"
        else:
            status = "zero_rows"
        parse_runs.append(
            dict(
                doc_id=doc_id,
                year=year,
                parser_version=_PARSE_VERSION,
                status=status,
                engines_attempted=",".join(engines_attempted) or "cascade-failed",
                raw_row_count=raw_counts.get(doc_id, 0),
                transaction_count=0,
                error_message=error_message,
                artifact_sha256=artifact_hashes.get(doc_id),
                ingestion_generation=ingestion_generation,
            )
        )

    attempted_doc_ids = [doc_id for doc_id, _, _ in parse_attempts]
    replacement_doc_ids = (
        df["doc_id"].astype(str).unique().tolist() if not df.empty else []
    )
    persisted = db.replace_transactions_for_docs(
        df,
        source="house_pdf",
        attempted_doc_ids=attempted_doc_ids,
        ingestion_generation=ingestion_generation,
        replacement_doc_ids=replacement_doc_ids,
        parse_runs=parse_runs,
    )
    by_status: dict[str, int] = {}
    for run in parse_runs:
        by_status[run["status"]] = by_status.get(run["status"], 0) + 1
    return {
        "attempted": len(parse_attempts),
        "parse_run_statuses": by_status,
        "persisted_transactions": sum(persisted.by_doc_total.values()),
    }


def _parse_house_year_tolerant(staging: Path, db: Database, year: int) -> dict:
    """Mirror HouseTransactionSource.parse_cached_pdfs with per-PDF quarantine."""
    src = _house_source(staging)
    try:
        ingestion_generation = db.get_latest_house_generation(year)
        if ingestion_generation is None:
            raise RuntimeError(f"house-parse {year}: no acquired generation")
        metadata = src.fetch_metadata(year)
        ptrs = metadata[metadata["FilingType"] == FilingType.PTR.value]
        pdf_dir = staging / str(year) / "pdfs"
        pdf_paths, existing_docs = _filter_existing_pdfs(ptrs, pdf_dir)
        if not pdf_paths:
            raise RuntimeError(f"house-parse {year}: no PDF files found in {pdf_dir}")

        # PDFs failing validation carry no trustworthy hash and must never
        # become terminal. Selection is semantic rather than parser-version
        # specific so a validated Gemini/no_txs result is not redundantly sent
        # back through the deterministic cascade.
        artifact_hashes = {
            path.stem: sha
            for path in pdf_paths
            if (sha := _validated_pdf_sha256(path)) is not None
        }
        unresolved_doc_ids = set(
            db.get_unresolved_house_doc_ids(year, ingestion_generation)
        )
        keep_mask = (
            existing_docs["DocID"]
            .astype(str)
            .map(lambda doc_id: doc_id in unresolved_doc_ids)
            .to_numpy()
        )
        skipped_terminal = len(pdf_paths) - int(keep_mask.sum())
        pdf_paths = [path for path, keep in zip(pdf_paths, keep_mask) if keep]
        existing_docs = (
            existing_docs[keep_mask]
            if len(keep_mask)
            else existing_docs.iloc[0:0]
        )
        if not pdf_paths:
            return {"attempted": 0, "skipped_cached": skipped_terminal}

        # Terminal successes/no_txs were removed above. Previously attempted
        # nonterminal documents have already failed the full cascade, so do not
        # spend the bounded text pass on them again. Every unseen document gets
        # a short text-only chance before any OCR-capable work starts. This
        # prevents a few long OCR tails from starving text-resolvable filings.
        previously_attempted = {
            str(row[0])
            for row in db.conn.execute(
                """
                SELECT DISTINCT doc_id
                FROM pdf_parse_runs
                WHERE year = ? AND ingestion_generation = ?
                  AND status NOT IN ('success', 'no_txs')
                """,
                [year, ingestion_generation],
            ).fetchall()
        }
        order = sorted(
            range(len(pdf_paths)),
            key=lambda index: (
                _pdf_page_count_hint(pdf_paths[index]),
                pdf_paths[index].stem,
            ),
        )
        pdf_paths = [pdf_paths[index] for index in order]
        existing_docs = existing_docs.iloc[order].reset_index(drop=True)
        member_lookup = _build_member_lookup(existing_docs)
        unseen_paths = [
            path for path in pdf_paths if path.stem not in previously_attempted
        ]
        retry_paths = [
            path for path in pdf_paths if path.stem in previously_attempted
        ]

        settings = _settings_for(staging)
        workers = settings.data.get_workers()
        persist_batch_size = max(1, workers)
        attempted = 0
        persisted_transactions = 0
        by_status: dict[str, int] = {}

        def persist_completed(results: list) -> None:
            nonlocal attempted, persisted_transactions
            if not results:
                return
            batch = _persist_house_parse_batch(
                db,
                year=year,
                ingestion_generation=ingestion_generation,
                member_lookup=member_lookup,
                artifact_hashes=artifact_hashes,
                results=results,
            )
            attempted += int(batch["attempted"])
            persisted_transactions += int(batch["persisted_transactions"])
            for status, count in batch["parse_run_statuses"].items():
                by_status[status] = by_status.get(status, 0) + int(count)

        completed: list = []
        deferred_paths: list[Path] = []
        with Pool(workers) as pool:
            for result in pool.imap_unordered(
                _tolerant_text_parse_worker, unseen_paths, chunksize=1
            ):
                if _engine_error_detail(result[2]) is not None:
                    deferred_paths.append(result[0])
                    continue
                completed.append(result)
                if len(completed) >= persist_batch_size:
                    persist_completed(completed)
                    completed = []
            persist_completed(completed)

            # Unseen documents that need OCR run before old retries. Each
            # expensive result is committed immediately so interruption cannot
            # discard a completed OCR/parser attempt.
            full_cascade_paths = deferred_paths + retry_paths
            for result in pool.imap_unordered(
                _tolerant_parse_worker, full_cascade_paths, chunksize=1
            ):
                persist_completed([result])

        return {
            "attempted": attempted,
            "skipped_cached": skipped_terminal,
            "parse_run_statuses": by_status,
            "persisted_transactions": persisted_transactions,
            "ingestion_generation": ingestion_generation,
        }
    finally:
        src.close()


def _house_inventory_rows(db: Database, year: int, gen: str) -> list[dict]:
    """Build source-report rows for every authoritative House PTR artifact.

    A House document can be accepted by either the deterministic parser or the
    validated Gemini OCR fallback.  The parser family is part of the accepted
    provenance, so the report source must match the transaction source.  Only
    one artifact-bound terminal run may authorize each PTR; ambiguous or
    inconsistent runs fail closed instead of silently choosing a fallback.
    """
    metadata = db.conn.execute(
        """
        SELECT doc_id, first_name, last_name, filing_date
        FROM house_generation_metadata
        WHERE archive_year = ? AND generation_id = ?
          AND filing_type = 'P'
        ORDER BY doc_id
        """,
        [year, gen],
    ).fetchall()
    meta_by_id: dict[str, tuple[object, object, object]] = {}
    for doc_id, first, last, filing_date in metadata:
        key = str(doc_id)
        if key in meta_by_id:
            raise RuntimeError(f"house inventory {year}/{key}: duplicate member metadata")
        meta_by_id[key] = (first, last, filing_date)

    artifact_rows = db.conn.execute(
        """
        SELECT doc_id, artifact_sha256
        FROM house_pdf_artifacts
        WHERE archive_year = ? AND generation_id = ?
        ORDER BY doc_id
        """,
        [year, gen],
    ).fetchall()
    artifacts: dict[str, str] = {}
    for doc_id, artifact_sha256 in artifact_rows:
        key = str(doc_id)
        if key in artifacts:
            raise RuntimeError(f"house inventory {year}/{key}: duplicate artifact")
        if not isinstance(artifact_sha256, str) or re.fullmatch(
            r"[0-9a-f]{64}", artifact_sha256
        ) is None:
            raise RuntimeError(f"house inventory {year}/{key}: missing or invalid artifact sha")
        artifacts[key] = artifact_sha256

    expected_ids = set(meta_by_id)
    artifact_ids = set(artifacts)
    missing_artifacts = sorted(expected_ids - artifact_ids)
    if missing_artifacts:
        raise RuntimeError(
            f"house inventory {year}: missing artifact(s): "
            + ", ".join(missing_artifacts[:10])
        )
    unexpected_artifacts = sorted(artifact_ids - expected_ids)
    if unexpected_artifacts:
        raise RuntimeError(
            f"house inventory {year}: artifact(s) outside PTR scope: "
            + ", ".join(unexpected_artifacts[:10])
        )

    runs = db.conn.execute(
        """
        SELECT doc_id, parser_version, status, raw_row_count,
               transaction_count, error_message, artifact_sha256
        FROM pdf_parse_runs
        WHERE year = ? AND ingestion_generation = ?
          AND status IN ('success', 'no_txs')
        ORDER BY doc_id, parsed_at DESC NULLS LAST, parser_version
        """,
        [year, gen],
    ).fetchall()
    runs_by_doc: dict[str, list[tuple]] = {}
    for run in runs:
        doc_id = str(run[0])
        if doc_id in expected_ids:
            runs_by_doc.setdefault(doc_id, []).append(run)

    transaction_rows = db.conn.execute(
        """
        SELECT doc_id, source, artifact_sha256, COUNT(*) AS row_count,
               COUNT(*) FILTER (
                   WHERE source_record_id IS DISTINCT FROM doc_id
               ) AS bad_record_ids,
               COUNT(*) FILTER (
                   WHERE chamber IS NULL OR LOWER(TRIM(chamber)) <> 'house'
               ) AS bad_chambers
        FROM transactions
        WHERE ingestion_generation = ?
        GROUP BY doc_id, source, artifact_sha256
        """,
        [gen],
    ).fetchall()
    tx_counts: dict[tuple[str, str, str | None], tuple[int, int, int]] = {}
    tx_keys_by_doc: dict[str, set[tuple[str, str | None]]] = {}
    for (
        doc_id,
        source,
        artifact_sha256,
        row_count,
        bad_record_ids,
        bad_chambers,
    ) in transaction_rows:
        doc_key = str(doc_id)
        source_key = str(source)
        tx_counts[(doc_key, source_key, artifact_sha256)] = (
            int(row_count),
            int(bad_record_ids),
            int(bad_chambers),
        )
        if doc_key in expected_ids:
            tx_keys_by_doc.setdefault(doc_key, set()).add(
                (source_key, artifact_sha256)
            )

    rows: list[dict] = []
    for doc_id in sorted(expected_ids):
        first, last, filing_date = meta_by_id[doc_id]
        name_parts = [
            str(value).strip()
            for value in (first, last)
            if value is not None
            and str(value).strip()
            and str(value).strip().lower() not in {"nan", "nat", "none"}
        ]
        member = " ".join(name_parts)
        if not member or str(filing_date).strip().lower() in {"", "nat", "nan", "none"}:
            raise RuntimeError(f"house inventory {year}/{doc_id}: missing member metadata")

        artifact_sha = artifacts[doc_id]
        candidates: list[tuple[str, str, int, int, str | None]] = []
        for (
            run_doc_id,
            parser_version,
            status,
            raw_count,
            transaction_count,
            error_message,
            run_artifact_sha,
        ) in runs_by_doc.get(doc_id, []):
            if run_artifact_sha != artifact_sha:
                continue
            if not isinstance(parser_version, str) or not parser_version.strip():
                raise RuntimeError(
                    f"house inventory {year}/{doc_id}: terminal run has no parser version"
                )
            source = (
                "gemini_ocr"
                if "gemini" in parser_version.lower()
                else "house_pdf"
            )
            try:
                raw = int(raw_count)
                accepted = int(transaction_count)
            except (TypeError, ValueError):
                raise RuntimeError(
                    f"house inventory {year}/{doc_id}: terminal run has invalid row counts"
                ) from None
            actual, bad_record_ids, bad_chambers = tx_counts.get(
                (doc_id, source, artifact_sha), (0, 0, 0)
            )
            if bad_record_ids or bad_chambers:
                raise RuntimeError(
                    f"house inventory {year}/{doc_id}: persisted rows have invalid House identity"
                )
            if status == "success":
                if raw <= 0 or accepted != raw or actual != accepted:
                    raise RuntimeError(
                        f"house inventory {year}/{doc_id}: parsed row count mismatch "
                        f"raw={raw} run={accepted} persisted={actual} source={source}"
                    )
                outcome = SourceReportOutcome.PARSED.value
            elif status == SourceReportOutcome.NO_TXS.value:
                if raw != 0 or accepted != 0 or actual != 0:
                    raise RuntimeError(
                        f"house inventory {year}/{doc_id}: no_txs row count mismatch "
                        f"raw={raw} run={accepted} persisted={actual} source={source}"
                    )
                outcome = SourceReportOutcome.NO_TXS.value
            else:
                raise RuntimeError(
                    f"house inventory {year}/{doc_id}: unsupported terminal status {status!r}"
                )
            candidates.append((source, outcome, raw, accepted, error_message))

        if len(candidates) != 1:
            if not candidates:
                raise RuntimeError(
                    f"house inventory {year}/{doc_id}: no artifact-bound terminal parse run"
                )
            raise RuntimeError(
                f"house inventory {year}/{doc_id}: ambiguous terminal parse runs"
            )
        source, outcome, raw_count, accepted_count, error_message = candidates[0]
        stale_transaction_keys = sorted(
            tx_keys_by_doc.get(doc_id, set()) - {(source, artifact_sha)}
        )
        if stale_transaction_keys:
            raise RuntimeError(
                f"house inventory {year}/{doc_id}: persisted rows have stale "
                f"source/artifact bindings: {stale_transaction_keys}"
            )
        rows.append(
            {
                "source": source,
                "ingestion_generation": gen,
                "chamber": "house",
                "source_record_id": doc_id,
                "report_path": f"{year}/pdfs/{doc_id}.pdf",
                "member": member,
                "official_filing_date": filing_date,
                "outcome": outcome,
                "artifact_sha256": artifact_sha,
                "landing_sha256": artifact_sha,
                "paper_artifact_url": None,
                "paper_artifact_sha256": None,
                "error_message": error_message,
                "raw_row_count": raw_count,
                "accepted_row_count": accepted_count,
                "rejected_row_count": 0,
            }
        )
    return rows


def _refresh_house_completion(
    db: Database, house: dict, year: int
) -> tuple[list[str], int | None]:
    """Refresh one House generation's unresolved state and activation."""
    import pandas as pd  # noqa: PLC0415

    generation = house["generation_id"]
    db.conn.execute("BEGIN TRANSACTION")
    try:
        unresolved = db.get_unresolved_house_doc_ids(year, generation)
        house["unresolved_doc_ids"] = unresolved
        house["resolved_doc_count"] = house["ptr_count"] - len(unresolved)
        if unresolved:
            db.conn.execute(
                """
                UPDATE house_archive_generations
                SET parse_status = 'incomplete'
                WHERE archive_year = ? AND generation_id = ?
                """,
                [year, generation],
            )
        else:
            rows = _house_inventory_rows(db, year, generation)
            rows_by_source = {
                source: [
                    {column: row[column] for column in SOURCE_REPORT_INPUT_COLUMNS}
                    for row in rows
                    if row["source"] == source
                ]
                for source in ("house_pdf", "gemini_ocr")
            }
            for source, source_rows in rows_by_source.items():
                db.source_reports.replace_generation(
                    generation,
                    source,
                    "house",
                    pd.DataFrame(source_rows, columns=SOURCE_REPORT_INPUT_COLUMNS),
                    _in_transaction=True,
                )
            db.mark_house_generation_parse_complete(
                year, generation, _in_transaction=True
            )
        db.conn.execute("COMMIT")
    except Exception:
        db.conn.execute("ROLLBACK")
        raise

    if unresolved:
        house["parse_status"] = "incomplete"
        return unresolved, None
    house["parse_status"] = "complete"
    house["source_report_rows"] = len(rows)
    return unresolved, len(rows)


def _house_source_report_binding_violations(
    db: Database, year: int, generation: str
) -> list[str]:
    """Return report/parser binding violations for one complete House generation."""
    artifact_rows = db.conn.execute(
        """
        SELECT doc_id, artifact_sha256
        FROM house_pdf_artifacts
        WHERE archive_year = ? AND generation_id = ?
        ORDER BY doc_id
        """,
        [year, generation],
    ).fetchall()
    terminal_runs = db.conn.execute(
        """
        SELECT doc_id, parser_version, status, artifact_sha256
        FROM pdf_parse_runs
        WHERE ingestion_generation = ?
          AND status IN ('success', 'no_txs')
        """,
        [generation],
    ).fetchall()
    reports = db.conn.execute(
        """
        SELECT source, source_record_id, outcome, artifact_sha256
        FROM source_reports
        WHERE ingestion_generation = ?
          AND source IN ('house_pdf', 'gemini_ocr')
          AND LOWER(chamber) = 'house'
        """,
        [generation],
    ).fetchall()

    runs_by_artifact: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for doc_id, parser_version, status, artifact_sha in terminal_runs:
        if (
            doc_id is None
            or artifact_sha is None
            or not isinstance(parser_version, str)
            or not parser_version.strip()
        ):
            continue
        source = "gemini_ocr" if "gemini" in parser_version.lower() else "house_pdf"
        outcome = "parsed" if status == "success" else "no_txs"
        runs_by_artifact.setdefault((str(doc_id), str(artifact_sha)), []).append(
            (source, outcome)
        )

    reports_by_artifact: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for source, record_id, outcome, artifact_sha in reports:
        if record_id is None or artifact_sha is None:
            continue
        reports_by_artifact.setdefault((str(record_id), str(artifact_sha)), []).append(
            (str(source), str(outcome))
        )

    violations: list[str] = []
    for doc_id, artifact_sha in artifact_rows:
        key = (str(doc_id), str(artifact_sha))
        run_bindings = runs_by_artifact.get(key, [])
        report_bindings = reports_by_artifact.get(key, [])
        if len(run_bindings) != 1:
            violations.append(
                f"artifact {doc_id} has {len(run_bindings)} matching "
                "terminal parser run(s)"
            )
        if len(report_bindings) != 1:
            violations.append(
                f"artifact {doc_id} has {len(report_bindings)} matching "
                "House source report(s)"
            )
        if len(run_bindings) == 1 and len(report_bindings) == 1:
            if run_bindings[0] != report_bindings[0]:
                violations.append(
                    f"artifact {doc_id} parser/report binding mismatch: "
                    f"run={run_bindings[0]} report={report_bindings[0]}"
                )
    return violations


def _source_report_provenance_violations(
    db: Database, generation: str
) -> list[str]:
    """Return semantic provenance violations for reports in one generation."""
    reports = db.conn.execute(
        """
        SELECT source, chamber, source_record_id, outcome,
               artifact_sha256, landing_sha256, paper_artifact_url,
               paper_artifact_sha256
        FROM source_reports
        WHERE ingestion_generation = ?
        ORDER BY source, source_record_id
        """,
        [generation],
    ).fetchall()
    violations: list[str] = []
    for (
        source,
        chamber,
        record_id,
        outcome,
        artifact,
        landing,
        paper_url,
        paper_sha,
    ) in reports:
        identity = f"source={source} chamber={chamber} gen={generation} {record_id}"
        nonpaper_hashes_valid = (
            isinstance(artifact, str)
            and re.fullmatch(r"[0-9a-f]{64}", artifact) is not None
            and isinstance(landing, str)
            and re.fullmatch(r"[0-9a-f]{64}", landing) is not None
            and artifact == landing
        )
        if outcome in ("parsed", "no_txs") and not nonpaper_hashes_valid:
            violations.append(
                f"{identity}: {outcome} report has invalid artifact provenance"
            )
        if outcome in ("parsed", "no_txs") and (
            paper_url is not None or paper_sha is not None
        ):
            violations.append(
                f"{identity}: {outcome} report sets paper artifact fields"
            )
        if outcome == "no_txs" and (
            not isinstance(chamber, str) or chamber.strip().lower() != "house"
        ):
            violations.append(f"{identity}: no_txs report is not a House report")
        if outcome == "paper_only":
            paper_valid = (
                isinstance(chamber, str)
                and chamber.strip().lower() == "senate"
                and nonpaper_hashes_valid
                and isinstance(paper_sha, str)
                and re.fullmatch(r"[0-9a-f]{64}", paper_sha) is not None
                and _is_official_paper_url(paper_url)
            )
            if not paper_valid:
                violations.append(
                    f"{identity}: paper_only report has invalid Senate paper provenance"
                )
    return violations


def house_parse(args) -> None:
    staging = Path(args.staging)
    manifest = _load_manifest(staging)
    years = [int(y) for y in args.years] if args.years else HOUSE_YEARS
    db = Database(staging / "congress.duckdb", read_only=False)
    try:
        for year in years:
            house = manifest.get("house", {}).get(str(year))
            if house is None:
                print(f"house-parse {year}: skipped (not fetched)")
                continue
            if house.get("parse_status") == "complete" and not args.force:
                print(f"house-parse {year}: skipped (already complete)")
                continue
            result = _parse_house_year_tolerant(staging, db, year)
            previous_result = house.get("parse_result") or {}
            if result.get("attempted", 0) == 0 and previous_result:
                # Resumable skip: keep the original outcome telemetry.
                merged = dict(previous_result)
                merged["skipped_cached"] = result.get("skipped_cached", 0) + previous_result.get("skipped_cached", 0)
                result = merged
            house["parse_result"] = result
            unresolved, report_count = _refresh_house_completion(db, house, year)
            if unresolved:
                print(
                    f"house-parse {year}: INCOMPLETE — {len(unresolved)} unresolved "
                    f"({', '.join(unresolved[:10])}{'...' if len(unresolved) > 10 else ''})"
                )
            else:
                print(f"house-parse {year}: COMPLETE — {report_count} inventory rows persisted")
            _save_manifest(staging, manifest)
    finally:
        db.close()


def _ingest_cached_gemini_year(
    staging: Path, db: Database, year: int, cache_dir: Path
) -> dict:
    """Ingest only already-cached Gemini output bound to the exact staged PDF."""
    from scripts.gemini_ocr_common import (  # noqa: PLC0415
        CACHE_ENVELOPE_VERSION,
        OUTPUT_SCHEMA_VERSION,
        PROMPT_SHA256,
        parse_gemini_output,
        pdf_page_count,
        validate_transactions,
    )
    from scripts.ocr_zero_rows import insert_transactions  # noqa: PLC0415

    generation = db.get_latest_house_generation(year)
    if generation is None:
        raise RuntimeError(f"cached Gemini ingest {year}: no acquired generation")
    unresolved = db.get_unresolved_house_doc_ids(year, generation)
    artifacts = {
        str(doc_id): str(artifact_sha256)
        for doc_id, artifact_sha256 in db.conn.execute(
            """
            SELECT doc_id, artifact_sha256
            FROM house_pdf_artifacts
            WHERE archive_year = ? AND generation_id = ?
            """,
            [year, generation],
        ).fetchall()
    }
    accepted: list[dict] = []
    rejected: list[dict] = []
    missing: list[str] = []
    for doc_id in unresolved:
        cache_path = cache_dir / f"{doc_id}.json"
        if not cache_path.is_file():
            missing.append(doc_id)
            continue
        try:
            envelope = json.loads(cache_path.read_text())
            pdf_path = staging / str(year) / "pdfs" / f"{doc_id}.pdf"
            artifact_sha256 = artifacts.get(doc_id)
            if not artifact_sha256 or _sha256_file(pdf_path) != artifact_sha256:
                raise ValueError("staged PDF hash does not match acquired artifact")
            if envelope.get("cache_envelope_version") != CACHE_ENVELOPE_VERSION:
                raise ValueError("cache envelope version mismatch")
            if envelope.get("output_schema_version") != OUTPUT_SCHEMA_VERSION:
                raise ValueError("cache output schema version mismatch")
            if envelope.get("prompt_sha256") != PROMPT_SHA256:
                raise ValueError("cache prompt identity mismatch")
            if str(envelope.get("doc_id")) != doc_id:
                raise ValueError("cache document identity mismatch")
            if envelope.get("pdf_sha256") != artifact_sha256:
                raise ValueError("cache PDF hash mismatch")
            page_count = pdf_page_count(pdf_path)
            if int(envelope.get("pdf_page_count") or 0) != page_count:
                raise ValueError("cache PDF page-count mismatch")
            parser_version = str(envelope.get("parser_version") or "").strip()
            model = str(envelope.get("model") or "").strip()
            if not parser_version or not model:
                raise ValueError("cache lacks parser/model identity")
            parsed = parse_gemini_output(
                envelope.get("output"), expected_page_count=page_count
            )
            metadata = db.conn.execute(
                """
                SELECT filing_date, first_name, last_name
                FROM house_generation_metadata
                WHERE archive_year = ? AND generation_id = ?
                  AND doc_id = ? AND filing_type = 'P'
                """,
                [year, generation, doc_id],
            ).fetchone()
            if metadata is None:
                raise ValueError("cache document lacks generation metadata")
            filing_date, first_name, last_name = metadata
            expected_member = " ".join(
                str(part).strip() for part in (first_name, last_name) if part
            ).strip()
            validated, rejections = validate_transactions(
                doc_id,
                parsed.member,
                parsed.transactions,
                filing_date,
                expected_member,
            )
            fatal_rejections = {
                key: value
                for key, value in rejections.items()
                if key != "member_mismatch"
            }
            if fatal_rejections:
                raise ValueError(
                    "semantic validation failed: "
                    + json.dumps(fatal_rejections, sort_keys=True)
                )
            inserted = insert_transactions(
                doc_id,
                year,
                parsed.member,
                validated,
                db_path=str(staging / "congress.duckdb"),
                parser_version=parser_version,
                raw_count=parsed.raw_row_count,
                artifact_sha256=artifact_sha256,
                ingestion_generation=generation,
                engine_model=model,
            )
            expected_count = len(validated)
            run = db.conn.execute(
                """
                SELECT status, transaction_count, engines_attempted
                FROM pdf_parse_runs
                WHERE doc_id = ? AND parser_version = ?
                  AND artifact_sha256 = ? AND ingestion_generation = ?
                ORDER BY parsed_at DESC LIMIT 1
                """,
                [doc_id, parser_version, artifact_sha256, generation],
            ).fetchone()
            expected_status = "no_txs" if expected_count == 0 else "success"
            if run is None or run[0] != expected_status or int(run[1]) != expected_count:
                raise RuntimeError(
                    f"cache persistence mismatch: expected {expected_status}/{expected_count}, got {run}"
                )
            if str(run[2]) != model:
                raise RuntimeError("cache parser/model provenance mismatch after persistence")
            accepted.append(
                {
                    "doc_id": doc_id,
                    "parser_version": parser_version,
                    "model": model,
                    "transactions": expected_count,
                    "inserted": int(inserted),
                    "member_mismatch": int(rejections.get("member_mismatch", 0)),
                }
            )
        except Exception as exc:  # noqa: BLE001 -- one bad cache must stay unresolved
            rejected.append(
                {
                    "doc_id": doc_id,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return {
        "year": year,
        "generation": generation,
        "unresolved_before": len(unresolved),
        "accepted": accepted,
        "rejected": rejected,
        "missing_cache": missing,
    }


def house_cache_ocr(args) -> None:
    """Consume immutable, schema-validated Gemini caches without model I/O."""
    staging = Path(args.staging)
    manifest = _load_manifest(staging)
    years = [int(y) for y in args.years] if args.years else list(REQUIRED_HOUSE_YEARS)
    cache_dir = Path(args.cache_dir) if args.cache_dir else _REPO_ROOT / "data" / "gemini_cache"
    db = Database(staging / "congress.duckdb", read_only=False)
    try:
        records = manifest.setdefault("house_cache_ocr", {})
        for year in years:
            house = manifest.get("house", {}).get(str(year))
            if house is None:
                print(f"house-cache-ocr {year}: skipped (not fetched)")
                continue
            result = _ingest_cached_gemini_year(staging, db, year, cache_dir)
            unresolved, report_count = _refresh_house_completion(db, house, year)
            result["unresolved_after"] = len(unresolved)
            result["source_report_rows"] = report_count
            records[str(year)] = result
            _save_manifest(staging, manifest)
            print(
                f"house-cache-ocr {year}: accepted={len(result['accepted'])} "
                f"rejected={len(result['rejected'])} missing={len(result['missing_cache'])} "
                f"unresolved={len(unresolved)}"
            )
    finally:
        db.close()


# --------------------------------------------------------------------------
# Senate
# --------------------------------------------------------------------------

def senate(args) -> None:
    from analyzer.senate_efd import SenateEFDError, SenateEFDSource  # noqa: PLC0415

    staging = Path(args.staging)
    manifest = _load_manifest(staging)
    if manifest.get("senate", {}).get("status") in ("persisted", "quarantined") and not args.force:
        print(f"senate: skipped (already {manifest['senate']['status']})")
        return
    generation = manifest["generation"]
    end = date.today()
    db = Database(staging / "congress.duckdb", read_only=False)
    src = SenateEFDSource(
        data_dir=str(staging),
        read_only=False,
        db=db,
        ingestion_generation=generation,
    )
    try:
        try:
            df = src.fetch_all_trades(SENATE_START, end)
        except SenateEFDError as exc:
            summary = src.last_refresh_summary
            inventory = list(src.report_inventory)
            record = {
                "status": "quarantined",
                "error": str(exc),
                "start_date": str(SENATE_START),
                "end_date": str(end),
                "summary": None if summary is None else {
                    "found": summary.found,
                    "parsed": summary.parsed,
                    "paper_only": summary.paper_only,
                    "unavailable": summary.unavailable,
                    "failed": summary.failed,
                },
                "inventory_count": len(inventory),
                "inventory": [
                    {
                        "source_record_id": row.get("source_record_id"),
                        "member": row.get("member"),
                        "outcome": row.get("outcome"),
                        "error_message": row.get("error_message"),
                    }
                    for row in inventory
                ],
            }
            manifest["senate"] = record
            _save_manifest(staging, manifest)
            print(f"senate: QUARANTINED ({record['summary']})")
            return
        inserted = src.save_to_db(df)
        summary = src.last_refresh_summary
        assert summary is not None  # save_to_db raises without a complete summary
        record = {
            "status": "persisted",
            "start_date": str(SENATE_START),
            "end_date": str(end),
            "summary": {
                "found": summary.found,
                "parsed": summary.parsed,
                "paper_only": summary.paper_only,
                "unavailable": summary.unavailable,
                "failed": summary.failed,
            },
            "inserted_transactions": inserted,
            "report_count": len(src.report_inventory),
            "reports": [
                {
                    "source_record_id": row.get("source_record_id"),
                    "member": row.get("member"),
                    "official_filing_date": str(row.get("official_filing_date")),
                    "outcome": row.get("outcome"),
                    "artifact_sha256": row.get("artifact_sha256"),
                    "landing_sha256": row.get("landing_sha256"),
                    "paper_artifact_sha256": row.get("paper_artifact_sha256"),
                    "paper_artifact_url": row.get("paper_artifact_url"),
                    "error_message": row.get("error_message"),
                    "raw_row_count": row.get("raw_row_count"),
                    "accepted_row_count": row.get("accepted_row_count"),
                    "rejected_row_count": row.get("rejected_row_count"),
                }
                for row in src.report_inventory
            ],
        }
        canary_members = {
            member: [
                r["source_record_id"]
                for r in record["reports"]
                if member.split()[0] in r["member"] and r["outcome"] == "parsed"
            ]
            for member in ("Katie Britt", "Rick Scott")
        }
        record["canary_members"] = canary_members
        manifest["senate"] = record
        _save_manifest(staging, manifest)
        print(
            f"senate: PERSISTED — found={summary.found} parsed={summary.parsed} "
            f"paper_only={summary.paper_only} inserted={inserted}"
        )
    finally:
        src.close()
        db.close()


# --------------------------------------------------------------------------
# Prices
# --------------------------------------------------------------------------

def prices(args) -> None:
    """Gap-fill staged tickers via the accepted price acquisition + snapshot.

    If a verified sibling refresh was ingested, only staged tickers missing
    from the prices table are fetched (chunked); otherwise the full staged
    universe is fetched. Always ends with a value-hashed snapshot and an exact
    coverage report. Recorded as the fallback path when no sibling exists.
    """
    import re  # noqa: PLC0415
    import time  # noqa: PLC0415

    from analyzer.price_source import YFinancePriceSource  # noqa: PLC0415

    staging = Path(args.staging)
    manifest = _load_manifest(staging)
    if manifest.get("prices", {}).get("status") == "snapshotted" and not args.force:
        print("prices: skipped (already snapshotted)")
        return
    from datetime import timedelta  # noqa: PLC0415

    # previous_nyse_session includes the day itself; exclude today so an
    # in-progress session is never used as the completed-session bound.
    end = previous_nyse_session(date.today() - timedelta(days=1))
    db = Database(staging / "congress.duckdb", read_only=False)
    try:
        rows = db.conn.execute(
            "SELECT DISTINCT ticker FROM transactions WHERE ticker IS NOT NULL"
        ).fetchall()
        all_tickers = sorted(
            {
                str(row[0])
                for row in rows
                if str(row[0])
                and re.fullmatch(
                    r"^[A-Z]{1,5}(?:[.-][A-Z]{1,2})?$", str(row[0])
                )
            }
            | {"SPY"}
        )
        ticker_total = len({str(r[0]) for r in rows if str(r[0])})
        already = {
            str(row[0])
            for row in db.conn.execute(
                "SELECT DISTINCT ticker FROM prices"
            ).fetchall()
        }
        missing = sorted(set(all_tickers) - already)
        settings = _settings_for(staging)
        price_source = YFinancePriceSource(settings, read_only=False, db=db)
        from analyzer.exceptions import DataSourceError  # noqa: PLC0415

        fetched_by_sibling = len(already & set(all_tickers))
        fetched_here = 0
        unavailable_here: list[str] = []
        chunk_size = 100
        try:
            for offset in range(0, len(missing), chunk_size):
                chunk = missing[offset:offset + chunk_size]
                t0 = time.time()
                try:
                    matrix = price_source.get_prices(chunk, PRICE_START, end)
                    fetched_here += len(
                        {c for c in matrix.columns if c in set(chunk)}
                    )
                    print(
                        f"prices: chunk {offset // chunk_size + 1}/"
                        f"{(len(missing) + chunk_size - 1) // chunk_size} "
                        f"fetched {len(chunk)} ({time.time() - t0:.0f}s)"
                    )
                except DataSourceError:
                    # Accepted per-ticker recovery on batch gate failure:
                    # fetch each ticker individually, recording unavailable.
                    for ticker in chunk:
                        try:
                            price_source.get_prices(
                                [ticker], PRICE_START, end
                            )
                            fetched_here += 1
                        except DataSourceError:
                            unavailable_here.append(ticker)
                    print(
                        f"prices: chunk {offset // chunk_size + 1}/"
                        f"{(len(missing) + chunk_size - 1) // chunk_size} "
                        f"recovered per-ticker "
                        f"({len(chunk) - len([t for t in chunk if t in unavailable_here])} ok, "
                        f"{len([t for t in chunk if t in unavailable_here])} unavailable)"
                    )
        finally:
            price_source.close()
        snapshot = create_snapshot(db, all_tickers, PRICE_START, end)
        snapshot_path = staging / "price_snapshot.json"
        save_snapshot(snapshot, snapshot_path)
        record = {
            "status": "snapshotted",
            "start_date": str(PRICE_START),
            "end_date": str(end),
            "transaction_ticker_total": ticker_total,
            "eligible_tickers_requested": snapshot.requested_tickers,
            "covered_by_sibling_refresh": fetched_by_sibling,
            "fetched_by_this_stage": fetched_here,
            "unavailable_this_stage": sorted(set(unavailable_here)),
            "resolved_tickers": snapshot.resolved_tickers,
            "unresolved_tickers": list(snapshot.unresolved_tickers),
            "price_rows": snapshot.price_rows,
            "first_date": snapshot.first_date,
            "last_date": snapshot.last_date,
            "value_hash": snapshot.value_hash,
            "snapshot_path": str(snapshot_path),
            "path": (
                "sibling_refresh_plus_fallback_gap_fill"
                if fetched_by_sibling
                else "fallback_full_fetch"
            ),
        }
        manifest["prices"] = record
        _save_manifest(staging, manifest)
        print(
            f"prices: snapshot rows={snapshot.price_rows} "
            f"tickers={snapshot.resolved_tickers}/{snapshot.requested_tickers} "
            f"range={snapshot.first_date}..{snapshot.last_date} "
            f"hash={snapshot.value_hash[:16]} path={record['path']}"
        )
    finally:
        db.close()


# --------------------------------------------------------------------------
# Senate frozen-window ingest
# --------------------------------------------------------------------------

def ingest_senate_window(args) -> None:
    """Hash-verify and ingest a frozen Senate eFD window via persist_source_refresh.

    The window artifact (sibling senate sweep track) contains
    transactions.jsonl + report_inventory.jsonl with SHAs in its manifest.
    This atomically replaces the senate_efd source/chamber state.
    """
    staging = Path(args.staging)
    manifest = _load_manifest(staging)
    if manifest.get("senate_window", {}).get("status") == "ingested" and not args.force:
        print("senate-window: skipped (already ingested)")
        return
    window_dir = Path(args.window_dir) if args.window_dir else None
    if window_dir is None:
        senate_track = _sibling_search_dirs("senate", manifest["generation"])
        candidates = []
        for base in senate_track:
            if base.exists():
                candidates.extend(sorted(base.glob("window-*")))
        if not candidates:
            raise SystemExit("no senate window dir found (--window-dir or sibling track)")
        window_dir = candidates[-1]
    mpath = window_dir / "manifest.json"
    if not mpath.exists():
        raise SystemExit(f"window manifest not found: {mpath}")
    window_manifest = json.loads(mpath.read_text())
    verification = _verify_artifact_files(window_manifest, window_dir)
    if verification["mismatches"]:
        raise SystemExit(
            f"senate window hash mismatch: {verification['mismatches']}"
        )
    db = Database(staging / "congress.duckdb", read_only=False)
    try:
        ingest = _ingest_senate(window_dir, window_manifest, db)
        generation = window_manifest["generation"]
        record = {
            "status": "ingested",
            "path": str(window_dir),
            "generation": generation,
            "window": window_manifest.get("window"),
            "summary": window_manifest.get("summary"),
            "outcome_counts": window_manifest.get("outcome_counts"),
            "canaries": window_manifest.get("canaries"),
            "inserted_transactions": ingest["inserted_transactions"],
            "verification": verification,
            "replaced_previous_sweep": bool(
                manifest.get("consume", {}).get("senate", {}).get("status")
                == "ingested"
            ),
        }
        manifest["senate_window"] = record
        _save_manifest(staging, manifest)
        print(
            f"senate-window: INGESTED gen={generation} "
            f"reports={record['summary'].get('found')} "
            f"transactions={record['inserted_transactions']} (replaced prior sweep: "
            f"{record['replaced_previous_sweep']})"
        )
    finally:
        db.close()


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------

def _check(checks: dict, name: str, condition: bool, detail: str = "") -> None:
    checks[name] = {"passed": bool(condition), "detail": detail}


def _latest_accepted_house_generations(db: Database) -> dict[int, str]:
    """Return the latest complete House generation with no unresolved docs.

    ``canonical_transactions`` applies this semantic acceptance predicate
    before selecting one generation per archive year. Reuse the database's
    authoritative unresolved-document helper here instead of copying that
    predicate into staged verification.
    """
    rows = db.conn.execute(
        """
        SELECT archive_year, generation_id
        FROM house_archive_generations
        WHERE parse_status = 'complete'
        ORDER BY archive_year, promoted_at DESC, generation_id DESC
        """
    ).fetchall()
    accepted: dict[int, str] = {}
    for archive_year, generation_id in rows:
        if archive_year is None or generation_id is None:
            continue
        archive_year = int(archive_year)
        generation_id = str(generation_id)
        if archive_year in accepted:
            continue
        if not db.get_unresolved_house_doc_ids(archive_year, generation_id):
            accepted[archive_year] = generation_id
    return accepted


def _canonical_view_diff_counts(db: Database) -> tuple[int, int]:
    """Return canonical transaction IDs outside and missing from expectations."""
    accepted_house_generations = _latest_accepted_house_generations(db)
    if accepted_house_generations:
        generation_values = ", ".join(
            "(?, ?)" for _ in accepted_house_generations
        )
        generation_params = [
            value
            for archive_year, generation_id in accepted_house_generations.items()
            for value in (archive_year, generation_id)
        ]
        accepted_generations_cte = "VALUES " + generation_values
    else:
        generation_params = []
        accepted_generations_cte = (
            "SELECT CAST(NULL AS INTEGER) AS archive_year, "
            "CAST(NULL AS VARCHAR) AS generation_id WHERE FALSE"
        )
    canonical_extra, canonical_missing = db.conn.execute(
        f"""
        WITH accepted_house_generations(archive_year, generation_id) AS (
            {accepted_generations_cte}
        ), expected AS (
            SELECT t.id FROM transactions t
            WHERE (t.source = 'senate_efd' AND t.chamber = 'senate')
               OR (
                   t.source IN ('house_pdf', 'gemini_ocr')
                   AND EXISTS (
                       SELECT 1
                       FROM house_archive_generations generation
                       JOIN accepted_house_generations accepted
                         ON accepted.archive_year = generation.archive_year
                        AND accepted.generation_id = generation.generation_id
                       WHERE generation.generation_id = t.ingestion_generation
                   )
               )
        ), canonical_extra AS (
            SELECT id FROM canonical_transactions
            EXCEPT
            SELECT id FROM expected
        ), canonical_missing AS (
            SELECT id FROM expected
            EXCEPT
            SELECT id FROM canonical_transactions
        )
        SELECT
            (SELECT COUNT(*) FROM canonical_extra),
            (SELECT COUNT(*) FROM canonical_missing)
        """,
        generation_params,
    ).fetchone()
    return int(canonical_extra), int(canonical_missing)


def verify(args) -> None:
    staging = Path(args.staging)
    manifest = _load_manifest(staging)
    checks: dict = {}
    db_path = staging / "congress.duckdb"
    db = Database(db_path, read_only=True)
    incomplete_years: list[int] = []
    try:
        _check(
            checks,
            "read_only_audit",
            db.is_read_only,
            "staged database opened read-only",
        )
        _check(
            checks,
            "no_sibling_db_dependency",
            True,
            f"verification queried only {db_path}",
        )
        scope = manifest.get("scope")
        scope_years = scope.get("house_years") if isinstance(scope, dict) else None
        _check(
            checks,
            "rebuild_scope_present",
            isinstance(scope, dict)
            and isinstance(scope_years, list)
            and set(REQUIRED_HOUSE_YEARS).issubset(set(scope_years))
            and bool(scope.get("senate_start"))
            and bool(scope.get("price_start")),
            f"scope={scope} required_house_years={list(REQUIRED_HOUSE_YEARS)}",
        )
        house_manifest = manifest.get("house")
        try:
            declared_house_years = {
                int(year) for year in house_manifest
            } if isinstance(house_manifest, dict) else set()
        except (TypeError, ValueError):
            declared_house_years = set()
        _check(
            checks,
            "house_scope_declared",
            set(REQUIRED_HOUSE_YEARS).issubset(declared_house_years),
            f"declared={sorted(declared_house_years)} required={list(REQUIRED_HOUSE_YEARS)}",
        )

        # Current House completeness and the historical years actually consumed
        # by production validation are bound to the declared generation and the
        # artifact-level terminal predicate in Database.get_unresolved_house_doc_ids.
        # Older staged archives are diagnostics only and cannot block promotion.
        for year in REQUIRED_HOUSE_YEARS:
            entry = (
                house_manifest.get(str(year))
                if isinstance(house_manifest, dict)
                else None
            )
            generation = entry.get("generation_id") if isinstance(entry, dict) else None
            generation_ok = isinstance(generation, str) and bool(generation.strip())
            generation_row = None
            latest_row = None
            if generation_ok:
                generation_row = db.conn.execute(
                    """
                    SELECT generation_id, parse_status, ptr_count
                    FROM house_archive_generations
                    WHERE archive_year = ? AND generation_id = ?
                    """,
                    [year, generation],
                ).fetchone()
                latest_row = db.conn.execute(
                    """
                    SELECT generation_id
                    FROM house_archive_generations
                    WHERE archive_year = ?
                    ORDER BY promoted_at DESC, generation_id DESC
                    LIMIT 1
                    """,
                    [year],
                ).fetchone()
            declared_present = generation_ok and generation_row is not None
            latest_ok = declared_present and str(latest_row[0]) == generation
            _check(
                checks,
                f"house_{year}_declared_generation_present",
                declared_present,
                f"generation={generation!r}",
            )
            _check(
                checks,
                f"house_{year}_declared_generation_is_latest",
                latest_ok,
                f"declared={generation!r} "
                f"latest={latest_row[0] if latest_row else None!r}",
            )
            if not declared_present:
                incomplete_years.append(year)
                _check(
                    checks,
                    f"house_{year}_parse_status_complete",
                    False,
                    "declared generation is absent",
                )
                _check(
                    checks,
                    f"house_{year}_zero_unresolved_pdfs",
                    False,
                    "declared generation is absent",
                )
                _check(
                    checks,
                    f"house_{year}_artifact_scope_complete",
                    False,
                    "declared generation is absent",
                )
                continue

            _, db_status, ptr_count = generation_row
            unresolved = db.get_unresolved_house_doc_ids(year, str(generation))
            artifact_count = int(
                db.conn.execute(
                    """
                    SELECT COUNT(*) FROM house_pdf_artifacts
                    WHERE archive_year = ? AND generation_id = ?
                    """,
                    [year, generation],
                ).fetchone()[0]
            )
            manifest_complete = entry.get("parse_status") == "complete"
            complete = manifest_complete and db_status == "complete"
            _check(
                checks,
                f"house_{year}_parse_status_complete",
                complete,
                f"manifest={entry.get('parse_status')!r} database={db_status!r}",
            )
            _check(
                checks,
                f"house_{year}_zero_unresolved_pdfs",
                not unresolved,
                f"unresolved={unresolved[:10]}",
            )
            _check(
                checks,
                f"house_{year}_artifact_scope_complete",
                artifact_count == int(ptr_count),
                f"artifacts={artifact_count} ptr_count={ptr_count}",
            )
            if not (
                latest_ok
                and complete
                and not unresolved
                and artifact_count == int(ptr_count)
            ):
                incomplete_years.append(year)

        # 1. parse counts == persisted rows per document, artifact, and
        # generation.
        mismatches = db.conn.execute(
            """
            SELECT p.doc_id, p.ingestion_generation, p.artifact_sha256,
                   p.transaction_count,
                   (SELECT COUNT(*) FROM transactions t
                    WHERE t.doc_id = p.doc_id
                      AND t.source IN ('house_pdf', 'gemini_ocr')
                      AND t.ingestion_generation = p.ingestion_generation
                      AND t.artifact_sha256 IS NOT DISTINCT FROM p.artifact_sha256) AS actual
            FROM pdf_parse_runs p
            WHERE p.status IN ('success', 'no_txs')
              AND (
                  COALESCE(p.transaction_count, -1) != (
                      SELECT COUNT(*) FROM transactions t
                      WHERE t.doc_id = p.doc_id
                        AND t.source IN ('house_pdf', 'gemini_ocr')
                        AND t.ingestion_generation = p.ingestion_generation
                        AND t.artifact_sha256 IS NOT DISTINCT FROM p.artifact_sha256
                  )
                  OR (p.status = 'success' AND (
                      COALESCE(p.raw_row_count, 0) <= 0
                      OR COALESCE(p.transaction_count, 0) <= 0
                  ))
                  OR (p.status = 'no_txs' AND (
                      COALESCE(p.raw_row_count, -1) != 0
                      OR COALESCE(p.transaction_count, -1) != 0
                  ))
              )
            """,
        ).fetchall()
        _check(
            checks,
            "house_success_run_counts_equal_persisted_rows",
            not mismatches,
            f"mismatches={mismatches[:10]}",
        )

        # Source-report inventories must reconcile their outcome equation and
        # their accepted row counts with persisted source rows.
        report_groups = db.conn.execute(
            """
            SELECT ingestion_generation, source, chamber,
                   COUNT(*) AS found,
                   COUNT(*) FILTER (WHERE outcome = 'parsed') AS parsed,
                   COUNT(*) FILTER (WHERE outcome = 'no_txs') AS no_txs,
                   COUNT(*) FILTER (WHERE outcome = 'paper_only') AS paper_only,
                   COUNT(*) FILTER (WHERE outcome = 'unavailable') AS unavailable,
                   COUNT(*) FILTER (WHERE outcome = 'failed') AS failed
            FROM source_reports
            GROUP BY 1, 2, 3
            ORDER BY 1, 2, 3
            """,
        ).fetchall()
        bad_report_groups = [
            row for row in report_groups
            if row[3] != row[4] + row[5] + row[6] + row[7] + row[8]
        ]
        bad_report_counts = db.conn.execute(
            """
            SELECT ingestion_generation, source, chamber, source_record_id,
                   outcome, raw_row_count, accepted_row_count, rejected_row_count
            FROM source_reports
            WHERE raw_row_count IS NULL OR accepted_row_count IS NULL
               OR rejected_row_count IS NULL
               OR raw_row_count < 0 OR accepted_row_count < 0 OR rejected_row_count < 0
               OR raw_row_count != accepted_row_count + rejected_row_count
               OR (outcome = 'parsed' AND (accepted_row_count <= 0
                                           OR raw_row_count != accepted_row_count
                                           OR rejected_row_count != 0))
               OR (outcome = 'no_txs' AND (raw_row_count != 0
                                           OR accepted_row_count != 0
                                           OR rejected_row_count != 0))
               OR (outcome = 'paper_only' AND (raw_row_count != 0
                                               OR accepted_row_count != 0
                                               OR rejected_row_count != 0))
            ORDER BY ingestion_generation, source_record_id
            LIMIT 10
            """,
        ).fetchall()
        report_row_mismatches = db.conn.execute(
            """
            SELECT r.ingestion_generation, r.source, r.chamber,
                   r.source_record_id, r.accepted_row_count, COUNT(t.id) AS actual
            FROM source_reports r
            LEFT JOIN transactions t
              ON t.ingestion_generation = r.ingestion_generation
             AND t.source = r.source
             AND LOWER(t.chamber) = LOWER(r.chamber)
             AND t.source_record_id = r.source_record_id
             AND t.artifact_sha256 IS NOT DISTINCT FROM r.artifact_sha256
            GROUP BY 1, 2, 3, 4, 5
            HAVING r.accepted_row_count != COUNT(t.id)
            ORDER BY 1, 4
            LIMIT 10
            """,
        ).fetchall()
        _check(
            checks,
            "source_report_reconciliation_consistent",
            not bad_report_groups and not bad_report_counts and not report_row_mismatches,
            f"groups={bad_report_groups[:10]} counts={bad_report_counts[:10]} "
            f"rows={report_row_mismatches[:10]}",
        )

        # Chronology is a blocking semantic property, not an analysis filter.
        chronology = db.conn.execute(
            """
            SELECT COUNT(*) FROM transactions
            WHERE transaction_date IS NULL OR disclosure_date IS NULL
               OR transaction_date > disclosure_date
               OR (notification_date IS NOT NULL
                   AND notification_date < transaction_date)
               OR (chamber ILIKE 'senate'
                   AND notification_date IS NOT NULL
                   AND official_filing_date IS NOT NULL
                   AND notification_date > official_filing_date)
            """
        ).fetchone()[0]
        _check(
            checks,
            "chronology_valid",
            int(chronology) == 0,
            f"invalid_rows={chronology}",
        )

        # 3. duplicate policy on the source identity tuple
        duplicates = db.conn.execute(
            """
            SELECT source, chamber, source_record_id, source_row_id,
                   ingestion_generation, COUNT(*) AS n
            FROM transactions
            GROUP BY 1, 2, 3, 4, 5 HAVING COUNT(*) > 1
            ORDER BY 1, 2, 3, 4
            LIMIT 10
            """
        ).fetchall()
        _check(
            checks,
            "no_source_identity_duplicates",
            not duplicates,
            f"dups={duplicates[:10]}",
        )
        missing_identity = db.conn.execute(
            """
            SELECT source, chamber, source_record_id, source_row_id,
                   ingestion_generation
            FROM transactions
            WHERE source IN ('house_pdf', 'gemini_ocr', 'senate_efd')
              AND (chamber IS NULL OR TRIM(chamber) = ''
                   OR source_record_id IS NULL OR TRIM(source_record_id) = ''
                   OR source_row_id IS NULL OR TRIM(source_row_id) = ''
                   OR ingestion_generation IS NULL OR TRIM(ingestion_generation) = '')
            LIMIT 10
            """
        ).fetchall()
        _check(
            checks,
            "source_identity_complete",
            not missing_identity,
            f"missing={missing_identity[:10]}",
        )

        # 5. House source reports are checked for each required complete
        # generation; an incomplete generation cannot authorize its inventory.
        for year in REQUIRED_HOUSE_YEARS:
            house = (
                house_manifest.get(str(year))
                if isinstance(house_manifest, dict)
                else None
            )
            if not isinstance(house, dict) or house.get("parse_status") != "complete":
                continue
            generation = house.get("generation_id")
            if not isinstance(generation, str) or not generation:
                continue
            source_reconciliations = {
                source: db.source_reports.reconcile(generation, source, "house")
                for source in ("house_pdf", "gemini_ocr")
            }
            reconcile = {
                name: sum(part[name] for part in source_reconciliations.values())
                for name in next(iter(source_reconciliations.values()))
            }
            artifact_count = int(
                db.conn.execute(
                    """
                    SELECT COUNT(*) FROM house_pdf_artifacts
                    WHERE archive_year = ? AND generation_id = ?
                    """,
                    [year, generation],
                ).fetchone()[0]
            )
            expected = (
                reconcile["found"] == artifact_count
                and reconcile["found"] == (
                    reconcile["parsed"]
                    + reconcile["no_txs"]
                    + reconcile["paper_only"]
                    + reconcile["unavailable"]
                    + reconcile["failed"]
                )
                and reconcile["failed"] == 0
                and reconcile["unavailable"] == 0
            )
            binding_violations = _house_source_report_binding_violations(
                db, year, generation
            )
            provenance_violations = _source_report_provenance_violations(
                db, generation
            )
            expected = expected and not binding_violations and not provenance_violations
            _check(
                checks,
                f"house_{year}_source_report_reconciliation",
                expected,
                f"reconcile={reconcile} artifacts={artifact_count} "
                f"bindings={binding_violations[:10]} "
                f"provenance={provenance_violations[:10]}",
            )
        # 6. Senate completeness is read from the active local source state.
        # A frozen Senate window is equivalent to a persisted refresh once its
        # rows and report inventory have been ingested into this database.
        senate = manifest.get("senate")
        senate_window = manifest.get("senate_window")
        if not isinstance(senate, dict):
            senate = {}
        if (
            senate.get("status") != "persisted"
            and isinstance(senate_window, dict)
            and senate_window.get("status") == "ingested"
        ):
            senate = senate_window
        senate_status_ok = senate.get("status") in ("persisted", "ingested")
        intended_senate_generation = senate.get("generation") or manifest.get("generation")
        _check(
            checks,
            "senate_status_persisted",
            senate_status_ok,
            f"status={senate.get('status')!r}",
        )

        active_senate_generations = {
            str(row[0])
            for row in db.conn.execute(
                """
                SELECT DISTINCT ingestion_generation
                FROM source_reports
                WHERE source = 'senate_efd' AND chamber = 'senate'
                UNION
                SELECT DISTINCT ingestion_generation
                FROM transactions
                WHERE source = 'senate_efd' AND chamber = 'senate'
                """
            ).fetchall()
            if row[0] is not None
        }
        senate_null_generation = db.conn.execute(
            """
            SELECT COUNT(*) FROM (
                SELECT ingestion_generation FROM source_reports
                WHERE source = 'senate_efd' AND chamber = 'senate'
                UNION ALL
                SELECT ingestion_generation FROM transactions
                WHERE source = 'senate_efd' AND chamber = 'senate'
            )
            WHERE ingestion_generation IS NULL OR TRIM(ingestion_generation) = ''
            """
        ).fetchone()[0]
        senate_identity_ok = (
            isinstance(intended_senate_generation, str)
            and bool(intended_senate_generation.strip())
            and active_senate_generations == {intended_senate_generation}
            and int(senate_null_generation) == 0
        )
        _check(
            checks,
            "senate_generation_identity_consistent",
            senate_identity_ok,
            f"intended={intended_senate_generation!r} "
            f"active={sorted(active_senate_generations)} null={senate_null_generation}",
        )
        if isinstance(intended_senate_generation, str) and intended_senate_generation.strip():
            senate_reconcile = db.source_reports.reconcile(
                intended_senate_generation, "senate_efd", "senate"
            )
        else:
            senate_reconcile = {
                name: 0
                for name in (
                    "found",
                    "parsed",
                    "no_txs",
                    "paper_only",
                    "unavailable",
                    "failed",
                )
            }
        _check(
            checks,
            "senate_source_reports_exist",
            senate_reconcile["found"] > 0,
            f"generation={intended_senate_generation!r} reconcile={senate_reconcile}",
        )
        senate_provenance_violations = (
            _source_report_provenance_violations(db, intended_senate_generation)
            if isinstance(intended_senate_generation, str)
            and intended_senate_generation.strip()
            else []
        )
        _check(
            checks,
            "senate_report_reconciliation",
            not senate_provenance_violations
            and senate_reconcile["found"] == (
                senate_reconcile["parsed"]
                + senate_reconcile["no_txs"]
                + senate_reconcile["paper_only"]
                + senate_reconcile["unavailable"]
                + senate_reconcile["failed"]
            ),
            f"{senate_reconcile} provenance={senate_provenance_violations[:10]}",
        )
        _check(
            checks,
            "senate_zero_unavailable_failed",
            senate_reconcile["unavailable"] == 0
            and senate_reconcile["failed"] == 0,
            str(senate_reconcile),
        )
        senate_count_mismatches = []
        senate_missing_reports = []
        if isinstance(intended_senate_generation, str) and intended_senate_generation.strip():
            senate_count_mismatches = db.conn.execute(
                """
                SELECT r.source_record_id, r.accepted_row_count, COUNT(t.id) AS actual
                FROM source_reports r
                LEFT JOIN transactions t
                  ON t.ingestion_generation = r.ingestion_generation
                 AND t.source = r.source
                 AND t.chamber = r.chamber
                 AND t.source_record_id = r.source_record_id
                WHERE r.ingestion_generation = ?
                  AND r.source = 'senate_efd' AND r.chamber = 'senate'
                GROUP BY 1, 2
                HAVING r.accepted_row_count != COUNT(t.id)
                ORDER BY 1
                LIMIT 10
                """,
                [intended_senate_generation],
            ).fetchall()
            senate_missing_reports = db.conn.execute(
                """
                SELECT t.source_record_id, COUNT(*) AS rows
                FROM transactions t
                LEFT JOIN source_reports r
                  ON r.ingestion_generation = t.ingestion_generation
                 AND r.source = t.source
                 AND r.chamber = t.chamber
                 AND r.source_record_id = t.source_record_id
                WHERE t.ingestion_generation = ?
                  AND t.source = 'senate_efd' AND t.chamber = 'senate'
                  AND r.source_record_id IS NULL
                GROUP BY 1
                ORDER BY 1
                LIMIT 10
                """,
                [intended_senate_generation],
            ).fetchall()
        _check(
            checks,
            "senate_accepted_counts_equal_persisted_rows",
            not senate_count_mismatches,
            f"mismatches={senate_count_mismatches[:10]}",
        )
        _check(
            checks,
            "senate_transactions_have_source_reports",
            not senate_missing_reports,
            f"missing={senate_missing_reports[:10]}",
        )
        declared_summary = senate.get("summary")
        if isinstance(declared_summary, dict) and declared_summary:
            # Senate manifests use the Senate-specific ReportOutcome
            # contract, which has no no_txs value. Keep no_txs in the
            # database equation (it must remain zero) without requiring the
            # manifest to invent a field it never emits.
            summary_fields = (
                "found",
                "parsed",
                "paper_only",
                "unavailable",
                "failed",
            )
            summary_ok = all(
                field in declared_summary
                and declared_summary[field] == senate_reconcile[field]
                for field in summary_fields
            ) and senate_reconcile["no_txs"] == 0
        else:
            summary_ok = True
        _check(
            checks,
            "senate_manifest_summary_consistent",
            summary_ok,
            f"manifest={declared_summary} database={senate_reconcile}",
        )

        # 7. The canonical view must contain exactly the active Senate rows
        # plus rows from the latest semantically accepted House generation for
        # each year. ``get_unresolved_house_doc_ids`` is the same acceptance
        # predicate used by Database._init_canonical_transactions_view.
        canonical_extra, canonical_missing = _canonical_view_diff_counts(db)
        _check(
            checks,
            "canonical_view_complete_generations_only",
            int(canonical_extra) == 0 and int(canonical_missing) == 0,
            f"canonical_extra={canonical_extra} canonical_missing={canonical_missing}",
        )

        # 8. Price availability is explicit: unresolved tickers remain a
        # diagnostic, never an implicit zero-return observation.
        price_record = manifest.get("prices")
        if not isinstance(price_record, dict):
            price_record = {}
        snapshot_path = staging / "price_snapshot.json"
        price_detail: dict = {
            "status": price_record.get("status"),
            "snapshot": str(snapshot_path),
            "unresolved_tickers": price_record.get("unresolved_tickers", []),
        }
        price_ok = (
            price_record.get("status") == "snapshotted"
            and snapshot_path.is_file()
        )
        try:
            snapshot = json.loads(snapshot_path.read_text())
            coverage = snapshot.get("coverage_by_ticker")
            tickers = list(coverage) if isinstance(coverage, dict) else []
            unresolved = sorted(
                str(ticker) for ticker in snapshot.get("unresolved_tickers", [])
            )
            record_unresolved = sorted(
                str(ticker)
                for ticker in price_record.get("unresolved_tickers", [])
            )
            fields_match = all(
                price_record.get(field) == snapshot.get(field)
                for field in (
                    "requested_tickers",
                    "resolved_tickers",
                    "price_rows",
                    "value_hash",
                )
            ) and unresolved == record_unresolved
            hash_ok = bool(
                re.fullmatch(
                    r"[0-9a-fA-F]{64}", str(snapshot.get("value_hash", ""))
                )
            )
            coverage_ok = (
                isinstance(coverage, dict)
                and snapshot.get("requested_tickers") == len(tickers)
                and snapshot.get("resolved_tickers") + len(unresolved)
                == snapshot.get("requested_tickers")
                and set(unresolved).issubset(tickers)
            )
            actual_rows = actual_tickers = None
            if (
                price_record.get("start_date")
                and price_record.get("end_date")
                and tickers
            ):
                actual_rows, actual_tickers = db.conn.execute(
                    """
                    SELECT COUNT(*), COUNT(DISTINCT ticker)
                    FROM prices
                    WHERE date BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
                      AND ticker IN (SELECT UNNEST(?))
                      AND close > 0 AND isfinite(close)
                    """,
                    [
                        price_record["start_date"],
                        price_record["end_date"],
                        tickers,
                    ],
                ).fetchone()
            db_counts_ok = (
                actual_rows is not None
                and int(actual_rows) == int(snapshot.get("price_rows", -1))
                and int(actual_tickers) == int(snapshot.get("resolved_tickers", -1))
            )
            price_detail.update(
                {
                    "requested_tickers": snapshot.get("requested_tickers"),
                    "resolved_tickers": snapshot.get("resolved_tickers"),
                    "price_rows": snapshot.get("price_rows"),
                    "coverage": coverage_ok,
                    "database_counts": db_counts_ok,
                    "hash_format": hash_ok,
                }
            )
            price_ok = (
                price_ok
                and fields_match
                and hash_ok
                and coverage_ok
                and db_counts_ok
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError, duckdb.Error) as exc:
            price_detail["error"] = f"{type(exc).__name__}: {exc}"
            price_ok = False
        _check(
            checks,
            "price_diagnostics",
            price_ok,
            json.dumps(price_detail, sort_keys=True, default=str),
        )
    finally:
        db.close()

    failed = [name for name, check in checks.items() if not check["passed"]]
    manifest["verify"] = {
        "incomplete_years": sorted(set(incomplete_years)),
        "generation_complete": not failed,
        "incomplete_reasons": (
            [f"house_year_{year}" for year in sorted(set(incomplete_years))]
            + (
                []
                if checks.get("senate_status_persisted", {}).get("passed")
                else ["senate"]
            )
        ),
        "db_fingerprint": _database_fingerprint(db_path),
        "checks": checks,
    }
    _save_manifest(staging, manifest)
    print(f"verify: {len(checks) - len(failed)}/{len(checks)} checks passed")
    if failed:
        print(f"verify: FAILED {failed}")
        raise SystemExit(1)


def promote(args) -> None:
    """Install a verified staged database with an atomic filesystem swap."""
    staging = Path(args.staging)
    staged_db = staging / "congress.duckdb"
    live_db = _REPO_ROOT / "data" / "congress.duckdb"
    if not staged_db.is_file():
        raise SystemExit(f"staged database not found: {staged_db}")
    if staged_db.resolve() == live_db.resolve():
        raise SystemExit("staged database must not be the live database")

    # A checkpoint is the only writer interaction in this command. The live
    # database is never opened by DuckDB; it is copied and replaced as a file.
    _checkpoint_database(staged_db)
    verify(args)
    manifest = _load_manifest(staging)
    verification = manifest.get("verify") or {}
    failed = [
        name
        for name, check in (verification.get("checks") or {}).items()
        if not check.get("passed")
    ]
    if (
        failed
        or not verification.get("generation_complete")
        or verification.get("incomplete_years")
    ):
        raise SystemExit(
            "promote refused: staged verification is incomplete"
            + (f" ({failed})" if failed else "")
        )
    expected_fingerprint = verification.get("db_fingerprint")
    actual_fingerprint = _database_fingerprint(staged_db)
    if actual_fingerprint != expected_fingerprint:
        raise SystemExit(
            "promote refused: staged database changed after verification"
        )

    live_db.parent.mkdir(parents=True, exist_ok=True)
    backup = live_db.with_name(
        f"{live_db.name}.backup-{time.strftime('%Y%m%dT%H%M%S')}"
    )
    if backup.exists():
        backup = live_db.with_name(f"{backup.name}-{time.time_ns()}")
    if live_db.exists():
        shutil.copy2(live_db, backup)
        print(f"promote: backup {backup}")
    os.replace(staged_db, live_db)
    print(f"promote: installed {live_db} from {staging}")


# --------------------------------------------------------------------------
# Audit-gap repair (invariants: C1/C2/C3/C9)
# --------------------------------------------------------------------------

def repair_audit_gaps(args) -> None:
    """Make the staged DB satisfy the luna-invariants generation audit.

    1. Chronology: move every house row with transaction_date >
       disclosure_date (or outside the date domain) into
       house_transaction_quarantine (reason 'chronology_invalid'), then
       delete it. Docs that drop to zero rows become zero_rows (unresolved).
    2. Parse runs: transaction_count updated to the post-quarantine persisted
       count; only one terminal run per (doc, generation) remains (stale
       error/zero_rows runs are removed when a success run exists).
    3. source_row_id: every house row gets a non-blank deterministic
       source_row_id (fallback '<doc_id>:seq:<n>' when the parser emitted none).
    """
    staging = Path(args.staging)
    manifest = _load_manifest(staging)
    db = Database(staging / "congress.duckdb", read_only=False)
    try:
        from datetime import timedelta  # noqa: PLC0415

        today_plus = date.today() + timedelta(days=1)

        # 1. quarantine invalid-chronology / out-of-domain rows
        invalid = db.conn.execute(
            """
            SELECT id, doc_id, ingestion_generation, to_json(t) AS payload
            FROM transactions t
            WHERE source = 'house_pdf'
              AND (
                transaction_date IS NULL OR disclosure_date IS NULL
                OR transaction_date > disclosure_date
                OR transaction_date < DATE '1900-01-01'
                OR transaction_date > ?
              )
            """,
            [today_plus],
        ).fetchall()
        per_year: dict[str, int] = {}
        for row_id, doc_id, generation, payload in invalid:
            year = db.conn.execute(
                """
                SELECT archive_year FROM house_generation_metadata
                WHERE doc_id = ? AND generation_id = ? LIMIT 1
                """,
                [doc_id, generation],
            ).fetchone()
            archive_year = int(year[0]) if year else None
            if archive_year is None:
                continue
            db.conn.execute(
                """
                INSERT INTO house_transaction_quarantine (
                    archive_year, doc_id, generation_id, transaction_id,
                    transaction_json, reason
                ) VALUES (?, ?, ?, ?, ?, 'chronology_invalid')
                """,
                [archive_year, doc_id, generation, row_id, payload],
            )
            key = str(archive_year)
            per_year[key] = per_year.get(key, 0) + 1
        n_invalid = len(invalid)
        if n_invalid:
            db.conn.execute(
                """
                DELETE FROM transactions
                WHERE source = 'house_pdf'
                  AND (
                    transaction_date IS NULL OR disclosure_date IS NULL
                    OR transaction_date > disclosure_date
                    OR transaction_date < DATE '1900-01-01'
                    OR transaction_date > ?
                  )
                """,
                [today_plus],
            )
        # 2a. update parse-run counts to post-quarantine persisted counts
        db.conn.execute("""
            UPDATE pdf_parse_runs p
            SET transaction_count = (
                SELECT COUNT(*) FROM transactions t
                WHERE t.doc_id = p.doc_id
                  AND t.source = 'house_pdf'
                  AND t.ingestion_generation = p.ingestion_generation
            )
            WHERE p.status = 'success'
        """)
        # 2b. docs that dropped to zero rows become unresolved (zero_rows)
        db.conn.execute("""
            UPDATE pdf_parse_runs p
            SET status = 'zero_rows', transaction_count = 0,
                error_message = 'all extracted rows quarantined for invalid chronology'
            WHERE p.status = 'success'
              AND NOT EXISTS (
                SELECT 1 FROM transactions t
                WHERE t.doc_id = p.doc_id
                  AND t.source = 'house_pdf'
                  AND t.ingestion_generation = p.ingestion_generation
              )
        """)
        # 2c. keep only the terminal run per (doc, generation) when one exists
        db.conn.execute("""
            DELETE FROM pdf_parse_runs p
            USING pdf_parse_runs terminal
            WHERE terminal.doc_id = p.doc_id
              AND terminal.ingestion_generation = p.ingestion_generation
              AND terminal.status IN ('success', 'no_txs')
              AND p.status NOT IN ('success', 'no_txs')
        """)
        # 3. non-blank deterministic source_row_id for house rows
        db.conn.execute("""
            UPDATE transactions SET source_row_id =
                doc_id || ':seq:' || rn
            FROM (
                SELECT id, ROW_NUMBER() OVER (
                    PARTITION BY doc_id, ingestion_generation ORDER BY id
                ) AS rn
                FROM transactions
                WHERE source = 'house_pdf'
                  AND (source_row_id IS NULL OR TRIM(source_row_id) = '')
            ) numbered
            WHERE transactions.id = numbered.id
        """)
        # 4. notification_date must be NULL or within domain and after the
        # transaction date (audit C9 ordering rule); unverifiable OCR
        # notification claims are nulled, never guessed.
        bad_notifications = db.conn.execute(
            """
            SELECT id, doc_id, ingestion_generation, to_json(t) AS payload
            FROM transactions t
            WHERE notification_date IS NOT NULL
              AND (
                notification_date < DATE '1900-01-01'
                OR notification_date > ?
                OR (transaction_date IS NOT NULL
                    AND notification_date < transaction_date)
              )
            """,
            [today_plus],
        ).fetchall()
        for row_id, doc_id, generation, payload in bad_notifications:
            year = db.conn.execute(
                """
                SELECT archive_year FROM house_generation_metadata
                WHERE doc_id = ? AND generation_id = ? LIMIT 1
                """,
                [doc_id, generation],
            ).fetchone()
            archive_year = int(year[0]) if year else None
            if archive_year is None:
                continue
            db.conn.execute(
                """
                INSERT INTO house_transaction_quarantine (
                    archive_year, doc_id, generation_id, transaction_id,
                    transaction_json, reason
                ) VALUES (?, ?, ?, ?, ?, 'notification_date_invalid')
                """,
                [archive_year, doc_id, generation, row_id, payload],
            )
        db.conn.execute(
            """
            UPDATE transactions SET notification_date = NULL
            WHERE notification_date IS NOT NULL
              AND (
                notification_date < DATE '1900-01-01'
                OR notification_date > ?
                OR (transaction_date IS NOT NULL
                    AND notification_date < transaction_date)
              )
            """,
            [today_plus],
        )
        manifest["repair"] = {
            "chronology_quarantined_rows": n_invalid,
            "chronology_quarantined_by_year": per_year,
            "nulled_invalid_notification_dates": int(
                db.conn.execute(
                    """
                    SELECT COUNT(*) FROM house_transaction_quarantine
                    WHERE reason = 'notification_date_invalid'
                    """
                ).fetchone()[0]
            ),
            "source_row_id_backfilled": int(
                db.conn.execute(
                    """
                    SELECT COUNT(*) FROM transactions
                    WHERE source = 'house_pdf'
                      AND source_row_id LIKE ':seq:'
                    """
                ).fetchone()[0]
            ),
        }
        # refresh house manifest unresolved state
        for year in HOUSE_YEARS:
            house = manifest.get("house", {}).get(str(year))
            if house is None:
                continue
            gen = house["generation_id"]
            house["unresolved_doc_ids"] = db.get_unresolved_house_doc_ids(year, gen)
            house["resolved_doc_count"] = house["ptr_count"] - len(
                house["unresolved_doc_ids"]
            )
            pr = house.setdefault("parse_result", {})
            pr["persisted_transactions"] = int(
                db.conn.execute(
                    "SELECT COUNT(*) FROM transactions WHERE source='house_pdf' AND ingestion_generation=?",
                    [gen],
                ).fetchone()[0]
            )
        _save_manifest(staging, manifest)
        print(
            f"repair: quarantined {n_invalid} invalid-chronology rows; "
            f"source_row_id backfilled; parse runs reconciled"
        )
    finally:
        db.close()


# --------------------------------------------------------------------------
# Sibling track consumption
# --------------------------------------------------------------------------

SIBLING_TRACKS = ("senate", "ocr", "prices", "capitol", "metadata-audit", "invariants")


def _verify_artifact_files(manifest: dict, base: Path) -> dict:
    """Verify every artifact path in a sibling manifest against its sha256."""
    results = {"files_checked": 0, "mismatches": []}
    if not isinstance(manifest, dict):
        return results
    artifacts = (
        manifest.get("artifacts")
        or manifest.get("files")
        or manifest.get("staged_files_sha256")
        or {}
    )
    if not isinstance(artifacts, dict):
        return results
    if manifest.get("staged_files_sha256") and not (
        manifest.get("artifacts") or manifest.get("files")
    ):
        # staged_files_sha256 maps relative path -> sha256 string.
        artifacts = {
            name: {"path": name, "sha256": value}
            for name, value in artifacts.items()
        }
    _HEX64 = re.compile(r"^[0-9a-f]{64}$")
    for name, meta in artifacts.items():
        if isinstance(meta, str):
            if _HEX64.fullmatch(meta):
                expected_sha = meta
                path = base / name
            else:
                # Filename reference without a per-file hash claim; verify
                # existence only (value hashes are checked by the ingester).
                path = base / meta
                results["files_checked"] += 1
                if not path.exists():
                    results["mismatches"].append(f"{name}: missing ({meta})")
                continue
        elif isinstance(meta, dict):
            expected_sha = meta.get("sha256")
            path = base / (meta.get("path") or name)
        else:
            continue
        if not expected_sha or not isinstance(expected_sha, str):
            continue
        if not path.exists():
            results["mismatches"].append(f"{name}: missing")
            continue
        actual = _sha256_file(path)
        results["files_checked"] += 1
        if actual != expected_sha:
            results["mismatches"].append(f"{name}: sha {actual[:16]} != {expected_sha[:16]}")
    return results


def _sibling_search_dirs(track: str, generation: str) -> list[Path]:
    """Locate a sibling track's artifact dir in any sibling worktree."""
    candidates = [
        _REPO_ROOT / "data" / ".staging" / track / generation,
        _REPO_ROOT / ".staging" / track / generation,
    ]
    worktrees_root = _REPO_ROOT.parents[0]
    if worktrees_root.name == ".worktrees":
        for worktree in sorted(worktrees_root.iterdir()):
            if not worktree.is_dir() or worktree.name == "luna-rebuild":
                continue
            candidates.append(worktree / ".staging" / track / generation)
            candidates.append(worktree / "data" / ".staging" / track / generation)
    return candidates


def _coerce_sibling_frame(frame, *, count_columns, date_columns, text_columns):
    """Normalize string-encoded sibling frames ('None' -> None, counts -> int)."""
    import pandas as pd  # noqa: PLC0415

    frame = frame.copy()
    for column in frame.columns:
        if column in count_columns:
            numeric = pd.to_numeric(frame[column], errors="coerce")
            if numeric.isna().any():
                raise ValueError(
                    f"sibling frame column {column!r} has non-numeric/missing values"
                )
            frame[column] = numeric.astype("int64")
        elif column in date_columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
        elif column in text_columns:
            frame[column] = frame[column].map(
                lambda value: None
                if value is None or pd.isna(value)
                else str(value).strip() or None
            )
    return frame


def _ingest_senate(track_dir: Path, track_manifest: dict, db: Database) -> dict:
    """Persist a verified senate sweep via Database.persist_source_refresh."""
    import pandas as pd  # noqa: PLC0415

    generation = track_manifest["generation"]
    tx = pd.read_json(track_dir / "transactions.jsonl", lines=True)
    inv = pd.read_json(track_dir / "report_inventory.jsonl", lines=True)
    count_cols = ["raw_row_count", "accepted_row_count", "rejected_row_count"]
    date_cols = ["official_filing_date", "available_date", "disclosure_date",
                 "transaction_date", "notification_date", "filing_date"]
    text_cols = [
        "amends_source_record_id", "artifact_sha256", "asset_description",
        "chamber", "chamber_member_key", "doc_id", "expiry_date",
        "ingestion_generation", "instrument_type", "member", "member_key",
        "owner_code", "raw_asset_class", "raw_asset_description", "raw_owner",
        "raw_ticker", "raw_transaction_subtype", "source_record_id",
        "source_report_path", "source_row_id", "strike_price", "ticker",
        "ticker_candidate", "ticker_origin", "transaction_type",
        "landing_sha256", "paper_artifact_sha256", "paper_artifact_url",
        "error_message", "outcome", "report_path", "source",
    ]
    tx = _coerce_sibling_frame(tx, count_columns=[], date_columns=date_cols, text_columns=text_cols)
    inv = _coerce_sibling_frame(inv, count_columns=count_cols, date_columns=date_cols, text_columns=text_cols)
    for label, frame in (("transactions", tx), ("report inventory", inv)):
        if "ingestion_generation" not in frame.columns:
            raise ValueError(f"senate {label} lacks ingestion_generation provenance")
        generations = {
            str(value).strip()
            for value in frame["ingestion_generation"].dropna().tolist()
            if str(value).strip()
        }
        null_generation_rows = int(frame["ingestion_generation"].isna().sum())
        if len(frame) and (
            generations != {generation} or null_generation_rows != 0
        ):
            raise ValueError(
                f"senate {label} generation provenance mismatch: "
                f"manifest={generation!r} rows={sorted(generations)!r} "
                f"null_rows={null_generation_rows}"
            )
    inserted = db.persist_source_refresh(
        transactions=tx,
        reports=inv,
        source="senate_efd",
        chamber="senate",
        ingestion_generation=generation,
    )
    return {
        "inserted_transactions": inserted,
        "reports": len(inv),
        "transactions_frame_rows": len(tx),
        "summary": track_manifest.get("summary") or track_manifest.get("outcome_counts"),
        "canaries": track_manifest.get("canaries"),
        "window": track_manifest.get("window"),
    }


def _ingest_prices(track_dir: Path, track_manifest: dict, db: Database) -> dict:
    """Upsert a value-verified price refresh into the staged DB."""
    import pandas as pd  # noqa: PLC0415

    from analyzer.price_snapshot import _hash_price_values  # noqa: PLC0415

    parquet_file = track_dir / "prices.parquet"
    duckdb_file = track_dir / "refresh.duckdb"
    if parquet_file.exists():
        prices = pd.read_parquet(parquet_file)
    elif duckdb_file.exists():
        import duckdb as _duckdb  # noqa: PLC0415

        src_conn = _duckdb.connect(str(duckdb_file), read_only=True)
        try:
            tables = [
                r[0]
                for r in src_conn.execute(
                    "SELECT table_name FROM information_schema.tables ORDER BY 1"
                ).fetchall()
            ]
            if "prices" not in tables:
                raise ValueError(
                    f"price refresh duckdb has no prices table: {tables}"
                )
            prices = src_conn.execute(
                "SELECT ticker, date, close FROM prices WHERE close > 0 AND isfinite(close)"
            ).fetchdf()
        finally:
            src_conn.close()
    else:
        raise FileNotFoundError(f"price refresh artifacts missing: {track_dir}")
    if prices.empty:
        raise ValueError("price refresh contains no price rows")
    for column in ("ticker", "date", "close"):
        if column not in prices.columns:
            raise ValueError(f"price refresh missing column {column!r}")
    pivot = prices.pivot(index="date", columns="ticker", values="close")
    pivot.index = pd.DatetimeIndex(pd.to_datetime(pivot.index)).normalize()
    computed_hash = _hash_price_values(pivot)
    expected_hash = track_manifest.get("value_hash") or track_manifest.get("data_hash")
    if expected_hash and computed_hash != expected_hash:
        raise ValueError(
            f"price refresh value hash mismatch: computed={computed_hash[:16]} "
            f"manifest={expected_hash[:16]}"
        )
    db.upsert_prices(pivot)
    return {
        "upserted_rows": len(prices),
        "tickers": len(prices["ticker"].unique()),
        "range": f"{prices['date'].min()}..{prices['date'].max()}",
        "value_hash_verified": computed_hash,
        "value_hash_matches_manifest": bool(expected_hash) and computed_hash == expected_hash,
        "source": "prices.parquet" if parquet_file.exists() else "refresh.duckdb",
    }


def _ingest_ocr(track_dir: Path, track_manifest: dict, db: Database) -> dict:
    """Ingest verified local-OCR rows for unresolved House scans.

    Fail-closed guards: docs the track marked unresolved are never ingested
    (8221322 stays quarantined); rows with unparseable dates or transaction
    dates after the filing date are dropped with exact per-doc reporting.
    """
    import pandas as pd  # noqa: PLC0415

    from datetime import timedelta as _timedelta  # noqa: PLC0415

    today_plus = date.today() + _timedelta(days=1)
    rows_files = sorted(track_dir.glob("rows/*.jsonl"))
    track_unresolved = sorted((track_manifest.get("unresolved") or {}).keys())
    if not rows_files:
        return {"rows_files": [], "track_unresolved": track_unresolved, "ingested_docs": []}
    parser_version = str(
        track_manifest.get("parser_version")
        or f"v4-{track_manifest.get('engine', 'local-ocr')}"
    )
    total_ingested = 0
    ingested_docs: list[str] = []
    skipped_already_resolved: list[str] = []
    skipped_track_unresolved: list[str] = []
    dropped_rows: list[dict] = []
    for rows_file in rows_files:
        rows = pd.read_json(rows_file, lines=True)
        file_doc_id = rows_file.stem
        if rows.empty:
            if file_doc_id in track_unresolved:
                skipped_track_unresolved.append(file_doc_id)
            continue
        doc_id = rows["doc_id"].astype(str).iloc[0]
        if doc_id in track_unresolved:
            skipped_track_unresolved.append(doc_id)
            continue
        meta = db.conn.execute(
            """
            SELECT m.archive_year, g.generation_id,
                   m.first_name, m.last_name, m.filing_date,
                   a.artifact_sha256
            FROM house_generation_metadata m
            JOIN house_archive_generations g
              ON g.archive_year = m.archive_year
             AND g.generation_id = m.generation_id
            JOIN house_pdf_artifacts a
              ON a.archive_year = m.archive_year
             AND a.generation_id = m.generation_id
             AND a.doc_id = m.doc_id
            WHERE m.doc_id = ? AND m.archive_year = ?
            LIMIT 1
            """,
            [doc_id, int(track_manifest.get("year") or 2026)],
        ).fetchone()
        if meta is None:
            raise RuntimeError(f"ocr ingest: no staged house metadata for {doc_id}")
        archive_year, gen, first, last, filing_date, artifact_sha = meta
        resolved = db.conn.execute(
            """
            SELECT COUNT(*) FROM pdf_parse_runs
            WHERE doc_id = ? AND ingestion_generation = ?
              AND status IN ('success', 'no_txs')
            """,
            [doc_id, gen],
        ).fetchone()[0]
        if resolved:
            skipped_already_resolved.append(doc_id)
            continue
        filing_ts = pd.to_datetime(filing_date)
        frame = rows[rows["doc_id"].astype(str) == doc_id].copy()
        frame["transaction_date"] = pd.to_datetime(
            frame["transaction_date"], errors="coerce"
        )
        valid_mask = frame["transaction_date"].notna() & (
            frame["transaction_date"] <= filing_ts
        )
        n_dropped = int((~valid_mask).sum())
        if n_dropped:
            dropped_rows.append(
                {
                    "doc_id": doc_id,
                    "dropped": n_dropped,
                    "reasons": sorted(
                        str(value)
                        for value in frame.loc[~valid_mask, "transaction_date"].fillna("unparseable").unique()
                    ),
                }
            )
        frame = frame.loc[valid_mask].copy()
        if frame.empty:
            continue
        if "notification_date" in frame.columns:
            frame["notification_date"] = pd.to_datetime(
                frame["notification_date"], errors="coerce"
            )
            frame["notification_date"] = frame["notification_date"].where(
                frame["notification_date"].isna()
                | (
                    (frame["notification_date"] >= frame["transaction_date"])
                    & (frame["notification_date"] >= pd.Timestamp("1900-01-01"))
                    & (frame["notification_date"] <= pd.Timestamp(today_plus))
                )
            )
        frame["chamber"] = "house"
        frame["ingestion_generation"] = gen
        frame["source_record_id"] = doc_id
        frame["member"] = f"{first} {last}".strip()
        frame["disclosure_date"] = filing_ts
        frame["official_filing_date"] = filing_ts
        frame["artifact_sha256"] = artifact_sha
        frame["source_report_path"] = f"{archive_year}/pdfs/{doc_id}.pdf"
        attempted = [doc_id]
        count = len(frame)
        parse_runs = [
            dict(
                doc_id=doc_id,
                year=int(archive_year),
                parser_version=parser_version,
                status="success",
                engines_attempted=f"local_tesseract:rows:{count}",
                raw_row_count=count,
                transaction_count=0,
                artifact_sha256=artifact_sha,
                ingestion_generation=gen,
            )
        ]
        persisted = db.replace_transactions_for_docs(
            frame,
            source="house_pdf",
            attempted_doc_ids=attempted,
            ingestion_generation=gen,
            replacement_doc_ids=attempted,
            parse_runs=parse_runs,
        )
        total_ingested += sum(persisted.by_doc_total.values())
        ingested_docs.append(doc_id)
    return {
        "rows_files": [f.name for f in rows_files],
        "track_unresolved": track_unresolved,
        "skipped_track_unresolved": skipped_track_unresolved,
        "skipped_already_resolved": skipped_already_resolved,
        "ingested_docs": sorted(ingested_docs),
        "ingested_rows": total_ingested,
        "dropped_rows": dropped_rows,
        "parser_version": parser_version,
    }


def consume(args) -> None:
    staging = Path(args.staging)
    manifest = _load_manifest(staging)
    generation = manifest["generation"]
    db = Database(staging / "congress.duckdb", read_only=False)
    try:
        tracks = manifest.setdefault("consume", {})
        for track in SIBLING_TRACKS:
            entry = tracks.get(track, {})
            if entry.get("status") in ("ingested", "quarantined") and not args.force:
                print(f"consume {track}: skipped (already {entry['status']})")
                continue
            track_dir = None
            for candidate in _sibling_search_dirs(track, generation):
                if candidate.exists():
                    track_dir = candidate
                    break
            if track_dir is None:
                tracks[track] = {"status": "absent"}
                print(f"consume {track}: ABSENT")
                continue
            mpath = track_dir / "manifest.json"
            if not mpath.exists():
                tracks[track] = {"status": "unverified", "path": str(track_dir), "reason": "no manifest.json"}
                print(f"consume {track}: UNVERIFIED (no manifest.json)")
                continue
            track_manifest = json.loads(mpath.read_text())
            verification = _verify_artifact_files(track_manifest, track_dir)
            entry = {
                "status": "present",
                "path": str(track_dir),
                "verification": verification,
            }
            if verification["mismatches"]:
                entry["status"] = "mismatch"
                tracks[track] = entry
                print(f"consume {track}: MISMATCH — {verification['mismatches']}")
                continue
            try:
                if track == "senate":
                    entry["ingest"] = _ingest_senate(track_dir, track_manifest, db)
                    entry["status"] = "ingested"
                elif track == "prices":
                    entry["ingest"] = _ingest_prices(track_dir, track_manifest, db)
                    entry["status"] = "ingested"
                elif track == "ocr":
                    entry["ingest"] = _ingest_ocr(track_dir, track_manifest, db)
                    entry["status"] = "ingested"
                else:
                    entry["status"] = "recorded"
                    entry["manifest"] = track_manifest
            except Exception as exc:  # noqa: BLE001 -- quarantine boundary
                entry["status"] = "quarantined"
                entry["error"] = f"{type(exc).__name__}: {exc}"
                print(f"consume {track}: QUARANTINED ({exc})")
            tracks[track] = entry
            print(f"consume {track}: {entry['status']}")
            _save_manifest(staging, manifest)
    finally:
        db.close()


def house_activate(args) -> None:
    """Re-check completeness after OCR consumption; activate complete years."""
    staging = Path(args.staging)
    manifest = _load_manifest(staging)
    db = Database(staging / "congress.duckdb", read_only=False)
    try:
        for year in HOUSE_YEARS:
            house = manifest.get("house", {}).get(str(year))
            if house is None:
                continue
            if house.get("parse_status") == "complete" and not args.force:
                continue
            unresolved, report_count = _refresh_house_completion(db, house, year)
            if unresolved:
                print(
                    f"house-activate {year}: INCOMPLETE — {len(unresolved)} unresolved"
                )
            else:
                print(f"house-activate {year}: COMPLETE — {report_count} inventory rows")
            _save_manifest(staging, manifest)
    finally:
        db.close()


# --------------------------------------------------------------------------
# Manifest + verdict
# --------------------------------------------------------------------------

def finalize(args) -> None:
    staging = Path(args.staging)
    manifest = _load_manifest(staging)
    house = manifest.get("house", {})
    senate_window = manifest.get("senate_window")
    if senate_window and senate_window.get("status") == "ingested":
        senate = {
            "status": "consumed",
            "start_date": senate_window.get("window", {}).get("start_date"),
            "end_date": senate_window.get("window", {}).get("end_date"),
            "summary": senate_window.get("summary"),
            "inserted_transactions": senate_window.get("inserted_transactions"),
            "generation": senate_window.get("generation"),
        }
    else:
        senate = manifest.get("senate", {}) or {
            "status": "consumed",
            "start_date": "2025-08-09",
            "end_date": "2026-08-09",
            "summary": manifest.get("consume", {})
            .get("senate", {})
            .get("ingest", {})
            .get("summary", {}),
            "inserted_transactions": manifest.get("consume", {})
            .get("senate", {})
            .get("ingest", {})
            .get("inserted_transactions", 0),
        }
    prices = manifest.get("prices", {})
    verify = manifest.get("verify", {})

    total_artifacts = sum(
        entry.get("artifact_count", 0) for entry in house.values()
    )
    db = Database(staging / "congress.duckdb", read_only=True)
    try:
        total_house_rows = int(
            db.conn.execute(
                "SELECT COUNT(*) FROM transactions WHERE source='house_pdf'"
            ).fetchone()[0]
        )
        senate_rows = int(
            db.conn.execute(
                "SELECT COUNT(*) FROM transactions WHERE source='senate_efd'"
            ).fetchone()[0]
        )
    finally:
        db.close()

    resolved = sum(
        entry.get("resolved_doc_count", 0) for entry in house.values()
    )
    unresolved = {
        str(year): entry.get("unresolved_doc_ids", [])
        for year, entry in house.items()
        if entry.get("unresolved_doc_ids")
    }
    classification = {}
    for year, doc_ids in unresolved.items():
        text_layer = 0
        for doc_id in doc_ids:
            pdf = staging / str(year) / "pdfs" / f"{doc_id}.pdf"
            if not pdf.exists():
                continue
            try:
                probe = subprocess.run(  # noqa: S603
                    ["pdftotext", "-l", "1", str(pdf), "-"],
                    capture_output=True,
                    text=True,
                    timeout=15,
                    check=False,
                )
                if probe.stdout.strip():
                    text_layer += 1
            except Exception:  # noqa: BLE001 -- classification is best-effort
                continue
        classification[str(year)] = {
            "unresolved": len(doc_ids),
            "with_text_layer": text_layer,
            "image_only": len(doc_ids) - text_layer,
        }
    manifest["unresolved_classification"] = classification
    complete_years = sorted(
        int(year) for year, entry in house.items()
        if entry.get("parse_status") == "complete"
    )
    incomplete_years = sorted(
        int(year) for year, entry in house.items()
        if entry.get("parse_status") != "complete"
    )

    lines = [
        "STAGED AUTHORITATIVE CONGRESSIONAL DATABASE REBUILD — VERDICT",
        f"Generation: {manifest['generation']}",
        f"Created: {manifest['created_at']}  Git SHA: {manifest['git_sha']}",
        "",
        "House (2015-2026):",
        f"  archives fetched: {len(house)}; artifacts (PDFs with SHAs): {total_artifacts}",
        f"  complete (activated) generations: {len(complete_years)} ({complete_years or 'none'})",
        f"  incomplete generations: {len(incomplete_years)} ({incomplete_years or 'none'})",
        f"  resolved filings: {resolved}; unresolved filings: {sum(len(v) for v in unresolved.values())}",
        f"  persisted house rows (all generations): {total_house_rows}",
        f"  persisted senate rows: {senate_rows}",
    ]
    for year in incomplete_years:
        entry = house.get(str(year))
        if entry is None:
            lines.append(f"    {year}: no generation acquired (fetch failed/blocked)")
        else:
            missing = entry.get("unresolved_doc_ids", [])
            lines.append(
                f"    {year}: generation {entry['generation_id'][-8:]} incomplete — "
                f"{len(missing)} unresolved PDF(s) (fail-closed, not canonical)"
            )
    if senate and senate.get("summary"):
        lines.append("")
        lines.append(f"Senate (eFD {senate.get('start_date')}..{senate.get('end_date')}):")
        summary = senate.get("summary") or {}
        if senate.get("status") in ("persisted", "consumed"):
            lines.append(
                f"  persisted (frozen window {senate.get('generation', '')}): found={summary.get('found')} "
                f"parsed={summary.get('parsed')} "
                f"paper_only={summary.get('paper_only')} unavailable={summary.get('unavailable')} "
                f"failed={summary.get('failed')}; transactions={senate.get('inserted_transactions')}"
            )
        elif senate.get("status") == "quarantined":
            lines.append(
                f"  QUARANTINED (nothing persisted): found={summary.get('found')} "
                f"parsed={summary.get('parsed')} paper_only={summary.get('paper_only')} "
                f"unavailable={summary.get('unavailable')} failed={summary.get('failed')} — {senate.get('error')}"
            )
    if prices:
        lines.append("")
        lines.append("Prices:")
        lines.append(
            f"  snapshot through {prices.get('end_date')}: rows={prices.get('price_rows')} "
            f"tickers={prices.get('resolved_tickers')}/{prices.get('eligible_tickers_requested')} "
            f"range={prices.get('first_date')}..{prices.get('last_date')} "
            f"value_hash={prices.get('value_hash')}"
        )
        if prices.get("unresolved_tickers"):
            lines.append(
                f"  unresolved tickers ({len(prices['unresolved_tickers'])}): "
                + ", ".join(prices["unresolved_tickers"][:20])
            )
    lines.append("")
    lines.append("Invariants:")
    checks = verify.get("checks", {})
    if checks:
        for name in sorted(checks):
            check = checks[name]
            lines.append(f"  [{'PASS' if check['passed'] else 'FAIL'}] {name}")
    else:
        lines.append("  (verify stage not run)")
    lines.append("")
    lines.append(
        "Verdict: no profitability, alpha, or performance claims are made. "
        "This manifest attests only to data acquisition, provenance, and "
        "completeness accounting described above."
    )
    verdict = "\n".join(lines)
    manifest["verdict"] = verdict
    _save_manifest(staging, manifest)
    print(verdict)


def _bootstrap(args) -> None:
    root = _staging_root()
    root.mkdir(parents=True, exist_ok=True)
    generation = args.generation or time.strftime("rebuild-%Y%m%dT%H%M%S")
    staging = root / generation
    if staging.exists() and not args.force:
        print(f"bootstrap: {staging} already exists (use --force to recreate)")
        return
    staging.mkdir(parents=True, exist_ok=True)
    db = _fresh_database(staging)
    db.close()
    manifest = {
        "generation": generation,
        "created_at": _now(),
        "git_sha": _git_sha(),
        "scope": {
            "house_years": HOUSE_YEARS,
            "required_house_years": list(REQUIRED_HOUSE_YEARS),
            "senate_start": str(SENATE_START),
            "price_start": str(PRICE_START),
        },
    }
    _save_manifest(staging, manifest)
    print(f"bootstrap: {staging}")
    print(f"  fresh database: {staging / 'congress.duckdb'}")
    print(f"  manifest: {staging / 'manifest.json'}")


def _resolve_staging(args) -> Path:
    if args.staging:
        return Path(args.staging)
    latest = _latest_generation()
    if latest is None:
        raise SystemExit("no generation found; run bootstrap first")
    return _staging_root() / latest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging", help="staging dir (default: latest generation)")
    parser.add_argument("--force", action="store_true")
    sub = parser.add_subparsers(dest="stage", required=True)

    p_boot = sub.add_parser("bootstrap")
    p_boot.add_argument("--generation", help="explicit generation id")

    p_fetch = sub.add_parser("house-fetch")
    p_fetch.add_argument("--years", nargs="+", type=int)

    p_parse = sub.add_parser("house-parse")
    p_parse.add_argument("--years", nargs="+", type=int)

    p_cache_ocr = sub.add_parser("house-cache-ocr")
    p_cache_ocr.add_argument("--years", nargs="+", type=int)
    p_cache_ocr.add_argument("--cache-dir")

    sub.add_parser("senate")
    sub.add_parser("prices")
    sub.add_parser("consume")
    sub.add_parser("repair-audit-gaps")
    p_window = sub.add_parser("ingest-senate-window")
    p_window.add_argument("--window-dir")
    sub.add_parser("house-activate")
    sub.add_parser("verify")
    sub.add_parser("promote")
    sub.add_parser("finalize")

    args = parser.parse_args(argv)
    if args.stage == "bootstrap":
        _bootstrap(args)
        return
    staging = _resolve_staging(args)
    if not staging.exists():
        raise SystemExit(f"staging dir not found: {staging}")
    args.staging = str(staging)
    print(f"staging: {staging}")
    handlers = {
        "house-fetch": house_fetch,
        "house-parse": house_parse,
        "house-cache-ocr": house_cache_ocr,
        "senate": senate,
        "prices": prices,
        "consume": consume,
        "repair-audit-gaps": repair_audit_gaps,
        "ingest-senate-window": ingest_senate_window,
        "house-activate": house_activate,
        "verify": verify,
        "promote": promote,
        "finalize": finalize,
    }
    handlers[args.stage](args)


if __name__ == "__main__":
    main()
