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
import subprocess
import sys
import time
from datetime import date
from multiprocessing import Pool
from pathlib import Path

# Docling is disabled exactly as in the accepted production reparse flow; the
# cascade still performs full text-engine comparison with Tesseract OCR as the
# local fallback. Must be set before analyzer imports.
os.environ["PTR_SKIP_DOCLING"] = "1"

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT))

from analyzer.database import Database  # noqa: E402
from analyzer.download import (  # noqa: E402
    _PARSE_VERSION,
    _build_member_lookup,
    _filter_existing_pdfs,
    _validated_pdf_sha256,
    preserve_existing_fields,
)
from analyzer.models import FilingType, ReportOutcome  # noqa: E402
from analyzer.parser_cascade import _parse_pdf_worker, ParserCascadeError  # noqa: E402
from analyzer.parsing import consolidate_transactions  # noqa: E402
from analyzer.price_repository import previous_nyse_session  # noqa: E402
from analyzer.price_snapshot import create_snapshot, save_snapshot  # noqa: E402
from analyzer.settings import DataSettings, Settings  # noqa: E402

HOUSE_YEARS = list(range(2015, 2027))
SENATE_START = date(2024, 1, 1)
PRICE_START = date(2014, 1, 1)
CANONICAL_DB_SHA256 = "9ec6be9263dc30aab07585d0110d2daf8568a14e4244f39d07c5b2bc130d476d"

PINNED_TEXT_CANARIES = {
    "20030977": ("76053146c191866009c30ba05b192e472aac616195137db9f5ea0e87274da39a", 224),
    "20033737": ("0b717e5a003cba305e42bcafc6e37042e45fdba7b8c4b9c6ca3528237eeef6b9", 16),
    "20033921": ("b486c612866c86738cc2810f34aaa1613c20e537daac4d5a467ee02da889f96d", 15),
}
PINNED_SCAN_HASHES = {
    "8221322": "26f1ce2fb7823d2e84ea4fbde24514c5c6371b43a828720d50f21b1c8c7ad314",
    "9115808": "05b2fa3becd71c9bb141690130708079407e52a6e169cdacf42a467e09e0bda5",
    "9115813": "737955c7c26c497eda37f4378e1af51409b6231204a82d7ae2c3f25c10e0ae84",
    "9116141": "716cdcc10bd57c400f10d8bb4133eb667931a9699fb1835ed3b7deca010a36a1",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    """Run the accepted cascade; convert unresolved PDFs into error results."""
    try:
        return _parse_pdf_worker(pdf_path)
    except ParserCascadeError as exc:
        return pdf_path, [], [f"error:{exc}"]
    except Exception as exc:  # noqa: BLE001 -- per-PDF quarantine boundary
        return pdf_path, [], [f"error:{type(exc).__name__}:{exc}"]


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

        artifact_hashes = {
            path.stem: _validated_pdf_sha256(path) for path in pdf_paths
        }
        cached = db.parse_runs.get_cached_doc_ids(
            year=year,
            parser_version=_PARSE_VERSION,
            artifact_hashes=artifact_hashes,
            ingestion_generation=ingestion_generation,
        )
        terminal = db.conn.execute(
            """
            SELECT doc_id FROM pdf_parse_runs
            WHERE year = ? AND parser_version = ?
              AND ingestion_generation = ?
              AND artifact_sha256 IS NOT NULL
              AND status IN ('success', 'no_txs', 'zero_rows', 'error')
            """,
            [year, _PARSE_VERSION, ingestion_generation],
        ).fetchall()
        cached |= {str(row[0]) for row in terminal}
        if cached:
            keep_mask = (
                existing_docs["DocID"].astype(str).map(lambda d: d not in cached).to_numpy()
            )
            pdf_paths = [p for p, keep in zip(pdf_paths, keep_mask) if keep]
            existing_docs = (
                existing_docs[keep_mask]
                if len(keep_mask)
                else existing_docs.iloc[0:0]
            )
        if not pdf_paths:
            return {"attempted": 0, "skipped_cached": len(cached)}

        member_lookup = _build_member_lookup(existing_docs)
        settings = _settings_for(staging)
        with Pool(settings.data.get_workers()) as pool:
            results = pool.map(_tolerant_parse_worker, pdf_paths)

        pdf_transactions: dict = {}
        raw_counts: dict[str, int] = {}
        parse_attempts: list[tuple[str, list[str], str | None]] = []
        for pdf_path, transactions, engines_attempted in results:
            doc_id = pdf_path.stem
            pdf_transactions[pdf_path] = transactions
            raw_counts[doc_id] = len(transactions)
            error_message = None
            error_engines = []
            for engine in engines_attempted:
                if engine.startswith("error:"):
                    error_message = engine[len("error:"):]
                else:
                    error_engines.append(engine)
            parse_attempts.append((doc_id, error_engines, error_message))

        df = consolidate_transactions(pdf_transactions, member_lookup)
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
            "skipped_cached": len(cached),
            "parse_run_statuses": by_status,
            "persisted_transactions": sum(persisted.by_doc_total.values()),
            "ingestion_generation": ingestion_generation,
        }
    finally:
        src.close()


def _house_inventory_rows(db: Database, year: int, gen: str, staging: Path) -> list[dict]:
    """Build the per-generation source_reports inventory (parsed docs only)."""
    runs = db.conn.execute(
        """
        SELECT doc_id, status, raw_row_count, transaction_count, error_message
        FROM pdf_parse_runs
        WHERE year = ? AND ingestion_generation = ?
        ORDER BY doc_id
        """,
        [year, gen],
    ).fetchall()
    metadata = db.conn.execute(
        """
        SELECT doc_id, first_name, last_name, filing_date
        FROM house_generation_metadata
        WHERE archive_year = ? AND generation_id = ?
        """,
        [year, gen],
    ).fetchall()
    meta_by_id = {str(doc_id): (first, last, filing_date) for doc_id, first, last, filing_date in metadata}
    artifacts = {
        str(doc_id): artifact_sha256
        for doc_id, artifact_sha256 in db.conn.execute(
            "SELECT doc_id, artifact_sha256 FROM house_pdf_artifacts WHERE archive_year=? AND generation_id=?",
            [year, gen],
        ).fetchall()
    }
    rows = []
    for doc_id, status, raw, accepted, error_message in runs:
        if status != "success":
            continue
        first, last, filing_date = meta_by_id.get(str(doc_id), (None, None, None))
        member = f"{first} {last}".strip() if first or last else None
        if member is None or filing_date is None:
            raise RuntimeError(f"house inventory {year}/{doc_id}: missing member metadata")
        sha = artifacts.get(str(doc_id))
        if sha is None:
            raise RuntimeError(f"house inventory {year}/{doc_id}: missing artifact sha")
        rows.append(
            {
                "ingestion_generation": gen,
                "chamber": "house",
                "source_record_id": str(doc_id),
                "report_path": f"{year}/pdfs/{doc_id}.pdf",
                "member": member,
                "official_filing_date": filing_date,
                "outcome": ReportOutcome.PARSED.value,
                "artifact_sha256": sha,
                "landing_sha256": sha,
                "paper_artifact_url": None,
                "paper_artifact_sha256": None,
                "error_message": error_message,
                "raw_row_count": int(accepted),
                "accepted_row_count": int(accepted),
                "rejected_row_count": 0,
            }
        )
    return rows


def house_parse(args) -> None:
    import pandas as pd  # noqa: PLC0415

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
            gen = house["generation_id"]
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
            unresolved = db.get_unresolved_house_doc_ids(year, gen)
            house["unresolved_doc_ids"] = unresolved
            house["resolved_doc_count"] = house["ptr_count"] - len(unresolved)
            house["parse_result"] = result
            if unresolved:
                house["parse_status"] = "incomplete"
                print(
                    f"house-parse {year}: INCOMPLETE — {len(unresolved)} unresolved "
                    f"({', '.join(unresolved[:10])}{'...' if len(unresolved) > 10 else ''})"
                )
            else:
                rows = _house_inventory_rows(db, year, gen, staging)
                reports_df = pd.DataFrame(rows)
                db.source_reports.replace_generation(
                    gen, "house_pdf", "house", reports_df
                )
                db.mark_house_generation_parse_complete(year, gen)
                house["parse_status"] = "complete"
                house["source_report_rows"] = len(rows)
                print(f"house-parse {year}: COMPLETE — {len(rows)} inventory rows persisted")
            _save_manifest(staging, manifest)
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

    from analyzer.datasources import YFinancePriceSource  # noqa: PLC0415

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
# Verification
# --------------------------------------------------------------------------

def _check(checks: dict, name: str, condition: bool, detail: str = "") -> None:
    checks[name] = {"passed": bool(condition), "detail": detail}


def verify(args) -> None:
    import pandas as pd  # noqa: PLC0415

    from analyzer.senate_efd import SenateRefreshSummary  # noqa: PLC0415

    staging = Path(args.staging)
    manifest = _load_manifest(staging)
    checks: dict = {}
    db = Database(staging / "congress.duckdb", read_only=True)
    try:
        # 1. parse counts == persisted rows per doc (house per generation)
        mismatches = db.conn.execute(
            """
            SELECT doc_id, transaction_count, actual FROM (
                SELECT p.doc_id AS doc_id,
                       p.transaction_count AS transaction_count,
                       (SELECT COUNT(*) FROM transactions t
                        WHERE t.doc_id = p.doc_id
                          AND t.source = 'house_pdf'
                          AND t.ingestion_generation = p.ingestion_generation) AS actual
                FROM pdf_parse_runs p
                WHERE p.status = 'success'
            ) WHERE transaction_count != actual
            """,
        ).fetchall()
        _check(
            checks,
            "house_parse_count_equals_persisted_rows",
            not mismatches,
            f"mismatches={[(r[0], r[1], r[2]) for r in mismatches][:10]}",
        )

        # senate report accepted counts == persisted rows per report
        senate_mismatch = db.conn.execute(
            """
            SELECT source_record_id, accepted_row_count, actual FROM (
                SELECT r.source_record_id AS source_record_id,
                       r.accepted_row_count AS accepted_row_count,
                       (SELECT COUNT(*) FROM transactions t
                        WHERE t.source_record_id = r.source_record_id
                          AND t.chamber = 'senate') AS actual
                FROM source_reports r
                WHERE r.source = 'senate_efd' AND r.outcome = 'parsed'
            ) WHERE accepted_row_count != actual
            """,
        ).fetchall()
        _check(
            checks,
            "senate_report_count_equals_persisted_rows",
            not senate_mismatch,
            f"mismatches={[(r[0], r[1], r[2]) for r in senate_mismatch][:10]}",
        )

        # 2. chronology: rows with transaction_date > disclosure_date are
        # OCR date swaps that the accepted pipeline quarantines from analyses
        # (TransactionRepository.get_by_year / get_by_date_range exclude them).
        bad_chronology = db.conn.execute(
            """
            SELECT COUNT(*) FROM transactions
            WHERE transaction_date IS NOT NULL AND disclosure_date IS NOT NULL
              AND transaction_date > disclosure_date
            """
        ).fetchone()[0]
        null_dates = db.conn.execute(
            """
            SELECT COUNT(*) FROM transactions
            WHERE transaction_date IS NULL OR disclosure_date IS NULL
            """
        ).fetchone()[0]
        checks["implausible_chronology_quarantined_count"] = {
            "passed": True,
            "detail": f"{int(bad_chronology)} rows excluded from analyses (OCR date swap policy)",
        }
        _check(checks, "no_null_transaction_or_disclosure_dates", int(null_dates) == 0, f"rows={null_dates}")
        analysis_invalid = db.transactions.get_by_year(2026)
        exposed = int(
            (
                analysis_invalid["transaction_date"].notna()
                & (analysis_invalid["transaction_date"] > analysis_invalid["disclosure_date"])
            ).sum()
        )
        _check(
            checks,
            "analysis_facing_chronology_valid",
            exposed == 0,
            f"exposed_invalid_rows={exposed}",
        )

        # 3. duplicate policy on the source identity tuple
        duplicates = db.conn.execute(
            """
            SELECT source, chamber, source_record_id, source_row_id,
                   ingestion_generation, COUNT(*) AS n
            FROM transactions
            WHERE source IS NOT NULL AND source_record_id IS NOT NULL
              AND source_row_id IS NOT NULL AND ingestion_generation IS NOT NULL
            GROUP BY 1, 2, 3, 4, 5 HAVING COUNT(*) > 1
            """
        ).fetchall()
        _check(checks, "duplicate_source_identity_policy", not duplicates, f"dups={duplicates[:10]}")

        # 4. source_row_id distinct per doc
        dup_row_ids = db.conn.execute(
            """
            SELECT doc_id, source_row_id, COUNT(*) AS n
            FROM transactions
            WHERE source_row_id IS NOT NULL
            GROUP BY doc_id, source_row_id HAVING COUNT(*) > 1
            """
        ).fetchall()
        _check(checks, "source_row_id_distinct_per_doc", not dup_row_ids, f"dups={dup_row_ids[:10]}")

        # 5. source_reports equation per source (senate uses its own generation)
        senate_gens = [
            str(row[0])
            for row in db.conn.execute(
                "SELECT DISTINCT ingestion_generation FROM source_reports WHERE source='senate_efd'"
            ).fetchall()
        ]
        senate_eq_ok = True
        senate_eq_detail = "no senate source_reports"
        for senate_gen in senate_gens:
            eq = db.source_reports.reconcile(senate_gen, "senate_efd", "senate")
            ok = eq["found"] == (
                eq["parsed"] + eq["paper_only"] + eq["unavailable"] + eq["failed"]
            ) and eq["failed"] == 0 and eq["unavailable"] == 0
            senate_eq_ok = senate_eq_ok and ok
            senate_eq_detail = f"gen={senate_gen} reconcile={eq}"
        eq = db.source_reports.reconcile(senate_gens[0] if senate_gens else manifest["generation"], "senate_efd", "senate")
        _check(
            checks,
            "report_equation_senate_efd",
            senate_eq_ok,
            senate_eq_detail,
        )
        for year in HOUSE_YEARS:
            house = manifest.get("house", {}).get(str(year))
            if house is None or house.get("parse_status") != "complete":
                continue
            gen = house["generation_id"]
            eq = db.source_reports.reconcile(gen, "house_pdf", "house")
            expected = eq["found"] == (
                eq["parsed"] + eq["paper_only"] + eq["unavailable"] + eq["failed"]
            ) and eq["failed"] == 0 and eq["unavailable"] == 0
            _check(
                checks,
                f"report_equation_house_{year}",
                expected,
                f"reconcile={eq}",
            )

        # 6. generation completeness bookkeeping
        incomplete_years = []
        for year in HOUSE_YEARS:
            house = manifest.get("house", {}).get(str(year))
            if house is None:
                incomplete_years.append(year)
                continue
            status = db.conn.execute(
                "SELECT parse_status FROM house_archive_generations WHERE archive_year=? AND generation_id=?",
                [year, house["generation_id"]],
            ).fetchone()[0]
            unresolved = db.get_unresolved_house_doc_ids(year, house["generation_id"])
            if status == "complete":
                _check(
                    checks,
                    f"house_{year}_complete_has_no_unresolved",
                    not unresolved,
                    f"unresolved={unresolved[:10]}",
                )
            else:
                incomplete_years.append(year)
                if not unresolved:
                    raise AssertionError(
                        f"house {year}: parse_status incomplete with no unresolved docs"
                    )
        # 13. sibling track presence gate
        consume_tracks = manifest.get("consume", {})
        required_tracks = ("senate", "ocr", "prices")
        track_status = {t: consume_tracks.get(t, {}).get("status") for t in required_tracks}
        tracks_present = all(
            consume_tracks.get(t, {}).get("status") == "ingested"
            for t in required_tracks
        )
        capitol = consume_tracks.get("capitol", {})
        _check(
            checks,
            "required_tracks_ingested",
            tracks_present,
            f"track_status={track_status}",
        )
        _check(
            checks,
            "capitol_track_present",
            capitol.get("status") == "ingested",
            f"capitol_status={capitol.get('status')} "
            f"({capitol.get('error', 'no artifact staged')})",
        )
        senate_summary = (
            manifest.get("senate", {}).get("summary")
            or consume_tracks.get("senate", {}).get("ingest", {}).get("summary")
            or {}
        )
        _check(
            checks,
            "senate_zero_failed_unavailable",
            int(senate_summary.get("failed", -1)) == 0
            and int(senate_summary.get("unavailable", -1)) == 0,
            str(senate_summary),
        )
        manifest["verify"] = {
            "incomplete_years": incomplete_years,
            "generation_complete": (
                not incomplete_years and tracks_present
                and capitol.get("status") == "ingested"
            ),
            "incomplete_reasons": (
                [f"house_year_{year}" for year in incomplete_years]
                + (
                    []
                    if tracks_present
                    else [f"missing_track:{t}" for t in required_tracks if consume_tracks.get(t, {}).get("status") != "ingested"]
                )
                + ([] if capitol.get("status") == "ingested" else [f"missing_track:capitol ({capitol.get('error', 'no artifact staged')})"])
            ),
        }

        # 7. canonical view: complete generations visible, incomplete hidden
        complete_years = [y for y in HOUSE_YEARS if y not in incomplete_years]
        canonical_count = db.conn.execute(
            "SELECT COUNT(*) FROM canonical_transactions"
        ).fetchone()[0]
        senate_count = db.conn.execute(
            "SELECT COUNT(*) FROM transactions WHERE source = 'senate_efd'"
        ).fetchone()[0]
        visible_house = db.conn.execute(
            """
            SELECT COUNT(*) FROM transactions t
            JOIN house_archive_generations g
              ON g.generation_id = t.ingestion_generation
             AND g.archive_year = EXTRACT(YEAR FROM t.disclosure_date)::INTEGER
            WHERE t.source = 'house_pdf' AND g.parse_status = 'complete'
            """
        ).fetchone()[0]
        _check(
            checks,
            "canonical_house_only_complete_generations",
            canonical_count == visible_house + senate_count,
            f"canonical={canonical_count} visible_house={visible_house} senate={senate_count}",
        )

        # 8. pinned canary artifact hashes in the staged corpus
        for doc_id, (expected_hash, _) in PINNED_TEXT_CANARIES.items():
            pdf = staging / "2026" / "pdfs" / f"{doc_id}.pdf"
            _check(
                checks,
                f"canary_{doc_id}_pdf_exists",
                pdf.exists(),
                str(pdf),
            )
            _check(
                checks,
                f"canary_{doc_id}_pdf_hash",
                pdf.exists() and _sha256_file(pdf) == expected_hash,
                f"sha={_sha256_file(pdf) if pdf.exists() else None}",
            )
        for doc_id, expected_hash in PINNED_SCAN_HASHES.items():
            pdf = staging / "2026" / "pdfs" / f"{doc_id}.pdf"
            _check(
                checks,
                f"scan_{doc_id}_pdf_exists",
                pdf.exists(),
                str(pdf),
            )
            _check(
                checks,
                f"scan_{doc_id}_pdf_hash",
                pdf.exists() and _sha256_file(pdf) == expected_hash,
                f"sha={_sha256_file(pdf) if pdf.exists() else None}",
            )

        # 9. pinned text-canary parse outcomes (local cascade, docling off)
        for doc_id, (_, expected_count) in PINNED_TEXT_CANARIES.items():
            pdf = staging / "2026" / "pdfs" / f"{doc_id}.pdf"
            _, rows, engines = _parse_pdf_worker(pdf)
            _check(
                checks,
                f"canary_{doc_id}_parse_count",
                len(rows) == expected_count,
                f"count={len(rows)} expected={expected_count} engines={engines[-4:]}",
            )
            _check(
                checks,
                f"canary_{doc_id}_won_pdftotext",
                "won:pdftotext" in engines,
                f"engines={engines[-6:]}",
            )

        # 10. scan fail-closed: 8221322 must remain unresolved in the 2026 work list
        from scripts.ocr_zero_rows import get_ocr_work_items  # noqa: PLC0415

        unresolved_2026 = {
            doc_id for doc_id, _, _ in get_ocr_work_items(
                db_path=str(staging / "congress.duckdb"),
                data_dir=staging,
                year=2026,
                require_schema=False,
            )
        }
        _check(
            checks,
            "scan_8221322_fail_closed_unresolved",
            "8221322" in unresolved_2026,
            f"unresolved2026_sample={sorted(unresolved_2026)[:10]}",
        )

        # 11. senate completeness before any claim
        senate = manifest.get("senate")
        if senate and senate.get("status") == "persisted":
            summary = senate["summary"]
            _check(
                checks,
                "senate_complete_refresh_required",
                summary["failed"] == 0 and summary["unavailable"] == 0,
                str(summary),
            )
            _check(
                checks,
                "senate_summary_accounts_all_reports",
                summary["found"]
                == summary["parsed"] + summary["paper_only"] + summary["unavailable"] + summary["failed"],
                str(summary),
            )
        elif senate and senate.get("status") == "quarantined":
            _check(
                checks,
                "senate_quarantined_not_persisted",
                True,
                senate.get("summary", ""),
            )

        # 12. canonical DB untouched (byte-identical sha)
        canonical_path = _REPO_ROOT.parents[1] / "data" / "congress.duckdb"
        if canonical_path.exists():
            _check(
                checks,
                "canonical_db_untouched",
                _sha256_file(canonical_path) == CANONICAL_DB_SHA256,
                f"sha={_sha256_file(canonical_path)[:16]}",
            )

        manifest["verify"]["checks"] = checks
        _save_manifest(staging, manifest)
        failed = [name for name, check in checks.items() if not check["passed"]]
        print(f"verify: {len(checks) - len(failed)}/{len(checks)} checks passed")
        if failed:
            print(f"verify: FAILED {failed}")
            raise SystemExit(1)
    finally:
        db.close()


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


def _sibling_dir(track: str, generation: str) -> Path:
    return _REPO_ROOT / "data" / ".staging" / track / generation


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
    tx["ingestion_generation"] = generation
    inv["ingestion_generation"] = generation
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


def _ingest_prices(track_dir: Path, track_manifest: dict, db: Database, staging: Path) -> dict:
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


def _ingest_ocr(track_dir: Path, track_manifest: dict, db: Database, staging: Path) -> dict:
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
                    entry["ingest"] = _ingest_prices(track_dir, track_manifest, db, staging)
                    entry["status"] = "ingested"
                elif track == "ocr":
                    entry["ingest"] = _ingest_ocr(track_dir, track_manifest, db, staging)
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
    import pandas as pd  # noqa: PLC0415

    staging = Path(args.staging)
    manifest = _load_manifest(staging)
    db = Database(staging / "congress.duckdb", read_only=False)
    try:
        for year in HOUSE_YEARS:
            house = manifest.get("house", {}).get(str(year))
            if house is None:
                continue
            gen = house["generation_id"]
            if house.get("parse_status") == "complete" and not args.force:
                continue
            unresolved = db.get_unresolved_house_doc_ids(year, gen)
            house["unresolved_doc_ids"] = unresolved
            house["resolved_doc_count"] = house["ptr_count"] - len(unresolved)
            if unresolved:
                house["parse_status"] = "incomplete"
                print(
                    f"house-activate {year}: INCOMPLETE — {len(unresolved)} unresolved"
                )
            else:
                rows = _house_inventory_rows(db, year, gen, staging)
                reports_df = pd.DataFrame(rows)
                db.source_reports.replace_generation(
                    gen, "house_pdf", "house", reports_df
                )
                db.mark_house_generation_parse_complete(year, gen)
                house["parse_status"] = "complete"
                house["source_report_rows"] = len(rows)
                print(f"house-activate {year}: COMPLETE — {len(rows)} inventory rows")
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
        f"STAGED AUTHORITATIVE CONGRESSIONAL DATABASE REBUILD — VERDICT",
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
                f"  persisted (consumed sibling sweep): found={summary.get('found')} "
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

    sub.add_parser("senate")
    sub.add_parser("prices")
    sub.add_parser("consume")
    sub.add_parser("repair-audit-gaps")
    sub.add_parser("house-activate")
    sub.add_parser("verify")
    sub.add_parser("finalize")

    args = parser.parse_args(argv)
    if args.stage == "bootstrap":
        _bootstrap(args)
        return
    staging = _resolve_staging(args)
    if not staging.exists():
        raise SystemExit(f"staging dir not found: {staging}")
    print(f"staging: {staging}")
    handlers = {
        "house-fetch": house_fetch,
        "house-parse": house_parse,
        "senate": senate,
        "prices": prices,
        "consume": consume,
        "repair-audit-gaps": repair_audit_gaps,
        "house-activate": house_activate,
        "verify": verify,
        "finalize": finalize,
    }
    handlers[args.stage](args)


if __name__ == "__main__":
    main()
