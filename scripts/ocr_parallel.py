"""Parallel Gemini OCR with concurrent model calls and acknowledged DuckDB writes."""

import argparse
import json
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import duckdb

from scripts.gemini_ocr_common import (
    GEMINI_35_MODEL,
    GEMINI_35_PARSER_VERSION,
    GEMINI_PARSER_VERSION,
    MODEL,
    call_gemini,
    parse_gemini_output,
    pdf_page_count,
    validate_transactions,
)
from scripts.ocr_zero_rows import (
    _record_failed_ocr_attempt,
    get_ocr_work_items,
    insert_transactions,
    load_progress,
    mark_progress,
    save_progress,
)

DB_PATH = "data/congress.duckdb"
PROGRESS_PATH = "data/ocr_progress.json"
MAX_WORKERS = 15


write_q: queue.Queue = queue.Queue()
SENTINEL = object()


def _write_item(item):
    parser_version = item.get("parser_version", GEMINI_PARSER_VERSION)
    model = item.get("model", MODEL)
    if item["status"] in {"error", "rejected"}:
        connection = duckdb.connect(DB_PATH)
        try:
            _record_failed_ocr_attempt(
                connection,
                item["doc_id"],
                item["year"],
                item["raw_count"],
                item.get("error", ""),
                parser_version=parser_version,
                artifact_sha256=item.get("artifact_sha256"),
                ingestion_generation=item.get("ingestion_generation"),
                engine_model=model,
            )
        finally:
            connection.close()
        return 0

    inserted = insert_transactions(
        item["doc_id"],
        item["year"],
        item["member"],
        item["transactions"],
        db_path=DB_PATH,
        parser_version=parser_version,
        raw_count=item["raw_count"],
        artifact_sha256=item.get("artifact_sha256"),
        ingestion_generation=item.get("ingestion_generation"),
        engine_model=model,
        checkpoint=False,
    )
    if item["status"] == "success" and inserted <= 0:
        raise RuntimeError("validated OCR rows were not inserted")
    if item["status"] == "no_txs" and inserted != 0:
        raise RuntimeError("no_txs write unexpectedly inserted rows")
    return inserted


def _flush(batch):
    """Write each document and acknowledge its durable outcome to its worker."""
    for item in batch:
        try:
            inserted = _write_item(item)
        except Exception as exc:
            item["ack"].put((None, exc))
        else:
            item["ack"].put((inserted, None))


def db_writer():
    while True:
        item = write_q.get()
        if item is SENTINEL:
            return
        _flush([item])


def _acknowledged_write(item):
    acknowledgement: queue.Queue = queue.Queue(maxsize=1)
    item["ack"] = acknowledgement
    write_q.put(item)
    inserted, error = acknowledgement.get()
    if error is not None:
        raise error
    return inserted


def _record_failure(
    doc_id,
    year,
    status,
    raw_count,
    error,
    *,
    artifact_sha256=None,
    ingestion_generation=None,
    parser_version=GEMINI_PARSER_VERSION,
    model=MODEL,
):
    _acknowledged_write(
        {
            "doc_id": doc_id,
            "year": year,
            "status": status,
            "raw_count": raw_count,
            "error": str(error)[:1000],
            "artifact_sha256": artifact_sha256,
            "ingestion_generation": ingestion_generation,
            "parser_version": parser_version,
            "model": model,
        }
    )


def process_one(
    item,
    refresh=False,
    *,
    cache_dir=None,
    parser_version=GEMINI_PARSER_VERSION,
    model=MODEL,
):
    (
        doc_id,
        year,
        pdf_path,
        ingestion_generation,
        expected_artifact_sha256,
        filing_date,
        expected_member,
    ) = item
    output, error, artifact_metadata = call_gemini(
        pdf_path,
        doc_id=doc_id,
        refresh=refresh,
        cache_dir=cache_dir or str(Path(DB_PATH).parent / "gemini_cache"),
        timeout=120,
        parser_version=parser_version,
        model=model,
    )
    if (
        artifact_metadata is not None
        and artifact_metadata.sha256 != expected_artifact_sha256
    ):
        output = None
        error = (
            "OCR artifact hash changed after work selection: "
            f"expected={expected_artifact_sha256} actual={artifact_metadata.sha256}"
        )
    if output is None or error:
        _record_failure(
            doc_id,
            year,
            "error",
            0,
            error,
            artifact_sha256=expected_artifact_sha256,
            ingestion_generation=ingestion_generation,
            parser_version=parser_version,
            model=model,
        )
        return doc_id, year, "error", 0, error

    parsed = parse_gemini_output(
        output,
        expected_page_count=(artifact_metadata.page_count if artifact_metadata else None),
    )
    if parsed.no_transactions:
        _acknowledged_write(
            {
                "doc_id": doc_id,
                "year": year,
                "status": "no_txs",
                "member": parsed.member,
                "transactions": [],
                "raw_count": 0,
                "artifact_sha256": expected_artifact_sha256,
                "ingestion_generation": ingestion_generation,
                "parser_version": parser_version,
                "model": model,
            }
        )
        return doc_id, year, "no_txs", 0, []

    transactions, rejections = validate_transactions(
        doc_id,
        parsed.member,
        parsed.transactions,
        filing_date,
        expected_member,
    )
    fatal_rejections = {
        key: value
        for key, value in rejections.items()
        if key not in {"duplicate_collapsed", "member_mismatch"}
    }
    if fatal_rejections:
        status = "rejected" if "row_count_exceeds_cap" in fatal_rejections else "error"
        message = json.dumps(fatal_rejections, sort_keys=True)
        _record_failure(
            doc_id,
            year,
            status,
            parsed.raw_row_count,
            message,
            artifact_sha256=expected_artifact_sha256,
            ingestion_generation=ingestion_generation,
            parser_version=parser_version,
            model=model,
        )
        return doc_id, year, status, 0, fatal_rejections
    if not transactions:
        _record_failure(
            doc_id,
            year,
            "error",
            parsed.raw_row_count,
            "semantic_zero_after_raw_rows",
            artifact_sha256=expected_artifact_sha256,
            ingestion_generation=ingestion_generation,
            parser_version=parser_version,
            model=model,
        )
        return doc_id, year, "error", 0, {"semantic_zero_after_raw_rows": 1}

    member = transactions[0]["member"]
    inserted = _acknowledged_write(
        {
            "doc_id": doc_id,
            "year": year,
            "status": "success",
            "member": member,
            "transactions": transactions,
            "raw_count": parsed.raw_row_count,
            "artifact_sha256": expected_artifact_sha256,
            "ingestion_generation": ingestion_generation,
            "parser_version": parser_version,
            "model": model,
        }
    )
    return doc_id, year, "success", inserted, transactions


def _page_count_hint(item) -> tuple[int, int, str]:
    doc_id, year, pdf_path = item[:3]
    try:
        pages = pdf_page_count(pdf_path)
    except Exception:
        pages = 10**9
    return int(year), pages, str(doc_id)


def _bind_work_items(pending):
    """Bind unresolved PDFs to one immutable staged generation before threading."""
    connection = duckdb.connect(DB_PATH)
    try:
        bound = []
        for doc_id, year, pdf_path in pending:
            row = connection.execute(
                """
                SELECT a.generation_id, a.artifact_sha256,
                       m.filing_date, m.first_name, m.last_name
                FROM house_pdf_artifacts a
                JOIN house_generation_metadata m
                  ON m.archive_year = a.archive_year
                 AND m.generation_id = a.generation_id
                 AND m.doc_id = a.doc_id
                JOIN house_archive_generations g
                  ON g.archive_year = a.archive_year
                 AND g.generation_id = a.generation_id
                WHERE a.archive_year = ? AND a.doc_id = ?
                  AND m.filing_type = 'P'
                ORDER BY g.promoted_at DESC, a.generation_id DESC
                LIMIT 1
                """,
                [int(year), str(doc_id)],
            ).fetchone()
            if row is None or not row[0] or not row[1]:
                raise RuntimeError(
                    f"House OCR work item lacks generation/artifact binding: {year}/{doc_id}"
                )
            generation, artifact_sha256, filing_date, first_name, last_name = row
            member = " ".join(
                str(part).strip() for part in (first_name, last_name) if part
            ).strip()
            if not member:
                raise RuntimeError(
                    f"House OCR work item lacks member metadata: {year}/{doc_id}"
                )
            bound.append(
                (
                    str(doc_id),
                    int(year),
                    str(pdf_path),
                    str(generation),
                    str(artifact_sha256),
                    filing_date,
                    member,
                )
            )
        return bound
    finally:
        connection.close()


def main():
    global DB_PATH, PROGRESS_PATH, MAX_WORKERS

    parser = argparse.ArgumentParser(
        description="Parallel Gemini OCR for unresolved PDFs"
    )
    parser.add_argument(
        "--refresh", action="store_true", help="Ignore cached responses"
    )
    parser.add_argument("--db", default=DB_PATH, help="target staged DuckDB")
    parser.add_argument(
        "--data-dir", default="data", help="directory containing <year>/pdfs"
    )
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--progress", default=None)
    parser.add_argument("--years", nargs="+", type=int, default=None)
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--model", default=GEMINI_35_MODEL)
    parser.add_argument("--parser-version", default=GEMINI_35_PARSER_VERSION)
    args = parser.parse_args()

    DB_PATH = str(args.db)
    MAX_WORKERS = max(1, int(args.workers))
    data_dir = Path(args.data_dir)
    cache_dir = Path(args.cache_dir) if args.cache_dir else data_dir / "gemini_cache"
    PROGRESS_PATH = str(
        Path(args.progress) if args.progress else data_dir / "ocr_progress_parallel.json"
    )

    progress = load_progress(PROGRESS_PATH)
    if args.years:
        pending = []
        for year in args.years:
            pending.extend(
                get_ocr_work_items(
                    db_path=DB_PATH,
                    data_dir=data_dir,
                    year=int(year),
                    parser_version=args.parser_version,
                )
            )
    else:
        pending = get_ocr_work_items(
            db_path=DB_PATH,
            data_dir=data_dir,
            parser_version=args.parser_version,
        )
    pending = sorted(dict.fromkeys(pending), key=_page_count_hint)
    if args.max_docs is not None:
        pending = pending[: max(0, int(args.max_docs))]
    pending = _bind_work_items(pending)
    print(
        f"Current unresolved work: {len(pending)} (parallelism: {MAX_WORKERS}, "
        f"model={args.model}, parser={args.parser_version})"
    )
    if not pending:
        return

    writer_thread = threading.Thread(target=db_writer)
    writer_thread.start()
    started = time.time()
    completed = 0
    total_inserted = 0
    write_failures = []
    try:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {
                pool.submit(
                    process_one,
                    item,
                    args.refresh,
                    cache_dir=str(cache_dir),
                    parser_version=args.parser_version,
                    model=args.model,
                ): item
                for item in pending
            }
            for future in as_completed(futures):
                doc_id, year, *_ = futures[future]
                try:
                    result_doc, _, status, inserted, _ = future.result()
                except Exception as exc:
                    write_failures.append((doc_id, str(exc)))
                    print(f"  {doc_id} unconfirmed failure: {exc}")
                    continue
                completed += 1
                total_inserted += inserted
                progress_status = (
                    "success"
                    if status == "success"
                    else "no_txs"
                    if status == "no_txs"
                    else "errors"
                )
                mark_progress(progress, result_doc, progress_status)
                save_progress(progress, PROGRESS_PATH)
                elapsed = time.time() - started
                print(
                    f"  [{completed}/{len(pending)}] {result_doc} ({year}) "
                    f"{status} +{inserted} | total {total_inserted} | {elapsed:.0f}s"
                )
    finally:
        write_q.put(SENTINEL)
        writer_thread.join()

    if write_failures:
        sample = ", ".join(f"{doc}: {error}" for doc, error in write_failures[:10])
        raise RuntimeError(f"{len(write_failures)} unconfirmed OCR writes: {sample}")
    print(f"Done: {completed} PDFs, {total_inserted} confirmed rows")


if __name__ == "__main__":
    main()
