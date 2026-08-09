"""Parallel Gemini OCR with concurrent model calls and acknowledged DuckDB writes."""

import argparse
import json
import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import duckdb

from scripts.gemini_ocr_common import (
    GEMINI_PARSER_VERSION,
    call_gemini,
    parse_gemini_output,
    validate_transactions,
)
from scripts.ocr_zero_rows import (
    get_filing_date,
    get_metadata_member,
    get_ocr_work_items,
    insert_transactions,
    mark_progress,
    record_parse_run,
)

DB_PATH = "data/congress.duckdb"
PROGRESS_PATH = "data/ocr_progress.json"
MAX_WORKERS = 15


def get_zero_row_pdfs():
    return get_ocr_work_items(db_path=DB_PATH, data_dir=os.path.dirname(DB_PATH))


def load_progress():
    if os.path.exists(PROGRESS_PATH):
        with open(PROGRESS_PATH) as handle:
            return json.load(handle)
    return {"completed": [], "errors": [], "no_txs": []}


def save_progress(progress):
    temporary = PROGRESS_PATH + ".tmp"
    with open(temporary, "w") as handle:
        json.dump(progress, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, PROGRESS_PATH)


def parse_output(output):
    parsed = parse_gemini_output(output)
    return parsed.member, parsed.transactions


write_q: queue.Queue = queue.Queue()
SENTINEL = object()


def _write_item(item):
    if item["status"] in {"error", "rejected"}:
        connection = duckdb.connect(DB_PATH)
        try:
            record_parse_run(
                connection,
                item["doc_id"],
                item["year"],
                item["status"],
                item["raw_count"],
                0,
                item.get("error", ""),
                parser_version=GEMINI_PARSER_VERSION,
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
        parser_version=GEMINI_PARSER_VERSION,
        raw_count=item["raw_count"],
        artifact_sha256=item.get("artifact_sha256"),
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


def _record_failure(doc_id, year, status, raw_count, error):
    _acknowledged_write(
        {
            "doc_id": doc_id,
            "year": year,
            "status": status,
            "raw_count": raw_count,
            "error": str(error)[:1000],
        }
    )


def process_one(item, refresh=False):
    doc_id, year, pdf_path = item
    output, error, artifact_metadata = call_gemini(
        pdf_path,
        doc_id=doc_id,
        refresh=refresh,
        timeout=90,
        parser_version=GEMINI_PARSER_VERSION,
    )
    if output is None or error:
        _record_failure(doc_id, year, "error", 0, error)
        return doc_id, year, "error", 0, error

    parsed = parse_gemini_output(output)
    if parsed.no_transactions:
        _acknowledged_write(
            {
                "doc_id": doc_id,
                "year": year,
                "status": "no_txs",
                "member": parsed.member,
                "transactions": [],
                "raw_count": 0,
            }
        )
        return doc_id, year, "no_txs", 0, []

    connection = duckdb.connect(DB_PATH, read_only=True)
    try:
        filing_date = get_filing_date(connection, doc_id)
        expected_member = get_metadata_member(connection, doc_id)
    finally:
        connection.close()
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
        _record_failure(doc_id, year, status, parsed.raw_row_count, message)
        return doc_id, year, status, 0, fatal_rejections
    if not transactions:
        _record_failure(
            doc_id,
            year,
            "error",
            parsed.raw_row_count,
            "semantic_zero_after_raw_rows",
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
            "artifact_sha256": artifact_metadata.sha256,
        }
    )
    return doc_id, year, "success", inserted, transactions


def main():
    parser = argparse.ArgumentParser(
        description="Parallel Gemini OCR for unresolved PDFs"
    )
    parser.add_argument(
        "--refresh", action="store_true", help="Ignore cached responses"
    )
    args = parser.parse_args()

    progress = load_progress()
    pending = get_zero_row_pdfs()
    print(f"Current unresolved work: {len(pending)} (parallelism: {MAX_WORKERS})")
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
                pool.submit(process_one, item, args.refresh): item for item in pending
            }
            for future in as_completed(futures):
                doc_id, year, _ = futures[future]
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
                save_progress(progress)
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
