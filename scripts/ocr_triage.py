"""Classify unresolved House PDFs and prioritize the work that needs Gemini.

The database is opened read-only. Local retries use the existing 300-dpi OCR
sweep; ``--apply`` sends resolved rows through HouseTransactionSource's normal
parse-result persistence method. Gemini is never called here.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _entry in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

SCANNED_IMAGE = "scanned_image_needs_llm"
LOW_RESOLUTION = "bad_resolution_fixable_locally"
TEXT_MISSED = "parseable_missed_pattern"
CATEGORIES = (SCANNED_IMAGE, LOW_RESOLUTION, TEXT_MISSED)
LOCAL_RENDER_DPI = 300
DEFAULT_QUEUE_NAME = "gemini_ocr_priority_queue.json"


@dataclass(frozen=True)
class PdfProbe:
    page_count: int
    text_char_count: int
    dpi_estimate: float | None
    image_count: int
    errors: tuple[str, ...] = ()


def _run(command: list[str], timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(
        command, capture_output=True, text=True, timeout=timeout, check=False
    )


def _image_dpi(output: str) -> tuple[float | None, int]:
    """Estimate scan DPI from the largest embedded image on each page."""
    largest_by_page: dict[int, tuple[int, float]] = {}
    image_count = 0
    for line in output.splitlines():
        fields = line.split()
        if len(fields) < 14 or fields[2] != "image":
            continue
        try:
            page, width, height = int(fields[0]), int(fields[3]), int(fields[4])
            x_dpi, y_dpi = float(fields[12]), float(fields[13])
        except (ValueError, IndexError):
            continue
        image_count += 1
        area = width * height
        dpi = min(x_dpi, y_dpi)
        if page not in largest_by_page or area > largest_by_page[page][0]:
            largest_by_page[page] = (area, dpi)
    estimates = [value[1] for value in largest_by_page.values()]
    return (statistics.median(estimates) if estimates else None, image_count)


def probe_pdf(pdf_path: str | Path) -> PdfProbe:
    """Collect page count, text volume and embedded-image DPI with Poppler."""
    pdf_path = Path(pdf_path)
    errors: list[str] = []
    page_count = 0
    text_char_count = 0
    dpi_estimate = None
    image_count = 0

    for name, command, timeout in (
        ("pdfinfo", ["pdfinfo", str(pdf_path)], 30),
        ("pdftotext", ["pdftotext", str(pdf_path), "-"], 60),
        ("pdfimages", ["pdfimages", "-list", str(pdf_path)], 60),
    ):
        try:
            result = _run(command, timeout)
        except (OSError, subprocess.TimeoutExpired) as exc:
            errors.append(f"{name}: {exc}")
            continue
        if result.returncode:
            errors.append(
                f"{name}: {result.stderr.strip() or f'exit {result.returncode}'}"
            )
            continue
        if name == "pdfinfo":
            for line in result.stdout.splitlines():
                if line.startswith("Pages:"):
                    try:
                        page_count = int(line.split(":", 1)[1].strip())
                    except ValueError:
                        errors.append("pdfinfo: invalid page count")
                    break
        elif name == "pdftotext":
            text_char_count = len(result.stdout.strip())
        else:
            dpi_estimate, image_count = _image_dpi(result.stdout)

    return PdfProbe(
        page_count, text_char_count, dpi_estimate, image_count, tuple(errors)
    )


def classify_probe(probe: PdfProbe) -> str:
    """Classify from text-layer coverage first, then scan resolution."""
    if probe.page_count > 0 and probe.text_char_count >= max(
        120, 80 * probe.page_count
    ):
        return TEXT_MISSED
    if probe.dpi_estimate is not None and probe.dpi_estimate < LOCAL_RENDER_DPI:
        return LOW_RESOLUTION
    return SCANNED_IMAGE


def _as_date(value) -> date | None:
    if value is None:
        return None
    try:
        return (
            value.date()
            if hasattr(value, "date")
            else date.fromisoformat(str(value)[:10])
        )
    except (TypeError, ValueError):
        return None


def prioritize_queue(items: list[dict]) -> list[dict]:
    """Sort by historical House filing activity multiplied by recency rank."""
    dates = sorted(
        {
            _as_date(item.get("filing_date"))
            for item in items
            if _as_date(item.get("filing_date"))
        }
    )
    recency_rank = {
        filing_date: rank for rank, filing_date in enumerate(dates, start=1)
    }
    ranked = []
    for item in items:
        filing_date = _as_date(item.get("filing_date"))
        recency = recency_rank.get(filing_date, 0)
        prominence = max(1, int(item.get("member_prominence", 0)))
        ranked.append(
            {
                **item,
                "member_prominence": prominence,
                "recency_rank": recency,
                "priority_score": prominence * recency,
            }
        )
    return sorted(
        ranked,
        key=lambda item: (
            -item["priority_score"],
            -item["recency_rank"],
            -item["member_prominence"],
            str(item["doc_id"]),
        ),
    )


def load_unresolved_artifacts(db_path: str | Path, data_dir: str | Path) -> list[dict]:
    """Read unresolved artifacts from each year's latest House generation."""
    from analyzer.database import Database

    db_path = Path(db_path)
    if not db_path.is_file():
        return []
    db = Database(db_path, read_only=True)
    try:
        generations = db.conn.execute(
            """
            SELECT archive_year, generation_id
            FROM house_archive_generations
            QUALIFY row_number() OVER (
                PARTITION BY archive_year
                ORDER BY promoted_at DESC, generation_id DESC
            ) = 1
            ORDER BY archive_year
            """
        ).fetchall()
        prominence = {
            str(member).strip().casefold(): int(count)
            for member, count in db.conn.execute(
                """
                SELECT member, COUNT(DISTINCT doc_id)
                FROM transactions
                WHERE member IS NOT NULL
                  AND source IN ('house_pdf', 'gemini_ocr')
                  AND (chamber IS NULL OR lower(chamber) = 'house')
                GROUP BY member
                """
            ).fetchall()
        }
        artifacts = []
        for year, generation in generations:
            unresolved = set(db.get_unresolved_house_doc_ids(year, generation))
            if not unresolved:
                continue
            rows = db.conn.execute(
                """
                SELECT artifact.doc_id, artifact.artifact_sha256,
                       metadata.first_name, metadata.last_name,
                       metadata.filing_date
                FROM house_pdf_artifacts AS artifact
                JOIN house_generation_metadata AS metadata
                  ON metadata.archive_year = artifact.archive_year
                 AND metadata.generation_id = artifact.generation_id
                 AND metadata.doc_id = artifact.doc_id
                WHERE artifact.archive_year = ?
                  AND artifact.generation_id = ?
                  AND metadata.filing_type = 'P'
                """,
                [year, generation],
            ).fetchall()
            for doc_id, digest, first, last, filing_date in rows:
                if str(doc_id) not in unresolved:
                    continue
                member = " ".join(part for part in (first, last) if part).strip()
                artifacts.append(
                    {
                        "doc_id": str(doc_id),
                        "year": int(year),
                        "generation_id": str(generation),
                        "artifact_sha256": digest,
                        "member": member or None,
                        "first_name": first,
                        "last_name": last,
                        "filing_date": filing_date,
                        "member_prominence": prominence.get(member.casefold(), 0),
                        "pdf_path": str(
                            Path(data_dir) / str(year) / "pdfs" / f"{doc_id}.pdf"
                        ),
                    }
                )
        return artifacts
    finally:
        db.close()


def _local_reparse(item: dict) -> tuple[bool, dict, list[dict]]:
    """Run the existing Tesseract sweep at 300 dpi without staging or DB writes."""
    from analyzer.parsing.rows import parse_pdf_table
    from scripts import ocr_local_sweep

    prior_dpi, prior_docling = (
        ocr_local_sweep.RENDER_DPI,
        ocr_local_sweep.DOCLING_ENABLED,
    )
    ocr_local_sweep.RENDER_DPI = LOCAL_RENDER_DPI
    ocr_local_sweep.DOCLING_ENABLED = False
    try:
        result = ocr_local_sweep.process_document(
            item["doc_id"],
            item["year"],
            item["pdf_path"],
            {"member": item.get("member")},
        )
    finally:
        ocr_local_sweep.RENDER_DPI = prior_dpi
        ocr_local_sweep.DOCLING_ENABLED = prior_docling
    transactions = []
    header = [
        "Asset Name",
        "Transaction Type",
        "Transaction Date",
        "Notification Date",
        "Amount",
    ]
    for row in result.get("rows", []):
        table = [
            header,
            [
                row.get("asset_description"),
                row.get("transaction_type"),
                row.get("transaction_date_raw"),
                row.get("notification_date_raw"),
                None,
            ],
        ]
        parsed = parse_pdf_table(table)
        for transaction in parsed:
            transaction["source_row_id"] = row.get("source_row_id")
            transactions.append(transaction)
    return (
        result.get("status") == "resolved" and bool(transactions),
        result,
        transactions,
    )


def _persist_through_pipeline(
    results: list[tuple[dict, list[dict]]], db_path, data_dir
) -> tuple[int, list[str]]:
    """Persist locally recovered rows using the normal House save path."""
    from analyzer.database import Database
    from analyzer.download import HouseTransactionSource
    from analyzer.settings import Settings

    db = Database(db_path)
    settings = Settings()
    settings.data.data_dir = str(data_dir)
    source = HouseTransactionSource(settings, db=db)
    successes = 0
    errors = []
    try:
        for item, transactions in results:
            try:
                metadata = {
                    item["doc_id"]: {
                        "First": item.get("first_name") or "",
                        "Last": item.get("last_name") or "",
                        "FilingDate": item.get("filing_date"),
                    }
                }
                source._save_parse_results(
                    item["year"],
                    [(Path(item["pdf_path"]), transactions, ["ocr_triage_300dpi"])],
                    metadata,
                    item["generation_id"],
                )
                run = db.conn.execute(
                    """
                    SELECT status, transaction_count
                    FROM pdf_parse_runs
                    WHERE doc_id = ? AND ingestion_generation = ?
                      AND artifact_sha256 = ?
                    ORDER BY parsed_at DESC LIMIT 1
                    """,
                    [item["doc_id"], item["generation_id"], item["artifact_sha256"]],
                ).fetchone()
                if run is None or run[0] != "success" or int(run[1] or 0) == 0:
                    raise RuntimeError(
                        "normal pipeline did not persist a successful parse"
                    )
                successes += 1
            except Exception as exc:  # noqa: BLE001 -- fail one PDF, keep the batch moving
                errors.append(f"{item['doc_id']}: {type(exc).__name__}: {exc}")
    finally:
        source.close()
        db.close()
    return successes, errors


def _queue_work_items(items: list[dict], queue_path: str | Path) -> None:
    """Write cron-compatible work tuples in priority order as a JSON array."""
    queue_path = Path(queue_path)
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    work_items = [[item["doc_id"], item["year"], item["pdf_path"]] for item in items]
    queue_path.write_text(json.dumps(work_items, indent=2) + "\n", encoding="utf-8")


def run_triage(
    data_dir: str | Path = "data",
    *,
    db_path: str | Path | None = None,
    queue_path: str | Path | None = None,
    apply: bool = False,
) -> dict:
    data_dir = Path(data_dir)
    db_path = Path(db_path) if db_path else data_dir / "congress.duckdb"
    queue_path = Path(queue_path) if queue_path else data_dir / DEFAULT_QUEUE_NAME
    if not db_path.is_file():
        return {"database_available": False, "classifications": {}, "attempted": 0}

    artifacts = load_unresolved_artifacts(db_path, data_dir)
    classified = []
    for item in artifacts:
        probe = probe_pdf(item["pdf_path"])
        category = classify_probe(probe)
        classified.append({**item, "probe": probe, "category": category})

    counts = Counter(item["category"] for item in classified)
    queue = prioritize_queue(
        [item for item in classified if item["category"] == SCANNED_IMAGE]
    )
    _queue_work_items(queue, queue_path)

    local_items = [item for item in classified if item["category"] == LOW_RESOLUTION]
    local_successes = 0
    local_failures = 0
    persistable = []
    for item in local_items:
        try:
            succeeded, result, transactions = _local_reparse(item)
            if succeeded and result.get("artifact_sha256") != item.get(
                "artifact_sha256"
            ):
                raise RuntimeError("PDF bytes no longer match the acquired artifact")
        except Exception as exc:  # noqa: BLE001 -- a failed PDF does not stop the batch
            item["local_error"] = f"{type(exc).__name__}: {exc}"
            succeeded, transactions = False, []
        if succeeded:
            if apply:
                persistable.append((item, transactions))
            else:
                local_successes += 1
        else:
            local_failures += 1

    persistence_errors = []
    if apply and persistable:
        try:
            local_successes, persistence_errors = _persist_through_pipeline(
                persistable, db_path, data_dir
            )
            local_failures += len(persistable) - local_successes
        except Exception as exc:  # noqa: BLE001 -- pipeline setup failure affects this batch only
            local_failures += len(persistable)
            persistence_errors.append(f"{type(exc).__name__}: {exc}")

    return {
        "database_available": True,
        "unresolved": len(artifacts),
        "classifications": {
            category: counts.get(category, 0) for category in CATEGORIES
        },
        "attempted": len(local_items),
        "local_successes": local_successes,
        "local_failures": local_failures,
        "persistence_errors": persistence_errors,
        "queue_count": len(queue),
        "queue_path": str(queue_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--queue-file", default=None)
    parser.add_argument(
        "--apply", action="store_true", help="persist successful local retries"
    )
    args = parser.parse_args(argv)
    report = run_triage(
        args.data_dir,
        queue_path=args.queue_file,
        apply=args.apply,
    )
    if not report["database_available"]:
        print(
            f"triage unavailable: no database at {Path(args.data_dir) / 'congress.duckdb'}"
        )
        return 2
    print(
        "unresolved={unresolved} scanned_image_needs_llm={scanned} "
        "bad_resolution_fixable_locally={low} parseable_missed_pattern={text} "
        "local_attempts={attempted} local_successes={local_successes} "
        "local_failures={local_failures} queued={queue_count} queue={queue_path}".format(
            unresolved=report["unresolved"],
            scanned=report["classifications"][SCANNED_IMAGE],
            low=report["classifications"][LOW_RESOLUTION],
            text=report["classifications"][TEXT_MISSED],
            **{
                key: report[key]
                for key in (
                    "attempted",
                    "local_successes",
                    "local_failures",
                    "queue_count",
                    "queue_path",
                )
            },
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
