"""Live Senate eFD full-member sweep producing staged artifacts only.

Drives the accepted ``SenateEFDSource`` code (fetch/normalize/classify) for
every PTR filing listed by efdsearch in the sweep window and stages the
results under ``.staging/senate/<generation>/`` without touching any database:

  * ``report_inventory.jsonl`` — one record per listed filing with the accepted
    outcome enum (parsed/paper_only/unavailable/failed), artifact SHA-256s and
    raw/accepted/rejected row counts.
  * ``transactions.jsonl`` — the accepted normalized transaction rows for every
    parsed filing (same columns as the production persistence schema).
  * ``papers/<source_record_id>.pdf`` — the paper-only artifact bytes, byte
    verified against the accepted paper_artifact_sha256.
  * ``manifest.json`` — generation id, window, outcome counts, per-artifact
    SHA-256 + byte size, canary results, and quarantine record when the sweep
    is incomplete.

Fail-closed rules (mirroring the staged rebuild driver):
  * A sweep with any unavailable/failed filing is quarantined: nothing is
    claimed complete, the exact missing counts and per-report outcome rows are
    staged, and the exit code is 2.  No report is ever fabricated.
  * Katie Britt (1 transaction) and Rick Scott (12 transactions) live canaries
    must classify ``parsed`` with the expected accepted row counts.

Exit codes: 0 = complete, 2 = quarantined (artifacts staged, sweep incomplete),
1 = hard failure (e.g. efdsearch blocked before an inventory existed).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

# Add src to path so the analyzer package imports from this checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from analyzer.member_names import canonical_member_key  # noqa: E402
from analyzer.models import ReportOutcome  # noqa: E402
from analyzer.senate_efd import (  # noqa: E402
    SenateEFDError,
    SenateEFDSource,
    SenateReportFetchResult,
)

GENERATION = "gen-live-20260809"
_PAPER_URL_RE = re.compile(r"/search/view/paper(?:-filing)?/|\.pdf", re.I)
_CANARY_EXPECTED = {
    "KATIE BRITT": {"accepted_row_count": 1},
    "RICK SCOTT": {"accepted_row_count": 12},
}
# Frozen-window contract: both members must have at least one parsed report
# in the window; the exact 1/12 counts belong to the 2026 live canary filings.
_CANARY_EXPECTED_ANY = {
    "KATIE BRITT": None,
    "RICK SCOTT": None,
}


class SenateSweepError(Exception):
    """Hard failure before a complete inventory could be staged."""


class _NullDB:
    """Database stub: the sweep stages artifacts and never persists."""

    def close(self) -> None:  # pragma: no cover - trivial stub
        return None


class _StagingSenateSource(SenateEFDSource):
    """Accepted SenateEFDSource that also retains per-report fetch results.

    Only the private hooks are shadowed to capture what the accepted fetch
    loop already produces; every classification, normalization and hash comes
    from the unmodified accepted implementation.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.staged_results: dict[str, SenateReportFetchResult] = {}
        self.paper_bytes: dict[str, bytes] = {}

    def _request_with_retry(self, method: str, url: str, **kwargs: Any):
        response = super()._request_with_retry(method, url, **kwargs)
        if method == "GET" and _PAPER_URL_RE.search(str(url)):
            self.paper_bytes[str(response.url)] = response.content
        return response

    def _fetch_report_transactions(self, report_path: str) -> SenateReportFetchResult:
        result = super()._fetch_report_transactions(report_path)
        self.staged_results[report_path] = result
        return result


def _json_ready(value: Any) -> Any:
    """Convert pandas/numpy scalars to JSON-native values."""
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return value.isoformat()
    if isinstance(value, float):
        import math

        if math.isnan(value):
            return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):
            return str(value)
    return value


def inventory_records(inventory: list[dict]) -> list[dict]:
    """JSON-ready report inventory records (timestamps as ISO strings)."""
    records = []
    for row in inventory:
        record = dict(row)
        record["official_filing_date"] = _json_ready(row.get("official_filing_date"))
        records.append(record)
    return records


def write_inventory_jsonl(inventory: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in inventory_records(inventory):
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def write_transactions_jsonl(transactions: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in transactions.to_dict("records"):
            handle.write(
                json.dumps(
                    {key: _json_ready(value) for key, value in record.items()},
                    sort_keys=True,
                )
                + "\n"
            )


def rebuild_normalized_transactions(
    source: SenateEFDSource, inventory: list[dict]
) -> pd.DataFrame:
    """Rebuild accepted normalized rows from the retained per-report results.

    Used only when the sweep is quarantined and the accepted loop raised
    before returning its DataFrame.  Row construction mirrors the accepted
    ``fetch_all_trades`` loop exactly; normalization is the accepted
    ``_normalize``.
    """
    report_rows: list[dict] = []
    for row in inventory:
        if row["outcome"] != ReportOutcome.PARSED.value:
            continue
        result = source.staged_results.get(row["report_path"])
        if result is None:
            raise SenateSweepError(
                f"missing staged parse result for parsed report {row['report_path']}"
            )
        doc_id = source._path_to_doc_id(row["report_path"])
        source_record_id = source._path_to_source_record_id(row["report_path"])
        filed_date = row.get("official_filing_date")
        for transaction in result.transactions:
            report_rows.append(
                {
                    "doc_id": doc_id,
                    "source_record_id": source_record_id,
                    "source_report_path": row["report_path"],
                    "senator": row["member"],
                    "filed_date": filed_date,
                    "official_filing_date": filed_date,
                    "available_date": filed_date,
                    "amends_source_record_id": None,
                    "artifact_sha256": result.transaction_artifact_sha256,
                    "ingestion_generation": source.ingestion_generation,
                    **transaction,
                }
            )
    return source._normalize(report_rows)


def stage_paper_artifacts(
    source: SenateEFDSource,
    inventory: list[dict],
    papers_dir: Path,
) -> list[dict]:
    """Stage paper-only PDF bytes and byte-verify them against accepted SHA."""
    staged = []
    papers_dir.mkdir(parents=True, exist_ok=True)
    for row in inventory:
        if row["outcome"] != ReportOutcome.PAPER_ONLY.value:
            continue
        url = row.get("paper_artifact_url")
        expected_sha = row.get("paper_artifact_sha256")
        if not url or not expected_sha:
            raise SenateSweepError(
                f"paper_only report {row['report_path']} lacks paper artifact metadata"
            )
        content = source.paper_bytes.get(str(url))
        if content is None:
            response = source._request_with_retry("GET", str(url))
            content = response.content
        actual_sha = hashlib.sha256(content).hexdigest()
        if actual_sha != expected_sha:
            raise SenateSweepError(
                f"paper artifact SHA mismatch for {row['report_path']}: "
                f"staged={actual_sha[:16]} expected={expected_sha[:16]}"
            )
        target = papers_dir / f"{row['source_record_id']}.pdf"
        target.write_bytes(content)
        staged.append(
            {
                "source_record_id": row["source_record_id"],
                "file": target.relative_to(target.parents[2]).as_posix(),
                "sha256": actual_sha,
                "bytes": len(content),
            }
        )
    return staged


def verify_canaries(inventory: list[dict], expected: dict | None = None) -> dict:
    """Verify Katie Britt and Rick Scott canaries classify parsed.

    ``expected`` maps canonical member key to either an ``accepted_row_count``
    (at least one parsed report must match exactly) or ``None`` (at least one
    parsed report must exist; the exact 1/12 counts belong to the 2026 live
    canary filings, absent from the frozen 2021-2025 window).  Defaults to the
    live-window expectations.
    """
    if expected is None:
        expected = _CANARY_EXPECTED
    by_member: dict[str, list[dict]] = {}
    for row in inventory:
        key = canonical_member_key(row.get("member") or "")
        by_member.setdefault(key, []).append(row)

    results: dict[str, dict] = {}
    failures: list[str] = []
    for key, spec in sorted(expected.items()):
        expected_count = None if spec is None else spec["accepted_row_count"]
        rows = by_member.get(key, [])
        parsed_rows = [r for r in rows if r["outcome"] == ReportOutcome.PARSED.value]
        if expected_count is None:
            matching = parsed_rows
        else:
            matching = [
                r for r in parsed_rows if int(r["accepted_row_count"]) == expected_count
            ]
        result = {
            "member_key": key,
            "reports_found": len(rows),
            "parsed_reports": len(parsed_rows),
            "accepted_row_counts": sorted(
                int(r["accepted_row_count"]) for r in parsed_rows
            ),
            "canary_matches": [
                {
                    "source_record_id": r["source_record_id"],
                    "official_filing_date": _json_ready(r.get("official_filing_date")),
                    "accepted_row_count": int(r["accepted_row_count"]),
                }
                for r in matching
            ],
            "outcomes": sorted({r["outcome"] for r in rows}),
        }
        if not rows:
            failures.append(f"{key}: no report listed in sweep window")
        elif not matching:
            if expected_count is None:
                failures.append(f"{key}: no parsed report in sweep window")
            else:
                failures.append(
                    f"{key}: no parsed report with accepted_row_count "
                    f"{expected_count}; counts={result['accepted_row_counts']}"
                )
        results[key] = result
    return {"results": results, "failures": failures, "passed": not failures}


def sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest(), path.stat().st_size


def write_manifest(
    generation_dir: Path,
    *,
    generation: str,
    start_date: date,
    end_date: date,
    inventory: list[dict],
    summary: dict,
    transactions_file: Path | None,
    papers: list[dict],
    canaries: dict,
    quarantine: dict | None,
) -> Path:
    def artifact(path: Path) -> dict:
        sha, size = sha256_file(path)
        return {
            "file": path.relative_to(generation_dir).as_posix(),
            "sha256": sha,
            "bytes": size,
        }

    inventory_file = generation_dir / "report_inventory.jsonl"
    artifacts = {
        "report_inventory.jsonl": artifact(inventory_file),
    }
    if transactions_file is not None:
        artifacts["transactions.jsonl"] = artifact(transactions_file)
    for paper in papers:
        artifacts[paper["file"]] = {
            "file": paper["file"],
            "sha256": paper["sha256"],
            "bytes": paper["bytes"],
        }

    manifest = {
        "generation": generation,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": "senate_efd",
        "window": {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
        },
        "outcome_counts": {
            outcome.value: sum(
                1 for row in inventory if row["outcome"] == outcome.value
            )
            for outcome in ReportOutcome
        },
        "reports_found": len(inventory),
        "summary": summary,
        "artifacts": artifacts,
        "canaries": canaries,
        "status": "complete" if quarantine is None else "quarantined",
        "quarantine": quarantine,
    }
    manifest_path = generation_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest_path


def run_sweep(
    generation: str,
    staging_root: Path,
    start_date: date,
    end_date: date,
    canary_expect: str = "exact",
) -> int:
    generation_dir = staging_root / generation
    papers_dir = generation_dir / "papers"
    generation_dir.mkdir(parents=True, exist_ok=True)

    source = _StagingSenateSource(
        data_dir=str(generation_dir / "cache"),
        read_only=False,
        db=_NullDB(),
        ingestion_generation=generation,
    )
    summary: dict | None = None
    inventory: list[dict] = []
    transactions = pd.DataFrame()
    quarantine: dict | None = None
    try:
        try:
            transactions = source.fetch_all_trades(start_date, end_date)
        except SenateEFDError as exc:
            summary_obj = source.last_refresh_summary
            inventory = list(source.report_inventory)
            summary = (
                None
                if summary_obj is None
                else {
                    "found": summary_obj.found,
                    "parsed": summary_obj.parsed,
                    "paper_only": summary_obj.paper_only,
                    "unavailable": summary_obj.unavailable,
                    "failed": summary_obj.failed,
                }
            )
            quarantine = {
                "error": str(exc),
                "missing": {
                    "unavailable": summary_obj.unavailable if summary_obj else None,
                    "failed": summary_obj.failed if summary_obj else None,
                },
                "reports_found": len(inventory),
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
        else:
            summary_obj = source.last_refresh_summary
            inventory = list(source.report_inventory)
            summary = (
                None
                if summary_obj is None
                else {
                    "found": summary_obj.found,
                    "parsed": summary_obj.parsed,
                    "paper_only": summary_obj.paper_only,
                    "unavailable": summary_obj.unavailable,
                    "failed": summary_obj.failed,
                }
            )
    finally:
        source.close()

    if not inventory:
        raise SenateSweepError(
            f"no Senate report inventory produced for {start_date}..{end_date}"
        )

    # Stage the report inventory always.
    write_inventory_jsonl(inventory, generation_dir / "report_inventory.jsonl")

    # Stage normalized transactions (accepted df on success; rebuilt otherwise).
    transactions_file = generation_dir / "transactions.jsonl"
    if len(transactions):
        write_transactions_jsonl(transactions, transactions_file)
    else:
        rebuilt = rebuild_normalized_transactions(source, inventory)
        write_transactions_jsonl(rebuilt, transactions_file)

    # Stage paper-only PDFs, byte-verified against the accepted SHA.
    papers = stage_paper_artifacts(source, inventory, papers_dir)

    if canary_expect == "any":
        canary_expected = _CANARY_EXPECTED_ANY
    else:
        canary_expected = _CANARY_EXPECTED
    canaries = verify_canaries(inventory, expected=canary_expected)
    manifest_path = write_manifest(
        generation_dir,
        generation=generation,
        start_date=start_date,
        end_date=end_date,
        inventory=inventory,
        summary=summary,
        transactions_file=transactions_file,
        papers=papers,
        canaries=canaries,
        quarantine=quarantine,
    )

    counts = {} if summary is None else summary
    print(
        f"senate sweep {generation}: found={counts.get('found')} "
        f"parsed={counts.get('parsed')} paper_only={counts.get('paper_only')} "
        f"unavailable={counts.get('unavailable')} failed={counts.get('failed')}"
    )
    print(f"staged artifacts in {generation_dir}")
    print(f"manifest: {manifest_path}")
    for key, result in canaries["results"].items():
        print(f"canary {key}: {result}")

    if quarantine is not None:
        (generation_dir / "quarantine.json").write_text(
            json.dumps(quarantine, indent=2, sort_keys=True) + "\n"
        )
        print("sweep QUARANTINED — see quarantine.json for exact missing counts")
        return 2
    if canary_expect == "none":
        print("sweep COMPLETE — canary results recorded (no gate)")
        return 0
    if not canaries["passed"]:
        print("sweep COMPLETE but canary verification FAILED:")
        for failure in canaries["failures"]:
            print(f"  {failure}")
        return 2
    print("sweep COMPLETE — all canaries passed")
    return 0


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def merge_window(window_dir: Path) -> int:
    """Consolidate ``chunk-*`` subdirectory sweeps into one window staging.

    The frozen 2021-2025 window is fetched per calendar-year chunk; this
    merges chunk inventories, normalized transactions and paper artifacts into
    the window directory with a combined manifest, per-chunk references, and
    window-level canary verification (Britt/Scott at least one parsed report).
    Any quarantined chunk keeps the window quarantined with exact missing
    counts.
    """
    if not window_dir.is_dir():
        raise SenateSweepError(f"merge window directory not found: {window_dir}")
    chunks = sorted(window_dir.glob("chunk-*"))
    if not chunks:
        raise SenateSweepError(f"no chunk-* subdirectories under {window_dir}")

    inventory: list[dict] = []
    transactions: list[dict] = []
    papers: list[dict] = []
    chunk_manifests: list[dict] = []
    quarantines: list[dict] = []
    seen_paths: set[str] = set()
    seen_rows: set[tuple] = set()
    window_start: date | None = None
    window_end: date | None = None

    for chunk_dir in chunks:
        manifest_path = chunk_dir / "manifest.json"
        if not manifest_path.is_file():
            raise SenateSweepError(f"chunk {chunk_dir.name} missing manifest.json")
        chunk_manifest = json.loads(manifest_path.read_text())
        chunk_manifests.append(chunk_manifest)
        chunk_start = date.fromisoformat(chunk_manifest["window"]["start_date"])
        chunk_end = date.fromisoformat(chunk_manifest["window"]["end_date"])
        window_start = (
            chunk_start
            if window_start is None or chunk_start < window_start
            else window_start
        )
        window_end = (
            chunk_end if window_end is None or chunk_end > window_end else window_end
        )
        if chunk_manifest.get("status") == "quarantined":
            quarantines.append(
                {
                    "chunk": chunk_dir.name,
                    **chunk_manifest.get("quarantine", {}),
                }
            )

        inv_path = chunk_dir / "report_inventory.jsonl"
        for line in inv_path.read_text().splitlines():
            row = json.loads(line)
            if row["report_path"] in seen_paths:
                continue
            seen_paths.add(row["report_path"])
            inventory.append(row)

        tx_path = chunk_dir / "transactions.jsonl"
        for line in tx_path.read_text().splitlines():
            row = json.loads(line)
            key = (row["source_record_id"], row["source_row_id"])
            if key in seen_rows:
                continue
            seen_rows.add(key)
            transactions.append(row)

        papers_dir = chunk_dir / "papers"
        if papers_dir.is_dir():
            for pdf in sorted(papers_dir.glob("*.pdf")):
                target = window_dir / "papers" / pdf.name
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    target.write_bytes(pdf.read_bytes())
                elif target.read_bytes() != pdf.read_bytes():
                    raise SenateSweepError(
                        f"paper artifact collision for {pdf.name} across chunks"
                    )
                papers.append(
                    {
                        "source_record_id": pdf.stem,
                        "file": f"papers/{pdf.name}",
                        "sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
                        "bytes": pdf.stat().st_size,
                    }
                )

    inventory.sort(key=lambda r: str(r.get("official_filing_date", "")))
    transactions.sort(
        key=lambda r: (
            str(r.get("source_record_id", "")),
            str(r.get("source_row_id", "")),
        )
    )
    window_dir.mkdir(parents=True, exist_ok=True)
    write_inventory_jsonl(inventory, window_dir / "report_inventory.jsonl")
    tx_file = window_dir / "transactions.jsonl"
    with tx_file.open("w", encoding="utf-8") as handle:
        for row in transactions:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    summary = {
        "found": len(inventory),
        "parsed": sum(1 for r in inventory if r["outcome"] == "parsed"),
        "paper_only": sum(1 for r in inventory if r["outcome"] == "paper_only"),
        "unavailable": sum(1 for r in inventory if r["outcome"] == "unavailable"),
        "failed": sum(1 for r in inventory if r["outcome"] == "failed"),
    }
    canaries = verify_canaries(inventory, expected=_CANARY_EXPECTED_ANY)
    quarantine = None
    if quarantines:
        quarantine = {
            "error": "one or more chunks quarantined",
            "missing": {
                "unavailable": sum(
                    q.get("missing", {}).get("unavailable") or 0 for q in quarantines
                ),
                "failed": sum(
                    q.get("missing", {}).get("failed") or 0 for q in quarantines
                ),
            },
            "chunks": quarantines,
        }

    write_manifest(
        window_dir,
        generation=window_dir.name,
        start_date=window_start or date.today(),
        end_date=window_end or date.today(),
        inventory=inventory,
        summary=summary,
        transactions_file=tx_file,
        papers=papers,
        canaries=canaries,
        quarantine=quarantine,
    )
    (window_dir / "chunks.json").write_text(
        json.dumps(
            {
                "chunks": [c.name for c in chunks],
                "manifests": [
                    {
                        "chunk": c.name,
                        "file": c.relative_to(window_dir).as_posix() + "/manifest.json",
                        "sha256": sha256_file(c / "manifest.json")[0],
                    }
                    for c in chunks
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(
        f"merge {window_dir.name}: found={summary['found']} parsed={summary['parsed']} "
        f"paper_only={summary['paper_only']} unavailable={summary['unavailable']} "
        f"failed={summary['failed']} transactions={len(transactions)}"
    )
    for key, result in canaries["results"].items():
        print(f"canary {key}: {result}")
    if quarantine is not None:
        print("window QUARANTINED — see manifest quarantine for exact missing counts")
        return 2
    if not canaries["passed"]:
        print("merge COMPLETE but canary verification FAILED:")
        for failure in canaries["failures"]:
            print(f"  {failure}")
        return 2
    print("merge COMPLETE — all window canaries passed")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generation", default=GENERATION, help="Staging generation id"
    )
    parser.add_argument(
        "--staging-root",
        type=Path,
        default=Path(".staging") / "senate",
        help="Staging root directory (default: .staging/senate)",
    )
    parser.add_argument(
        "--start",
        default=None,
        help="Sweep start date YYYY-MM-DD (default: one year before end)",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="Sweep end date YYYY-MM-DD (default: today)",
    )
    parser.add_argument(
        "--canary-expect",
        choices=("exact", "any", "none"),
        default="exact",
        help=(
            "exact: Britt=1/Scott=12 parsed row counts gate the run (live "
            "canaries); any: at least one parsed report per canary member "
            "gates (frozen window); none: record canary results, no gate "
            "(per-chunk runs)"
        ),
    )
    parser.add_argument(
        "--merge-window",
        default=None,
        help=(
            "Consolidate chunk-* subdirectories under this window generation "
            "into a single inventory/transactions/manifest (relative to "
            "--staging-root)"
        ),
    )
    args = parser.parse_args(argv)

    if args.merge_window:
        try:
            return merge_window(args.staging_root / args.merge_window)
        except SenateSweepError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

    end_date = _parse_date(args.end) if args.end else date.today()
    if args.start:
        start_date = _parse_date(args.start)
    else:
        try:
            start_date = end_date.replace(year=end_date.year - 1)
        except ValueError:
            start_date = end_date.replace(year=end_date.year - 1, day=28)
    if start_date > end_date:
        print("error: --start must be on or before --end", file=sys.stderr)
        return 1

    try:
        return run_sweep(
            args.generation,
            args.staging_root,
            start_date,
            end_date,
            canary_expect=args.canary_expect,
        )
    except SenateSweepError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except SenateEFDError as exc:
        print(f"error: efdsearch sweep failed hard: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
