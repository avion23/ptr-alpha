#!/usr/bin/env python3
"""Live House archive metadata coverage audit (read-only).

Fetches the official House disclosure metadata archives (``{year}FD.ZIP``) for
an audited year range and reconciles the expected PTR doc IDs against a local
PDF inventory laid out as ``<local_dir>/<year>/pdfs/<doc_id>.pdf``. Emits a
coverage table JSON with per-year expected/local/missing/orphan counts and the
excluded legacy 2008-2014 range.

This is an audit only: the script never writes to ``data/`` or the database,
and the only network traffic is read-only GET of the public archives. Metadata
parsing reuses the production ``analyzer.download._read_first_text_from_zip``
and ``analyzer.parsing.metadata.normalize_house_metadata`` so the audit
measures exactly what production ingestion would consume.

Run (system interpreter, src on the path, like the other audit scripts):

    PYTHONPATH=src python3 scripts/audit_house_coverage.py \
        /path/to/local_pdfs /tmp/house_coverage.json

Exit status is 0 only when every audited year archive fetched and parsed;
coverage gaps (missing/orphan PDFs) are findings, not script failures.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

from analyzer.download import _read_first_text_from_zip
from analyzer.models import FilingType
from analyzer.parsing.metadata import normalize_house_metadata

METADATA_URL_TEMPLATE = (
    "https://disclosures-clerk.house.gov/public_disc/financial-pdfs/{year}FD.ZIP"
)
DEFAULT_FIRST_YEAR = 2015
DEFAULT_LAST_YEAR = 2026
DEFAULT_LEGACY_FIRST_YEAR = 2008
DEFAULT_LEGACY_LAST_YEAR = 2014
LEGACY_EXCLUSION_REASON = (
    "Pre-2015 archives use a different metadata schema (Filing Year / "
    "DisclosureType columns) and filing-type taxonomy with no 'P' PTR type; "
    "out of audit scope"
)


def fetch_metadata_archive(
    year: int, session: requests.Session, timeout: int
) -> tuple[int, bytes | None, str | None]:
    """GET the year archive; return (http_status, zip_bytes, last_modified)."""
    url = METADATA_URL_TEMPLATE.format(year=year)
    response = session.get(url, timeout=timeout)
    return response.status_code, response.content, response.headers.get("Last-Modified")


def parse_metadata(zip_bytes: bytes, year: int) -> Any:
    """Return the normalized metadata table for one archive year."""
    return normalize_house_metadata(_read_first_text_from_zip(zip_bytes, year))


def scan_local_inventory(local_dir: Path) -> dict[int, set[str]]:
    """Map every ``<local_dir>/<year>/pdfs`` directory to its PDF doc IDs."""
    inventory: dict[int, set[str]] = {}
    if not local_dir.is_dir():
        return inventory
    for year_dir in sorted(
        (entry for entry in local_dir.iterdir() if entry.is_dir()),
        key=lambda entry: entry.name,
    ):
        try:
            year = int(year_dir.name)
        except ValueError:
            continue
        pdf_dir = year_dir / "pdfs"
        if not pdf_dir.is_dir():
            continue
        doc_ids = {path.stem for path in pdf_dir.glob("*.pdf")}
        if doc_ids:
            inventory[year] = doc_ids
    return inventory


def build_year_coverage(
    metadata: Any, local_ids: set[str], year: int
) -> dict[str, Any]:
    """Reconcile one year's expected PTR doc IDs against the local inventory."""
    ptr_mask = metadata["FilingType"] == FilingType.PTR.value
    ptr_doc_ids = set(metadata.loc[ptr_mask, "DocID"].astype(str))
    expected_count = len(ptr_doc_ids)
    covered = len(ptr_doc_ids & local_ids)
    missing = sorted(ptr_doc_ids - local_ids)
    orphans = sorted(local_ids - ptr_doc_ids)
    return {
        "year": year,
        "metadata_records": int(len(metadata)),
        "ptr_expected": expected_count,
        "amendments": int((metadata["FilingType"] == FilingType.AMENDMENT.value).sum()),
        "local_pdfs": len(local_ids),
        "covered": covered,
        "coverage_ratio": round(covered / expected_count, 4)
        if expected_count
        else None,
        "missing": len(missing),
        "missing_doc_ids": missing,
        "orphans": len(orphans),
        "orphan_doc_ids": orphans,
        "filing_type_breakdown": dict(Counter(metadata["FilingType"].astype(str))),
    }


def run_audit(
    local_dir: Path,
    first_year: int,
    last_year: int,
    legacy_first_year: int,
    legacy_last_year: int,
    timeout: int,
    session: requests.Session,
) -> dict[str, Any]:
    audited_years = list(range(first_year, last_year + 1))
    audited_set = set(audited_years)
    legacy_years = list(range(legacy_first_year, legacy_last_year + 1))
    inventory = scan_local_inventory(local_dir)

    year_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for year in audited_years:
        local_ids = inventory.get(year, set())
        row: dict[str, Any] = {"year": year, "error": None}
        try:
            status, content, last_modified = fetch_metadata_archive(
                year, session, timeout
            )
            if status != 200 or content is None:
                raise RuntimeError(f"HTTP {status} fetching {year}FD.ZIP")
            metadata = parse_metadata(content, year)
            row.update(build_year_coverage(metadata, local_ids, year))
            row["metadata_last_modified"] = last_modified
            row["fetched_at"] = datetime.now(UTC).isoformat()
        except Exception as exc:  # one broken year must not hide the rest
            row.update(
                {
                    "metadata_records": None,
                    "ptr_expected": None,
                    "amendments": None,
                    "local_pdfs": len(local_ids),
                    "covered": 0,
                    "coverage_ratio": None,
                    "missing": None,
                    "missing_doc_ids": [],
                    "orphans": None,
                    "orphan_doc_ids": [],
                    "filing_type_breakdown": {},
                }
            )
            row["error"] = f"{type(exc).__name__}: {exc}"
            errors.append(row)
        year_rows.append(row)

    out_of_scope_years = sorted(set(inventory).difference(audited_set))
    out_of_scope = {str(year): sorted(inventory[year]) for year in out_of_scope_years}

    audited_rows = [row for row in year_rows if row["error"] is None]
    summary = {
        "years_audited": len(audited_years),
        "years_failed": len(errors),
        "expected_ptrs_total": sum(row["ptr_expected"] or 0 for row in audited_rows),
        "local_pdfs_total": sum(row["local_pdfs"] for row in audited_rows),
        "covered_total": sum(row["covered"] for row in audited_rows),
        "missing_total": sum(row["missing"] or 0 for row in audited_rows),
        "orphans_total": sum(row["orphans"] or 0 for row in audited_rows),
    }

    return {
        "audit": "house_metadata_coverage",
        "generated_at": datetime.now(UTC).isoformat(),
        "scope": {
            "local_pdf_dir": str(local_dir),
            "audited_years": audited_years,
            "metadata_url_template": METADATA_URL_TEMPLATE,
            "pdf_layout": "<local_dir>/<year>/pdfs/<doc_id>.pdf",
        },
        "excluded_legacy": {
            "years": legacy_years,
            "reason": LEGACY_EXCLUSION_REASON,
        },
        "local_out_of_scope_pdfs": out_of_scope,
        "summary": summary,
        "years": year_rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Live House archive metadata coverage audit (read-only network; "
            "no data/ or database writes)."
        )
    )
    parser.add_argument("local_pdf_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--first-year", type=int, default=DEFAULT_FIRST_YEAR)
    parser.add_argument("--last-year", type=int, default=DEFAULT_LAST_YEAR)
    parser.add_argument(
        "--legacy-first-year",
        type=int,
        default=DEFAULT_LEGACY_FIRST_YEAR,
        help="Excluded legacy archive range start (default 2008).",
    )
    parser.add_argument(
        "--legacy-last-year",
        type=int,
        default=DEFAULT_LEGACY_LAST_YEAR,
        help="Excluded legacy archive range end (default 2014).",
    )
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args(argv)

    if args.first_year > args.last_year:
        parser.error("--first-year must not exceed --last-year")
    if args.legacy_first_year > args.legacy_last_year:
        parser.error("--legacy-first-year must not exceed --legacy-last-year")

    with requests.Session() as session:
        payload = run_audit(
            local_dir=args.local_pdf_dir,
            first_year=args.first_year,
            last_year=args.last_year,
            legacy_first_year=args.legacy_first_year,
            legacy_last_year=args.legacy_last_year,
            timeout=args.timeout,
            session=session,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")

    summary = payload["summary"]
    print(
        json.dumps(
            {
                "years_audited": summary["years_audited"],
                "years_failed": summary["years_failed"],
                "expected_ptrs_total": summary["expected_ptrs_total"],
                "local_pdfs_total": summary["local_pdfs_total"],
                "covered_total": summary["covered_total"],
                "missing_total": summary["missing_total"],
                "orphans_total": summary["orphans_total"],
            },
            indent=2,
        )
    )
    return 1 if summary["years_failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
