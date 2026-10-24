#!/usr/bin/env python3
"""Reclassify staged no_txs envelopes that lack nothing-to-report evidence.

The v1 no_txs rule treated cover-classified pages (filer block only) as
zero-transaction evidence; 82 of the first 120 staged 2015 no_txs docs were
Gemini-resolved one-page PTRs that local OCR simply could not read.  The
fixed rule (commit bf8cebe) requires explicit nothing-to-report evidence on
EVERY page for terminal no_txs.

This pass reconciles already-staged envelopes (rows are empty for no_txs, so
only docs/<doc>.json changes) and reassembles manifest.json:

  * no_txs envelopes whose pages were only cover-classified are demoted to
    unresolved (fail-closed, left in the unresolved pool);
  * envelopes previously demoted that actually carry nothing-to-report
    evidence on every page are restored to no_txs (the first pass
    over-counted the summary line).

Safe to re-run; idempotent.

Usage:
    python scripts/ocr_local_reclassify_no_txs.py --out .staging/ocr2-local/gen-live-20260810
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if __name__ == "__main__":
    for _entry in (str(REPO_ROOT), str(REPO_ROOT / "src")):
        if _entry not in sys.path:
            sys.path.insert(0, _entry)

from scripts.ocr_local_sweep import sha256_file, write_manifest  # noqa: E402

REASON_NOTHING = "reports no transactions"
REASON_COVER = "cover page (no transaction rows)"


def _atomic_write_json(path: Path, envelope: dict) -> None:
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.stem}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(envelope, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass


def _page_evidence(reasons: list[str]) -> tuple[dict[int, list[str]], bool]:
    """Collect page readings without treating repeated lines as new pages."""
    pages: dict[int, list[str]] = {}
    malformed = False
    for reason in reasons:
        if not isinstance(reason, str):
            malformed = True
            continue
        if not reason.startswith("page "):
            continue
        match = re.fullmatch(r"page ([1-9]\d*): (.+)", reason)
        if match is None:
            malformed = True
            continue
        page, reading = match.groups()
        pages.setdefault(int(page), []).append(reading)
    return pages, malformed


def _page_inventory(envelope: dict) -> tuple[int, list[int]]:
    """Validate the staged page partition and return its finite page set."""
    page_count = envelope.get("page_count")
    covered = envelope.get("covered_pages")
    uncovered = envelope.get("uncovered_pages")
    if type(page_count) is not int or page_count < 1:
        raise ValueError("page_count must be a positive integer")
    if not isinstance(covered, list) or not isinstance(uncovered, list):
        raise ValueError("covered_pages and uncovered_pages must be lists")

    if len(covered) + len(uncovered) != page_count:
        raise ValueError("page inventory does not cover page_count")
    pages = set()
    for inventory in (covered, uncovered):
        for page in inventory:
            if type(page) is not int or not 1 <= page <= page_count or page in pages:
                raise ValueError(
                    "page inventory must contain unique in-range integers"
                )
            pages.add(page)
    return page_count, sorted(pages)


def _bound_rows_path(
    out_root: Path, envelope_path: Path, envelope: dict
) -> Path | None:
    """Return only this envelope's nonsymlink row file inside the staging root."""
    doc_id = envelope.get("doc_id")
    if (
        not isinstance(doc_id, str)
        or not doc_id
        or doc_id in {".", ".."}
        or "/" in doc_id
        or "\\" in doc_id
        or envelope_path.stem != doc_id
    ):
        return None
    rows_file = f"rows/{doc_id}.jsonl"
    if envelope.get("rows_file") != rows_file:
        return None

    root = out_root.resolve()
    rows_dir = root / "rows"
    rows_path = rows_dir / f"{doc_id}.jsonl"
    if rows_dir.is_symlink() or rows_path.is_symlink():
        return None
    try:
        if rows_dir.resolve() != rows_dir or rows_path.resolve() != rows_path:
            return None
    except (OSError, RuntimeError):
        return None
    return rows_path


def _row_artifact_matches(
    out_root: Path,
    envelope_path: Path,
    envelope: dict,
    manifest_hashes: dict | None,
    *,
    empty: bool,
) -> bool:
    """Check recorded row hashes; for empty claims, also require zero rows."""
    rows_path = _bound_rows_path(out_root, envelope_path, envelope)
    if rows_path is None:
        return False
    rows_file = envelope["rows_file"]
    if empty and (
        type(envelope.get("row_count")) is not int or envelope["row_count"] != 0
    ):
        return False
    try:
        if not rows_path.is_file():
            return False
        if empty and rows_path.stat().st_size != 0:
            return False
        actual_sha = sha256_file(rows_path)
    except OSError:
        return False
    return envelope.get("rows_file_sha256") == actual_sha and (
        manifest_hashes is None or manifest_hashes.get(rows_file) == actual_sha
    )


def reclassify_no_txs(out_root: str | Path) -> dict:
    """Idempotent two-way reconciliation of the no_txs evidence rule."""
    out_root = Path(out_root).resolve()
    docs_dir = out_root / "docs"
    if docs_dir.is_symlink():
        raise ValueError("staging docs directory must not be a symlink")
    demoted: list[str] = []
    restored: list[str] = []
    kept: list[str] = []

    manifest_path = out_root / "manifest.json"
    if manifest_path.is_symlink():
        raise ValueError("staging manifest must not be a symlink")
    manifest_hashes = None
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if not isinstance(manifest, dict):
            raise ValueError("staging manifest must be an object")
        hashes = manifest.get("staged_files_sha256")
        manifest_hashes = hashes if isinstance(hashes, dict) else {}

    documents = []
    for envelope_path in sorted(docs_dir.glob("*.json")):
        if envelope_path.is_symlink() or envelope_path.resolve() != envelope_path:
            raise ValueError(f"{envelope_path.name}: envelope escapes staging docs")
        envelope = json.loads(envelope_path.read_text())
        if not isinstance(envelope, dict):
            raise ValueError(f"{envelope_path.name}: envelope must be an object")
        doc_id = envelope.get("doc_id")
        if (
            not isinstance(doc_id, str)
            or not doc_id
            or envelope_path.stem != doc_id
            or doc_id in {".", ".."}
            or "/" in doc_id
            or "\\" in doc_id
        ):
            raise ValueError(f"{envelope_path.name}: doc_id does not match envelope")
        if _bound_rows_path(out_root, envelope_path, envelope) is None:
            raise ValueError(f"{doc_id}: rows_file is not bound to this staged document")
        if type(envelope.get("year")) is not int:
            raise ValueError(f"{doc_id}: year must be an integer")
        if type(envelope.get("row_count")) is not int or envelope["row_count"] < 0:
            raise ValueError(f"{doc_id}: row_count must be a nonnegative integer")
        status = envelope.get("status")
        if not isinstance(status, str) or status not in {
            "resolved",
            "unresolved",
            "no_txs",
        }:
            raise ValueError(f"{doc_id}: invalid staging status")
        if not isinstance(envelope.get("covered_pages"), list) or not isinstance(
            envelope.get("uncovered_pages"), list
        ):
            raise ValueError(f"{doc_id}: page inventories must be lists")
        if "page_count" not in envelope:
            raise ValueError(f"{doc_id}: missing page_count")
        if not isinstance(envelope.get("reasons"), list):
            raise ValueError(f"{doc_id}: reasons must be a list")
        canary = envelope.get("canary")
        if canary is not None and not isinstance(canary, dict):
            raise ValueError(f"{doc_id}: canary must be an object or null")

        reasons = envelope["reasons"]
        marker = any(
            isinstance(reason, str) and reason.startswith("reclassified:")
            for reason in reasons
        )
        page_count = envelope["page_count"]
        if page_count is None:
            if (
                status != "unresolved"
                or envelope["covered_pages"]
                or envelope["uncovered_pages"]
            ):
                raise ValueError(
                    f"{doc_id}: null page_count has inconsistent status/inventory"
                )
            page_inventory = None
        else:
            page_count, page_inventory = _page_inventory(envelope)
            if status in {"resolved", "no_txs"} and (
                len(envelope["covered_pages"]) != page_count
                or envelope["uncovered_pages"]
            ):
                raise ValueError(f"{doc_id}: {status} page inventory is inconsistent")
            if status == "unresolved" and marker and (
                envelope["covered_pages"]
                or len(envelope["uncovered_pages"]) != page_count
            ):
                raise ValueError(f"{doc_id}: reclassified page inventory is inconsistent")

        all_nothing = None
        nothing_pages = cover_pages = 0
        if status == "no_txs" or (status == "unresolved" and marker):
            if page_count is None:
                raise ValueError(f"{doc_id}: reclassification requires a page_count")
            page_evidence, malformed_page_evidence = _page_evidence(reasons)
            if malformed_page_evidence:
                raise ValueError(f"{doc_id}: malformed page evidence")
            all_nothing = len(page_evidence) == page_count and all(
                1 <= page <= page_count and readings == [REASON_NOTHING]
                for page, readings in page_evidence.items()
            )
            nothing_pages = sum(
                readings == [REASON_NOTHING] for readings in page_evidence.values()
            )
            cover_pages = sum(
                any(reading.startswith(REASON_COVER) for reading in readings)
                for readings in page_evidence.values()
            )
        needs_empty_rows = status == "no_txs" or (status == "unresolved" and marker)
        if needs_empty_rows or marker:
            if not _row_artifact_matches(
                out_root,
                envelope_path,
                envelope,
                manifest_hashes,
                empty=needs_empty_rows,
            ):
                claim = "zero-row claim" if needs_empty_rows else "recorded row hashes"
                raise ValueError(
                    f"{doc_id}: refusing reclassification; "
                    f"{claim} conflict with the staged row artifact or hash"
                )
        documents.append(
            (
                envelope_path,
                envelope,
                page_count,
                page_inventory,
                all_nothing,
                marker,
                nothing_pages,
                cover_pages,
            )
        )

    for (
        envelope_path,
        envelope,
        page_count,
        page_inventory,
        all_nothing,
        marker,
        nothing_pages,
        cover_pages,
    ) in documents:
        if page_count is None or all_nothing is None:
            continue
        doc_id = envelope["doc_id"]
        reasons = envelope["reasons"]
        if envelope.get("status") == "no_txs":
            if all_nothing:
                kept.append(doc_id)
                continue
            envelope["status"] = "unresolved"
            envelope["uncovered_pages"] = page_inventory
            envelope["covered_pages"] = []
            if not marker:
                envelope.setdefault("reasons", []).append(
                    "reclassified: cover pages are not zero-transaction evidence "
                    f"(nothing-to-report {nothing_pages}/{page_count}, "
                    f"cover {cover_pages})"
                )
            _atomic_write_json(envelope_path, envelope)
            demoted.append(doc_id)
            continue
        # restore previously-demoted envelopes that do carry all-page
        # nothing-to-report evidence (first pass over-counted the summary)
        if envelope.get("status") == "unresolved" and marker and all_nothing:
            envelope["status"] = "no_txs"
            envelope["uncovered_pages"] = []
            envelope["covered_pages"] = page_inventory
            envelope["reasons"] = [
                r
                for r in reasons
                if not (isinstance(r, str) and r.startswith("reclassified:"))
            ]
            _atomic_write_json(envelope_path, envelope)
            restored.append(doc_id)
    write_manifest(out_root, kind="sweep", data_dir=out_root)
    return {"demoted": demoted, "restored": restored, "kept": kept}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="shared staging root")
    args = parser.parse_args(argv)
    result = reclassify_no_txs(args.out)
    print(
        f"reclassify: demoted {len(result['demoted'])} -> unresolved; "
        f"restored {len(result['restored'])} -> no_txs; kept {len(result['kept'])} no_txs"
    )
    if result["demoted"]:
        print("demoted sample:", result["demoted"][:10])
    if result["restored"]:
        print("restored sample:", result["restored"][:10])
    return 0


if __name__ == "__main__":
    sys.exit(main())
