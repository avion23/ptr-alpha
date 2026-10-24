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


def _row_artifact_matches(
    out_root: Path,
    envelope: dict,
    manifest_hashes: dict | None,
    *,
    empty: bool,
) -> bool:
    """Check recorded row hashes; for empty claims, also require zero rows."""
    doc_id = envelope.get("doc_id")
    rows_file = f"rows/{doc_id}.jsonl"
    if envelope.get("rows_file") != rows_file:
        return False
    if empty and (
        type(envelope.get("row_count")) is not int or envelope["row_count"] != 0
    ):
        return False
    rows_path = out_root / rows_file
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
    out_root = Path(out_root)
    docs_dir = out_root / "docs"
    demoted: list[str] = []
    restored: list[str] = []
    kept: list[str] = []

    manifest_path = out_root / "manifest.json"
    manifest_hashes = None
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        hashes = manifest.get("staged_files_sha256")
        manifest_hashes = hashes if isinstance(hashes, dict) else {}

    documents = []
    for envelope_path in sorted(docs_dir.glob("*.json")):
        envelope = json.loads(envelope_path.read_text())
        reasons = envelope.get("reasons") or []
        marker = any(
            isinstance(reason, str) and reason.startswith("reclassified:")
            for reason in reasons
        )
        needs_empty_rows = envelope.get("status") == "no_txs" or (
            envelope.get("status") == "unresolved" and marker
        )
        if needs_empty_rows or marker:
            if not _row_artifact_matches(
                out_root, envelope, manifest_hashes, empty=needs_empty_rows
            ):
                claim = "zero-row claim" if needs_empty_rows else "recorded row hashes"
                raise ValueError(
                    f"{envelope.get('doc_id')}: refusing reclassification; "
                    f"{claim} conflict with the staged row artifact or hash"
                )
        documents.append((envelope_path, envelope))

    for envelope_path, envelope in documents:
        doc_id = envelope["doc_id"]
        page_count = int(envelope.get("page_count") or 0)
        reasons = envelope.get("reasons") or []
        page_evidence, malformed_page_evidence = _page_evidence(reasons)
        nothing_pages = sum(
            readings == [REASON_NOTHING] for readings in page_evidence.values()
        )
        cover_pages = sum(
            any(reading.startswith(REASON_COVER) for reading in readings)
            for readings in page_evidence.values()
        )
        all_nothing = (
            page_count > 0
            and not malformed_page_evidence
            and set(page_evidence) == set(range(1, page_count + 1))
            and all(
                page_evidence[page] == [REASON_NOTHING]
                for page in range(1, page_count + 1)
            )
        )
        marker = any(
            isinstance(reason, str) and reason.startswith("reclassified:")
            for reason in reasons
        )
        if envelope.get("status") == "no_txs":
            if all_nothing:
                kept.append(doc_id)
                continue
            envelope["status"] = "unresolved"
            envelope["uncovered_pages"] = list(range(1, page_count + 1))
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
            envelope["covered_pages"] = list(range(1, page_count + 1))
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
