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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.ocr_local_sweep import write_manifest  # noqa: E402

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


def _page_evidence(reasons: list[str]) -> tuple[int, int]:
    """Count page-scoped nothing-to-report vs cover-page reasons."""
    nothing = sum(
        1
        for r in reasons
        if re.match(r"^page \d+: " + re.escape(REASON_NOTHING), r)
    )
    cover = sum(
        1
        for r in reasons
        if re.match(r"^page \d+: " + re.escape(REASON_COVER), r)
    )
    return nothing, cover


def reclassify_no_txs(out_root: str | Path) -> dict:
    """Idempotent two-way reconciliation of the no_txs evidence rule."""
    out_root = Path(out_root)
    docs_dir = out_root / "docs"
    demoted: list[str] = []
    restored: list[str] = []
    kept: list[str] = []
    for envelope_path in sorted(docs_dir.glob("*.json")):
        envelope = json.loads(envelope_path.read_text())
        doc_id = envelope["doc_id"]
        page_count = int(envelope.get("page_count") or 0)
        reasons = envelope.get("reasons") or []
        nothing_pages, cover_pages = _page_evidence(reasons)
        all_nothing = (
            nothing_pages == page_count and cover_pages == 0 and page_count > 0
        )
        marker = next((r for r in reasons if r.startswith("reclassified:")), None)
        if envelope.get("status") == "no_txs":
            if all_nothing:
                kept.append(doc_id)
                continue
            envelope["status"] = "unresolved"
            envelope["uncovered_pages"] = list(range(1, page_count + 1))
            envelope["covered_pages"] = []
            if marker is None:
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
        if marker is not None and all_nothing:
            envelope["status"] = "no_txs"
            envelope["uncovered_pages"] = []
            envelope["covered_pages"] = list(range(1, page_count + 1))
            envelope["reasons"] = [
                r for r in reasons if not r.startswith("reclassified:")
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
