#!/usr/bin/env python3
"""Compatibility executable for the operational OCR drain."""
import os
import sys
import time

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if __name__ == "__main__":
    for _entry in (str(REPO_ROOT), str(REPO_ROOT / "src")):
        if _entry not in sys.path:
            sys.path.insert(0, _entry)

from analyzer.ocr import runner
from analyzer.ocr.common import is_quota_error
from analyzer.ocr.ingestion import DB_PATH, COOLDOWN, approved_data_root

def _run_visionb_staging(remaining, vision_b_config, *, data_dir=None, _data_root=None) -> None:
    """GLM-primary drain: write STAGING records, never canonical rows.

    Engine A is never contacted and the canonical DB is never opened here:
    every reading lands as a JSON staging record under
    ``data/staging/glm-backlog`` (via ``scripts.glm_stage``'s staging
    functions), and canonical ingestion happens ONLY through
    ``glm_stage promote`` after the pixel audit. Already-staged docs are
    skipped; quota exhaustion stops the batch cleanly; auth/config defects
    alert and stop (every remaining doc would fail the same way).
    """
    if __package__:
        from scripts import glm_stage
    else:
        import glm_stage

    if _data_root is None:
        with approved_data_root(data_dir or os.path.dirname(DB_PATH) or ".") as root:
            return _run_visionb_staging(remaining, vision_b_config, _data_root=root)
    data_dir = _data_root
    for i, (doc_id, year, path) in enumerate(remaining):
        idx = i + 1
        print(f"\n[{idx}/{len(remaining)}] {doc_id} ({year})...", flush=True)
        if glm_stage.load_staging_record(data_dir, year, doc_id) is not None:
            print(
                "  already staged; skipping (canonical ingestion happens "
                "only via glm_stage promote)",
                flush=True,
            )
            continue
        time.sleep(COOLDOWN)
        record = glm_stage.build_staging_record(doc_id, year, path, vision_b_config, data_dir=data_dir)
        error = record.get("error") or ""
        if record["parse_status"] == "error" and (
            record.get("error_kind") == "quota" or is_quota_error(error)
        ):
            print("  Quota exhausted; stopping batch", flush=True)
            return
        if record["parse_status"] == "error" and record.get("error_kind") in (
            "auth",
            "config",
        ):
            print(
                f"ALERT: visionb {record.get('error_kind')} defect: {error}; "
                "stopping batch",
                flush=True,
            )
            return
        glm_stage.write_staging_record(data_dir, record)
        if record["parse_status"] == "error":
            print(
                f"  ERROR: {error} (staged as error record; canonical "
                "ingestion happens only via glm_stage promote)",
                flush=True,
            )
            continue
        print(
            f"  STAGED {record['parse_status']} rows={len(record['rows'])} "
            "(canonical ingestion happens only via glm_stage promote)",
            flush=True,
        )



def main():
    runner.main(staging_runner=_run_visionb_staging)


if __name__ == "__main__":
    main()
