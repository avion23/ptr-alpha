"""Operational OCR entrypoints delegate to the single durable coordinator."""
import argparse
import json
import os
from pathlib import Path

from analyzer.ocr.paths import DATA_DIR


def _ocr_engine_mode(env=None) -> str:
    from analyzer.vision_engines import dual_read_enabled

    source = os.environ if env is None else env
    if dual_read_enabled(source):
        return "dual"
    mode = str(source.get("OCR_ENGINE") or "gemini").strip().lower()
    if mode not in {"gemini", "visionb", "dual"}:
        raise ValueError(f"unsupported OCR_ENGINE: {mode!r}")
    return mode


def main(*, staging_runner=None) -> int:
    """Keep the executable, not its old competing database-writing loop."""
    from analyzer.ocr.coordinator import run_pipeline
    from analyzer.ocr.ingestion import WORK_MANIFEST_DEFAULT_PATH, export_work_manifest
    from analyzer.persistence.ownership import resolve_read_path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--year", type=int, action="append")
    parser.add_argument("--refresh", action="store_true", help="Explicit current-recipe refresh")
    parser.add_argument("--export-work-manifest", nargs="?", const=WORK_MANIFEST_DEFAULT_PATH)
    args = parser.parse_args()
    if args.export_work_manifest:
        result = export_work_manifest(args.export_work_manifest, data_dir=args.data_dir,
                                      db_path=str(resolve_read_path(Path(args.data_dir) / "congress.duckdb")))
        print(json.dumps(result, default=str))
        return 0
    mode = _ocr_engine_mode()
    if mode == "dual":
        parser.error("dual-read drain retired; configure explicit coordinator lanes after policy review")
    result = run_pipeline(args.data_dir, args.year, local_workers=0,
                          gemini_workers=1 if mode == "gemini" else 0,
                          glm_workers=1 if mode == "visionb" else 0,
                          refresh_recipe=args.refresh)
    print(json.dumps(result, default=str))
    return int(result["unresolved"] > 0 or bool(result["missing_years"]))


def run_gemini_ocr_for_year(year: int, data_dir: str = str(DATA_DIR),
                            refresh: bool = False, force_partial_canonical: bool = False):
    from analyzer.ocr.coordinator import run_pipeline

    if force_partial_canonical:
        raise ValueError("the durable pipeline does not admit forced partial canonical rowsets")
    if _ocr_engine_mode() == "dual":
        raise ValueError("dual-read drain retired; coordinator Gemini lane is single-engine only")
    result = run_pipeline(data_dir, [year], local_workers=0, gemini_workers=1,
                          refresh_recipe=refresh)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
