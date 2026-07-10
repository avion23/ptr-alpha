#!/usr/bin/env python3
"""Non-destructively run the production parser cascade over cached House PDFs."""

import argparse
import io
import json
import logging
import os
import time
import warnings
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

from analyzer.parser_cascade import _is_valid_pdf, _parse_pdf_worker


def _parse_one(path_string: str) -> dict:
    path = Path(path_string)
    data_dir_string = str(path.parents[2])

    def redact(value: str) -> str:
        return value.replace(data_dir_string, "$DATA_DIR")

    diagnostic_stream = io.StringIO()
    handler = logging.StreamHandler(diagnostic_stream)
    cascade_logger = logging.getLogger("analyzer.parser_cascade")
    previous_level = cascade_logger.level
    cascade_logger.addHandler(handler)
    cascade_logger.setLevel(logging.DEBUG)
    try:
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            parsed_path, transactions, engines = _parse_pdf_worker(path)
        return {
            "path": str(parsed_path),
            "transactions": len(transactions),
            "engines": engines,
            "error": None,
            "warnings": [
                redact(f"{warning.category.__name__}: {warning.message}")
                for warning in caught_warnings
            ],
            "diagnostics": [
                redact(line) for line in diagnostic_stream.getvalue().splitlines()
            ],
        }
    except Exception as exc:
        return {
            "path": str(path),
            "transactions": 0,
            "engines": [],
            "error": f"{type(exc).__name__}: {exc}",
            "warnings": [],
            "diagnostics": [
                redact(line) for line in diagnostic_stream.getvalue().splitlines()
            ],
        }
    finally:
        cascade_logger.removeHandler(handler)
        cascade_logger.setLevel(previous_level)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--with-docling", action="store_true")
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be at least 1")
    data_dir = args.data_dir.resolve()

    if not args.with_docling:
        os.environ["PTR_SKIP_DOCLING"] = "1"

    paths = sorted(data_dir.glob("*/pdfs/*.pdf"))
    invalid = [path for path in paths if not _is_valid_pdf(path)]
    invalid_set = set(invalid)
    parse_paths = [path for path in paths if path not in invalid_set]
    started = time.time()
    with Pool(args.workers) as pool:
        results = list(
            pool.imap_unordered(_parse_one, map(str, parse_paths), chunksize=1)
        )

    for result in results:
        result["path"] = str(Path(result["path"]).relative_to(data_dir))

    payload = {
        "scope": str(args.data_dir),
        "documents": len(paths),
        "invalid_pdfs": [str(path.relative_to(data_dir)) for path in invalid],
        "elapsed_seconds": round(time.time() - started, 2),
        "results": sorted(results, key=lambda result: result["path"]),
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")

    outcomes = Counter(
        result["engines"][-1] if result["engines"] else "exception"
        for result in results
    )
    print(
        json.dumps(
            {
                "documents": len(paths),
                "invalid": len(invalid),
                "zero": sum(not result["transactions"] for result in results),
                "exceptions": sum(bool(result["error"]) for result in results),
                "transactions": sum(result["transactions"] for result in results),
                "outcomes": outcomes,
                "elapsed_seconds": payload["elapsed_seconds"],
            },
            indent=2,
        )
    )
    return 1 if invalid or any(result["error"] for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
