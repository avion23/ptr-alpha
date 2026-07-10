#!/usr/bin/env python3
"""Non-destructively run the production parser cascade over cached House PDFs."""

import argparse
import json
import os
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

from analyzer.parser_cascade import _is_valid_pdf, _parse_pdf_worker


def _parse_one(path_string: str) -> dict:
    path = Path(path_string)
    try:
        parsed_path, transactions, engines = _parse_pdf_worker(path)
        return {
            "path": str(parsed_path),
            "transactions": len(transactions),
            "engines": engines,
            "error": None,
        }
    except BaseException as exc:
        return {
            "path": str(path),
            "transactions": 0,
            "engines": [],
            "error": f"{type(exc).__name__}: {exc}",
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--with-docling", action="store_true")
    args = parser.parse_args()

    if not args.with_docling:
        os.environ["PTR_SKIP_DOCLING"] = "1"

    paths = sorted(args.data_dir.glob("*/pdfs/*.pdf"))
    invalid = [str(path) for path in paths if not _is_valid_pdf(path)]
    started = time.time()
    with Pool(args.workers) as pool:
        results = list(pool.imap_unordered(_parse_one, map(str, paths), chunksize=1))

    payload = {
        "scope": str(args.data_dir),
        "documents": len(paths),
        "invalid_pdfs": invalid,
        "elapsed_seconds": round(time.time() - started, 2),
        "results": sorted(results, key=lambda result: result["path"]),
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")

    outcomes = Counter(
        result["engines"][-1] if result["engines"] else "exception"
        for result in results
    )
    print(json.dumps({
        "documents": len(paths),
        "invalid": len(invalid),
        "zero": sum(not result["transactions"] for result in results),
        "exceptions": sum(bool(result["error"]) for result in results),
        "transactions": sum(result["transactions"] for result in results),
        "outcomes": outcomes,
        "elapsed_seconds": payload["elapsed_seconds"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
