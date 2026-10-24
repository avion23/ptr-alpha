"""Isolated, machine-readable year OCR boundary used by the analyzer CLI."""
import argparse
from contextlib import redirect_stdout
import json
import sys

from analyzer.ocr.runner import run_gemini_ocr_for_year


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--data-dir", required=True)
    args = parser.parse_args()
    with redirect_stdout(sys.stderr):
        inserted = run_gemini_ocr_for_year(args.year, data_dir=args.data_dir)
    print(json.dumps({"schema": "ocr-year-summary-v1", "inserted": inserted}))


if __name__ == "__main__":
    main()
