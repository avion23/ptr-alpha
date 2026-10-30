"""Run Gemini extraction through the single-owner OCR coordinator."""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if __name__ == "__main__":
    for entry in (str(REPO_ROOT), str(REPO_ROOT / "src")):
        if entry not in sys.path:
            sys.path.insert(0, entry)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--years", nargs="+", type=int)
    parser.add_argument("--workers", type=int, default=15)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")

    from analyzer.ocr.coordinator import run_pipeline

    return run_pipeline(
        data_dir=args.data_dir,
        years=args.years,
        gemini_workers=args.workers,
    )


if __name__ == "__main__":
    main()
