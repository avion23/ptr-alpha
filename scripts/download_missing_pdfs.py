#!/usr/bin/env python3
"""Download and reconcile House PTR PDFs for specified archive years."""

import asyncio
import logging
import sys
from datetime import date
from pathlib import Path

# Add src to path so analyzer imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from analyzer.database import Database
from analyzer.download import HouseTransactionSource
from analyzer.settings import Settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


async def download_missing_pdfs(years: list[int]) -> None:
    """Use the production archive-aware fetch and completeness checks."""
    settings = Settings()
    data_dir = Path(settings.data.data_dir)
    db = Database(data_dir / "congress.duckdb", read_only=False)
    source = HouseTransactionSource(settings, read_only=False, db=db)
    try:
        for archive_year in years:
            logger.info("=== Processing House archive %d ===", archive_year)
            summary = await source._fetch_and_cache_pdfs_async(
                archive_year,
                refresh_metadata=archive_year == date.today().year,
            )
            logger.info(
                "Archive %d complete: metadata=%d PTR=%d valid=%d "
                "downloaded=%d skipped=%d orphan=%d",
                summary.archive_year,
                summary.metadata_count,
                summary.ptr_count,
                summary.valid_pdf_count,
                summary.downloaded_count,
                summary.skipped_count,
                summary.orphan_pdf_count,
            )
    finally:
        source.close()
        db.close()


def main() -> None:
    years = [int(value) for value in sys.argv[1:]] or [date.today().year]
    logger.info("Downloading missing PDFs for House archives: %s", years)
    asyncio.run(download_missing_pdfs(years))
    logger.info("Done.")


if __name__ == "__main__":
    main()
