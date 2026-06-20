#!/usr/bin/env python3
"""Download missing House PTR PDFs for specified years.

Reuses the download logic from analyzer.datasources.HouseTransactionSource
but as a standalone script so it can run without the CLI framework.
"""
import asyncio
import logging
import sys
from pathlib import Path

# Add src to path so analyzer imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import aiohttp
import requests_cache
import pandas as pd

from analyzer.database import Database
from analyzer.models import DownloadResult, DownloadStatus, FilingType
from analyzer.parsing import normalize_house_metadata
from analyzer.settings import Settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def fetch_metadata_for_year(
    settings: Settings, db: Database, year: int, session: requests_cache.CachedSession
) -> pd.DataFrame:
    """Fetch or load cached metadata for a given year."""
    if db.metadata_exists(year):
        logger.info(f"Loading cached metadata for {year}")
        return db.get_metadata(year)

    metadata_url = settings.sources.house_metadata_url.format(year=year)
    logger.info(f"Downloading metadata for {year}")
    response = session.get(metadata_url, timeout=30)
    if response.status_code != 200:
        logger.error(f"Failed to fetch metadata for {year}: HTTP {response.status_code}")
        return pd.DataFrame()

    import zipfile
    from io import BytesIO
    from datetime import datetime

    with zipfile.ZipFile(BytesIO(response.content)) as z:
        text_files = [f for f in z.namelist() if f.endswith(".txt")]
        if not text_files:
            logger.error(f"No text files in metadata ZIP for {year}")
            return pd.DataFrame()
        with z.open(text_files[0]) as f:
            content = f.read().decode("utf-8", errors="ignore")

    df = normalize_house_metadata(content)
    df["fetched_at"] = datetime.now()
    df = df.rename(
        columns={
            "DocID": "doc_id",
            "First": "first_name",
            "Last": "last_name",
            "FilingDate": "filing_date",
            "FilingType": "filing_type",
        }
    )
    db.upsert_metadata(df)
    logger.info(f"Cached metadata for {year}: {len(df)} records")
    return db.get_metadata(year)


async def download_pdf(session, doc_id, pdf_path, url):
    """Download a single PDF, skipping if it already exists."""
    if pdf_path.exists():
        return DownloadResult(doc_id=doc_id, status=DownloadStatus.SKIPPED)

    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
            if response.status == 200:
                content = await response.read()
                with open(pdf_path, "wb") as f:
                    f.write(content)
                return DownloadResult(doc_id=doc_id, status=DownloadStatus.SUCCESS)
            else:
                return DownloadResult(
                    doc_id=doc_id,
                    status=DownloadStatus.FAILED,
                    status_code=response.status,
                    error_message=f"HTTP {response.status}",
                )
    except (aiohttp.ClientError, OSError, asyncio.TimeoutError) as e:
        return DownloadResult(doc_id=doc_id, status=DownloadStatus.ERROR, error_message=str(e))


async def download_missing_pdfs(years: list[int]):
    settings = Settings()
    data_dir = Path(settings.data.data_dir)
    db = Database(data_dir / "congress.duckdb", read_only=True)
    session = requests_cache.CachedSession(
        cache_name=str(data_dir / "http_cache"),
        backend="sqlite",
        expire_after=3600,
    )

    for year in years:
        logger.info(f"=== Processing year {year} ===")
        metadata = fetch_metadata_for_year(settings, db, year, session)
        if metadata.empty:
            continue

        ptrs = metadata[metadata["FilingType"] == FilingType.PTR.value]
        pdf_dir = data_dir / str(year) / "pdfs"
        pdf_dir.mkdir(parents=True, exist_ok=True)

        doc_ids = ptrs["DocID"].values
        pdf_paths = [pdf_dir / f"{doc_id}.pdf" for doc_id in doc_ids]
        urls = [settings.sources.house_pdf_url.format(year=year, doc_id=doc_id) for doc_id in doc_ids]

        missing = sum(1 for p in pdf_paths if not p.exists())
        logger.info(f"Year {year}: {len(doc_ids)} PTR filings, {missing} missing PDFs, {len(doc_ids) - missing} already downloaded")

        if missing == 0:
            continue

        connector = aiohttp.TCPConnector(limit=10)
        async with aiohttp.ClientSession(connector=connector) as aio_session:
            async with asyncio.TaskGroup() as tg:
                tasks = [
                    tg.create_task(download_pdf(aio_session, doc_id, pdf_path, url))
                    for doc_id, pdf_path, url in zip(doc_ids, pdf_paths, urls)
                ]

        results = [task.result() for task in tasks]
        downloaded = sum(1 for r in results if r.status == DownloadStatus.SUCCESS)
        skipped = sum(1 for r in results if r.status == DownloadStatus.SKIPPED)
        failed = sum(1 for r in results if r.status in (DownloadStatus.FAILED, DownloadStatus.ERROR))

        logger.info(f"Year {year} complete: {downloaded} downloaded, {skipped} skipped, {failed} failed")

    db.close()
    session.close()


def main():
    years = [2021, 2022, 2023]
    if len(sys.argv) > 1:
        years = [int(y) for y in sys.argv[1:]]
    logger.info(f"Downloading missing PDFs for years: {years}")
    asyncio.run(download_missing_pdfs(years))
    logger.info("Done.")


if __name__ == "__main__":
    main()
