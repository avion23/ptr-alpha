import requests
import requests_cache
import yfinance as yf
import zipfile
import camelot
import os
import pathlib
import logging
import asyncio
import aiohttp
from datetime import datetime
from io import BytesIO
from multiprocessing import Pool
from analyzer.exceptions import DataSourceError, ParsingError
from analyzer.parsing import (
    parse_pdf_table,
    normalize_house_metadata,
    consolidate_transactions,
    extract_tables_with_ocr,
)
from analyzer.models import DownloadResult, DownloadStatus, FilingType
from analyzer.interfaces import TransactionSource, PriceSource
from analyzer.database import Database

logger = logging.getLogger(__name__)

_cache_initialized = False


def _ensure_request_cache():
    global _cache_initialized
    if not _cache_initialized:
        requests_cache.install_cache("http_cache", backend="sqlite", expire_after=3600)
        _cache_initialized = True


def _parse_pdf_worker(pdf_path):
    try:
        transactions = []
        tables = camelot.read_pdf(str(pdf_path), pages="all", flavor="lattice")
        for table in tables:
            transactions.extend(parse_pdf_table(table.data))
        if not tables:
            tables = camelot.read_pdf(str(pdf_path), pages="all", flavor="stream")
            for table in tables:
                transactions.extend(parse_pdf_table(table.data))
        if not transactions:
            ocr_tables = extract_tables_with_ocr(pdf_path)
            for table in ocr_tables:
                transactions.extend(parse_pdf_table(table))
        return pdf_path, transactions
    except Exception as e:
        logger.warning(f"Failed to parse PDF {pdf_path}: {e}")
        return pdf_path, []


class HouseTransactionSource(TransactionSource):
    def __init__(self, settings, read_only: bool = False):
        self.settings = settings
        self.data_dir = pathlib.Path(settings.data.data_dir)
        self.metadata_url_template = settings.sources.house_metadata_url
        self.pdf_url_template = settings.sources.house_pdf_url
        self.parallel_workers = settings.data.get_workers()
        _ensure_request_cache()
        self.db = Database(self.data_dir / "congress.duckdb", read_only=read_only)

    def close(self):
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def get_transactions(self, year):
        if not self.db.transactions_exist(year):
            raise DataSourceError(
                f"No cached data found for {year}. Run 'insider-trading parse --year {year}' first."
            )

        df = self.db.get_transactions(year)
        logger.info(f"Loaded {len(df)} cached transactions for {year}")
        return df

    def fetch_metadata(self, year, refresh=False):
        if not refresh and self.db.metadata_exists(year):
            logger.info(f"Loading cached metadata for {year}")
            return self.db.get_metadata(year)

        if refresh:
            self.db.clear_metadata(year)

        metadata_url = self.metadata_url_template.format(year=year)
        try:
            logger.info(f"Downloading metadata for {year} from House disclosures")
            response = requests.get(metadata_url, timeout=30)
            if response.status_code != 200:
                raise DataSourceError(
                    f"Failed to fetch metadata for {year}, status: {response.status_code}"
                )

            with zipfile.ZipFile(BytesIO(response.content)) as z:
                text_files = [f for f in z.namelist() if f.endswith(".txt")]
                if not text_files:
                    raise ParsingError(
                        f"No text files found in metadata ZIP for {year}"
                    )

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

            self.db.upsert_metadata(df)
            logger.info(f"Cached metadata for {year}: {len(df)} records")

            return self.db.get_metadata(year)
        except requests.RequestException as e:
            raise DataSourceError(f"Failed to fetch metadata for {year}: {e}")

    async def _download_pdf_async(self, session, doc_id, pdf_path, url):
        if pdf_path.exists():
            return DownloadResult(doc_id=doc_id, status=DownloadStatus.SKIPPED)

        try:
            async with session.get(
                url, timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
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
            return DownloadResult(
                doc_id=doc_id, status=DownloadStatus.ERROR, error_message=str(e)
            )

    async def _fetch_and_cache_pdfs_async(self, year):
        metadata = self.fetch_metadata(year)
        ptrs = metadata[metadata["FilingType"] == FilingType.PTR.value]
        pdf_dir = self.data_dir / str(year) / "pdfs"
        os.makedirs(pdf_dir, exist_ok=True)

        logger.info(f"Processing {len(ptrs)} PTR filings for {year}")

        connector = aiohttp.TCPConnector(limit=10)
        async with aiohttp.ClientSession(connector=connector) as session:
            async with asyncio.TaskGroup() as tg:
                tasks = []
                for _, row in ptrs.iterrows():
                    doc_id = row["DocID"]
                    pdf_path = pdf_dir / f"{doc_id}.pdf"
                    url = self.pdf_url_template.format(year=year, doc_id=doc_id)
                    task = tg.create_task(
                        self._download_pdf_async(session, doc_id, pdf_path, url)
                    )
                    tasks.append(task)

            results = [task.result() for task in tasks]

        downloaded = sum(1 for r in results if r.status == DownloadStatus.SUCCESS)
        skipped = sum(1 for r in results if r.status == DownloadStatus.SKIPPED)
        failed = sum(
            1
            for r in results
            if r.status in (DownloadStatus.FAILED, DownloadStatus.ERROR)
        )

        logger.info(
            f"PDF download complete: {downloaded} downloaded, {skipped} skipped, {failed} failed"
        )

    def fetch_and_cache_pdfs(self, year):
        return asyncio.run(self._fetch_and_cache_pdfs_async(year))

    def parse_cached_pdfs(self, year):
        metadata = self.fetch_metadata(year)
        ptrs = metadata[metadata["FilingType"] == FilingType.PTR.value]
        pdf_dir = self.data_dir / str(year) / "pdfs"

        if not pdf_dir.exists():
            raise DataSourceError(f"PDF directory not found: {pdf_dir}")

        logger.info(f"Parsing {len(ptrs)} PDFs for {year}")

        pdf_paths = []
        member_lookup = {}
        for _, row in ptrs.iterrows():
            pdf_path = pdf_dir / f"{row['DocID']}.pdf"
            if pdf_path.exists():
                pdf_paths.append(pdf_path)
                member_lookup[row["DocID"]] = {
                    "First": row["First"],
                    "Last": row["Last"],
                    "FilingDate": row["FilingDate"],
                }

        if not pdf_paths:
            raise DataSourceError(f"No PDF files found in {pdf_dir}")

        with Pool(self.parallel_workers) as pool:
            results = pool.map(_parse_pdf_worker, pdf_paths)

        pdf_transactions = {
            pdf_path: transactions for pdf_path, transactions in results
        }
        df = consolidate_transactions(pdf_transactions, member_lookup)

        if df.empty:
            raise ParsingError("No transactions found after parsing all PDFs")

        self.db.upsert_transactions(df)
        logger.info(f"Saved {len(df)} transactions to database")


class YFinancePriceSource(PriceSource):
    def __init__(self, settings, read_only: bool = False):
        self.settings = settings
        self.data_dir = pathlib.Path(settings.data.data_dir)
        self.db = Database(self.data_dir / "congress.duckdb", read_only=read_only)

    def close(self):
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def get_prices(self, tickers, start, end):
        if len(tickers) == 0:
            raise DataSourceError("No tickers provided for price fetching")

        all_tickers = sorted(list(set(tickers) | {"SPY"}))

        cached_prices = self.db.get_prices(all_tickers, start, end)
        if not cached_prices.empty:
            logger.info(
                f"Loaded cached prices: {len(cached_prices.columns)} tickers, {len(cached_prices)} dates"
            )

        missing_tickers, missing_dates = self.db.get_missing_price_data(
            all_tickers, start, end
        )

        if not missing_tickers and not missing_dates:
            logger.info(f"Using fully cached prices for {len(all_tickers)} tickers")
            available_tickers = [t for t in all_tickers if t in cached_prices.columns]
            return cached_prices[available_tickers].dropna(axis=1, how="all")

        fetch_tickers = missing_tickers if missing_tickers else all_tickers
        logger.info(
            f"Fetching price data for {len(fetch_tickers)} tickers using yfinance"
        )

        data = yf.download(
            fetch_tickers,
            start=start,
            end=end,
            progress=False,
            threads=True,
            auto_adjust=True,
        )

        if data.empty:
            if not cached_prices.empty:
                logger.warning("yfinance failed, using cached data")
                available_tickers = [
                    t for t in all_tickers if t in cached_prices.columns
                ]
                return cached_prices[available_tickers].dropna(axis=1, how="all")
            raise DataSourceError(
                "No price data could be fetched from yfinance. Data source may be blocked or down."
            )

        new_prices = (
            data["Close"]
            if len(fetch_tickers) > 1
            else data["Close"].to_frame(fetch_tickers[0])
        )
        new_prices = new_prices.dropna(axis=1, how="all")

        self.db.upsert_prices(new_prices)
        logger.info(f"Cached {len(new_prices.columns)} tickers to database")

        prices = self.db.get_prices(all_tickers, start, end)

        failed_tickers = sorted(set(all_tickers) - set(prices.columns))
        success_count = len([t for t in all_tickers if t in prices.columns])
        success_rate = success_count / len(all_tickers)

        if success_rate < 0.75:
            raise DataSourceError(
                f"Price fetch failure rate too high: {(1 - success_rate) * 100:.1f}% failed ({len(failed_tickers)}/{len(all_tickers)}). Analysis would be unreliable."
            )

        if failed_tickers:
            logger.warning(
                f"Failed to fetch price data for {len(failed_tickers)} tickers: {', '.join(failed_tickers[:10])}{'...' if len(failed_tickers) > 10 else ''}"
            )

        logger.info(
            f"Successfully fetched prices for {success_count}/{len(all_tickers)} tickers ({success_rate * 100:.1f}% success)"
        )
        available_tickers = [t for t in all_tickers if t in prices.columns]
        return prices[available_tickers].dropna(axis=1, how="all")
