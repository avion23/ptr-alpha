import asyncio
import logging
import zipfile
from datetime import date, datetime
from io import BytesIO
from multiprocessing import Pool
from pathlib import Path

import aiohttp
import camelot
import pandas as pd
import requests
import requests_cache
import yfinance as yf

from analyzer.database import Database
from analyzer.exceptions import DataSourceError, ParsingError
from analyzer.interfaces import PriceSource, TransactionSource
from analyzer.settings import Settings
from analyzer.models import DownloadResult, DownloadStatus, FilingType
from analyzer.ticker_resolver import TickerResolver
from analyzer.parsing import (
    consolidate_transactions,
    extract_tables_with_docling,
    extract_tables_with_ocr,
    extract_tables_with_pdfplumber,
    extract_tables_with_pdftotext,
    normalize_house_metadata,
    parse_pdf_table,
    _parse_ocr_text_to_rows,
)

logger = logging.getLogger(__name__)


def _parse_pdf_worker(pdf_path: Path) -> tuple[Path, list[dict], list[str]]:
    transactions = []
    engines_attempted = []

    # 1) pdfplumber — benchmark winner for text-based PDFs (0.075s avg).
    # Handles encrypted PDFs natively; returns 0 on scanned images.
    try:
        pp_tables = extract_tables_with_pdfplumber(pdf_path)
        if pp_tables:
            engines_attempted.append("pdfplumber")
            for table in pp_tables:
                txs = parse_pdf_table(table)
                if txs:
                    transactions.extend(txs)
    except Exception as e:
        logger.debug(f"pdfplumber failed for {pdf_path}: {e}")

    # 2) camelot lattice — was the previous primary; keeps PDFs with ruling lines
    if not transactions:
        try:
            tables = camelot.read_pdf(str(pdf_path), pages="all", flavor="lattice")
            engines_attempted.append("lattice")
            for table in tables:
                data = table.data
                # Fix 1: If lattice produced a 1-column table (null bytes collapse),
                # split cell content by newlines and parse as OCR text
                if data and len(data[0]) == 1:
                    for row in data:
                        cell_text = row[0] if row else ""
                        if cell_text:
                            ocr_rows = _parse_ocr_text_to_rows(cell_text)
                            if ocr_rows:
                                ocr_table = [['Asset Name', 'Transaction Type', 'Transaction Date', 'Amount']] + ocr_rows
                                transactions.extend(parse_pdf_table(ocr_table))
                else:
                    transactions.extend(parse_pdf_table(data))
        except Exception as e:
            logger.debug(f"Lattice failed for {pdf_path}: {e}")

    # 3) camelot stream — fallback for unrulled tables
    if not transactions:
        try:
            tables = camelot.read_pdf(str(pdf_path), pages="all", flavor="stream")
            engines_attempted.append("stream")
            # Fix 2: Try ALL detected tables, not just the first one
            for table in tables:
                txs = parse_pdf_table(table.data)
                if txs:
                    transactions.extend(txs)
                    break  # Found transactions, stop scanning
        except Exception as e:
            logger.debug(f"Stream failed for {pdf_path}: {e}")

    # 4) pdftotext — handles encrypted PDFs where camelot/pdfplumber return nothing
    if not transactions:
        try:
            pdftext_tables = extract_tables_with_pdftotext(pdf_path)
            engines_attempted.append("pdftotext")
            for table in pdftext_tables:
                txs = parse_pdf_table(table)
                if txs:
                    transactions.extend(txs)
                    break
        except Exception as e:
            logger.debug(f"pdftotext failed for {pdf_path}: {e}")

    # 5) Docling — OCR fallback for SCANNED IMAGE PDFs (no text layer).
    # Benchmark winner over Marker (MIT license, no Cyrillic-E bug, better
    # accuracy 75% vs 58%). Slow (13-300s) but only runs when all text-layer
    # parsers return nothing.
    if not transactions:
        engines_attempted.append("docling")
        try:
            docling_tables = extract_tables_with_docling(pdf_path)
            for table in docling_tables:
                txs = parse_pdf_table(table)
                if txs:
                    transactions.extend(txs)
                    break
        except Exception as e:
            logger.debug(f"Docling failed for {pdf_path}: {e}")

    # 6) pytesseract — last resort, kept for backward compatibility when
    # docling/uvx is unavailable.
    if not transactions:
        try:
            ocr_tables = extract_tables_with_ocr(pdf_path)
            engines_attempted.append("ocr")
            for table in ocr_tables:
                transactions.extend(parse_pdf_table(table))
        except Exception as e:
            logger.debug(f"OCR failed for {pdf_path}: {e}")

    return pdf_path, transactions, engines_attempted


class HouseTransactionSource(TransactionSource):
    def __init__(self, settings: Settings, read_only: bool = False):
        self.settings = settings
        self.data_dir = Path(settings.data.data_dir)
        self.metadata_url_template = settings.sources.house_metadata_url
        self.pdf_url_template = settings.sources.house_pdf_url
        self.parallel_workers = settings.data.get_workers()
        self.session = requests_cache.CachedSession(
            cache_name=str(self.data_dir / "http_cache"),
            backend="sqlite",
            expire_after=3600,
        )
        self.db = Database(self.data_dir / "congress.duckdb", read_only=read_only)

    def close(self) -> None:
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        return False

    def get_transactions(self, year: int) -> pd.DataFrame:
        if not self.db.transactions_exist(year):
            raise DataSourceError(
                f"No cached data found for {year}. Run 'ptr-alpha parse --year {year}' first."
            )

        df = self.db.get_transactions(year)
        logger.info(f"Loaded {len(df)} cached transactions for {year}")
        return df

    def fetch_metadata(self, year: int, refresh: bool = False) -> pd.DataFrame:
        if not refresh and self.db.metadata_exists(year):
            logger.info(f"Loading cached metadata for {year}")
            return self.db.get_metadata(year)

        if refresh:
            self.db.clear_metadata(year)

        metadata_url = self.metadata_url_template.format(year=year)
        try:
            logger.info(f"Downloading metadata for {year} from House disclosures")
            response = self.session.get(metadata_url, timeout=30)
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

        except Exception as e:
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
        pdf_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Processing {len(ptrs)} PTR filings for {year}")

        connector = aiohttp.TCPConnector(limit=10)
        async with aiohttp.ClientSession(connector=connector) as session:
            async with asyncio.TaskGroup() as tg:
                doc_ids = ptrs["DocID"].values
                pdf_paths = [pdf_dir / f"{doc_id}.pdf" for doc_id in doc_ids]
                urls = [self.pdf_url_template.format(year=year, doc_id=doc_id) for doc_id in doc_ids]
                tasks = [
                    tg.create_task(self._download_pdf_async(session, doc_id, pdf_path, url))
                    for doc_id, pdf_path, url in zip(doc_ids, pdf_paths, urls)
                ]

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

    def fetch_and_cache_pdfs(self, year: int) -> None:
        return asyncio.run(self._fetch_and_cache_pdfs_async(year))

    def parse_cached_pdfs(self, year: int) -> None:
        metadata = self.fetch_metadata(year)
        ptrs = metadata[metadata["FilingType"] == FilingType.PTR.value]
        pdf_dir = self.data_dir / str(year) / "pdfs"

        if not pdf_dir.exists():
            raise DataSourceError(f"PDF directory not found: {pdf_dir}")

        logger.info(f"Parsing {len(ptrs)} PDFs for {year}")

        doc_ids = ptrs["DocID"].values
        all_pdf_paths = [pdf_dir / f"{doc_id}.pdf" for doc_id in doc_ids]
        exists_mask = [p.exists() for p in all_pdf_paths]
        pdf_paths = [p for p, e in zip(all_pdf_paths, exists_mask) if e]
        existing_docs = ptrs[exists_mask]
        member_lookup = dict(zip(
            existing_docs["DocID"].values,
            [{"First": f, "Last": last, "FilingDate": d} for f, last, d in zip(
                existing_docs["First"].values, existing_docs["Last"].values, existing_docs["FilingDate"].values
            )]
        ))

        if not pdf_paths:
            raise DataSourceError(f"No PDF files found in {pdf_dir}")

        with Pool(self.parallel_workers) as pool:
            results = pool.map(_parse_pdf_worker, pdf_paths)

        pdf_transactions = {}
        for pdf_path, transactions, engines_attempted in results:
            doc_id = pdf_path.stem
            pdf_transactions[pdf_path] = transactions
            status = "success" if transactions else "zero_rows"
            self.db.upsert_parse_run(
                doc_id=doc_id,
                year=year,
                parser_version="v2",
                status=status,
                engines_attempted=",".join(engines_attempted),
                raw_row_count=0,
                transaction_count=len(transactions),
            )

        df = consolidate_transactions(pdf_transactions, member_lookup)

        if df.empty:
            raise ParsingError("No transactions found after parsing all PDFs")

        # Delete old rows for each doc_id before inserting new ones
        for doc_id in df["doc_id"].unique():
            self.db.delete_transactions_for_doc(doc_id)

        self.db.upsert_transactions(df)
        logger.info(f"Saved {len(df)} transactions to database")


class YFinancePriceSource(PriceSource):
    def __init__(self, settings: Settings, read_only: bool = False):
        self.settings = settings
        self.data_dir = Path(settings.data.data_dir)
        self.db = Database(self.data_dir / "congress.duckdb", read_only=read_only)
        self.resolver = TickerResolver()

    def close(self) -> None:
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        return False

    def get_prices(self, tickers: list[str], start: date, end: date) -> pd.DataFrame:
        if len(tickers) == 0:
            raise DataSourceError("No tickers provided for price fetching")

        all_tickers = sorted(list(set(tickers) | {"SPY"}))

        # Resolve raw tickers to yfinance-compatible symbols
        resolutions = self.resolver.resolve_batch(all_tickers)
        # Mapping: raw_ticker -> price_symbol
        raw_to_yf: dict[str, str] = {r.raw_ticker: r.price_symbol for r in resolutions.values()}
        # Reverse mapping: yfinance symbol -> raw_ticker (for renaming back)
        yf_to_raw: dict[str, str] = {}
        for raw, sym in raw_to_yf.items():
            if sym not in yf_to_raw:
                yf_to_raw[sym] = raw

        # Log resolutions
        for r in resolutions.values():
            if r.status != "valid":
                logger.info(f"Ticker resolution: {r.notes}")

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
        # Resolve the tickers we need to fetch
        fetch_resolved = sorted(set(raw_to_yf.get(t, t) for t in fetch_tickers))
        logger.info(
            f"Fetching price data for {len(fetch_resolved)} tickers using yfinance"
        )

        try:
            data = yf.download(
                fetch_resolved,
                start=start,
                end=end,
                progress=False,
                threads=True,
                auto_adjust=True,
            )
        except Exception as e:
            logger.warning(f"yfinance request failed ({e}), falling back to cached data")
            data = pd.DataFrame()

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
            if len(fetch_resolved) > 1
            else data["Close"].to_frame(fetch_resolved[0])
        )
        new_prices = new_prices.dropna(axis=1, how="all")

        # Rename columns from yfinance symbols back to raw tickers for storage
        # Build reverse mapping: yfinance symbol -> list of raw tickers
        yf_to_raws: dict[str, list[str]] = {}
        for raw, sym in raw_to_yf.items():
            yf_to_raws.setdefault(sym, []).append(raw)
        # Only rename when yfinance symbol maps to exactly one raw ticker
        rename_map = {
            sym: raws[0]
            for sym, raws in yf_to_raws.items()
            if sym in new_prices.columns and len(raws) == 1
        }
        new_prices = new_prices.rename(columns=rename_map)

        if self.db.is_read_only:
            logger.info(
                f"Read-only mode: merging {len(new_prices.columns)} fetched tickers with cache"
            )
            # Drop duplicate columns (keep cached, fill gaps from new)
            merged = pd.concat([cached_prices, new_prices], axis=1)
            merged = merged.loc[:, ~merged.columns.duplicated(keep="first")]
            merged = merged[~merged.index.duplicated(keep="last")]
            prices = merged
        else:
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
