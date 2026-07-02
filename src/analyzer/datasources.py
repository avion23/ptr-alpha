import asyncio
import logging
import os
import re
import time
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

_VALID_TICKER_RE = re.compile(r"^[A-Z]{1,5}([.-][A-Z]{1,2})?$")


# ── Per-PDF worker: cascades 5 PDF engines until transactions are found ─

def _is_valid_pdf(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    with open(path, "rb") as f:
        return f.read(5) == b"%PDF-"


def _result_quality(txs: list[dict]) -> float:
    if not txs:
        return 0.0
    valid = sum(
        1 for tx in txs
        if tx.get("transaction_date") and tx.get("amount_midpoint")
    )
    return valid / len(txs)

def _parse_pdf_worker(pdf_path: Path) -> tuple[Path, list[dict], list[str]]:
    """Cascade through pdfplumber → camelot lattice → camelot stream →
    pdftotext → Docling OCR → tesseract OCR. Each engine is tried only when
    all text-layer parsers returned 0 transactions; this minimises OCR work
    while allowing better text-layer engines to beat low-quality results."""
    skip_docling = os.environ.get("PTR_SKIP_DOCLING") == "1"
    engines_attempted: list[str] = []
    best_transactions: list[dict] = []
    best_engine = ""
    best_quality = 0.0

    for engine_fn, engine_name in [
        (_try_pdfplumber, "pdfplumber"),
        (_try_camelot_lattice, "lattice"),
        (_try_camelot_stream, "stream"),
        (_try_pdftotext, "pdftotext"),
    ]:
        transactions = engine_fn(pdf_path)
        engines_attempted.append(engine_name)
        if transactions:
            quality = _result_quality(transactions)
            if quality >= 0.7:
                engines_attempted.append(f"won:{engine_name}")
                return pdf_path, transactions, engines_attempted
            if (quality, len(transactions)) > (best_quality, len(best_transactions)):
                best_transactions = transactions
                best_engine = engine_name
                best_quality = quality

    if best_transactions:
        engines_attempted.append(f"won:{best_engine}")
        return pdf_path, best_transactions, engines_attempted

    transactions: list[dict] = []

    # Docling + tesseract: only run if all text-layer engines returned 0
    if not skip_docling:
        engines_attempted.append("docling")
        transactions = _try_docling(pdf_path)
        if transactions:
            engines_attempted.append("won:docling")
            return pdf_path, transactions, engines_attempted

    engines_attempted.append("ocr")
    transactions = _try_tesseract(pdf_path)
    if transactions:
        engines_attempted.append("won:ocr")

    return pdf_path, transactions, engines_attempted


def _try_pdfplumber(pdf_path: Path) -> list[dict]:
    """Benchmark winner for text-based PDFs (0.075s avg). Handles encrypted
    PDFs natively; returns 0 on scanned images."""
    try:
        pp_tables = extract_tables_with_pdfplumber(pdf_path)
    except Exception as e:
        logger.debug(f"pdfplumber failed for {pdf_path}: {e}")
        return []
    if not pp_tables:
        return []
    txs: list[dict] = []
    for table in pp_tables:
        txs.extend(parse_pdf_table(table))
    return txs


def _try_camelot_lattice(pdf_path: Path) -> list[dict]:
    """Previous primary parser; keeps PDFs with ruling lines. Lattice
    sometimes collapses to 1-column when null bytes are present — in that
    case we re-parse each cell as OCR text."""
    try:
        tables = camelot.read_pdf(str(pdf_path), pages="all", flavor="lattice")
    except Exception as e:
        logger.debug(f"Lattice failed for {pdf_path}: {e}")
        return []
    txs: list[dict] = []
    for table in tables:
        data = table.data
        if data and len(data[0]) == 1:
            for row in data:
                cell_text = row[0] if row else ""
                if cell_text:
                    ocr_rows = _parse_ocr_text_to_rows(cell_text)
                    if ocr_rows:
                        ocr_table = [
                            ['Asset Name', 'Transaction Type', 'Transaction Date', 'Amount']
                        ] + ocr_rows
                        txs.extend(parse_pdf_table(ocr_table))
        else:
            txs.extend(parse_pdf_table(data))
    return txs


def _try_camelot_stream(pdf_path: Path) -> list[dict]:
    """Fallback for unrulled tables. Try ALL detected tables and stop at
    the first one that yields transactions (Fix 2: don't just scan table[0])."""
    try:
        tables = camelot.read_pdf(str(pdf_path), pages="all", flavor="stream")
    except Exception as e:
        logger.debug(f"Stream failed for {pdf_path}: {e}")
        return []
    for table in tables:
        txs = parse_pdf_table(table.data)
        if txs:
            return txs
    return []


def _try_pdftotext(pdf_path: Path) -> list[dict]:
    """Handles encrypted PDFs where camelot/pdfplumber return nothing."""
    try:
        pdftext_tables = extract_tables_with_pdftotext(pdf_path)
    except Exception as e:
        logger.debug(f"pdftotext failed for {pdf_path}: {e}")
        return []
    for table in pdftext_tables:
        txs = parse_pdf_table(table)
        if txs:
            return txs
    return []


def _try_docling(pdf_path: Path) -> list[dict]:
    """OCR fallback for SCANNED IMAGE PDFs (no text layer). Slow (13-300s)
    but only runs when all text-layer parsers return nothing.

    Skip when PTR_SKIP_DOCLING=1 (set during the bulk first pass; stragglers
    are re-parsed with Docling in a second pass using reduced worker count
    to avoid OOM — each Docling proc uses ~2GB).
    """
    try:
        docling_tables = extract_tables_with_docling(pdf_path)
    except Exception as e:
        logger.debug(f"Docling failed for {pdf_path}: {e}")
        return []
    for table in docling_tables:
        txs = parse_pdf_table(table)
        if txs:
            return txs
    return []


def _try_tesseract(pdf_path: Path) -> list[dict]:
    """Last-resort OCR. Kept for backward compatibility when docling/uvx
    is unavailable."""
    try:
        ocr_tables = extract_tables_with_ocr(pdf_path)
    except Exception as e:
        logger.debug(f"OCR failed for {pdf_path}: {e}")
        return []
    txs: list[dict] = []
    for table in ocr_tables:
        txs.extend(parse_pdf_table(table))
    return txs


# ── HouseTransactionSource: House disclosure download + parse driver ──

class HouseTransactionSource(TransactionSource):
    def __init__(self, settings: Settings, read_only: bool = False, db: Database | None = None):
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
        self._owns_db = db is None
        self.db = db if db is not None else Database(self.data_dir / "congress.duckdb", read_only=read_only)

    def close(self) -> None:
        if self._owns_db:
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
            return self._download_and_upsert_metadata(year, metadata_url)
        except requests.RequestException as e:
            raise DataSourceError(f"Failed to fetch metadata for {year}: {e}")
        except Exception as e:
            raise DataSourceError(f"Failed to fetch metadata for {year}: {e}")

    def _download_and_upsert_metadata(self, year: int, metadata_url: str) -> pd.DataFrame:
        logger.info(f"Downloading metadata for {year} from House disclosures")
        response = self.session.get(metadata_url, timeout=30)
        if response.status_code != 200:
            raise DataSourceError(
                f"Failed to fetch metadata for {year}, status: {response.status_code}"
            )

        content = _read_first_text_from_zip(response.content, year)
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

    async def _download_pdf_async(self, session, doc_id, pdf_path, url):
        if _is_valid_pdf(pdf_path):
            return DownloadResult(doc_id=doc_id, status=DownloadStatus.SKIPPED)

        try:
            async with session.get(
                url, timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    content = await response.read()
                    if not content.startswith(b"%PDF-"):
                        return DownloadResult(
                            doc_id=doc_id,
                            status=DownloadStatus.FAILED,
                            status_code=response.status,
                            error_message="not a PDF (got HTML error page?)",
                        )
                    tmp_path = pdf_path.with_suffix(".pdf.tmp")
                    with open(tmp_path, "wb") as f:
                        f.write(content)
                    os.replace(tmp_path, pdf_path)
                    return DownloadResult(doc_id=doc_id, status=DownloadStatus.SUCCESS)
                return DownloadResult(
                    doc_id=doc_id,
                    status=DownloadStatus.FAILED,
                    status_code=response.status,
                    error_message=f"HTTP {response.status}",
                )
        except (aiohttp.ClientError, OSError, asyncio.TimeoutError) as e:
            return DownloadResult(
                doc_id=doc_id, status=DownloadStatus.ERROR, error_message=str(e),
            )

    async def _fetch_and_cache_pdfs_async(self, year):
        metadata = self.fetch_metadata(year)
        ptrs = metadata[metadata["FilingType"] == FilingType.PTR.value]
        pdf_dir = self.data_dir / str(year) / "pdfs"
        pdf_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Processing {len(ptrs)} PTR filings for {year}")

        doc_ids = ptrs["DocID"].values
        pdf_paths = [pdf_dir / f"{doc_id}.pdf" for doc_id in doc_ids]
        urls = [
            self.pdf_url_template.format(year=year, doc_id=doc_id)
            for doc_id in doc_ids
        ]

        connector = aiohttp.TCPConnector(limit=10)
        async with aiohttp.ClientSession(connector=connector) as session:
            async with asyncio.TaskGroup() as tg:
                tasks = [
                    tg.create_task(self._download_pdf_async(session, doc_id, pdf_path, url))
                    for doc_id, pdf_path, url in zip(doc_ids, pdf_paths, urls)
                ]

            results = [task.result() for task in tasks]

        self._log_download_summary(results)

    def _log_download_summary(self, results: list[DownloadResult]) -> None:
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

        pdf_paths, existing_docs = _filter_existing_pdfs(ptrs, pdf_dir)
        if not pdf_paths:
            raise DataSourceError(f"No PDF files found in {pdf_dir}")

        member_lookup = _build_member_lookup(existing_docs)
        logger.info(f"Parsing {len(ptrs)} PDFs for {year}")

        with Pool(self.parallel_workers) as pool:
            results = pool.map(_parse_pdf_worker, pdf_paths)

        self._save_parse_results(year, results, member_lookup)

    def _save_parse_results(
        self, year: int, results: list, member_lookup: dict,
    ) -> None:
        pdf_transactions: dict = {}
        stale_docs: dict[str, int] = {}
        for pdf_path, transactions, engines_attempted in results:
            doc_id = pdf_path.stem
            pdf_transactions[pdf_path] = transactions
            status = "success" if transactions else "zero_rows"
            self.db.upsert_parse_run(
                doc_id=doc_id,
                year=year,
                parser_version="v3",
                status=status,
                engines_attempted=",".join(engines_attempted),
                raw_row_count=0,
                transaction_count=len(transactions),
            )
            if not transactions:
                existing_rows = self.db.count_transactions_for_doc(doc_id)
                if existing_rows:
                    stale_docs[doc_id] = existing_rows

        if stale_docs:
            doc_ids = list(stale_docs)[:10]
            logger.warning(
                "%d docs parsed to zero rows but retain %d existing DB rows (stale?): %s",
                len(stale_docs),
                sum(stale_docs.values()),
                ", ".join(doc_ids),
            )

        df = consolidate_transactions(pdf_transactions, member_lookup)
        if df.empty:
            raise ParsingError("No transactions found after parsing all PDFs")

        for doc_id in df["doc_id"].unique():
            self.db.delete_transactions_for_doc(doc_id)

        self.db.upsert_transactions(df, source="house_pdf")
        logger.info(f"Saved {len(df)} transactions to database")


def _read_first_text_from_zip(zip_bytes: bytes, year: int) -> str:
    """Open the metadata ZIP and return the first .txt file's content."""
    with zipfile.ZipFile(BytesIO(zip_bytes)) as z:
        text_files = [f for f in z.namelist() if f.endswith(".txt")]
        if not text_files:
            raise ParsingError(f"No text files found in metadata ZIP for {year}")
        with z.open(text_files[0]) as f:
            return f.read().decode("utf-8", errors="ignore")


def _filter_existing_pdfs(ptrs: pd.DataFrame, pdf_dir: Path) -> tuple[list[Path], pd.DataFrame]:
    """Return (existing PDF paths, the subset of `ptrs` whose PDFs are on disk)."""
    doc_ids = ptrs["DocID"].values
    all_pdf_paths = [pdf_dir / f"{doc_id}.pdf" for doc_id in doc_ids]
    exists_mask = [p.exists() for p in all_pdf_paths]
    pdf_paths = [p for p, e in zip(all_pdf_paths, exists_mask) if e]
    existing_docs = ptrs[exists_mask]
    return pdf_paths, existing_docs


def _build_member_lookup(existing_docs: pd.DataFrame) -> dict:
    """Map doc_id -> {First, Last, FilingDate} for downstream transaction join."""
    return dict(zip(
        existing_docs["DocID"].values,
        [{"First": f, "Last": last, "FilingDate": d} for f, last, d in zip(
            existing_docs["First"].values,
            existing_docs["Last"].values,
            existing_docs["FilingDate"].values,
        )],
    ))


# ── YFinancePriceSource: yfinance-backed price fetcher with cache merge ──

class YFinancePriceSource(PriceSource):
    def __init__(self, settings: Settings, read_only: bool = False, db: Database | None = None):
        self.settings = settings
        self.data_dir = Path(settings.data.data_dir)
        self._owns_db = db is None
        self.db = db if db is not None else Database(self.data_dir / "congress.duckdb", read_only=read_only)
        self.resolver = TickerResolver()

    def close(self) -> None:
        if self._owns_db:
            self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        return False

    def get_prices(self, tickers: list[str], start: date, end: date) -> pd.DataFrame:
        if len(tickers) == 0:
            raise DataSourceError("No tickers provided for price fetching")

        clean_tickers = _clean_tickers(tickers)
        all_tickers = sorted(
            list(set(t for t in clean_tickers if _VALID_TICKER_RE.match(str(t))) | {"SPY"})
        )

        raw_to_yf, yf_to_raw = _resolve_tickers(all_tickers)
        cached_prices = self.db.get_prices(all_tickers, start, end)
        if not cached_prices.empty:
            logger.info(
                f"Loaded cached prices: {len(cached_prices.columns)} tickers, "
                f"{len(cached_prices)} dates"
            )

        missing_tickers, missing_dates = self.db.get_missing_price_data(
            all_tickers, start, end,
        )

        if not missing_tickers and not missing_dates:
            logger.info(f"Using fully cached prices for {len(all_tickers)} tickers")
            available_tickers = [t for t in all_tickers if t in cached_prices.columns]
            return cached_prices[available_tickers].dropna(axis=1, how="all")

        return self._fetch_and_merge_prices(
            all_tickers, raw_to_yf, yf_to_raw, cached_prices, start, end,
            missing_tickers,
        )

    def _fetch_and_merge_prices(
        self, all_tickers, raw_to_yf, yf_to_raw, cached_prices, start, end,
        missing_tickers,
    ) -> pd.DataFrame:
        """Fetch missing data from yfinance and merge with the cache."""
        fetch_tickers = missing_tickers if missing_tickers else all_tickers
        fetch_resolved = sorted(set(raw_to_yf.get(t, t) for t in fetch_tickers))

        logger.info(
            f"Fetching price data for {len(fetch_resolved)} tickers using yfinance"
        )

        data = self._download_yfinance(fetch_resolved, start, end)
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
        new_prices = self._rename_yf_columns(new_prices, raw_to_yf)

        if self.db.is_read_only:
            logger.info(
                f"Read-only mode: merging {len(new_prices.columns)} fetched tickers with cache"
            )
            merged = pd.concat([cached_prices, new_prices], axis=1)
            merged = merged.loc[:, ~merged.columns.duplicated(keep="first")]
            prices = merged[~merged.index.duplicated(keep="last")]
        else:
            self.db.upsert_prices(new_prices)
            logger.info(f"Cached {len(new_prices.columns)} tickers to database")
            prices = self.db.get_prices(all_tickers, start, end)

        return _validate_and_log_prices(prices, all_tickers)

    def _download_yfinance(self, fetch_resolved: list[str], start, end) -> pd.DataFrame:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return yf.download(
                    fetch_resolved,
                    start=start,
                    end=end,
                    progress=False,
                    threads=True,
                    auto_adjust=True,
                )
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = 2 ** (attempt + 1)
                    logger.warning(
                        f"yfinance request failed (attempt {attempt + 1}/{max_retries}: {e}), "
                        f"retrying in {delay}s"
                    )
                    time.sleep(delay)
                else:
                    logger.warning(
                        f"yfinance request failed after {max_retries} attempts ({e}), "
                        "falling back to cached data"
                    )
                    return pd.DataFrame()
        return pd.DataFrame()

    def _rename_yf_columns(self, new_prices: pd.DataFrame, raw_to_yf: dict) -> pd.DataFrame:
        """Rename yf-symbol columns back to their raw tickers so downstream
        consumers see consistent identifiers across sources."""
        yf_to_raws: dict[str, list[str]] = {}
        for raw, sym in raw_to_yf.items():
            yf_to_raws.setdefault(sym, []).append(raw)
        # Only rename when a yf symbol maps to exactly one raw ticker (avoid collision).
        rename_map = {
            sym: raws[0]
            for sym, raws in yf_to_raws.items()
            if sym in new_prices.columns and len(raws) == 1
        }
        return new_prices.rename(columns=rename_map)


def _clean_tickers(tickers: list[str]) -> list[str]:
    """Filter out NaN/None/empty/garbage tickers from the input list."""
    return [t for t in tickers if t and str(t).strip() and str(t) != "nan"]


def _resolve_tickers(all_tickers: list[str]) -> tuple[dict, dict]:
    """Build (raw -> yf, yf -> raw) mapping via TickerResolver."""
    resolver = TickerResolver()
    resolutions = resolver.resolve_batch(all_tickers)
    raw_to_yf = {r.raw_ticker: r.price_symbol for r in resolutions.values()}
    yf_to_raw: dict[str, str] = {}
    for raw, sym in raw_to_yf.items():
        if sym not in yf_to_raw:
            yf_to_raw[sym] = raw
    for r in resolutions.values():
        if r.status != "valid":
            logger.info(f"Ticker resolution: {r.notes}")
    return raw_to_yf, yf_to_raw


def _validate_and_log_prices(prices: pd.DataFrame, all_tickers: list[str]) -> pd.DataFrame:
    """Fail loudly when too many tickers couldn't be fetched (>25%)."""
    failed_tickers = sorted(set(all_tickers) - set(prices.columns))
    success_count = len([t for t in all_tickers if t in prices.columns])
    success_rate = success_count / len(all_tickers)

    if success_rate < 0.75:
        raise DataSourceError(
            f"Price fetch failure rate too high: {(1 - success_rate) * 100:.1f}% failed "
            f"({len(failed_tickers)}/{len(all_tickers)}). Analysis would be unreliable."
        )
    if failed_tickers:
        logger.warning(
            f"Failed to fetch price data for {len(failed_tickers)} tickers: "
            f"{', '.join(failed_tickers[:10])}"
            f"{'...' if len(failed_tickers) > 10 else ''}"
        )

    logger.info(
        f"Successfully fetched prices for {success_count}/{len(all_tickers)} "
        f"tickers ({success_rate * 100:.1f}% success)"
    )
    available_tickers = [t for t in all_tickers if t in prices.columns]
    return prices[available_tickers].dropna(axis=1, how="all")
