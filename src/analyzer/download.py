"""House disclosure download and PDF parse driver."""

import asyncio
import logging
import os
import zipfile
from datetime import datetime
from io import BytesIO
from multiprocessing import Pool
from pathlib import Path

import aiohttp
import pandas as pd
import requests
import requests_cache

from analyzer.database import Database
from analyzer.exceptions import DataSourceError, ParsingError
from analyzer.interfaces import TransactionSource
from analyzer.settings import Settings
from analyzer.models import DownloadResult, DownloadStatus, FilingType
from analyzer.parsing import (
    consolidate_transactions,
    normalize_house_metadata,
)
from analyzer.parser_cascade import _is_valid_pdf, _parse_pdf_worker

logger = logging.getLogger(__name__)


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
        self.session.close()
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

        metadata_url = self.metadata_url_template.format(year=year)
        try:
            return self._download_and_upsert_metadata(year, metadata_url, replace=refresh)
        except requests.RequestException as e:
            raise DataSourceError(f"Failed to fetch metadata for {year}: {e}")
        except Exception as e:
            raise DataSourceError(f"Failed to fetch metadata for {year}: {e}")

    def _download_and_upsert_metadata(
        self, year: int, metadata_url: str, *, replace: bool = False,
    ) -> pd.DataFrame:
        logger.info(f"Downloading metadata for {year} from House disclosures")
        response = self.session.get(metadata_url, timeout=30)
        if response.status_code != 200:
            raise DataSourceError(
                f"Failed to fetch metadata for {year}, status: {response.status_code}"
            )

        content = _read_first_text_from_zip(response.content, year)
        df = normalize_house_metadata(content)
        if "FilingType" not in df.columns:
            raise ParsingError("Missing required metadata column: FilingType")
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

        if replace:
            self.db.replace_metadata(year, df)
        else:
            self.db.upsert_metadata(df)
        logger.info(f"Cached metadata for {year}: {len(df)} records")
        return self.db.get_metadata(year)

    async def _download_pdf_async(self, session, doc_id, pdf_path, url):
        if _is_valid_pdf(pdf_path):
            return DownloadResult(doc_id=doc_id, status=DownloadStatus.SKIPPED)

        tmp_path = pdf_path.with_suffix(".pdf.tmp")
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
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                logger.warning("Unable to remove temporary PDF %s", tmp_path)

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
        failures = [r for r in results if r.status in (DownloadStatus.FAILED, DownloadStatus.ERROR)]
        if failures:
            sample = ", ".join(str(r.doc_id) for r in failures[:10])
            raise DataSourceError(
                f"Failed to download {len(failures)} of {len(results)} House PDFs; doc IDs: {sample}"
            )

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
        zero_row_doc_ids: list[str] = []
        parse_runs: list[dict] = []
        for pdf_path, transactions, engines_attempted in results:
            doc_id = pdf_path.stem
            pdf_transactions[pdf_path] = transactions
            status = "success" if transactions else "zero_rows"
            parse_runs.append(dict(
                doc_id=doc_id,
                year=year,
                parser_version="v3",
                status=status,
                engines_attempted=",".join(engines_attempted),
                raw_row_count=0,
                transaction_count=len(transactions),
            ))
            if not transactions:
                zero_row_doc_ids.append(doc_id)

        stale_docs = self.db.count_transactions_for_docs(zero_row_doc_ids)
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

        df = preserve_existing_fields(df, self.db)

        self.db.replace_transactions_for_docs(df, source="house_pdf")
        # Do not publish successful parse audit records until the corresponding
        # transaction replacement has succeeded.
        for parse_run in parse_runs:
            self.db.upsert_parse_run(**parse_run)
        logger.info(f"Saved {len(df)} transactions to database")


# ── Helpers ──

def _is_blank(value) -> bool:
    """True when a field carries no usable value (None / NaN / empty string)."""
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def _identity_key(member, transaction_date, transaction_type):
    """Normalize the identity tuple so NaN/None match each other on lookup.

    Date-like values (``datetime.date``, ``datetime.datetime``,
    ``pandas.Timestamp``) are coerced to a plain ``date`` so the key built from a
    freshly consolidated ``df`` matches one built from DB rows — DuckDB returns
    DATE columns as ``pd.Timestamp``, while parsed rows use ``datetime.date``.
    """
    def _norm(v):
        if v is None:
            return None
        if isinstance(v, float) and pd.isna(v):
            return None
        if isinstance(v, str) and v.strip() == "":
            return None
        if not isinstance(v, str) and hasattr(v, "date"):
            try:
                return v.date()
            except Exception:
                return v
        return v

    return (_norm(member), _norm(transaction_date), _norm(transaction_type))


def preserve_existing_fields(df: pd.DataFrame, db) -> pd.DataFrame:
    """Carry forward previously-resolved fields so re-parsing MERGEs, not clobbers.

    Re-parsing deletes and re-inserts a document's transactions. If the fresh
    parse yields a null/empty ``ticker`` (or ``amount_raw``) for a transaction
    that a previous parse had correctly resolved, the good data would be
    overwritten with NULL. This merges the previously-resolved values back into
    ``df``, matched per ``doc_id`` by ``(member, transaction_date,
    transaction_type)`` — the transaction identity. Identity fields themselves
    are never modified.

    Behaviour:
    * new ticker null/empty/blank     -> keep existing ticker if present
    * new ticker present and valid    -> keep new (never downgrade)
    * new amount_raw null/empty       -> keep existing amount_raw if present
    * multiple existing rows for identity -> value carried only when all
      existing rows for that identity agree on a single non-blank value;
      disagreement (or no non-blank value) leaves the field blank (safe, never
      guess)
    * doc has no existing rows         -> no-op
    """
    if df.empty:
        return df
    df = df.copy()

    for doc_id in df["doc_id"].unique():
        existing = db.get_transactions_for_doc(doc_id)
        if existing is None or existing.empty:
            continue

        lookup: dict[tuple, dict] = {}
        for _, er in existing.iterrows():
            key = _identity_key(er.get("member"), er.get("transaction_date"), er.get("transaction_type"))
            slot = lookup.setdefault(key, {"tickers": set(), "amounts": set()})
            if not _is_blank(er.get("ticker")):
                slot["tickers"].add(er["ticker"])
            if not _is_blank(er.get("amount_raw")):
                slot["amounts"].add(er["amount_raw"])

        for idx in df.index[df["doc_id"] == doc_id]:
            key = _identity_key(df.at[idx, "member"], df.at[idx, "transaction_date"], df.at[idx, "transaction_type"])
            slot = lookup.get(key)
            if slot is None:
                continue
            if _is_blank(df.at[idx, "ticker"]) and len(slot["tickers"]) == 1:
                df.at[idx, "ticker"] = next(iter(slot["tickers"]))
            if _is_blank(df.at[idx, "amount_raw"]) and len(slot["amounts"]) == 1:
                df.at[idx, "amount_raw"] = next(iter(slot["amounts"]))

    return df


def _read_first_text_from_zip(zip_bytes: bytes, year: int) -> str:
    """Open the metadata ZIP and return its unambiguous metadata text member."""
    with zipfile.ZipFile(BytesIO(zip_bytes)) as z:
        text_files = [f for f in z.namelist() if f.lower().endswith(".txt") and not f.endswith("/")]
        if not text_files:
            raise ParsingError(f"No text files found in metadata ZIP for {year}")
        decoded: dict[str, str] = {}
        for name in text_files:
            with z.open(name) as f:
                raw = f.read()
            try:
                decoded[name] = raw.decode("utf-8-sig")
            except UnicodeDecodeError:
                # Historical names can contain Windows-1252 punctuation. This
                # fallback preserves every byte instead of silently dropping it.
                logger.warning("Metadata text file %s is Windows-1252, not UTF-8", name)
                decoded[name] = raw.decode("cp1252")

        required = {"DocID", "First", "Last", "FilingDate"}
        candidates = []
        for name, content in decoded.items():
            lines = content.splitlines()
            if not lines:
                continue
            header = {c.strip().lstrip("\ufeff") for c in lines[0].split("\t")}
            if required.issubset(header):
                candidates.append(name)
        if not candidates:
            raise ParsingError(f"No House metadata table found in ZIP for {year}")
        preferred = [name for name in candidates if str(year) in Path(name).stem]
        if len(preferred) == 1:
            return decoded[preferred[0]]
        if len(candidates) != 1:
            raise ParsingError(
                f"Ambiguous metadata ZIP for {year}: {', '.join(candidates[:10])}"
            )
        return decoded[candidates[0]]


def _filter_existing_pdfs(ptrs: pd.DataFrame, pdf_dir: Path) -> tuple[list[Path], pd.DataFrame]:
    """Return (existing PDF paths, the subset of `ptrs` whose PDFs are on disk)."""
    doc_ids = ptrs["DocID"].values
    all_pdf_paths = [pdf_dir / f"{doc_id}.pdf" for doc_id in doc_ids]
    exists_mask = [_is_valid_pdf(p) for p in all_pdf_paths]
    pdf_paths = [p for p, e in zip(all_pdf_paths, exists_mask) if e]
    existing_docs = ptrs[exists_mask]
    return pdf_paths, existing_docs


def _build_member_lookup(existing_docs: pd.DataFrame) -> dict:
    """Map doc_id -> {First, Last, FilingDate} for downstream transaction join."""
    identity_columns = ["DocID", "First", "Last", "FilingDate"]
    distinct = existing_docs[identity_columns].drop_duplicates()
    duplicate_ids = distinct.loc[
        distinct["DocID"].astype(str).duplicated(keep=False), "DocID"
    ].astype(str).unique()
    if len(duplicate_ids):
        raise ParsingError(
            "Duplicate metadata DocID(s) would make member attribution ambiguous: "
            + ", ".join(duplicate_ids[:10])
        )
    existing_docs = distinct.drop_duplicates(subset=["DocID"])
    return dict(zip(
        existing_docs["DocID"].values,
        [{"First": f, "Last": last, "FilingDate": d} for f, last, d in zip(
            existing_docs["First"].values,
            existing_docs["Last"].values,
            existing_docs["FilingDate"].values,
        )],
    ))
