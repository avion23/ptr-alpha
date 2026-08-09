"""House disclosure download and PDF parse driver."""

import asyncio
import hashlib
import logging
import os
import shutil
import uuid
import zipfile
from dataclasses import dataclass
from datetime import UTC, date, datetime
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
_PARSE_VERSION = "v4-deterministic"


@dataclass(frozen=True, slots=True)
class HouseFetchSummary:
    archive_year: int
    metadata_count: int
    ptr_count: int
    valid_pdf_count: int
    downloaded_count: int
    skipped_count: int
    orphan_pdf_count: int
    removed_doc_count: int = 0
    quarantined_pdf_count: int = 0
    generation_id: str | None = None
    generation_status: str = "incomplete"


@dataclass(frozen=True, slots=True)
class HousePdfAcquisition:
    doc_id: str
    status: DownloadStatus
    status_code: int = 0
    error_message: str = ""
    artifact_sha256: str | None = None
    etag: str | None = None
    last_modified: str | None = None
    content_length: int | None = None


# ── HouseTransactionSource: House disclosure download + parse driver ──


class HouseTransactionSource(TransactionSource):
    def __init__(
        self, settings: Settings, read_only: bool = False, db: Database | None = None
    ):
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
        self.db = (
            db
            if db is not None
            else Database(self.data_dir / "congress.duckdb", read_only=read_only)
        )

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
        if not self.db.transactions_exist(year, sources=("house_pdf", "gemini_ocr")):
            raise DataSourceError(
                f"No cached data found for {year}. Run 'ptr-alpha parse --year {year}' first."
            )

        df = self.db.get_transactions(year, sources=("house_pdf", "gemini_ocr"))
        logger.info(f"Loaded {len(df)} cached transactions for {year}")
        return df

    def fetch_metadata(self, archive_year: int, refresh: bool = False) -> pd.DataFrame:
        if not refresh and self.db.metadata_exists(archive_year):
            logger.info("Loading cached metadata for archive %d", archive_year)
            return self.db.get_metadata(archive_year)

        metadata_url = self.metadata_url_template.format(year=archive_year)
        try:
            return self._download_and_upsert_metadata(
                archive_year,
                metadata_url,
                replace=refresh,
                bypass_cache=refresh,
            )
        except requests.RequestException as e:
            raise DataSourceError(
                f"Failed to fetch metadata archive {archive_year}: {e}"
            ) from e
        except (DataSourceError, ParsingError):
            raise
        except Exception as e:
            raise DataSourceError(
                f"Failed to fetch metadata archive {archive_year}: {e}"
            ) from e

    def _acquire_metadata_archive(
        self,
        archive_year: int,
        *,
        bypass_cache: bool,
    ) -> tuple[pd.DataFrame, dict]:
        metadata_url = self.metadata_url_template.format(year=archive_year)
        logger.info(
            "Downloading metadata archive %d from House disclosures", archive_year
        )
        if bypass_cache:
            with self.session.cache_disabled():
                response = self.session.get(metadata_url, timeout=30)
        else:
            response = self.session.get(metadata_url, timeout=30)
        if response.status_code != 200:
            raise DataSourceError(
                f"Failed to fetch metadata archive {archive_year}, "
                f"status: {response.status_code}"
            )

        content = _read_first_text_from_zip(response.content, archive_year)
        df = normalize_house_metadata(content)
        if "FilingType" not in df.columns:
            raise ParsingError("Missing required metadata column: FilingType")
        df["fetched_at"] = datetime.now()
        df["archive_year"] = archive_year
        df = df.rename(
            columns={
                "DocID": "doc_id",
                "First": "first_name",
                "Last": "last_name",
                "FilingDate": "filing_date",
                "FilingType": "filing_type",
            }
        )
        headers = getattr(response, "headers", {})
        provenance = {
            "metadata_sha256": hashlib.sha256(response.content).hexdigest(),
            "metadata_http_status": response.status_code,
            "metadata_etag": headers.get("ETag"),
            "metadata_last_modified": headers.get("Last-Modified"),
        }
        return df, provenance

    def _download_and_upsert_metadata(
        self,
        archive_year: int,
        metadata_url: str,
        *,
        replace: bool = False,
        bypass_cache: bool = False,
    ) -> pd.DataFrame:
        del metadata_url  # URL is derived from authoritative archive settings.
        df, _ = self._acquire_metadata_archive(
            archive_year, bypass_cache=bypass_cache
        )
        if replace:
            self.db.replace_metadata(archive_year, df)
        else:
            self.db.upsert_metadata(df)
        logger.info(
            "Cached metadata archive %d: %d records", archive_year, len(df)
        )
        return self.db.get_metadata(archive_year)

    async def _download_pdf_async(self, session, doc_id, pdf_path, url):
        existing_sha = _validated_pdf_sha256(pdf_path)
        if existing_sha:
            return HousePdfAcquisition(
                doc_id=str(doc_id),
                status=DownloadStatus.SKIPPED,
                artifact_sha256=existing_sha,
                content_length=pdf_path.stat().st_size,
            )

        tmp_path = pdf_path.with_suffix(".pdf.tmp")
        try:
            async with session.get(
                url, timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status != 200:
                    return HousePdfAcquisition(
                        doc_id=str(doc_id),
                        status=DownloadStatus.FAILED,
                        status_code=response.status,
                        error_message=f"HTTP {response.status}",
                    )
                content = await response.read()
                if not _valid_pdf_bytes(content):
                    return HousePdfAcquisition(
                        doc_id=str(doc_id),
                        status=DownloadStatus.FAILED,
                        status_code=response.status,
                        error_message="invalid or truncated PDF artifact",
                    )
                with open(tmp_path, "wb") as artifact:
                    artifact.write(content)
                os.replace(tmp_path, pdf_path)
                return HousePdfAcquisition(
                    doc_id=str(doc_id),
                    status=DownloadStatus.SUCCESS,
                    status_code=response.status,
                    artifact_sha256=hashlib.sha256(content).hexdigest(),
                    etag=response.headers.get("ETag"),
                    last_modified=response.headers.get("Last-Modified"),
                    content_length=len(content),
                )
        except (aiohttp.ClientError, OSError, asyncio.TimeoutError) as exc:
            return HousePdfAcquisition(
                doc_id=str(doc_id),
                status=DownloadStatus.ERROR,
                error_message=str(exc),
            )
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                logger.warning("Unable to remove temporary PDF %s", tmp_path)

    async def _fetch_and_cache_pdfs_async(
        self,
        archive_year: int,
        *,
        refresh_metadata: bool | None = None,
    ) -> HouseFetchSummary:
        if refresh_metadata is None:
            refresh_metadata = archive_year == date.today().year
        authoritative = refresh_metadata or not self.db.metadata_exists(archive_year)
        if authoritative:
            metadata_df, metadata_provenance = self._acquire_metadata_archive(
                archive_year, bypass_cache=True
            )
        else:
            metadata = self.db.get_metadata(archive_year)
            metadata_df = metadata.rename(
                columns={
                    "DocID": "doc_id",
                    "First": "first_name",
                    "Last": "last_name",
                    "FilingDate": "filing_date",
                    "FilingType": "filing_type",
                    "ArchiveYear": "archive_year",
                }
            )
            metadata_df["fetched_at"] = datetime.now()
            serialized = metadata_df.sort_values("doc_id").to_csv(index=False).encode()
            metadata_provenance = {
                "metadata_sha256": hashlib.sha256(serialized).hexdigest(),
                "metadata_http_status": None,
                "metadata_etag": None,
                "metadata_last_modified": None,
            }

        ptrs = metadata_df[metadata_df["filing_type"] == FilingType.PTR.value]
        doc_ids = ptrs["doc_id"].astype(str).tolist()
        official_doc_ids = set(doc_ids)
        generation_id = (
            datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
            + "-"
            + uuid.uuid4().hex[:12]
        )
        archive_dir = self.data_dir / str(archive_year)
        canonical_pdf_dir = archive_dir / "pdfs"
        stage_root = (
            self.data_dir
            / ".staging"
            / "house"
            / str(archive_year)
            / generation_id
        )
        stage_pdf_dir = stage_root / "pdfs"
        stage_pdf_dir.mkdir(parents=True, exist_ok=False)
        expected_hashes = self.db.get_house_artifact_hashes(archive_year)

        logger.info(
            "Staging %d PTR filings for archive %d (authoritative=%s)",
            len(doc_ids),
            archive_year,
            authoritative,
        )
        results: list[HousePdfAcquisition] = []
        downloads: list[tuple[str, Path, str]] = []
        for doc_id in doc_ids:
            staged_path = stage_pdf_dir / f"{doc_id}.pdf"
            canonical_path = canonical_pdf_dir / f"{doc_id}.pdf"
            canonical_sha = _validated_pdf_sha256(canonical_path)
            if (
                not authoritative
                and canonical_sha
                and expected_hashes.get(doc_id) == canonical_sha
            ):
                shutil.copy2(canonical_path, staged_path)
                results.append(
                    HousePdfAcquisition(
                        doc_id=doc_id,
                        status=DownloadStatus.SKIPPED,
                        artifact_sha256=canonical_sha,
                        content_length=staged_path.stat().st_size,
                    )
                )
                continue
            downloads.append(
                (
                    doc_id,
                    staged_path,
                    self.pdf_url_template.format(
                        year=archive_year, doc_id=doc_id
                    ),
                )
            )

        try:
            connector = aiohttp.TCPConnector(limit=10)
            async with aiohttp.ClientSession(connector=connector) as session:
                async with asyncio.TaskGroup() as task_group:
                    tasks = [
                        task_group.create_task(
                            self._download_pdf_async(session, doc_id, path, url)
                        )
                        for doc_id, path, url in downloads
                    ]
                results.extend(task.result() for task in tasks)

            result_by_id = {result.doc_id: result for result in results}
            missing_doc_ids = sorted(
                doc_id
                for doc_id in doc_ids
                if not _validated_pdf_sha256(
                    stage_pdf_dir / f"{doc_id}.pdf"
                )
            )
            if missing_doc_ids:
                details = []
                for doc_id in missing_doc_ids:
                    failure = result_by_id.get(doc_id)
                    reason = (
                        failure.error_message
                        if failure and failure.error_message
                        else "missing staged artifact"
                    )
                    details.append(f"{doc_id} ({reason})")
                raise DataSourceError(
                    f"Incomplete House archive {archive_year}: "
                    f"{len(doc_ids) - len(missing_doc_ids)}/{len(doc_ids)} "
                    f"valid PTR PDFs; missing {len(missing_doc_ids)}: "
                    + ", ".join(details)
                )

            old_pdf_ids = (
                {path.stem for path in canonical_pdf_dir.glob("*.pdf")}
                if canonical_pdf_dir.exists()
                else set()
            )
            orphan_ids = sorted(old_pdf_ids - official_doc_ids)
            backup_pdf_dir = stage_root / "previous-pdfs"
            quarantine_dir = archive_dir / "quarantine" / generation_id
            moved_orphans: list[tuple[Path, Path]] = []
            old_generation_moved = False
            new_generation_promoted = False
            try:
                archive_dir.mkdir(parents=True, exist_ok=True)
                if canonical_pdf_dir.exists():
                    os.replace(canonical_pdf_dir, backup_pdf_dir)
                    old_generation_moved = True
                os.replace(stage_pdf_dir, canonical_pdf_dir)
                new_generation_promoted = True
                quarantine_dir.mkdir(parents=True, exist_ok=False)
                quarantined_artifacts = []
                for doc_id in orphan_ids:
                    old_path = backup_pdf_dir / f"{doc_id}.pdf"
                    quarantine_path = quarantine_dir / old_path.name
                    if old_path.exists():
                        os.replace(old_path, quarantine_path)
                        moved_orphans.append((quarantine_path, old_path))
                    quarantined_artifacts.append(
                        {
                            "doc_id": doc_id,
                            "artifact_sha256": (
                                _sha256_file(quarantine_path)
                                if quarantine_path.exists()
                                else None
                            ),
                            "quarantine_path": str(quarantine_path),
                        }
                    )
                artifacts = []
                for doc_id in doc_ids:
                    result = result_by_id[doc_id]
                    canonical_path = canonical_pdf_dir / f"{doc_id}.pdf"
                    artifacts.append(
                        {
                            "doc_id": doc_id,
                            "artifact_sha256": _validated_pdf_sha256(
                                canonical_path
                            ),
                            "http_status": getattr(result, "status_code", None) or None,
                            "etag": getattr(result, "etag", None),
                            "last_modified": getattr(result, "last_modified", None),
                            "content_length": canonical_path.stat().st_size,
                        }
                    )
                removed_counts = self.db.promote_house_archive(
                    archive_year=archive_year,
                    metadata_df=metadata_df,
                    generation_id=generation_id,
                    artifacts=artifacts,
                    quarantined_artifacts=quarantined_artifacts,
                    **metadata_provenance,
                )
            except Exception:
                for quarantine_path, old_path in reversed(moved_orphans):
                    if quarantine_path.exists():
                        os.replace(quarantine_path, old_path)
                if new_generation_promoted and canonical_pdf_dir.exists():
                    os.replace(canonical_pdf_dir, stage_pdf_dir)
                if old_generation_moved and backup_pdf_dir.exists():
                    os.replace(backup_pdf_dir, canonical_pdf_dir)
                shutil.rmtree(quarantine_dir, ignore_errors=True)
                raise

            shutil.rmtree(backup_pdf_dir, ignore_errors=True)
            shutil.rmtree(stage_root, ignore_errors=True)
        except Exception:
            shutil.rmtree(stage_root, ignore_errors=True)
            raise

        self._log_download_summary(results)
        summary = HouseFetchSummary(
            archive_year=archive_year,
            metadata_count=len(metadata_df),
            ptr_count=len(doc_ids),
            valid_pdf_count=len(doc_ids),
            downloaded_count=sum(
                result.status == DownloadStatus.SUCCESS for result in results
            ),
            skipped_count=sum(
                result.status == DownloadStatus.SKIPPED for result in results
            ),
            orphan_pdf_count=len(orphan_ids),
            removed_doc_count=len(removed_counts),
            quarantined_pdf_count=len(moved_orphans),
            generation_id=generation_id,
            generation_status="incomplete",
        )
        logger.info(
            "House archive %d promoted: metadata=%d PTR=%d valid=%d "
            "downloaded=%d skipped=%d removed=%d quarantined=%d",
            summary.archive_year,
            summary.metadata_count,
            summary.ptr_count,
            summary.valid_pdf_count,
            summary.downloaded_count,
            summary.skipped_count,
            summary.removed_doc_count,
            summary.quarantined_pdf_count,
        )
        return summary

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

    def fetch_and_cache_pdfs(
        self,
        archive_year: int,
        *,
        refresh_metadata: bool | None = None,
    ) -> HouseFetchSummary:
        return asyncio.run(
            self._fetch_and_cache_pdfs_async(
                archive_year,
                refresh_metadata=refresh_metadata,
            )
        )

    def parse_cached_pdfs(self, year: int, *, force: bool = False) -> None:
        metadata = self.fetch_metadata(year)
        ptrs = metadata[metadata["FilingType"] == FilingType.PTR.value]
        pdf_dir = self.data_dir / str(year) / "pdfs"

        if not pdf_dir.exists():
            raise DataSourceError(f"PDF directory not found: {pdf_dir}")

        pdf_paths, existing_docs = _filter_existing_pdfs(ptrs, pdf_dir)
        if not pdf_paths:
            raise DataSourceError(f"No PDF files found in {pdf_dir}")

        if not force:
            ingestion_generation = self.db.get_latest_house_generation(year)
            if ingestion_generation is None:
                raise DataSourceError(
                    f"No acquired House generation exists for archive {year}"
                )
            artifact_hashes = {
                path.stem: sha256
                for path in pdf_paths
                if (sha256 := _validated_pdf_sha256(path))
            }
            cached = self.db.parse_runs.get_cached_doc_ids(
                year=year,
                parser_version=_PARSE_VERSION,
                artifact_hashes=artifact_hashes,
                ingestion_generation=ingestion_generation,
            )
            if cached:
                keep_mask = (
                    existing_docs["DocID"]
                    .astype(str)
                    .map(lambda d: d not in cached)
                    .to_numpy()
                )
                skipped = int((~keep_mask).sum()) if len(keep_mask) else 0
                pdf_paths = [p for p, keep in zip(pdf_paths, keep_mask) if keep]
                existing_docs = (
                    existing_docs[keep_mask]
                    if len(keep_mask)
                    else existing_docs.iloc[0:0]
                )
                logger.info(
                    "Skipping %d already-parsed PDFs for %d (%d remain)",
                    skipped,
                    year,
                    len(pdf_paths),
                )

        if not pdf_paths:
            logger.info("All PDFs for %d already parsed; nothing to do", year)
            return

        member_lookup = _build_member_lookup(existing_docs)
        logger.info(f"Parsing {len(pdf_paths)} PDFs for {year}")

        with Pool(self.parallel_workers) as pool:
            results = pool.map(_parse_pdf_worker, pdf_paths)

        self._save_parse_results(year, results, member_lookup)

    def _save_parse_results(
        self,
        year: int,
        results: list,
        member_lookup: dict,
    ) -> None:
        pdf_transactions: dict = {}
        parse_attempts: list[tuple[str, list[str]]] = []
        raw_transaction_counts: dict[str, int] = {}
        for pdf_path, transactions, engines_attempted in results:
            doc_id = pdf_path.stem
            pdf_transactions[pdf_path] = transactions
            raw_transaction_counts[doc_id] = len(transactions)
            parse_attempts.append((doc_id, engines_attempted))

        df = consolidate_transactions(pdf_transactions, member_lookup)
        transaction_counts = (
            df["doc_id"].astype(str).value_counts().to_dict() if not df.empty else {}
        )
        artifact_hashes = {
            path.stem: _validated_pdf_sha256(path) for path in pdf_transactions
        }
        ingestion_generation = (
            self.db.get_latest_house_generation(year)
            or f"legacy-untracked-{year}"
        )
        if not df.empty:
            df["chamber"] = "house"
            df["ingestion_generation"] = ingestion_generation
            df["source_record_id"] = df["doc_id"].astype(str)
            df["official_filing_date"] = df["disclosure_date"]
            df["artifact_sha256"] = df["doc_id"].astype(str).map(artifact_hashes)
            if "asset_description" in df.columns:
                df["raw_asset_description"] = df["asset_description"]
        parse_runs = [
            dict(
                doc_id=doc_id,
                year=year,
                parser_version=_PARSE_VERSION,
                status="success" if transaction_counts.get(doc_id, 0) else "zero_rows",
                engines_attempted=",".join(engines_attempted),
                raw_row_count=raw_transaction_counts.get(doc_id, 0),
                # Database.replace_transactions_for_docs overwrites this with
                # the actual persisted count inside the replacement transaction.
                transaction_count=0,
                artifact_sha256=artifact_hashes.get(doc_id),
                ingestion_generation=ingestion_generation,
            )
            for doc_id, engines_attempted in parse_attempts
        ]

        zero_row_doc_ids = [
            doc_id
            for doc_id, _ in parse_attempts
            if transaction_counts.get(doc_id, 0) == 0
        ]
        stale_docs = self.db.count_transactions_for_docs(zero_row_doc_ids)
        if stale_docs:
            doc_ids = list(stale_docs)[:10]
            logger.warning(
                "%d docs parsed to zero rows but retain %d existing DB rows (stale?): %s",
                len(stale_docs),
                sum(stale_docs.values()),
                ", ".join(doc_ids),
            )

        if not df.empty:
            df = preserve_existing_fields(df, self.db)

        attempted_doc_ids = [doc_id for doc_id, _ in parse_attempts]
        persisted = self.db.replace_transactions_for_docs(
            df,
            source="house_pdf",
            attempted_doc_ids=attempted_doc_ids,
            ingestion_generation=ingestion_generation,
            replacement_doc_ids=(
                df["doc_id"].astype(str).unique().tolist()
                if not df.empty
                else []
            ),
            parse_runs=parse_runs,
        )
        logger.info(
            "Persisted %d transactions across %d parsed PDFs "
            "(%d zero-row results; canonical=%d raw=%d database rows)",
            sum(persisted.by_doc_total.values()),
            len(parse_attempts),
            len(zero_row_doc_ids),
            persisted.total_current_rows,
            persisted.total_raw_rows,
        )
        logger.debug("Persisted rows by doc/source: %s", persisted.by_doc_source)


# ── Helpers ──


def _valid_pdf_bytes(content: bytes) -> bool:
    return content.startswith(b"%PDF-") and b"%%EOF" in content[-2048:]


def _validated_pdf_sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        content = path.read_bytes()
    except OSError:
        return None
    if not _valid_pdf_bytes(content):
        return None
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path: Path) -> str:
    with open(path, "rb") as artifact:
        return hashlib.file_digest(artifact, "sha256").hexdigest()


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
            key = _identity_key(
                er.get("member"), er.get("transaction_date"), er.get("transaction_type")
            )
            slot = lookup.setdefault(key, {"tickers": set(), "amounts": set()})
            if not _is_blank(er.get("ticker")):
                slot["tickers"].add(er["ticker"])
            if not _is_blank(er.get("amount_raw")):
                slot["amounts"].add(er["amount_raw"])

        for idx in df.index[df["doc_id"] == doc_id]:
            key = _identity_key(
                df.at[idx, "member"],
                df.at[idx, "transaction_date"],
                df.at[idx, "transaction_type"],
            )
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
        text_files = [
            f
            for f in z.namelist()
            if f.lower().endswith(".txt") and not f.endswith("/")
        ]
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


def _filter_existing_pdfs(
    ptrs: pd.DataFrame, pdf_dir: Path
) -> tuple[list[Path], pd.DataFrame]:
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
    duplicate_ids = (
        distinct.loc[distinct["DocID"].astype(str).duplicated(keep=False), "DocID"]
        .astype(str)
        .unique()
    )
    if len(duplicate_ids):
        raise ParsingError(
            "Duplicate metadata DocID(s) would make member attribution ambiguous: "
            + ", ".join(duplicate_ids[:10])
        )
    existing_docs = distinct.drop_duplicates(subset=["DocID"])
    return dict(
        zip(
            existing_docs["DocID"].values,
            [
                {"First": f, "Last": last, "FilingDate": d}
                for f, last, d in zip(
                    existing_docs["First"].values,
                    existing_docs["Last"].values,
                    existing_docs["FilingDate"].values,
                )
            ],
        )
    )
