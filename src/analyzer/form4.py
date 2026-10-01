"""Official SEC Form 4 source for open-market insider trades."""

from __future__ import annotations

import logging
import os
import re
import time
import xml.etree.ElementTree as ET
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from urllib.parse import quote, urlsplit
from uuid import uuid4

import pandas as pd
import requests

from analyzer.database import Database
from analyzer.interfaces import TransactionSource
from analyzer.transaction_repository import _BASE_WRITE_COLUMNS

logger = logging.getLogger(__name__)

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data"
SEC_DAILY_INDEX_URL = "https://www.sec.gov/Archives/edgar/daily-index"
DEFAULT_USER_AGENT = "ptr-alpha research contact@example.com"
REQUEST_TIMEOUT = 30
REQUEST_INTERVAL = 1 / 8
MAX_RETRIES = 5
RETRY_BASE_SECONDS = 1.0
_ACCESSION_RE = re.compile(r"\d{10}-\d{2}-\d{6}\Z")
_XML_BLOCK_RE = re.compile(rb"<XML>\s*(.*?)\s*</XML>", re.IGNORECASE | re.DOTALL)
_POSITIVE_10B5_RE = re.compile(
    r"(?:pursuant to|under|in accordance with|subject to)\s+"
    r"(?:a\s+)?(?:written\s+)?(?:trading\s+)?(?:rule\s+)?"
    r"10b5\s*[-‐‑–—]?\s*1|"
    r"10b5\s*[-‐‑–—]?\s*1.{0,80}\bplan\b",
    re.IGNORECASE,
)
_NEGATIVE_10B5_RE = re.compile(
    r"\b(?:not|no|without)\b.{0,50}10b5\s*[-‐‑–—]?\s*1|"
    r"10b5\s*[-‐‑–—]?\s*1.{0,50}\b(?:not|no|without)\b",
    re.IGNORECASE,
)
_TRADE_PRICE_RE = re.compile(r"\bshares\s+@\s+\$([0-9]+(?:\.[0-9]+)?)\b")
_FORM4_COLUMNS = _BASE_WRITE_COLUMNS + (
    "source_record_id",
    "source_row_id",
    "source_report_path",
    "raw_owner",
    "official_filing_date",
    "ingestion_generation",
    "raw_transaction_subtype",
    "raw_asset_description",
)
_SWEEP_CANDIDATE_COLUMNS = (
    "ticker",
    "actor_id",
    "kind",
    "source",
    "entry_ref",
    "event_date",
    "disclosure_date",
    "corroboration",
    "position_evidence",
    "as_of",
)


class Form4Error(Exception):
    """Raised when SEC Form 4 data is unavailable or fails validation."""


def _accession_digits(accession):
    accession = str(accession)
    if not _ACCESSION_RE.fullmatch(accession):
        raise ValueError(f"Invalid SEC accession: {accession!r}")
    return accession.replace("-", "")


class _SECClient:
    def __init__(self, user_agent, headers):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent, **headers})
        self._last_request_at = 0.0

    def close(self):
        self.session.close()

    def get(self, url, error_type):
        for attempt in range(MAX_RETRIES):
            delay = REQUEST_INTERVAL - (time.monotonic() - self._last_request_at)
            if delay > 0:
                time.sleep(delay)
            self._last_request_at = time.monotonic()
            try:
                response = self.session.get(url, timeout=REQUEST_TIMEOUT)
            except requests.RequestException as exc:
                if attempt + 1 == MAX_RETRIES:
                    raise error_type(f"SEC request failed for {url}: {exc}") from exc
                time.sleep(RETRY_BASE_SECONDS * (2**attempt))
                continue
            if response.status_code in (429, 503):
                if attempt + 1 == MAX_RETRIES:
                    raise error_type(
                        f"SEC rate-limited {url} with HTTP {response.status_code} "
                        f"after {MAX_RETRIES} attempts"
                    )
                retry_after = response.headers.get("Retry-After")
                try:
                    backoff = (
                        float(retry_after)
                        if retry_after
                        else RETRY_BASE_SECONDS * (2**attempt)
                    )
                except ValueError:
                    backoff = RETRY_BASE_SECONDS * (2**attempt)
                logger.warning(
                    "SEC HTTP %s for %s; retrying in %.1fs",
                    response.status_code,
                    url,
                    backoff,
                )
                time.sleep(max(backoff, 0))
                continue
            if response.status_code != 200:
                raise error_type(
                    f"SEC returned HTTP {response.status_code} for {url}"
                )
            final = urlsplit(response.url)
            expected = urlsplit(url)
            if final.scheme != "https" or final.hostname != expected.hostname:
                raise error_type(
                    f"SEC request redirected outside the official host: {url}"
                )
            return response
        raise error_type(f"SEC retry limit reached for {url}")

    def submissions(self, cik, error_type):
        url = SEC_SUBMISSIONS_URL.format(cik=str(int(cik)).zfill(10))
        try:
            data = self.get(url, error_type).json()
        except (ValueError, requests.RequestException) as exc:
            raise error_type(
                f"SEC submissions index for CIK {cik} is invalid"
            ) from exc
        if not isinstance(data, dict):
            raise error_type(f"SEC submissions index for CIK {cik} is not an object")
        return data


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _first_element(element: ET.Element, name: str) -> ET.Element | None:
    return next(
        (node for node in element.iter() if _local_name(node.tag) == name), None
    )


def _field_value(element: ET.Element, field: str) -> str:
    node = _first_element(element, field)
    if node is None:
        return ""
    value = _first_element(node, "value")
    return " ".join((value if value is not None else node).itertext()).strip()


def _clean_text(value: str) -> str:
    return " ".join(value.split())


def _parse_decimal(value: str, label: str, accession: str) -> Decimal:
    try:
        number = Decimal(value)
    except (InvalidOperation, ValueError):
        raise Form4Error(
            f"Form 4 {accession} has invalid {label}: {value!r}"
        ) from None
    if not number.is_finite() or number <= 0:
        raise Form4Error(f"Form 4 {accession} has invalid {label}: {value!r}")
    return number


def _is_10b5_1(root: ET.Element) -> bool:
    checkbox = _first_element(root, "aff10b5One")
    if checkbox is not None:
        value = _clean_text(" ".join(checkbox.itertext())).lower()
        if value in {"1", "true", "yes", "checked", "x"}:
            return True
        if value in {"0", "false", "no", ""}:
            return False

    footnotes = " ".join(
        _clean_text(" ".join(node.itertext()))
        for node in root.iter()
        if _local_name(node.tag) == "footnote"
    )
    return bool(_POSITIVE_10B5_RE.search(footnotes)) and not bool(
        _NEGATIVE_10B5_RE.search(footnotes)
    )


def parse_form4_xml(
    content: bytes | str,
    *,
    accession: str,
    source_report_path: str,
    official_filing_date: date | str,
    ingestion_generation: str | None = None,
) -> pd.DataFrame:
    """Parse a filing into canonical open-market purchase and sale rows.

    Sales (code S) feed officer holdings so closed positions stop scoring;
    only purchases initiate setups downstream.
    """
    try:
        root = ET.fromstring(content)
    except ET.ParseError as exc:
        raise Form4Error(f"Form 4 {accession} is not valid XML: {exc}") from exc
    if _local_name(root.tag) != "ownershipDocument":
        raise Form4Error(f"SEC filing {accession} is not an ownership document")

    issuer = _first_element(root, "issuer")
    ticker = (
        _clean_text(_field_value(issuer, "issuerTradingSymbol"))
        if issuer is not None
        else ""
    )
    owner_id = _first_element(root, "reportingOwnerId")
    owner = (
        _clean_text(_field_value(owner_id, "rptOwnerName"))
        if owner_id is not None
        else ""
    )

    try:
        filing_date = date.fromisoformat(str(official_filing_date))
    except ValueError:
        raise Form4Error(
            f"Form 4 {accession} has invalid filing date: {official_filing_date!r}"
        ) from None

    trades = [
        node
        for node in root.iter()
        if _local_name(node.tag) == "nonDerivativeTransaction"
    ]
    is_10b5_1 = _is_10b5_1(root)
    rows: list[dict] = []
    for row_number, trade in enumerate(trades, start=1):
        code = _field_value(trade, "transactionCode").upper()
        if code not in ("P", "S"):
            continue
        is_purchase = code == "P"

        transaction_date = _field_value(trade, "transactionDate")
        shares_raw = _field_value(trade, "transactionShares")
        price_raw = _field_value(trade, "transactionPricePerShare")
        missing = [
            name
            for name, value in (
                ("owner name", owner),
                ("ticker", ticker),
                ("transaction date", transaction_date),
                ("shares", shares_raw),
                ("price", price_raw),
            )
            if not value
        ]
        if missing:
            raise Form4Error(
                f"Form 4 {accession} {'purchase' if is_purchase else 'sale'} "
                f"row {row_number} is missing " + ", ".join(missing)
            )
        try:
            parsed_date = date.fromisoformat(transaction_date)
        except ValueError:
            raise Form4Error(
                f"Form 4 {accession} has invalid transaction date: "
                f"{transaction_date!r}"
            ) from None

        shares = _parse_decimal(shares_raw, "transaction shares", accession)
        price = _parse_decimal(price_raw, "transaction price", accession)
        amount = shares * price
        security = _clean_text(_field_value(trade, "securityTitle")) or "Common Stock"
        shares_text = format(shares.normalize(), "f")
        price_text = format(price.normalize(), "f")
        description = f"{security}; {shares_text} shares @ ${price_text}; "
        description += f"is_10b5_1={str(is_10b5_1).lower()}"
        rows.append(
            {
                "doc_id": f"form4-{accession}",
                "member": owner,
                "ticker": ticker.upper(),
                "transaction_date": parsed_date,
                "disclosure_date": filing_date,
                "transaction_type": "Purchase" if is_purchase else "Sale",
                "owner_code": None,
                "amount_raw": f"${amount:,.2f}",
                "amount_midpoint": float(amount),
                "instrument_type": "Common Stock",
                "strike_price": None,
                "expiry_date": None,
                "created_at": None,
                "asset_description": description,
                "source": "form4",
                "source_record_id": accession,
                "source_row_id": f"nonDerivativeTransaction:{row_number:06d}",
                "source_report_path": source_report_path,
                "raw_owner": owner,
                "official_filing_date": filing_date,
                "ingestion_generation": ingestion_generation,
                "raw_transaction_subtype": code,
                "raw_asset_description": description,
            }
        )
    return pd.DataFrame(rows, columns=_FORM4_COLUMNS)


def _ownership_xml(content: bytes) -> bytes:
    blocks = _XML_BLOCK_RE.findall(content)
    if len(blocks) == 1:
        return blocks[0]
    if not blocks and content.lstrip().startswith(b"<?xml"):
        return content
    raise Form4Error(
        f"SEC complete submission must contain exactly one ownership XML block; "
        f"found {len(blocks)}"
    )


def _is_officer_filing(content: bytes) -> bool:
    try:
        root = ET.fromstring(content)
    except ET.ParseError as exc:
        raise Form4Error(f"SEC filing is not valid XML: {exc}") from exc
    if _local_name(root.tag) != "ownershipDocument":
        raise Form4Error("SEC filing is not an ownership document")
    owner = next(
        (node for node in root.iter() if _local_name(node.tag) == "reportingOwner"),
        None,
    )
    if owner is None:
        return False
    relationship = _first_element(owner, "reportingOwnerRelationship")
    if relationship is None:
        return False
    return _clean_text(_field_value(relationship, "isOfficer")).lower() in {
        "1",
        "true",
        "yes",
        "x",
    }


def candidates_from_sweep(df: pd.DataFrame) -> pd.DataFrame:
    """Convert sweep purchase rows to candidates; sales never initiate."""
    rows = []
    as_of = datetime.now(UTC).date()
    for row in df.to_dict("records"):
        if str(row.get("transaction_type") or "").strip().lower() != "purchase":
            continue
        description = str(row.get("raw_asset_description") or "")
        price = _TRADE_PRICE_RE.search(description)
        owner = " ".join(str(row.get("raw_owner") or "").split()).upper()
        ticker = " ".join(str(row.get("ticker") or "").split()).upper()
        if not price or not owner or not ticker:
            continue
        rows.append(
            {
                "ticker": ticker,
                "actor_id": f"officer:{owner}",
                "kind": "officer",
                "source": "form4",
                "entry_ref": float(price.group(1)),
                "event_date": row["transaction_date"],
                "disclosure_date": row["official_filing_date"],
                "corroboration": False,
                "position_evidence": False,
                "as_of": as_of,
            }
        )
    return pd.DataFrame(rows, columns=_SWEEP_CANDIDATE_COLUMNS)


class Form4Source(TransactionSource):
    """Fetch SEC Form 4 filings and persist Table I open-market purchases."""

    def __init__(
        self,
        data_dir: str | Path = "data",
        read_only: bool = False,
        db: Database | None = None,
        ingestion_generation: str | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._owns_db = db is None
        self.db = (
            db
            if db is not None
            else Database(self.data_dir / "congress.duckdb", read_only=read_only)
        )
        self._sec = _SECClient(
            os.environ.get("SEC_USER_AGENT") or DEFAULT_USER_AGENT,
            {"Accept-Encoding": "gzip, deflate"},
        )
        self.session = self._sec.session
        self.ingestion_generation = ingestion_generation
        self._ticker_map: dict[str, int] | None = None

    def close(self) -> None:
        self._sec.close()
        if self._owns_db:
            self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        return False

    def get_transactions(self, year: int) -> pd.DataFrame:
        return self.db.get_transactions(year, source="form4")

    def _request(self, url: str) -> requests.Response:
        return self._sec.get(url, Form4Error)

    def _load_ticker_map(self) -> dict[str, int]:
        if self._ticker_map is not None:
            return self._ticker_map
        try:
            data = self._request(SEC_TICKERS_URL).json()
        except (ValueError, requests.RequestException) as exc:
            raise Form4Error(f"SEC company ticker index is invalid: {exc}") from exc
        if not isinstance(data, dict):
            raise Form4Error("SEC company ticker index is not an object")

        tickers: dict[str, int] = {}
        for item in data.values():
            if not isinstance(item, dict):
                raise Form4Error("SEC company ticker index has an invalid record")
            ticker = str(item.get("ticker") or "").strip().upper()
            cik = item.get("cik_str")
            if not ticker or isinstance(cik, bool) or not str(cik or "").isdigit():
                raise Form4Error("SEC company ticker index has a missing ticker or CIK")
            if ticker in tickers and tickers[ticker] != int(cik):
                raise Form4Error(
                    f"SEC company ticker index has ambiguous ticker {ticker}"
                )
            tickers[ticker] = int(cik)
        self._ticker_map = tickers
        return tickers

    def _submissions(self, cik: int) -> list[dict]:
        data = self._sec.submissions(cik, Form4Error)
        try:
            recent = data["filings"]["recent"]
            columns = (
                recent["accessionNumber"],
                recent["form"],
                recent["filingDate"],
                recent["primaryDocument"],
            )
            records = list(zip(*columns, strict=True))
        except (KeyError, TypeError, ValueError) as exc:
            raise Form4Error(
                f"SEC submissions index for CIK {cik} has an invalid schema"
            ) from exc

        filings: list[dict] = []
        for accession, form, filed, document in records:
            if form not in {"4", "4/A"}:
                continue
            try:
                _accession_digits(accession)
            except ValueError:
                raise Form4Error(f"SEC returned invalid Form 4 accession: {accession!r}")
            try:
                filing_date = date.fromisoformat(str(filed))
            except ValueError:
                raise Form4Error(
                    f"SEC returned invalid filing date for {accession}: {filed!r}"
                ) from None
            document = str(document or "")
            document_parts = document.split("/")
            if (
                not document
                or "\\" in document
                or any(part in {"", ".", ".."} for part in document_parts)
            ):
                raise Form4Error(
                    f"SEC returned invalid primary document for {accession}"
                )
            filings.append(
                {
                    "accession": str(accession),
                    "filing_date": filing_date,
                    "primary_document": document,
                    "ownership_document": f"{accession}.txt",
                }
            )
        return filings

    @staticmethod
    def _parse_master_index(content: bytes | str) -> list[dict]:
        text = (
            content.decode("utf-8-sig", errors="replace")
            if isinstance(content, bytes)
            else content
        )
        filings = []
        in_records = False
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not in_records:
                if line.strip().startswith("-----"):
                    in_records = True
                continue
            if not line.strip():
                continue
            fields = line.split("|")
            if len(fields) != 5:
                logger.warning(
                    "Skipping malformed SEC master index line %d", line_number
                )
                continue
            raw_cik, _, form, raw_filing_date, filename = (
                field.strip() for field in fields
            )
            if form not in {"4", "4/A"}:
                continue
            try:
                if not raw_cik.isdigit():
                    raise ValueError
                cik = int(raw_cik)
                filing_date = date.fromisoformat(raw_filing_date)
                parts = filename.split("/")
                if (
                    len(parts) < 4
                    or parts[:2] != ["edgar", "data"]
                    or not parts[2].isdigit()
                    or int(parts[2]) != cik
                    or any(
                        part in {"", ".", ".."} or "\\" in part for part in parts[2:]
                    )
                ):
                    raise ValueError
                if len(parts) == 4 and re.fullmatch(r"\d{18}\.txt", parts[3]):
                    accession_digits = parts[3][:-4]
                elif len(parts) >= 5 and re.fullmatch(r"\d{18}", parts[3]):
                    accession_digits = parts[3]
                else:
                    raise ValueError
                if not re.fullmatch(r"\d{18}", accession_digits):
                    raise ValueError
                if int(accession_digits[:10]) != cik:
                    raise ValueError
            except ValueError:
                logger.warning(
                    "Skipping malformed SEC master index line %d", line_number
                )
                continue
            filings.append(
                {
                    "cik": cik,
                    "accession": (
                        f"{accession_digits[:10]}-{accession_digits[10:12]}-"
                        f"{accession_digits[12:]}"
                    ),
                    "filing_date": filing_date,
                    "ownership_document": f"{accession_digits}.txt",
                }
            )
        return filings

    @staticmethod
    def _daily_index_url(filing_day: date) -> str:
        quarter = (filing_day.month - 1) // 3 + 1
        return (
            f"{SEC_DAILY_INDEX_URL}/{filing_day.year}/QTR{quarter}/"
            f"master.{filing_day:%Y%m%d}.idx"
        )

    def sweep_recent(
        self, days_back: int = 7, *, limit_filings: int = 500
    ) -> pd.DataFrame:
        """Fetch recent officer Form 4 open-market buys across the filing universe."""
        empty = pd.DataFrame(columns=_FORM4_COLUMNS)
        if days_back < 1 or limit_filings < 1:
            return empty

        today = datetime.now(UTC).date()
        start_date = today - timedelta(days=days_back - 1)
        filings = []
        for offset in range(days_back):
            filing_day = today - timedelta(days=offset)
            try:
                content = self._request(self._daily_index_url(filing_day)).content
            except Form4Error as exc:
                if "HTTP 404" not in str(exc):
                    logger.warning(
                        "Could not fetch SEC daily index for %s: %s", filing_day, exc
                    )
                continue
            filings.extend(self._parse_master_index(content))

        recent = sorted(
            (
                filing
                for filing in filings
                if start_date <= filing["filing_date"] <= today
            ),
            key=lambda filing: filing["filing_date"],
            reverse=True,
        )[:limit_filings]
        rows = []
        for filing in recent:
            url = self._archive_url(filing["cik"], filing)
            try:
                xml = _ownership_xml(self._request(url).content)
                if not _is_officer_filing(xml):
                    continue
                parsed = parse_form4_xml(
                    xml,
                    accession=filing["accession"],
                    source_report_path=url,
                    official_filing_date=filing["filing_date"],
                    ingestion_generation=self.ingestion_generation,
                )
                if not parsed.empty:
                    rows.append(parsed)
            except Exception as exc:
                logger.warning(
                    "Skipping SEC Form 4 filing %s: %s",
                    filing["accession"],
                    exc,
                    exc_info=True,
                )
        return pd.concat(rows, ignore_index=True) if rows else empty

    @staticmethod
    def _archive_url(cik: int, filing: dict) -> str:
        accession_no_dashes = _accession_digits(filing["accession"])
        document = quote(filing["ownership_document"], safe="._-")
        return f"{SEC_ARCHIVES_URL}/{cik}/{accession_no_dashes}/{document}"

    def fetch_accession(self, ticker: str, accession: str) -> pd.DataFrame:
        """Fetch one exact filing; useful for audit and filing-level refreshes."""
        normalized_ticker = ticker.strip().upper()
        try:
            cik = self._load_ticker_map()[normalized_ticker]
        except KeyError:
            raise Form4Error(f"Ticker not found in SEC company index: {ticker!r}") from None
        filing = next(
            (
                record
                for record in self._submissions(cik)
                if record["accession"] == accession
            ),
            None,
        )
        if filing is None:
            raise Form4Error(f"SEC submissions do not list Form 4 accession {accession}")
        url = self._archive_url(cik, filing)
        response = self._request(url)
        return parse_form4_xml(
            _ownership_xml(response.content),
            accession=filing["accession"],
            source_report_path=url,
            official_filing_date=filing["filing_date"],
            ingestion_generation=self.ingestion_generation,
        )

    def fetch_ticker_trades(self, ticker: str, year: int) -> pd.DataFrame:
        normalized_ticker = ticker.strip().upper()
        try:
            cik = self._load_ticker_map()[normalized_ticker]
        except KeyError:
            raise Form4Error(f"Ticker not found in SEC company index: {ticker!r}") from None
        frames = []
        for filing in self._submissions(cik):
            if filing["filing_date"].year != year:
                continue
            url = self._archive_url(cik, filing)
            response = self._request(url)
            parsed = parse_form4_xml(
                _ownership_xml(response.content),
                accession=filing["accession"],
                source_report_path=url,
                official_filing_date=filing["filing_date"],
                ingestion_generation=self.ingestion_generation,
            )
            if not parsed.empty:
                frames.append(parsed)
        result = (
            pd.concat(frames, ignore_index=True)
            if frames
            else pd.DataFrame(columns=_FORM4_COLUMNS)
        )
        logger.info(
            "Fetched %d Form 4 purchase rows for %s in %d",
            len(result),
            normalized_ticker,
            year,
        )
        return result

    def fetch_all_trades(
        self, year: int, tickers: list[str] | tuple[str, ...] | None = None
    ) -> pd.DataFrame:
        """Fetch recent Form 4 purchases for the requested tickers or SEC index."""
        ticker_map = self._load_ticker_map()
        selected = (
            sorted({ticker.strip().upper() for ticker in tickers})
            if tickers is not None
            else sorted(ticker_map)
        )
        unknown = sorted(set(selected) - set(ticker_map))
        if unknown:
            raise Form4Error(f"Tickers not found in SEC company index: {unknown}")
        frames = [self.fetch_ticker_trades(ticker, year) for ticker in selected]
        nonempty = [frame for frame in frames if not frame.empty]
        return (
            pd.concat(nonempty, ignore_index=True)
            if nonempty
            else pd.DataFrame(columns=_FORM4_COLUMNS)
        )

    def save_to_db(self, df: pd.DataFrame) -> int:
        if tuple(df.columns) != _FORM4_COLUMNS:
            raise Form4Error(
                "Form 4 transaction columns do not match the canonical schema"
            )
        if not self.ingestion_generation or not self.ingestion_generation.strip():
            raise Form4Error("ingestion_generation is required for Form 4 persistence")
        if not df.empty:
            if not df["source"].eq("form4").all() or not df[
                "ingestion_generation"
            ].eq(self.ingestion_generation).all():
                raise Form4Error("Form 4 source or ingestion generation does not match")
            required = [
                "doc_id",
                "member",
                "ticker",
                "transaction_date",
                "disclosure_date",
                "source_record_id",
                "source_row_id",
                "source_report_path",
                "raw_owner",
                "official_filing_date",
                "raw_transaction_subtype",
            ]
            if df[required].isna().any().any():
                raise Form4Error("Form 4 transaction provenance is incomplete")
            from analyzer.transaction_repository import drop_already_stored

            df = drop_already_stored(self.db, df, source="form4")
        return self.db.upsert_transactions(df, source="form4")

    def fetch_and_save_ticker(self, ticker: str, year: int) -> int:
        return self.save_to_db(self.fetch_ticker_trades(ticker, year))

    def fetch_and_save_all(
        self, year: int, tickers: list[str] | tuple[str, ...] | None = None
    ) -> int:
        return self.save_to_db(self.fetch_all_trades(year, tickers))

    def fetch_and_save_sweep(self, days_back: int = 7, limit_filings: int = 500) -> int:
        self.ingestion_generation = uuid4().hex
        return self.save_to_db(
            self.sweep_recent(days_back, limit_filings=limit_filings)
        )
