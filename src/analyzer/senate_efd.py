"""Senate eFD PTR scraper — official Senate financial-disclosure source.

Fetches Periodic Transaction Reports (PTRs) directly from
https://efdsearch.senate.gov, the authoritative source for U.S. Senate
trading disclosures under the STOCK Act.

The site is fronted by Akamai Bot Manager, which 403/503s plain HTTP clients
on the strength of their TLS/HTTP2 fingerprint. We therefore use
``curl_cffi`` with browser impersonation, which reproduces a real Chrome
fingerprint and is accepted. The Django app also requires a CSRF agreement
handshake before the search endpoint will respond.

Senate filings are loaded into an isolated data directory (e.g. ``data/senate``)
so chamber separation is exact without a schema change on the main database.

Known limitation: efdsearch exposes no authoritative amendment/supersession
pointer. Every official report row is preserved under its own source record;
no cross-report amendment or restatement relationship is inferred.
"""

import hashlib
import logging
import random
import re
import time
from dataclasses import dataclass
from datetime import date, timedelta
from numbers import Integral
from pathlib import Path
from typing import Literal
from urllib.parse import urljoin, urlparse

import pandas as pd
from bs4 import BeautifulSoup
from curl_cffi import requests as cffi_requests

from analyzer.database import Database
from analyzer.interfaces import TransactionSource
from analyzer.member_names import chamber_scoped_member_key, canonical_member_key
from analyzer.models import Chamber, ReportOutcome, TickerOrigin
from analyzer.parsing.cells import (
    _extract_amount_midpoint,
    _extract_owner_code,
    _extract_ticker,
)

logger = logging.getLogger(__name__)

EFD_BASE = "https://efdsearch.senate.gov"
BROWSER_IMPERSONATE = "chrome124"
REQUEST_TIMEOUT = 30
SEARCH_PAGE_SIZE = 100
SEARCH_PAGE_DELAY = 0.3
PTR_FETCH_DELAY = 0.2
SEARCH_WINDOW_DAYS = 90
MAX_RETRY_ATTEMPTS = 4
RETRY_BASE_DELAY = 1.0
RETRY_JITTER = 0.3

_LINK_RE = re.compile(r'href="(?P<path>/search/view/ptr/[^"]+)"')
_CSRF_RE = re.compile(r'name="csrfmiddlewaretoken" value="([^"]+)"')
_UUID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I
)

_HEADER_NAME_MAP = {
    "source_row_id": "#",
    "transaction_date": "transaction date",
    "notification_date": "notification date",
    "owner": "owner",
    "ticker": "ticker",
    "asset_name": "asset name",
    "asset_type": "asset type",
    "tx_type": "type",
    "amount": "amount",
}
_REQUIRED_HEADERS = {"transaction_date", "owner", "asset_name", "tx_type", "amount"}
_EQUITY_TICKER_RE = re.compile(r"^[A-Z]{1,5}(?:[.-][A-Z]{1,2})?$")
_NON_EQUITY_RE = re.compile(
    r"(?:RATE\s*/?\s*COUPON|\bBONDS?\b|\bNOTES?\b|\bDEBENTURES?\b|"
    r"\bTREASUR(?:Y|IES)\b|\bMUNICIPAL\b|\bMUNI\b|"
    r"\bFIXED[ -]INCOME\b|\bCERTIFICATES? OF DEPOSIT\b|"
    r"\bCOMMERCIAL PAPER\b|\bCORPORATE DEBT\b|"
    r"\bGOVERNMENT SECURIT(?:Y|IES)\b|\bMORTGAGE[- ]BACKED\b|"
    r"\bPROMISSORY\b|\bGO BDS?\b|\bREV BDS?\b|"
    r"\bMATUR(?:E|ES|ITY|ITIES)\s*:)",
    re.I,
)
_RESERVED_INFERRED_TICKERS = frozenset(
    {"COUPON", "BOND", "BONDS", "NOTE", "NOTES", "STOCK", "TICKER"}
)
_CANONICAL_PTR_PATH_RE = re.compile(
    r"^/search/view/ptr/"
    r"(?P<source_record_id>[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
    r"[0-9a-f]{4}-[0-9a-f]{12})/$"
)
_NORMALIZED_TRANSACTION_COLUMNS = (
    "doc_id",
    "chamber",
    "source_record_id",
    "source_row_id",
    "source_report_path",
    "member",
    "member_key",
    "chamber_member_key",
    "ticker",
    "raw_ticker",
    "ticker_candidate",
    "ticker_origin",
    "transaction_date",
    "disclosure_date",
    "official_filing_date",
    "available_date",
    "notification_date",
    "transaction_type",
    "raw_transaction_subtype",
    "owner_code",
    "raw_owner",
    "amount_raw",
    "amount_midpoint",
    "instrument_type",
    "raw_asset_class",
    "strike_price",
    "expiry_date",
    "asset_description",
    "raw_asset_description",
    "amends_source_record_id",
    "ingestion_generation",
    "artifact_sha256",
)
_PERSISTED_COLUMN_DERIVATIONS = {
    "report_binding": frozenset(
        {
            "doc_id",
            "chamber",
            "source_record_id",
            "source_report_path",
            "member",
            "member_key",
            "chamber_member_key",
            "disclosure_date",
            "official_filing_date",
            "available_date",
            "ingestion_generation",
            "artifact_sha256",
        }
    ),
    "official_row": frozenset(
        {
            "source_row_id",
            "raw_ticker",
            "transaction_date",
            "notification_date",
            "raw_transaction_subtype",
            "raw_owner",
            "amount_raw",
            "raw_asset_class",
            "raw_asset_description",
        }
    ),
    "raw_derivation": frozenset(
        {
            "ticker",
            "ticker_candidate",
            "ticker_origin",
            "transaction_type",
            "owner_code",
            "amount_midpoint",
            "instrument_type",
            "asset_description",
        }
    ),
    "unsupported_nullable": frozenset(
        {"strike_price", "expiry_date", "amends_source_record_id"}
    ),
}
_DERIVATION_COLUMNS = [
    column for columns in _PERSISTED_COLUMN_DERIVATIONS.values() for column in columns
]
assert len(_DERIVATION_COLUMNS) == len(set(_DERIVATION_COLUMNS))
assert set(_DERIVATION_COLUMNS) == set(_NORMALIZED_TRANSACTION_COLUMNS)

_PAPER_ARTIFACT_RE = re.compile(
    r"(?:/search/view/(?:paper|paper-filing)/|\.pdf(?:$|[?#]))", re.I
)


@dataclass(frozen=True, slots=True)
class SenateReportFetchResult:
    outcome: ReportOutcome
    transactions: tuple[dict, ...] = ()
    landing_sha256: str | None = None
    paper_artifact_sha256: str | None = None
    paper_artifact_url: str | None = None
    error_message: str | None = None

    @property
    def transaction_artifact_sha256(self) -> str | None:
        return self.landing_sha256


@dataclass(frozen=True, slots=True)
class SenateRefreshSummary:
    found: int
    parsed: int
    paper_only: int
    unavailable: int
    failed: int

    def __post_init__(self) -> None:
        accounted = self.parsed + self.paper_only + self.unavailable + self.failed
        if self.found != accounted:
            raise ValueError(
                f"Senate report accounting mismatch: found={self.found}, "
                f"accounted={accounted}"
            )

    @property
    def complete(self) -> bool:
        return self.failed == 0 and self.unavailable == 0

    def require_complete(self) -> None:
        if not self.complete:
            raise SenateEFDError(
                "Senate refresh incomplete: "
                f"found={self.found}, parsed={self.parsed}, "
                f"paper_only={self.paper_only}, unavailable={self.unavailable}, "
                f"failed={self.failed}"
            )


class SenateEFDError(Exception):
    pass


class SenateRowValidationError(SenateEFDError):
    def __init__(self, raw_count: int, accepted_count: int, rejected_count: int):
        self.raw_count = raw_count
        self.accepted_count = accepted_count
        self.rejected_count = rejected_count
        super().__init__(
            "Malformed Senate report rows: "
            f"raw={raw_count}, accepted={accepted_count}, rejected={rejected_count}"
        )


class SenateEFDBlockedError(SenateEFDError):
    """Raised when efdsearch blocks the client or the backend fails."""


class SenateEFDSource(TransactionSource):
    """Scrapes official Senate PTR filings from efdsearch.senate.gov."""

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
        self._session: cffi_requests.Session | None = None
        self._csrf_token: str | None = None
        self.ingestion_generation = ingestion_generation
        self.report_inventory: list[dict] = []
        self.last_refresh_summary: SenateRefreshSummary | None = None

    def close(self) -> None:
        if self._session is not None:
            self._session.close()
            self._session = None
        if self._owns_db:
            self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        return False

    def get_transactions(self, year: int) -> pd.DataFrame:
        df = self.db.get_transactions(year)
        if df.empty:
            raise SenateEFDError(
                f"No cached Senate eFD data for {year}. "
                "Run 'ptr-alpha fetch-senate-efd' first."
            )
        logger.info("Loaded %d cached Senate eFD transactions for %d", len(df), year)
        return df

    def _request_with_retry(
        self,
        method: Literal["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "TRACE"],
        url: str,
        *,
        attempts: int = MAX_RETRY_ATTEMPTS,
        **kwargs,
    ) -> cffi_requests.Response:
        """Bounded retries with exponential backoff + jitter.

        - 429: honors Retry-After (or backoff), then raises on exhaustion.
        - 403/401: refreshes the CSRF session once, then raises if still blocked.
        - 5xx/521: retries with backoff, then raises.
        - Other codes (e.g. 404): returned to the caller.
        """
        session = self._require_session()
        refreshed = False
        for attempt in range(1, attempts + 1):
            try:
                resp = session.request(method, url, timeout=REQUEST_TIMEOUT, **kwargs)
            except cffi_requests.RequestsError as e:
                if attempt == attempts:
                    raise SenateEFDBlockedError(
                        f"eFD request failed after {attempts} attempts: {e}"
                    ) from e
                delay = RETRY_BASE_DELAY * (2 ** (attempt - 1)) + random.uniform(
                    0, RETRY_JITTER
                )  # nosec B311
                logger.warning("eFD request error (%s); retrying in %.1fs", e, delay)
                time.sleep(delay)
                continue

            if resp.status_code == 200:
                return resp

            if resp.status_code == 429:
                if attempt == attempts:
                    raise SenateEFDBlockedError(
                        "eFD rate-limited (HTTP 429) after retries"
                    )
                retry_after = resp.headers.get("Retry-After")
                try:
                    delay = (
                        float(retry_after)
                        if retry_after
                        else RETRY_BASE_DELAY * (2 ** (attempt - 1))
                    )
                except ValueError:
                    delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                logger.warning("eFD HTTP 429; sleeping %.1fs", delay)
                time.sleep(delay)
                continue

            if resp.status_code in (401, 403):
                if not refreshed and attempt < attempts:
                    logger.warning(
                        "eFD auth/block response HTTP %d; refreshing session",
                        resp.status_code,
                    )
                    self._open_session()
                    session = self._require_session()
                    refreshed = True
                    time.sleep(RETRY_BASE_DELAY)
                    continue
                raise SenateEFDBlockedError(
                    f"eFD blocked (HTTP {resp.status_code}); session refresh did not help"
                )

            if resp.status_code in (500, 502, 503, 504, 521):
                if attempt == attempts:
                    raise SenateEFDBlockedError(
                        f"eFD server error HTTP {resp.status_code} after {attempts} attempts"
                    )
                delay = RETRY_BASE_DELAY * (2 ** (attempt - 1)) + random.uniform(
                    0, RETRY_JITTER
                )  # nosec B311
                logger.warning(
                    "eFD HTTP %d; retrying in %.1fs", resp.status_code, delay
                )
                time.sleep(delay)
                continue

            return resp
        raise SenateEFDBlockedError("unreachable retry state")

    def _open_session(self) -> str:
        """Perform the CSRF agreement handshake. Return the CSRF cookie token."""
        if self._session is not None:
            try:
                self._session.close()
            except Exception:  # noqa: B110  # nosec B110
                pass
        session = cffi_requests.Session(impersonate=BROWSER_IMPERSONATE)
        self._session = session
        resp = session.get(f"{EFD_BASE}/search/", timeout=REQUEST_TIMEOUT)
        if resp.status_code != 200:
            raise SenateEFDBlockedError(
                f"eFD search page returned HTTP {resp.status_code}"
            )
        m = _CSRF_RE.search(resp.text)
        if not m:
            raise SenateEFDError(
                "csrfmiddlewaretoken not found; eFD layout may have changed"
            )
        form_token = m.group(1)

        agree = session.post(
            f"{EFD_BASE}/search/home/",
            data={"csrfmiddlewaretoken": form_token, "prohibition_agreement": "1"},
            headers={"Referer": f"{EFD_BASE}/search/home/", "Origin": EFD_BASE},
            timeout=REQUEST_TIMEOUT,
        )
        if agree.status_code != 200:
            raise SenateEFDBlockedError(
                f"eFD agreement POST returned HTTP {agree.status_code}"
            )

        csrf_cookie = session.cookies.get("csrftoken")
        if not csrf_cookie:
            raise SenateEFDError("eFD session did not return a csrftoken cookie")
        self._csrf_token = csrf_cookie
        logger.info("Opened eFD session (CSRF handshake complete)")
        return csrf_cookie

    def _require_session(self) -> cffi_requests.Session:
        if self._session is None:
            self._open_session()
        assert self._session is not None  # nosec B101
        return self._session

    def _search_reports(self, start_date: date, end_date: date) -> list[dict]:
        """Paginate the PTR search over immutable 90-day windows.

        Windowed queries stop live insertions from shifting later pages.
        Report paths are de-duplicated across pages and windows so a filing
        is fetched at most once.
        """
        reports: dict[str, dict] = {}
        cursor = start_date
        while cursor <= end_date:
            win_end = min(cursor + timedelta(days=SEARCH_WINDOW_DAYS - 1), end_date)
            self._search_window(cursor, win_end, reports)
            cursor = win_end + timedelta(days=1)
        logger.info(
            "eFD search found %d unique PTR filings (%s..%s)",
            len(reports),
            start_date,
            end_date,
        )
        return list(reports.values())

    def _search_window(
        self, start_date: date, end_date: date, reports: dict[str, dict]
    ) -> None:
        start_offset = 0
        before = len(reports)
        stale_pages = 0
        while True:
            payload = {
                "report_types": "[11]",
                "filer_types": "[]",
                "submitted_start_date": start_date.strftime("%m/%d/%Y 00:00:00"),
                "submitted_end_date": end_date.strftime("%m/%d/%Y 23:59:59"),
                "candidate_state": "",
                "senator_state": "",
                "office_id": "",
                "first_name": "",
                "last_name": "",
                "draw": "1",
                "start": str(start_offset),
                "length": str(SEARCH_PAGE_SIZE),
            }
            headers = {
                "Referer": f"{EFD_BASE}/search/",
                "X-Requested-With": "XMLHttpRequest",
                "X-CSRFToken": self._csrf_token,
            }
            resp = self._request_with_retry(
                "POST", f"{EFD_BASE}/search/report/data/", data=payload, headers=headers
            )
            try:
                data = resp.json()
            except Exception as e:
                raise SenateEFDError(
                    f"eFD search returned non-JSON (HTTP {resp.status_code})"
                ) from e
            rows = data.get("data", [])
            for row in rows:
                first_name, last_name, display, link_html, filed_date = row[:5]
                link_m = _LINK_RE.search(str(link_html))
                if not link_m:
                    continue
                if "candidate" in str(display).lower():
                    continue
                path = link_m.group("path")
                if path in reports:
                    continue
                name = re.sub(r"\s+", " ", f"{first_name} {last_name}".strip()).strip(
                    " ,"
                )
                reports[path] = {
                    "senator": name,
                    "report_path": path,
                    "filed_date": self._parse_date(filed_date),
                }
            start_offset += SEARCH_PAGE_SIZE
            if start_offset >= data.get("recordsFiltered", 0):
                break
            if len(reports) == before:
                stale_pages += 1
                if stale_pages >= 3:
                    logger.warning(
                        "eFD window %s..%s pagination stopped after %d stable pages",
                        start_date,
                        end_date,
                        stale_pages,
                    )
                    break
            else:
                before = len(reports)
                stale_pages = 0
            time.sleep(SEARCH_PAGE_DELAY)

    @staticmethod
    def _is_non_equity_asset(asset_name: str, asset_type: str) -> bool:
        normalized_class = SenateEFDSource._normalize_instrument_type(asset_type)
        if normalized_class == "bond":
            return True
        return bool(_NON_EQUITY_RE.search(asset_name))

    @classmethod
    def _resolve_ticker(
        cls, ticker_raw: str, asset_name: str, asset_type: str
    ) -> tuple[str | None, str | None, TickerOrigin]:
        raw = str(ticker_raw or "").strip().upper()
        if raw and raw != "--":
            if not _EQUITY_TICKER_RE.fullmatch(raw):
                return None, None, TickerOrigin.INVALID
            return raw, None, TickerOrigin.OFFICIAL

        if cls._is_non_equity_asset(asset_name, asset_type):
            return None, None, TickerOrigin.NON_EQUITY

        inferred = _extract_ticker(asset_name)
        if not inferred:
            return None, None, TickerOrigin.MISSING
        inferred = inferred.strip().upper()
        if inferred in _RESERVED_INFERRED_TICKERS:
            return None, inferred, TickerOrigin.INVALID
        if not _EQUITY_TICKER_RE.fullmatch(inferred):
            return None, inferred, TickerOrigin.INVALID

        asset_class = cls._normalize_instrument_type(asset_type)
        if asset_class in {"stock", "fund", "call", "put"}:
            return inferred, None, TickerOrigin.ASSET_DESCRIPTION
        return None, inferred, TickerOrigin.UNVERIFIED

    @staticmethod
    def _official_url_matches(actual_url: str, expected_url: str) -> bool:
        try:
            actual = urlparse(str(actual_url))
            expected = urlparse(expected_url)
            actual_port = actual.port
        except (TypeError, ValueError):
            return False
        return (
            actual.scheme == "https"
            and actual.hostname == "efdsearch.senate.gov"
            and actual.username is None
            and actual.password is None
            and actual_port in {None, 443}
            and actual.path.rstrip("/") == expected.path.rstrip("/")
        )

    @staticmethod
    def _is_allowlisted_paper_url(url: str) -> bool:
        try:
            parsed = urlparse(url)
            parsed_port = parsed.port
        except (TypeError, ValueError):
            return False
        return (
            parsed.scheme == "https"
            and parsed.hostname == "efdsearch.senate.gov"
            and parsed.username is None
            and parsed.password is None
            and parsed_port in {None, 443}
            and (
                parsed.path.startswith("/media/")
                or parsed.path.startswith("/search/view/paper/")
                or parsed.path.startswith("/search/view/paper-filing/")
            )
        )

    @staticmethod
    def _paper_artifact_url(soup: BeautifulSoup) -> str | None:
        for link in soup.find_all("a", href=True):
            href = str(link.get("href") or "").strip()
            if not _PAPER_ARTIFACT_RE.search(href):
                continue
            absolute = urljoin(EFD_BASE, href)
            if SenateEFDSource._is_allowlisted_paper_url(absolute):
                return absolute
        return None

    def _fetch_paper_artifact(
        self, paper_url: str
    ) -> tuple[str | None, str | None, str | None]:
        response = self._request_with_retry("GET", paper_url)
        final_url = str(response.url)
        if not self._official_url_matches(final_url, paper_url):
            return None, None, "paper artifact redirected outside its official path"
        if not self._is_allowlisted_paper_url(final_url):
            return None, None, "paper artifact final URL is not allowlisted"
        if response.status_code != 200:
            return None, None, f"paper artifact returned HTTP {response.status_code}"
        content = response.content
        if not content.startswith(b"%PDF-"):
            return None, None, "paper artifact response is not a PDF"
        return hashlib.sha256(content).hexdigest(), final_url, None

    def _fetch_report_transactions(self, report_path: str) -> SenateReportFetchResult:
        """GET one PTR detail and return a classified, hashed parse result."""
        expected_url = f"{EFD_BASE}{report_path}"
        resp = self._request_with_retry("GET", expected_url)
        landing_sha256 = hashlib.sha256(resp.content).hexdigest()
        if not self._official_url_matches(resp.url, expected_url):
            return SenateReportFetchResult(
                outcome=ReportOutcome.FAILED,
                landing_sha256=landing_sha256,
                error_message=(
                    f"eFD filing {report_path} redirected outside its official path"
                ),
            )
        if resp.status_code in (404, 410):
            logger.warning(
                "eFD filing %s unavailable (HTTP %d)",
                report_path,
                resp.status_code,
            )
            return SenateReportFetchResult(
                outcome=ReportOutcome.UNAVAILABLE,
                landing_sha256=landing_sha256,
            )
        if resp.status_code != 200:
            return SenateReportFetchResult(
                outcome=ReportOutcome.FAILED,
                landing_sha256=landing_sha256,
                error_message=(
                    f"eFD filing {report_path} returned HTTP {resp.status_code}"
                ),
            )

        soup = BeautifulSoup(resp.text, "html.parser")
        title = soup.title.get_text(" ", strip=True) if soup.title else ""
        if "periodic transaction report" not in title.lower():
            return SenateReportFetchResult(
                outcome=ReportOutcome.FAILED,
                landing_sha256=landing_sha256,
                error_message=(
                    f"eFD filing {report_path} returned an unrecognized page: "
                    f"title={title!r}"
                ),
            )

        table = soup.find("table", {"class": "table-striped"})
        if table is None:
            paper_artifact_url = self._paper_artifact_url(soup)
            if not paper_artifact_url:
                return SenateReportFetchResult(
                    outcome=ReportOutcome.FAILED,
                    landing_sha256=landing_sha256,
                    error_message=(
                        f"PTR {report_path} has no transaction table or "
                        "allowlisted paper artifact link"
                    ),
                )
            paper_sha256, final_paper_url, paper_error = self._fetch_paper_artifact(
                paper_artifact_url
            )
            if paper_error:
                return SenateReportFetchResult(
                    outcome=ReportOutcome.FAILED,
                    landing_sha256=landing_sha256,
                    error_message=paper_error,
                )
            return SenateReportFetchResult(
                outcome=ReportOutcome.PAPER_ONLY,
                landing_sha256=landing_sha256,
                paper_artifact_sha256=paper_sha256,
                paper_artifact_url=final_paper_url,
            )

        thead = table.find("thead")
        tbody = table.find("tbody")
        if thead is None or tbody is None:
            return SenateReportFetchResult(
                outcome=ReportOutcome.FAILED,
                landing_sha256=landing_sha256,
                error_message=f"PTR {report_path} has no parseable header/body",
            )
        headers = [
            th.get_text(strip=True).lower() for th in thead.find_all(["th", "td"])
        ]
        col: dict[str, int] = {}
        for key, label in _HEADER_NAME_MAP.items():
            if label in headers:
                col[key] = headers.index(label)
        missing = _REQUIRED_HEADERS - set(col)
        if missing:
            return SenateReportFetchResult(
                outcome=ReportOutcome.FAILED,
                landing_sha256=landing_sha256,
                error_message=(
                    f"PTR {report_path} table missing columns {sorted(missing)}; "
                    f"headers={headers}"
                ),
            )

        out: list[dict] = []
        for row_position, tr in enumerate(tbody.find_all("tr"), start=1):
            cells = [td.get_text(strip=True) for td in tr.find_all("td")]
            if not cells:
                continue

            def cell(key: str) -> str:
                idx = col.get(key)
                if idx is None or idx >= len(cells):
                    return ""
                return cells[idx]

            asset_name = cell("asset_name")
            asset_type = cell("asset_type")
            ticker_raw = cell("ticker")
            ticker, ticker_candidate, ticker_origin = self._resolve_ticker(
                ticker_raw, asset_name, asset_type
            )
            owner_raw = cell("owner")
            tx_subtype_raw = cell("tx_type")
            amount_raw = cell("amount")
            official_row_id = cell("source_row_id").strip()
            source_row_id = (
                f"official:{official_row_id}"
                if official_row_id
                else f"table:{row_position:06d}"
            )
            out.append(
                {
                    "source_row_id": source_row_id,
                    "ticker": ticker,
                    "ticker_raw": ticker_raw or None,
                    "ticker_candidate": ticker_candidate,
                    "ticker_origin": ticker_origin.value,
                    "asset_name": asset_name,
                    "asset_type": asset_type,
                    "owner": _extract_owner_code(owner_raw),
                    "owner_raw": owner_raw or None,
                    "type": tx_subtype_raw,
                    "transaction_subtype_raw": tx_subtype_raw or None,
                    "transaction_date": cell("transaction_date"),
                    "notification_date": cell("notification_date") or None,
                    "amount_range": amount_raw,
                    "amount_range_raw": amount_raw or None,
                }
            )

        if not out:
            return SenateReportFetchResult(
                outcome=ReportOutcome.FAILED,
                landing_sha256=landing_sha256,
                error_message=f"PTR {report_path} transaction table is empty",
            )
        source_row_ids = [str(row.get("source_row_id") or "").strip() for row in out]
        if any(not source_row_id for source_row_id in source_row_ids) or len(
            source_row_ids
        ) != len(set(source_row_ids)):
            return SenateReportFetchResult(
                outcome=ReportOutcome.FAILED,
                transactions=tuple(out),
                landing_sha256=landing_sha256,
                error_message=(
                    f"PTR {report_path} has blank or duplicate source row IDs"
                ),
            )
        return SenateReportFetchResult(
            outcome=ReportOutcome.PARSED,
            transactions=tuple(out),
            landing_sha256=landing_sha256,
        )

    def fetch_all_trades(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
        chamber: str | None = "senate",
    ) -> pd.DataFrame:
        """Fetch all Senate PTR transactions in the date range.

        ``chamber`` is accepted for interface symmetry but the source is
        Senate-only; any non-None value other than 'senate' raises.
        """
        if chamber is not None and chamber.lower() != "senate":
            raise SenateEFDError(
                f"SenateEFDSource is senate-only; got chamber={chamber!r}"
            )
        end_date = end_date or date.today()
        start_date = start_date or self._one_year_before(end_date)

        self._open_session()
        reports = self._search_reports(start_date, end_date)
        normalized_reports: list[pd.DataFrame] = []
        counts = {outcome: 0 for outcome in ReportOutcome}
        self.report_inventory = []
        for report in reports:
            doc_id = self._path_to_doc_id(report["report_path"])
            source_record_id = self._path_to_source_record_id(report["report_path"])
            try:
                result = self._fetch_report_transactions(report["report_path"])
            except SenateEFDBlockedError:
                raise
            except SenateEFDError as exc:
                result = SenateReportFetchResult(
                    outcome=ReportOutcome.FAILED,
                    error_message=str(exc),
                )
                logger.error("Failed to fetch %s: %s", report["report_path"], exc)

            raw_row_count = len(result.transactions)
            accepted_row_count = 0
            rejected_row_count = (
                raw_row_count if result.outcome is ReportOutcome.FAILED else 0
            )
            if result.outcome is ReportOutcome.PARSED:
                report_rows = [
                    {
                        "doc_id": doc_id,
                        "source_record_id": source_record_id,
                        "source_report_path": report["report_path"],
                        "senator": report["senator"],
                        "filed_date": report["filed_date"],
                        "official_filing_date": report["filed_date"],
                        "available_date": report["filed_date"],
                        "amends_source_record_id": None,
                        "artifact_sha256": result.transaction_artifact_sha256,
                        "ingestion_generation": self.ingestion_generation,
                        **transaction,
                    }
                    for transaction in result.transactions
                ]
                try:
                    normalized_report = self._normalize(report_rows)
                except SenateRowValidationError as exc:
                    accepted_row_count = exc.accepted_count
                    rejected_row_count = exc.rejected_count
                    result = SenateReportFetchResult(
                        outcome=ReportOutcome.FAILED,
                        landing_sha256=result.landing_sha256,
                        paper_artifact_sha256=result.paper_artifact_sha256,
                        paper_artifact_url=result.paper_artifact_url,
                        error_message=str(exc),
                    )
                else:
                    accepted_row_count = len(normalized_report)
                    normalized_reports.append(normalized_report)

            counts[result.outcome] += 1
            self.report_inventory.append(
                {
                    "chamber": Chamber.SENATE.value,
                    "source_record_id": source_record_id,
                    "report_path": report["report_path"],
                    "member": report["senator"],
                    "official_filing_date": report["filed_date"],
                    "outcome": result.outcome.value,
                    "artifact_sha256": result.transaction_artifact_sha256,
                    "landing_sha256": result.landing_sha256,
                    "paper_artifact_sha256": result.paper_artifact_sha256,
                    "paper_artifact_url": result.paper_artifact_url,
                    "error_message": result.error_message,
                    "raw_row_count": raw_row_count,
                    "accepted_row_count": accepted_row_count,
                    "rejected_row_count": rejected_row_count,
                    "ingestion_generation": self.ingestion_generation,
                }
            )
            time.sleep(PTR_FETCH_DELAY)

        summary = SenateRefreshSummary(
            found=len(reports),
            parsed=counts[ReportOutcome.PARSED],
            paper_only=counts[ReportOutcome.PAPER_ONLY],
            unavailable=counts[ReportOutcome.UNAVAILABLE],
            failed=counts[ReportOutcome.FAILED],
        )
        self.last_refresh_summary = summary
        summary.require_complete()

        normalized = (
            pd.concat(normalized_reports, ignore_index=True)
            if normalized_reports
            else self._normalize([])
        )
        logger.info(
            "Fetched %d Senate transactions from %d filings "
            "(%d parsed, %d paper-only, %d unavailable)",
            len(normalized),
            summary.found,
            summary.parsed,
            summary.paper_only,
            summary.unavailable,
        )
        normalized.attrs["refresh_summary"] = {
            "found": summary.found,
            "parsed": summary.parsed,
            "paper_only": summary.paper_only,
            "unavailable": summary.unavailable,
            "failed": summary.failed,
        }
        normalized.attrs["report_inventory"] = list(self.report_inventory)
        return normalized

    @staticmethod
    def _path_to_source_record_id(report_path: str) -> str:
        match = _UUID_RE.search(report_path)
        return match.group(0) if match else report_path

    @staticmethod
    def _path_to_doc_id(report_path: str) -> str:
        match = _UUID_RE.search(report_path)
        if match:
            return match.group(0)
        return (
            "efd-"
            + hashlib.sha1(report_path.encode(), usedforsecurity=False).hexdigest()[:16]
        )

    @staticmethod
    def _one_year_before(d: date) -> date:
        try:
            return d.replace(year=d.year - 1)
        except ValueError:
            return d.replace(year=d.year - 1, day=28)

    @staticmethod
    def _dates_are_valid(
        transaction_date: pd.Timestamp | None,
        official_filing_date: pd.Timestamp | None,
        available_date: pd.Timestamp | None,
        notification_date: pd.Timestamp | None,
    ) -> bool:
        required = (transaction_date, official_filing_date, available_date)
        if any(value is None or not pd.notna(value) for value in required):
            return False
        minimum = pd.Timestamp("1900-01-01")
        maximum = pd.Timestamp(date.today() + timedelta(days=1))
        if any(value < minimum or value > maximum for value in required):
            return False
        if transaction_date > available_date:
            return False
        if available_date != official_filing_date:
            return False
        if notification_date is None:
            return True
        if not pd.notna(notification_date):
            return False
        return transaction_date <= notification_date <= official_filing_date

    def _normalize(self, trades: list[dict]) -> pd.DataFrame:
        columns = list(_NORMALIZED_TRANSACTION_COLUMNS)
        if not trades:
            return pd.DataFrame(columns=columns)

        rows = []
        rejected = 0
        for transaction in trades:
            amount_raw, midpoint = _extract_amount_midpoint(
                transaction.get("amount_range")
            )
            member = transaction.get("senator")
            available_date = self._parse_date(transaction.get("available_date"))
            official_filing_date = self._parse_date(
                transaction.get("official_filing_date")
            )
            row = {
                "doc_id": transaction.get("doc_id"),
                "chamber": Chamber.SENATE.value,
                "source_record_id": transaction.get("source_record_id"),
                "source_row_id": transaction.get("source_row_id"),
                "source_report_path": transaction.get("source_report_path"),
                "member": member,
                "member_key": canonical_member_key(member),
                "chamber_member_key": chamber_scoped_member_key(
                    member, Chamber.SENATE.value
                ),
                "ticker": transaction.get("ticker"),
                "raw_ticker": transaction.get("ticker_raw"),
                "ticker_candidate": transaction.get("ticker_candidate"),
                "ticker_origin": transaction.get("ticker_origin"),
                "transaction_date": self._parse_date(
                    transaction.get("transaction_date")
                ),
                "disclosure_date": available_date,
                "official_filing_date": official_filing_date,
                "available_date": available_date,
                "notification_date": self._parse_date(
                    transaction.get("notification_date")
                ),
                "transaction_type": self._normalize_tx_type(transaction.get("type")),
                "raw_transaction_subtype": transaction.get("transaction_subtype_raw"),
                "owner_code": transaction.get("owner"),
                "raw_owner": transaction.get("owner_raw"),
                "amount_raw": amount_raw or transaction.get("amount_range_raw"),
                "amount_midpoint": midpoint,
                "instrument_type": self._normalize_instrument_type(
                    transaction.get("asset_type")
                ),
                "raw_asset_class": transaction.get("asset_type"),
                "strike_price": None,
                "expiry_date": None,
                "asset_description": transaction.get("asset_name"),
                "raw_asset_description": transaction.get("asset_name"),
                "amends_source_record_id": transaction.get("amends_source_record_id"),
                "ingestion_generation": transaction.get("ingestion_generation"),
                "artifact_sha256": transaction.get("artifact_sha256"),
            }
            if (
                not row["doc_id"]
                or not str(row["source_record_id"] or "").strip()
                or not row["member"]
                or not str(row["source_row_id"] or "").strip()
                or not self._dates_are_valid(
                    row["transaction_date"],
                    row["official_filing_date"],
                    row["available_date"],
                    row["notification_date"],
                )
                or row["transaction_type"] not in {"Purchase", "Sale", "Exchange"}
                or not row["raw_transaction_subtype"]
                or not row["amount_raw"]
                or not row["raw_asset_description"]
                or not row["ticker_origin"]
                or (
                    row["ticker_origin"] == TickerOrigin.UNVERIFIED.value
                    and (row["ticker"] is not None or not row["ticker_candidate"])
                )
                or not row["artifact_sha256"]
            ):
                rejected += 1
                continue
            rows.append(row)

        frame = pd.DataFrame(rows, columns=columns)
        if not frame.empty:
            duplicate_mask = frame.duplicated(
                subset=["source_record_id", "source_row_id"], keep=False
            )
            duplicate_count = int(duplicate_mask.sum())
            if duplicate_count:
                rejected += duplicate_count
                frame = frame.loc[~duplicate_mask].copy()
        if rejected:
            raise SenateRowValidationError(
                raw_count=len(trades),
                accepted_count=len(frame),
                rejected_count=rejected,
            )
        return frame

    @staticmethod
    def _parse_date(value):
        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        try:
            if not pd.notna(value):
                return None
        except (TypeError, ValueError):
            return None
        try:
            parsed = pd.Timestamp(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if not pd.notna(parsed):
            return None
        if parsed.tzinfo is not None:
            parsed = parsed.tz_convert("UTC").tz_localize(None)
        return parsed.normalize()

    @staticmethod
    def _normalize_tx_type(raw):
        if not raw:
            return raw
        lower = raw.lower().strip()
        if lower == "purchase" or lower.startswith("purchase"):
            return "Purchase"
        if lower.startswith("sale"):
            return "Sale"
        if lower.startswith("exchange"):
            return "Exchange"
        return raw.strip().title()

    @staticmethod
    def _normalize_instrument_type(raw):
        if not raw:
            return "unknown"
        lower = raw.lower()
        if any(
            term in lower
            for term in (
                "bond",
                "note",
                "treasury",
                "municipal",
                "debt",
                "fixed income",
                "commercial paper",
                "certificate of deposit",
                "government security",
            )
        ):
            return "bond"
        if "call" in lower:
            return "call"
        if "put" in lower:
            return "put"
        if "fund" in lower or "etf" in lower:
            return "fund"
        if "stock" in lower or "equity" in lower:
            return "stock"
        return "other"

    @staticmethod
    def _provenance_text(value) -> str | None:
        if value is None:
            return None
        try:
            if not pd.notna(value):
                return None
        except (TypeError, ValueError):
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _is_sha256(value) -> bool:
        text = SenateEFDSource._provenance_text(value)
        return bool(text and re.fullmatch(r"[0-9a-f]{64}", text))

    @classmethod
    def _validate_ticker_provenance(cls, df: pd.DataFrame) -> None:
        for row in df[["ticker", "ticker_candidate", "ticker_origin"]].itertuples(
            index=False
        ):
            ticker = cls._provenance_text(row.ticker)
            candidate = cls._provenance_text(row.ticker_candidate)
            try:
                origin = TickerOrigin(row.ticker_origin)
            except (TypeError, ValueError):
                raise SenateEFDError(
                    f"Unknown Senate ticker_origin: {row.ticker_origin!r}"
                ) from None

            ticker_is_valid = bool(ticker and _EQUITY_TICKER_RE.fullmatch(ticker))
            candidate_is_valid = bool(
                candidate and _EQUITY_TICKER_RE.fullmatch(candidate)
            )
            if origin in {
                TickerOrigin.OFFICIAL,
                TickerOrigin.ASSET_DESCRIPTION,
            }:
                if (
                    not ticker_is_valid
                    or candidate is not None
                    or (
                        origin is TickerOrigin.ASSET_DESCRIPTION
                        and ticker in _RESERVED_INFERRED_TICKERS
                    )
                ):
                    raise SenateEFDError(
                        f"Invalid Senate {origin.value} ticker provenance"
                    )
                continue
            if origin is TickerOrigin.UNVERIFIED:
                if (
                    ticker is not None
                    or not candidate_is_valid
                    or candidate in _RESERVED_INFERRED_TICKERS
                ):
                    raise SenateEFDError(
                        "Unverified Senate ticker requires only a valid candidate"
                    )
                continue
            if origin in {TickerOrigin.NON_EQUITY, TickerOrigin.MISSING}:
                if ticker is not None or candidate is not None:
                    raise SenateEFDError(
                        f"Senate {origin.value} ticker provenance must be empty"
                    )
                continue
            if origin is TickerOrigin.INVALID:
                if ticker is not None or (
                    candidate_is_valid and candidate not in _RESERVED_INFERRED_TICKERS
                ):
                    raise SenateEFDError(
                        "Invalid Senate ticker provenance is inconsistent"
                    )
                # Candidate is NULL for a rejected raw_ticker. Otherwise it is
                # retained only when syntax or a reserved token made it invalid.
                continue
            raise SenateEFDError(f"Unsupported Senate ticker_origin: {origin.value}")

    def save_to_db(self, df: pd.DataFrame) -> int:
        persist_refresh = getattr(self.db, "persist_source_refresh", None)
        if not callable(persist_refresh):
            raise SenateEFDError(
                "Database does not implement atomic persist_source_refresh; "
                "refusing to drop Senate provenance or report inventory"
            )
        if not self.ingestion_generation:
            raise SenateEFDError(
                "ingestion_generation is required for Senate refresh persistence"
            )
        if self.last_refresh_summary is None or not self.last_refresh_summary.complete:
            raise SenateEFDError(
                "A complete Senate report inventory is required before persistence"
            )
        if len(self.report_inventory) != self.last_refresh_summary.found:
            raise SenateEFDError(
                "Senate report inventory count does not match refresh summary"
            )

        required_columns = set(_NORMALIZED_TRANSACTION_COLUMNS)
        missing = required_columns - set(df.columns)
        if missing:
            raise SenateEFDError(
                f"Senate transaction provenance columns missing: {sorted(missing)}"
            )

        required_values = [
            "doc_id",
            "chamber",
            "source_record_id",
            "source_row_id",
            "source_report_path",
            "member",
            "official_filing_date",
            "available_date",
            "amount_raw",
            "raw_transaction_subtype",
            "ticker_origin",
            "raw_asset_description",
            "ingestion_generation",
            "artifact_sha256",
        ]
        if not df.empty and df[required_values].isna().any().any():
            raise SenateEFDError("Senate transaction provenance values are incomplete")
        nonblank_columns = [
            "doc_id",
            "source_record_id",
            "source_row_id",
            "source_report_path",
            "member",
            "amount_raw",
            "raw_transaction_subtype",
            "raw_asset_description",
            "ingestion_generation",
            "artifact_sha256",
        ]
        if not df.empty and any(
            df[column].map(self._provenance_text).isna().any()
            for column in nonblank_columns
        ):
            raise SenateEFDError(
                "Senate transaction provenance text values must be nonblank"
            )
        if (
            not df.empty
            and df.duplicated(
                subset=["source_record_id", "source_row_id"], keep=False
            ).any()
        ):
            raise SenateEFDError(
                "Senate source_record_id/source_row_id values must be unique"
            )
        for row in df[
            [
                "transaction_date",
                "official_filing_date",
                "available_date",
                "notification_date",
            ]
        ].itertuples(index=False):
            if not self._dates_are_valid(
                self._parse_date(row.transaction_date),
                self._parse_date(row.official_filing_date),
                self._parse_date(row.available_date),
                self._parse_date(row.notification_date),
            ):
                raise SenateEFDError(
                    "Senate transaction dates are incomplete or invalid"
                )
        self._validate_ticker_provenance(df)
        if not df.empty and not df["artifact_sha256"].map(self._is_sha256).all():
            raise SenateEFDError("Senate transaction artifact hashes are invalid")
        if not df.empty and (
            not df["chamber"].eq(Chamber.SENATE.value).all()
            or not df["ingestion_generation"].eq(self.ingestion_generation).all()
        ):
            raise SenateEFDError(
                "Senate transaction chamber/generation does not match refresh"
            )

        required_report_fields = {
            "chamber",
            "source_record_id",
            "report_path",
            "member",
            "official_filing_date",
            "outcome",
            "artifact_sha256",
            "landing_sha256",
            "paper_artifact_sha256",
            "paper_artifact_url",
            "error_message",
            "raw_row_count",
            "accepted_row_count",
            "rejected_row_count",
            "ingestion_generation",
        }
        report_hashes: dict[str, str] = {}
        report_bindings: dict[str, tuple[str, str, pd.Timestamp, ReportOutcome]] = {}
        outcome_counts = {outcome.value: 0 for outcome in ReportOutcome}
        transaction_counts = df.groupby("source_record_id").size().to_dict()
        for report in self.report_inventory:
            missing_report_fields = required_report_fields - set(report)
            if missing_report_fields:
                raise SenateEFDError(
                    "Senate report inventory fields missing: "
                    f"{sorted(missing_report_fields)}"
                )
            source_record_id = self._provenance_text(report["source_record_id"])
            if (
                source_record_id is None
                or report["source_record_id"] != source_record_id
            ):
                raise SenateEFDError("Senate report inventory has no source_record_id")
            if source_record_id in report_hashes:
                raise SenateEFDError(
                    f"Duplicate Senate report inventory ID: {source_record_id}"
                )
            report_path = self._provenance_text(report["report_path"])
            if report_path is None or report["report_path"] != report_path:
                raise SenateEFDError(
                    f"Senate report path/record ID mismatch: {source_record_id}"
                )
            path_match = _CANONICAL_PTR_PATH_RE.fullmatch(report_path)
            if (
                path_match is None
                or path_match.group("source_record_id") != source_record_id
            ):
                raise SenateEFDError(
                    f"Senate report path/record ID mismatch: {source_record_id}"
                )
            report_member = self._provenance_text(report["member"])
            if report_member is None:
                raise SenateEFDError(
                    f"Senate report member is blank: {source_record_id}"
                )
            filing_date = self._parse_date(report["official_filing_date"])
            if not self._dates_are_valid(filing_date, filing_date, filing_date, None):
                raise SenateEFDError(
                    f"Senate report filing date is invalid: {source_record_id}"
                )
            if (
                report["chamber"] != Chamber.SENATE.value
                or report["ingestion_generation"] != self.ingestion_generation
            ):
                raise SenateEFDError(
                    "Senate report inventory chamber/generation does not match refresh"
                )
            try:
                outcome = ReportOutcome(report["outcome"])
            except (TypeError, ValueError):
                raise SenateEFDError(
                    f"Unknown Senate report outcome: {report['outcome']!r}"
                ) from None
            outcome_counts[outcome.value] += 1

            count_values = {
                name: report[name]
                for name in (
                    "raw_row_count",
                    "accepted_row_count",
                    "rejected_row_count",
                )
            }
            if any(
                isinstance(value, bool) or not isinstance(value, Integral) or value < 0
                for value in count_values.values()
            ):
                raise SenateEFDError(
                    f"Senate report row counts are invalid: {source_record_id}"
                )
            raw_count = int(count_values["raw_row_count"])
            accepted_count = int(count_values["accepted_row_count"])
            rejected_count = int(count_values["rejected_row_count"])
            if raw_count != accepted_count + rejected_count:
                raise SenateEFDError(
                    f"Senate report row accounting is invalid: {source_record_id}"
                )

            transaction_count = int(transaction_counts.get(source_record_id, 0))
            if outcome is ReportOutcome.PARSED:
                if (
                    accepted_count == 0
                    or rejected_count != 0
                    or raw_count != accepted_count
                    or transaction_count != accepted_count
                ):
                    raise SenateEFDError(
                        "Parsed Senate report count does not match transaction rows: "
                        f"{source_record_id}"
                    )
            elif transaction_count != 0:
                raise SenateEFDError(
                    f"Nonparsed Senate report has transactions: {source_record_id}"
                )

            artifact_sha256 = self._provenance_text(report["artifact_sha256"])
            landing_sha256 = self._provenance_text(report["landing_sha256"])
            paper_sha256 = self._provenance_text(report["paper_artifact_sha256"])
            paper_url = self._provenance_text(report["paper_artifact_url"])
            if outcome in {ReportOutcome.PARSED, ReportOutcome.PAPER_ONLY}:
                if (
                    not self._is_sha256(artifact_sha256)
                    or not self._is_sha256(landing_sha256)
                    or artifact_sha256 != landing_sha256
                ):
                    raise SenateEFDError(
                        f"Senate report landing hash is invalid: {source_record_id}"
                    )
            elif artifact_sha256 != landing_sha256 or any(
                value is not None and not self._is_sha256(value)
                for value in (artifact_sha256, landing_sha256)
            ):
                raise SenateEFDError(
                    f"Senate report artifact hash is invalid: {source_record_id}"
                )

            if outcome is ReportOutcome.PAPER_ONLY:
                if (
                    raw_count != 0
                    or not self._is_sha256(paper_sha256)
                    or not paper_url
                    or not self._is_allowlisted_paper_url(paper_url)
                ):
                    raise SenateEFDError(
                        f"Senate paper artifact provenance is invalid: {source_record_id}"
                    )
            elif paper_sha256 is not None or paper_url is not None:
                raise SenateEFDError(
                    f"Nonpaper Senate report has paper provenance: {source_record_id}"
                )
            report_hashes[source_record_id] = artifact_sha256 or ""
            report_bindings[source_record_id] = (
                report_member,
                report_path,
                filing_date,
                outcome,
            )

        expected_counts = {
            "found": self.last_refresh_summary.found,
            "parsed": self.last_refresh_summary.parsed,
            "paper_only": self.last_refresh_summary.paper_only,
            "unavailable": self.last_refresh_summary.unavailable,
            "failed": self.last_refresh_summary.failed,
        }
        actual_counts = {
            "found": len(self.report_inventory),
            **outcome_counts,
        }
        if actual_counts != expected_counts:
            raise SenateEFDError(
                "Senate report outcome counts do not match refresh summary: "
                f"expected={expected_counts}, actual={actual_counts}"
            )
        unknown_transaction_reports = set(transaction_counts) - set(report_hashes)
        if unknown_transaction_reports:
            raise SenateEFDError(
                "Senate transactions have no report inventory: "
                f"{sorted(unknown_transaction_reports)}"
            )
        binding_columns = [
            "doc_id",
            "source_record_id",
            "source_report_path",
            "member",
            "member_key",
            "chamber_member_key",
            "transaction_date",
            "disclosure_date",
            "official_filing_date",
            "available_date",
            "transaction_type",
            "raw_transaction_subtype",
            "ticker",
            "raw_ticker",
            "ticker_candidate",
            "ticker_origin",
            "asset_description",
            "raw_asset_description",
            "raw_asset_class",
            "strike_price",
            "expiry_date",
            "amends_source_record_id",
            "owner_code",
            "raw_owner",
            "amount_raw",
            "amount_midpoint",
            "instrument_type",
            "artifact_sha256",
        ]
        for row in df[binding_columns].itertuples(index=False):
            source_record_id = self._provenance_text(row.source_record_id)
            binding = report_bindings.get(source_record_id or "")
            if binding is None:
                raise SenateEFDError(
                    f"Senate transaction has no report inventory: {source_record_id}"
                )
            report_member, report_path, filing_date, outcome = binding
            if outcome is not ReportOutcome.PARSED:
                raise SenateEFDError(
                    f"Nonparsed Senate report has transactions: {source_record_id}"
                )
            if self._provenance_text(row.doc_id) != self._path_to_doc_id(report_path):
                raise SenateEFDError(
                    f"Senate transaction doc_id mismatch: {source_record_id}"
                )
            if row.member != report_member:
                raise SenateEFDError(
                    f"Senate transaction member mismatch: {source_record_id}"
                )
            if row.member_key != canonical_member_key(
                row.member
            ) or row.chamber_member_key != chamber_scoped_member_key(
                row.member, Chamber.SENATE.value
            ):
                raise SenateEFDError(
                    f"Senate transaction member key mismatch: {source_record_id}"
                )
            if row.source_report_path != report_path:
                raise SenateEFDError(
                    f"Senate transaction report path mismatch: {source_record_id}"
                )
            transaction_filing_dates = (
                self._parse_date(row.disclosure_date),
                self._parse_date(row.official_filing_date),
                self._parse_date(row.available_date),
            )
            if any(value != filing_date for value in transaction_filing_dates):
                raise SenateEFDError(
                    f"Senate transaction filing date binding mismatch: {source_record_id}"
                )

            if row.asset_description != row.raw_asset_description:
                raise SenateEFDError(
                    f"Senate transaction asset description mismatch: {source_record_id}"
                )
            if any(
                self._provenance_text(value) is not None
                for value in (
                    row.strike_price,
                    row.expiry_date,
                    row.amends_source_record_id,
                )
            ):
                raise SenateEFDError(
                    f"Unsupported Senate derived value is nonnull: {source_record_id}"
                )
            resolved_ticker, resolved_candidate, resolved_origin = self._resolve_ticker(
                row.raw_ticker,
                self._provenance_text(row.raw_asset_description) or "",
                self._provenance_text(row.raw_asset_class) or "",
            )
            actual_ticker = self._provenance_text(row.ticker)
            actual_candidate = self._provenance_text(row.ticker_candidate)
            if (
                actual_ticker != resolved_ticker
                or actual_candidate != resolved_candidate
                or row.ticker_origin != resolved_origin.value
            ):
                raise SenateEFDError(
                    f"Senate transaction ticker/raw binding mismatch: {source_record_id}"
                )
            if (
                self._normalize_tx_type(row.raw_transaction_subtype)
                != row.transaction_type
            ):
                raise SenateEFDError(
                    f"Senate transaction subtype binding mismatch: {source_record_id}"
                )
            expected_owner = _extract_owner_code(
                self._provenance_text(row.raw_owner) or ""
            )
            if expected_owner != row.owner_code:
                raise SenateEFDError(
                    f"Senate transaction owner binding mismatch: {source_record_id}"
                )
            rebound_amount, rebound_midpoint = _extract_amount_midpoint(row.amount_raw)
            midpoint_matches = (
                pd.isna(rebound_midpoint) and pd.isna(row.amount_midpoint)
            ) or rebound_midpoint == row.amount_midpoint
            if rebound_amount != row.amount_raw or not midpoint_matches:
                raise SenateEFDError(
                    f"Senate transaction amount binding mismatch: {source_record_id}"
                )
            if (
                self._normalize_instrument_type(row.raw_asset_class)
                != row.instrument_type
            ):
                raise SenateEFDError(
                    f"Senate transaction asset class binding mismatch: {source_record_id}"
                )
            if report_hashes[source_record_id] != row.artifact_sha256:
                raise SenateEFDError(
                    "Senate transaction artifact hash does not match report landing: "
                    f"{source_record_id}"
                )

        inserted = persist_refresh(
            transactions=df,
            reports=pd.DataFrame(self.report_inventory),
            source="senate_efd",
            chamber=Chamber.SENATE.value,
            ingestion_generation=self.ingestion_generation,
        )
        logger.info(
            "Persisted %d Senate eFD transactions and %d report outcomes",
            inserted,
            len(self.report_inventory),
        )
        return inserted

    def fetch_and_save_all(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
        chamber: str | None = "senate",
    ) -> int:
        df = self.fetch_all_trades(start_date, end_date, chamber)
        return self.save_to_db(df)
