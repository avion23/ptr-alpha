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

Known limitation: efdsearch exposes no amendment/supersession metadata. An
amended filing that restates transactions with changed amounts is stored
alongside the original (the site itself behaves this way). Exact restatements
across filings within one run are de-duplicated keeping the latest filing.
"""

import hashlib
import logging
import random
import re
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Literal

import pandas as pd
from bs4 import BeautifulSoup
from curl_cffi import requests as cffi_requests

from analyzer.database import Database
from analyzer.interfaces import TransactionSource
from analyzer.parsing.cells import _extract_amount_midpoint, _extract_owner_code, _extract_ticker

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
_UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I)

_HEADER_NAME_MAP = {
    "transaction_date": "transaction date",
    "owner": "owner",
    "ticker": "ticker",
    "asset_name": "asset name",
    "asset_type": "asset type",
    "tx_type": "type",
    "amount": "amount",
}
_REQUIRED_HEADERS = {"transaction_date", "owner", "asset_name", "tx_type", "amount"}


class SenateEFDError(Exception):
    pass


class SenateEFDBlockedError(SenateEFDError):
    """Raised when efdsearch blocks the client or the backend fails."""


class SenateEFDSource(TransactionSource):
    """Scrapes official Senate PTR filings from efdsearch.senate.gov."""

    def __init__(self, data_dir: str | Path = "data", read_only: bool = False, db: Database | None = None):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._owns_db = db is None
        self.db = db if db is not None else Database(self.data_dir / "congress.duckdb", read_only=read_only)
        self._session: cffi_requests.Session | None = None
        self._csrf_token: str | None = None

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
                delay = RETRY_BASE_DELAY * (2 ** (attempt - 1)) + random.uniform(0, RETRY_JITTER)
                logger.warning("eFD request error (%s); retrying in %.1fs", e, delay)
                time.sleep(delay)
                continue

            if resp.status_code == 200:
                return resp

            if resp.status_code == 429:
                if attempt == attempts:
                    raise SenateEFDBlockedError("eFD rate-limited (HTTP 429) after retries")
                retry_after = resp.headers.get("Retry-After")
                try:
                    delay = float(retry_after) if retry_after else RETRY_BASE_DELAY * (2 ** (attempt - 1))
                except ValueError:
                    delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                logger.warning("eFD HTTP 429; sleeping %.1fs", delay)
                time.sleep(delay)
                continue

            if resp.status_code in (401, 403):
                if not refreshed and attempt < attempts:
                    logger.warning(
                        "eFD auth/block response HTTP %d; refreshing session", resp.status_code
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
                delay = RETRY_BASE_DELAY * (2 ** (attempt - 1)) + random.uniform(0, RETRY_JITTER)
                logger.warning("eFD HTTP %d; retrying in %.1fs", resp.status_code, delay)
                time.sleep(delay)
                continue

            return resp
        raise SenateEFDBlockedError("unreachable retry state")

    def _open_session(self) -> str:
        """Perform the CSRF agreement handshake. Return the CSRF cookie token."""
        if self._session is not None:
            try:
                self._session.close()
            except Exception:
                pass
        session = cffi_requests.Session(impersonate=BROWSER_IMPERSONATE)
        self._session = session
        resp = session.get(f"{EFD_BASE}/search/", timeout=REQUEST_TIMEOUT)
        if resp.status_code != 200:
            raise SenateEFDBlockedError(f"eFD search page returned HTTP {resp.status_code}")
        m = _CSRF_RE.search(resp.text)
        if not m:
            raise SenateEFDError("csrfmiddlewaretoken not found; eFD layout may have changed")
        form_token = m.group(1)

        agree = session.post(
            f"{EFD_BASE}/search/home/",
            data={"csrfmiddlewaretoken": form_token, "prohibition_agreement": "1"},
            headers={"Referer": f"{EFD_BASE}/search/home/", "Origin": EFD_BASE},
            timeout=REQUEST_TIMEOUT,
        )
        if agree.status_code != 200:
            raise SenateEFDBlockedError(f"eFD agreement POST returned HTTP {agree.status_code}")

        csrf_cookie = session.cookies.get("csrftoken")
        if not csrf_cookie:
            raise SenateEFDError("eFD session did not return a csrftoken cookie")
        self._csrf_token = csrf_cookie
        logger.info("Opened eFD session (CSRF handshake complete)")
        return csrf_cookie

    def _require_session(self) -> cffi_requests.Session:
        if self._session is None:
            self._open_session()
        assert self._session is not None
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
        logger.info("eFD search found %d unique PTR filings (%s..%s)", len(reports), start_date, end_date)
        return list(reports.values())

    def _search_window(self, start_date: date, end_date: date, reports: dict[str, dict]) -> None:
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
                name = re.sub(r"\s+", " ", f"{first_name} {last_name}".strip()).strip(" ,")
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
                        start_date, end_date, stale_pages,
                    )
                    break
            else:
                before = len(reports)
                stale_pages = 0
            time.sleep(SEARCH_PAGE_DELAY)

    def _fetch_report_transactions(self, report_path: str) -> list[dict]:
        """GET one PTR detail and parse its transaction table from headers."""
        resp = self._request_with_retry("GET", f"{EFD_BASE}{report_path}")
        if resp.status_code in (404, 410):
            logger.warning("eFD filing %s unavailable (HTTP %d); skipping", report_path, resp.status_code)
            return []
        soup = BeautifulSoup(resp.text, "html.parser")
        table = soup.find("table", {"class": "table-striped"})
        if table is None:
            # Paper/PDF-only filing: no transaction table.
            return []
        thead = table.find("thead")
        tbody = table.find("tbody")
        if thead is None or tbody is None:
            raise SenateEFDError(f"PTR {report_path} has no parseable header/body")
        headers = [th.get_text(strip=True).lower() for th in thead.find_all(["th", "td"])]
        col: dict[str, int] = {}
        for key, label in _HEADER_NAME_MAP.items():
            if label in headers:
                col[key] = headers.index(label)
        missing = _REQUIRED_HEADERS - set(col)
        if missing:
            raise SenateEFDError(
                f"PTR {report_path} table missing columns {sorted(missing)}; headers={headers}"
            )
        out: list[dict] = []
        for tr in tbody.find_all("tr"):
            cells = [td.get_text(strip=True) for td in tr.find_all("td")]
            if not cells:
                continue

            def cell(key: str) -> str:
                idx = col.get(key)
                if idx is None or idx >= len(cells):
                    return ""
                return cells[idx]

            asset_name = cell("asset_name")
            ticker_raw = cell("ticker").strip()
            if ticker_raw and ticker_raw != "--":
                ticker = ticker_raw.upper()
            else:
                ticker = _extract_ticker(asset_name)
            out.append({
                "ticker": ticker,
                "asset_name": asset_name,
                "asset_type": cell("asset_type"),
                "owner": _extract_owner_code(cell("owner")),
                "type": cell("tx_type"),
                "transaction_date": cell("transaction_date"),
                "amount_range": cell("amount"),
            })
        return out

    @staticmethod
    def _dedupe_restatements(raw: list[dict]) -> list[dict]:
        """Drop exact transaction restatements across filings within one run.

        An amendment re-filing restates identical transactions under a new
        document id. Keep only the latest-filed copy. Amendments that alter
        amounts or tickers are not de-duplicated because efdsearch exposes
        no supersession metadata.
        """
        best: dict[tuple, dict] = {}
        for t in raw:
            key = (
                t.get("senator"), t.get("ticker"), t.get("transaction_date"),
                t.get("type"), t.get("amount_range"), t.get("owner"), t.get("asset_name"),
            )
            prev = best.get(key)
            if prev is None or (t.get("filed_date") or pd.Timestamp.min) >= (prev.get("filed_date") or pd.Timestamp.min):
                best[key] = t
        dropped = len(raw) - len(best)
        if dropped:
            logger.warning("Dropped %d restated transactions across filings (amendments)", dropped)
        return list(best.values())

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
            raise SenateEFDError(f"SenateEFDSource is senate-only; got chamber={chamber!r}")
        end_date = end_date or date.today()
        start_date = start_date or self._one_year_before(end_date)

        self._open_session()
        reports = self._search_reports(start_date, end_date)
        raw: list[dict] = []
        failures = 0
        for report in reports:
            doc_id = self._path_to_doc_id(report["report_path"])
            try:
                txns = self._fetch_report_transactions(report["report_path"])
            except SenateEFDBlockedError:
                raise
            except SenateEFDError as e:
                failures += 1
                logger.error("Failed to parse %s: %s", report["report_path"], e)
                continue
            for tx in txns:
                raw.append({
                    "doc_id": doc_id,
                    "senator": report["senator"],
                    "filed_date": report["filed_date"],
                    **tx,
                })
            time.sleep(PTR_FETCH_DELAY)
        if failures:
            logger.warning("Failed to parse %d of %d filings", failures, len(reports))
        raw = self._dedupe_restatements(raw)
        logger.info("Fetched %d raw Senate transactions from %d filings", len(raw), len(reports))
        return self._normalize(raw)

    @staticmethod
    def _path_to_doc_id(report_path: str) -> str:
        m = _UUID_RE.search(report_path)
        if m:
            return m.group(0)
        return "efd-" + hashlib.sha1(report_path.encode()).hexdigest()[:16]

    @staticmethod
    def _one_year_before(d: date) -> date:
        try:
            return d.replace(year=d.year - 1)
        except ValueError:
            return d.replace(year=d.year - 1, day=28)

    def _normalize(self, trades: list[dict]) -> pd.DataFrame:
        columns = [
            "doc_id", "member", "ticker", "transaction_date", "disclosure_date",
            "transaction_type", "owner_code", "amount_raw", "amount_midpoint",
            "instrument_type", "strike_price", "expiry_date", "asset_description",
        ]
        if not trades:
            return pd.DataFrame(columns=columns)

        rows = []
        rejected = 0
        for t in trades:
            amount_raw, midpoint = _extract_amount_midpoint(t.get("amount_range"))
            row = {
                "doc_id": t.get("doc_id"),
                "member": t.get("senator"),
                "ticker": t.get("ticker"),
                "transaction_date": self._parse_date(t.get("transaction_date")),
                "disclosure_date": self._parse_date(t.get("filed_date")),
                "transaction_type": self._normalize_tx_type(t.get("type")),
                "owner_code": t.get("owner"),
                "amount_raw": amount_raw or t.get("amount_range"),
                "amount_midpoint": midpoint,
                "instrument_type": self._normalize_instrument_type(t.get("asset_type")),
                "strike_price": None,
                "expiry_date": None,
                "asset_description": t.get("asset_name"),
            }
            if (
                not row["doc_id"]
                or not row["member"]
                or row["transaction_date"] is None
                or row["disclosure_date"] is None
                or not row["transaction_type"]
            ):
                rejected += 1
                continue
            rows.append(row)
        if rejected:
            logger.warning("Rejected %d invalid Senate eFD transaction rows", rejected)
        return pd.DataFrame(rows, columns=columns)

    @staticmethod
    def _parse_date(value):
        if not value:
            return None
        try:
            return pd.Timestamp(value)
        except Exception:
            return None

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
            return "stock"
        lower = raw.lower()
        if "call" in lower:
            return "call"
        if "put" in lower:
            return "put"
        return "stock"

    def save_to_db(self, df: pd.DataFrame) -> int:
        if df.empty:
            return 0
        inserted = self.db.upsert_transactions(df, source="senate_efd")
        logger.info("Inserted %d new Senate eFD transactions from %d records", inserted, len(df))
        return inserted

    def fetch_and_save_all(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
        chamber: str | None = "senate",
    ) -> int:
        df = self.fetch_all_trades(start_date, end_date, chamber)
        return self.save_to_db(df)
