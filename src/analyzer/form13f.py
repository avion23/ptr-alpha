"""Form 13F position-change source."""

from __future__ import annotations

import logging
import math
import os
import re
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path
from typing import Callable

import pandas as pd
import requests

from analyzer.database import Database
from analyzer.form4 import (
    SEC_ARCHIVES_URL,
    SEC_SUBMISSIONS_URL,
    _SECClient,
    _accession_digits,
)
from analyzer.interfaces import TransactionSource
from analyzer.manager_watchlist import WATCHLIST_MANAGERS
from analyzer.transaction_repository import _BASE_WRITE_COLUMNS

logger = logging.getLogger(__name__)

SEC_USER_AGENT = os.environ.get(
    "SEC_USER_AGENT", "InsiderTrading 13F research contact admin@example.com"
)

_OUTPUT_PROVENANCE_COLUMNS = (
    "source_record_id",
    "source_row_id",
    "source_report_path",
    "raw_owner",
    "official_filing_date",
    "raw_transaction_subtype",
    "raw_asset_description",
    "ingestion_generation",
)
_OUTPUT_COLUMNS = _BASE_WRITE_COLUMNS + _OUTPUT_PROVENANCE_COLUMNS
_SHARE_SUBTYPE = "INFERRED_POSITION_INCREASE"
_INCREASE_COLUMNS = (
    "manager_key",
    "manager",
    "ticker",
    "event_date",
    "disclosure_date",
    "cusip",
    "added_shares",
    "raw_asset_description",
    "source",
    "accession",
    "report_url",
)
_CANDIDATE_COLUMNS = (
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


def _canonical_row(
    *,
    manager_name,
    ticker,
    event_date,
    disclosure_date,
    added_shares,
    amount_midpoint,
    issuer,
    cusip,
    accession,
    report_url,
    ingestion_generation,
) -> dict:
    """One canonical 13F row; shared by watchlist and discovery paths."""
    return {
        "doc_id": f"13f-{accession}",
        "member": manager_name,
        "ticker": ticker,
        "transaction_date": event_date,
        "disclosure_date": disclosure_date,
        "transaction_type": "Purchase",
        "owner_code": None,
        "amount_raw": f"{added_shares:,} shares",
        "amount_midpoint": amount_midpoint,
        "instrument_type": "Common Stock",
        "strike_price": None,
        "expiry_date": None,
        "created_at": None,
        "asset_description": issuer,
        "source": "13f",
        "source_record_id": accession,
        "source_row_id": cusip,
        "source_report_path": report_url,
        "raw_owner": manager_name,
        "official_filing_date": disclosure_date,
        "raw_transaction_subtype": _SHARE_SUBTYPE,
        "raw_asset_description": issuer,
        "ingestion_generation": ingestion_generation,
    }


class ThirteenFError(Exception):
    """Raised when a 13F filing cannot be safely used."""


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _parse_information_table(xml_content: bytes | str) -> dict[str, dict[str, object]]:
    try:
        root = ET.fromstring(xml_content)
    except (ET.ParseError, TypeError, ValueError) as exc:
        raise ThirteenFError(f"Unparseable 13F information table: {exc}") from exc

    entries = [
        element for element in root.iter() if _local_name(element.tag) == "infoTable"
    ]
    holdings: dict[str, dict[str, object]] = {}
    for entry in entries:
        fields = {
            _local_name(child.tag): (child.text or "").strip()
            for child in entry.iter()
            if child is not entry
        }
        issuer = fields.get("nameOfIssuer", "")
        cusip = fields.get("cusip", "").upper()
        shares_raw = fields.get("sshPrnamt", "")
        amount_type = fields.get("sshPrnamtType", "").upper()
        if not issuer or not cusip or not shares_raw:
            raise ThirteenFError(
                "13F information table has a row missing issuer, CUSIP, or shares"
            )
        if amount_type != "SH" or fields.get("putCall"):
            continue
        try:
            shares = int(str(shares_raw).replace(",", ""))
        except ValueError as exc:
            raise ThirteenFError(f"Invalid 13F share count for CUSIP {cusip}") from exc
        if shares < 0:
            raise ThirteenFError(f"Negative 13F share count for CUSIP {cusip}")

        existing = holdings.get(cusip)
        if existing is None:
            holdings[cusip] = {"issuer": issuer, "shares": shares}
        elif str(existing["issuer"]).casefold() == issuer.casefold():
            existing["shares"] = int(str(existing["shares"])) + shares
        else:
            raise ThirteenFError(f"Conflicting issuer names for CUSIP {cusip}")

    if not entries or not holdings:
        raise ThirteenFError(
            "13F information table is empty or has no usable share rows"
        )
    return holdings


def _recent_quarters(submissions: dict) -> list[dict[str, str]]:
    filings = submissions.get("filings")
    recent = filings.get("recent") if isinstance(filings, dict) else None
    required = ("form", "reportDate", "filingDate", "accessionNumber")
    if not isinstance(recent, dict) or any(
        not isinstance(recent.get(field), list) for field in required
    ):
        raise ThirteenFError("SEC submissions response has no recent filing index")
    if len({len(recent[field]) for field in required}) != 1:
        raise ThirteenFError(
            "SEC submissions filing index columns have different lengths"
        )

    by_report_date: dict[date, dict[str, str]] = {}
    rows = zip(*(recent[field] for field in required))
    for form, report_raw, filed_raw, accession in rows:
        if form not in {"13F-HR", "13F-HR/A"}:
            continue
        try:
            report_date = date.fromisoformat(report_raw)
            filing_date = date.fromisoformat(filed_raw)
        except (TypeError, ValueError):
            continue
        if not accession:
            continue
        try:
            _accession_digits(accession)
        except ValueError:
            raise ThirteenFError(
                f"SEC submissions filing index has invalid accession: {accession!r}"
            ) from None
        candidate = {
            "report_date": report_date.isoformat(),
            "filing_date": filing_date.isoformat(),
            "accession": str(accession),
        }
        previous = by_report_date.get(report_date)
        if previous is None or (candidate["filing_date"], candidate["accession"]) > (
            previous["filing_date"],
            previous["accession"],
        ):
            by_report_date[report_date] = candidate

    selected = [by_report_date[key] for key in sorted(by_report_date, reverse=True)[:2]]
    if len(selected) != 2:
        raise ThirteenFError(
            "SEC submissions index has fewer than two usable 13F quarters"
        )
    return selected


class ThirteenFSource(TransactionSource):
    """Compare 13F positions against ticker keywords or across all holdings."""

    def __init__(
        self,
        *,
        watchlist: dict[str, set[str]],
        data_dir: str | Path = "data",
        read_only: bool = False,
        db: Database | None = None,
        ingestion_generation: str | None = None,
        price_lookup: Callable[[str, date], float | None] | None = None,
        managers: dict[str, dict[str, str]] | None = None,
        ticker_map: dict[str, str] | None = None,
    ):
        if (
            not isinstance(watchlist, dict)
            or not watchlist
            or any(
                not isinstance(ticker, str)
                or not ticker.strip()
                or not isinstance(keywords, set)
                or not keywords
                or any(
                    not isinstance(keyword, str) or not keyword.strip()
                    for keyword in keywords
                )
                for ticker, keywords in watchlist.items()
            )
        ):
            raise ThirteenFError(
                "A non-empty ticker-to-issuer-keywords watchlist is required"
            )
        if not SEC_USER_AGENT.strip():
            raise ThirteenFError("SEC_USER_AGENT must be non-empty")
        if ingestion_generation is not None and not ingestion_generation.strip():
            raise ThirteenFError("ingestion_generation must be non-empty when supplied")

        self.watchlist = {
            ticker: set(keywords) for ticker, keywords in watchlist.items()
        }
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._owns_db = db is None
        self.db = (
            db
            if db is not None
            else Database(self.data_dir / "congress.duckdb", read_only=read_only)
        )
        self.ingestion_generation = ingestion_generation
        self.price_lookup = price_lookup
        self.managers = WATCHLIST_MANAGERS if managers is None else managers
        if not self.managers:
            raise ThirteenFError("At least one 13F manager is required")
        if ticker_map is not None and (
            not isinstance(ticker_map, dict)
            or any(
                not isinstance(cusip, str)
                or not cusip.strip()
                or not isinstance(ticker, str)
                or not ticker.strip()
                for cusip, ticker in ticker_map.items()
            )
        ):
            raise ThirteenFError("ticker_map must map non-empty CUSIPs to tickers")
        self.ticker_map = {
            cusip.strip().upper(): ticker.strip()
            for cusip, ticker in (ticker_map or {}).items()
        }
        self._sec = _SECClient(
            SEC_USER_AGENT,
            {"Accept": "application/json, application/xml, text/xml"},
        )
        self.session = self._sec.session

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
        return self.db.get_transactions(year, source="13f")

    def _get(self, url: str) -> requests.Response:
        return self._sec.get(url, ThirteenFError)

    def _load_submissions(self, cik: str) -> dict:
        return self._sec.submissions(cik, ThirteenFError)

    def _load_filing(
        self, cik: str, filing: dict[str, str]
    ) -> tuple[dict[str, dict[str, object]], str]:
        cik_path = str(int(cik))
        accession_path = _accession_digits(filing["accession"])
        directory_url = f"{SEC_ARCHIVES_URL}/{cik_path}/{accession_path}"
        index_url = f"{directory_url}/index.json"
        try:
            index = self._get(index_url).json()
        except (ValueError, requests.JSONDecodeError) as exc:
            raise ThirteenFError(f"Unparseable SEC filing index: {index_url}") from exc
        files = (
            index.get("directory", {}).get("item", [])
            if isinstance(index, dict)
            else []
        )
        info_tables = [
            item.get("name", "")
            for item in files
            if isinstance(item, dict)
            and str(item.get("name", "")).lower().endswith(".xml")
            and any(
                marker in re.sub(r"[^a-z]", "", str(item.get("name", "")).lower())
                for marker in ("infotable", "informationtable", "inftbl")
            )
        ]
        if len(info_tables) != 1:
            raise ThirteenFError(
                f"SEC filing index has {len(info_tables)} information-table XML files: {index_url}"
            )
        report_url = f"{directory_url}/{info_tables[0]}"
        response = self._get(report_url)
        return _parse_information_table(response.content), report_url

    def _manager_transactions(self, manager: dict[str, str]) -> list[dict]:
        filings = _recent_quarters(self._load_submissions(manager["cik"]))
        previous_filing, current_filing = reversed(filings)
        previous, _ = self._load_filing(manager["cik"], previous_filing)
        current, report_url = self._load_filing(manager["cik"], current_filing)
        return self._position_changes(
            manager, current_filing, current, previous, report_url
        )

    def _manager_increases(self, key: str, manager: dict[str, str]) -> list[dict]:
        filings = _recent_quarters(self._load_submissions(manager["cik"]))
        previous_filing, current_filing = reversed(filings)
        previous, _ = self._load_filing(manager["cik"], previous_filing)
        current, report_url = self._load_filing(manager["cik"], current_filing)
        return self._position_increases(
            key, manager, current_filing, current, previous, report_url
        )

    def _position_increases(
        self,
        manager_key: str,
        manager: dict[str, str],
        current_filing: dict[str, str],
        current: dict[str, dict[str, object]],
        previous: dict[str, dict[str, object]],
        report_url: str,
    ) -> list[dict]:
        quarter_end = date.fromisoformat(current_filing["report_date"])
        disclosure_date = date.fromisoformat(current_filing["filing_date"])
        rows = []
        for cusip, holding in current.items():
            shares = int(str(holding["shares"]))
            added_shares = shares - int(str(previous.get(cusip, {}).get("shares", 0)))
            if added_shares <= 0:
                continue

            issuer = str(holding["issuer"])
            issuer_folded = issuer.casefold()
            matches = {
                ticker
                for ticker, keywords in self.watchlist.items()
                if any(keyword.casefold() in issuer_folded for keyword in keywords)
            }
            ticker = (
                next(iter(matches))
                if len(matches) == 1
                else self.ticker_map.get(cusip.upper())
            )
            rows.append(
                {
                    "manager_key": manager_key,
                    "manager": manager["name"],
                    "ticker": ticker,
                    "event_date": quarter_end,
                    "disclosure_date": disclosure_date,
                    "cusip": cusip,
                    "added_shares": added_shares,
                    "raw_asset_description": issuer,
                    "source": "13f",
                    "accession": current_filing["accession"],
                    "report_url": report_url,
                }
            )
        return rows

    def _position_changes(
        self,
        manager: dict[str, str],
        current_filing: dict[str, str],
        current: dict[str, dict[str, object]],
        previous: dict[str, dict[str, object]],
        report_url: str,
    ) -> list[dict]:
        quarter_end = date.fromisoformat(current_filing["report_date"])
        disclosure_date = date.fromisoformat(current_filing["filing_date"])
        rows = []
        for cusip, holding in current.items():
            issuer = str(holding["issuer"])
            issuer_folded = issuer.casefold()
            matches = {
                ticker
                for ticker, keywords in self.watchlist.items()
                if any(keyword.casefold() in issuer_folded for keyword in keywords)
            }
            if len(matches) != 1:
                logger.debug(
                    "Skipping 13F holding %s (%s): issuer matched %s watchlist tickers",
                    cusip,
                    issuer,
                    len(matches),
                )
                continue

            shares = int(str(holding["shares"]))
            added_shares = shares - int(str(previous.get(cusip, {}).get("shares", 0)))
            if added_shares <= 0:
                continue

            ticker = matches.pop()
            price = (
                self.price_lookup(ticker, quarter_end) if self.price_lookup else None
            )
            if price is not None:
                try:
                    price = float(price)
                except (TypeError, ValueError) as exc:
                    raise ThirteenFError(
                        f"Invalid quarter-end price for {ticker}"
                    ) from exc
                if not math.isfinite(price) or price < 0:
                    raise ThirteenFError(f"Invalid quarter-end price for {ticker}")

            rows.append(
                _canonical_row(
                    manager_name=manager["name"],
                    ticker=ticker,
                    event_date=quarter_end,
                    disclosure_date=disclosure_date,
                    added_shares=added_shares,
                    amount_midpoint=added_shares * price
                    if price is not None
                    else None,
                    issuer=issuer,
                    cusip=cusip,
                    accession=current_filing["accession"],
                    report_url=report_url,
                    ingestion_generation=self.ingestion_generation,
                )
            )
        return rows

    def fetch_all_trades(self) -> pd.DataFrame:
        rows = []
        failures = []
        for key, manager in self.managers.items():
            try:
                rows.extend(self._manager_transactions(manager))
            except ThirteenFError as exc:
                failures.append(f"{key}: {exc}")
                logger.warning("Skipping 13F manager %s: %s", key, exc)
        if failures:
            logger.warning(
                "13F skipped %d of %d managers", len(failures), len(self.managers)
            )
        if not rows:
            raise ThirteenFError(
                "No 13F manager produced position changes; "
                + "; ".join(failures)
            )
        return pd.DataFrame(rows, columns=_OUTPUT_COLUMNS)

    def fetch_all_increases(self) -> pd.DataFrame:
        rows = []
        failures = []
        for key, manager in self.managers.items():
            try:
                rows.extend(self._manager_increases(key, manager))
            except ThirteenFError as exc:
                failures.append(f"{key}: {exc}")
                logger.warning("Skipping 13F manager %s: %s", key, exc)
        if failures:
            logger.warning(
                "13F skipped %d of %d managers", len(failures), len(self.managers)
            )
        if len(failures) == len(self.managers):
            raise ThirteenFError("All 13F managers failed; " + "; ".join(failures))
        increases = pd.DataFrame(rows, columns=_INCREASE_COLUMNS)
        if rows:
            increases["ticker"] = pd.Series(
                [row["ticker"] for row in rows], dtype=object
            )
        return increases

    def save_to_db(self, transactions: pd.DataFrame) -> int:
        if tuple(transactions.columns) != _OUTPUT_COLUMNS:
            raise ThirteenFError(
                "13F transaction columns do not match the canonical schema"
            )
        if not self.ingestion_generation or not self.ingestion_generation.strip():
            raise ThirteenFError("ingestion_generation is required for 13F persistence")
        if not transactions.empty:
            if not transactions["source"].eq("13f").all() or not transactions[
                "ingestion_generation"
            ].eq(self.ingestion_generation).all():
                raise ThirteenFError("13F source or ingestion generation does not match")
        return self.db.upsert_transactions(transactions, source="13f")

    def fetch_and_save_all(self) -> int:
        return self.save_to_db(self.fetch_all_trades())

    def save_increases(self, increases: pd.DataFrame) -> int:
        """Persist discovery-scope increases as canonical rows.

        Discovery frames carry no dollar amounts (no price lookup at this
        scope), so amount_midpoint stays NULL and downstream sizing treats
        these rows as corroboration with unknown size. Ticker may be NULL
        for unmatched issuers; those rows persist for review.
        """
        if not set(_INCREASE_COLUMNS) <= set(increases.columns):
            raise ThirteenFError(
                "13F increase frame does not match the discovery schema"
            )
        if not self.ingestion_generation or not self.ingestion_generation.strip():
            raise ThirteenFError("ingestion_generation is required for 13F persistence")
        rows = []
        for row in increases.to_dict("records"):
            rows.append(
                _canonical_row(
                    manager_name=row.get("manager"),
                    ticker=row.get("ticker"),
                    event_date=row.get("event_date"),
                    disclosure_date=row.get("disclosure_date"),
                    added_shares=row.get("added_shares") or 0,
                    amount_midpoint=None,
                    issuer=row.get("raw_asset_description"),
                    cusip=row.get("cusip"),
                    accession=row.get("accession"),
                    report_url=row.get("report_url"),
                    ingestion_generation=self.ingestion_generation,
                )
            )
        canonical = pd.DataFrame(rows, columns=_OUTPUT_COLUMNS)
        return self.save_to_db(canonical)


def candidates_from_increases(df: pd.DataFrame) -> pd.DataFrame:
    candidates = pd.DataFrame(index=df.index)
    candidates["ticker"] = df["ticker"].astype(object)
    candidates.loc[candidates["ticker"].isna(), "ticker"] = None
    candidates["actor_id"] = "manager:" + df["manager_key"].astype(str)
    candidates["kind"] = "manager"
    candidates["source"] = "13f"
    candidates["entry_ref"] = None
    candidates["event_date"] = df["event_date"]
    candidates["disclosure_date"] = df["disclosure_date"]
    candidates["corroboration"] = True
    candidates["position_evidence"] = False
    candidates["as_of"] = pd.Timestamp.now(tz="UTC").normalize()
    return candidates.loc[:, _CANDIDATE_COLUMNS].reset_index(drop=True)


__all__ = ["ThirteenFError", "ThirteenFSource", "candidates_from_increases"]
