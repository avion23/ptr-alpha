"""Form 13F position-change source."""

from __future__ import annotations

import json
import logging
import os
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path

import pandas as pd
import requests

from analyzer.asset_ticker_map import resolve_asset_ticker
from analyzer.database import Database
from analyzer.form4 import (
    SEC_ARCHIVES_URL,
    _SECClient,
    _accession_digits,
    _acceptance_datetime,
)
from analyzer.interfaces import TransactionSource
from analyzer.manager_watchlist import WATCHLIST_MANAGERS
from analyzer.transaction_repository import _BASE_WRITE_COLUMNS

logger = logging.getLogger(__name__)

SEC_USER_AGENT = os.environ.get(
    "SEC_USER_AGENT", "InsiderTrading 13F research contact admin@example.com"
)

_OUTPUT_PROVENANCE_COLUMNS = (
    "acceptance_datetime",
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
_SNAPSHOT_SUBTYPE = "13F_SHARE_POSITION_SNAPSHOT"
_INCREASE_COLUMNS = (
    "acceptance_datetime",
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
    acceptance_datetime=None,
    transaction_type="Purchase",
    amount_raw=None,
    raw_transaction_subtype=_SHARE_SUBTYPE,
) -> dict:
    """One canonical 13F row; shared by watchlist and discovery paths."""
    row_id = cusip
    if row_id is None or (isinstance(row_id, float) and pd.isna(row_id)) or not str(
        row_id
    ).strip():
        # Unmatched issuers have no CUSIP; fall back to a stable composite so
        # re-fetches dedupe instead of duplicating (or colliding on None).
        # Best effort: identical composites for truly identical rows dedupe
        # correctly; near-identical review rows may need human eyes.
        row_id = (
            f"{issuer}|{ticker or 'NOTICKER'}|{event_date}|{added_shares}"
        )
    return {
        "doc_id": f"13f-{accession}",
        "member": manager_name,
        "ticker": ticker,
        "transaction_date": event_date,
        "disclosure_date": disclosure_date,
        "acceptance_datetime": acceptance_datetime,
        "transaction_type": transaction_type,
        "owner_code": None,
        "amount_raw": amount_raw or f"{added_shares:,} shares",
        "amount_midpoint": amount_midpoint,
        "instrument_type": "Common Stock",
        "strike_price": None,
        "expiry_date": None,
        "created_at": None,
        "asset_description": issuer,
        "source": "13f",
        "source_record_id": accession,
        "source_row_id": row_id,
        "source_report_path": report_url,
        "raw_owner": manager_name,
        "official_filing_date": disclosure_date,
        "raw_transaction_subtype": raw_transaction_subtype,
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
    portfolio_value_total = 0
    portfolio_value_complete = True
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
        value_raw = fields.get("value", "")
        if not issuer or not cusip or not shares_raw:
            raise ThirteenFError(
                "13F information table has a row missing issuer, CUSIP, or shares"
            )
        try:
            value_thousands = int(str(value_raw).replace(",", "")) if value_raw else None
        except ValueError as exc:
            raise ThirteenFError(f"Invalid 13F reported value for CUSIP {cusip}") from exc
        if value_thousands is None:
            portfolio_value_complete = False
        elif value_thousands < 0:
            raise ThirteenFError(f"Negative 13F reported value for CUSIP {cusip}")
        else:
            portfolio_value_total += value_thousands
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
            holdings[cusip] = {
                "issuer": issuer,
                "shares": shares,
                "value_thousands": value_thousands,
            }
        elif str(existing["issuer"]).casefold() == issuer.casefold():
            existing["shares"] = int(str(existing["shares"])) + shares
            existing["value_thousands"] = (
                int(existing["value_thousands"]) + value_thousands
                if existing["value_thousands"] is not None
                and value_thousands is not None
                else None
            )
        else:
            raise ThirteenFError(f"Conflicting issuer names for CUSIP {cusip}")

    if not entries or not holdings:
        raise ThirteenFError(
            "13F information table is empty or has no usable share rows"
        )
    portfolio_total = portfolio_value_total if portfolio_value_complete else None
    for holding in holdings.values():
        holding["portfolio_value_thousands"] = portfolio_total
    return holdings


def _portfolio_percentages(holdings: dict[str, dict[str, object]]) -> dict[str, float | None]:
    """Share of the reported 13F value, or unknown if the filing is incomplete."""
    totals = {holding.get("portfolio_value_thousands") for holding in holdings.values()}
    total = next(iter(totals)) if len(totals) == 1 else None
    if total is None or not holdings:
        return {cusip: None for cusip in holdings}
    if int(total) <= 0:
        return {cusip: None for cusip in holdings}
    return {
        cusip: (
            int(holding["value_thousands"]) / int(total) * 100.0
            if holding.get("value_thousands") is not None
            else None
        )
        for cusip, holding in holdings.items()
    }


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
        xml_files = [
            item.get("name", "")
            for item in files
            if isinstance(item, dict)
            and str(item.get("name", "")).lower().endswith(".xml")
        ]
        # SEC index.json types are generic (text.gif), and table filenames can
        # be numeric (Berkshire's 53405.xml). Identify the XML schema, not names.
        # Bound discovery rather than guessing when an index is unexpectedly large.
        if len(xml_files) > 8:
            raise ThirteenFError(f"Too many XML files to inspect safely: {index_url}")
        info_tables = []
        for name in xml_files:
            report_url = f"{directory_url}/{name}"
            content = self._get(report_url).content
            try:
                root = ET.fromstring(content)
            except (ET.ParseError, TypeError, ValueError) as exc:
                raise ThirteenFError(f"Unparseable SEC filing XML: {report_url}") from exc
            if root.tag == (
                "{http://www.sec.gov/edgar/document/thirteenf/informationtable}"
                "informationTable"
            ):
                info_tables.append((report_url, content))
        if len(info_tables) != 1:
            raise ThirteenFError(
                f"SEC filing index has {len(info_tables)} information-table XML files: {index_url}"
            )
        report_url, content = info_tables[0]
        header = self._get(f"{directory_url}/{filing['accession']}.txt").content
        filing["acceptance_datetime"] = _acceptance_datetime(content=header)
        return _parse_information_table(content), report_url

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
            ticker = self.ticker_map.get(cusip.upper()) or resolve_asset_ticker(issuer)
            if ticker is None and len(matches) == 1:
                ticker = next(iter(matches))
            if ticker is None:
                continue
            rows.append(
                {
                    "manager_key": manager_key,
                    "manager": manager["name"],
                    "ticker": ticker,
                    "event_date": quarter_end,
                    "disclosure_date": disclosure_date,
                    "acceptance_datetime": current_filing.get("acceptance_datetime"),
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
        previous_values = _portfolio_percentages(previous)
        current_values = _portfolio_percentages(current)
        previous_complete = bool(previous_values) and all(
            value is not None for value in previous_values.values()
        )
        current_complete = bool(current_values) and all(
            value is not None for value in current_values.values()
        )
        for cusip in sorted(set(current) | set(previous)):
            holding = current.get(cusip) or previous[cusip]
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

            shares_held = int(str(current.get(cusip, {}).get("shares", 0)))
            previous_shares = int(str(previous.get(cusip, {}).get("shares", 0)))
            change_shares = shares_held - previous_shares
            state = "held" if shares_held > 0 else "absent"
            current_pct = current_values.get(
                cusip, 0.0 if current_complete else None
            )
            previous_pct = previous_values.get(
                cusip, 0.0 if previous_complete else None
            )
            raw_position = json.dumps(
                {
                    "format": "13f-share-snapshot-v1",
                    "shares_held": shares_held,
                    "portfolio_pct": current_pct,
                    "portfolio_pct_basis": "reported 13F value (all information-table entries)",
                    "change_shares": change_shares,
                    "change_portfolio_pct": (
                        current_pct - previous_pct
                        if current_pct is not None and previous_pct is not None
                        else None
                    ),
                    "state": state,
                    "dollar_valuation": {
                        "status": "not_performed",
                        "reason": "share snapshot; no dollar-size comparison used",
                    },
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            transaction_type = (
                "Purchase" if change_shares > 0 else
                "Sale" if change_shares < 0 else "Position Snapshot"
            )

            ticker = matches.pop()
            rows.append(
                _canonical_row(
                    manager_name=manager["name"],
                    ticker=ticker,
                    event_date=quarter_end,
                    disclosure_date=disclosure_date,
                    added_shares=change_shares,
                    amount_midpoint=None,
                    issuer=issuer,
                    cusip=cusip,
                    accession=current_filing["accession"],
                    report_url=report_url,
                    ingestion_generation=self.ingestion_generation,
                    acceptance_datetime=current_filing.get("acceptance_datetime"),
                    transaction_type=transaction_type,
                    amount_raw=raw_position,
                    raw_transaction_subtype=_SNAPSHOT_SUBTYPE,
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
            from analyzer.transaction_repository import drop_already_stored

            transactions = drop_already_stored(self.db, transactions, source="13f")
        return self.db.upsert_transactions(transactions, source="13f")

    def fetch_and_save_all(self) -> int:
        return self.save_to_db(self.fetch_all_trades())

    def save_increases(self, increases: pd.DataFrame) -> int:
        """Persist discovery-scope increases as canonical rows.

        Discovery frames carry no dollar amounts (no price lookup at this
        scope), so amount_midpoint stays NULL and downstream sizing treats
        these rows as corroboration with unknown size.
        """
        if not (set(_INCREASE_COLUMNS) - {"acceptance_datetime"}) <= set(increases.columns):
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
                    acceptance_datetime=row.get("acceptance_datetime"),
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
