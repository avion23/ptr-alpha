"""Fail-closed Capitol Trades reconciliation client.

Capitol Trades is a third-party aggregate, not an official disclosure source.  This
module fetches and normalizes its records only for reconciliation against official
House and Senate data.  It deliberately refuses to write canonical transactions.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from analyzer.interfaces import TransactionSource
from analyzer.parsing.cells import _extract_amount_midpoint as _parse_amount_midpoint

logger = logging.getLogger(__name__)

BASE_URL = "https://trades.telep.io/api"
DEFAULT_PAGE_SIZE = 50
PAGE_DELAY_SECONDS = 0.3

TX_TYPE_MAP = {
    "purchase": "Purchase",
    "sale": "Sale",
    "sale (full)": "Sale",
    "sale (partial)": "Sale",
    "exchange": "Exchange",
}

_REQUIRED_TRADE_FIELDS = frozenset(
    {
        "politician_name",
        "chamber",
        "state",
        "party",
        "ticker",
        "asset_name",
        "asset_type",
        "transaction_type",
        "transaction_date",
        "disclosure_date",
        "amount_text",
        "amount_min",
        "amount_max",
        "filing_url",
        "doc_id",
    }
)
_SOURCE_ID_FIELDS = ("id", "trade_id", "transaction_id")
_INTERNAL_PREFIX = "_capitol_"


class CapitolTradesError(Exception):
    pass


class CapitolTradesSource(TransactionSource):
    """Read third-party trades for reconciliation; never persist them as canonical."""

    def __init__(
        self,
        data_dir: str | Path = "data",
        read_only: bool = True,
        db: Any | None = None,
    ):
        # These attributes remain for caller compatibility.  Reconciliation does
        # not create a data directory, open a database, or mutate caller-owned DBs.
        self.data_dir = Path(data_dir)
        self.read_only = read_only
        self.db = db
        self.session = requests.Session()
        self.session.headers["Accept"] = "application/json"

    def close(self) -> None:
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        return False

    def get_transactions(self, year: int) -> pd.DataFrame:
        raise CapitolTradesError(
            "Capitol Trades is reconciliation-only and cannot serve canonical transactions"
        )

    def fetch_trades(
        self,
        politician_name: str,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> pd.DataFrame:
        endpoint = (
            f"{BASE_URL}/politicians/{requests.utils.quote(politician_name)}/trades"
        )
        df = self._normalize(self._paginate(endpoint))
        df = self._filter_dates(df, start_date, end_date)
        logger.info(
            "Fetched %d reconciliation records for %s from Capitol Trades",
            len(df),
            politician_name,
        )
        return df

    def fetch_all_trades(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
        chamber: str | None = None,
    ) -> pd.DataFrame:
        if chamber is not None:
            chamber = chamber.lower().strip()
            if chamber not in {"house", "senate"}:
                raise CapitolTradesError(
                    f"Invalid chamber {chamber!r}; expected 'house' or 'senate'"
                )

        endpoint = f"{BASE_URL}/trades"
        all_trades = self._paginate(endpoint)
        if chamber is not None:
            all_trades = [t for t in all_trades if t["chamber"].lower() == chamber]
        df = self._normalize(all_trades)
        df = self._filter_dates(df, start_date, end_date)
        logger.info("Fetched %d Capitol Trades reconciliation records", len(df))
        return df

    def save_to_db(self, df: pd.DataFrame) -> int:
        raise CapitolTradesError(
            "Capitol Trades is reconciliation-only; canonical database writes are forbidden"
        )

    def fetch_and_save_politician(
        self,
        politician_name: str,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> int:
        df = self.fetch_trades(politician_name, start_date, end_date)
        return self.save_to_db(df)

    def fetch_and_save_all(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
        chamber: str | None = None,
    ) -> int:
        df = self.fetch_all_trades(start_date, end_date, chamber)
        return self.save_to_db(df)

    def _paginate(self, url: str) -> list[dict]:
        """Fetch an exact, stable result set or fail without returning partial data."""
        all_trades: list[dict] = []
        page_fingerprints: set[str] = set()
        source_record_ids: set[str] = set()
        artifact_occurrences: Counter[str] = Counter()
        expected_metadata: tuple[int, int, int] | None = None
        page = 1

        while True:
            params = {"page": page, "per_page": DEFAULT_PAGE_SIZE}
            try:
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
            except (requests.RequestException, ValueError) as exc:
                raise CapitolTradesError(f"API request failed: {exc}") from exc

            trades, metadata = self._validate_page(
                data, requested_page=page, expected_metadata=expected_metadata
            )
            if expected_metadata is None:
                expected_metadata = metadata
            total, pages, _ = metadata

            page_fingerprint = self._artifact_sha256(trades)
            if trades and page_fingerprint in page_fingerprints:
                raise CapitolTradesError(
                    f"Pagination repeated page content at page {page}; refusing partial data"
                )
            page_fingerprints.add(page_fingerprint)

            for position, trade in enumerate(trades, start=1):
                source_record_id = self._raw_source_record_id(trade)
                if source_record_id is not None:
                    if source_record_id in source_record_ids:
                        raise CapitolTradesError(
                            "Pagination repeated source record ID "
                            f"{source_record_id!r} at page {page}"
                        )
                    source_record_ids.add(source_record_id)

                artifact_sha256 = self._trade_artifact_sha256(trade)
                artifact_occurrences[artifact_sha256] += 1
                annotated = dict(trade)
                annotated.update(
                    {
                        "_capitol_endpoint": url,
                        "_capitol_params": json.dumps(params, sort_keys=True),
                        "_capitol_page": page,
                        "_capitol_position": position,
                        "_capitol_artifact_sha256": artifact_sha256,
                        "_capitol_occurrence": artifact_occurrences[artifact_sha256],
                    }
                )
                all_trades.append(annotated)

            logger.debug(
                "Validated Capitol Trades page %d/%d (%d/%d records)",
                page,
                pages,
                len(all_trades),
                total,
            )
            if page >= pages:
                break
            page += 1
            time.sleep(PAGE_DELAY_SECONDS)

        if expected_metadata is None:
            raise CapitolTradesError("API returned no pagination metadata")
        total, _, _ = expected_metadata
        if len(all_trades) != total:
            raise CapitolTradesError(
                f"Incomplete API response: expected {total} records, received {len(all_trades)}"
            )
        return all_trades

    def _validate_page(
        self,
        data: Any,
        *,
        requested_page: int,
        expected_metadata: tuple[int, int, int] | None,
    ) -> tuple[list[dict], tuple[int, int, int]]:
        if not isinstance(data, dict):
            raise CapitolTradesError("API schema error: response must be an object")
        missing = {"trades", "page", "per_page", "pages", "total"} - data.keys()
        if missing:
            raise CapitolTradesError(
                f"API schema error: missing response fields {sorted(missing)}"
            )

        page = self._strict_int(data["page"], "page", minimum=1)
        per_page = self._strict_int(data["per_page"], "per_page", minimum=1)
        pages = self._strict_int(data["pages"], "pages", minimum=1)
        total = self._strict_int(data["total"], "total", minimum=0)
        if page != requested_page:
            raise CapitolTradesError(
                f"Pagination page mismatch: requested {requested_page}, received {page}"
            )

        calculated_pages = max(1, math.ceil(total / per_page))
        if pages != calculated_pages:
            raise CapitolTradesError(
                "Pagination metadata mismatch: "
                f"total={total}, per_page={per_page} requires pages={calculated_pages}, "
                f"received pages={pages}"
            )
        metadata = (total, pages, per_page)
        if expected_metadata is not None and metadata != expected_metadata:
            raise CapitolTradesError(
                "Pagination metadata changed during fetch: "
                f"expected {expected_metadata}, received {metadata}"
            )

        trades = data["trades"]
        if not isinstance(trades, list):
            raise CapitolTradesError("API schema error: trades must be a list")
        expected_count = min(per_page, max(total - ((page - 1) * per_page), 0))
        if len(trades) != expected_count:
            raise CapitolTradesError(
                f"Incomplete page {page}: expected {expected_count} records, "
                f"received {len(trades)}"
            )
        for index, trade in enumerate(trades):
            self._validate_trade_schema(trade, page=page, index=index)
        return trades, metadata

    def _validate_trade_schema(self, trade: Any, *, page: int, index: int) -> None:
        location = f"page {page} record {index}"
        if not isinstance(trade, dict):
            raise CapitolTradesError(
                f"API schema error at {location}: record must be an object"
            )
        missing = _REQUIRED_TRADE_FIELDS - trade.keys()
        if missing:
            raise CapitolTradesError(
                f"API schema error at {location}: missing fields {sorted(missing)}"
            )
        if (
            not isinstance(trade["politician_name"], str)
            or not trade["politician_name"].strip()
        ):
            raise CapitolTradesError(
                f"API schema error at {location}: politician_name must be non-empty"
            )
        chamber = trade["chamber"]
        if not isinstance(chamber, str) or chamber.lower() not in {"house", "senate"}:
            raise CapitolTradesError(
                f"API schema error at {location}: invalid chamber {chamber!r}"
            )
        tx_type = trade["transaction_type"]
        if not isinstance(tx_type, str) or not tx_type.strip():
            raise CapitolTradesError(
                f"API schema error at {location}: transaction_type must be non-empty"
            )
        for field in ("transaction_date", "disclosure_date"):
            if self._parse_date(trade[field]) is None:
                raise CapitolTradesError(
                    f"API schema error at {location}: invalid {field} {trade[field]!r}"
                )
        for field in (
            "state",
            "party",
            "ticker",
            "asset_name",
            "asset_type",
            "amount_text",
            "filing_url",
        ):
            if trade[field] is not None and not isinstance(trade[field], str):
                raise CapitolTradesError(
                    f"API schema error at {location}: {field} must be a string or null"
                )
        if trade["doc_id"] is not None and (
            isinstance(trade["doc_id"], bool)
            or not isinstance(trade["doc_id"], (str, int))
        ):
            raise CapitolTradesError(
                f"API schema error at {location}: doc_id must be a string, integer, or null"
            )
        for field in ("amount_min", "amount_max"):
            value = trade[field]
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, (int, float))
            ):
                raise CapitolTradesError(
                    f"API schema error at {location}: {field} must be numeric or null"
                )
        self._raw_source_record_id(trade)

    def _normalize(self, trades: list[dict]) -> pd.DataFrame:
        columns = [
            "doc_id",
            "member",
            "ticker",
            "transaction_date",
            "disclosure_date",
            "transaction_type",
            "owner_code",
            "amount_raw",
            "amount_midpoint",
            "instrument_type",
            "strike_price",
            "expiry_date",
            "asset_description",
            "chamber",
            "source_record_id",
            "official_filing_date",
            "available_date",
            "notification_date",
            "amends_source_record_id",
            "raw_transaction_subtype",
            "ticker_origin",
            "raw_asset_class",
            "raw_asset_description",
            "ingestion_generation",
            "artifact_sha256",
            "state",
            "party",
            "filing_url",
            "source_endpoint",
            "source_params",
            "source_page",
            "source_position",
            "source_filing_id",
        ]
        if not trades:
            return pd.DataFrame(columns=columns)

        artifact_occurrences: Counter[str] = Counter()
        rows = []
        for index, trade in enumerate(trades):
            self._validate_trade_schema(
                trade, page=trade.get("_capitol_page", 1), index=index
            )
            artifact_sha256 = trade.get(
                "_capitol_artifact_sha256"
            ) or self._trade_artifact_sha256(trade)
            occurrence = trade.get("_capitol_occurrence")
            if occurrence is None:
                artifact_occurrences[artifact_sha256] += 1
                occurrence = artifact_occurrences[artifact_sha256]

            amount_min = trade.get("amount_min")
            amount_max = trade.get("amount_max")
            midpoint = self._compute_midpoint(amount_min, amount_max)
            if midpoint is None and trade.get("amount_text"):
                _, midpoint = _parse_amount_midpoint(trade["amount_text"])

            raw_tx_type = trade["transaction_type"].strip()
            tx_type = TX_TYPE_MAP.get(raw_tx_type.lower(), raw_tx_type.title())
            source_record_id = self._raw_source_record_id(trade)
            chamber = trade["chamber"].lower()
            raw_doc_id = trade.get("doc_id")
            source_filing_id = str(raw_doc_id) if raw_doc_id is not None else None
            doc_id = source_filing_id or self._synthetic_doc_id(
                chamber=chamber,
                source_record_id=source_record_id,
                artifact_sha256=artifact_sha256,
                occurrence=int(occurrence),
            )
            disclosure_date = self._parse_date(trade["disclosure_date"])
            asset_name = trade.get("asset_name")
            rows.append(
                {
                    "doc_id": doc_id,
                    "member": trade["politician_name"].strip(),
                    "ticker": trade.get("ticker"),
                    "transaction_date": self._parse_date(trade["transaction_date"]),
                    "disclosure_date": disclosure_date,
                    "transaction_type": tx_type,
                    "owner_code": None,
                    "amount_raw": trade.get("amount_text"),
                    "amount_midpoint": midpoint,
                    "instrument_type": self._normalize_instrument_type(
                        trade.get("asset_type")
                    ),
                    "strike_price": None,
                    "expiry_date": None,
                    "asset_description": asset_name,
                    "chamber": chamber,
                    "source_record_id": source_record_id,
                    # Capitol Trades is not an official source. Preserve its raw
                    # disclosure_date above, but do not manufacture official dates.
                    "official_filing_date": None,
                    "available_date": None,
                    "notification_date": None,
                    "amends_source_record_id": None,
                    "raw_transaction_subtype": raw_tx_type,
                    "ticker_origin": "capitol_trades_api"
                    if trade.get("ticker")
                    else None,
                    "raw_asset_class": trade.get("asset_type"),
                    "raw_asset_description": asset_name,
                    "ingestion_generation": None,
                    "artifact_sha256": artifact_sha256,
                    "state": trade.get("state"),
                    "party": trade.get("party"),
                    "filing_url": trade.get("filing_url"),
                    "source_endpoint": trade.get("_capitol_endpoint"),
                    "source_params": trade.get("_capitol_params"),
                    "source_page": trade.get("_capitol_page"),
                    "source_position": trade.get("_capitol_position"),
                    "source_filing_id": source_filing_id,
                }
            )
        return pd.DataFrame(rows, columns=columns)

    def _filter_dates(
        self,
        df: pd.DataFrame,
        start_date: date | None,
        end_date: date | None,
    ) -> pd.DataFrame:
        if start_date is not None and end_date is not None and end_date < start_date:
            raise CapitolTradesError("end_date must be on or after start_date")
        if df.empty:
            return df
        if start_date:
            df = df[df["disclosure_date"] >= pd.Timestamp(start_date)]
        if end_date:
            df = df[df["disclosure_date"] <= pd.Timestamp(end_date)]
        return df

    @staticmethod
    def _strict_int(value: Any, field: str, *, minimum: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise CapitolTradesError(
                f"API schema error: {field} must be an integer >= {minimum}"
            )
        return value

    @classmethod
    def _raw_source_record_id(cls, trade: dict) -> str | None:
        present = [
            (field, trade[field])
            for field in _SOURCE_ID_FIELDS
            if trade.get(field) is not None
        ]
        if not present:
            return None
        if any(
            isinstance(value, bool) or not isinstance(value, (str, int))
            for _, value in present
        ):
            raise CapitolTradesError(
                f"API schema error: source record IDs must be strings or integers: {present!r}"
            )
        values = {str(value) for _, value in present}
        if len(values) != 1:
            raise CapitolTradesError(
                f"API schema error: conflicting source record IDs {present!r}"
            )
        value = next(iter(values)).strip()
        if not value:
            raise CapitolTradesError(
                "API schema error: source record ID must be non-empty"
            )
        return value

    @classmethod
    def _trade_artifact_sha256(cls, trade: dict) -> str:
        raw = {k: v for k, v in trade.items() if not k.startswith(_INTERNAL_PREFIX)}
        return cls._artifact_sha256(raw)

    @staticmethod
    def _artifact_sha256(value: Any) -> str:
        payload = json.dumps(
            value, sort_keys=True, separators=(",", ":"), default=str
        ).encode()
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _synthetic_doc_id(
        *,
        chamber: str,
        source_record_id: str | None,
        artifact_sha256: str,
        occurrence: int,
    ) -> str:
        stable_identity = (
            f"source-id:{source_record_id}"
            if source_record_id is not None
            else f"artifact:{artifact_sha256}:occurrence:{occurrence}"
        )
        components = f"{chamber}|{stable_identity}"
        digest = hashlib.sha256(components.encode()).hexdigest()[:20]
        return f"ct-{chamber}-{digest}"

    @staticmethod
    def _compute_midpoint(amount_min, amount_max) -> float | None:
        if amount_min is not None and amount_max is not None:
            try:
                return (float(amount_min) + float(amount_max)) / 2.0
            except (TypeError, ValueError):
                return None
        return None

    @staticmethod
    def _parse_date(value) -> pd.Timestamp | None:
        if not value:
            return None
        try:
            parsed = pd.Timestamp(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if pd.isna(parsed):
            return None
        return parsed

    @staticmethod
    def _normalize_instrument_type(raw: str | None) -> str:
        if not raw:
            return "stock"
        lower = raw.lower()
        if "call" in lower:
            return "call"
        if "put" in lower:
            return "put"
        return "stock"
