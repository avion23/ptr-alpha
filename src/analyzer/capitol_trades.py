"""Fail-closed Capitol Trades reconciliation client.

Capitol Trades is a third-party aggregate, not an official disclosure source. This
module fetches and normalizes its records only for reconciliation against official
House and Senate data. It deliberately refuses canonical transaction reads/writes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import tempfile
import time
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
_NULL_ID_SENTINELS = frozenset({"", "none", "null", "nan", "<null>"})


class CapitolTradesError(Exception):
    pass


class CapitolTradesSource(TransactionSource):
    """Read third-party trades for reconciliation; never persist canonical rows."""

    def __init__(
        self,
        data_dir: str | Path = "data",
        read_only: bool = True,
        db: Any | None = None,
        *,
        generation: str,
    ):
        if not isinstance(generation, str) or not generation.strip():
            raise CapitolTradesError(
                "A non-empty reconciliation generation is required"
            )
        # Retained for caller compatibility. Reconciliation does not create a data
        # directory, open a database, or mutate caller-owned database handles.
        self.data_dir = Path(data_dir)
        self.read_only = read_only
        self.db = db
        self.generation = generation.strip()
        self._last_page_artifacts: list[dict[str, Any]] = []
        self._last_source_reported: dict[str, int] | None = None
        self._last_result: pd.DataFrame | None = None
        self._last_result_metadata: dict[str, Any] | None = None
        self.session = requests.Session()
        self.session.headers["Accept"] = "application/json"

    def close(self) -> None:
        self.session.close()

    @property
    def last_page_artifacts(self) -> list[dict[str, Any]]:
        """Return a detached copy of validated page provenance."""
        return json.loads(json.dumps(self._last_page_artifacts))

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
        politician_name = politician_name.strip()
        endpoint = (
            f"{BASE_URL}/politicians/{requests.utils.quote(politician_name)}/trades"
        )
        raw_trades = self._paginate(endpoint)
        df = self._normalize(raw_trades)
        df = self._filter_dates(df, start_date, end_date)
        result = self._finalize_reconciliation(
            df,
            fetched_raw_count=len(raw_trades),
            selection={
                "politician": politician_name,
                "chamber": None,
                "start_date": start_date.isoformat() if start_date else None,
                "end_date": end_date.isoformat() if end_date else None,
            },
        )
        logger.info(
            "Fetched %d reconciliation records for %s from Capitol Trades",
            len(result),
            politician_name,
        )
        return result

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
        raw_trades = self._paginate(endpoint)
        df = self._normalize(raw_trades)
        if chamber is not None:
            df = df[df["chamber"] == chamber]
        df = self._filter_dates(df, start_date, end_date)
        result = self._finalize_reconciliation(
            df,
            fetched_raw_count=len(raw_trades),
            selection={
                "politician": None,
                "chamber": chamber,
                "start_date": start_date.isoformat() if start_date else None,
                "end_date": end_date.isoformat() if end_date else None,
            },
        )
        logger.info("Fetched %d Capitol Trades reconciliation records", len(result))
        return result

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

    def write_reconciliation_artifact(self, output: str | Path) -> Path:
        """Write only the exact retained result of the latest validated fetch."""
        output = Path(output)
        if output.exists():
            raise CapitolTradesError(
                f"Refusing to overwrite reconciliation artifact: {output}"
            )
        if not output.parent.exists():
            raise CapitolTradesError(
                f"Reconciliation artifact parent directory does not exist: {output.parent}"
            )
        if (
            not self._last_page_artifacts
            or self._last_result is None
            or self._last_result_metadata is None
        ):
            raise CapitolTradesError(
                "Cannot write artifact without a complete validated and filtered API fetch"
            )

        metadata = self._last_result_metadata
        reported_total = metadata["source_reported"]["total"]
        fetched_raw_count = metadata["fetched_raw_count"]
        accounted_count = (
            metadata["emitted_count"]
            + metadata["filtered_count"]
            + metadata["rejected_count"]
        )
        if reported_total != fetched_raw_count or fetched_raw_count != accounted_count:
            raise CapitolTradesError("Retained reconciliation count equation changed")

        result_json = self._normalized_result_json(self._last_result)
        result_sha256 = hashlib.sha256(result_json.encode()).hexdigest()
        expected_sha256 = self._last_result_metadata["normalized_result_sha256"]
        if result_sha256 != expected_sha256:
            raise CapitolTradesError("Retained reconciliation result digest changed")
        records = json.loads(result_json)
        emitted_count = self._last_result_metadata["emitted_count"]
        if len(records) != emitted_count:
            raise CapitolTradesError("Retained reconciliation result count changed")

        manifest = {
            "schema_version": 1,
            "artifact_type": "capitol_trades_reconciliation",
            "reconciliation_only": True,
            "source": "capitol_trades",
            "ingestion_generation": self.generation,
            **self._last_result_metadata,
            "pages": self._last_page_artifacts,
            "records": records,
        }
        payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=output.parent,
                prefix=f".{output.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary.write(payload)
                temporary.flush()
                os.fsync(temporary.fileno())
                temporary_name = temporary.name
            try:
                os.link(temporary_name, output)
            except FileExistsError as exc:
                raise CapitolTradesError(
                    f"Refusing to overwrite reconciliation artifact: {output}"
                ) from exc
        finally:
            if temporary_name is not None:
                Path(temporary_name).unlink(missing_ok=True)
        return output

    def _finalize_reconciliation(
        self,
        df: pd.DataFrame,
        *,
        fetched_raw_count: int,
        selection: dict[str, Any],
    ) -> pd.DataFrame:
        if self._last_source_reported is None:
            raise CapitolTradesError("Missing validated source pagination totals")
        reported_total = self._last_source_reported["total"]
        if fetched_raw_count != reported_total:
            raise CapitolTradesError(
                "Reconciliation count mismatch: "
                f"reported={reported_total}, fetched={fetched_raw_count}"
            )

        result = df.reset_index(drop=True).copy(deep=True)
        result_json = self._normalized_result_json(result)
        emitted_count = len(result)
        rejected_count = 0
        filtered_count = fetched_raw_count - emitted_count - rejected_count
        if filtered_count < 0 or (
            emitted_count + filtered_count + rejected_count != fetched_raw_count
        ):
            raise CapitolTradesError("Invalid reconciliation filter accounting")

        self._last_result = result.copy(deep=True)
        self._last_result_metadata = {
            "selection": selection,
            "source_reported": dict(self._last_source_reported),
            "fetched_raw_count": fetched_raw_count,
            "emitted_count": emitted_count,
            "filtered_count": filtered_count,
            "rejected_count": rejected_count,
            "record_count": emitted_count,
            "normalized_result_encoding": "canonical-json-records-v1",
            "normalized_result_sha256": hashlib.sha256(
                result_json.encode()
            ).hexdigest(),
        }
        return result.copy(deep=True)

    @staticmethod
    def _normalized_result_json(df: pd.DataFrame) -> str:
        records = json.loads(
            df.to_json(
                orient="records",
                date_format="iso",
                date_unit="us",
                double_precision=15,
                force_ascii=False,
            )
        )
        return json.dumps(
            records,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )

    def _paginate(self, url: str) -> list[dict]:
        """Fetch an exact, stable result set or fail without returning partial data."""
        all_trades: list[dict] = []
        page_body_hashes: set[str] = set()
        source_record_ids: set[str] = set()
        no_id_fingerprints: set[str] = set()
        page_artifacts: list[dict[str, Any]] = []
        expected_metadata: tuple[int, int, int] | None = None
        self._last_page_artifacts = []
        self._last_source_reported = None
        self._last_result = None
        self._last_result_metadata = None
        page = 1

        while True:
            params = {"page": page, "per_page": DEFAULT_PAGE_SIZE}
            try:
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                raw_body = response.content
                if not isinstance(raw_body, bytes) or not raw_body:
                    raise CapitolTradesError(
                        f"API response page {page} has no raw response bytes"
                    )
                data = response.json()
            except CapitolTradesError:
                raise
            except (requests.RequestException, ValueError) as exc:
                raise CapitolTradesError(f"API request failed: {exc}") from exc

            trades, metadata = self._validate_page(
                data, requested_page=page, expected_metadata=expected_metadata
            )
            if expected_metadata is None:
                expected_metadata = metadata
            total, pages, _ = metadata

            page_body_sha256 = hashlib.sha256(raw_body).hexdigest()
            if page_body_sha256 in page_body_hashes:
                raise CapitolTradesError(
                    f"Pagination repeated raw page bytes at page {page}"
                )
            page_body_hashes.add(page_body_sha256)
            page_artifacts.append(
                {
                    "page": page,
                    "endpoint": url,
                    "parameters": params,
                    "artifact_sha256": page_body_sha256,
                    "response_bytes": len(raw_body),
                }
            )

            for position, trade in enumerate(trades, start=1):
                source_record_id = self._raw_source_record_id(trade)
                record_fingerprint = self._trade_fingerprint(trade)
                if source_record_id is not None:
                    if source_record_id in source_record_ids:
                        raise CapitolTradesError(
                            "Pagination repeated source record ID "
                            f"{source_record_id!r} at page {page}"
                        )
                    source_record_ids.add(source_record_id)
                elif record_fingerprint in no_id_fingerprints:
                    raise CapitolTradesError(
                        "Ambiguous duplicate Capitol record without a stable source ID "
                        f"at page {page} position {position}"
                    )
                else:
                    no_id_fingerprints.add(record_fingerprint)

                annotated = dict(trade)
                annotated.update(
                    {
                        "_capitol_endpoint": url,
                        "_capitol_params": json.dumps(params, sort_keys=True),
                        "_capitol_page": page,
                        "_capitol_position": position,
                        "_capitol_artifact_sha256": page_body_sha256,
                        "_capitol_record_fingerprint": record_fingerprint,
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
        total, pages, per_page = expected_metadata
        self._last_page_artifacts = page_artifacts
        self._last_source_reported = {
            "total": total,
            "pages": pages,
            "per_page": per_page,
        }
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
        politician_name = self._normalize_model_text(trade["politician_name"])
        if politician_name is None:
            raise CapitolTradesError(
                f"API schema error at {location}: politician_name must be non-empty"
            )
        chamber = self._normalize_model_text(trade["chamber"])
        if chamber is None or chamber.casefold() not in {"house", "senate"}:
            raise CapitolTradesError(
                f"API schema error at {location}: invalid chamber {trade['chamber']!r}"
            )
        tx_type = self._normalize_model_text(trade["transaction_type"])
        if tx_type is None:
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
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise CapitolTradesError(
                    f"API schema error at {location}: {field} must be finite numeric or null"
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

        source_record_ids: set[str] = set()
        no_id_fingerprints: set[str] = set()
        rows = []
        for index, trade in enumerate(trades):
            self._validate_trade_schema(
                trade, page=trade.get("_capitol_page", 1), index=index
            )
            artifact_sha256 = trade.get("_capitol_artifact_sha256")
            if not self._is_sha256(artifact_sha256):
                raise CapitolTradesError(
                    "Normalized records require the SHA-256 of exact raw response-page bytes"
                )
            record_fingerprint = trade.get(
                "_capitol_record_fingerprint"
            ) or self._trade_fingerprint(trade)
            source_record_id = self._raw_source_record_id(trade)
            if source_record_id is not None:
                if source_record_id in source_record_ids:
                    raise CapitolTradesError(
                        f"Duplicate source record ID {source_record_id!r} during normalization"
                    )
                source_record_ids.add(source_record_id)
            elif record_fingerprint in no_id_fingerprints:
                raise CapitolTradesError(
                    "Ambiguous duplicate Capitol record without a stable source ID"
                )
            else:
                no_id_fingerprints.add(record_fingerprint)

            amount_min = trade.get("amount_min")
            amount_max = trade.get("amount_max")
            midpoint = self._compute_midpoint(amount_min, amount_max)
            if midpoint is None and trade.get("amount_text"):
                _, midpoint = _parse_amount_midpoint(trade["amount_text"])

            raw_tx_type = self._normalize_model_text(trade["transaction_type"])
            if raw_tx_type is None:
                raise CapitolTradesError("Missing normalized transaction subtype")
            tx_type = TX_TYPE_MAP.get(raw_tx_type.casefold(), raw_tx_type.title())
            chamber = self._normalize_model_text(trade["chamber"])
            if chamber is None:
                raise CapitolTradesError("Missing normalized chamber")
            chamber = chamber.casefold()
            source_filing_id = self._normalize_filing_id(trade.get("doc_id"))
            doc_id = source_filing_id or self._synthetic_doc_id(
                chamber=chamber,
                source_record_id=source_record_id,
                record_fingerprint=record_fingerprint,
            )
            disclosure_date = self._parse_date(trade["disclosure_date"])
            asset_name = self._normalize_model_text(trade.get("asset_name"))
            asset_type = self._normalize_model_text(trade.get("asset_type"))
            ticker = self._normalize_model_text(trade.get("ticker"))
            if ticker is not None:
                ticker = ticker.upper()
            member = self._normalize_model_text(trade["politician_name"])
            if member is None:
                raise CapitolTradesError("Missing normalized politician name")
            rows.append(
                {
                    "doc_id": doc_id,
                    "member": member,
                    "ticker": ticker,
                    "transaction_date": self._parse_date(trade["transaction_date"]),
                    "disclosure_date": disclosure_date,
                    "transaction_type": tx_type,
                    "owner_code": None,
                    "amount_raw": trade.get("amount_text"),
                    "amount_midpoint": midpoint,
                    "instrument_type": self._normalize_instrument_type(asset_type),
                    "strike_price": None,
                    "expiry_date": None,
                    "asset_description": asset_name,
                    "chamber": chamber,
                    "source_record_id": source_record_id,
                    # This aggregator cannot establish official availability dates.
                    "official_filing_date": None,
                    "available_date": None,
                    "notification_date": None,
                    "amends_source_record_id": None,
                    "raw_transaction_subtype": trade["transaction_type"],
                    "ticker_origin": "source_reported" if ticker is not None else None,
                    "raw_asset_class": trade.get("asset_type"),
                    "raw_asset_description": trade.get("asset_name"),
                    "ingestion_generation": self.generation,
                    "artifact_sha256": artifact_sha256,
                    "state": self._normalize_model_upper_text(trade.get("state")),
                    "party": self._normalize_model_upper_text(trade.get("party")),
                    "filing_url": self._normalize_model_text(trade.get("filing_url")),
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
        values: set[str] = set()
        present: list[tuple[str, Any]] = []
        for field in _SOURCE_ID_FIELDS:
            if field not in trade or trade[field] is None:
                continue
            value = trade[field]
            present.append((field, value))
            if isinstance(value, bool) or not isinstance(value, (str, int)):
                raise CapitolTradesError(
                    "API schema error: source record IDs must be strings or integers: "
                    f"{present!r}"
                )
            normalized = cls._normalize_id_text(value)
            if normalized is not None:
                values.add(normalized)
        if not values:
            return None
        if len(values) != 1:
            raise CapitolTradesError(
                f"API schema error: conflicting source record IDs {present!r}"
            )
        return next(iter(values))

    @classmethod
    def _trade_fingerprint(cls, trade: dict) -> str:
        payload = json.dumps(
            cls._normalized_identity_payload(trade),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(payload).hexdigest()

    @classmethod
    def _normalized_identity_payload(cls, trade: dict) -> dict[str, Any]:
        transaction_date = cls._parse_date(trade.get("transaction_date"))
        disclosure_date = cls._parse_date(trade.get("disclosure_date"))
        return {
            "source_record_id": cls._raw_source_record_id(trade),
            "source_filing_id": cls._normalize_filing_id(trade.get("doc_id")),
            "politician_name": cls._canonical_identity_text(
                trade.get("politician_name")
            ),
            "chamber": cls._canonical_identity_text(trade.get("chamber")),
            "state": cls._canonical_identity_text(trade.get("state")),
            "party": cls._canonical_identity_text(trade.get("party")),
            "ticker": cls._canonical_identity_text(trade.get("ticker")),
            "asset_name": cls._canonical_identity_text(trade.get("asset_name")),
            "asset_type": cls._canonical_identity_text(trade.get("asset_type")),
            "transaction_type": cls._canonical_identity_text(
                trade.get("transaction_type")
            ),
            "transaction_date": (
                transaction_date.isoformat() if transaction_date is not None else None
            ),
            "disclosure_date": (
                disclosure_date.isoformat() if disclosure_date is not None else None
            ),
            "amount_text": cls._canonical_identity_text(trade.get("amount_text")),
            "amount_min": cls._normalize_identity_number(trade.get("amount_min")),
            "amount_max": cls._normalize_identity_number(trade.get("amount_max")),
            "filing_url": cls._canonical_identity_text(trade.get("filing_url")),
        }

    @staticmethod
    def _normalize_identity_number(value: Any) -> float | None:
        if value is None:
            return None
        return float(value)

    @staticmethod
    def _is_sha256(value: Any) -> bool:
        if not isinstance(value, str) or len(value) != 64:
            return False
        return all(character in "0123456789abcdef" for character in value)

    @classmethod
    def _normalize_filing_id(cls, value: Any) -> str | None:
        return cls._normalize_id_text(value)

    @staticmethod
    def _normalize_id_text(value: Any) -> str | None:
        """Normalize only fields whose domain is an identifier."""
        if value is None:
            return None
        normalized = " ".join(str(value).split())
        if normalized.casefold() in _NULL_ID_SENTINELS:
            return None
        return normalized

    @staticmethod
    def _normalize_model_text(value: Any) -> str | None:
        """Trim modeling text without interpreting words such as NAN or null."""
        if value is None:
            return None
        normalized = " ".join(str(value).split())
        return normalized or None

    @staticmethod
    def _canonical_identity_text(value: Any) -> str | None:
        """Canonicalize non-ID identity text without applying ID sentinels."""
        if value is None:
            return None
        return " ".join(str(value).split()).casefold()

    @classmethod
    def _normalize_model_upper_text(cls, value: Any) -> str | None:
        normalized = cls._normalize_model_text(value)
        return normalized.upper() if normalized is not None else None

    @staticmethod
    def _synthetic_doc_id(
        *,
        chamber: str,
        source_record_id: str | None,
        record_fingerprint: str,
    ) -> str:
        stable_identity = (
            f"source-id:{source_record_id}"
            if source_record_id is not None
            else f"record-fingerprint:{record_fingerprint}"
        )
        digest = hashlib.sha256(f"{chamber}|{stable_identity}".encode()).hexdigest()[
            :20
        ]
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
    def _normalize_instrument_type(raw: str | None) -> str | None:
        if raw is None:
            return None
        lower = raw.casefold()
        if "call" in lower:
            return "call"
        if "put" in lower:
            return "put"
        if "stock" in lower:
            return "stock"
        return lower
