"""Capitol Trades API client — backup data source for congressional trading data.

Fetches from https://trades.telep.io/api (free, no auth).
Supports both per-politician and global trade listing with pagination.
"""

import logging
from datetime import date
from pathlib import Path

import pandas as pd
import requests

from analyzer.database import Database
from analyzer.interfaces import TransactionSource

logger = logging.getLogger(__name__)

BASE_URL = "https://trades.telep.io/api"
DEFAULT_PAGE_SIZE = 50

# Mapping from API transaction_type values to our canonical types
TX_TYPE_MAP = {
    "purchase": "Purchase",
    "sale": "Sale Full",
    "sale (full)": "Sale Full",
    "sale (partial)": "Sale Partial",
    "exchange": "Exchange",
}


class CapitolTradesError(Exception):
    pass


class CapitolTradesSource(TransactionSource):
    """Fetches congressional trading data from the Capitol Trades API."""

    def __init__(self, data_dir: str | Path = "data", read_only: bool = False):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db = Database(self.data_dir / "congress.duckdb", read_only=read_only)
        self.session = requests.Session()
        self.session.headers["Accept"] = "application/json"

    def close(self) -> None:
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        return False

    def get_transactions(self, year: int) -> pd.DataFrame:
        """TransactionSource interface — returns transactions for a given year."""
        df = self.db.get_transactions(year)
        if df.empty:
            raise CapitolTradesError(
                f"No cached Capitol Trades data for {year}. "
                "Run 'ptr-alpha fetch-capitol' first."
            )
        logger.info(f"Loaded {len(df)} cached Capitol Trades transactions for {year}")
        return df

    def fetch_trades(
        self,
        politician_name: str,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> pd.DataFrame:
        """Fetch trades for a specific politician."""
        all_trades = self._paginate(
            f"{BASE_URL}/politicians/{requests.utils.quote(politician_name)}/trades"
        )
        df = self._normalize(all_trades)
        df = self._filter_dates(df, start_date, end_date)
        logger.info(
            f"Fetched {len(df)} trades for {politician_name} from Capitol Trades API"
        )
        return df

    def fetch_all_trades(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
        chamber: str | None = None,
    ) -> pd.DataFrame:
        """Fetch all recent trades, optionally filtered by chamber."""
        all_trades = self._paginate(f"{BASE_URL}/trades")
        if chamber:
            all_trades = [t for t in all_trades if t.get("chamber") == chamber]
        df = self._normalize(all_trades)
        df = self._filter_dates(df, start_date, end_date)
        logger.info(f"Fetched {len(df)} trades from Capitol Trades API")
        return df

    def save_to_db(self, df: pd.DataFrame) -> int:
        """Upsert trades into the database. Returns count inserted."""
        if df.empty:
            return 0
        self.db.upsert_transactions(df)
        logger.info(f"Saved {len(df)} Capitol Trades transactions to database")
        return len(df)

    def fetch_and_save_politician(
        self,
        politician_name: str,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> int:
        """Fetch politician trades and save to DB. Returns count saved."""
        df = self.fetch_trades(politician_name, start_date, end_date)
        return self.save_to_db(df)

    def fetch_and_save_all(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
        chamber: str | None = None,
    ) -> int:
        """Fetch all trades and save to DB. Returns count saved."""
        df = self.fetch_all_trades(start_date, end_date, chamber)
        return self.save_to_db(df)

    def _paginate(self, url: str) -> list[dict]:
        """Fetch all pages from a paginated API endpoint."""
        all_trades: list[dict] = []
        page = 1
        per_page = DEFAULT_PAGE_SIZE

        while True:
            params = {"page": page, "per_page": per_page}
            try:
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
            except requests.RequestException as e:
                raise CapitolTradesError(f"API request failed: {e}")

            data = response.json()
            trades = data.get("trades", [])
            all_trades.extend(trades)

            total_pages = data.get("pages", 1)
            logger.debug(
                f"Page {page}/{total_pages}: fetched {len(trades)} trades "
                f"(total so far: {len(all_trades)}/{data.get('total', '?')})"
            )

            if page >= total_pages:
                break
            page += 1

        return all_trades

    def _normalize(self, trades: list[dict]) -> pd.DataFrame:
        """Convert API response trades to our canonical schema."""
        if not trades:
            return pd.DataFrame(
                columns=[
                    "doc_id", "member", "ticker", "transaction_date",
                    "disclosure_date", "transaction_type", "owner_code",
                    "amount_raw", "amount_midpoint", "instrument_type",
                    "strike_price", "expiry_date",
                ]
            )

        rows = []
        for t in trades:
            amount_min = t.get("amount_min")
            amount_max = t.get("amount_max")
            midpoint = self._compute_midpoint(amount_min, amount_max)

            tx_type_raw = t.get("transaction_type", "")
            tx_type = TX_TYPE_MAP.get(tx_type_raw, tx_type_raw.title())

            rows.append({
                "doc_id": str(t.get("doc_id", "")),
                "member": t.get("politician_name", ""),
                "ticker": t.get("ticker"),
                "transaction_date": self._parse_date(t.get("transaction_date")),
                "disclosure_date": self._parse_date(t.get("disclosure_date")),
                "transaction_type": tx_type,
                "owner_code": None,
                "amount_raw": t.get("amount_text"),
                "amount_midpoint": midpoint,
                "instrument_type": t.get("asset_type"),
                "strike_price": None,
                "expiry_date": None,
            })

        df = pd.DataFrame(rows)
        # Drop rows with missing critical fields
        df = df.dropna(subset=["doc_id", "member", "transaction_date"])
        return df

    def _filter_dates(
        self,
        df: pd.DataFrame,
        start_date: date | None,
        end_date: date | None,
    ) -> pd.DataFrame:
        """Filter dataframe to date range based on disclosure_date."""
        if df.empty:
            return df
        if start_date:
            df = df[df["disclosure_date"] >= pd.Timestamp(start_date)]
        if end_date:
            df = df[df["disclosure_date"] <= pd.Timestamp(end_date)]
        return df

    @staticmethod
    def _compute_midpoint(amount_min, amount_max) -> float | None:
        """Compute midpoint from API amount range, or return None."""
        if amount_min is not None and amount_max is not None:
            try:
                return (float(amount_min) + float(amount_max)) / 2.0
            except (TypeError, ValueError):
                return None
        return None

    @staticmethod
    def _parse_date(value) -> pd.Timestamp | None:
        """Parse a date string into a Timestamp."""
        if not value:
            return None
        try:
            return pd.Timestamp(value)
        except Exception:
            return None
