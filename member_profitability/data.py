"""Read-only database loading and signal computation."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from analyzer.database import Database
from analyzer.member_names import canonical_member_key
from analyzer.signals import calculate_signal_potential

from member_profitability.config import (
    DECAY_LAMBDA,
    HORIZON,
    PRICE_END_BUFFER_DAYS,
    TX_END,
    TX_START,
)


def load_transactions_and_prices(
    db_path: str | Path,
    tx_start: str | pd.Timestamp = TX_START,
    tx_end: str | pd.Timestamp = TX_END,
):
    """Return transactions and prices from an explicit, read-only database.

    Transaction selection stops at ``tx_end``. The later price boundary exists
    only to mature forward returns and is never passed to a transaction query.
    """
    path = Path(db_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Database does not exist: {path}")

    start = pd.Timestamp(tx_start).normalize()
    end = pd.Timestamp(tx_end).normalize()
    if start > end:
        raise ValueError("tx_start must be on or before tx_end")
    price_end = end + pd.Timedelta(days=PRICE_END_BUFFER_DAYS)

    db = Database(path, read_only=True)
    try:
        all_tx = db.get_transactions_by_date_range(start, end)
        if all_tx.empty:
            raise ValueError(f"No transactions between {start.date()} and {end.date()}")
        all_tx = _canonicalize_members(all_tx)
        all_tickers = sorted(
            set(
                ticker
                for ticker in all_tx["ticker"].dropna().unique()
                if isinstance(ticker, str) and ticker.strip()
            )
            | {"SPY"}
        )
        prices = db.get_prices(all_tickers, start, price_end)
        # Entry prices are transaction-bearing rows. Keep their query bounded by
        # tx_end; future market prices come only from the separate prices query.
        entry_prices = db.get_entry_prices(all_tickers, start, end)
    finally:
        db.close()

    entry_prices = _canonicalize_members(entry_prices)
    if not entry_prices.empty:
        latest = pd.to_datetime(entry_prices["disclosure_date"]).max()
        if latest > end:
            raise RuntimeError(
                f"Entry-price query crossed transaction boundary: {latest.date()} > {end.date()}"
            )
    return all_tx, prices, entry_prices, all_tickers


def _canonicalize_members(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "member" not in df.columns:
        return df.copy()
    result = df.copy()
    result["member"] = result["member"].map(canonical_member_key)
    return result[result["member"] != ""]


def compute_signals(entry_prices: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    return calculate_signal_potential(
        entry_prices,
        prices,
        [HORIZON],
        decay_lambda=DECAY_LAMBDA,
    )


def print_loaded_data(t0: float, all_tx: pd.DataFrame, all_tickers: list[str]) -> None:
    print(f"  Data loaded in {time.time() - t0:.1f}s")
    print(f"  Transactions: {len(all_tx)}, Tickers: {len(all_tickers)}")
