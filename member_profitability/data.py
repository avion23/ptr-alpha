"""Database loading and signal computation for member profitability analysis."""

from __future__ import annotations

import time

import pandas as pd

from analyzer.database import Database
from analyzer.signals import calculate_signal_potential

from member_profitability.config import (
    DECAY_LAMBDA,
    HORIZON,
    PRICE_END_BUFFER_DAYS,
    TX_END,
    TX_START,
)


def load_transactions_and_prices():
    """Open the DB and return (all_tx, prices, entry_prices, all_tickers)."""
    db = Database("data/congress.duckdb")
    tx_start = pd.Timestamp(TX_START)
    tx_end = pd.Timestamp(TX_END)
    all_tx = db.get_transactions_by_date_range(tx_start, tx_end)

    price_end = tx_end + pd.Timedelta(days=PRICE_END_BUFFER_DAYS)
    all_tickers = sorted(
        set(t for t in all_tx["ticker"].dropna().unique() if isinstance(t, str)) | {"SPY"}
    )

    prices = db.get_prices(all_tickers, tx_start, price_end)
    entry_prices = db.get_entry_prices(all_tickers, tx_start, price_end)
    return all_tx, prices, entry_prices, all_tickers


def compute_signals(entry_prices: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    return calculate_signal_potential(
        entry_prices, prices, [HORIZON], decay_lambda=DECAY_LAMBDA,
    )


def print_loaded_data(t0: float, all_tx: pd.DataFrame, all_tickers: list[str]) -> None:
    print(f"  Data loaded in {time.time()-t0:.1f}s")
    print(f"  Transactions: {len(all_tx)}, Tickers: {len(all_tickers)}")
