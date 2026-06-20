"""Fetch ALL trades from Capitol Trades API and save to database.

Uses per_page=200 for efficient pagination.
"""

import sys
import time
from pathlib import Path

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import requests

from analyzer.capitol_trades import CapitolTradesSource

BASE_URL = "https://trades.telep.io/api"
PER_PAGE = 200


def fetch_all_trades() -> list[dict]:
    """Fetch all trades from the API using per_page=200 pagination."""
    session = requests.Session()
    session.headers["Accept"] = "application/json"

    all_trades: list[dict] = []
    page = 1

    while True:
        params = {"page": page, "per_page": PER_PAGE}
        resp = session.get(f"{BASE_URL}/trades", params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        trades = data.get("trades", [])
        all_trades.extend(trades)

        total_pages = data.get("pages", 1)
        total = data.get("total", "?")
        print(f"  Page {page}/{total_pages}: +{len(trades)} trades (cumulative: {len(all_trades)}/{total})")

        if page >= total_pages:
            break
        page += 1
        time.sleep(0.3)  # be polite

    return all_trades


def main():
    print("=== Capitol Trades Fetcher ===")
    print(f"Fetching from {BASE_URL} (per_page={PER_PAGE})\n")

    # 1. Fetch all trades from API
    t0 = time.time()
    raw_trades = fetch_all_trades()
    fetch_time = time.time() - t0
    print(f"\nFetched {len(raw_trades)} raw trades in {fetch_time:.1f}s")

    # 2. Normalize and save using CapitolTradesSource
    df = None
    with CapitolTradesSource(data_dir="data") as source:
        df = source._normalize(raw_trades)
        print(f"Normalized to {len(df)} trades (dropped invalid rows)")

        saved = source.save_to_db(df)
        print(f"Saved {saved} trades to database")

    # 3. Summary
    print("\n--- Summary ---")
    if df is not None and not df.empty:
        print(f"Unique members:  {df['member'].nunique()}")
        print(f"Unique tickers:  {df['ticker'].nunique()}")
        print(f"Date range:      {df['transaction_date'].min()} to {df['transaction_date'].max()}")
        tx_types = df["transaction_type"].value_counts()
        print("Transaction types:")
        for t, count in tx_types.items():
            print(f"  {t}: {count}")
    print("\nDone.")


if __name__ == "__main__":
    main()
