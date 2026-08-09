"""Fetch Capitol Trades records for reconciliation only.

The core client owns pagination and validation.  This script never writes the
third-party aggregate into the canonical congressional transaction database.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from analyzer.capitol_trades import BASE_URL, CapitolTradesSource


def fetch_all_trades() -> pd.DataFrame:
    """Return a fail-closed, normalized reconciliation frame via the core client."""
    with CapitolTradesSource(data_dir="data", read_only=True) as source:
        return source.fetch_all_trades()


def main() -> None:
    print("=== Capitol Trades Reconciliation Fetcher ===")
    print(f"Fetching from {BASE_URL}; canonical database writes are disabled.\n")

    df = fetch_all_trades()
    print(f"Validated {len(df)} reconciliation records")
    if not df.empty:
        print(f"Unique members:  {df['member'].nunique()}")
        print(f"Unique tickers:  {df['ticker'].nunique()}")
        print(
            f"Date range:      {df['transaction_date'].min()} to "
            f"{df['transaction_date'].max()}"
        )
        print("Chambers:")
        for chamber, count in df["chamber"].value_counts().items():
            print(f"  {chamber}: {count}")
        print("Transaction types:")
        for tx_type, count in df["transaction_type"].value_counts().items():
            print(f"  {tx_type}: {count}")
    print("\nReconciliation only: no canonical transactions were saved.")


if __name__ == "__main__":
    main()
