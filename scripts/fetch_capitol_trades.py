"""Fetch a fail-closed Capitol Trades reconciliation artifact.

The core client owns pagination, validation, normalization, and manifest writing.
This script never writes third-party rows into the canonical transaction database.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from analyzer.capitol_trades import BASE_URL, CapitolTradesSource


def fetch_all_trades(*, generation: str, output: str | Path) -> pd.DataFrame:
    """Fetch through the core client and require a reconciliation manifest output."""
    with CapitolTradesSource(
        data_dir="data", read_only=True, generation=generation
    ) as source:
        df = source.fetch_all_trades()
        source.write_reconciliation_artifact(output)
        return df


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Fetch Capitol Trades for reconciliation only"
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--generation", required=True)
    args = parser.parse_args(argv)

    print("=== Capitol Trades Reconciliation Fetcher ===")
    print(f"Fetching from {BASE_URL}; canonical database writes are disabled.\n")
    df = fetch_all_trades(generation=args.generation, output=args.output)
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
    print(f"\nReconciliation artifact: {args.output}")
    print("No canonical transactions were saved.")


if __name__ == "__main__":
    main()
