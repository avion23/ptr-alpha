"""Fetch a fail-closed Capitol Trades reconciliation staging artifact.

Reconciliation only: this script never writes third-party rows into the
canonical transaction database. The core client owns pagination, validation,
normalization, and the reconciliation manifest (which includes raw page
SHA-256 hashes and the selection/accounting counts). This script only stages
that artifact to --output for later reconciliation.

The live endpoint is currently unavailable (HTTP 503). When any part of the
fetch fails before an artifact exists, the script fails closed: no artifact is
written and a scheduled retry marker is recorded so an operator scheduler can
retry the staging fetch later.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from analyzer.capitol_trades import BASE_URL, CapitolTradesError, CapitolTradesSource

RETRY_MARKER_SCHEMA_VERSION = 1
RETRY_MARKER_TYPE = "capitol_trades_scheduled_retry"


def default_retry_marker(output: str | Path) -> Path:
    """Return the retry marker path derived from the staging artifact output."""
    output = Path(output)
    return output.parent / f"{output.name}.retry.json"


def fetch_all_trades(*, generation: str, output: str | Path) -> pd.DataFrame:
    """Fetch through the fail-closed core client and require the artifact write."""
    with CapitolTradesSource(
        data_dir="data", read_only=True, generation=generation
    ) as source:
        df = source.fetch_all_trades()
        source.write_reconciliation_artifact(output)
        return df


def write_retry_marker(
    *,
    generation: str,
    output: str | Path,
    retry_marker: str | Path,
    failure_reason: str,
) -> Path:
    """Record that the staging fetch failed and a scheduled retry is pending."""
    marker = Path(retry_marker)
    payload = {
        "schema_version": RETRY_MARKER_SCHEMA_VERSION,
        "marker_type": RETRY_MARKER_TYPE,
        "reconciliation_only": True,
        "source": "capitol_trades",
        "ingestion_generation": generation,
        "target_output": str(Path(output)),
        "failure_reason": failure_reason,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    marker.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return marker


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Fetch Capitol Trades for reconciliation only (fail closed)"
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="staging artifact path"
    )
    parser.add_argument(
        "--generation", required=True, help="non-empty reconciliation generation"
    )
    parser.add_argument(
        "--retry-marker",
        type=Path,
        default=None,
        help="scheduled retry marker path (default: <output>.retry.json)",
    )
    args = parser.parse_args(argv)
    retry_marker = args.retry_marker or default_retry_marker(args.output)

    print("=== Capitol Trades Reconciliation Fetcher ===")
    print(f"Fetching from {BASE_URL}; canonical database writes are disabled.\n")
    try:
        df = fetch_all_trades(generation=args.generation, output=args.output)
    except CapitolTradesError as exc:
        if not args.output.exists():
            # Fail closed: no artifact exists, so the fetch must be retried later.
            write_retry_marker(
                generation=args.generation,
                output=args.output,
                retry_marker=retry_marker,
                failure_reason=str(exc),
            )
            print(
                f"Scheduled retry marker written: {retry_marker}",
                file=sys.stderr,
            )
        print(
            f"FAILED (no reconciliation artifact written): {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    retry_marker.unlink(missing_ok=True)
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
