"""Point-in-time member profitability research command.

Run with an explicit input database and output path::

    python -m member_profitability.main --db /path/congress.duckdb --output /tmp/member.json
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from member_profitability.analysis import (
    combined_metrics_analysis,
    spearman_correlations_per_metric,
    tier_analysis,
    trade_count_reliability,
)
from member_profitability.config import DATA_SCOPE, TX_END, TX_START
from member_profitability.data import (
    compute_signals,
    load_transactions_and_prices,
    print_loaded_data,
)
from member_profitability.position_sizing import position_sizing_grid_search
from member_profitability.reporting import best_predictors, build_output_dict, write_output
from member_profitability.walk_forward import collect_window_results, generate_windows


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    started = time.time()
    all_tx, prices, entry_prices, all_tickers = load_transactions_and_prices(
        args.db, args.tx_start, args.tx_end
    )
    print_loaded_data(started, all_tx, all_tickers)
    signals = compute_signals(entry_prices, prices)
    windows = generate_windows(signals)
    if len(windows) < 2:
        raise RuntimeError(
            "At least two non-overlapping windows are required for retrospective validation"
        )

    # The last window is selection-isolated in this run, but the history was
    # previously explored and therefore supports retrospective validation only.
    research_windows = windows[:-1]
    all_wf = collect_window_results(signals, research_windows)
    if all_wf.empty:
        raise RuntimeError("No valid point-in-time research windows")

    correlations = spearman_correlations_per_metric(all_wf)
    tiers = tier_analysis(all_wf)
    trade_counts = trade_count_reliability(all_wf)
    combined = combined_metrics_analysis(all_wf)
    position_research = position_sizing_grid_search(signals, windows)
    output = build_output_dict(
        signals,
        all_tx,
        all_tickers,
        windows,
        all_wf,
        correlations,
        tiers,
        trade_counts,
        position_research,
        combined,
        args.db,
    )
    output["recommendations"] = best_predictors(
        correlations, combined, tiers, position_research
    )
    write_output(output, args.output)

    print(f"Data scope: {DATA_SCOPE}")
    print(
        "Retrospective validation status: "
        f"{position_research['retrospective_validation_status']}"
    )
    print("Profitability claim: not established")
    print(f"Elapsed: {time.time() - started:.1f}s")
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, type=Path, help="Explicit input DuckDB path")
    parser.add_argument("--output", required=True, type=Path, help="Output JSON path")
    parser.add_argument("--tx-start", default=TX_START, type=_iso_date)
    parser.add_argument("--tx-end", default=TX_END, type=_iso_date)
    return parser.parse_args(argv)


def _iso_date(value: str) -> str:
    try:
        return str(pd.Timestamp(value).date())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("date must be parseable as YYYY-MM-DD") from exc


if __name__ == "__main__":
    raise SystemExit(main())
