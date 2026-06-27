"""Optimize profit: walk-forward sweep entry point.

Walks through every combination of (scoring function, top_n, min_buyers,
allocation) and reports per-combo metrics + best-by-Sharpe / best-by-return
/ best-by-return-to-DD picks.

Run directly:
    python -m optimize_profit.main
"""

import sys

sys.argv = ["ptr-alpha"]  # prevent typer from parsing sweep args

import itertools  # noqa: E402
import time  # noqa: E402
from datetime import date  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402

from analyzer import analysis  # noqa: E402
from analyzer._memo import clear_all_caches  # noqa: E402
from analyzer.database import Database  # noqa: E402
from analyzer.signals import constants as sig_constants  # noqa: E402

from optimize_profit.precompute import precompute_walk_forward_data  # noqa: E402
from optimize_profit.reporting import (  # noqa: E402
    print_baseline,
    print_best_by_ratio,
    print_best_by_return,
    print_best_by_sharpe,
    print_summary_tables,
)
from optimize_profit.scoring import SCORING_FUNCTIONS  # noqa: E402
from optimize_profit.walk_forward import run_walk_forward  # noqa: E402


def main():
    t0 = time.time()

    db = Database(Path("data") / "congress.duckdb", read_only=True)
    signals, all_tx, prices = _load_data(db)

    precomputed, _ = _compute_signals_and_precompute(signals, all_tx, prices)
    if not precomputed:
        print("ERROR: No walk-forward periods with data. Check date ranges.")
        return

    results = _run_sweep(signals, all_tx, prices, precomputed, t0)
    results_df = pd.DataFrame(results)

    print_baseline(results_df)
    print_best_by_sharpe(results_df)
    print_best_by_return(results_df)
    print_best_by_ratio(results_df)
    print_summary_tables(results_df)

    out_path = Path("data/optimize_profit_results.csv")
    results_df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")


def _load_data(db: Database) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load transactions, prices, and entry_prices from the database."""
    tx_start = pd.Timestamp("2021-10-07")
    tx_end = pd.Timestamp("2025-06-30")
    all_tx = db.get_transactions_by_date_range(tx_start, tx_end)

    price_start = tx_start
    price_end = pd.Timestamp("2025-06-30") + pd.Timedelta(days=130)
    all_tickers = sorted(
        set(t for t in all_tx["ticker"].dropna().unique() if isinstance(t, str))
        | {"SPY"}
    )

    prices = db.get_prices(all_tickers, price_start, price_end)
    entry_prices = db.get_entry_prices(all_tickers, price_start, price_end)
    print(f"Data loaded: {len(all_tx)} transactions, {prices.shape[1]} tickers")
    return entry_prices, all_tx, prices


def _compute_signals_and_precompute(signals, all_tx, prices) -> tuple[dict, int]:
    """Compute signal features for the default horizon, then precompute
    per-as_of_date data shared across all sweep combos."""
    horizon = 90
    signals_df = analysis.calculate_signal_potential(signals, prices, [horizon])
    print(f"Signals computed: {len(signals_df)}")

    start_date = date(2022, 1, 1)
    end_date = date(2025, 6, 30)
    as_of_dates = pd.date_range(start_date, end_date, freq="30D")
    print(f"Walk-forward: {len(as_of_dates)} periods from {start_date} to {end_date}")

    min_buyers_list = [1, 2, 3]
    print("Precomputing walk-forward data...")
    precomputed = precompute_walk_forward_data(
        signals_df, all_tx, prices,
        as_of_dates, horizon,
        lookback_days=60,
        training_lookback_days=365,
        min_buyers_list=min_buyers_list,
    )
    print(f"  {len(precomputed)}/{len(as_of_dates)} periods have data")
    return precomputed, horizon


def _run_sweep(signals, all_tx, prices, precomputed, t0: float) -> list:
    """Iterate the full parameter grid and return a list of result dicts."""
    param_grid = {
        "scoring_fn": list(SCORING_FUNCTIONS.keys()),
        "top_n": [3, 5],
        "min_buyers": [1, 2, 3],
        "allocation": ["equal", "signal"],
        "decay_lambda": [0.003, 0.005],
    }
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))
    total = len(combinations)
    print(f"Grid: {total} combinations ({len(SCORING_FUNCTIONS)} scorings x "
          f"{len(param_grid['top_n'])} top_n x {len(param_grid['min_buyers'])} min_buyers x "
          f"{len(param_grid['allocation'])} allocations x "
          f"{len(param_grid['decay_lambda'])} decay)")

    results = []
    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))

        # Save originals, set current combo values
        orig_decay = sig_constants.DECAY_LAMBDA
        sig_constants.DECAY_LAMBDA = params["decay_lambda"]

        clear_all_caches()

        try:
            metrics = run_walk_forward(
                signals, all_tx, prices,
                precomputed,
                scoring_fn=SCORING_FUNCTIONS[params["scoring_fn"]],
                top_n=params["top_n"],
                min_buyers=params["min_buyers"],
                allocation=params["allocation"],
                max_dd_pct=50,
            )
            results.append({**params, **metrics})
            _maybe_log_progress(i, total, params, metrics, t0)
        finally:
            # Restore originals
            sig_constants.DECAY_LAMBDA = orig_decay

    elapsed = time.time() - t0
    print(f"\nSweep completed in {elapsed:.1f}s ({total} combos)")
    return results


def _maybe_log_progress(i: int, total: int, params: dict, metrics: dict, t0: float) -> None:
    if not ((i + 1) % 18 == 0 or i == 0 or i == total - 1):
        return
    elapsed = time.time() - t0
    rate = (i + 1) / elapsed
    eta = (total - i - 1) / rate if rate > 0 else 0
    print(
        f"  [{i+1:3d}/{total}] "
        f"{params['scoring_fn']:22s} "
        f"top={params['top_n']} "
        f"mb={params['min_buyers']} "
        f"{params['allocation']:6s} "
        f"decay={params.get('decay_lambda', 0):.3f} "
        f"→ ret={metrics['total_return_pct']:+7.1f}% "
        f"sharpe={metrics['sharpe']:+5.2f} "
        f"DD={metrics['max_drawdown_pct']:6.1f}% "
        f"wr={metrics['win_rate_pct']:4.0f}% "
        f"({rate:.1f}/s ETA {eta:.0f}s)"
    )


if __name__ == "__main__":
    main()
