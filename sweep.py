"""Parameter sweep for backtest optimization.

Phase 1: coarse grid. Precompute signals per (horizon, decay_lambda) pair,
then iterate backtest params. ~7 min for 216 combos.
"""

from __future__ import annotations

import itertools
import os
import sys
import time
from dataclasses import asdict
from datetime import date
from pathlib import Path

import pandas as pd

sys.argv = ["ptr-alpha"]  # prevent typer from parsing sweep args

from analyzer.database import Database
from analyzer import analysis
from analyzer.pipeline import BacktestParams
from analyzer import signals as signals_mod
from analyzer.validation import SweepResult, run_single_backtest  # noqa: F401  (re-exported)


def main():
    db = Database(Path("data") / "congress.duckdb", read_only=True)

    # Load data once from DB (no yfinance — avoids rate limits)
    tx_start = pd.Timestamp("2021-10-07")
    tx_end = pd.Timestamp("2025-06-30")
    all_transactions = db.get_transactions_by_date_range(tx_start, tx_end)

    price_start = tx_start
    price_end = pd.Timestamp("2025-06-30") + pd.Timedelta(days=130)
    all_tickers = sorted(set(t for t in all_transactions["ticker"].dropna().unique() if isinstance(t, str)) | {"SPY"})

    prices = db.get_prices(all_tickers, price_start, price_end)
    entry_prices = db.get_entry_prices(all_tickers, price_start, price_end)

    print(f"Data loaded: {len(all_transactions)} transactions, {prices.shape[1]} tickers")

    # Parameter grid — phase 1: coarse sweep
    param_grid = {
        "horizon": [60, 90, 120],
        "frequency_days": [30, 90],
        "training_lookback_days": [180, 365],
        "min_buyers": [2, 3, 5],
        "top_n": [3, 5],
        "decay_lambda": [0.001, 0.005, 0.02],
        "bayes_prior_strength": [5, 20, 50],
        "scoring_mode": ["shrunk_alpha", "consistency"],
    }

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))
    total = len(combinations)
    print(f"Total parameter combinations: {total}")

    # Precompute signals per (horizon, decay_lambda) pair
    unique_horizons = set(param_grid["horizon"])
    unique_decays = set(param_grid["decay_lambda"])
    signal_cache: dict[tuple[int, float], pd.DataFrame] = {}

    print(f"Precomputing signals for {len(unique_horizons) * len(unique_decays)} (horizon, decay) pairs...")
    t0 = time.time()
    for h in unique_horizons:
        for d in unique_decays:
            # Pass decay_lambda explicitly — the module global is only read by
            # calculate_signal_potential when no arg is given, but being
            # explicit avoids any ambiguity.
            sigs = analysis.calculate_signal_potential(
                entry_prices, prices, [h], decay_lambda=d,
            )
            signal_cache[(h, d)] = sigs
    print(f"Signal precomputation done in {time.time() - t0:.1f}s")

    # Decide serial vs parallel. Parallel uses fork (Linux/macOS) so child
    # processes inherit the precomputed signal_cache + prices + transactions
    # without re-loading from disk. Each child runs a disjoint subset of
    # combos with its own lru_cache. Override via SWEEP_WORKERS env var.
    workers = int(os.environ.get("SWEEP_WORKERS", "1"))
    if workers < 1:
        workers = 1

    results: list[SweepResult] = []
    start_time = time.time()

    if workers == 1:
        results = _run_serial(
            combinations, keys, signal_cache, all_transactions, prices,
        )
    else:
        results = _run_parallel(
            combinations, keys, signal_cache, all_transactions, prices, workers,
        )

    elapsed = time.time() - start_time
    print(f"\nSweep completed in {elapsed:.1f}s ({total} combos, workers={workers})")

    # Save results
    results_df = pd.DataFrame([asdict(r) for r in results])
    out_path = Path("data/sweep_results.csv")
    results_df.to_csv(out_path, index=False)
    print(f"Results saved to {out_path}")

    # Top 10 by alpha_slope
    print("\n=== Top 10 by alpha_slope (rank5 - rank1) ===")
    cols = [
        "horizon", "frequency_days", "training_lookback_days",
        "min_buyers", "top_n", "decay_lambda", "bayes_prior_strength",
        "overall_alpha", "alpha_slope", "win_rate", "sharpe", "total_recs",
    ]
    top_slope = results_df.nlargest(10, "alpha_slope")
    print(top_slope[cols].to_string(index=False))

    # Top 10 by sharpe
    print("\n=== Top 10 by Sharpe ratio ===")
    top_sharpe = results_df.nlargest(10, "sharpe")
    print(top_sharpe[cols].to_string(index=False))

    # Top 10 by overall_alpha
    print("\n=== Top 10 by overall alpha ===")
    top_alpha = results_df.nlargest(10, "overall_alpha")
    print(top_alpha[cols].to_string(index=False))

    # Bottom 5
    print("\n=== Bottom 5 (worst alpha_slope) ===")
    bottom = results_df.nsmallest(5, "alpha_slope")
    print(bottom[cols].to_string(index=False))


# ---------------------------------------------------------------------------
# Serial / parallel sweep drivers
# ---------------------------------------------------------------------------

# Module-level globals populated by main() before forking workers. Children
# inherit these via fork on Linux/macOS.
_SWEEP_CTX: dict = {}


def _run_serial(
    combinations, keys, signal_cache, all_transactions, prices,
) -> list[SweepResult]:
    total = len(combinations)
    results: list[SweepResult] = []
    start_time = time.time()
    for i, combo in enumerate(combinations):
        params_dict = dict(zip(keys, combo))
        result = _eval_combo(
            params_dict, keys, signal_cache, all_transactions, prices,
        )
        results.append(result)
        if (i + 1) % 50 == 0 or i == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (total - i - 1) / rate if rate > 0 else 0
            print(
                f"  [{i+1}/{total}] "
                f"alpha={result.overall_alpha:+.1f}% "
                f"slope={result.alpha_slope:+.1f}% "
                f"win={result.win_rate:.0f}% "
                f"sharpe={result.sharpe:+.2f} "
                f"({rate:.1f}/s, ETA {eta:.0f}s)"
            )
    return results


def _eval_combo(params_dict, keys, signal_cache, all_transactions, prices):
    """Evaluate one parameter combination and return its SweepResult."""
    params = BacktestParams(
        start_date=date(2022, 1, 1),
        end_date=date(2025, 6, 30),
        horizon=params_dict["horizon"],
        lookback_days=60,
        training_lookback_days=params_dict["training_lookback_days"],
        min_buyers=params_dict["min_buyers"],
        top_n=params_dict["top_n"],
        threshold=5.0,
        frequency_days=params_dict["frequency_days"],
    )
    sigs = signal_cache[(params_dict["horizon"], params_dict["decay_lambda"])]
    return run_single_backtest(
        all_transactions, prices, params, sigs,
        bayes_prior_strength=params_dict["bayes_prior_strength"],
        decay_lambda=params_dict["decay_lambda"],
        scoring_mode=params_dict.get("scoring_mode", "shrunk_alpha"),
    )


def _worker_run(combo_group):
    """Worker entry point: evaluate a group of parameter combinations.

    Inherits _SWEEP_CTX from parent via fork. Uses functools.lru_cache
    (via @df_memoize) for per-worker memoization (no cross-process locking).
    """
    keys = _SWEEP_CTX["keys"]
    signal_cache = _SWEEP_CTX["signal_cache"]
    all_transactions = _SWEEP_CTX["all_transactions"]
    prices = _SWEEP_CTX["prices"]

    out: list[SweepResult] = []
    t0 = time.time()
    for params_dict in combo_group:
        out.append(
            _eval_combo(
                params_dict, keys, signal_cache, all_transactions, prices,
            )
        )
    pid = os.getpid()
    print(
        f"  worker {pid}: {len(combo_group)} combos in {time.time() - t0:.1f}s",
        file=sys.stderr,
    )
    return out


def _run_parallel(
    combinations, keys, signal_cache, all_transactions, prices, workers,
) -> list[SweepResult]:
    """Partition combos across worker processes.

    Groups combinations by (horizon, decay_lambda) so each worker handles a
    coherent subset that maximizes per-worker cache reuse, then dispatches
    groups to a fork-based process pool.
    """
    import multiprocessing as mp

    # Sort combinations into (horizon, decay_lambda) buckets so workers
    # inherit maximal cache reuse.
    keys_index = {k: i for i, k in enumerate(keys)}
    h_idx = keys_index["horizon"]
    d_idx = keys_index["decay_lambda"]
    buckets: dict[tuple, list[dict]] = {}
    for combo in combinations:
        # Extract bucket key directly from combo via precomputed indices
        # (avoids re-reading from the dict, which would require explicit
        # construction of the dict to satisfy static key analysis).
        bkey = (combo[h_idx], combo[d_idx])
        pd_dict = dict(zip(keys, combo))
        buckets.setdefault(bkey, []).append(pd_dict)

    bucket_list = list(buckets.values())

    # Populate the inherited context for forked workers.
    _SWEEP_CTX.clear()
    _SWEEP_CTX["keys"] = keys
    _SWEEP_CTX["signal_cache"] = signal_cache
    _SWEEP_CTX["all_transactions"] = all_transactions
    _SWEEP_CTX["prices"] = prices

    # Fork is required to inherit signal_cache + prices without re-loading.
    # On macOS Python 3.8+ this is no longer the default; we opt in explicitly.
    ctx = mp.get_context("fork")
    # Cap effective worker count at the number of (horizon, decay) buckets:
    # subdividing buckets trades cache reuse for parallelism and empirically
    # never wins for this dataset because per-bucket cold work dominates.
    effective_workers = min(workers, len(bucket_list))
    print(
        f"Running parallel sweep: {effective_workers} workers "
        f"(requested {workers}, capped at {len(bucket_list)} buckets), "
        f"{len(combinations)} total combos",
        file=sys.stderr,
    )
    with ctx.Pool(processes=effective_workers) as pool:
        # One work chunk per (horizon, decay) bucket. Each chunk has maximal
        # cache reuse since all of its combos share the same signal_cache df.
        grouped = [list(b) for b in bucket_list]
        results_nested = pool.map(_worker_run, grouped)

    # Flatten in bucket order (deterministic; bucket_list order is stable
    # because we built it from a dict whose insertion order matches the
    # itertools.product order for the param grid).
    flat: list[SweepResult] = []
    for r in results_nested:
        flat.extend(r)
    return flat


if __name__ == "__main__":
    main()
