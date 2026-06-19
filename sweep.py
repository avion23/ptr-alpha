"""Parameter sweep for backtest optimization.

Phase 1: coarse grid. Precompute signals per (horizon, decay_lambda) pair,
then iterate backtest params. ~7 min for 216 combos.
"""

from __future__ import annotations

import itertools
import sys
import time
from dataclasses import dataclass, asdict
from datetime import date
from pathlib import Path

import pandas as pd

sys.argv = ["ptr-alpha"]  # prevent typer from parsing sweep args

from analyzer.database import Database
from analyzer import analysis
from analyzer.pipeline import BacktestParams
from analyzer import signals as signals_mod
from analyzer import member_ranking as mr_mod
from analyzer import member_skill as ms_mod


@dataclass
class SweepResult:
    horizon: int
    frequency_days: int
    training_lookback_days: int
    min_buyers: int
    top_n: int
    decay_lambda: float
    bayes_prior_strength: float
    total_recs: int = 0
    dates_evaluated: int = 0
    overall_alpha: float = 0.0
    overall_return: float = 0.0
    rank1_alpha: float = 0.0
    rank5_alpha: float = 0.0
    alpha_slope: float = 0.0
    win_rate: float = 0.0
    sharpe: float = 0.0
    max_drawdown: float = 0.0


def run_single_backtest(
    all_transactions: pd.DataFrame,
    prices: pd.DataFrame,
    params: BacktestParams,
    signals: pd.DataFrame,
    bayes_prior_strength: float,
    decay_lambda: float,
    cache=None,
) -> SweepResult:
    """Run one backtest with given params and return metrics."""
    old_bayes = signals_mod.BAYES_PRIOR_STRENGTH
    old_decay = signals_mod.DECAY_LAMBDA
    signals_mod.BAYES_PRIOR_STRENGTH = bayes_prior_strength
    signals_mod.DECAY_LAMBDA = decay_lambda
    mr_mod.BAYES_PRIOR_STRENGTH = bayes_prior_strength
    ms_mod.BAYES_PRIOR_STRENGTH = bayes_prior_strength

    try:
        as_of_dates = pd.date_range(
            params.start_date, params.end_date, freq=f"{params.frequency_days}D"
        )

        all_results = []
        for as_of in as_of_dates:
            as_of_ts = pd.Timestamp(as_of)
            recs = analysis.backtest_recommendations(
                signals, all_transactions, as_of_ts,
                horizon=params.horizon,
                lookback_days=params.lookback_days,
                min_buyers=params.min_buyers,
                top_n=params.top_n,
                threshold=params.threshold,
                prices_df=prices,
                training_lookback_days=params.training_lookback_days,
                cache=cache,
            )
            if recs.empty:
                continue
            try:
                evaluated = analysis.evaluate_backtest(
                    recs, prices, as_of_ts, params.horizon
                )
                evaluated = evaluated.dropna(subset=["bt_return_pct"])
                evaluated.insert(0, "as_of_date", as_of_ts.date())
                all_results.append(evaluated)
            except Exception:
                continue

        if not all_results:
            return SweepResult(
                horizon=params.horizon,
                frequency_days=params.frequency_days,
                training_lookback_days=params.training_lookback_days,
                min_buyers=params.min_buyers,
                top_n=params.top_n,
                decay_lambda=decay_lambda,
                bayes_prior_strength=bayes_prior_strength,
            )

        combined = pd.concat(all_results, ignore_index=True)
        valid = combined.dropna(subset=["bt_alpha_pct"])

        result = SweepResult(
            horizon=params.horizon,
            frequency_days=params.frequency_days,
            training_lookback_days=params.training_lookback_days,
            min_buyers=params.min_buyers,
            top_n=params.top_n,
            decay_lambda=decay_lambda,
            bayes_prior_strength=bayes_prior_strength,
            total_recs=len(combined),
            dates_evaluated=int(valid["as_of_date"].nunique()),
            overall_alpha=round(float(valid["bt_alpha_pct"].mean()), 2),
            overall_return=round(float(valid["bt_return_pct"].mean()), 2),
        )

        rank_alpha = valid.groupby("rank")["bt_alpha_pct"].mean()
        if 1 in rank_alpha.index:
            result.rank1_alpha = round(float(rank_alpha.loc[1]), 2)
        if 5 in rank_alpha.index:
            result.rank5_alpha = round(float(rank_alpha.loc[5]), 2)
        result.alpha_slope = round(result.rank5_alpha - result.rank1_alpha, 2)

        result.win_rate = round(float((valid["bt_alpha_pct"] > 0).mean()) * 100, 1)

        if len(valid) > 1:
            mean_alpha = valid["bt_alpha_pct"].mean()
            std_alpha = valid["bt_alpha_pct"].std()
            if std_alpha > 0:
                periods_per_year = 365 / params.frequency_days
                result.sharpe = round(
                    float(mean_alpha / std_alpha * (periods_per_year ** 0.5)), 2
                )

        cumulative = (1 + valid["bt_alpha_pct"] / 100).cumprod()
        rolling_max = cumulative.cummax()
        drawdown = (cumulative - rolling_max) / rolling_max
        result.max_drawdown = round(float(drawdown.min()) * 100, 2)

        return result
    finally:
        signals_mod.BAYES_PRIOR_STRENGTH = old_bayes
        signals_mod.DECAY_LAMBDA = old_decay
        mr_mod.BAYES_PRIOR_STRENGTH = old_bayes
        ms_mod.BAYES_PRIOR_STRENGTH = old_bayes


def main():
    db = Database(Path("data") / "congress.duckdb", read_only=False)

    # Load data once from DB (no yfinance — avoids rate limits)
    tx_start = pd.Timestamp("2021-10-07")
    tx_end = pd.Timestamp("2025-06-30")
    all_transactions = db.get_transactions_by_date_range(tx_start, tx_end)

    price_start = tx_start
    price_end = pd.Timestamp("2025-06-30") + pd.Timedelta(days=130)
    all_tickers = sorted(set(all_transactions["ticker"].unique().tolist()) | {"SPY"})

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

    # Long-lived memoization cache shared across all 648 combos. Subcomputations
    # whose results depend only on a subset of the sweep params (entry value,
    # signal features, rank_members, ticker-member perf) are reused across
    # combos that share those params. This is the single biggest speedup:
    # most leaf work depends only on (ticker, as_of_date, horizon, decay).
    from analyzer.sweep_cache import BacktestCache
    cache = BacktestCache()

    results = []
    start_time = time.time()

    for i, combo in enumerate(combinations):
        params_dict = dict(zip(keys, combo))
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

        result = run_single_backtest(
            all_transactions, prices, params, sigs,
            bayes_prior_strength=params_dict["bayes_prior_strength"],
            decay_lambda=params_dict["decay_lambda"],
            cache=cache,
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

    elapsed = time.time() - start_time
    print(f"\nSweep completed in {elapsed:.1f}s ({total} combos)")
    print(f"Cache stats: {cache.stats}")

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


if __name__ == "__main__":
    main()
