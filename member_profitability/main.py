"""Walk-forward analysis of congress member profitability.

Determines which metrics actually predict future alpha and optimizes
position sizing parameters. Run via: ``python -m member_profitability.main``.

Package layout:
  - config.py         constants
  - data.py           DB loading + signal computation
  - walk_forward.py   rolling train/test windows + per-window metrics
  - analysis.py       Spearman correlations, tiers, trade-count thresholds
  - position_sizing.py top_n × min_buyers grid search
  - reporting.py      output JSON + recommendations
  - main.py           entry point
"""

import sys

sys.argv = ["ptr-alpha"]

import time  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402

from member_profitability.analysis import (  # noqa: E402
    combined_metrics_analysis,
    spearman_correlations_per_metric,
    summarize_combined_metrics,
    tier_analysis,
    trade_count_reliability,
)
from member_profitability.data import (  # noqa: E402
    compute_signals,
    load_transactions_and_prices,
    print_loaded_data,
)
from member_profitability.position_sizing import position_sizing_grid_search  # noqa: E402
from member_profitability.reporting import (  # noqa: E402
    best_predictors,
    build_output_dict,
    write_output,
)
from member_profitability.walk_forward import collect_window_results, generate_windows  # noqa: E402


def main():
    t0 = time.time()

    print("Loading data...")
    all_tx, prices, entry_prices, all_tickers = load_transactions_and_prices()
    print_loaded_data(t0, all_tx, all_tickers)

    print("Computing signals...")
    t1 = time.time()
    sigs = compute_signals(entry_prices, prices)
    print(f"  Signals computed in {time.time()-t1:.1f}s")
    print(f"  Total signals: {len(sigs)}")

    print("\nRunning walk-forward analysis...")
    t2 = time.time()
    windows = generate_windows(sigs)
    print(f"  Date range: {pd.Timestamp(sigs['disclosure_date'].min()).date()} to "
          f"{pd.Timestamp(sigs['disclosure_date'].max()).date()}")
    print(f"  Windows to evaluate: {len(windows)}")

    all_wf = collect_window_results(sigs, windows)
    if all_wf.empty:
        print("ERROR: No valid windows found!")
        sys.exit(1)
    wf_time = time.time() - t2
    print(f"  Walk-forward completed in {wf_time:.1f}s")
    print(f"\nTotal window-period observations: {len(all_wf)}")

    correlations = spearman_correlations_per_metric(all_wf)
    tier_results = tier_analysis(all_wf)
    trade_count_analysis = trade_count_reliability(all_wf)
    combined_results = combined_metrics_analysis(all_wf)
    position_results = position_sizing_grid_search(sigs, windows)

    _print_correlations(correlations)
    _print_tier_analysis(tier_results)
    _print_trade_count(trade_count_analysis)
    _print_position_grid(position_results)
    _print_combined_metrics(combined_results)

    valid_windows = int(all_wf["window"].nunique())
    output = build_output_dict(
        sigs, all_tx, all_tickers, windows, valid_windows, all_wf,
        correlations, tier_results, trade_count_analysis,
        position_results, combined_results,
    )
    output["recommendations"] = best_predictors(
        correlations, combined_results, tier_results, position_results,
    )
    write_output(output, Path("data/member_analysis.json"))

    _print_executive_summary(correlations, output["recommendations"]["key_findings"], t0)


def _print_correlations(correlations: dict) -> None:
    print("\n" + "=" * 60)
    print("1. SPEARMAN CORRELATIONS: Training Metric Rank vs Test Alpha")
    print("=" * 60)
    for metric, data in correlations.items():
        if data["n_windows"] > 0:
            print(f"  {metric:25s}: mean={data['mean_spearman']:+.4f}, "
                  f"median={data['median_spearman']:+.4f}, "
                  f"sig={int(data['pct_significant'] * data['n_windows'] / 100)}/{data['n_windows']} "
                  f"({data['pct_significant']:.0f}%)")
        else:
            print(f"  {metric:25s}: insufficient data")


def _print_tier_analysis(tier_results: dict) -> None:
    print("\n" + "=" * 60)
    print("2. TIER ANALYSIS: Top 10% vs Bottom 10% by Training Metric")
    print("=" * 60)
    for metric, data in tier_results.items():
        if data["n_observations"] > 0:
            direction = "OUTPERFORMS" if data["top_outperforms"] else "UNDERPERFORMS"
            print(f"  {metric:25s}: top10={data['top_10pct_mean_alpha']:+.4f}% "
                  f"bot10={data['bottom_10pct_mean_alpha']:+.4f}% "
                  f"lift={data['alpha_lift']:+.4f}% ({direction}, p={data['lift_p_value']:.4f})")
        else:
            print(f"  {metric:25s}: insufficient data")


def _print_trade_count(trade_count_analysis: dict) -> None:
    print("\n" + "=" * 60)
    print("3. MIN TRADE COUNT FOR RELIABILITY")
    print("=" * 60)
    for min_trades, data in trade_count_analysis.items():
        if "shrunk_alpha_corr" in data:
            print(f"  min_trades={min_trades:2d}: n={data['n_members']:5d}, "
                  f"mean_alpha={data['mean_test_alpha']:+.4f}%, "
                  f"corr(shrunk_alpha, test)={data['shrunk_alpha_corr']:+.4f} "
                  f"(p={data['corr_p_value']:.4f})")
        else:
            print(f"  min_trades={min_trades:2d}: n={data['n_members']:5d}, "
                  f"mean_alpha={data['mean_test_alpha']:+.4f}%")


def _print_position_grid(position_results: list[dict]) -> None:
    print("\n" + "=" * 60)
    print("4. OPTIMAL POSITION SIZING (top_n × min_buyers grid)")
    print("=" * 60)
    print(f"\n  {'top_n':>5s} {'min_b':>5s} {'picks':>6s} {'avg_α%':>8s} {'sharpe':>8s} {'win%':>6s}")
    print("  " + "-" * 45)
    for r in position_results:
        print(f"  {r['top_n']:5d} {r['min_buyers']:5d} {r['total_picks']:6d} "
              f"{r['avg_spy_alpha_pct']:+8.4f} {r['sharpe_proxy']:8.4f} {r['win_rate_pct']:6.1f}")


def _print_combined_metrics(combined_results: dict) -> None:
    print("\n" + "=" * 60)
    print("5. BEST COMBINED PREDICTOR (multi-metric)")
    print("=" * 60)
    summary = summarize_combined_metrics(combined_results)
    print("  Combined metric correlations with test alpha:")
    for name, data in summary.items():
        print(f"    {name:25s}: mean_spearman={data['mean_spearman']:+.4f}")


def _print_executive_summary(correlations: dict, findings: list[str], t0: float) -> None:
    print("\n" + "=" * 60)
    print("EXECUTIVE SUMMARY")
    print("=" * 60)

    print("\nTop predictors of future member alpha (Spearman correlation):")
    sorted_corrs = sorted(
        correlations.items(),
        key=lambda x: abs(x[1]["mean_spearman"]),
        reverse=True,
    )
    for name, data in sorted_corrs[:5]:
        print(f"  {name:25s}: {data['mean_spearman']:+.4f}")

    print("\nKey findings:")
    for i, finding in enumerate(findings, 1):
        print(f"  {i}. {finding}")

    print(f"\nTotal analysis time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
