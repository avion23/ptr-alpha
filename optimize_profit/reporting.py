"""Reporting helpers for the optimize_profit sweep.

Prints baseline, best-by-Sharpe, best-by-return, best-by-return/DD-ratio,
plus a summary table per scoring function. Each helper takes a
`results_df` (the sweep output) and writes to stdout.
"""

import numpy as np
import pandas as pd

from optimize_profit.scoring import SCORING_FUNCTIONS


def print_baseline(results_df: pd.DataFrame) -> None:
    baseline = results_df[
        (results_df["scoring_fn"] == "shrunk_alpha")
        & (results_df["top_n"] == 5)
        & (results_df["min_buyers"] == 2)
        & (results_df["allocation"] == "equal")
    ]
    if baseline.empty:
        return
    b = baseline.iloc[0]
    print(f"\n{'=' * 70}")
    print("BASELINE (shrunk_alpha, top=5, mb=2, equal-weight)")
    print(f"{'=' * 70}")
    print(f"  Return:  {b['total_return_pct']:+.1f}%")
    print(f"  Sharpe:  {b['sharpe']:+.2f}")
    print(f"  DD:      {b['max_drawdown_pct']:.1f}%")
    print(f"  Win%:    {b['win_rate_pct']:.0f}%")


def print_best_by_sharpe(results_df: pd.DataFrame) -> None:
    valid = _valid_results(results_df)
    if valid.empty:
        return
    best = valid.nlargest(1, "sharpe").iloc[0]
    print(f"\n{'=' * 70}")
    print("BEST BY SHARPE (max DD < 30%)")
    print(f"{'=' * 70}")
    print(f"  Scoring:  {best['scoring_fn']}")
    print(f"  Top N:    {best['top_n']}")
    print(f"  Min Buy:  {best['min_buyers']}")
    print(f"  Alloc:    {best['allocation']}")
    print(f"  Return:   {best['total_return_pct']:+.1f}%")
    print(f"  Sharpe:   {best['sharpe']:+.2f}")
    print(f"  DD:       {best['max_drawdown_pct']:.1f}%")
    print(f"  Win%:     {best['win_rate_pct']:.0f}%")

    baseline = _baseline_row(results_df)
    if baseline is not None:
        ret_imp = best["total_return_pct"] - baseline["total_return_pct"]
        sharpe_imp = best["sharpe"] - baseline["sharpe"]
        dd_imp = best["max_drawdown_pct"] - baseline["max_drawdown_pct"]
        print(f"\n  vs baseline: return {ret_imp:+.1f}pp, "
              f"sharpe {sharpe_imp:+.2f}, DD {dd_imp:+.1f}pp")


def print_best_by_return(results_df: pd.DataFrame) -> None:
    valid = _valid_results(results_df)
    if valid.empty:
        return
    best = valid.nlargest(1, "total_return_pct").iloc[0]
    print(f"\n{'=' * 70}")
    print("BEST BY TOTAL RETURN (max DD < 30%)")
    print(f"{'=' * 70}")
    print(f"  Scoring:  {best['scoring_fn']}")
    print(f"  Top N:    {best['top_n']}")
    print(f"  Min Buy:  {best['min_buyers']}")
    print(f"  Alloc:    {best['allocation']}")
    print(f"  Return:   {best['total_return_pct']:+.1f}%")
    print(f"  Sharpe:   {best['sharpe']:+.2f}")
    print(f"  DD:       {best['max_drawdown_pct']:.1f}%")
    print(f"  Win%:     {best['win_rate_pct']:.0f}%")


def print_best_by_ratio(results_df: pd.DataFrame) -> None:
    valid = _valid_results(results_df).copy()
    if valid.empty:
        return
    valid["ret_dd_ratio"] = (
        valid["total_return_pct"]
        / np.abs(valid["max_drawdown_pct"].values).clip(min=1)
    )
    best = valid.nlargest(1, "ret_dd_ratio").iloc[0]
    print(f"\n{'=' * 70}")
    print("BEST RETURN/DRAWDOWN RATIO (max DD < 30%)")
    print(f"{'=' * 70}")
    print(f"  Scoring:  {best['scoring_fn']}")
    print(f"  Top N:    {best['top_n']}")
    print(f"  Min Buy:  {best['min_buyers']}")
    print(f"  Alloc:    {best['allocation']}")
    print(f"  Return:   {best['total_return_pct']:+.1f}%")
    print(f"  Sharpe:   {best['sharpe']:+.2f}")
    print(f"  DD:       {best['max_drawdown_pct']:.1f}%")
    print(f"  Win%:     {best['win_rate_pct']:.0f}%")
    print(f"  Ret/DD:   {best['ret_dd_ratio']:.2f}")


def print_summary_tables(results_df: pd.DataFrame) -> None:
    display_cols = [
        "scoring_fn", "top_n", "min_buyers", "allocation",
        "total_return_pct", "sharpe", "max_drawdown_pct", "win_rate_pct", "n_periods",
    ]
    available = [c for c in display_cols if c in results_df.columns]

    print(f"\n{'=' * 90}")
    print("ALL RESULTS (sorted by Sharpe)")
    print(f"{'=' * 90}")
    print(
        results_df.nlargest(len(results_df), "sharpe")[available].to_string(index=False)
    )

    print(f"\n{'=' * 70}")
    print("SCORING FUNCTION SUMMARY (best config per scoring)")
    print(f"{'=' * 70}")
    summary_rows = []
    for scoring_fn_name in SCORING_FUNCTIONS:
        subset = results_df[results_df["scoring_fn"] == scoring_fn_name]
        if subset.empty:
            continue
        best = subset.nlargest(1, "sharpe").iloc[0]
        summary_rows.append({
            "scoring": scoring_fn_name,
            "best_return": best["total_return_pct"],
            "best_sharpe": best["sharpe"],
            "best_dd": best["max_drawdown_pct"],
            "best_top_n": best["top_n"],
            "best_mb": best["min_buyers"],
            "best_alloc": best["allocation"],
        })
    summary_df = pd.DataFrame(summary_rows).sort_values("best_sharpe", ascending=False)
    print(summary_df.to_string(index=False))


def _valid_results(results_df: pd.DataFrame) -> pd.DataFrame:
    return results_df[results_df["max_drawdown_pct"] > -30].copy()


def _baseline_row(results_df: pd.DataFrame):
    baseline = results_df[
        (results_df["scoring_fn"] == "shrunk_alpha")
        & (results_df["top_n"] == 5)
        & (results_df["min_buyers"] == 2)
        & (results_df["allocation"] == "equal")
    ]
    if baseline.empty:
        return None
    return baseline.iloc[0]
