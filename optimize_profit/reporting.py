"""Concise reporting for selection, untouched holdout, and canaries."""

from __future__ import annotations

import pandas as pd


def print_selection(selected: pd.Series, n_trials: int) -> None:
    print("\n=== SELECTION WINDOW ONLY ===")
    print(f"Trials: {n_trials}")
    print(
        f"Frozen config: {selected['scoring_fn']}, top={selected['top_n']}, "
        f"min_buyers={selected['min_buyers']}, allocation={selected['allocation']}, "
        f"decay={selected['decay_lambda']}"
    )
    print(
        f"Selection return={selected['total_return_pct']:+.2f}% "
        f"SPY={selected['spy_total_return_pct']:+.2f}% "
        f"alpha Sharpe={selected['alpha_sharpe']:+.2f} "
        f"BH q={selected['bh_q_value']:.4f}"
    )


def print_holdout(metrics: dict, spy_metrics: dict, constant_metrics: dict) -> None:
    print("\n=== UNTOUCHED HOLDOUT (FROZEN CONFIG, ONE EVALUATION) ===")
    print(
        f"Strategy return={metrics['total_return_pct']:+.2f}% "
        f"SPY={spy_metrics['total_return_pct']:+.2f}% "
        f"mean alpha={metrics['mean_alpha_pct']:+.3f}% "
        f"alpha Sharpe={metrics['alpha_sharpe']:+.2f} "
        f"periods={metrics['n_periods']} coverage={metrics['coverage_pct']:.1f}%"
    )
    print(
        f"Constant-score canary return={constant_metrics['total_return_pct']:+.2f}% "
        f"alpha Sharpe={constant_metrics['alpha_sharpe']:+.2f}"
    )


def print_verdict(robust: bool, reasons: list[str], artifact_dir) -> None:
    if robust:
        print(
            "\nVERDICT: HOLDOUT ROBUSTNESS PASSED. Evidence supports further paper trading; "
            "it is not a guaranteed-profit claim."
        )
    else:
        print("\nVERDICT: NO VALIDATED PROFIT CLAIM.")
        for reason in reasons:
            print(f"  - {reason}")
    print(f"Artifacts: {artifact_dir}")


# Legacy helpers intentionally avoid “best” claims. They remain import-compatible.
def print_baseline(results_df: pd.DataFrame) -> None:
    print(f"Selection trials recorded: {len(results_df)}")


def print_best_by_sharpe(results_df: pd.DataFrame) -> None:
    print("Best-by-Sharpe reporting removed: use chronological selection + holdout.")


def print_best_by_return(results_df: pd.DataFrame) -> None:
    print("Best-by-return reporting removed: use chronological selection + holdout.")


def print_best_by_ratio(results_df: pd.DataFrame) -> None:
    print("Best-by-ratio reporting removed: use chronological selection + holdout.")


def print_summary_tables(results_df: pd.DataFrame) -> None:
    columns = [
        column
        for column in (
            "trial_id",
            "scoring_fn",
            "top_n",
            "min_buyers",
            "allocation",
            "decay_lambda",
            "alpha_sharpe",
            "mean_alpha_pct",
            "bh_q_value",
        )
        if column in results_df
    ]
    print(
        results_df.sort_values("alpha_sharpe", ascending=False)[columns].to_string(
            index=False
        )
    )
