"""Reporting for retrospective validation and a locked future test."""

from __future__ import annotations

import pandas as pd


def print_selection(
    selected: pd.Series, n_trials: int, null_empirical_p: float
) -> None:
    print("\n=== RETROSPECTIVE SELECTION WINDOW ===")
    print(f"Trials: {n_trials}")
    print(
        f"Locked config: {selected['scoring_fn']}, top={selected['top_n']}, "
        f"min_buyers={selected['min_buyers']}, allocation={selected['allocation']}, "
        f"decay={selected['decay_lambda']}"
    )
    print(
        f"Selection return={selected['total_return_pct']:+.2f}% "
        f"SPY={selected['spy_total_return_pct']:+.2f}% "
        f"alpha Sharpe={selected['alpha_sharpe']:+.2f} "
        f"Bonferroni={bool(selected['bonferroni_significant'])} "
        f"permutation p={null_empirical_p:.4f}"
    )


def print_retrospective(
    metrics: dict, spy_metrics: dict, constant_metrics: dict
) -> None:
    print("\n=== RETROSPECTIVE VALIDATION (2024-07 THROUGH 2025-06) ===")
    print("This interval is reused historical data and is not the locked final test.")
    print(
        f"Strategy return={metrics['total_return_pct']:+.2f}% "
        f"SPY={spy_metrics['total_return_pct']:+.2f}% "
        f"mean opportunity alpha={metrics['mean_alpha_pct']:+.3f}% "
        f"alpha Sharpe={metrics['alpha_sharpe']:+.2f} "
        f"periods={metrics['n_periods']} cash={metrics['n_cash_periods']}"
    )
    print(
        f"Constant-score canary return={constant_metrics['total_return_pct']:+.2f}% "
        f"alpha Sharpe={constant_metrics['alpha_sharpe']:+.2f}"
    )


def print_verdict(robust: bool, reasons: list[str], artifact_dir, final_start) -> None:
    print("\nVERDICT: NO FINAL OUT-OF-SAMPLE PROFIT CLAIM.")
    if robust:
        print(
            "Retrospective gates passed, but the locked final test remains unexecuted."
        )
    else:
        for reason in reasons:
            print(f"  - {reason}")
    print(f"Locked final test starts {final_start}; it was not read or evaluated.")
    print(f"Artifacts: {artifact_dir}")


# Import-compatible legacy helpers. They never make optimization claims.
def print_baseline(results_df: pd.DataFrame) -> None:
    print(f"Retrospective selection trials recorded: {len(results_df)}")


def print_best_by_sharpe(results_df: pd.DataFrame) -> None:
    print("Best-by-Sharpe claims removed; use locked chronological phases.")


def print_best_by_return(results_df: pd.DataFrame) -> None:
    print("Best-by-return claims removed; use locked chronological phases.")


def print_best_by_ratio(results_df: pd.DataFrame) -> None:
    print("Best-by-ratio claims removed; use locked chronological phases.")


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
            "bonferroni_significant",
        )
        if column in results_df
    ]
    print(
        results_df.sort_values("alpha_sharpe", ascending=False)[columns].to_string(
            index=False
        )
    )
