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
    print(
        f"Locked final test starts {final_start}; its rows were not analytically queried."
    )
    print("The database file was read only to compute its whole-file SHA-256.")
    print(f"Artifacts: {artifact_dir}")
