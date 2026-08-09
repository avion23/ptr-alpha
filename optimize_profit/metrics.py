"""Metrics for non-overlapping walk-forward portfolio periods."""

from __future__ import annotations

import numpy as np
import pandas as pd


def summarize_walk_forward(
    all_returns: list[dict],
    stopped: bool,
    *,
    periods_per_year: float = 365.0 / 90.0,
) -> dict:
    """Aggregate independent, non-overlapping portfolio periods.

    The equity curve is anchored at initial wealth 1.0 so a loss in the first
    period contributes to maximum drawdown.
    """
    if not all_returns:
        return _empty_metrics(stopped)

    rets_df = pd.DataFrame(all_returns)
    period_rets = rets_df["portfolio_return_pct"].to_numpy(dtype=float) / 100
    spy_rets = (
        rets_df["spy_return_pct"].to_numpy(dtype=float) / 100
        if "spy_return_pct" in rets_df
        else np.zeros(len(rets_df), dtype=float)
    )
    alpha_rets = period_rets - spy_rets

    cumulative = np.concatenate(([1.0], np.cumprod(1 + period_rets)))
    spy_cumulative = np.concatenate(([1.0], np.cumprod(1 + spy_rets)))
    total_return = float(cumulative[-1] - 1) * 100
    spy_total_return = float(spy_cumulative[-1] - 1) * 100

    return {
        "total_return_pct": round(total_return, 2),
        "spy_total_return_pct": round(spy_total_return, 2),
        "excess_total_return_pct": round(total_return - spy_total_return, 2),
        "mean_alpha_pct": round(float(alpha_rets.mean() * 100), 3),
        "sharpe": round(annualized_sharpe(period_rets, periods_per_year), 2),
        "alpha_sharpe": round(annualized_sharpe(alpha_rets, periods_per_year), 2),
        "max_drawdown_pct": round(max_drawdown(cumulative), 1),
        "win_rate_pct": round(float((period_rets > 0).mean() * 100), 1),
        "n_periods": len(rets_df),
        "avg_positions": avg_positions_from(rets_df),
        "stopped_early": stopped,
    }


def annualized_sharpe(period_rets: np.ndarray, periods_per_year: float) -> float:
    if len(period_rets) > 1 and np.std(period_rets, ddof=1) > 0:
        return float(
            np.mean(period_rets)
            / np.std(period_rets, ddof=1)
            * np.sqrt(periods_per_year)
        )
    return 0.0


def max_drawdown(cumulative: np.ndarray) -> float:
    rolling_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - rolling_max) / rolling_max
    return float(drawdown.min() * 100)


def avg_positions_from(rets_df: pd.DataFrame) -> float:
    col = "n_positions" if "n_positions" in rets_df.columns else "n_periods"
    return round(float(rets_df[col].mean()), 1)


def _empty_metrics(stopped: bool) -> dict:
    return {
        "total_return_pct": 0.0,
        "spy_total_return_pct": 0.0,
        "excess_total_return_pct": 0.0,
        "mean_alpha_pct": 0.0,
        "sharpe": 0.0,
        "alpha_sharpe": 0.0,
        "max_drawdown_pct": 0.0,
        "win_rate_pct": 0.0,
        "n_periods": 0,
        "avg_positions": 0.0,
        "stopped_early": stopped,
    }
