"""Metrics for non-overlapping scheduled portfolio observations."""

from __future__ import annotations

import numpy as np
import pandas as pd


def summarize_walk_forward(
    all_returns: list[dict], *, periods_per_year: float = 365.0 / 90.0
) -> dict:
    """Aggregate one return for every scheduled non-overlapping observation.

    ``terminal_observation_drawdown_pct`` uses only period-end observations. It
    is not an intraperiod or trading maximum drawdown and cannot drive a stop.
    """
    if not all_returns:
        return _empty_metrics()

    rets_df = pd.DataFrame(all_returns)
    period_rets = rets_df["portfolio_return_pct"].to_numpy(dtype=float) / 100
    spy_rets = rets_df["spy_return_pct"].to_numpy(dtype=float) / 100
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
        "terminal_observation_drawdown_pct": round(
            terminal_observation_drawdown(cumulative), 1
        ),
        "win_rate_pct": round(float((period_rets > 0).mean() * 100), 1),
        "n_periods": len(rets_df),
        "n_cash_periods": int((rets_df["n_positions"] == 0).sum()),
        "avg_positions": avg_positions_from(rets_df),
    }


def annualized_sharpe(period_rets: np.ndarray, periods_per_year: float) -> float:
    if len(period_rets) > 1 and np.std(period_rets, ddof=1) > 0:
        return float(
            np.mean(period_rets)
            / np.std(period_rets, ddof=1)
            * np.sqrt(periods_per_year)
        )
    return 0.0


def terminal_observation_drawdown(cumulative: np.ndarray) -> float:
    rolling_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - rolling_max) / rolling_max
    return float(drawdown.min() * 100)


def avg_positions_from(rets_df: pd.DataFrame) -> float:
    return round(float(rets_df["n_positions"].mean()), 1)


def _empty_metrics() -> dict:
    return {
        "total_return_pct": 0.0,
        "spy_total_return_pct": 0.0,
        "excess_total_return_pct": 0.0,
        "mean_alpha_pct": 0.0,
        "sharpe": 0.0,
        "alpha_sharpe": 0.0,
        "terminal_observation_drawdown_pct": 0.0,
        "win_rate_pct": 0.0,
        "n_periods": 0,
        "n_cash_periods": 0,
        "avg_positions": 0.0,
    }
