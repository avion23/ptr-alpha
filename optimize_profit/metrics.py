"""Summary metrics for a walk-forward backtest.

Aggregates per-period portfolio returns into a single metrics dict:
total return, annualized Sharpe, max drawdown, win rate, n_periods,
avg_positions, stopped_early flag.
"""

import numpy as np
import pandas as pd

SQRT_12 = np.sqrt(12)


def summarize_walk_forward(all_returns: list, stopped: bool) -> dict:
    """Aggregate per-period returns into a single metrics dict.

    Returns an "empty" metrics dict (all zeros) when there are no periods.
    """
    if not all_returns:
        return _empty_metrics(stopped=False)

    rets_df = pd.DataFrame(all_returns)
    period_rets = rets_df["portfolio_return_pct"].values / 100
    cumulative = np.cumprod(1 + period_rets)
    total_return = float(cumulative[-1] - 1) * 100

    sharpe = annualized_sharpe(period_rets)
    max_dd = max_drawdown(cumulative)
    win_rate = float((period_rets > 0).mean() * 100)
    avg_positions = avg_positions_from(rets_df)

    return {
        "total_return_pct": round(total_return, 2),
        "sharpe": round(sharpe, 2),
        "max_drawdown_pct": round(max_dd, 1),
        "win_rate_pct": round(win_rate, 1),
        "n_periods": len(rets_df),
        "avg_positions": avg_positions,
        "stopped_early": stopped,
    }


def annualized_sharpe(period_rets: np.ndarray) -> float:
    """Annualized Sharpe assuming monthly rebalance (sqrt(12) factor)."""
    if len(period_rets) > 1 and np.std(period_rets) > 0:
        return float(np.mean(period_rets) / np.std(period_rets) * SQRT_12)
    return 0.0


def max_drawdown(cumulative: np.ndarray) -> float:
    """Max drawdown of an equity curve, returned as a positive percentage."""
    rolling_max = np.maximum.accumulate(cumulative)
    dd = (cumulative - rolling_max) / rolling_max
    return float(dd.min() * 100)


def avg_positions_from(rets_df: pd.DataFrame) -> float:
    """Mean of the n_positions column (fallback to n_periods)."""
    col = "n_positions" if "n_positions" in rets_df.columns else "n_periods"
    return round(float(rets_df[col].mean()), 1)


def _empty_metrics(stopped: bool) -> dict:
    return {
        "total_return_pct": 0, "sharpe": 0, "max_drawdown_pct": 0,
        "win_rate_pct": 0, "n_periods": 0, "avg_positions": 0,
        "stopped_early": stopped,
    }
