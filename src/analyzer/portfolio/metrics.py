"""Aggregate portfolio performance metrics from a return series.

Takes the output of `simulate_portfolio_returns` and computes:
- total cumulative return (portfolio + SPY)
- annualized return
- annualized Sharpe (sqrt(12) factor for monthly-ish rebalance)
- max drawdown on equity curve
- win rate (% of periods with positive return)
- average alpha vs SPY
- total position count
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SQRT_12 = np.sqrt(12)
DEFAULT_AVG_DAYS_PER_PERIOD = 30


def compute_portfolio_metrics(portfolio_returns: pd.DataFrame) -> dict:
    """Compute aggregate portfolio performance metrics from return series."""
    if portfolio_returns.empty:
        return {}

    rets = portfolio_returns["portfolio_return"].values
    spy_rets = portfolio_returns["spy_return"].values

    cumulative = _cumulative_return(rets)
    spy_cumulative = _cumulative_return(spy_rets)
    ann_return = _annualized_return(cumulative, len(rets))
    sharpe = _sharpe_ratio(rets)
    max_dd = _max_drawdown(rets)
    win_rate = _win_rate(rets)
    avg_alpha = _avg_alpha(portfolio_returns)
    total_positions = _total_positions(portfolio_returns)

    return {
        "total_return_pct": round(cumulative * 100, 2),
        "annualized_return_pct": round(ann_return, 2),
        "sharpe_ratio": round(sharpe, 3),
        "max_drawdown_pct": round(max_dd, 2),
        "win_rate_pct": round(win_rate, 1),
        "avg_alpha_per_period_pct": round(avg_alpha, 2),
        "spy_total_return_pct": round(spy_cumulative * 100, 2),
        "n_periods": len(rets),
        "total_positions": total_positions,
    }


def _cumulative_return(rets: np.ndarray) -> float:
    return float(np.prod(1 + rets / 100) - 1)


def _annualized_return(cumulative: float, n_periods: int) -> float:
    """Annualize assuming ~30-day rebalance frequency."""
    if n_periods <= 0:
        return 0.0
    years = n_periods * DEFAULT_AVG_DAYS_PER_PERIOD / 365
    if years <= 0:
        return 0.0
    return ((1 + cumulative) ** (1 / years) - 1) * 100


def _sharpe_ratio(rets: np.ndarray) -> float:
    """Annualized Sharpe (sqrt(12) factor for monthly-ish rebalance)."""
    if len(rets) > 1 and np.std(rets) > 0:
        return float(np.mean(rets) / np.std(rets, ddof=1) * SQRT_12)
    return 0.0


def _max_drawdown(rets: np.ndarray) -> float:
    """Max drawdown on cumulative equity curve (returned as negative %)."""
    equity = np.cumprod(1 + rets / 100)
    peak = np.maximum.accumulate(equity)
    drawdowns = (equity - peak) / peak
    return float(np.min(drawdowns)) * 100 if len(drawdowns) > 0 else 0.0


def _win_rate(rets: np.ndarray) -> float:
    return float(np.mean(rets > 0) * 100) if len(rets) > 0 else 0.0


def _avg_alpha(portfolio_returns: pd.DataFrame) -> float:
    return float(portfolio_returns["portfolio_alpha"].values.mean())


def _total_positions(portfolio_returns: pd.DataFrame) -> int:
    if "num_positions" in portfolio_returns.columns:
        return int(portfolio_returns["num_positions"].sum())
    return 0
