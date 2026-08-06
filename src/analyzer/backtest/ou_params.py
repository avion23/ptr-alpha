"""Ornstein-Uhlenbeck (OU) parameter estimation for ticker signals.

Fits a mean-reverting OU process to a ticker's historical return curves and
returns the entry value V(0) and optimal holding horizon. Falls back to a
global prior (curves across all tickers) when the ticker has no own
disclosure history.
"""

from __future__ import annotations

import pandas as pd

from analyzer._memo import df_memoize

from analyzer.backtest.curves import _build_global_curves, _build_ticker_curves


@df_memoize(copy=False)
def _compute_ticker_ou_params(
    ticker: str,
    signals_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    as_of_date: pd.Timestamp,
    horizon: int,
    rho: float = 0.000137,
) -> tuple[float | None, int]:
    """Compute both OU entry value V(0) and optimal horizon for a ticker.

    Builds curves once and fits OU once, returning both V0 and optimal
    holding period.  Falls back to global prior (average across all
    tickers) when the ticker has no own disclosure history.

    Returns (v0, optimal_horizon).
    """
    from analyzer.return_process import compute_entry_value_and_horizon

    ticker_col = ticker in prices_df.columns if hasattr(prices_df, "columns") else False

    curves: list = []
    if ticker_col:
        curves = _build_ticker_curves(
            ticker, signals_df, prices_df, as_of_date, horizon
        )

    if not curves:
        curves = _build_global_curves(signals_df, prices_df, as_of_date, horizon)

    if not curves:
        return (None, horizon)

    v0, _mu, _theta, optimal_h = compute_entry_value_and_horizon(curves, rho=rho)
    return v0, optimal_h


@df_memoize(copy=False)
def _compute_ticker_entry_value(
    ticker: str,
    signals_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    as_of_date: pd.Timestamp,
    horizon: int,
    rho: float = 0.000137,
) -> float | None:
    """Compute OU entry value V(0) for a ticker from historical return curves."""
    v0, _ = _compute_ticker_ou_params(
        ticker, signals_df, prices_df, as_of_date, horizon, rho
    )
    return v0


@df_memoize(copy=False)
def _compute_ticker_optimal_horizon(
    ticker: str,
    signals_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    as_of_date: pd.Timestamp,
    horizon: int,
    min_horizon: int = 20,
    max_horizon: int = 120,
) -> int:
    """Compute optimal holding period for a ticker from historical curves."""
    _v0, optimal_h = _compute_ticker_ou_params(
        ticker, signals_df, prices_df, as_of_date, horizon
    )
    return max(min_horizon, min(max_horizon, optimal_h))
