"""Bayesian math helpers for member ranking.

Shrinkage toward market prior, Bayes factors against market null hypothesis,
and per-member historical win rate on a specific ticker. Reads the module
global `BAYES_PRIOR_STRENGTH` from `analyzer.signals` (override per-call
via the `_bayes_prior_strength` keyword).
"""

from __future__ import annotations

from math import exp, lgamma, log

import numpy as np
import pandas as pd

from analyzer import signals as _signals
from analyzer._memo import df_memoize
from analyzer.models import TransactionType


def bayesian_win_probability(
    wins: int,
    losses: int,
    market_prior: float = 0.55,
    prior_strength: float | None = None,
) -> float:
    ps = prior_strength if prior_strength is not None else _signals.BAYES_PRIOR_STRENGTH
    alpha = market_prior * ps
    beta = (1 - market_prior) * ps
    return (alpha + wins) / (alpha + beta + wins + losses)


def bayes_factor_against_market(
    wins: int,
    losses: int,
    market_prior: float = 0.55,
    prior_strength: float | None = None,
) -> float:
    observations = wins + losses
    if observations == 0:
        return 1.0
    ps = prior_strength if prior_strength is not None else _signals.BAYES_PRIOR_STRENGTH
    market_prior = float(np.clip(market_prior, 1e-6, 1 - 1e-6))
    alpha = market_prior * ps
    beta = (1 - market_prior) * ps
    log_marginal = (
        lgamma(alpha + wins)
        + lgamma(beta + losses)
        - lgamma(alpha + beta + observations)
        - lgamma(alpha)
        - lgamma(beta)
        + lgamma(alpha + beta)
    )
    log_market = wins * log(market_prior) + losses * log(1 - market_prior)
    return exp(float(np.clip(log_marginal - log_market, -50, 50)))


@df_memoize(copy=False)
def _compute_ticker_member_performance(
    signals_df: pd.DataFrame,
    ticker: str,
    horizon: int,
    prior_strength: float | None = None,
) -> dict[str, tuple[float, int]]:
    """Per-member Bayesian-shrunk win rate on a specific ticker from historical signals.

    Returns {member: (shrunk_win_rate, trade_count)} for members with >= 1 trade.
    """
    ps = prior_strength if prior_strength is not None else _signals.BAYES_PRIOR_STRENGTH
    if signals_df.empty or "ticker" not in signals_df.columns:
        return {}

    purchases = _ticker_purchase_subset(signals_df, ticker, horizon)
    if purchases.empty:
        return {}

    global_win_rate = _global_purchase_win_rate(signals_df, horizon)
    return _shrunk_per_member(purchases, global_win_rate, ps)


def _ticker_purchase_subset(signals_df: pd.DataFrame, ticker: str, horizon: int) -> pd.DataFrame:
    return signals_df[
        (signals_df["ticker"] == ticker)
        & (signals_df["horizon_days"] == horizon)
        & (signals_df["signal_type"] == TransactionType.PURCHASE.value)
    ]


def _global_purchase_win_rate(signals_df: pd.DataFrame, horizon: int) -> float:
    all_purchases = signals_df[
        (signals_df["horizon_days"] == horizon)
        & (signals_df["signal_type"] == TransactionType.PURCHASE.value)
    ]
    all_returns = all_purchases["decayed_return_pct"].dropna()
    return float((all_returns > 0).mean()) if len(all_returns) > 0 else 0.5


def _shrunk_per_member(purchases: pd.DataFrame, global_win_rate: float, prior_strength: float) -> dict[str, tuple[float, int]]:
    result: dict[str, tuple[float, int]] = {}
    for member, grp in purchases.groupby("member"):
        returns = grp["decayed_return_pct"].dropna()
        if len(returns) == 0:
            continue
        wins = int((returns > 0).sum())
        n = len(returns)
        # Bayesian shrinkage: pull toward global win rate
        shrunk_wr = (global_win_rate * prior_strength + wins) / (prior_strength + n)
        result[member] = (shrunk_wr, n)
    return result
