"""Bayesian math helpers for member ranking.

Reads the module global `BAYES_PRIOR_STRENGTH` from `analyzer.signals` unless
a prior strength is supplied per call.
"""

from __future__ import annotations

from analyzer import signals as _signals


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
