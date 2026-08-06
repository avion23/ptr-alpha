"""Bayesian math helpers for member ranking.

Reads the module global `BAYES_PRIOR_STRENGTH` from `analyzer.signals` unless
a prior strength is supplied per call.
"""

from __future__ import annotations

from math import exp, lgamma, log

import numpy as np

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
