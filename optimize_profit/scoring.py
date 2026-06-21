"""Scoring functions for the walk-forward sweep.

Each function takes a `member_rankings` DataFrame (output of `rank_members`)
and returns `{member: score}` where higher = better (for ranking purposes).

All functions are continuous and differentiable in their inputs (so they're
amenable to gradient-based downstream tuning).

Available scoring modes:
  - shrunk_alpha          baseline alpha
  - inverted_alpha        fixes the negative-correlation bug
  - trade_frequency       log(1 + trades) — activity-weighted
  - consistency           prob_up * log(1 + trades)
  - bayesian_quality      bayes_win_prob * shrunk_alpha
  - neg_bayesian_quality  inverted combined signal
  - smooth_trade_thresh   expit-smoothed trade threshold at 5
  - softplus_quality      softplus-smoothed threshold at 3
  - sharpe                risk-adjusted return ratio
"""

import numpy as np
from scipy.special import expit, softplus as _softplus


def score_shrunk_alpha(member_rankings):
    """Baseline: shrunk_alpha (INVERTED — picks worst performers)."""
    return dict(zip(member_rankings["member"], member_rankings["shrunk_alpha"]))


def score_inverted_alpha(member_rankings):
    """Inverted shrunk_alpha — fixes the negative correlation."""
    return dict(zip(member_rankings["member"], -member_rankings["shrunk_alpha"]))


def score_trade_frequency(member_rankings):
    """log(1 + trade_count) — more active members score higher."""
    return dict(
        zip(member_rankings["member"], np.log1p(member_rankings["purchase_trades"]))
    )


def score_consistency(member_rankings):
    """prob_up * log(1 + trades) — consistent winners with volume."""
    prob = member_rankings["prob_up_given_buy"].values
    trades = np.log1p(member_rankings["purchase_trades"].values)
    return dict(zip(member_rankings["member"], prob * trades))


def score_bayesian_quality(member_rankings):
    """bayes_win_prob * shrunk_alpha — combined signal."""
    bayes = member_rankings["bayes_win_prob"].values
    alpha = member_rankings["shrunk_alpha"].values
    return dict(zip(member_rankings["member"], bayes * alpha))


def score_neg_bayesian_quality(member_rankings):
    """-bayes_win_prob * shrunk_alpha — inverted combined signal."""
    bayes = member_rankings["bayes_win_prob"].values
    alpha = member_rankings["shrunk_alpha"].values
    return dict(zip(member_rankings["member"], -bayes * alpha))


def score_smooth_trade_threshold(member_rankings):
    """Sigmoid-thresholded trade count weighted by prob_up.

    Uses expit for smooth thresholding at 5 trades (differentiable).
    """
    trades = member_rankings["purchase_trades"].values.astype(float)
    prob = member_rankings["prob_up_given_buy"].values
    trade_weight = expit((trades - 5) / 2.0)  # sigmoid centered at 5
    return dict(zip(member_rankings["member"], prob * trade_weight))


def score_softplus_quality(member_rankings):
    """Softplus-thresholded trades weighted by bayes_win_prob.

    softplus(x) = log(1 + exp(x)) — smooth approximation to ReLU.
    Threshold at 3 trades, normalized.
    """
    trades = member_rankings["purchase_trades"].values.astype(float)
    bayes = member_rankings["bayes_win_prob"].values
    # softplus(trades - 3) smoothly activates above 3 trades
    trade_signal = _softplus(trades - 3)
    norm = float(_softplus(np.array([10.0])).item())  # normalize by softplus(10)
    trade_weight = trade_signal / norm
    return dict(zip(member_rankings["member"], bayes * trade_weight))


def score_sharpe(member_rankings):
    """Sharpe ratio — risk-adjusted return."""
    return dict(zip(member_rankings["member"], member_rankings["sharpe_ratio"]))


SCORING_FUNCTIONS = {
    "shrunk_alpha": score_shrunk_alpha,
    "inverted_alpha": score_inverted_alpha,
    "trade_frequency": score_trade_frequency,
    "consistency": score_consistency,
    "bayesian_quality": score_bayesian_quality,
    "neg_bayesian_quality": score_neg_bayesian_quality,
    "smooth_trade_thresh": score_smooth_trade_threshold,
    "softplus_quality": score_softplus_quality,
    "sharpe": score_sharpe,
}
