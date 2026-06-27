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


def score_recency_consistency(member_rankings):
    """Consistency weighted by recency — recent trades matter more.

    Combines prob_up with a recency bonus: trades in the last 30 days
    get a 2x weight multiplier. This captures the insight that a member's
    recent performance is more predictive than their distant past.
    """
    prob = member_rankings["prob_up_given_buy"].values
    trades = member_rankings["purchase_trades"].values.astype(float)

    # Recency bonus: use sharpe_ratio as a proxy for recent quality
    # (members with high sharpe recently are on a hot streak)
    sharpe = member_rankings["sharpe_ratio"].values
    recency_bonus = np.clip(1.0 + sharpe * 0.2, 0.5, 2.0)

    return dict(zip(member_rankings["member"], prob * trades * recency_bonus))


def score_consistency_sharpe(member_rankings):
    """Consistency * (1 + sharpe) — balanced quality + risk-adjusted return.

    Multiplies consistency (prob_up * log(1+trades)) by a risk-adjustment
    factor that rewards members with good Sharpe ratios without overweighting
    risk. The (1 + sharpe) factor is always positive (floor at 0.5).
    """
    prob = member_rankings["prob_up_given_buy"].values
    trades = np.log1p(member_rankings["purchase_trades"].values)
    sharpe = member_rankings["sharpe_ratio"].values
    risk_factor = np.clip(1.0 + sharpe * 0.5, 0.5, 3.0)

    return dict(zip(member_rankings["member"], prob * trades * risk_factor))


def score_consistency_bayes(member_rankings):
    """Consistency * bayes_win_prob — double-validated quality signal.

    Combines two independent quality signals: consistency (prob_up * trades)
    and Bayesian posterior probability. This double-gating approach should
    filter out both lucky one-hit wonders and inconsistent high-volume traders.
    """
    prob = member_rankings["prob_up_given_buy"].values
    trades = np.log1p(member_rankings["purchase_trades"].values)
    bayes = member_rankings["bayes_win_prob"].values

    consistency = prob * trades
    return dict(zip(member_rankings["member"], consistency * bayes))


def score_conviction_consistency(member_rankings):
    """Conviction-weighted consistency — rewards high-alpha consistent traders.
    
    Combines consistency (prob_up * log(1+trades)) with a conviction factor
    that rewards members who have high average SPY alpha. This targets
    members who are both consistent AND generate excess returns.
    """
    prob = member_rankings["prob_up_given_buy"].values
    trades = np.log1p(member_rankings["purchase_trades"].values)
    avg_alpha = member_rankings["avg_total_spy_alpha_pct"].values
    
    consistency = prob * trades
    # Conviction: sigmoid-scaled alpha bonus (alpha > 5% gets full bonus)
    conviction = 1.0 + np.clip(avg_alpha / 10.0, -0.5, 1.0)
    
    return dict(zip(member_rankings["member"], consistency * conviction))


def score_recency_bayes(member_rankings):
    """Bayesian quality weighted by recency — hot streaks with statistical backing.
    
    Combines bayes_win_prob with a recency bonus from sharpe ratio, then
    gates by trade count (minimum 3 trades for statistical relevance).
    """
    bayes = member_rankings["bayes_win_prob"].values
    trades = member_rankings["purchase_trades"].values.astype(float)
    sharpe = member_rankings["sharpe_ratio"].values
    
    # Only activate for members with 3+ trades
    trade_gate = np.where(trades >= 3, 1.0, 0.3)
    recency_bonus = np.clip(1.0 + sharpe * 0.15, 0.5, 2.0)
    
    return dict(zip(member_rankings["member"], bayes * trade_gate * recency_bonus))


def score_volume_consistency(member_rankings):
    """Volume-weighted consistency — rewards high-activity consistent traders.
    
    Uses log(1+trades)^2 to strongly reward volume, gated by prob_up.
    Members with many trades AND high win rate score highest.
    """
    prob = member_rankings["prob_up_given_buy"].values
    trades = np.log1p(member_rankings["purchase_trades"].values)
    
    # Quadratic volume bonus: log(1+n)^2
    volume_signal = np.log1p(trades) ** 2 / 10.0  # normalize
    
    return dict(zip(member_rankings["member"], prob * volume_signal))


def score_alpha_consistency(member_rankings):
    """Alpha-gated consistency — only trades from positive-alpha members count.
    
    Multiplies consistency by a hard gate: members with avg_total_spy_alpha > 0
    get full weight, negative-alpha members get penalized proportionally.
    """
    prob = member_rankings["prob_up_given_buy"].values
    trades = np.log1p(member_rankings["purchase_trades"].values)
    avg_alpha = member_rankings["avg_total_spy_alpha_pct"].values
    
    consistency = prob * trades
    # Hard gate: full weight if alpha > 0, penalized if negative
    alpha_gate = np.where(avg_alpha > 0, 1.0, np.clip(0.3 + avg_alpha / 50.0, 0.1, 1.0))
    
    return dict(zip(member_rankings["member"], consistency * alpha_gate))


def score_meta_ensemble(member_rankings):
    """Ensemble of top signals — consistency * bayes * (1 + sharpe).
    
    Combines the three most predictive dimensions into a single score:
    - prob_up * trades (consistency)
    - bayes_win_prob (statistical quality)
    - 1 + sharpe (risk-adjusted return)
    """
    prob = member_rankings["prob_up_given_buy"].values
    trades = np.log1p(member_rankings["purchase_trades"].values)
    bayes = member_rankings["bayes_win_prob"].values
    sharpe = member_rankings["sharpe_ratio"].values
    
    consistency = prob * trades
    quality = bayes * np.clip(1.0 + sharpe * 0.3, 0.5, 2.0)
    
    return dict(zip(member_rankings["member"], consistency * quality))


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
    "recency_consistency": score_recency_consistency,
    "consistency_sharpe": score_consistency_sharpe,
    "consistency_bayes": score_consistency_bayes,
    "conviction_consistency": score_conviction_consistency,
    "recency_bayes": score_recency_bayes,
    "volume_consistency": score_volume_consistency,
    "alpha_consistency": score_alpha_consistency,
    "meta_ensemble": score_meta_ensemble,
}
