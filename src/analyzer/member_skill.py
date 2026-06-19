from __future__ import annotations

from dataclasses import dataclass, field
from math import exp, log, sqrt

import numpy as np
import pandas as pd

from analyzer.analysis import TransactionType, _collapse_to_episodes
from analyzer.signals import BAYES_PRIOR_STRENGTH


@dataclass
class MemberSkillPosterior:
    member: str
    alpha_mean: float  # posterior mean
    alpha_std: float   # posterior std
    n_episodes: int    # number of fully-elapsed episodes
    shrinkage: float   # how much pulled toward global mean
    sector_skills: dict[str, float] = field(default_factory=dict)  # sector-specific skill


def _recency_weight(
    disclosure_date: pd.Timestamp,
    ref_date: pd.Timestamp,
    half_life_days: int,
) -> float:
    """Exponential decay weight based on days since disclosure."""
    days_ago = (ref_date - disclosure_date).days
    if days_ago < 0:
        days_ago = 0
    return exp(-days_ago * log(2) / half_life_days)


def _compute_member_raw_alphas(
    signals_df: pd.DataFrame,
    horizon: int,
    ref_date: pd.Timestamp,
    recency_half_life_days: int,
) -> dict[str, tuple[float, float, int]]:
    """Compute recency-weighted alpha per member.

    Returns {member: (weighted_alpha, weight_sum, n_episodes)}.
    """
    eligible = signals_df[
        (signals_df["horizon_days"] == horizon)
        & (signals_df["signal_type"] == TransactionType.PURCHASE.value)
        & (signals_df["disclosure_date"] <= ref_date - pd.Timedelta(days=horizon))
        & (signals_df["spy_alpha_pct"].notna())
    ].copy()

    if eligible.empty:
        return {}

    collapsed = _collapse_to_episodes(eligible)

    result: dict[str, tuple[float, float, int]] = {}
    for member, grp in collapsed.groupby("member"):
        alpha_sum = 0.0
        weight_sum = 0.0
        n_episodes = 0
        for _, row in grp.iterrows():
            alpha = float(row["spy_alpha_pct"])
            weight = _recency_weight(
                pd.Timestamp(row["disclosure_date"]),
                ref_date,
                recency_half_life_days,
            )
            alpha_sum += alpha * weight
            weight_sum += weight
            n_episodes += 1
        if weight_sum > 0:
            result[member] = (alpha_sum / weight_sum, weight_sum, n_episodes)
    return result


def _compute_member_sector_skills(
    signals_df: pd.DataFrame,
    member: str,
    horizon: int,
    ref_date: pd.Timestamp,
    recency_half_life_days: int,
) -> dict[str, float]:
    """Compute sector-specific skill for a member (ticker as sector proxy).

    Returns {ticker: weighted_alpha}.
    """
    member_signals = signals_df[
        (signals_df["member"] == member)
        & (signals_df["horizon_days"] == horizon)
        & (signals_df["signal_type"] == TransactionType.PURCHASE.value)
        & (signals_df["disclosure_date"] <= ref_date - pd.Timedelta(days=horizon))
        & (signals_df["spy_alpha_pct"].notna())
    ].copy()

    if member_signals.empty:
        return {}

    sector_alphas: dict[str, tuple[float, float]] = {}
    for ticker, grp in member_signals.groupby("ticker"):
        alpha_sum = 0.0
        weight_sum = 0.0
        for _, row in grp.iterrows():
            alpha = float(row["spy_alpha_pct"])
            weight = _recency_weight(
                pd.Timestamp(row["disclosure_date"]),
                ref_date,
                recency_half_life_days,
            )
            alpha_sum += alpha * weight
            weight_sum += weight
        if weight_sum > 0:
            sector_alphas[ticker] = alpha_sum / weight_sum
    return sector_alphas


def estimate_member_skills(
    signals_df: pd.DataFrame,
    min_episodes: int = 1,
    prior_strength: float | None = None,
    recency_half_life_days: int = 365,
    horizon: int = 90,
    ref_date: pd.Timestamp | None = None,
) -> dict[str, MemberSkillPosterior]:
    """Estimate Bayesian posterior skill for each member.

    Uses empirical Bayes:
    1. Compute per-member historical alpha (weighted by recency)
    2. Compute global mean and variance of member alphas
    3. Shrink each member's estimate toward global mean

    Args:
        signals_df: Historical signal data with columns including
            member, ticker, disclosure_date, signal_type, horizon_days,
            spy_alpha_pct.
        min_episodes: Minimum episodes required to include a member.
        prior_strength: Strength of the prior (pseudo-observations).
            Defaults to BAYES_PRIOR_STRENGTH (20) for consistency with
            member_ranking module.
        recency_half_life_days: Half-life for recency weighting in days.
        horizon: Horizon in days to filter signals.
        ref_date: Reference date for recency weighting. MUST be provided
            for backtesting to avoid look-ahead bias. Defaults to now()
            only for live analysis.

    Returns:
        Mapping of member name to MemberSkillPosterior.
    """
    if signals_df.empty:
        return {}

    if prior_strength is None:
        prior_strength = BAYES_PRIOR_STRENGTH  # 20, unified with member_ranking
    if ref_date is None:
        ref_date = pd.Timestamp.now()
    raw = _compute_member_raw_alphas(
        signals_df, horizon, ref_date, recency_half_life_days
    )

    if not raw:
        return {}

    # Filter members with enough episodes
    qualifying = {
        m: (alpha, w, n)
        for m, (alpha, w, n) in raw.items()
        if n >= min_episodes
    }

    if not qualifying:
        # Fall back: use all members with at least 1 episode
        qualifying = raw

    # Global parameters from qualifying members
    alphas = np.array([v[0] for v in qualifying.values()])
    global_mean = float(np.mean(alphas))
    global_var = float(np.var(alphas)) if len(alphas) > 1 else 0.0
    global_std = sqrt(global_var)

    posteriors: dict[str, MemberSkillPosterior] = {}
    for member, (raw_alpha, weight_sum, n) in qualifying.items():
        shrinkage = prior_strength / (n + prior_strength)
        posterior_mean = (1 - shrinkage) * raw_alpha + shrinkage * global_mean
        posterior_std = global_std / sqrt(n + prior_strength)

        sector_skills = _compute_member_sector_skills(
            signals_df, member, horizon, ref_date, recency_half_life_days
        )

        posteriors[member] = MemberSkillPosterior(
            member=member,
            alpha_mean=posterior_mean,
            alpha_std=posterior_std,
            n_episodes=n,
            shrinkage=shrinkage,
            sector_skills=sector_skills,
        )

    return posteriors


def score_members_for_ticker(
    ticker: str,
    members_bought: list[str],
    skills: dict[str, MemberSkillPosterior],
    n_bootstrap: int = 1000,
) -> tuple[float, float]:
    """Score a ticker based on buying members' skills.

    Returns (expected_alpha, uncertainty).
    Uses posterior predictive: samples from each member's posterior,
    combines via weighted average (weights = inverse uncertainty).
    """
    relevant = [s for s in members_bought if s in skills]
    if not relevant:
        return (0.0, 1.0)

    posteriors = [skills[m] for m in relevant]

    # Weight by inverse uncertainty (1/std), floor at small epsilon
    weights = []
    for p in posteriors:
        w = 1.0 / max(p.alpha_std, 1e-6)
        weights.append(w)
    weight_arr = np.array(weights)
    weight_sum = weight_arr.sum()
    if weight_sum == 0:
        return (0.0, 1.0)
    weight_arr = weight_arr / weight_sum

    # Expected alpha = weighted mean of posterior means
    means = np.array([p.alpha_mean for p in posteriors])
    expected_alpha = float(np.dot(weight_arr, means))

    # Uncertainty via bootstrap
    rng = np.random.default_rng(42)
    bootstrap_alphas = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sampled = np.array([
            rng.normal(p.alpha_mean, max(p.alpha_std, 1e-6))
            for p in posteriors
        ])
        bootstrap_alphas[i] = np.dot(weight_arr, sampled)

    uncertainty = float(np.std(bootstrap_alphas))
    return (expected_alpha, uncertainty)
