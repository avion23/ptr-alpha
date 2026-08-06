from __future__ import annotations

from dataclasses import dataclass, field
from math import exp, log, sqrt

import numpy as np
import pandas as pd

from analyzer import signals as _signals
from analyzer.analysis import TransactionType, _collapse_to_episodes


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
) -> dict[str, tuple[float, float, int, float]]:
    """Compute recency-weighted alpha per member.

    Returns {member: (weighted_alpha, weight_sum, n_episodes, weight_sq_sum)}.
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

    # Vectorized recency weight computation (no iterrows)
    days_ago = (ref_date - pd.to_datetime(collapsed["disclosure_date"])).dt.days
    days_ago = days_ago.clip(lower=0)
    weights = np.exp(-days_ago.values * np.log(2) / recency_half_life_days)

    collapsed["_weight_vals"] = weights
    collapsed["_weight_sq_vals"] = weights**2
    collapsed["_alpha_weighted"] = collapsed["spy_alpha_pct"].values * weights

    grp = collapsed.groupby("member")
    weight_sums = grp["_weight_vals"].sum()
    weight_sq_sums = grp["_weight_sq_vals"].sum()
    alpha_sums = grp["_alpha_weighted"].sum()
    n_episodes = grp.size()

    # Filter groups with weight_sum > 0
    valid = weight_sums > 0
    result: dict[str, tuple[float, float, int, float]] = {}
    for member in weight_sums.index[valid]:
        w = float(weight_sums[member])
        result[member] = (
            float(alpha_sums[member]) / w,
            w,
            int(n_episodes[member]),
            float(weight_sq_sums[member]),
        )
    return result


def _compute_member_sector_skills_from_group(
    member_signals: pd.DataFrame,
    horizon: int,
    ref_date: pd.Timestamp,
    recency_half_life_days: int,
) -> dict[str, float]:
    """Compute sector-specific skill from a pre-filtered member group.

    Same logic as _compute_member_sector_skills but assumes the caller has
    already filtered by member. Avoids repeated full-table boolean masks.
    """
    if member_signals.empty:
        return {}

    filtered = member_signals[
        (member_signals["horizon_days"] == horizon)
        & (member_signals["signal_type"] == TransactionType.PURCHASE.value)
        & (member_signals["disclosure_date"] <= ref_date - pd.Timedelta(days=horizon))
        & (member_signals["spy_alpha_pct"].notna())
    ]

    if filtered.empty:
        return {}

    days_ago = (ref_date - pd.to_datetime(filtered["disclosure_date"])).dt.days
    days_ago = days_ago.clip(lower=0)
    weight = np.exp(-days_ago.values * np.log(2) / recency_half_life_days)
    alpha_weighted = filtered["spy_alpha_pct"].values * weight

    weight_sums = pd.Series(weight, index=filtered.index).groupby(filtered["ticker"]).sum()
    alpha_sums = pd.Series(alpha_weighted, index=filtered.index).groupby(filtered["ticker"]).sum()

    valid = weight_sums > 0
    return {t: float(alpha_sums[t]) / float(weight_sums[t]) for t in weight_sums.index[valid]}


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
        prior_strength = _signals.BAYES_PRIOR_STRENGTH  # 20, unified with member_ranking
    if ref_date is None:
        ref_date = pd.Timestamp.now()
    raw = _compute_member_raw_alphas(
        signals_df, horizon, ref_date, recency_half_life_days
    )

    if not raw:
        return {}

    # Filter members with enough episodes
    qualifying = {
        m: (alpha, w, n, w_sq)
        for m, (alpha, w, n, w_sq) in raw.items()
        if n >= min_episodes
    }

    if not qualifying:
        # Fall back: use all members with at least 1 episode
        qualifying = raw

    # Global parameters from qualifying members
    alphas = np.array([v[0] for v in qualifying.values()])
    global_mean = float(np.mean(alphas))
    global_var = float(np.var(alphas)) if len(alphas) > 1 else 0.0

    posteriors: dict[str, MemberSkillPosterior] = {}
    # Pre-group signals_df by member to avoid repeated full-table scans
    member_groups: dict[str, pd.DataFrame] = {}
    for m in qualifying:
        if m not in member_groups:
            member_groups[m] = signals_df[signals_df["member"] == m]

    pooled_residual_sum = 0.0
    pooled_weight_sum = 0.0
    for member, (raw_alpha, _, _, _) in qualifying.items():
        member_signals = member_groups[member]
        eligible = member_signals[
            (member_signals["horizon_days"] == horizon)
            & (member_signals["signal_type"] == TransactionType.PURCHASE.value)
            & (member_signals["disclosure_date"] <= ref_date - pd.Timedelta(days=horizon))
            & (member_signals["spy_alpha_pct"].notna())
        ].copy()
        collapsed = _collapse_to_episodes(eligible)
        days_ago = (ref_date - pd.to_datetime(collapsed["disclosure_date"])).dt.days
        weights = np.exp(
            -days_ago.clip(lower=0).values * np.log(2) / recency_half_life_days
        )
        residuals = collapsed["spy_alpha_pct"].to_numpy() - raw_alpha
        pooled_residual_sum += float(np.dot(weights, residuals**2))
        pooled_weight_sum += float(weights.sum())

    residual_dof = pooled_weight_sum - len(qualifying)
    within_var = pooled_residual_sum / residual_dof if residual_dof > 0 else global_var
    if not np.isfinite(within_var) or within_var < 0:
        within_var = global_var

    variance_floor = 1e-12
    for member, (raw_alpha, weight_sum, n, weight_sq_sum) in qualifying.items():
        if weight_sum > 0 and weight_sq_sum > 0:
            effective_n = weight_sum**2 / weight_sq_sum
        else:
            effective_n = float(n)

        shrinkage = prior_strength / (effective_n + prior_strength)
        posterior_mean = (1 - shrinkage) * raw_alpha + shrinkage * global_mean
        sigma_sq = max(within_var, variance_floor)
        tau_sq = max(global_var, variance_floor)
        posterior_var = 1.0 / (effective_n / sigma_sq + 1.0 / tau_sq)
        posterior_std = sqrt(max(posterior_var, variance_floor**2))

        member_signals = member_groups.get(member, pd.DataFrame())
        sector_skills = _compute_member_sector_skills_from_group(
            member_signals, horizon, ref_date, recency_half_life_days
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

    # Uncertainty via bootstrap (vectorized — no per-iteration loop)
    rng = np.random.default_rng(42)
    # Generate all samples at once: shape (n_bootstrap, n_posteriors)
    means_arr = np.array([p.alpha_mean for p in posteriors])
    stds_arr = np.array([max(p.alpha_std, 1e-6) for p in posteriors])
    all_samples = rng.normal(means_arr[None, :], stds_arr[None, :], size=(n_bootstrap, len(posteriors)))
    bootstrap_alphas = all_samples @ weight_arr

    uncertainty = float(np.std(bootstrap_alphas))
    return (expected_alpha, uncertainty)
