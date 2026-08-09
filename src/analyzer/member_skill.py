from __future__ import annotations

from dataclasses import dataclass, field
from math import exp, log, sqrt

import numpy as np
import pandas as pd

from analyzer import signals as _signals
from analyzer.analysis import TransactionType, _collapse_to_episodes


_OUTCOME_COL = "total_spy_alpha_pct"


@dataclass
class MemberSkillPosterior:
    """Descriptive, noncausal posterior association for one member."""

    member: str
    alpha_mean: float
    alpha_std: float
    n_episodes: int
    shrinkage: float
    ticker_skills: dict[str, float] = field(default_factory=dict)


def _recency_weight(
    disclosure_date: pd.Timestamp,
    ref_date: pd.Timestamp,
    half_life_days: int,
) -> float:
    """Exponential power-likelihood weight based on days since disclosure."""
    if half_life_days <= 0:
        raise ValueError("recency_half_life_days must be positive")
    days_ago = max((ref_date - disclosure_date).days, 0)
    return exp(-days_ago * log(2) / half_life_days)


def _eligible_signals(
    signals_df: pd.DataFrame,
    horizon: int,
    ref_date: pd.Timestamp,
) -> pd.DataFrame:
    if _OUTCOME_COL not in signals_df.columns:
        raise ValueError(
            "Member skill requires endpoint SPY alpha; total_spy_alpha_pct is missing"
        )
    eligible = signals_df[
        (signals_df["horizon_days"] == horizon)
        & (signals_df["signal_type"] == TransactionType.PURCHASE.value)
        & (signals_df["disclosure_date"] <= ref_date - pd.Timedelta(days=horizon))
        & signals_df[_OUTCOME_COL].notna()
    ].copy()
    if "window_complete" in eligible.columns:
        eligible = eligible[eligible["window_complete"].fillna(False).astype(bool)]
    return eligible


def _weighted_member_rows(
    signals_df: pd.DataFrame,
    horizon: int,
    ref_date: pd.Timestamp,
    recency_half_life_days: int,
) -> pd.DataFrame:
    if recency_half_life_days <= 0:
        raise ValueError("recency_half_life_days must be positive")
    eligible = _eligible_signals(signals_df, horizon, ref_date)
    if eligible.empty:
        return eligible
    collapsed = _collapse_to_episodes(eligible)
    days_ago = (ref_date - pd.to_datetime(collapsed["disclosure_date"])).dt.days
    collapsed["_weight_vals"] = np.exp(
        -days_ago.clip(lower=0).to_numpy(dtype=float)
        * np.log(2)
        / recency_half_life_days
    )
    return collapsed


def _compute_member_raw_alphas(
    signals_df: pd.DataFrame,
    horizon: int,
    ref_date: pd.Timestamp,
    recency_half_life_days: int,
) -> dict[str, tuple[float, float, int, float]]:
    """Return weighted endpoint alpha, weight sum, episode count, squared-weight sum."""
    collapsed = _weighted_member_rows(
        signals_df, horizon, ref_date, recency_half_life_days
    )
    if collapsed.empty:
        return {}

    collapsed["_weight_sq_vals"] = collapsed["_weight_vals"] ** 2
    collapsed["_alpha_weighted"] = collapsed[_OUTCOME_COL].to_numpy(
        dtype=float
    ) * collapsed["_weight_vals"].to_numpy(dtype=float)
    grp = collapsed.groupby("member")
    weight_sums = grp["_weight_vals"].sum()
    weight_sq_sums = grp["_weight_sq_vals"].sum()
    alpha_sums = grp["_alpha_weighted"].sum()
    n_episodes = grp.size()

    result: dict[str, tuple[float, float, int, float]] = {}
    for member in weight_sums.index[weight_sums > 0]:
        weight_sum = float(weight_sums[member])
        result[member] = (
            float(alpha_sums[member]) / weight_sum,
            weight_sum,
            int(n_episodes[member]),
            float(weight_sq_sums[member]),
        )
    return result


def _compute_member_ticker_skills_from_group(
    member_signals: pd.DataFrame,
    horizon: int,
    ref_date: pd.Timestamp,
    recency_half_life_days: int,
) -> dict[str, float]:
    """Return descriptive endpoint-alpha means keyed by ticker."""
    if member_signals.empty:
        return {}
    collapsed = _weighted_member_rows(
        member_signals, horizon, ref_date, recency_half_life_days
    )
    if collapsed.empty:
        return {}
    weighted = collapsed[_OUTCOME_COL] * collapsed["_weight_vals"]
    weight_sums = collapsed["_weight_vals"].groupby(collapsed["ticker"]).sum()
    alpha_sums = weighted.groupby(collapsed["ticker"]).sum()
    return {
        ticker: float(alpha_sums[ticker] / weight_sums[ticker])
        for ticker in weight_sums.index[weight_sums > 0]
    }


def estimate_member_skills(
    signals_df: pd.DataFrame,
    min_episodes: int = 1,
    prior_strength: float | None = None,
    recency_half_life_days: int = 365,
    horizon: int = 90,
    ref_date: pd.Timestamp | None = None,
) -> dict[str, MemberSkillPosterior]:
    """Estimate descriptive normal-normal member associations.

    Endpoint excess alpha is modeled with a common Normal prior. Recency
    weights are power-likelihood information weights, so uniformly older data
    have less information. Posterior means and variances use the same data and
    prior precisions. These associations are not causal member skill: ticker,
    sector, co-buyer, and market-regime confounding remain. Production trading
    must not consume them until time-blocked validation supports that use.
    """
    if signals_df.empty:
        return {}
    if min_episodes < 1:
        raise ValueError("min_episodes must be positive")
    if prior_strength is None:
        prior_strength = float(_signals.BAYES_PRIOR_STRENGTH)
    if prior_strength <= 0:
        raise ValueError("prior_strength must be positive")
    if recency_half_life_days <= 0:
        raise ValueError("recency_half_life_days must be positive")
    if ref_date is None:
        ref_date = pd.Timestamp.now()

    collapsed = _weighted_member_rows(
        signals_df, horizon, ref_date, recency_half_life_days
    )
    if collapsed.empty:
        return {}
    raw = _compute_member_raw_alphas(
        signals_df, horizon, ref_date, recency_half_life_days
    )
    qualifying = {
        member: values for member, values in raw.items() if values[2] >= min_episodes
    }
    if not qualifying:
        return {}

    member_means = np.array([values[0] for values in qualifying.values()], dtype=float)
    global_mean = float(member_means.mean())

    pooled_sse = 0.0
    pooled_dof = 0.0
    member_groups = {member: group for member, group in collapsed.groupby("member")}
    for member, (raw_alpha, weight_sum, _, weight_sq_sum) in qualifying.items():
        group = member_groups[member]
        weights = group["_weight_vals"].to_numpy(dtype=float)
        residuals = group[_OUTCOME_COL].to_numpy(dtype=float) - raw_alpha
        pooled_sse += float(np.dot(weights, residuals**2))
        pooled_dof += max(weight_sum - weight_sq_sum / weight_sum, 0.0)

    observed_between = (
        float(np.var(member_means, ddof=1)) if len(member_means) > 1 else 0.0
    )
    within_var = pooled_sse / pooled_dof if pooled_dof > 0 else observed_between
    all_values = collapsed[_OUTCOME_COL].to_numpy(dtype=float)
    variance_scale = max(float(np.var(all_values)), 1.0)
    variance_floor = variance_scale * 1e-8
    within_var = max(float(within_var), variance_floor)

    # Keep the common prior fixed under a uniform rescaling of recency
    # information. Otherwise making every observation older can collapse the
    # estimated between-member variance and create false certainty.
    tau_sq = max(observed_between, variance_floor)
    prior_precision = float(prior_strength) / tau_sq

    posteriors: dict[str, MemberSkillPosterior] = {}
    for member, (raw_alpha, weight_sum, n, _) in qualifying.items():
        data_precision = weight_sum / within_var
        posterior_precision = data_precision + prior_precision
        posterior_mean = (
            data_precision * raw_alpha + prior_precision * global_mean
        ) / posterior_precision
        posterior_std = sqrt(1.0 / posterior_precision)
        shrinkage = prior_precision / posterior_precision
        posteriors[member] = MemberSkillPosterior(
            member=member,
            alpha_mean=float(posterior_mean),
            alpha_std=float(posterior_std),
            n_episodes=n,
            shrinkage=float(shrinkage),
            ticker_skills=_compute_member_ticker_skills_from_group(
                member_groups[member],
                horizon,
                ref_date,
                recency_half_life_days,
            ),
        )
    return posteriors


def score_members_for_ticker(
    ticker: str,
    members_bought: list[str],
    skills: dict[str, MemberSkillPosterior],
) -> tuple[float, float]:
    """Combine independent diagnostic member posteriors by precision.

    Returns a descriptive `(mean, standard_error)` pair. The calculation is
    not ticker-conditioned and must not be used as a tradable score. `ticker`
    identifies the candidate for callers and must be non-empty.
    """
    if not ticker:
        raise ValueError("ticker must be non-empty")
    posteriors = [skills[member] for member in members_bought if member in skills]
    if not posteriors:
        return (0.0, 1.0)

    variances = np.array(
        [max(posterior.alpha_std**2, 1e-12) for posterior in posteriors], dtype=float
    )
    precisions = 1.0 / variances
    normalized = precisions / precisions.sum()
    means = np.array([posterior.alpha_mean for posterior in posteriors], dtype=float)
    expected_alpha = float(np.dot(normalized, means))
    uncertainty = sqrt(1.0 / float(precisions.sum()))
    return expected_alpha, uncertainty
