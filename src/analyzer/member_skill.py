from __future__ import annotations

from dataclasses import dataclass
from math import exp, log, sqrt

import numpy as np
import pandas as pd

from analyzer import signals as _signals
from analyzer.analysis import TransactionType, _collapse_to_episodes
from analyzer.member_names import canonical_member_key
from analyzer.member_ranking.bayes import normal_normal_posteriors


_OUTCOME_COL = "total_spy_alpha_pct"


@dataclass(frozen=True, slots=True)
class MemberSkillPosterior:
    """Descriptive, noncausal posterior association for one member."""

    member: str
    alpha_mean: float
    alpha_std: float
    n_episodes: int
    effective_information: float
    shrinkage: float


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


def _weighted_member_rows(
    signals_df: pd.DataFrame,
    *,
    horizon: int,
    ref_date: pd.Timestamp,
    recency_half_life_days: int,
) -> pd.DataFrame:
    if recency_half_life_days <= 0:
        raise ValueError("recency_half_life_days must be positive")
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
    if eligible.empty:
        return eligible

    collapsed = _collapse_to_episodes(eligible)
    days_ago = (ref_date - pd.to_datetime(collapsed["disclosure_date"])).dt.days
    collapsed["_information_weight"] = np.exp(
        -days_ago.clip(lower=0).to_numpy(dtype=float)
        * np.log(2)
        / recency_half_life_days
    )
    return collapsed


def estimate_member_skills(
    signals_df: pd.DataFrame,
    *,
    ref_date: pd.Timestamp,
    min_episodes: int = 1,
    prior_strength: float | None = None,
    recency_half_life_days: int = 365,
    horizon: int = 90,
) -> dict[str, MemberSkillPosterior]:
    """Estimate descriptive normal-normal member associations.

    Endpoint excess alpha is modeled by the same common normal-normal fit used
    by member ranking. Recency weights scale effective information. Effects are
    predictive associations, not causal skill, and production trading rejects
    them until time-blocked validation supports their use.
    """
    if signals_df.empty:
        return {}
    if min_episodes < 1:
        raise ValueError("min_episodes must be positive")
    if prior_strength is None:
        prior_strength = float(_signals.BAYES_PRIOR_STRENGTH)
    if prior_strength <= 0:
        raise ValueError("prior_strength must be positive")

    collapsed = _weighted_member_rows(
        signals_df,
        horizon=horizon,
        ref_date=pd.Timestamp(ref_date),
        recency_half_life_days=recency_half_life_days,
    )
    if collapsed.empty:
        return {}

    counts = collapsed.groupby("member").size()
    qualifying_members = counts[counts >= min_episodes].index
    collapsed = collapsed[collapsed["member"].isin(qualifying_members)]
    if collapsed.empty:
        return {}

    fit = normal_normal_posteriors(
        collapsed[_OUTCOME_COL].to_numpy(dtype=float),
        collapsed["member"].to_numpy(dtype=object),
        information_weights=collapsed["_information_weight"].to_numpy(dtype=float),
        prior_strength=float(prior_strength),
    )
    return {
        member: MemberSkillPosterior(
            member=str(member),
            alpha_mean=float(row["posterior_mean"]),
            alpha_std=float(row["posterior_std"]),
            n_episodes=int(counts[member]),
            effective_information=float(row["effective_information"]),
            shrinkage=float(row["shrinkage"]),
        )
        for member, row in fit.iterrows()
    }


def score_member_posteriors(
    members: list[str],
    skills: dict[str, MemberSkillPosterior],
) -> tuple[float, float]:
    """Combine unique diagnostic member posteriors by inverse variance."""
    requested_identities = [canonical_member_key(member) for member in members]
    if any(not identity for identity in requested_identities):
        raise ValueError("member identities must be non-empty")
    if len(set(requested_identities)) != len(requested_identities):
        raise ValueError("duplicate member identities are not independent evidence")

    skill_identities = [canonical_member_key(member) for member in skills]
    if any(not identity for identity in skill_identities):
        raise ValueError("skill member identities must be non-empty")
    if len(set(skill_identities)) != len(skill_identities):
        raise ValueError("skills contain duplicate canonical member identities")

    by_identity = {
        canonical_member_key(member): posterior for member, posterior in skills.items()
    }
    posteriors = [
        by_identity[identity]
        for identity in requested_identities
        if identity in by_identity
    ]
    if not posteriors:
        return (0.0, 1.0)

    variances = np.array(
        [max(posterior.alpha_std**2, np.finfo(float).tiny) for posterior in posteriors],
        dtype=float,
    )
    precisions = 1.0 / variances
    normalized = precisions / precisions.sum()
    means = np.array([posterior.alpha_mean for posterior in posteriors], dtype=float)
    return float(np.dot(normalized, means)), sqrt(1.0 / float(precisions.sum()))
