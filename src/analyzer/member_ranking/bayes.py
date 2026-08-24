"""Bayesian math helpers for member ranking.

Reads the module global `BAYES_PRIOR_STRENGTH` from `analyzer.signals` unless
a prior strength is supplied per call.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analyzer import signals as _signals


def bayesian_win_probability(
    wins: int,
    losses: int,
    market_prior: float = 0.55,
    prior_strength: float | None = None,
) -> float:
    ps = prior_strength if prior_strength is not None else _signals.BAYES_PRIOR_STRENGTH
    if wins < 0 or losses < 0:
        raise ValueError("wins and losses must be non-negative")
    if not 0 < market_prior < 1:
        raise ValueError("market_prior must be strictly between zero and one")
    if ps <= 0 or not np.isfinite(ps):
        raise ValueError("prior_strength must be positive and finite")
    alpha = market_prior * ps
    beta = (1 - market_prior) * ps
    return (alpha + wins) / (alpha + beta + wins + losses)


def normal_normal_posteriors(
    outcomes,
    groups,
    *,
    information_weights=None,
    prior_strength: float = 1.0,
) -> pd.DataFrame:
    """Fit one scale-equivariant empirical normal-normal model.

    Hyperparameters use unweighted observations so uniformly aging every row
    changes information, not the estimated population. ``information_weights``
    are power-likelihood weights: each group's effective information is their
    sum. Returned member effects are descriptive associations, not causal.

    The between-group variance is the non-negative method-of-moments estimate.
    When the observed spread of group means is fully explained by sampling
    noise, the model therefore approaches complete pooling instead of imposing
    an artificial lower bound equal to one observation's residual variance.
    """
    values = np.asarray(outcomes, dtype=float)
    labels = np.asarray(groups, dtype=object)
    if values.ndim != 1 or labels.ndim != 1 or len(values) != len(labels):
        raise ValueError("outcomes and groups must be aligned one-dimensional arrays")
    if len(values) == 0:
        return pd.DataFrame(
            columns=[
                "posterior_mean",
                "posterior_std",
                "shrinkage",
                "effective_information",
                "global_mean",
                "within_var",
                "between_var",
            ]
        )
    if not np.isfinite(values).all():
        raise ValueError("outcomes must be finite")
    if pd.isna(labels).any():
        raise ValueError("groups must be non-null")
    if prior_strength <= 0 or not np.isfinite(prior_strength):
        raise ValueError("prior_strength must be positive and finite")

    if information_weights is None:
        weights = np.ones(len(values), dtype=float)
    else:
        weights = np.asarray(information_weights, dtype=float)
        if weights.ndim != 1 or len(weights) != len(values):
            raise ValueError("information_weights must align with outcomes")
        if not np.isfinite(weights).all() or np.any(weights <= 0):
            raise ValueError("information_weights must be positive and finite")

    frame = pd.DataFrame({"group": labels, "outcome": values, "weight": weights})
    unweighted = frame.groupby("group", sort=False)["outcome"]
    group_counts = unweighted.size().astype(float)
    group_means = unweighted.mean().astype(float)
    global_mean = float(group_means.mean())

    residuals = frame["outcome"] - frame["group"].map(group_means)
    within_dof = len(frame) - len(group_means)
    if within_dof > 0:
        within_var = float(np.dot(residuals, residuals) / within_dof)
    elif len(values) > 1:
        within_var = float(np.var(values, ddof=1))
    else:
        within_var = 0.0

    magnitude = max(
        float(np.max(np.abs(values))),
        float(np.ptp(values)),
        np.sqrt(np.finfo(float).tiny),
    )
    # Keep reciprocal-scale quantities representable. Smallest subnormal
    # variances overflow when inverted, even though the input is valid.
    variance_floor = max(
        (np.finfo(float).eps * magnitude) ** 2,
        1.0 / np.finfo(float).max,
    )
    within_var = max(within_var, variance_floor)

    observed_between = (
        float(np.var(group_means.to_numpy(dtype=float), ddof=1))
        if len(group_means) > 1
        else 0.0
    )
    mean_sampling_var = float((within_var / group_counts).mean())
    moment_between_var = observed_between - mean_sampling_var
    # A variance component may legitimately be unresolved at zero. Keeping only
    # a representability floor yields full pooling in that case; forcing it to
    # at least within_var would manufacture heterogeneity and under-shrink noisy
    # member histories.
    between_var = max(moment_between_var, variance_floor)

    weighted_sum = (frame["outcome"] * frame["weight"]).groupby(frame["group"]).sum()
    information = frame.groupby("group", sort=False)["weight"].sum()
    weighted_means = weighted_sum / information
    group_order = group_means.index
    information = information.reindex(group_order).astype(float)
    weighted_means = weighted_means.reindex(group_order).astype(float)

    # Work in the variance domain instead of forming precision reciprocals.
    # This remains finite for exact-zero data at the representable variance
    # floor and is algebraically identical to normal-normal precision updates.
    denominator = information * between_var + prior_strength * within_var
    shrinkage = prior_strength * within_var / denominator
    posterior_mean = (1.0 - shrinkage) * weighted_means + shrinkage * global_mean
    posterior_var = (within_var / denominator) * between_var

    return pd.DataFrame(
        {
            "posterior_mean": posterior_mean,
            "posterior_std": np.sqrt(posterior_var),
            "shrinkage": shrinkage,
            "effective_information": information,
            "global_mean": global_mean,
            "within_var": within_var,
            "between_var": between_var,
        },
        index=group_order,
    )
