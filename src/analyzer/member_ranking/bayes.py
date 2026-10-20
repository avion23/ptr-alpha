"""Empirical normal-normal partial pooling for descriptive member research."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def normal_normal_posteriors(outcomes, groups, *, independent_groups=None) -> pd.DataFrame:
    """Fit the existing member-pooled model and expose its reporting units.

    ``groups`` still defines estimator groups (members). Optional
    ``independent_groups`` labels independent date/member units for reporting
    only; they do not alter the estimator or prior.
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
                "global_mean",
                "within_var",
                "between_var",
                "episode_count",
                "member_group_count",
                "independent_group_count",
            ]
        )
    if not np.isfinite(values).all():
        raise ValueError("outcomes must be finite")
    if pd.isna(labels).any():
        raise ValueError("groups must be non-null")
    independent = None
    if independent_groups is not None:
        independent = np.asarray(independent_groups, dtype=object)
        if independent.ndim != 1 or len(independent) != len(values):
            raise ValueError("independent_groups must align with outcomes")
        if pd.isna(independent).any():
            raise ValueError("independent_groups must be non-null")

    frame = pd.DataFrame({"group": labels, "outcome": values})
    grouped = frame.groupby("group", sort=False)["outcome"]
    counts = grouped.size().astype(float)
    means = grouped.mean().astype(float)
    global_mean = float(means.mean())

    residuals = frame["outcome"] - frame["group"].map(means)
    within_dof = len(frame) - len(means)
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
    variance_floor = max(
        (np.finfo(float).eps * magnitude) ** 2,
        1.0 / np.finfo(float).max,
    )
    within_var = max(within_var, variance_floor)

    observed_between = (
        float(np.var(means.to_numpy(dtype=float), ddof=1)) if len(means) > 1 else 0.0
    )
    mean_sampling_var = float((within_var / counts).mean())
    moment_between_var = observed_between - mean_sampling_var
    if moment_between_var <= 0:
        # The observed spread of group means is fully explained by sampling
        # noise, so the method-of-moments population variance truncates to the
        # representability floor and every posterior collapses to the global
        # mean (shrinkage identically 1). Warn so a downstream ranks table is
        # read as alphabetical ties, not signal.
        warnings.warn(
            "Member between-variance is unresolved: observed spread of member "
            f"means ({observed_between:.6g}) does not exceed mean sampling "
            f"noise ({mean_sampling_var:.6g}), so empirical pooling collapses "
            f"every member to the global mean ({global_mean:.6g}); shrunk "
            f"ranks tie and fall back to alphabetical order "
            f"(rows/episodes={len(values)}, member_groups={len(means)}, "
            f"independent_date_member_groups="
            f"{_independent_group_count(labels, independent)}).",
            UserWarning,
            stacklevel=2,
        )
    elif independent is None:
        # rank_members currently supplies member labels only, so this runtime
        # report makes its missing independent date/member unit explicit.
        warnings.warn(
            "Member posterior report: rows/episodes="
            f"{len(values)}, member_groups={len(means)}, "
            "independent_date_member_groups=UNKNOWN (not supplied).",
            UserWarning,
            stacklevel=2,
        )
    between_var = max(moment_between_var, variance_floor)

    denominator = counts * between_var + within_var
    shrinkage = within_var / denominator
    posterior_mean = (1.0 - shrinkage) * means + shrinkage * global_mean
    posterior_var = shrinkage * between_var

    result = pd.DataFrame(
        {
            "posterior_mean": posterior_mean,
            "posterior_std": np.sqrt(posterior_var),
            "shrinkage": shrinkage,
            "global_mean": global_mean,
            "within_var": within_var,
            "between_var": between_var,
        },
        index=means.index,
    )
    result["episode_count"] = counts.astype(int)
    result["member_group_count"] = len(means)
    result["independent_group_count"] = (
        _independent_groups_per_member(labels, independent).reindex(means.index)
        if independent is not None
        else None
    )
    return result


def _independent_groups_per_member(labels, independent):
    frame = pd.DataFrame({"member": labels, "independent": independent})
    return frame.groupby("member", sort=False)["independent"].nunique()


def _independent_group_count(labels, independent):
    if independent is None:
        return "UNKNOWN (not supplied)"
    return int(_independent_groups_per_member(labels, independent).sum())
