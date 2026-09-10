"""Empirical normal-normal partial pooling for descriptive member research."""

from __future__ import annotations

import numpy as np
import pandas as pd


def normal_normal_posteriors(outcomes, groups) -> pd.DataFrame:
    """Fit an unweighted, scale-equivariant empirical normal-normal model."""
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
            ]
        )
    if not np.isfinite(values).all():
        raise ValueError("outcomes must be finite")
    if pd.isna(labels).any():
        raise ValueError("groups must be non-null")

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
    between_var = max(observed_between - mean_sampling_var, variance_floor)

    denominator = counts * between_var + within_var
    shrinkage = within_var / denominator
    posterior_mean = (1.0 - shrinkage) * means + shrinkage * global_mean
    posterior_var = shrinkage * between_var

    return pd.DataFrame(
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
