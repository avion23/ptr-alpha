"""Dependence-aware corrections for strategy search and backtest selection.

The production gate is deliberately conservative:

* marginal bootstrap p-values are Bonferroni-corrected over the full family;
* max-stat resampling is synchronized only for trials with identical calendar
  support; and
* support groups are combined with another Bonferroni bound rather than by
  pretending that ordinal observations from different calendars are aligned.

This module contains statistical primitives only. It does not authorize a
strategy for deployment; the validation ledger owns that decision.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


def bonferroni_correction(n_tests: int, alpha: float = 0.05) -> float:
    """Return the per-test threshold controlling family-wise error."""
    if n_tests <= 0:
        raise ValueError(f"n_tests must be positive, got {n_tests}")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be between zero and one, got {alpha}")
    return alpha / n_tests


def benjamini_hochberg(
    p_values: list[float] | np.ndarray, alpha: float = 0.05
) -> np.ndarray:
    """Apply the Benjamini-Hochberg false-discovery-rate procedure."""
    p = np.asarray(p_values, dtype=float)
    if p.size == 0:
        raise ValueError("p_values must not be empty")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be between zero and one, got {alpha}")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("p_values must be finite probabilities")

    sorted_indices = np.argsort(p)
    sorted_p = p[sorted_indices]
    critical_values = np.arange(1, len(p) + 1) / len(p) * alpha
    below = sorted_p <= critical_values
    rejected = np.zeros(len(p), dtype=bool)
    if below.any():
        last = int(np.flatnonzero(below)[-1])
        rejected[sorted_indices[: last + 1]] = True
    return rejected


@dataclass(frozen=True, slots=True)
class MaxStatBootstrapResult:
    """Centered circular-block bootstrap output for one strategy family."""

    marginal_p_values: np.ndarray
    adjusted_p_values: np.ndarray
    observed_statistics: np.ndarray
    null_statistics: np.ndarray
    null_max_statistics: np.ndarray
    n_bootstrap: int
    seed: int
    assumptions: tuple[str, ...]
    support_group_count: int = 1


def _hac_tstat(values: np.ndarray, lag: int) -> float:
    """Bartlett-kernel HAC t-statistic for a sample mean."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        return 0.0
    lag = max(0, min(int(lag), n - 1))
    mean = float(x.mean())
    demeaned = x - mean
    gamma = np.array(
        [np.dot(demeaned[k:], demeaned[: n - k]) / n for k in range(lag + 1)]
    )
    if lag == 0:
        long_run_variance = float(gamma[0])
    else:
        weights = 1.0 - np.arange(1, lag + 1) / (lag + 1)
        long_run_variance = float(gamma[0] + 2.0 * np.dot(weights, gamma[1:]))
    standard_error = math.sqrt(max(long_run_variance, 0.0) / n)
    if standard_error < 1e-14:
        if mean > 0:
            return math.inf
        if mean < 0:
            return -math.inf
        return 0.0
    return float(mean / standard_error)


def _normalize_series(value: pd.Series, trial_id: int) -> pd.Series:
    series = pd.Series(value, dtype=float).dropna().sort_index()
    if series.empty:
        raise ValueError(f"trial {trial_id} has no finite observations")
    if not series.index.is_unique:
        raise ValueError(f"trial {trial_id} calendar index must be unique")
    if not np.isfinite(series.to_numpy(dtype=float)).all():
        raise ValueError(f"trial {trial_id} contains non-finite observations")
    return series


def _calendar_support_groups(
    series_by_trial: dict[int, pd.Series],
) -> tuple[dict[int, pd.Series], list[list[int]]]:
    """Partition trials by exact post-NaN calendar support."""
    normalized = {
        int(trial_id): _normalize_series(value, int(trial_id))
        for trial_id, value in series_by_trial.items()
    }
    groups: list[list[int]] = []
    for trial_id in sorted(normalized):
        for group in groups:
            if normalized[trial_id].index.equals(normalized[group[0]].index):
                group.append(trial_id)
                break
        else:
            groups.append([trial_id])
    return normalized, groups


def _circular_block_sample(
    centered: np.ndarray,
    starts: np.ndarray,
    block_length: int,
) -> np.ndarray:
    """Sample one fixed-length circular moving-block path."""
    n = len(centered)
    n_blocks = math.ceil(n / block_length)
    offsets = np.arange(block_length)
    blocks = [centered[(int(start) + offsets) % n] for start in starts[:n_blocks]]
    return np.concatenate(blocks)[:n]


def max_stat_moving_block_bootstrap(
    series_by_trial: dict[int, pd.Series],
    lags_by_trial: dict[int, int],
    block_lengths_by_trial: dict[int, int],
    *,
    n_bootstrap: int,
    seed: int,
) -> MaxStatBootstrapResult:
    """Run a centered, calendar-aware, one-sided family bootstrap.

    Trials sharing exactly the same calendar index are resampled jointly with
    common circular block starts. This preserves observed contemporaneous and
    serial dependence within that support group. Trials on different calendars
    are never paired by ordinal row number; their groupwise max-stat p-values
    are combined with a Bonferroni bound across support groups.

    Each trial must contain at least two complete blocks. Marginal p-values are
    returned for the separate full-family Bonferroni gate used by validation.
    """
    if not series_by_trial:
        raise ValueError("series_by_trial must not be empty")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")

    trial_ids = sorted(int(value) for value in series_by_trial)
    expected = set(trial_ids)
    if set(lags_by_trial) != expected or set(block_lengths_by_trial) != expected:
        raise ValueError(
            "lags and block lengths must match every actual trial_id exactly"
        )

    normalized, support_groups = _calendar_support_groups(series_by_trial)
    position_by_trial = {
        trial_id: position for position, trial_id in enumerate(trial_ids)
    }
    observed = np.empty(len(trial_ids), dtype=float)
    null_statistics = np.empty((n_bootstrap, len(trial_ids)), dtype=float)
    marginal = np.empty(len(trial_ids), dtype=float)
    group_adjusted = np.empty(len(trial_ids), dtype=float)
    group_null_maxima: list[np.ndarray] = []
    rng = np.random.default_rng(seed)

    for group in support_groups:
        n = len(normalized[group[0]])
        max_blocks = 0
        centered: dict[int, np.ndarray] = {}
        for trial_id in group:
            block_length = int(block_lengths_by_trial[trial_id])
            if block_length < 1:
                raise ValueError("block lengths must be positive")
            if n < 2 * block_length:
                raise ValueError(
                    f"trial {trial_id} has {n} observations; "
                    f"at least {2 * block_length} are required for two blocks"
                )
            values = normalized[trial_id].to_numpy(dtype=float)
            centered[trial_id] = values - float(values.mean())
            observed[position_by_trial[trial_id]] = _hac_tstat(
                values, int(lags_by_trial[trial_id])
            )
            max_blocks = max(max_blocks, math.ceil(n / block_length))

        # Common integer starts mean the same calendar date starts every block
        # for every trial in this support group. Circular blocks avoid an edge
        # rule that would otherwise differ when block lengths differ.
        starts = rng.integers(0, n, size=(n_bootstrap, max_blocks))
        group_positions = [position_by_trial[trial_id] for trial_id in group]
        for bootstrap_index in range(n_bootstrap):
            for trial_id in group:
                position = position_by_trial[trial_id]
                sampled = _circular_block_sample(
                    centered[trial_id],
                    starts[bootstrap_index],
                    int(block_lengths_by_trial[trial_id]),
                )
                null_statistics[bootstrap_index, position] = _hac_tstat(
                    sampled, int(lags_by_trial[trial_id])
                )

        group_null = np.max(null_statistics[:, group_positions], axis=1)
        group_null_maxima.append(group_null)
        for trial_id in group:
            position = position_by_trial[trial_id]
            statistic = observed[position]
            marginal[position] = (
                1.0
                + float(np.sum(null_statistics[:, position] >= statistic))
            ) / (n_bootstrap + 1.0)
            group_adjusted[position] = (
                1.0 + float(np.sum(group_null >= statistic))
            ) / (n_bootstrap + 1.0)

    support_group_count = len(support_groups)
    adjusted = np.minimum(group_adjusted * support_group_count, 1.0)
    # Diagnostic only. Formal cross-group control is the Bonferroni factor above;
    # no unobserved dependence is invented between different calendar supports.
    null_max = np.max(np.vstack(group_null_maxima), axis=0)
    return MaxStatBootstrapResult(
        marginal_p_values=marginal,
        adjusted_p_values=adjusted,
        observed_statistics=observed,
        null_statistics=null_statistics,
        null_max_statistics=null_max,
        n_bootstrap=n_bootstrap,
        seed=seed,
        assumptions=(
            "per-date net-alpha series is locally stationary within circular blocks",
            "block starts are synchronized only inside exact calendar-support groups",
            "support-group max-stat p-values use Bonferroni control across calendars",
            "the full-family marginal Bonferroni gate remains the "
            "arbitrary-dependence control",
        ),
        support_group_count=support_group_count,
    )


def deflated_sharpe_ratio(
    observed_sharpe: float,
    n_trials: int,
    n_observations: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Return the descriptive probability used by the legacy DSR report."""
    if n_trials <= 0:
        raise ValueError(f"n_trials must be positive, got {n_trials}")
    if n_observations <= 0:
        raise ValueError(f"n_observations must be positive, got {n_observations}")
    e_max_sr = _expected_max_sharpe(n_trials, n_observations, skew, kurtosis)
    se_sq = (
        1 - skew * observed_sharpe + (kurtosis - 1) / 4 * observed_sharpe**2
    ) / n_observations
    se = math.sqrt(max(se_sq, 0.0))
    if se == 0:
        return 1.0 if observed_sharpe >= e_max_sr else 0.0
    return float(stats.norm.cdf((observed_sharpe - e_max_sr) / se))


def _expected_max_sharpe(
    n_trials: int,
    n_observations: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    if n_trials == 1:
        return 0.0
    gamma = 0.5772156649015329
    base = math.sqrt(2 * math.log(n_trials)) - (
        gamma / (2 * math.sqrt(2 * math.log(n_trials)))
    )
    excess_kurt = kurtosis - 3.0
    adjustment = skew / (6 * math.sqrt(n_observations)) + excess_kurt / (
        24 * n_observations
    )
    return base + adjustment


def min_backtest_length(
    sharpe: float,
    sigma: float = 1.0,
    alpha: float = 0.05,
) -> float:
    """Return the minimum number of annualized Sharpe periods for a z-test."""
    del sigma  # retained in the public signature for compatibility
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")
    if sharpe <= 0:
        return float("inf")
    z_alpha = stats.norm.ppf(1 - alpha)
    return float((z_alpha / sharpe) ** 2)


def alpha_ttest(
    mean_alpha: float,
    std_alpha: float,
    n_observations: int,
) -> tuple[float, float]:
    """Return a conventional two-sided t-test for a sample mean."""
    if n_observations <= 1:
        return 0.0, 1.0
    se = std_alpha / math.sqrt(n_observations)
    if se == 0:
        statistic = (
            math.inf if mean_alpha > 0 else -math.inf if mean_alpha < 0 else 0.0
        )
        return statistic, 1.0
    statistic = mean_alpha / se
    p_value = 2 * (1 - stats.t.cdf(abs(statistic), df=n_observations - 1))
    return float(statistic), float(p_value)


@dataclass(frozen=True, slots=True)
class SnoopingReport:
    """Dependence-aware diagnostics for one selected configuration."""

    n_tests: int
    alpha_slope: float
    overall_alpha: float
    sharpe: float
    n_observations: int
    dates_evaluated: int
    t_statistic: float
    p_value_raw: float
    bonferroni_threshold: float
    p_value_bonferroni: float
    significant_bonferroni: bool
    bh_rejected: bool
    bh_adjusted_alpha: float
    dsr: float
    significant_dsr: bool
    min_years: float
    max_stat_p_value: float = 1.0
    deployable: bool = False
    inference_method: str = (
        "calendar_grouped_circular_block_bootstrap_bonferroni_max_stat"
    )

    @property
    def significant_bonferroni_any(self) -> bool:
        return self.p_value_raw < self.bonferroni_threshold


def analyze_snooping(
    sweep_results: pd.DataFrame,
    best_config: dict | None = None,
    n_tests: int | None = None,
    alpha: float = 0.05,
    *,
    per_date_returns: dict[int, pd.Series] | None = None,
    lags_by_trial: dict[int, int] | None = None,
    block_lengths_by_trial: dict[int, int] | None = None,
    n_permutations: int = 999,
    seed: int = 0,
) -> SnoopingReport:
    """Analyze one requested configuration from a complete trial family."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")
    if sweep_results.empty:
        raise ValueError("sweep_results must not be empty")
    if "trial_id" not in sweep_results.columns:
        raise ValueError("sweep_results must contain actual trial_id values")
    trial_ids = [int(value) for value in sweep_results["trial_id"]]
    if len(set(trial_ids)) != len(trial_ids):
        raise ValueError("trial_id values must be unique")
    actual_ids = set(trial_ids)
    if not isinstance(per_date_returns, dict) or set(per_date_returns) != actual_ids:
        raise ValueError(
            "per_date_returns must be a complete dict keyed by actual trial_id"
        )
    if not isinstance(lags_by_trial, dict) or set(lags_by_trial) != actual_ids:
        raise ValueError("lags_by_trial must match every actual trial_id exactly")
    if (
        not isinstance(block_lengths_by_trial, dict)
        or set(block_lengths_by_trial) != actual_ids
    ):
        raise ValueError(
            "block_lengths_by_trial must match every actual trial_id exactly"
        )
    trials = len(sweep_results)
    if n_tests is not None and int(n_tests) != trials:
        raise ValueError("n_tests must equal the complete supplied family size")

    if best_config is None:
        row_position = int(
            np.argmax(sweep_results["overall_alpha"].to_numpy(dtype=float))
        )
    else:
        mask = pd.Series(True, index=sweep_results.index)
        for key, value in best_config.items():
            if key not in sweep_results.columns:
                raise ValueError(f"Unknown config key: {key}")
            mask &= sweep_results[key] == value
        positions = np.flatnonzero(mask.to_numpy())
        if len(positions) == 0:
            raise ValueError(f"No matching config found for {best_config}")
        row_position = int(positions[0])
    row = sweep_results.iloc[row_position]
    selected_trial_id = int(row["trial_id"])

    bootstrap = max_stat_moving_block_bootstrap(
        {
            int(key): pd.Series(value, dtype=float)
            for key, value in per_date_returns.items()
        },
        {int(key): int(value) for key, value in lags_by_trial.items()},
        {int(key): int(value) for key, value in block_lengths_by_trial.items()},
        n_bootstrap=n_permutations,
        seed=seed,
    )
    ordered_ids = sorted(actual_ids)
    selected_position = ordered_ids.index(selected_trial_id)
    statistic = float(bootstrap.observed_statistics[selected_position])
    bootstrap_p = float(bootstrap.marginal_p_values[selected_position])
    max_stat_p = float(bootstrap.adjusted_p_values[selected_position])
    bonferroni_threshold = bonferroni_correction(trials, alpha)
    significant_bonferroni = bootstrap_p <= bonferroni_threshold
    selected_series = pd.Series(
        per_date_returns[selected_trial_id], dtype=float
    ).dropna()

    clean = selected_series.to_numpy(dtype=float)
    observed_sharpe = float(row.get("sharpe", 0.0))
    skew = float(stats.skew(clean, bias=False)) if len(clean) > 2 else 0.0
    kurtosis = (
        float(stats.kurtosis(clean, fisher=False, bias=False))
        if len(clean) > 3
        else 3.0
    )
    dsr = deflated_sharpe_ratio(
        observed_sharpe=observed_sharpe,
        n_trials=trials,
        n_observations=max(len(clean), 1),
        skew=skew if math.isfinite(skew) else 0.0,
        kurtosis=kurtosis if math.isfinite(kurtosis) else 3.0,
    )
    return SnoopingReport(
        n_tests=trials,
        alpha_slope=float(row.get("alpha_slope", 0.0)),
        overall_alpha=float(selected_series.mean()) if len(clean) else 0.0,
        sharpe=observed_sharpe,
        n_observations=len(clean),
        dates_evaluated=len(clean),
        t_statistic=statistic,
        p_value_raw=bootstrap_p,
        bonferroni_threshold=bonferroni_threshold,
        p_value_bonferroni=min(bootstrap_p * trials, 1.0),
        significant_bonferroni=significant_bonferroni,
        bh_rejected=False,
        bh_adjusted_alpha=bonferroni_threshold,
        dsr=dsr,
        significant_dsr=dsr > 0.95,
        min_years=min_backtest_length(observed_sharpe, alpha=alpha),
        max_stat_p_value=max_stat_p,
        deployable=False,
    )
