"""Data snooping corrections for multiple hypothesis testing.

Corrects for the multiple comparisons problem when running parameter sweeps.
With 648 tested combinations, some will show spurious alpha by chance.

References:
- Bonferroni: Bonferroni (1936)
- Benjamini-Hochberg: Benjamini & Hochberg (1995)
- Deflated Sharpe Ratio: Bailey & López de Prado (2014)
- Minimum Backtest Length: Bailey & López de Prado (2012)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


def bonferroni_correction(n_tests: int, alpha: float = 0.05) -> float:
    """Compute the Bonferroni-corrected significance threshold.

    The Bonferroni correction controls the family-wise error rate (FWER)
    by dividing the significance level by the number of tests.

    Args:
        n_tests: Number of hypotheses tested.
        alpha: Desired family-wise significance level (default 0.05).

    Returns:
        Adjusted p-value threshold. A result is significant if
        p < threshold.

    Raises:
        ValueError: If n_tests <= 0 or alpha <= 0.
    """
    if n_tests <= 0:
        raise ValueError(f"n_tests must be positive, got {n_tests}")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be between zero and one, got {alpha}")
    return alpha / n_tests


def benjamini_hochberg(
    p_values: list[float] | np.ndarray, alpha: float = 0.05
) -> np.ndarray:
    """Apply the Benjamini-Hochberg procedure to control the false discovery rate.

    Controls FDR at level `alpha` rather than FWER, making it more powerful
    than Bonferroni when many tests are conducted.

    Args:
        p_values: Array of p-values from the hypothesis tests.
        alpha: Desired false discovery rate (default 0.05).

    Returns:
        Boolean array indicating which hypotheses are rejected.

    Raises:
        ValueError: If p_values is empty or alpha <= 0.
    """
    p = np.asarray(p_values, dtype=float)
    if p.size == 0:
        raise ValueError("p_values must not be empty")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be between zero and one, got {alpha}")

    n = len(p)
    # Sort p-values and track original indices
    sorted_indices = np.argsort(p)
    sorted_p = p[sorted_indices]

    # BH critical values: (rank / n) * alpha
    ranks = np.arange(1, n + 1)
    critical_values = ranks / n * alpha

    # Find the largest rank where p <= critical value
    below = sorted_p <= critical_values
    if not below.any():
        return np.zeros(n, dtype=bool)

    max_rank = np.max(np.where(below)[0]) + 1  # 1-indexed

    # Reject all hypotheses with rank <= max_rank
    rejected = np.zeros(n, dtype=bool)
    rejected[sorted_indices[:max_rank]] = True
    return rejected


@dataclass(frozen=True, slots=True)
class MaxStatBootstrapResult:
    """Centered moving-block bootstrap output for a family of strategies."""

    marginal_p_values: np.ndarray
    adjusted_p_values: np.ndarray
    observed_statistics: np.ndarray
    null_statistics: np.ndarray
    null_max_statistics: np.ndarray
    n_bootstrap: int
    seed: int
    assumptions: tuple[str, ...]


def _hac_tstat(values: np.ndarray, lag: int) -> float:
    """HAC t-statistic used internally to avoid a validation import cycle."""
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


def max_stat_moving_block_bootstrap(
    series_by_trial: dict[int, pd.Series],
    lags_by_trial: dict[int, int],
    block_lengths_by_trial: dict[int, int],
    *,
    n_bootstrap: int,
    seed: int,
) -> MaxStatBootstrapResult:
    """Centered one-sided moving-block bootstrap with a family max statistic.

    Each trial is centered under its zero-mean null. Shared block-start uniforms
    are used across trials, preserving local serial order and cross-trial
    dependence for aligned schedules. Configurations with different frequencies
    use the same uniforms mapped to their own ordinal schedule. Bonferroni remains
    the arbitrary-dependence gate; max-stat is an additional empirical gate.

    The bootstrap is refused unless each series contains at least two complete
    blocks. This prevents an asymptotic p-value from replacing inadequate null
    support.
    """
    if not series_by_trial:
        raise ValueError("series_by_trial must not be empty")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    trial_ids = sorted(series_by_trial)
    expected = set(trial_ids)
    if set(lags_by_trial) != expected or set(block_lengths_by_trial) != expected:
        raise ValueError(
            "lags and block lengths must match every actual trial_id exactly"
        )

    centered: dict[int, np.ndarray] = {}
    observed = np.empty(len(trial_ids), dtype=float)
    blocks_needed = 0
    for position, trial_id in enumerate(trial_ids):
        values = pd.Series(series_by_trial[trial_id], dtype=float).dropna().to_numpy()
        block_length = int(block_lengths_by_trial[trial_id])
        if block_length < 1:
            raise ValueError("block lengths must be positive")
        if len(values) < 2 * block_length:
            raise ValueError(
                f"trial {trial_id} has {len(values)} observations; "
                f"at least {2 * block_length} are required for two blocks"
            )
        centered[trial_id] = values - float(values.mean())
        observed[position] = _hac_tstat(values, lags_by_trial[trial_id])
        blocks_needed = max(blocks_needed, math.ceil(len(values) / block_length))

    rng = np.random.default_rng(seed)
    shared_uniforms = rng.random((n_bootstrap, blocks_needed))
    null_statistics = np.empty((n_bootstrap, len(trial_ids)), dtype=float)
    for bootstrap_index in range(n_bootstrap):
        for position, trial_id in enumerate(trial_ids):
            values = centered[trial_id]
            block_length = int(block_lengths_by_trial[trial_id])
            n_starts = len(values) - block_length + 1
            n_blocks = math.ceil(len(values) / block_length)
            samples = []
            for uniform in shared_uniforms[bootstrap_index, :n_blocks]:
                start = min(int(uniform * n_starts), n_starts - 1)
                samples.append(values[start : start + block_length])
            bootstrap_values = np.concatenate(samples)[: len(values)]
            null_statistics[bootstrap_index, position] = _hac_tstat(
                bootstrap_values, lags_by_trial[trial_id]
            )
    null_max = np.max(null_statistics, axis=1)
    marginal = np.asarray(
        [
            (1.0 + float(np.sum(null_statistics[:, position] >= statistic)))
            / (n_bootstrap + 1.0)
            for position, statistic in enumerate(observed)
        ]
    )
    adjusted = np.asarray(
        [
            (1.0 + float(np.sum(null_max >= statistic))) / (n_bootstrap + 1.0)
            for statistic in observed
        ]
    )
    return MaxStatBootstrapResult(
        marginal_p_values=marginal,
        adjusted_p_values=adjusted,
        observed_statistics=observed,
        null_statistics=null_statistics,
        null_max_statistics=null_max,
        n_bootstrap=n_bootstrap,
        seed=seed,
        assumptions=(
            "per-date net-alpha series is locally stationary within moving blocks",
            "ordinal schedules with different frequencies share dependence through block-start uniforms",
            "Bonferroni is the controlling arbitrary-dependence gate",
        ),
    )


def deflated_sharpe_ratio(
    observed_sharpe: float,
    n_trials: int,
    n_observations: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Compute the Deflated Sharpe Ratio (DSR).

    Adjusts the observed Sharpe ratio for the number of strategies tried,
    the non-normality of returns, and the estimation error.

    Based on Bailey & López de Prado (2014), "The Deflated Sharpe Ratio".

    Args:
        observed_sharpe: The Sharpe ratio of the best strategy found.
        n_trials: Number of strategies/parameter combinations tested.
        n_observations: Number of independent observations used to compute
            the Sharpe ratio.
        skew: Skewness of the returns distribution.
        kurtosis: Total kurtosis of the returns distribution (default 3.0
            for normal distribution; internally converted to excess kurtosis).

    Returns:
        The Deflated Sharpe Ratio (probability that the Sharpe ratio is
        not due to chance). Values close to 1.0 indicate genuine alpha.
    """
    if n_trials <= 0:
        raise ValueError(f"n_trials must be positive, got {n_trials}")
    if n_observations <= 0:
        raise ValueError(f"n_observations must be positive, got {n_observations}")

    # Expected max Sharpe under null (no skill), assuming iid normal returns.
    # E[max(SR)] ≈ sqrt(2 * ln(n)) for large n, with corrections for
    # finite samples and non-normality.
    e_max_sr = _expected_max_sharpe(n_trials, n_observations, skew, kurtosis)

    # Standard error of the Sharpe ratio under the null.
    # The expression under the sqrt can go negative with extreme skew;
    # clamp to zero in that case (the SE floor is then 0).
    se_sq = (
        1 - skew * observed_sharpe + (kurtosis - 1) / 4 * observed_sharpe**2
    ) / n_observations
    se = math.sqrt(max(se_sq, 0.0))

    if se == 0:
        return 1.0 if observed_sharpe >= e_max_sr else 0.0

    # DSR = probability that a standard normal exceeds the deflated z-score.
    z = (observed_sharpe - e_max_sr) / se
    return float(stats.norm.cdf(z))


def _expected_max_sharpe(
    n_trials: int,
    n_observations: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Expected maximum Sharpe ratio under the null hypothesis.

    Uses the approximation from Bailey & López de Prado (2014):
        E[max(SR)] ≈ sqrt(2 * ln(n)) * (1 - γ/(2*ln(n)))
                      + skew/(6*sqrt(n_obs)) * ... (higher-order terms)

    where γ ≈ 0.5772 is the Euler-Mascheroni constant.
    """
    if n_trials == 1:
        return 0.0
    gamma = 0.5772156649015329  # Euler-Mascheroni constant

    # Base term: expected max of n iid standard normals
    base = math.sqrt(2 * math.log(n_trials)) - (
        gamma / (2 * math.sqrt(2 * math.log(n_trials)))
    )

    # Adjust for non-normality (skew and excess kurtosis)
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
    """Compute the minimum number of years of data needed for significance.

    Given an observed Sharpe ratio, how many years of data do we need
    for the result to be statistically significant at level `alpha`?

    Uses the formula: T >= (z_alpha * sigma / mu)^2
    where mu = sharpe * sigma / sqrt(T), solving for T.

    More precisely, for a t-test with T observations:
        T >= (t_alpha * sigma / mu)^2

    Args:
        sharpe: Observed annualized Sharpe ratio.
        sigma: Standard deviation of annual returns (default 1.0).
        alpha: Desired significance level (default 0.05).

    Returns:
        Minimum number of years of data required.
    """
    if sharpe <= 0:
        return float("inf")  # Cannot achieve significance with non-positive Sharpe

    # z_alpha for one-sided test (we want to show positive alpha)
    z_alpha = stats.norm.ppf(1 - alpha)

    # For a Sharpe ratio: SR = mu / sigma * sqrt(T)
    # mu = SR * sigma / sqrt(T)
    # t-stat = mu / (sigma / sqrt(T)) = SR * sqrt(T)
    # We need SR * sqrt(T) >= z_alpha
    # Therefore: T >= (z_alpha / SR)^2
    t_min = (z_alpha / sharpe) ** 2

    # Convert from observations to years (assuming monthly observations
    # as a reasonable default for financial data; 12 obs/year).
    # But actually, we should express this in terms of the observation
    # frequency. Since we don't know, we return observations and let
    # the caller convert. However, the task asks for years, so we
    # assume the Sharpe is annualized and the formula gives years directly.
    return t_min


def alpha_ttest(
    mean_alpha: float,
    std_alpha: float,
    n_observations: int,
) -> tuple[float, float]:
    """Compute the t-statistic and p-value for the mean alpha.

    Args:
        mean_alpha: Sample mean of the alpha (e.g., alpha_slope or overall_alpha).
        std_alpha: Sample standard deviation of the alpha.
        n_observations: Number of independent observations.

    Returns:
        Tuple of (t-statistic, two-sided p-value).
    """
    if n_observations <= 1:
        return 0.0, 1.0
    se = std_alpha / math.sqrt(n_observations)
    if se == 0:
        return (
            float("inf") if mean_alpha > 0 else float("-inf") if mean_alpha < 0 else 0.0
        ), 1.0
    t_stat = mean_alpha / se
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n_observations - 1))
    return float(t_stat), float(p_value)


@dataclass(frozen=True, slots=True)
class SnoopingReport:
    """Results of data snooping analysis for a single strategy configuration."""

    n_tests: int
    alpha_slope: float
    overall_alpha: float
    sharpe: float
    n_observations: int
    dates_evaluated: int

    # T-test results
    t_statistic: float
    p_value_raw: float

    # Corrections
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
    inference_method: str = "centered_moving_block_bootstrap_bonferroni_max_stat"

    @property
    def significant_bonferroni_any(self) -> bool:
        """Is the result still significant after Bonferroni correction?"""
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
    """Analyze one requested configuration from a complete family of returns.

    Every family row must carry a unique ``trial_id`` and callers must provide
    exactly one return series, lag, and block length for each actual ID. Summary
    rows, lone series, positional guesses, missing IDs, and extra IDs are
    refused. All rewarded p-values come from the centered moving-block bootstrap.
    """
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
    release_ready = n_permutations >= 999
    deployable = bool(
        release_ready
        and float(selected_series.mean()) > 0
        and significant_bonferroni
        and max_stat_p <= alpha
    )

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
        deployable=deployable,
    )
