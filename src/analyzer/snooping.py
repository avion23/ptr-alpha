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
    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha}")
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
    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha}")

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
class MaxStatPermutationResult:
    """Dependence-preserving one-sided max-stat permutation output."""

    adjusted_p_values: np.ndarray
    observed_statistics: np.ndarray
    null_max_statistics: np.ndarray
    n_permutations: int
    block_days: int
    seed: int


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


def max_stat_block_permutation(
    series_by_trial: dict[int, pd.Series],
    lags_by_trial: dict[int, int],
    *,
    n_permutations: int,
    block_days: int,
    seed: int,
) -> MaxStatPermutationResult:
    """Compute one-sided max-stat p-values with synchronized calendar blocks.

    One random sign is applied to every observation in a calendar block and to
    every configuration that has an observation in that block. This preserves
    serial dependence inside blocks and cross-configuration dependence. The
    returned p-values include the standard +1 finite-permutation correction.
    """
    if not series_by_trial:
        raise ValueError("series_by_trial must not be empty")
    if n_permutations < 1:
        raise ValueError("n_permutations must be positive")
    if block_days < 1:
        raise ValueError("block_days must be positive")

    trial_ids = sorted(series_by_trial)
    normalized: dict[int, pd.Series] = {}
    all_dates: list[pd.Timestamp] = []
    for trial_id in trial_ids:
        series = pd.Series(series_by_trial[trial_id], dtype=float).dropna().sort_index()
        series.index = pd.DatetimeIndex(series.index)
        normalized[trial_id] = series
        all_dates.extend(pd.Timestamp(value) for value in series.index)
    if not all_dates:
        raise ValueError("permutation series contain no observations")

    origin = min(all_dates).normalize()
    block_ids_by_trial = {
        trial_id: np.asarray(
            [
                int((pd.Timestamp(value).normalize() - origin).days // block_days)
                for value in series.index
            ],
            dtype=int,
        )
        for trial_id, series in normalized.items()
    }
    block_ids = sorted(
        {int(block) for blocks in block_ids_by_trial.values() for block in blocks}
    )
    block_position = {block: position for position, block in enumerate(block_ids)}

    observed = np.asarray(
        [
            _hac_tstat(normalized[trial_id].to_numpy(), lags_by_trial.get(trial_id, 0))
            for trial_id in trial_ids
        ],
        dtype=float,
    )
    rng = np.random.default_rng(seed)
    null_max = np.empty(n_permutations, dtype=float)
    for permutation in range(n_permutations):
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=len(block_ids))
        maximum = -math.inf
        for trial_id in trial_ids:
            trial_signs = np.asarray(
                [
                    signs[block_position[int(block)]]
                    for block in block_ids_by_trial[trial_id]
                ],
                dtype=float,
            )
            statistic = _hac_tstat(
                normalized[trial_id].to_numpy() * trial_signs,
                lags_by_trial.get(trial_id, 0),
            )
            maximum = max(maximum, statistic)
        null_max[permutation] = maximum

    adjusted = np.asarray(
        [
            (1.0 + float(np.sum(null_max >= statistic))) / (n_permutations + 1.0)
            for statistic in observed
        ],
        dtype=float,
    )
    return MaxStatPermutationResult(
        adjusted_p_values=adjusted,
        observed_statistics=observed,
        null_max_statistics=null_max,
        n_permutations=n_permutations,
        block_days=block_days,
        seed=seed,
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
    inference_method: str = "return_series_bonferroni_max_stat"

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
    per_date_returns: pd.Series | dict[int, pd.Series] | None = None,
    lags_by_trial: dict[int, int] | None = None,
    n_permutations: int = 999,
    block_days: int = 90,
    seed: int = 0,
) -> SnoopingReport:
    """Analyze one requested configuration from its actual return series.

    The former implementation inferred a within-strategy standard error from
    cross-configuration summary rows. That inference is invalid and is now
    refused. Callers must supply per-date net-alpha returns. Bonferroni controls
    family-wise error under arbitrary dependence; synchronized block max-stat
    permutations provide a second dependence-aware gate.
    """
    if per_date_returns is None:
        raise ValueError("per_date_returns is required for coherent snooping inference")
    if sweep_results.empty:
        raise ValueError("sweep_results must not be empty")

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

    if isinstance(per_date_returns, dict):
        series_by_trial = {
            int(key): pd.Series(value, dtype=float)
            for key, value in per_date_returns.items()
        }
        selected_series = series_by_trial.get(row_position)
        if selected_series is None:
            raise ValueError("per_date_returns has no series for the requested config")
    else:
        selected_series = pd.Series(per_date_returns, dtype=float)
        series_by_trial = {row_position: selected_series}

    trials = int(n_tests if n_tests is not None else len(sweep_results))
    if trials < len(sweep_results):
        raise ValueError("n_tests cannot be smaller than the sweep result count")
    lags = dict(lags_by_trial or {})
    lag = int(lags.get(row_position, 0))
    statistic = _hac_tstat(selected_series.dropna().to_numpy(), lag)
    raw_p = (
        float(stats.norm.sf(statistic))
        if math.isfinite(statistic)
        else (0.0 if statistic > 0 else 1.0)
    )
    bonferroni_threshold = bonferroni_correction(trials, alpha)
    significant_bonferroni = raw_p <= bonferroni_threshold

    max_stat = max_stat_block_permutation(
        series_by_trial,
        {trial_id: int(lags.get(trial_id, 0)) for trial_id in series_by_trial},
        n_permutations=n_permutations,
        block_days=block_days,
        seed=seed,
    )
    trial_ids = sorted(series_by_trial)
    selected_permutation_position = trial_ids.index(row_position)
    max_stat_p = float(max_stat.adjusted_p_values[selected_permutation_position])
    release_ready = n_permutations >= 999
    deployable = bool(
        release_ready
        and float(selected_series.mean()) > 0
        and significant_bonferroni
        and max_stat_p <= alpha
    )

    clean = selected_series.dropna().to_numpy(dtype=float)
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
        p_value_raw=raw_p,
        bonferroni_threshold=bonferroni_threshold,
        p_value_bonferroni=min(raw_p * trials, 1.0),
        significant_bonferroni=significant_bonferroni,
        bh_rejected=False,
        bh_adjusted_alpha=bonferroni_threshold,
        dsr=dsr,
        significant_dsr=dsr > 0.95,
        min_years=min_backtest_length(observed_sharpe, alpha=alpha),
        max_stat_p_value=max_stat_p,
        deployable=deployable,
    )
