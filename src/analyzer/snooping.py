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


def benjamini_hochberg(p_values: list[float] | np.ndarray, alpha: float = 0.05) -> np.ndarray:
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
        (1 - skew * observed_sharpe + (kurtosis - 1) / 4 * observed_sharpe**2)
        / n_observations
    )
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
    gamma = 0.5772156649015329  # Euler-Mascheroni constant

    # Base term: expected max of n iid standard normals
    base = math.sqrt(2 * math.log(n_trials)) - (
        gamma / (2 * math.sqrt(2 * math.log(n_trials)))
    )

    # Adjust for non-normality (skew and excess kurtosis)
    excess_kurt = kurtosis - 3.0
    adjustment = (
        skew / (6 * math.sqrt(n_observations))
        + excess_kurt / (24 * n_observations)
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
        return (float("inf") if mean_alpha > 0 else float("-inf") if mean_alpha < 0 else 0.0), 1.0
    t_stat = mean_alpha / se
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n_observations - 1))
    return float(t_stat), float(p_value)


@dataclass
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

    @property
    def significant_bonferroni_any(self) -> bool:
        """Is the result still significant after Bonferroni correction?"""
        return self.p_value_raw < self.bonferroni_threshold


def analyze_snooping(
    sweep_results: pd.DataFrame,
    best_config: dict | None = None,
    n_tests: int = 648,
    alpha: float = 0.05,
) -> SnoopingReport:
    """Run the full snooping analysis on sweep results.

    Args:
        sweep_results: DataFrame from the parameter sweep.
        best_config: Dict of parameters identifying the best config.
            If None, uses the config with highest alpha_slope.
        n_tests: Total number of hypotheses tested (default 648).
        alpha: Significance level (default 0.05).

    Returns:
        SnoopingReport with all corrections applied.
    """
    if best_config is not None:
        mask = pd.Series(True, index=sweep_results.index)
        for k, v in best_config.items():
            mask &= sweep_results[k] == v
        best_row = sweep_results[mask]
        if best_row.empty:
            raise ValueError(f"No matching config found for {best_config}")
        row = best_row.iloc[0]
    else:
        row = sweep_results.loc[sweep_results["alpha_slope"].idxmax()]

    alpha_slope = float(row["alpha_slope"])
    overall_alpha = float(row["overall_alpha"])
    sharpe = float(row["sharpe"])
    dates_evaluated = int(row.get("dates_evaluated", 0))

    # --- T-test on alpha_slope ---
    # Use the sweep-wide std of alpha_slope as a proxy for within-strategy std
    all_std = float(sweep_results["alpha_slope"].std())
    all_mean = float(sweep_results["alpha_slope"].mean())
    n_obs = max(dates_evaluated, 2)

    # Estimate within-strategy std: use the sweep std scaled by sqrt(N)
    # This is a conservative estimate; ideally we'd have per-period alphas.
    t_stat, p_raw = alpha_ttest(alpha_slope, all_std, n_obs)

    # --- Bonferroni ---
    bonf_thresh = bonferroni_correction(n_tests, alpha)
    sig_bonf = p_raw < bonf_thresh

    # --- Benjamini-Hochberg ---
    # Convert all configs' alpha_slope to pseudo-p-values using the sweep distribution
    all_slopes = sweep_results["alpha_slope"].values
    all_t_stats = (all_slopes - all_mean) / (all_std / math.sqrt(n_obs))
    all_p_values = 2 * (1 - stats.t.cdf(np.abs(all_t_stats), df=n_obs - 1))
    bh_rejected = benjamini_hochberg(all_p_values, alpha)

    # Find our config's position
    best_idx = int(sweep_results["alpha_slope"].idxmax())
    bh_our_rejected = bool(bh_rejected[best_idx])

    # --- Deflated Sharpe Ratio ---
    skew = float(sweep_results["sharpe"].skew()) if len(sweep_results) > 2 else 0.0
    kurt = float(sweep_results["sharpe"].kurtosis()) + 3.0 if len(sweep_results) > 3 else 3.0
    dsr = deflated_sharpe_ratio(
        observed_sharpe=sharpe,
        n_trials=n_tests,
        n_observations=n_obs,
        skew=skew,
        kurtosis=kurt,
    )
    sig_dsr = dsr > 0.95  # 95% confidence threshold

    # --- Minimum backtest length ---
    min_yrs = min_backtest_length(sharpe, alpha=alpha)

    return SnoopingReport(
        n_tests=n_tests,
        alpha_slope=alpha_slope,
        overall_alpha=overall_alpha,
        sharpe=sharpe,
        n_observations=n_obs,
        dates_evaluated=dates_evaluated,
        t_statistic=t_stat,
        p_value_raw=p_raw,
        bonferroni_threshold=bonf_thresh,
        p_value_bonferroni=min(p_raw * n_tests, 1.0),
        significant_bonferroni=sig_bonf,
        bh_rejected=bh_our_rejected,
        bh_adjusted_alpha=alpha,
        dsr=dsr,
        significant_dsr=sig_dsr,
        min_years=min_yrs,
    )
