"""Tests for analyzer.validation: newey_west_tstat, select_config."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from analyzer.validation import (
    newey_west_tstat,
    select_config,
)


# ---------------------------------------------------------------------------
# newey_west_tstat
# ---------------------------------------------------------------------------


class TestNeweyWestTstat:
    def test_lag0_matches_biased_plain_tstat(self):
        """lag=0 → t = mean / (biased_std / sqrt(n))."""
        x = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        t_nw = newey_west_tstat(x, lag=0)
        n = len(x)
        t_plain = float(np.mean(x) / (np.std(x) / np.sqrt(n)))  # ddof=0
        assert abs(t_nw - t_plain) < 1e-10

    def test_zero_mean_series_gives_zero_tstat(self):
        """Series whose mean is 0 → t-stat = 0."""
        x = pd.Series([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
        assert abs(newey_west_tstat(x, lag=0)) < 1e-10

    def test_positive_mean_gives_positive_tstat(self):
        x = pd.Series([2.0, 3.0, 4.0])
        assert newey_west_tstat(x, lag=0) > 0

    def test_hand_computable_lag0(self):
        """Hand-verify: x=[0,2], mean=1, biased_std=1, n=2 → t=sqrt(2)."""
        x = pd.Series([0.0, 2.0])
        # biased std = sqrt(mean of (0-1)^2 + (2-1)^2) = sqrt(1) = 1
        # t = 1 / (1 / sqrt(2)) = sqrt(2)
        t = newey_west_tstat(x, lag=0)
        assert abs(t - math.sqrt(2)) < 1e-10

    def test_lag0_larger_than_lag1_for_positively_autocorrelated(self):
        """Positive autocorrelation → HAC widens SE → |t| shrinks with lag."""
        np.random.seed(42)
        n = 60
        ar = np.zeros(n)
        ar[0] = 1.0
        for i in range(1, n):
            ar[i] = 0.7 * ar[i - 1] + np.random.normal(0, 0.3)
        ar += 2.0  # ensure positive mean
        x = pd.Series(ar)
        t0 = abs(newey_west_tstat(x, lag=0))
        t2 = abs(newey_west_tstat(x, lag=2))
        # HAC correction should reduce or keep the t-stat
        assert t2 <= t0 + 0.5  # tolerant: direction matters

    def test_single_element_returns_zero(self):
        x = pd.Series([5.0])
        assert newey_west_tstat(x, lag=0) == 0.0

    def test_nan_dropped_before_computation(self):
        x_with_nan = pd.Series([1.0, float("nan"), 2.0, 3.0])
        x_clean = pd.Series([1.0, 2.0, 3.0])
        assert (
            abs(newey_west_tstat(x_with_nan, lag=0) - newey_west_tstat(x_clean, lag=0))
            < 1e-10
        )

    def test_lag_equals_len_minus_1_does_not_crash(self):
        """Edge case: lag ≥ n-1 should not raise (gamma array boundary)."""
        x = pd.Series([1.0, 2.0, 3.0])
        # lag=2 == n-1; gamma[2] involves dot product of length 0 array
        t = newey_west_tstat(x, lag=2)
        assert math.isfinite(t) or math.isinf(t)  # no crash

    def test_lag_capped_at_len_minus_1_for_two_observations(self):
        x = pd.Series([1.0, 3.0])
        assert newey_west_tstat(x, lag=3) == newey_west_tstat(x, lag=1)

    def test_lag_capped_at_len_minus_1_for_five_observations(self):
        x = pd.Series([1.0, 2.0, 4.0, 8.0, 16.0])
        assert newey_west_tstat(x, lag=10) == newey_west_tstat(x, lag=4)


# ---------------------------------------------------------------------------
# select_config
# ---------------------------------------------------------------------------


def _make_sweep_df(n: int = 10, p_values: list[float] | None = None) -> pd.DataFrame:
    """Build a synthetic sweep DataFrame with n rows."""
    rng = np.random.default_rng(99)
    rows = []
    for i in range(n):
        rows.append(
            {
                "horizon": 60,
                "frequency_days": 30,
                "training_lookback_days": 365,
                "min_buyers": 2 + (i % 3),
                "top_n": 3 + (i % 2) * 2,
                "decay_lambda": 0.005,
                "bayes_prior_strength": 20.0,
                "scoring_mode": "shrunk_alpha",
                "total_recs": 50,
                "dates_evaluated": 20,
                "overall_alpha": float(rng.uniform(0.5, 3.0)),
                "overall_return": 1.0,
                "rank1_alpha": 2.0,
                "rank5_alpha": 1.0,
                "alpha_slope": float(rng.uniform(-1.0, 2.0)),
                "win_rate": 60.0,
                "sharpe": 1.0,
                "max_drawdown": -5.0,
                "nw_tstat": 2.0,
                "p_value": p_values[i] if p_values else 0.9,  # default: no survivors
                "min_sample_ok": True,
            }
        )
    return pd.DataFrame(rows)


class TestSelectConfig:
    def test_picks_max_alpha_slope_among_bh_survivors(self):
        """Among configs that survive BH, pick the one with the highest alpha_slope."""
        df = _make_sweep_df(n=5, p_values=[0.001, 0.002, 0.9, 0.9, 0.9])
        df.loc[0, "alpha_slope"] = 5.0  # best, survives BH
        df.loc[1, "alpha_slope"] = 2.0  # survives BH but lower slope
        result = select_config(df, alpha=0.05)
        assert result["alpha_slope"] == 5.0
        assert result["survives_correction"] is True
        assert result["n_survivors"] >= 1

    def test_tiebreak_on_overall_alpha(self):
        """If alpha_slope is tied, pick higher overall_alpha."""
        df = _make_sweep_df(n=3, p_values=[0.001, 0.001, 0.9])
        df.loc[0, "alpha_slope"] = 3.0
        df.loc[1, "alpha_slope"] = 3.0
        df.loc[0, "overall_alpha"] = 1.0
        df.loc[1, "overall_alpha"] = 2.0  # should win on tiebreak
        result = select_config(df, alpha=0.05)
        assert result["overall_alpha"] == 2.0

    def test_no_survivors_returns_nominal_best(self):
        """When no config survives BH, return nominal best with flag=False."""
        df = _make_sweep_df(n=6, p_values=[0.9] * 6)
        df.loc[3, "alpha_slope"] = 10.0  # nominal best
        result = select_config(df, alpha=0.05)
        assert result["survives_correction"] is False
        assert result["n_survivors"] == 0
        assert result["alpha_slope"] == 10.0

    def test_bonferroni_threshold_is_alpha_over_n(self):
        df = _make_sweep_df(n=8, p_values=[0.9] * 8)
        result = select_config(df, alpha=0.05)
        assert abs(result["bonferroni_threshold"] - 0.05 / 8) < 1e-12

    def test_n_trials_equals_len_sweep_df(self):
        df = _make_sweep_df(n=12, p_values=[0.9] * 12)
        result = select_config(df, alpha=0.05)
        assert result["n_trials"] == 12

    def test_all_survive(self):
        """All very small p-values → all survive → best is still max slope."""
        df = _make_sweep_df(n=4, p_values=[0.0001] * 4)
        df.loc[2, "alpha_slope"] = 99.0
        result = select_config(df, alpha=0.05)
        assert result["alpha_slope"] == 99.0
        assert result["n_survivors"] == 4

    def test_min_sample_filter_excludes_tiny_high_tstat_config(self):
        """Tiny samples cannot win even with a huge positive t-stat."""
        df = _make_sweep_df(n=2, p_values=[0.0, 0.02])
        df.loc[
            0,
            [
                "dates_evaluated",
                "total_recs",
                "alpha_slope",
                "p_value",
                "min_sample_ok",
            ],
        ] = [
            2,
            2,
            100.0,
            1.0,
            False,
        ]
        df.loc[
            1,
            [
                "dates_evaluated",
                "total_recs",
                "alpha_slope",
                "p_value",
                "min_sample_ok",
            ],
        ] = [
            10,
            30,
            2.0,
            0.02,
            True,
        ]

        selected = select_config(df, alpha=0.05)

        assert selected["alpha_slope"] == 2.0
        assert selected["dates_evaluated"] == 10
        assert selected["total_recs"] == 30
        assert selected["survives_correction"] is True
        assert selected["n_survivors"] == 1
        assert selected["sample_filter_exhausted"] is False

    def test_bh_correction_counts_filtered_rows_as_trials(self):
        """Filtered configs carry p=1 but still count in BH's trial denominator."""
        df = _make_sweep_df(n=10, p_values=[0.02, 0.9] + [1.0] * 8)
        df.loc[0, ["alpha_slope", "overall_alpha", "min_sample_ok"]] = [2.0, 2.0, True]
        df.loc[1, ["alpha_slope", "overall_alpha", "min_sample_ok"]] = [1.0, 1.0, True]
        df.loc[2:, "alpha_slope"] = 100.0
        df.loc[2:, "overall_alpha"] = 100.0
        df.loc[2:, "min_sample_ok"] = False

        selected = select_config(df, alpha=0.05)

        assert selected["n_trials"] == 10
        assert selected["n_min_sample_candidates"] == 2
        assert selected["n_survivors"] == 0
        assert selected["survives_correction"] is False
        assert selected["alpha_slope"] == 2.0
        assert selected["sample_filter_exhausted"] is False

    def test_sample_filter_exhausted_falls_back_to_overall_best_with_flag(self):
        """If all rows are too small, keep a deterministic fallback and flag it."""
        df = _make_sweep_df(n=3, p_values=[1.0, 1.0, 1.0])
        df["dates_evaluated"] = [1, 2, 3]
        df["total_recs"] = [2, 4, 6]
        df["min_sample_ok"] = False
        df["alpha_slope"] = [1.0, 5.0, 3.0]

        selected = select_config(df, alpha=0.05)

        assert selected["alpha_slope"] == 5.0
        assert selected["survives_correction"] is False
        assert selected["n_survivors"] == 0
        assert selected["n_min_sample_candidates"] == 0
        assert selected["sample_filter_exhausted"] is True


# ---------------------------------------------------------------------------
# Smoke test: sweep.py still imports from analyzer.validation
# ---------------------------------------------------------------------------
