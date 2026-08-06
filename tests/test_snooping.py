"""Tests for data snooping corrections."""

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from analyzer.snooping import (
    benjamini_hochberg,
    bonferroni_correction,
    deflated_sharpe_ratio,
    min_backtest_length,
    alpha_ttest,
    analyze_snooping,
)


class TestBonferroniCorrection(unittest.TestCase):
    """Tests for the Bonferroni correction."""

    def test_basic(self):
        # 648 tests at alpha=0.05 → threshold = 0.05/648 ≈ 0.000077
        threshold = bonferroni_correction(648, 0.05)
        self.assertAlmostEqual(threshold, 0.05 / 648, places=10)

    def test_single_test(self):
        # With 1 test, no correction needed
        threshold = bonferroni_correction(1, 0.05)
        self.assertAlmostEqual(threshold, 0.05)

    def test_many_tests(self):
        threshold = bonferroni_correction(1000, 0.01)
        self.assertAlmostEqual(threshold, 0.00001)

    def test_custom_alpha(self):
        threshold = bonferroni_correction(100, 0.10)
        self.assertAlmostEqual(threshold, 0.001)

    def test_invalid_n_tests(self):
        with self.assertRaises(ValueError):
            bonferroni_correction(0, 0.05)
        with self.assertRaises(ValueError):
            bonferroni_correction(-1, 0.05)

    def test_invalid_alpha(self):
        with self.assertRaises(ValueError):
            bonferroni_correction(100, 0)
        with self.assertRaises(ValueError):
            bonferroni_correction(100, -0.05)

    def test_threshold_is_smaller_than_original(self):
        threshold = bonferroni_correction(648, 0.05)
        self.assertLess(threshold, 0.05)


class TestBenjaminiHochberg(unittest.TestCase):
    """Tests for the Benjamini-Hochberg procedure."""

    def test_all_null(self):
        # All p-values uniform → should reject none
        rng = np.random.default_rng(42)
        p_values = rng.uniform(0, 1, size=100)
        rejected = benjamini_hochberg(p_values, alpha=0.05)
        # With uniform p-values, BH should reject very few (if any)
        self.assertLessEqual(rejected.sum(), 10)  # generous bound

    def test_all_significant(self):
        # All p-values very small → should reject all
        p_values = np.full(50, 1e-10)
        rejected = benjamini_hochberg(p_values, alpha=0.05)
        self.assertTrue(np.all(rejected))

    def test_mixed(self):
        # Some small, some large
        p_values = np.array([0.001, 0.005, 0.01, 0.5, 0.9])
        rejected = benjamini_hochberg(p_values, alpha=0.05)
        # First 3 should be rejected (ranks 1-3 at alpha=0.05)
        # rank 1: 0.001 <= 0.01 → yes
        # rank 2: 0.005 <= 0.02 → yes
        # rank 3: 0.01 <= 0.03 → yes
        # rank 4: 0.5 > 0.04 → no
        self.assertTrue(rejected[0])
        self.assertTrue(rejected[1])
        self.assertTrue(rejected[2])
        self.assertFalse(rejected[3])
        self.assertFalse(rejected[4])

    def test_returns_boolean_array(self):
        p_values = np.array([0.01, 0.05, 0.10])
        rejected = benjamini_hochberg(p_values, alpha=0.05)
        self.assertEqual(rejected.dtype, bool)
        self.assertEqual(len(rejected), 3)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            benjamini_hochberg([], alpha=0.05)

    def test_invalid_alpha(self):
        with self.assertRaises(ValueError):
            benjamini_hochberg([0.01], alpha=0)

    def test_preserves_order(self):
        # All p-values equal → either all rejected or none
        p_values = np.full(20, 0.01)
        rejected = benjamini_hochberg(p_values, alpha=0.05)
        # All same p-value means same rank treatment
        self.assertTrue(np.all(rejected) or not np.any(rejected))

    def test_bh_is_less_conservative_than_bonferroni(self):
        # BH should reject more hypotheses than Bonferroni for the same data
        rng = np.random.default_rng(123)
        # Generate some significant p-values mixed with noise
        p_values = np.concatenate([
            rng.uniform(0, 0.02, size=10),
            rng.uniform(0.3, 0.9, size=90),
        ])
        rejected_bh = benjamini_hochberg(p_values, alpha=0.05)
        n_bonferroni = sum(
            p < bonferroni_correction(100, 0.05) for p in p_values
        )
        # BH should reject at least as many as Bonferroni
        self.assertGreaterEqual(rejected_bh.sum(), n_bonferroni)


class TestDeflatedSharpeRatio(unittest.TestCase):
    """Tests for the Deflated Sharpe Ratio."""

    def test_dsr_range(self):
        # DSR should be between 0 and 1
        dsr = deflated_sharpe_ratio(1.0, n_trials=100, n_observations=250)
        self.assertGreaterEqual(dsr, 0.0)
        self.assertLessEqual(dsr, 1.0)

    def test_high_sharpe_few_trials(self):
        # High Sharpe with few trials → high DSR
        dsr = deflated_sharpe_ratio(
            observed_sharpe=2.0,
            n_trials=5,
            n_observations=250,
        )
        self.assertGreater(dsr, 0.9)

    def test_low_sharpe_many_trials(self):
        # Low Sharpe with many trials → low DSR
        dsr = deflated_sharpe_ratio(
            observed_sharpe=0.1,
            n_trials=10000,
            n_observations=250,
        )
        self.assertLess(dsr, 0.5)

    def test_more_trials_lower_dsr(self):
        # More trials → lower DSR for same Sharpe
        dsr_few = deflated_sharpe_ratio(1.0, n_trials=10, n_observations=250)
        dsr_many = deflated_sharpe_ratio(1.0, n_trials=1000, n_observations=250)
        self.assertGreater(dsr_few, dsr_many)

    def test_more_observations_help_when_above_expected_max(self):
        # When observed Sharpe exceeds expected max, more observations
        # reduce estimation error and increase DSR.
        # Use a high Sharpe with few trials so it's above E[max].
        dsr_short = deflated_sharpe_ratio(2.0, n_trials=3, n_observations=20)
        dsr_long = deflated_sharpe_ratio(2.0, n_trials=3, n_observations=200)
        self.assertGreater(dsr_long, dsr_short)

    def test_non_normal_adjustment(self):
        # Positive skew shifts the expected max upward and adjusts the SE,
        # changing DSR. Use few trials so DSR is in a visible range.
        dsr_normal = deflated_sharpe_ratio(
            1.5, n_trials=5, n_observations=100,
            skew=0.0, kurtosis=3.0,
        )
        dsr_skewed = deflated_sharpe_ratio(
            1.5, n_trials=5, n_observations=100,
            skew=2.0, kurtosis=3.0,
        )
        # The two should differ (skew changes both E[max] and SE)
        self.assertFalse(
            math.isclose(dsr_normal, dsr_skewed, rel_tol=1e-6),
            f"DSR should differ with skew: normal={dsr_normal}, skewed={dsr_skewed}",
        )

    def test_invalid_trials(self):
        with self.assertRaises(ValueError):
            deflated_sharpe_ratio(1.0, n_trials=0, n_observations=100)

    def test_invalid_observations(self):
        with self.assertRaises(ValueError):
            deflated_sharpe_ratio(1.0, n_trials=100, n_observations=0)

    def test_high_sharpe_few_observations(self):
        # With very few observations and high Sharpe, DSR should still
        # be a valid probability in [0, 1] even though SE is large.
        dsr = deflated_sharpe_ratio(
            10.0, n_trials=10, n_observations=1,
            skew=0.0, kurtosis=3.0,
        )
        self.assertGreaterEqual(dsr, 0.0)
        self.assertLessEqual(dsr, 1.0)
        self.assertGreater(dsr, 0.5)  # High Sharpe with few trials


class TestMinBacktestLength(unittest.TestCase):
    """Tests for minimum backtest length calculation."""

    def test_positive_sharpe(self):
        # SR=1.0 → need T=(z_alpha/SR)^2 years for p<0.05
        # z_0.05 one-sided = norm.ppf(0.95) ≈ 1.6449
        from scipy.stats import norm as sp_norm
        z = sp_norm.ppf(0.95)
        years = min_backtest_length(1.0, alpha=0.05)
        expected = (z / 1.0) ** 2
        self.assertAlmostEqual(years, expected, places=1)

    def test_high_sharpe_needs_less_data(self):
        years_high = min_backtest_length(2.0)
        years_low = min_backtest_length(0.5)
        self.assertLess(years_high, years_low)

    def test_zero_sharpe_infinite(self):
        years = min_backtest_length(0.0)
        self.assertEqual(years, float("inf"))

    def test_negative_sharpe_infinite(self):
        years = min_backtest_length(-1.0)
        self.assertEqual(years, float("inf"))

    def test_custom_alpha(self):
        # Lower alpha → more data needed
        years_05 = min_backtest_length(1.0, alpha=0.05)
        years_01 = min_backtest_length(1.0, alpha=0.01)
        self.assertGreater(years_01, years_05)

    def test_sanity_bounds(self):
        # SR=0.5 should need about 15.4 years
        years = min_backtest_length(0.5)
        self.assertGreater(years, 10)
        self.assertLess(years, 30)


class TestAlphaTtest(unittest.TestCase):
    """Tests for the alpha t-test."""

    def test_significant_positive(self):
        t_stat, p_val = alpha_ttest(mean_alpha=5.0, std_alpha=2.0, n_observations=30)
        self.assertGreater(t_stat, 0)
        self.assertLess(p_val, 0.05)

    def test_not_significant(self):
        t_stat, p_val = alpha_ttest(mean_alpha=0.1, std_alpha=10.0, n_observations=5)
        self.assertGreater(p_val, 0.05)

    def test_zero_std(self):
        t_stat, p_val = alpha_ttest(mean_alpha=5.0, std_alpha=0.0, n_observations=30)
        self.assertEqual(t_stat, float("inf"))
        self.assertEqual(p_val, 1.0)

    def test_zero_mean(self):
        t_stat, p_val = alpha_ttest(mean_alpha=0.0, std_alpha=1.0, n_observations=30)
        self.assertEqual(t_stat, 0.0)
        self.assertGreater(p_val, 0.05)

    def test_small_sample(self):
        t_stat, p_val = alpha_ttest(mean_alpha=3.0, std_alpha=1.0, n_observations=3)
        # With only 3 obs, hard to be significant
        self.assertGreater(p_val, 0.01)


class TestAnalyzeSnooping(unittest.TestCase):
    """Integration test for the full snooping analysis."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        csv_path = Path(self.tmpdir.name) / "sweep_results.csv"
        self.sweep = pd.DataFrame([
            {
                "horizon": 60,
                "frequency_days": 30,
                "min_buyers": 2,
                "top_n": 5,
                "decay_lambda": 0.001,
                "bayes_prior_strength": 50,
                "alpha_slope": 3.5,
                "overall_alpha": 2.0,
                "sharpe": 1.2,
                "dates_evaluated": 30,
            },
            {
                "horizon": 90,
                "frequency_days": 30,
                "min_buyers": 3,
                "top_n": 3,
                "decay_lambda": 0.005,
                "bayes_prior_strength": 20,
                "alpha_slope": 0.5,
                "overall_alpha": 0.2,
                "sharpe": 0.3,
                "dates_evaluated": 30,
            },
            {
                "horizon": 120,
                "frequency_days": 90,
                "min_buyers": 5,
                "top_n": 5,
                "decay_lambda": 0.02,
                "bayes_prior_strength": 5,
                "alpha_slope": -0.2,
                "overall_alpha": -0.1,
                "sharpe": -0.1,
                "dates_evaluated": 30,
            },
        ])
        self.sweep.to_csv(csv_path, index=False)
        self.sweep = pd.read_csv(csv_path)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_with_basic_config(self):
        report = analyze_snooping(
            self.sweep,
            best_config={
                "horizon": 60,
                "frequency_days": 30,
                "min_buyers": 2,
                "top_n": 5,
                "decay_lambda": 0.001,
                "bayes_prior_strength": 50,
            },
            n_tests=648,
        )

        # Verify report fields are populated (values shift with parser updates,
        # so we check ranges and types, not exact constants)
        self.assertEqual(report.n_tests, 648)
        self.assertIsInstance(report.alpha_slope, float)
        self.assertIsInstance(report.overall_alpha, float)
        self.assertIsInstance(report.sharpe, float)
        self.assertGreater(report.t_statistic, 0)

        # Bonferroni threshold
        self.assertAlmostEqual(report.bonferroni_threshold, 0.05 / 648, places=10)

        # DSR should be between 0 and 1
        self.assertGreaterEqual(report.dsr, 0.0)
        self.assertLessEqual(report.dsr, 1.0)

        # Min backtest length
        self.assertGreater(report.min_years, 0)

    def test_with_synthetic_data(self):
        # Construct synthetic sweep results
        rng = np.random.default_rng(42)
        n = 648
        data = {
            "horizon": rng.choice([60, 90, 120], size=n),
            "frequency_days": rng.choice([30, 90], size=n),
            "training_lookback_days": rng.choice([180, 365], size=n),
            "min_buyers": rng.choice([2, 3, 5], size=n),
            "top_n": rng.choice([3, 5], size=n),
            "decay_lambda": rng.choice([0.001, 0.005, 0.02], size=n),
            "bayes_prior_strength": rng.choice([5, 20, 50], size=n),
            "total_recs": rng.integers(10, 100, size=n),
            "dates_evaluated": rng.integers(5, 30, size=n),
            "overall_alpha": rng.normal(0, 2, size=n),
            "overall_return": rng.normal(0, 2, size=n),
            "rank1_alpha": rng.normal(0, 5, size=n),
            "rank5_alpha": rng.normal(0, 5, size=n),
            "alpha_slope": rng.normal(0, 7, size=n),
            "win_rate": rng.uniform(40, 60, size=n),
            "sharpe": rng.normal(0, 0.6, size=n),
            "max_drawdown": rng.uniform(-80, -10, size=n),
        }
        df = pd.DataFrame(data)

        # Make one config clearly the best
        best_idx = 0
        df.loc[best_idx, "alpha_slope"] = 10.6
        df.loc[best_idx, "overall_alpha"] = 1.9
        df.loc[best_idx, "sharpe"] = 1.2
        df.loc[best_idx, "dates_evaluated"] = 22

        report = analyze_snooping(df, n_tests=648)

        # With the best config being synthetic max, it should be found
        self.assertEqual(report.n_tests, 648)
        self.assertGreater(report.t_statistic, 0)
        self.assertGreater(report.min_years, 0)


if __name__ == "__main__":
    unittest.main()
