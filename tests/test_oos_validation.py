"""Smoke tests for analyzer.oos_validation module."""
import unittest
from unittest.mock import patch

import pandas as pd

from analyzer import oos_validation
from analyzer.oos_validation import (
    BEST_CONFIG,
    _degradation_ratio,
    _print_fold,
    run_backtest_split,
)


class TestBestConfig(unittest.TestCase):

    def test_has_required_keys(self):
        required = {"horizon", "frequency_days", "min_buyers", "top_n"}
        self.assertTrue(required.issubset(BEST_CONFIG.keys()))

    def test_values_are_sensible(self):
        self.assertGreaterEqual(BEST_CONFIG["horizon"], 1)
        self.assertGreaterEqual(BEST_CONFIG["min_buyers"], 1)
        self.assertGreaterEqual(BEST_CONFIG["top_n"], 1)
        self.assertGreaterEqual(BEST_CONFIG["frequency_days"], 1)


class TestDegradationRatio(unittest.TestCase):

    def test_zero_is_alpha_returns_zero(self):
        # Avoid division by zero
        self.assertEqual(_degradation_ratio(0.0, 5.0), 0.0)

    def test_healthy_decay_between_half_and_one(self):
        ratio = _degradation_ratio(10.0, 7.0)
        self.assertEqual(ratio, 0.7)

    def test_alpha_grew_above_one(self):
        ratio = _degradation_ratio(5.0, 8.0)
        self.assertEqual(ratio, 1.6)

    def test_degraded_below_half(self):
        ratio = _degradation_ratio(10.0, 2.0)
        self.assertEqual(ratio, 0.2)


class TestRunBacktestSplit(unittest.TestCase):

    def test_returns_zero_metrics_on_empty_data(self):
        empty_tx = pd.DataFrame()
        empty_prices = pd.DataFrame()
        empty_entries = pd.DataFrame()

        from datetime import date
        with patch("analyzer.analysis.calculate_signal_potential") as mock_calc:
            mock_calc.return_value = pd.DataFrame()
            result = run_backtest_split(
                empty_tx, empty_prices, empty_entries,
                date(2024, 1, 1), date(2024, 6, 1),
                horizon=60,
            )
        self.assertEqual(result["N"], 0)
        self.assertEqual(result["alpha"], 0)


class TestPrintFold(unittest.TestCase):

    def test_does_not_raise(self):
        # _print_fold writes to stdout; just ensure no exception.
        result = {
            "label": "Test", "N": 5, "alpha": 1.5, "slope": 0.5,
            "t": 1.0, "win%": 60.0, "r1": 1.0, "r5": 0.5,
        }
        _print_fold(result)  # should not raise


class TestModuleImports(unittest.TestCase):

    def test_all_public_functions_exported(self):
        # Ensure the public API is importable
        assert callable(oos_validation.run_backtest_split)
        assert callable(oos_validation.run_split_oos)
        assert callable(oos_validation.run_walk_forward)


if __name__ == "__main__":
    unittest.main()