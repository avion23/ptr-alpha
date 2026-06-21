"""Smoke tests for analyzer.signals.core submodule."""
import unittest

import numpy as np
import pandas as pd

from analyzer.exceptions import AnalysisError


def _make_prices():
    dates = pd.date_range("2024-01-01", "2024-06-01", freq="D")
    np.random.seed(42)
    return pd.DataFrame({
        "AAPL": 100 + np.cumsum(np.random.randn(len(dates)) * 0.5),
        "SPY": 400 + np.cumsum(np.random.randn(len(dates)) * 1),
    }, index=dates)


class TestCoreImports(unittest.TestCase):

    def test_module_imports(self):
        import analyzer.signals.core
        self.assertTrue(callable(analyzer.signals.core.calculate_signal_potential))
        self.assertTrue(callable(analyzer.signals.core.compute_signal_potential_with_member_decay))

    def test_private_helpers_callable(self):
        import analyzer.signals.core as c
        self.assertTrue(callable(c._compute_ticker_signals))
        self.assertTrue(callable(c._validate_inputs))
        self.assertTrue(callable(c._resolve_tickers))
        self.assertTrue(callable(c._explode_by_horizon))
        self.assertTrue(callable(c._extract_metadata_arrays))
        self.assertTrue(callable(c._precompute_spy_log_returns))
        self.assertTrue(callable(c._allocate_result_arrays))


class TestValidateInputs(unittest.TestCase):

    def test_empty_entry_prices_raises(self):
        from analyzer.signals.core import _validate_inputs
        prices = _make_prices()
        with self.assertRaises(AnalysisError):
            _validate_inputs(pd.DataFrame(), prices)

    def test_empty_prices_raises(self):
        from analyzer.signals.core import _validate_inputs
        entry = pd.DataFrame({
            "member": ["A"], "ticker": ["AAPL"],
            "disclosure_date": pd.to_datetime(["2024-02-01"]),
            "transaction_type": ["Purchase"], "entry_price": [100.0],
        })
        with self.assertRaises(AnalysisError):
            _validate_inputs(entry, pd.DataFrame())

    def test_missing_columns_raises(self):
        from analyzer.signals.core import _validate_inputs
        prices = _make_prices()
        bad = pd.DataFrame({"member": ["A"]})
        with self.assertRaises(AnalysisError):
            _validate_inputs(bad, prices)


class TestAllocateResultArrays(unittest.TestCase):

    def test_correct_shape(self):
        from analyzer.signals.core import _allocate_result_arrays
        n = 10
        arrays = _allocate_result_arrays(n)
        expected_keys = {
            "r_peak", "r_trough", "r_decayed_ret", "r_disc_baseline",
            "r_last_price", "r_spy_cum", "r_spy_wsum", "r_spy_first", "r_spy_last",
        }
        self.assertEqual(set(arrays.keys()), expected_keys)
        for key, arr in arrays.items():
            self.assertEqual(len(arr), n)


class TestCalculateSignalPotential(unittest.TestCase):

    def test_basic_call_produces_dataframe(self):
        from analyzer.signals.core import calculate_signal_potential
        prices = _make_prices()
        entry = pd.DataFrame({
            "member": ["Alice", "Bob"],
            "ticker": ["AAPL", "AAPL"],
            "disclosure_date": pd.to_datetime(["2024-02-01", "2024-03-01"]),
            "transaction_type": ["Purchase", "Purchase"],
            "entry_price": [102.0, 105.0],
        })
        result = calculate_signal_potential(entry, prices, [30, 60])
        self.assertFalse(result.empty)
        self.assertIn("signal_score" if "signal_score" in result.columns else "peak_potential_pct", result.columns)


class TestWithMemberDecay(unittest.TestCase):

    def test_empty_decay_map_falls_back_to_default(self):
        from analyzer.signals.core import compute_signal_potential_with_member_decay
        prices = _make_prices()
        entry = pd.DataFrame({
            "member": ["Alice"],
            "ticker": ["AAPL"],
            "disclosure_date": pd.to_datetime(["2024-02-01"]),
            "transaction_type": ["Purchase"],
            "entry_price": [102.0],
        })
        # Empty map -> just use default decay
        result = compute_signal_potential_with_member_decay(
            entry, prices, [30], member_decay_map={},
        )
        self.assertFalse(result.empty)


if __name__ == "__main__":
    unittest.main()
