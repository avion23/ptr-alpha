"""Smoke tests for analyzer.signals module."""
import unittest

import numpy as np
import pandas as pd

from analyzer.signals import (
    CONVICTION_WEIGHT_ALPHA,
    CONVICTION_WEIGHT_REALIZED,
    calculate_signal_potential,
    get_member_signals,
    get_top_signals,
)
from analyzer.exceptions import AnalysisError


class TestSignalConstants(unittest.TestCase):

    def test_conviction_weights_sum_to_one(self):
        # Conviction weights define the signal_score blend
        self.assertEqual(CONVICTION_WEIGHT_ALPHA + CONVICTION_WEIGHT_REALIZED, 1.0)
        self.assertGreaterEqual(CONVICTION_WEIGHT_ALPHA, 0.0)
        self.assertGreaterEqual(CONVICTION_WEIGHT_REALIZED, 0.0)


class TestCalculateSignalPotential(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        self.dates = pd.date_range("2024-01-01", "2024-06-01", freq="D")
        self.prices_df = pd.DataFrame({
            "AAPL": 100 + np.cumsum(np.random.randn(len(self.dates)) * 0.5),
            "SPY": 400 + np.cumsum(np.random.randn(len(self.dates)) * 1),
        }, index=self.dates)

        self.entry_prices = pd.DataFrame({
            "member": ["Alice", "Bob"],
            "ticker": ["AAPL", "AAPL"],
            "disclosure_date": pd.to_datetime(["2024-02-01", "2024-03-01"]),
            "transaction_type": ["Purchase", "Purchase"],
            "entry_price": [102.0, 105.0],
        })

    def test_returns_dataframe_with_expected_columns(self):
        result = calculate_signal_potential(
            self.entry_prices, self.prices_df, [30, 60],
        )
        expected_cols = {
            "member", "ticker", "disclosure_date", "signal_type",
            "horizon_days", "entry_price", "peak_potential_pct",
            "decayed_return_pct", "spy_alpha_pct", "total_return_pct",
            "total_spy_alpha_pct", "decayed_spy_return_pct",
        }
        self.assertTrue(expected_cols.issubset(set(result.columns)))

    def test_explodes_across_horizons(self):
        result = calculate_signal_potential(
            self.entry_prices, self.prices_df, [30, 60, 90],
        )
        # 2 entries * 3 horizons = 6 rows
        self.assertEqual(len(result), 6)
        self.assertEqual(set(result["horizon_days"].unique()), {30, 60, 90})

    def test_handles_missing_ticker(self):
        entry_prices = pd.DataFrame({
            "member": ["Alice"],
            "ticker": ["MISSING"],
            "disclosure_date": pd.to_datetime(["2024-02-01"]),
            "transaction_type": ["Purchase"],
            "entry_price": [100.0],
        })
        result = calculate_signal_potential(entry_prices, self.prices_df, [30])
        # Missing ticker should produce NaN signal values rather than crash
        self.assertEqual(len(result), 1)
        self.assertTrue(pd.isna(result.iloc[0]["spy_alpha_pct"]))

    def test_incomplete_forward_window_is_not_shortened(self):
        entry = self.entry_prices.iloc[[0]].copy()
        entry["disclosure_date"] = pd.Timestamp("2024-05-20")
        result = calculate_signal_potential(entry, self.prices_df, [180])

        row = result.iloc[0]
        for column in (
            "peak_potential_pct", "decayed_return_pct", "spy_alpha_pct",
            "total_return_pct", "total_spy_alpha_pct",
        ):
            self.assertTrue(pd.isna(row[column]), column)

    def test_weekend_window_end_is_considered_complete(self):
        entry = self.entry_prices.iloc[[0]].copy()
        entry["disclosure_date"] = pd.Timestamp("2024-04-30")
        result = calculate_signal_potential(entry, self.prices_df, [30])
        self.assertTrue(pd.notna(result.iloc[0]["total_spy_alpha_pct"]))


class TestGetTopSignals(unittest.TestCase):

    def _make_signals(self) -> pd.DataFrame:
        return pd.DataFrame({
            "member": ["Alice", "Bob", "Charlie"],
            "ticker": ["AAPL", "AAPL", "AAPL"],
            "disclosure_date": pd.to_datetime([
                "2024-02-01", "2024-02-15", "2024-03-01",
            ]),
            "signal_type": ["Purchase", "Purchase", "Purchase"],
            "horizon_days": [90, 90, 90],
            "entry_price": [100.0, 105.0, 110.0],
            "peak_potential_pct": [5.0, 8.0, 3.0],
            "decayed_return_pct": [2.0, 4.0, 1.0],
            "spy_alpha_pct": [1.0, 2.5, -0.5],
            "total_return_pct": [3.0, 5.0, 0.5],
            "total_spy_alpha_pct": [2.0, 3.5, -0.5],
        })

    def test_raises_on_empty(self):
        with self.assertRaises(AnalysisError):
            get_top_signals(pd.DataFrame())

    def test_returns_top_n_sorted_by_score(self):
        signals = self._make_signals()
        top = get_top_signals(signals, horizon=90, top_n=2)
        self.assertEqual(len(top), 2)
        self.assertIn("signal_score", top.columns)


class TestGetMemberSignals(unittest.TestCase):

    def test_raises_on_empty(self):
        with self.assertRaises(AnalysisError):
            get_member_signals(pd.DataFrame(), member="Alice")


if __name__ == "__main__":
    unittest.main()
