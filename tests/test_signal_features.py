import unittest
from datetime import date

import numpy as np
import pandas as pd

from analyzer.signal_features import (
    CrashHazard,
    SignalFeatures,
    compute_disclosure_lag_weight,
    compute_signal_features,
    estimate_crash_hazard,
)


class TestDisclosureLagWeight(unittest.TestCase):

    def test_zero_lag_returns_one(self):
        self.assertEqual(compute_disclosure_lag_weight(0), 1.0)

    def test_half_life_returns_half(self):
        result = compute_disclosure_lag_weight(60, half_life=60.0)
        self.assertAlmostEqual(result, 0.5, places=2)

    def test_large_lag_near_zero(self):
        result = compute_disclosure_lag_weight(365, half_life=60.0)
        self.assertLess(result, 0.05)

    def test_monotonically_decreasing(self):
        w0 = compute_disclosure_lag_weight(0)
        w30 = compute_disclosure_lag_weight(30)
        w60 = compute_disclosure_lag_weight(60)
        w120 = compute_disclosure_lag_weight(120)
        self.assertGreater(w0, w30)
        self.assertGreater(w30, w60)
        self.assertGreater(w60, w120)


class TestCrashHazard(unittest.TestCase):

    def _make_features(self, **overrides) -> SignalFeatures:
        defaults = dict(
            ticker="TEST",
            disclosure_date=date(2024, 1, 1),
            lag_days=30,
            pre_disclosure_return=0.05,
            pre_disclosure_alpha=0.02,
            max_drawdown_to_entry=0.10,
            volatility_20d=0.30,
            drawdown_from_ath=0.15,
            days_since_ipo=365,
            n_buyers_30d=2,
        )
        defaults.update(overrides)
        return SignalFeatures(**defaults)

    def test_high_vol_increases_crash_prob(self):
        low_vol = self._make_features(volatility_20d=0.15)
        high_vol = self._make_features(volatility_20d=0.80)

        crash_low = estimate_crash_hazard(low_vol)
        crash_high = estimate_crash_hazard(high_vol)

        self.assertGreater(crash_high.crash_prob, crash_low.crash_prob)

    def test_large_drawdown_increases_crash_prob(self):
        small_dd = self._make_features(drawdown_from_ath=0.05)
        large_dd = self._make_features(drawdown_from_ath=0.50)

        crash_small = estimate_crash_hazard(small_dd)
        crash_large = estimate_crash_hazard(large_dd)

        self.assertGreater(crash_large.crash_prob, crash_small.crash_prob)

    def test_crash_prob_bounded(self):
        features = self._make_features(volatility_20d=2.0, drawdown_from_ath=0.9, lag_days=300)
        crash = estimate_crash_hazard(features)
        self.assertGreater(crash.crash_prob, 0.0)
        self.assertLessEqual(crash.crash_prob, 1.0)

    def test_var_95_negative(self):
        features = self._make_features()
        crash = estimate_crash_hazard(features)
        self.assertLess(crash.var_95, 0.0)

    def test_cvar_95_worse_than_var(self):
        features = self._make_features()
        crash = estimate_crash_hazard(features)
        self.assertLess(crash.cvar_95, crash.var_95)


class TestComputeSignalFeatures(unittest.TestCase):

    def setUp(self):
        dates = pd.date_range("2024-01-01", "2024-06-01", freq="D")
        np.random.seed(42)
        self.prices_df = pd.DataFrame({
            "AAPL": 100 + np.cumsum(np.random.randn(len(dates)) * 0.5),
            "SPY": 400 + np.cumsum(np.random.randn(len(dates)) * 1),
        }, index=dates)

        self.all_tx = pd.DataFrame({
            "member": ["Alice", "Bob", "Charlie"],
            "ticker": ["AAPL", "AAPL", "MSFT"],
            "transaction_type": ["Purchase", "Purchase", "Purchase"],
            "disclosure_date": pd.to_datetime(["2024-03-01", "2024-03-15", "2024-03-10"]),
        })

    def test_returns_valid_features_no_nan(self):
        features = compute_signal_features(
            ticker="AAPL",
            disclosure_date=date(2024, 3, 1),
            transaction_date=date(2024, 2, 15),
            prices_df=self.prices_df,
            all_tx=self.all_tx,
            as_of_date=date(2024, 3, 1),
        )

        self.assertEqual(features.ticker, "AAPL")
        self.assertEqual(features.lag_days, 15)  # Feb 15 to Mar 1, 2024 (leap year)
        self.assertIsInstance(features.pre_disclosure_return, float)
        self.assertIsInstance(features.volatility_20d, float)
        self.assertIsInstance(features.drawdown_from_ath, float)
        self.assertGreaterEqual(features.volatility_20d, 0.0)
        self.assertGreaterEqual(features.drawdown_from_ath, 0.0)

    def test_lag_days_computed_correctly(self):
        features = compute_signal_features(
            ticker="AAPL",
            disclosure_date=date(2024, 3, 10),
            transaction_date=date(2024, 3, 1),
            prices_df=self.prices_df,
            all_tx=self.all_tx,
            as_of_date=date(2024, 3, 10),
        )
        self.assertEqual(features.lag_days, 9)

    def test_n_buyers_counts_recent_purchases(self):
        features = compute_signal_features(
            ticker="AAPL",
            disclosure_date=date(2024, 3, 1),
            transaction_date=date(2024, 2, 15),
            prices_df=self.prices_df,
            all_tx=self.all_tx,
            as_of_date=date(2024, 3, 1),
        )
        # Only Alice disclosed AAPL by as_of_date (Mar 1); Bob disclosed Mar 15
        self.assertEqual(features.n_buyers_30d, 1)

    def test_days_since_ipo_computed(self):
        features = compute_signal_features(
            ticker="AAPL",
            disclosure_date=date(2024, 3, 1),
            transaction_date=date(2024, 2, 15),
            prices_df=self.prices_df,
            all_tx=self.all_tx,
            as_of_date=date(2024, 3, 1),
        )
        self.assertIsNotNone(features.days_since_ipo)
        self.assertGreater(features.days_since_ipo, 0)

    def test_missing_ticker_prices(self):
        features = compute_signal_features(
            ticker="MISSING",
            disclosure_date=date(2024, 3, 1),
            transaction_date=date(2024, 2, 15),
            prices_df=self.prices_df,
            all_tx=self.all_tx,
            as_of_date=date(2024, 3, 1),
        )
        self.assertEqual(features.volatility_20d, 0.0)
        self.assertEqual(features.drawdown_from_ath, 0.0)


class TestLagAndCrashPenaltyIntegration(unittest.TestCase):

    def test_lag_reduces_score(self):
        base_score = 10.0
        lag_0 = compute_disclosure_lag_weight(0)
        lag_90 = compute_disclosure_lag_weight(90)

        score_fresh = base_score * lag_0
        score_stale = base_score * lag_90

        self.assertGreater(score_fresh, score_stale)

    def test_crash_penalty_reduces_score(self):
        base_score = 10.0

        low_risk = CrashHazard(crash_prob=0.05, expected_return=-0.005, var_95=-0.15, cvar_95=-0.19)
        high_risk = CrashHazard(crash_prob=0.40, expected_return=-0.04, var_95=-0.30, cvar_95=-0.38)

        score_low = base_score * (1 - low_risk.crash_prob)
        score_high = base_score * (1 - high_risk.crash_prob)

        self.assertGreater(score_low, score_high)
        self.assertAlmostEqual(score_low, 9.5, places=1)
        self.assertAlmostEqual(score_high, 6.0, places=1)

    def test_combined_penalties(self):
        base_score = 10.0
        lag_weight = compute_disclosure_lag_weight(60)  # ~0.5
        crash_prob = 0.20

        adjusted = base_score * lag_weight * (1 - crash_prob)
        self.assertLess(adjusted, base_score)
        self.assertGreater(adjusted, 0.0)


if __name__ == "__main__":
    unittest.main()
