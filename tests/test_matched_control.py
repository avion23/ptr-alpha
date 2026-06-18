from __future__ import annotations

import unittest
from datetime import date

import numpy as np
import pandas as pd

from analyzer.matched_control import (
    MatchedControlResult,
    _compute_max_drawdown,
    _compute_realized_volatility,
    _market_cap_tier,
    find_matched_controls,
    run_matched_control_backtest,
)


def _make_prices(tickers: list[str], n_days: int = 200, seed: int = 42) -> pd.DataFrame:
    """Create synthetic price data with controllable drift/vol per ticker."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    data = {}
    for i, ticker in enumerate(tickers):
        drift = 0.0003 * (i + 1)  # varying drift
        vol = 0.01 + 0.005 * (i % 5)  # varying vol
        returns = rng.normal(drift, vol, n_days)
        prices = 100.0 * np.cumprod(1 + returns)
        data[ticker] = prices
    return pd.DataFrame(data, index=dates)


def _make_signals(rows: list[dict]) -> pd.DataFrame:
    base = {
        "member": [],
        "ticker": [],
        "disclosure_date": [],
        "signal_type": [],
        "horizon_days": [],
        "entry_price": [],
        "decayed_return_pct": [],
        "peak_potential_pct": [],
        "spy_alpha_pct": [],
        "total_return_pct": [],
        "total_spy_alpha_pct": [],
    }
    for row in rows:
        for key in base:
            base[key].append(row.get(key))
    df = pd.DataFrame(base)
    df["disclosure_date"] = pd.to_datetime(df["disclosure_date"])
    return df


def _make_transactions(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    df["disclosure_date"] = pd.to_datetime(df["disclosure_date"])
    return df


class TestComputeRealizedVolatility(unittest.TestCase):
    def setUp(self):
        self.prices = _make_prices(["A", "B"], n_days=100)

    def test_returns_float_for_valid_ticker(self):
        result = _compute_realized_volatility(self.prices, "A", pd.Timestamp("2024-03-01"))
        self.assertIsInstance(result, float)
        self.assertGreater(result, 0)

    def test_returns_none_for_missing_ticker(self):
        result = _compute_realized_volatility(self.prices, "MISSING", pd.Timestamp("2024-03-01"))
        self.assertIsNone(result)

    def test_returns_none_for_insufficient_data(self):
        result = _compute_realized_volatility(self.prices, "A", pd.Timestamp("2024-01-02"))
        self.assertIsNone(result)


class TestComputeMaxDrawdown(unittest.TestCase):
    def test_returns_float_for_valid_ticker(self):
        prices = _make_prices(["X"], n_days=100)
        result = _compute_max_drawdown(prices, "X", pd.Timestamp("2024-03-01"))
        self.assertIsInstance(result, float)
        self.assertLessEqual(result, 0)

    def test_returns_none_for_missing_ticker(self):
        prices = _make_prices(["X"], n_days=100)
        result = _compute_max_drawdown(prices, "MISSING", pd.Timestamp("2024-03-01"))
        self.assertIsNone(result)


class TestMarketCapTier(unittest.TestCase):
    def test_mega(self):
        self.assertEqual(_market_cap_tier(300e9), "mega")

    def test_large(self):
        self.assertEqual(_market_cap_tier(50e9), "large")

    def test_mid(self):
        self.assertEqual(_market_cap_tier(5e9), "mid")

    def test_small(self):
        self.assertEqual(_market_cap_tier(500e6), "small")


class TestFindMatchedControls(unittest.TestCase):
    def setUp(self):
        self.tickers = ["TREAT", "C1", "C2", "C3", "C4", "C5"]
        self.prices = _make_prices(self.tickers, n_days=100)
        self.signals = _make_signals([])

        # Provide sector data externally to avoid yfinance calls
        self.sector_data = {
            "TREAT": {"sector": "Tech", "market_cap": 50e9},
            "C1": {"sector": "Tech", "market_cap": 45e9},
            "C2": {"sector": "Tech", "market_cap": 55e9},
            "C3": {"sector": "Health", "market_cap": 48e9},
            "C4": {"sector": "Tech", "market_cap": 1e9},
            "C5": {"sector": "Tech", "market_cap": 52e9},
        }

    def test_returns_correct_number_of_controls(self):
        controls = find_matched_controls(
            "TREAT", date(2024, 3, 1), self.tickers, self.prices, self.signals,
            n_controls=3, sector_data=self.sector_data,
        )
        self.assertLessEqual(len(controls), 3)
        self.assertGreater(len(controls), 0)

    def test_treatment_ticker_excluded(self):
        controls = find_matched_controls(
            "TREAT", date(2024, 3, 1), self.tickers, self.prices, self.signals,
            n_controls=10, sector_data=self.sector_data,
        )
        self.assertNotIn("TREAT", controls)

    def test_controls_are_from_available_tickers(self):
        controls = find_matched_controls(
            "TREAT", date(2024, 3, 1), self.tickers, self.prices, self.signals,
            n_controls=10, sector_data=self.sector_data,
        )
        for c in controls:
            self.assertIn(c, self.tickers)

    def test_returns_empty_for_no_candidates(self):
        controls = find_matched_controls(
            "TREAT", date(2024, 3, 1), ["TREAT"], self.prices, self.signals,
            n_controls=10, sector_data=self.sector_data,
        )
        self.assertEqual(controls, [])

    def test_sector_filtering(self):
        """Controls should prefer same-sector tickers."""
        controls = find_matched_controls(
            "TREAT", date(2024, 3, 1), self.tickers, self.prices, self.signals,
            n_controls=10, sector_data=self.sector_data,
        )
        # C3 is Health sector — should be deprioritized vs Tech sector tickers
        if len(controls) >= 3:
            self.assertIn("C1", controls)
            self.assertIn("C2", controls)
            # C3 should rank lower than same-sector peers
            if "C3" in controls:
                idx_c3 = controls.index("C3")
                idx_c1 = controls.index("C1") if "C1" in controls else len(controls)
                self.assertGreater(idx_c3, idx_c1)


class TestFindMatchedControlsMatching(unittest.TestCase):
    """Test that matching criteria are respected."""

    def setUp(self):
        # Create prices with very different vol profiles
        rng = np.random.default_rng(99)
        dates = pd.bdate_range("2024-01-01", periods=100)
        self.prices = pd.DataFrame(index=dates)
        # Treatment: low vol
        self.prices["TREAT"] = 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, 100))
        # High-vol controls
        self.prices["HV1"] = 100 * np.cumprod(1 + rng.normal(0.0005, 0.05, 100))
        self.prices["HV2"] = 100 * np.cumprod(1 + rng.normal(0.0005, 0.04, 100))
        # Low-vol controls (similar to treatment)
        self.prices["LV1"] = 100 * np.cumprod(1 + rng.normal(0.0005, 0.012, 100))
        self.prices["LV2"] = 100 * np.cumprod(1 + rng.normal(0.0005, 0.009, 100))

        self.signals = _make_signals([])
        self.sector_data = {
            "TREAT": {"sector": "Tech", "market_cap": 50e9},
            "HV1": {"sector": "Tech", "market_cap": 48e9},
            "HV2": {"sector": "Tech", "market_cap": 52e9},
            "LV1": {"sector": "Tech", "market_cap": 47e9},
            "LV2": {"sector": "Tech", "market_cap": 51e9},
        }
        self.all_tickers = list(self.sector_data.keys())

    def test_volatility_matching_prefers_similar_vol(self):
        controls = find_matched_controls(
            "TREAT", date(2024, 4, 1), self.all_tickers, self.prices, self.signals,
            n_controls=2, sector_data=self.sector_data,
        )
        # Low-vol controls should be preferred over high-vol ones
        for c in controls:
            self.assertIn(c, ["LV1", "LV2"], f"Expected low-vol control, got {c}")


class TestRunMatchedControlBacktest(unittest.TestCase):
    def setUp(self):
        # Build a minimal but complete dataset
        dates = pd.bdate_range("2024-01-01", periods=300)
        rng = np.random.default_rng(42)
        self.prices = pd.DataFrame(index=dates)
        for t in ["AAPL", "MSFT", "GOOG", "AMZN", "SPY"]:
            self.prices[t] = 100 * np.cumprod(1 + rng.normal(0.0004, 0.015, 300))

        # Signals: enough history for training
        self.signals = _make_signals([
            {
                "member": "Alice", "ticker": "AAPL",
                "disclosure_date": "2023-06-01",
                "signal_type": "Purchase", "horizon_days": 90,
                "entry_price": 95.0, "decayed_return_pct": 15.0,
                "peak_potential_pct": 25.0, "spy_alpha_pct": 10.0,
                "total_return_pct": 20.0, "total_spy_alpha_pct": 12.0,
            },
            {
                "member": "Bob", "ticker": "AAPL",
                "disclosure_date": "2023-08-01",
                "signal_type": "Purchase", "horizon_days": 90,
                "entry_price": 98.0, "decayed_return_pct": 12.0,
                "peak_potential_pct": 20.0, "spy_alpha_pct": 8.0,
                "total_return_pct": 18.0, "total_spy_alpha_pct": 10.0,
            },
            {
                "member": "Alice", "ticker": "MSFT",
                "disclosure_date": "2023-07-01",
                "signal_type": "Purchase", "horizon_days": 90,
                "entry_price": 90.0, "decayed_return_pct": 10.0,
                "peak_potential_pct": 18.0, "spy_alpha_pct": 6.0,
                "total_return_pct": 14.0, "total_spy_alpha_pct": 8.0,
            },
            {
                "member": "Bob", "ticker": "MSFT",
                "disclosure_date": "2023-09-01",
                "signal_type": "Purchase", "horizon_days": 90,
                "entry_price": 92.0, "decayed_return_pct": 8.0,
                "peak_potential_pct": 15.0, "spy_alpha_pct": 5.0,
                "total_return_pct": 12.0, "total_spy_alpha_pct": 7.0,
            },
        ])

        self.transactions = _make_transactions([
            {
                "member": "Alice", "ticker": "CAND1",
                "transaction_date": "2024-08-01", "disclosure_date": "2024-08-05",
                "transaction_type": "Purchase",
            },
            {
                "member": "Bob", "ticker": "CAND1",
                "transaction_date": "2024-08-02", "disclosure_date": "2024-08-06",
                "transaction_type": "Purchase",
            },
            {
                "member": "Alice", "ticker": "CAND2",
                "transaction_date": "2024-08-10", "disclosure_date": "2024-08-14",
                "transaction_type": "Purchase",
            },
            {
                "member": "Bob", "ticker": "CAND2",
                "transaction_date": "2024-08-11", "disclosure_date": "2024-08-15",
                "transaction_type": "Purchase",
            },
        ])

        self.sector_data = {
            "CAND1": {"sector": "Tech", "market_cap": 50e9},
            "CAND2": {"sector": "Tech", "market_cap": 45e9},
            "AAPL": {"sector": "Tech", "market_cap": 200e9},
            "MSFT": {"sector": "Tech", "market_cap": 180e9},
            "GOOG": {"sector": "Tech", "market_cap": 150e9},
            "AMZN": {"sector": "Tech", "market_cap": 160e9},
            "SPY": {"sector": "ETF", "market_cap": 0},
        }

    def test_returns_expected_columns(self):
        # Extend prices far enough to cover horizon
        dates_ext = pd.bdate_range("2024-01-01", periods=400)
        rng = np.random.default_rng(42)
        prices_ext = pd.DataFrame(index=dates_ext)
        for t in ["CAND1", "CAND2", "AAPL", "MSFT", "GOOG", "AMZN", "SPY"]:
            prices_ext[t] = 100 * np.cumprod(1 + rng.normal(0.0004, 0.015, 400))

        result = run_matched_control_backtest(
            self.signals, self.transactions, prices_ext,
            start_date=date(2024, 9, 1),
            end_date=date(2024, 10, 1),
            horizon=90, top_n=5, frequency_days=14, n_controls=3,
            min_buyers=2, lookback_days=60, threshold=5.0,
            training_lookback_days=365,
        )
        expected_cols = {
            "as_of_date", "ticker", "rank", "alpha", "excess_alpha",
            "control_mean_alpha", "n_controls", "sector", "volatility", "drawdown",
        }
        if not result.empty:
            self.assertTrue(expected_cols.issubset(set(result.columns)))

    def test_excess_alpha_equals_alpha_minus_control_mean(self):
        dates_ext = pd.bdate_range("2024-01-01", periods=400)
        rng = np.random.default_rng(42)
        prices_ext = pd.DataFrame(index=dates_ext)
        for t in ["CAND1", "CAND2", "AAPL", "MSFT", "GOOG", "AMZN", "SPY"]:
            prices_ext[t] = 100 * np.cumprod(1 + rng.normal(0.0004, 0.015, 400))

        result = run_matched_control_backtest(
            self.signals, self.transactions, prices_ext,
            start_date=date(2024, 9, 1),
            end_date=date(2024, 10, 1),
            horizon=90, top_n=5, frequency_days=14, n_controls=3,
            min_buyers=2, lookback_days=60, threshold=5.0,
            training_lookback_days=365,
        )
        if not result.empty:
            for _, row in result.iterrows():
                expected = round(row["alpha"] - row["control_mean_alpha"], 2)
                self.assertAlmostEqual(row["excess_alpha"], expected, places=1)


class TestMatchedControlResultDataclass(unittest.TestCase):
    def test_fields(self):
        r = MatchedControlResult(
            as_of_date=date(2024, 1, 1),
            treatment_ticker="AAPL",
            treatment_alpha=5.0,
            control_tickers=["MSFT", "GOOG"],
            control_alphas=[3.0, 2.0],
            control_mean_alpha=2.5,
            excess_alpha=2.5,
            n_controls=2,
        )
        self.assertEqual(r.treatment_ticker, "AAPL")
        self.assertEqual(r.excess_alpha, 2.5)
        self.assertEqual(r.n_controls, 2)


if __name__ == "__main__":
    unittest.main()
