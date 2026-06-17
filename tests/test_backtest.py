import unittest
from datetime import date

import numpy as np
import pandas as pd

from analyzer.analysis import (
    backtest_recommendations,
    evaluate_backtest,
    summarize_backtest,
    _price_at_or_before,
)
from analyzer.exceptions import AnalysisError

from .conftest import DatabaseTestCase


def _make_signals(rows):
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


def _make_transactions(rows):
    df = pd.DataFrame(rows)
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    df["disclosure_date"] = pd.to_datetime(df["disclosure_date"])
    return df


class TestPriceAtOrBefore(unittest.TestCase):

    def setUp(self):
        self.prices = pd.DataFrame(
            {"AAPL": [100.0, 101.0, 102.0], "GOOG": [200.0, np.nan, 202.0]},
            index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        )

    def test_returns_price_on_exact_date(self):
        result = _price_at_or_before(self.prices, "AAPL", pd.Timestamp("2024-01-02"))
        self.assertEqual(result, 101.0)

    def test_returns_last_price_before_date(self):
        result = _price_at_or_before(self.prices, "AAPL", pd.Timestamp("2024-01-05"))
        self.assertEqual(result, 102.0)

    def test_returns_none_for_missing_ticker(self):
        result = _price_at_or_before(self.prices, "MSFT", pd.Timestamp("2024-01-02"))
        self.assertIsNone(result)

    def test_returns_none_for_date_before_first(self):
        result = _price_at_or_before(self.prices, "AAPL", pd.Timestamp("2023-12-31"))
        self.assertIsNone(result)

    def test_skips_nan_values(self):
        result = _price_at_or_before(self.prices, "GOOG", pd.Timestamp("2024-01-02"))
        self.assertEqual(result, 200.0)


class TestBacktestRecommendations(unittest.TestCase):

    def setUp(self):
        self.as_of = pd.Timestamp("2025-01-01")
        horizon = 90
        elapsed_cutoff = self.as_of - pd.Timedelta(days=horizon)

        self.signals = _make_signals([
            {
                "member": "Alpha", "ticker": "AAPL",
                "disclosure_date": elapsed_cutoff - pd.Timedelta(days=30),
                "signal_type": "Purchase", "horizon_days": 90,
                "entry_price": 100.0, "decayed_return_pct": 20.0,
                "peak_potential_pct": 30.0, "spy_alpha_pct": 15.0,
                "total_return_pct": 25.0, "total_spy_alpha_pct": 18.0,
            },
            {
                "member": "Alpha", "ticker": "MSFT",
                "disclosure_date": elapsed_cutoff - pd.Timedelta(days=20),
                "signal_type": "Purchase", "horizon_days": 90,
                "entry_price": 200.0, "decayed_return_pct": 15.0,
                "peak_potential_pct": 25.0, "spy_alpha_pct": 10.0,
                "total_return_pct": 18.0, "total_spy_alpha_pct": 12.0,
            },
            {
                "member": "Beta", "ticker": "GOOG",
                "disclosure_date": elapsed_cutoff - pd.Timedelta(days=25),
                "signal_type": "Purchase", "horizon_days": 90,
                "entry_price": 50.0, "decayed_return_pct": -10.0,
                "peak_potential_pct": 5.0, "spy_alpha_pct": -15.0,
                "total_return_pct": -12.0, "total_spy_alpha_pct": -18.0,
            },
        ])

        self.recent_transactions = _make_transactions([
            {
                "member": "Alpha", "ticker": "CAND",
                "transaction_date": "2024-12-10", "disclosure_date": "2024-12-15",
                "transaction_type": "Purchase",
            },
            {
                "member": "Beta", "ticker": "CAND",
                "transaction_date": "2024-12-12", "disclosure_date": "2024-12-17",
                "transaction_type": "Purchase",
            },
        ])

    def test_produces_recommendations_for_multi_buyer_ticker(self):
        recs = backtest_recommendations(
            self.signals, self.recent_transactions, self.as_of,
            horizon=90, lookback_days=60, min_buyers=2, top_n=10, threshold=5.0,
        )
        self.assertFalse(recs.empty)
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs.iloc[0]["ticker"], "CAND")
        self.assertIn("rank", recs.columns)
        self.assertEqual(recs.iloc[0]["rank"], 1)

    def test_returns_empty_when_no_elapsed_training_data(self):
        recent_signals = _make_signals([
            {
                "member": "Alpha", "ticker": "AAPL",
                "disclosure_date": self.as_of - pd.Timedelta(days=10),
                "signal_type": "Purchase", "horizon_days": 90,
                "entry_price": 100.0, "decayed_return_pct": 20.0,
                "peak_potential_pct": 30.0, "spy_alpha_pct": 15.0,
                "total_return_pct": 25.0, "total_spy_alpha_pct": 18.0,
            },
        ])
        recs = backtest_recommendations(
            recent_signals, self.recent_transactions, self.as_of,
            horizon=90, lookback_days=60, min_buyers=2, top_n=10, threshold=5.0,
        )
        self.assertTrue(recs.empty)

    def test_returns_empty_when_no_recent_candidates(self):
        old_transactions = _make_transactions([
            {
                "member": "Alpha", "ticker": "CAND",
                "transaction_date": "2024-01-10", "disclosure_date": "2024-01-15",
                "transaction_type": "Purchase",
            },
            {
                "member": "Beta", "ticker": "CAND",
                "transaction_date": "2024-01-12", "disclosure_date": "2024-01-17",
                "transaction_type": "Purchase",
            },
        ])
        recs = backtest_recommendations(
            self.signals, old_transactions, self.as_of,
            horizon=90, lookback_days=60, min_buyers=2, top_n=10, threshold=5.0,
        )
        self.assertTrue(recs.empty)

    def test_filters_below_min_buyers(self):
        single_buyer = _make_transactions([
            {
                "member": "Alpha", "ticker": "SOLO",
                "transaction_date": "2024-12-10", "disclosure_date": "2024-12-15",
                "transaction_type": "Purchase",
            },
        ])
        recs = backtest_recommendations(
            self.signals, single_buyer, self.as_of,
            horizon=90, lookback_days=60, min_buyers=2, top_n=10, threshold=5.0,
        )
        self.assertTrue(recs.empty)

    def test_no_lookahead_future_signals_excluded_from_training(self):
        signals_with_future = pd.concat([
            self.signals,
            _make_signals([{
                "member": "Alpha", "ticker": "FUT",
                "disclosure_date": self.as_of - pd.Timedelta(days=30),
                "signal_type": "Purchase", "horizon_days": 90,
                "entry_price": 100.0, "decayed_return_pct": -50.0,
                "peak_potential_pct": -40.0, "spy_alpha_pct": -45.0,
                "total_return_pct": -50.0, "total_spy_alpha_pct": -48.0,
            }]),
        ], ignore_index=True)

        recs_elapsed = backtest_recommendations(
            self.signals, self.recent_transactions, self.as_of,
            horizon=90, lookback_days=60, min_buyers=2, top_n=10, threshold=5.0,
        )
        recs_with_future = backtest_recommendations(
            signals_with_future, self.recent_transactions, self.as_of,
            horizon=90, lookback_days=60, min_buyers=2, top_n=10, threshold=5.0,
        )

        self.assertEqual(
            recs_elapsed.iloc[0]["signal_score"],
            recs_with_future.iloc[0]["signal_score"],
        )


class TestEvaluateBacktest(unittest.TestCase):

    def setUp(self):
        dates = pd.date_range("2024-12-01", "2025-06-01", freq="D")
        n = len(dates)
        self.prices = pd.DataFrame(
            {
                "AAPL": [100.0 + i * 0.1 for i in range(n)],
                "GOOG": [200.0 - i * 0.05 for i in range(n)],
                "SPY": [400.0 + i * 0.02 for i in range(n)],
            },
            index=dates,
        )
        self.recommendations = pd.DataFrame(
            {
                "rank": [1, 2],
                "ticker": ["AAPL", "GOOG"],
                "signal_score": [50.0, 30.0],
                "num_buyers": [3, 2],
            }
        )

    def test_computes_forward_returns(self):
        as_of = pd.Timestamp("2025-01-01")
        result = evaluate_backtest(self.recommendations, self.prices, as_of, horizon=90)

        self.assertIn("bt_return_pct", result.columns)
        self.assertIn("bt_alpha_pct", result.columns)
        self.assertIn("bt_spy_return_pct", result.columns)
        self.assertIsNotNone(result.iloc[0]["bt_return_pct"])

    def test_aapl_beats_spy(self):
        as_of = pd.Timestamp("2025-01-01")
        result = evaluate_backtest(self.recommendations, self.prices, as_of, horizon=90)

        aapl_return = result[result["ticker"] == "AAPL"].iloc[0]["bt_return_pct"]
        spy_return = result[result["ticker"] == "AAPL"].iloc[0]["bt_spy_return_pct"]
        aapl_alpha = result[result["ticker"] == "AAPL"].iloc[0]["bt_alpha_pct"]

        self.assertAlmostEqual(aapl_alpha, aapl_return - spy_return, places=1)
        self.assertGreater(aapl_return, spy_return)

    def test_empty_recommendations_returns_empty(self):
        result = evaluate_backtest(
            pd.DataFrame(), self.prices, pd.Timestamp("2025-01-01"), 90
        )
        self.assertTrue(result.empty)

    def test_missing_exit_price_raises(self):
        short_prices = self.prices.loc[:"2025-01-15"].copy()
        with self.assertRaises(AnalysisError):
            evaluate_backtest(
                self.recommendations, short_prices, pd.Timestamp("2025-01-01"), horizon=90
            )

    def test_missing_entry_price_raises(self):
        no_aapl = self.prices.drop(columns=["AAPL"])
        with self.assertRaises(AnalysisError):
            evaluate_backtest(
                self.recommendations, no_aapl, pd.Timestamp("2025-01-01"), horizon=90
            )


class TestSummarizeBacktest(unittest.TestCase):

    def test_groups_by_rank(self):
        results = pd.DataFrame({
            "rank": [1, 1, 2, 2],
            "bt_return_pct": [10.0, -5.0, 20.0, 15.0],
            "bt_alpha_pct": [5.0, -8.0, 15.0, 10.0],
        })
        summary = summarize_backtest(results)
        self.assertEqual(len(summary), 3)
        rank1 = summary[summary["rank"] == 1].iloc[0]
        rank2 = summary[summary["rank"] == 2].iloc[0]
        self.assertEqual(rank1["count"], 2)
        self.assertEqual(rank1["win_rate_pct"], 50.0)
        self.assertAlmostEqual(rank1["avg_return_pct"], 2.5)
        self.assertEqual(rank2["count"], 2)
        self.assertEqual(rank2["win_rate_pct"], 100.0)

    def test_includes_overall_row(self):
        results = pd.DataFrame({
            "rank": [1, 2],
            "bt_return_pct": [10.0, -5.0],
            "bt_alpha_pct": [5.0, -8.0],
        })
        summary = summarize_backtest(results)
        overall = summary[summary["rank"] == "ALL"].iloc[0]
        self.assertEqual(overall["count"], 2)
        self.assertEqual(overall["win_rate_pct"], 50.0)
        self.assertAlmostEqual(overall["avg_return_pct"], 2.5)

    def test_empty_results_returns_empty(self):
        results = pd.DataFrame({"rank": [], "bt_return_pct": [], "bt_alpha_pct": []})
        summary = summarize_backtest(results)
        self.assertTrue(summary.empty)

    def test_all_nan_returns_dropped(self):
        results = pd.DataFrame({
            "rank": [1, 2],
            "bt_return_pct": [10.0, None],
            "bt_alpha_pct": [5.0, None],
        })
        summary = summarize_backtest(results)
        self.assertEqual(len(summary), 2)
        self.assertEqual(summary[summary["rank"] == 1].iloc[0]["count"], 1)


class TestDatabaseDateRange(DatabaseTestCase):

    def test_filters_by_date_range(self):
        df = pd.DataFrame({
            "doc_id": ["d1", "d2", "d3"],
            "member": ["Alice", "Bob", "Charlie"],
            "ticker": ["AAPL", "GOOG", "MSFT"],
            "transaction_date": pd.to_datetime(["2024-01-01", "2024-06-01", "2024-12-01"]),
            "disclosure_date": pd.to_datetime(["2024-01-05", "2024-06-05", "2024-12-05"]),
            "transaction_type": ["Purchase", "Purchase", "Purchase"],
        })
        self.db.upsert_transactions(df)

        result = self.db.get_transactions_by_date_range(
            date(2024, 3, 1), date(2024, 10, 1)
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["ticker"], "GOOG")

    def test_excludes_transaction_date_after_disclosure(self):
        df = pd.DataFrame({
            "doc_id": ["good", "bad"],
            "member": ["Alice", "Bob"],
            "ticker": ["AAPL", "GOOG"],
            "transaction_date": pd.to_datetime(["2024-06-01", "2024-12-01"]),
            "disclosure_date": pd.to_datetime(["2024-06-05", "2024-06-05"]),
            "transaction_type": ["Purchase", "Purchase"],
        })
        self.db.upsert_transactions(df)

        result = self.db.get_transactions_by_date_range(
            date(2024, 1, 1), date(2024, 12, 31)
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["ticker"], "AAPL")

    def test_empty_range_returns_empty(self):
        df = pd.DataFrame({
            "doc_id": ["d1"],
            "member": ["Alice"],
            "ticker": ["AAPL"],
            "transaction_date": pd.to_datetime(["2024-06-01"]),
            "disclosure_date": pd.to_datetime(["2024-06-05"]),
            "transaction_type": ["Purchase"],
        })
        self.db.upsert_transactions(df)

        result = self.db.get_transactions_by_date_range(
            date(2025, 1, 1), date(2025, 6, 1)
        )
        self.assertTrue(result.empty)



if __name__ == "__main__":
    unittest.main()
