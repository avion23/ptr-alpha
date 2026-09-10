import unittest
import pandas as pd
import numpy as np
from analyzer.analysis import (
    calculate_signal_potential,
    rank_members,
    get_top_signals,
    get_member_signals,
    get_analysis_table,
    score_ticker_by_buyers,
)
from analyzer.exceptions import AnalysisError
from analyzer.models import AnalysisMode

from .conftest import make_entry_prices


class TestAnalysis(unittest.TestCase):
    def setUp(self):
        self.sample_transactions = pd.DataFrame(
            {
                "member": ["Alice", "Bob", "Alice", "Charlie"],
                "ticker": ["AAPL", "GOOGL", "MSFT", "AAPL"],
                "disclosure_date": pd.to_datetime(
                    ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]
                ),
                "transaction_type": ["Purchase", "Sale", "Purchase", "Purchase"],
                "owner_code": [None, None, "DC", None],
                "amount_midpoint": [8000.5, 8000.5, 32500.5, 100000.0],
            }
        )

        dates = pd.date_range("2023-12-15", "2024-05-15", freq="D")
        np.random.seed(42)
        self.sample_prices = pd.DataFrame(
            {
                "AAPL": 100 + np.cumsum(np.random.randn(len(dates)) * 0.5),
                "GOOGL": 2000 + np.cumsum(np.random.randn(len(dates)) * 2),
                "MSFT": 300 + np.cumsum(np.random.randn(len(dates)) * 1),
                "SPY": 400 + np.cumsum(np.random.randn(len(dates)) * 1),
            },
            index=dates,
        )

        self.entry_prices = make_entry_prices(
            self.sample_transactions, self.sample_prices
        )

    def test_calculate_signal_potential_basic(self):
        signals = calculate_signal_potential(
            self.entry_prices, self.sample_prices, [30, 90]
        )

        self.assertFalse(signals.empty)
        self.assertEqual(len(signals), 8)

        required_cols = [
            "member",
            "ticker",
            "disclosure_date",
            "signal_type",
            "horizon_days",
            "entry_price",
            "peak_potential_pct",
        ]
        for col in required_cols:
            self.assertIn(col, signals.columns)

        self.assertTrue(all(h in [30, 90] for h in signals["horizon_days"].unique()))
        self.assertTrue(
            all(st in ["Purchase", "Sale"] for st in signals["signal_type"].unique())
        )
        self.assertIn("owner_code", signals.columns)
        self.assertIn("amount_midpoint", signals.columns)

        self.assertFalse(signals["peak_potential_pct"].isna().any())
        self.assertTrue((signals["entry_price"] > 0).all())

    def test_score_ticker_by_buyers_reports_filing_lag_without_noop_factors(
        self,
    ):
        transactions = pd.DataFrame(
            {
                "member": ["Alice", "Charlie"],
                "ticker": ["AAPL", "AAPL"],
                "transaction_date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                "disclosure_date": pd.to_datetime(["2024-01-03", "2024-01-04"]),
                "transaction_type": ["Purchase", "Purchase"],
                "owner_code": [None, "DC"],
                "amount_midpoint": [100000.0, 100000.0],
            }
        )
        score = score_ticker_by_buyers(
            "AAPL",
            transactions,
            min_buyers=2,
            as_of_date=pd.Timestamp("2024-02-01"),
        )

        self.assertEqual(score.iloc[0]["signal_score"], 2.0)
        self.assertEqual(score.iloc[0]["max_trade_to_disclosure_days"], 2)
        self.assertEqual(score.iloc[0]["median_trade_to_disclosure_days"], 2.0)
        for obsolete in (
            "size_factor",
            "owner_factor",
            "convergence_factor",
            "ticker_perf_factor",
            "avg_buyer_performance",
            "best_buyer_performance",
            "total_buyer_trades",
        ):
            self.assertNotIn(obsolete, score.columns)
        self.assertEqual(
            score.iloc[0]["scorer_provenance"], "identity_free_distinct_buyer_count_v2"
        )

    def test_calculate_signal_potential_empty_input(self):
        with self.assertRaises(AnalysisError):
            calculate_signal_potential(pd.DataFrame(), self.sample_prices)

        with self.assertRaises(AnalysisError):
            calculate_signal_potential(self.entry_prices, pd.DataFrame())

    def test_calculate_signal_potential_missing_columns(self):
        bad_data = self.entry_prices.drop(columns=["ticker"])
        with self.assertRaises(AnalysisError):
            calculate_signal_potential(bad_data, self.sample_prices)

    def test_calculate_signal_potential_purchase_vs_sale(self):
        signals = calculate_signal_potential(
            self.entry_prices, self.sample_prices, [30]
        )

        purchases = signals[signals["signal_type"] == "Purchase"]
        sales = signals[signals["signal_type"] == "Sale"]

        self.assertEqual(len(purchases), 3)
        self.assertEqual(len(sales), 1)

        for _, row in purchases.iterrows():
            self.assertTrue(row["peak_potential_pct"] >= -100)

        for _, row in sales.iterrows():
            self.assertTrue(row["peak_potential_pct"] >= -100)

    def test_rank_members_basic(self):
        signals = calculate_signal_potential(
            self.entry_prices, self.sample_prices, [90]
        )
        rankings = rank_members(signals, horizon=90)

        self.assertFalse(rankings.empty)
        self.assertIn("member", rankings.columns)
        self.assertIn("shrunk_alpha_pct", rankings.columns)
        self.assertIn("purchase_episodes", rankings.columns)

    def test_rank_members_empty_input(self):
        with self.assertRaises(AnalysisError):
            rank_members(pd.DataFrame())

    def test_rank_members_filters_by_horizon(self):
        signals = pd.DataFrame(
            {
                "member": ["Alice", "Alice"],
                "ticker": ["AAPL", "AAPL"],
                "disclosure_date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
                "signal_type": ["Purchase", "Purchase"],
                "horizon_days": [30, 90],
                "window_complete": [True, True],
                "total_return_pct": [-40.0, 50.0],
                "total_spy_alpha_pct": [-45.0, 45.0],
            }
        )

        r30 = rank_members(signals, horizon=30)
        r90 = rank_members(signals, horizon=90)

        self.assertEqual(r30.iloc[0]["avg_spy_alpha_pct"], -45.0)
        self.assertEqual(r90.iloc[0]["avg_spy_alpha_pct"], 45.0)
        self.assertEqual(r30.iloc[0]["purchase_episodes"], 1)
        self.assertEqual(r90.iloc[0]["purchase_episodes"], 1)

    def test_get_top_signals_basic(self):
        signals = calculate_signal_potential(
            self.entry_prices, self.sample_prices, [90]
        )
        top_signals = get_top_signals(signals, horizon=90, top_n=2)

        self.assertFalse(top_signals.empty)
        self.assertLessEqual(len(top_signals), 2)

        for col in ["member", "ticker", "disclosure_date", "total_spy_alpha_pct"]:
            self.assertIn(col, top_signals.columns)

        if len(top_signals) > 1:
            alpha = top_signals["total_spy_alpha_pct"].values
            self.assertTrue((alpha[:-1] >= alpha[1:]).all())

    def test_get_top_signals_empty_input(self):
        with self.assertRaises(AnalysisError):
            get_top_signals(pd.DataFrame())

    def test_get_member_signals_basic(self):
        signals = calculate_signal_potential(
            self.entry_prices, self.sample_prices, [90]
        )
        member_signals = get_member_signals(signals, "Alice", horizon=90, top_n=5)

        self.assertFalse(member_signals.empty)
        if "signal_type" in member_signals.columns:
            self.assertTrue(
                all(s in ["Purchase"] for s in member_signals["signal_type"].unique())
            )

        for col in ["ticker", "disclosure_date", "total_spy_alpha_pct"]:
            self.assertIn(col, member_signals.columns)

    def test_get_member_signals_nonexistent_member(self):
        signals = calculate_signal_potential(
            self.entry_prices, self.sample_prices, [90]
        )
        with self.assertRaises(AnalysisError):
            get_member_signals(signals, "NonExistent", horizon=90, top_n=5)

    def test_get_analysis_table_member_filter(self):
        signals = calculate_signal_potential(
            self.entry_prices, self.sample_prices, [90]
        )
        table = get_analysis_table(
            signals, AnalysisMode.MEMBER_SIGNALS, "Alice", 90, 5
        )

        self.assertFalse(table.empty)
        self.assertIn("ticker", table.columns)

    def test_get_analysis_table_member_mode_requires_member(self):
        signals = calculate_signal_potential(
            self.entry_prices, self.sample_prices, [90]
        )

        with self.assertRaisesRegex(ValueError, "member_filter is required"):
            get_analysis_table(signals, AnalysisMode.MEMBER_SIGNALS, None, 90, 5)

    def test_get_analysis_table_top_signals(self):
        signals = calculate_signal_potential(
            self.entry_prices, self.sample_prices, [90]
        )
        table = get_analysis_table(signals, AnalysisMode.TOP_SIGNALS, None, 90, 5)

        self.assertFalse(table.empty)
        for col in ["member", "ticker", "disclosure_date", "total_spy_alpha_pct"]:
            self.assertIn(col, table.columns)

    def test_get_analysis_table_rank_members(self):
        signals = calculate_signal_potential(
            self.entry_prices, self.sample_prices, [90]
        )
        table = get_analysis_table(
            signals, AnalysisMode.MEMBER_RANKINGS, None, 90, 1
        )

        self.assertFalse(table.empty)
        self.assertTrue("member" in table.columns)
        self.assertEqual(len(table), 1)

    def test_score_ticker_by_buyers_canonicalizes_buyer_identity_for_gate(self):
        transactions = pd.DataFrame(
            {
                "member": [
                    "Donald Sternoff Beyer",
                    "Donald Sternoff Honorable Beyer",
                    "Tim Moore",
                    "Tim Moore",
                ],
                "ticker": ["AAPL"] * 4,
                "transaction_date": pd.to_datetime(["2024-01-01"] * 4),
                "disclosure_date": pd.to_datetime(
                    [
                        "2024-01-02",
                        "2024-01-03",
                        "2024-01-04",
                        "2024-01-05",
                    ]
                ),
                "transaction_type": ["Purchase"] * 4,
            }
        )
        score = score_ticker_by_buyers(
            "AAPL",
            transactions,
            min_buyers=3,
            as_of_date=pd.Timestamp("2024-02-01"),
        )

        self.assertEqual(score.iloc[0]["num_buyers"], 2)
        self.assertIn("minimum buyer threshold", score.iloc[0]["note"])

    def test_rank_members_skips_members_with_all_nan_returns(self):
        signals = pd.DataFrame(
            {
                "member": ["Alice", "Bob"],
                "ticker": ["AAPL", "GOOGL"],
                "disclosure_date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                "signal_type": ["Purchase", "Purchase"],
                "horizon_days": [90, 90],
                "window_complete": [True, True],
                "total_return_pct": [12.0, float("nan")],
                "total_spy_alpha_pct": [10.0, float("nan")],
            }
        )

        rankings = rank_members(signals, horizon=90)

        self.assertEqual(len(rankings), 1)
        self.assertEqual(rankings.iloc[0]["member"], "Alice")
        self.assertFalse(np.isnan(rankings.iloc[0]["avg_spy_alpha_pct"]))

    def test_missing_price_windows_do_not_count_as_zero_return_trades(self):
        entry_prices = pd.DataFrame(
            {
                "member": ["Alice", "Alice"],
                "ticker": ["AAPL", "MSFT"],
                "disclosure_date": pd.to_datetime(["2024-01-01", "2024-06-01"]),
                "transaction_type": ["Purchase", "Purchase"],
                "entry_price": [100.0, 200.0],
            }
        )
        price_dates = pd.date_range("2024-01-01", "2024-02-05", freq="D")
        prices = pd.DataFrame(
            {
                "AAPL": np.linspace(100.0, 110.0, len(price_dates)),
                "MSFT": [np.nan] * len(price_dates),
                "SPY": [100.0] * len(price_dates),
            },
            index=price_dates,
        )

        signals = calculate_signal_potential(entry_prices, prices, [30])
        rankings = rank_members(signals, horizon=30)

        self.assertTrue(
            np.isnan(
                signals.loc[signals["ticker"] == "MSFT", "decayed_return_pct"].iloc[0]
            )
        )
        self.assertEqual(rankings.iloc[0]["purchase_episodes"], 1)

    def test_sale_peak_potential_nan_with_incomplete_ticker_coverage(self):
        transactions = pd.DataFrame(
            {
                "member": ["Alice"],
                "ticker": ["AAPL"],
                "disclosure_date": pd.to_datetime(["2024-01-15"]),
                "transaction_type": ["Sale"],
                "owner_code": [None],
                "amount_midpoint": [50000.0],
            }
        )

        dates = pd.date_range("2023-12-15", "2024-02-15", freq="D")
        np.random.seed(99)
        prices = pd.DataFrame(
            {
                "AAPL": 100 + np.cumsum(np.random.randn(len(dates)) * 0.5),
            },
            index=dates,
        )

        entry_prices = make_entry_prices(transactions, prices)
        signals = calculate_signal_potential(entry_prices, prices, [90])

        sales = signals[signals["signal_type"] == "Sale"]
        self.assertEqual(len(sales), 1)
        self.assertTrue(np.isnan(sales.iloc[0]["peak_potential_pct"]))
        self.assertFalse(sales.iloc[0]["window_complete"])

    def test_total_spy_alpha_uses_actual_spy_return(self):
        dates = pd.date_range("2024-01-01", "2024-04-01", freq="D")
        entry_prices = pd.DataFrame(
            {
                "member": ["Alice"],
                "ticker": ["AAPL"],
                "disclosure_date": pd.to_datetime(["2024-01-01"]),
                "transaction_type": ["Purchase"],
                "entry_price": [150.0],
            }
        )
        # AAPL goes 150 -> 165, SPY goes 400 -> 420 over 91 days
        prices = pd.DataFrame(
            {
                "AAPL": np.linspace(150, 165, len(dates)),
                "SPY": np.linspace(400, 420, len(dates)),
            },
            index=dates,
        )

        signals = calculate_signal_potential(entry_prices, prices, [30])

        self.assertEqual(len(signals), 1)
        row = signals.iloc[0]

        # Over 30-day window: AAPL and SPY prices at day 30
        horizon_days = 30
        spy_entry = 400.0
        spy_exit = 400.0 + (420.0 - 400.0) * horizon_days / (len(dates) - 1)
        aapl_exit = 150.0 + (165.0 - 150.0) * horizon_days / (len(dates) - 1)
        actual_spy_return_pct = (spy_exit / spy_entry - 1) * 100
        total_return_pct = (aapl_exit / 150.0 - 1) * 100
        expected_alpha = total_return_pct - actual_spy_return_pct

        self.assertAlmostEqual(row["total_spy_alpha_pct"], expected_alpha, places=2)

    def test_decayed_spy_return_pct_column_present(self):
        entry_prices = pd.DataFrame(
            {
                "member": ["Alice"],
                "ticker": ["AAPL"],
                "disclosure_date": pd.to_datetime(["2024-01-01"]),
                "transaction_type": ["Purchase"],
                "entry_price": [100.0],
            }
        )
        dates = pd.date_range("2024-01-01", "2024-04-01", freq="D")
        prices = pd.DataFrame(
            {
                "AAPL": 100 + np.cumsum(np.random.randn(len(dates)) * 0.5),
                "SPY": 400 + np.cumsum(np.random.randn(len(dates)) * 1),
            },
            index=dates,
        )

        signals = calculate_signal_potential(entry_prices, prices, [30])
        self.assertIn("decayed_spy_return_pct", signals.columns)
        self.assertIn("total_spy_alpha_pct", signals.columns)


class TestEpisodeCollapse(unittest.TestCase):
    def test_rank_members_deduplicates_only_same_public_event(self):
        signals = pd.DataFrame(
            {
                "member": ["Alice"] * 4,
                "ticker": ["AAPL", "AAPL", "AAPL", "MSFT"],
                "signal_type": ["Purchase"] * 4,
                "horizon_days": [90] * 4,
                "window_complete": [True] * 4,
                "disclosure_date": pd.to_datetime(
                    ["2024-01-01", "2024-01-01", "2024-01-10", "2024-01-02"]
                ),
                "total_return_pct": [10.0, 10.0, 8.0, 5.0],
                "total_spy_alpha_pct": [5.0, 5.0, 3.0, 2.0],
            }
        )
        rankings = rank_members(signals, horizon=90)
        self.assertEqual(rankings.iloc[0]["purchase_episodes"], 3)


class TestSoloBuyerConsensusScoring(unittest.TestCase):
    def _solo_transaction(self):
        return pd.DataFrame(
            {
                "member": ["Pelosi"],
                "ticker": ["AVGO"],
                "transaction_date": pd.to_datetime(["2024-01-01"]),
                "disclosure_date": pd.to_datetime(["2024-01-03"]),
                "transaction_type": ["Purchase"],
            }
        )

    def test_single_buyer_score_is_one_when_threshold_allows_it(self):
        score = score_ticker_by_buyers(
            "AVGO",
            self._solo_transaction(),
            min_buyers=1,
            as_of_date=pd.Timestamp("2024-02-01"),
        )
        self.assertEqual(score.iloc[0]["signal_score"], 1.0)

    def test_minimum_distinct_buyer_gate_remains(self):
        score = score_ticker_by_buyers(
            "AVGO",
            self._solo_transaction(),
            min_buyers=2,
            as_of_date=pd.Timestamp("2024-02-01"),
        )

        self.assertEqual(score.iloc[0]["signal_score"], 0.0)
        self.assertIn("minimum buyer threshold", score.iloc[0]["note"])


if __name__ == "__main__":
    unittest.main()
