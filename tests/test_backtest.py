import unittest
from datetime import date
from unittest.mock import patch

import numpy as np
import pandas as pd

from analyzer.analysis import (
    backtest_recommendations as _backtest_recommendations,
    evaluate_backtest,
    summarize_backtest,
)
from analyzer.backtest.recommend import _candidate_tickers, _filter_equity_rows

from .conftest import DatabaseTestCase


def backtest_recommendations(*args, **kwargs):
    """Run legacy ranking tests with an explicit non-default scoring mode."""
    kwargs.setdefault("scoring_mode", "shrunk_alpha")
    return _backtest_recommendations(*args, **kwargs)


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
        "instrument_type": [],
        "ticker_origin": [],
    }
    for row in rows:
        enriched = {"instrument_type": "stock", "ticker_origin": "official"} | row
        for key in base:
            base[key].append(enriched.get(key))
    df = pd.DataFrame(base)
    df["disclosure_date"] = pd.to_datetime(df["disclosure_date"])
    # Synthetic realized labels explicitly record their executable maturity.
    df["label_window_end"] = [
        pd.Timestamp(row["disclosure_date"])
        + pd.Timedelta(days=int(row["horizon_days"]) + 1)
        for row in rows
    ]
    df["window_complete"] = [row.get("window_complete", True) for row in rows]
    return df


def _make_transactions(rows):
    df = pd.DataFrame(rows)
    if "instrument_type" not in df.columns:
        df["instrument_type"] = "stock"
    if "ticker_origin" not in df.columns:
        df["ticker_origin"] = "official"
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    df["disclosure_date"] = pd.to_datetime(df["disclosure_date"])
    return df


class TestBacktestRecommendations(unittest.TestCase):
    def setUp(self):
        self.as_of = pd.Timestamp("2025-01-01")
        horizon = 90
        elapsed_cutoff = self.as_of - pd.Timedelta(days=horizon)

        self.signals = _make_signals(
            [
                {
                    "member": "Alpha",
                    "ticker": "AAPL",
                    "disclosure_date": elapsed_cutoff - pd.Timedelta(days=30),
                    "signal_type": "Purchase",
                    "horizon_days": 90,
                    "entry_price": 100.0,
                    "decayed_return_pct": 20.0,
                    "peak_potential_pct": 30.0,
                    "spy_alpha_pct": 15.0,
                    "total_return_pct": 25.0,
                    "total_spy_alpha_pct": 18.0,
                },
                {
                    "member": "Alpha",
                    "ticker": "MSFT",
                    "disclosure_date": elapsed_cutoff - pd.Timedelta(days=20),
                    "signal_type": "Purchase",
                    "horizon_days": 90,
                    "entry_price": 200.0,
                    "decayed_return_pct": 15.0,
                    "peak_potential_pct": 25.0,
                    "spy_alpha_pct": 10.0,
                    "total_return_pct": 18.0,
                    "total_spy_alpha_pct": 12.0,
                },
                {
                    "member": "Beta",
                    "ticker": "GOOG",
                    "disclosure_date": elapsed_cutoff - pd.Timedelta(days=25),
                    "signal_type": "Purchase",
                    "horizon_days": 90,
                    "entry_price": 50.0,
                    "decayed_return_pct": -10.0,
                    "peak_potential_pct": 5.0,
                    "spy_alpha_pct": -15.0,
                    "total_return_pct": -12.0,
                    "total_spy_alpha_pct": -18.0,
                },
            ]
        )

        self.recent_transactions = _make_transactions(
            [
                {
                    "member": "Alpha",
                    "ticker": "CAND",
                    "transaction_date": "2024-12-10",
                    "disclosure_date": "2024-12-15",
                    "transaction_type": "Purchase",
                },
                {
                    "member": "Beta",
                    "ticker": "CAND",
                    "transaction_date": "2024-12-12",
                    "disclosure_date": "2024-12-17",
                    "transaction_type": "Purchase",
                },
            ]
        )

    @patch("analyzer.backtest.recommend.score_ticker_by_buyers")
    def test_default_consensus_cold_start_passes_absolute_as_of(self, score):
        score.return_value = pd.DataFrame(
            {
                "ticker": ["AAPL"],
                "num_buyers": [2],
                "rated_buyers": [0],
                "signal_score": [1.5],
                "signal_score_raw": [1.5],
                "scoring_mode": ["consensus"],
            }
        )
        recommendations = _backtest_recommendations(
            pd.DataFrame(),
            self.recent_transactions,
            self.as_of,
            horizon=90,
            lookback_days=60,
            min_buyers=2,
        )
        self.assertEqual(recommendations.iloc[0]["scoring_mode"], "consensus")
        kwargs = score.call_args.kwargs
        self.assertEqual(kwargs["scoring_mode"], "consensus")
        self.assertEqual(kwargs["as_of_date"], self.as_of)

    def test_produces_recommendations_for_multi_buyer_ticker(self):
        recs = backtest_recommendations(
            self.signals,
            self.recent_transactions,
            self.as_of,
            horizon=90,
            lookback_days=60,
            min_buyers=2,
            top_n=10,
            threshold=5.0,
            scoring_mode="consensus",
        )
        self.assertFalse(recs.empty)
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs.iloc[0]["ticker"], "CAND")
        self.assertIn("rank", recs.columns)
        self.assertEqual(recs.iloc[0]["rank"], 1)

    def test_returns_empty_when_no_elapsed_training_data(self):
        recent_signals = _make_signals(
            [
                {
                    "member": "Alpha",
                    "ticker": "AAPL",
                    "disclosure_date": self.as_of - pd.Timedelta(days=10),
                    "signal_type": "Purchase",
                    "horizon_days": 90,
                    "entry_price": 100.0,
                    "decayed_return_pct": 20.0,
                    "peak_potential_pct": 30.0,
                    "spy_alpha_pct": 15.0,
                    "total_return_pct": 25.0,
                    "total_spy_alpha_pct": 18.0,
                },
            ]
        )
        recs = backtest_recommendations(
            recent_signals,
            self.recent_transactions,
            self.as_of,
            horizon=90,
            lookback_days=60,
            min_buyers=2,
            top_n=10,
            threshold=5.0,
        )
        self.assertTrue(recs.empty)

    def test_returns_empty_when_no_recent_candidates(self):
        old_transactions = _make_transactions(
            [
                {
                    "member": "Alpha",
                    "ticker": "CAND",
                    "transaction_date": "2024-01-10",
                    "disclosure_date": "2024-01-15",
                    "transaction_type": "Purchase",
                },
                {
                    "member": "Beta",
                    "ticker": "CAND",
                    "transaction_date": "2024-01-12",
                    "disclosure_date": "2024-01-17",
                    "transaction_type": "Purchase",
                },
            ]
        )
        recs = backtest_recommendations(
            self.signals,
            old_transactions,
            self.as_of,
            horizon=90,
            lookback_days=60,
            min_buyers=2,
            top_n=10,
            threshold=5.0,
        )
        self.assertTrue(recs.empty)

    def test_filters_below_min_buyers(self):
        single_buyer = _make_transactions(
            [
                {
                    "member": "Alpha",
                    "ticker": "SOLO",
                    "transaction_date": "2024-12-10",
                    "disclosure_date": "2024-12-15",
                    "transaction_type": "Purchase",
                },
            ]
        )
        recs = backtest_recommendations(
            self.signals,
            single_buyer,
            self.as_of,
            horizon=90,
            lookback_days=60,
            min_buyers=2,
            top_n=10,
            threshold=5.0,
        )
        self.assertTrue(recs.empty)

    def test_no_lookahead_future_signals_excluded_from_training(self):
        signals_with_future = pd.concat(
            [
                self.signals,
                _make_signals(
                    [
                        {
                            "member": "Alpha",
                            "ticker": "FUT",
                            "disclosure_date": self.as_of - pd.Timedelta(days=30),
                            "signal_type": "Purchase",
                            "horizon_days": 90,
                            "entry_price": 100.0,
                            "decayed_return_pct": -50.0,
                            "peak_potential_pct": -40.0,
                            "spy_alpha_pct": -45.0,
                            "total_return_pct": -50.0,
                            "total_spy_alpha_pct": -48.0,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

        recs_elapsed = backtest_recommendations(
            self.signals,
            self.recent_transactions,
            self.as_of,
            horizon=90,
            lookback_days=60,
            min_buyers=2,
            top_n=10,
            threshold=5.0,
            scoring_mode="consensus",
        )
        recs_with_future = backtest_recommendations(
            signals_with_future,
            self.recent_transactions,
            self.as_of,
            horizon=90,
            lookback_days=60,
            min_buyers=2,
            top_n=10,
            threshold=5.0,
            scoring_mode="consensus",
        )

        self.assertEqual(
            recs_elapsed.iloc[0]["signal_score"],
            recs_with_future.iloc[0]["signal_score"],
        )

    def test_no_lookahead_in_ticker_perf_signals(self):
        """Regression: not-elapsed signals must not leak into ticker_perf_signals."""
        as_of = pd.Timestamp("2025-01-01")
        horizon = 90
        elapsed_cutoff = as_of - pd.Timedelta(days=horizon)

        # 3 fully-elapsed signals for Alpha on CAND (enough for TICKER_PERF_MIN_TRADES)
        elapsed_signals = [
            {
                "member": "Alpha",
                "ticker": "CAND",
                "disclosure_date": elapsed_cutoff - pd.Timedelta(days=90),
                "signal_type": "Purchase",
                "horizon_days": 90,
                "entry_price": 50.0,
                "decayed_return_pct": 10.0,
                "peak_potential_pct": 15.0,
                "spy_alpha_pct": 5.0,
                "total_return_pct": 12.0,
                "total_spy_alpha_pct": 8.0,
            },
            {
                "member": "Alpha",
                "ticker": "CAND",
                "disclosure_date": elapsed_cutoff - pd.Timedelta(days=60),
                "signal_type": "Purchase",
                "horizon_days": 90,
                "entry_price": 50.0,
                "decayed_return_pct": -5.0,
                "peak_potential_pct": 5.0,
                "spy_alpha_pct": -3.0,
                "total_return_pct": -4.0,
                "total_spy_alpha_pct": -2.0,
            },
            {
                "member": "Alpha",
                "ticker": "CAND",
                "disclosure_date": elapsed_cutoff - pd.Timedelta(days=30),
                "signal_type": "Purchase",
                "horizon_days": 90,
                "entry_price": 50.0,
                "decayed_return_pct": 8.0,
                "peak_potential_pct": 12.0,
                "spy_alpha_pct": 4.0,
                "total_return_pct": 9.0,
                "total_spy_alpha_pct": 5.0,
            },
        ]
        # Not-elapsed signal: disclosed 30 days before as_of, horizon 90 not yet elapsed
        not_elapsed_signal = {
            "member": "Alpha",
            "ticker": "CAND",
            "disclosure_date": as_of - pd.Timedelta(days=30),
            "signal_type": "Purchase",
            "horizon_days": 90,
            "entry_price": 50.0,
            "decayed_return_pct": 500.0,
            "peak_potential_pct": 600.0,
            "spy_alpha_pct": 500.0,
            "total_return_pct": 500.0,
            "total_spy_alpha_pct": 500.0,
        }

        signals_elapsed_only = _make_signals(elapsed_signals)
        signals_with_leak = _make_signals(elapsed_signals + [not_elapsed_signal])

        recent_txns = _make_transactions(
            [
                {
                    "member": "Alpha",
                    "ticker": "CAND",
                    "transaction_date": "2024-12-10",
                    "disclosure_date": "2024-12-15",
                    "transaction_type": "Purchase",
                },
                {
                    "member": "Beta",
                    "ticker": "CAND",
                    "transaction_date": "2024-12-12",
                    "disclosure_date": "2024-12-17",
                    "transaction_type": "Purchase",
                },
            ]
        )

        recs_elapsed = backtest_recommendations(
            signals_elapsed_only,
            recent_txns,
            as_of,
            horizon=90,
            lookback_days=60,
            min_buyers=2,
            top_n=10,
            threshold=5.0,
        )
        recs_with_leak = backtest_recommendations(
            signals_with_leak,
            recent_txns,
            as_of,
            horizon=90,
            lookback_days=60,
            min_buyers=2,
            top_n=10,
            threshold=5.0,
        )

        self.assertFalse(recs_elapsed.empty)
        self.assertFalse(recs_with_leak.empty)
        self.assertEqual(
            recs_elapsed.iloc[0]["signal_score"],
            recs_with_leak.iloc[0]["signal_score"],
            "Not-elapsed signal leaked into ticker_perf_signals scoring",
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

    def test_truncated_ticker_is_unavailable_without_a_delisting_guess(self):
        full_dates = pd.date_range("2024-12-01", "2025-06-01", freq="D")
        short_end = pd.Timestamp("2025-01-15")
        n_short = len(pd.date_range("2024-12-01", short_end, freq="D"))
        n_full = len(full_dates)
        prices = pd.DataFrame(
            {
                "AAPL": [100.0] * n_short + [np.nan] * (n_full - n_short),
                "GOOG": [200.0] * n_short + [np.nan] * (n_full - n_short),
                "SPY": [400.0 + i * 0.02 for i in range(n_full)],
            },
            index=full_dates,
        )

        result = evaluate_backtest(
            self.recommendations, prices, pd.Timestamp("2025-01-01"), horizon=90
        )

        self.assertEqual(result["bt_return_pct"].notna().sum(), 0)
        self.assertTrue((result["bt_coverage"] == "unavailable").all())
        self.assertFalse(result["bt_delisted"].any())
        self.assertEqual(result.attrs["n_unavailable"], 2)
        self.assertEqual(result.attrs["n_delisted"], 0)

    def test_missing_entry_price_skips_row(self):
        no_aapl = self.prices.drop(columns=["AAPL"])
        result = evaluate_backtest(
            self.recommendations, no_aapl, pd.Timestamp("2025-01-01"), horizon=90
        )
        # AAPL has no price → skipped, GOOG still evaluated
        self.assertEqual(len(result.dropna(subset=["bt_return_pct"])), 1)


class TestSummarizeBacktest(unittest.TestCase):
    def test_equal_funds_once_per_rebalance_date(self):
        results = pd.DataFrame(
            {
                "as_of_date": ["2025-01-01", "2025-01-02", "2025-01-02"],
                "rank": [1, 1, 2],
                "bt_return_pct": [0.0, 10.0, 20.0],
                "bt_alpha_pct": [0.0, 5.0, 15.0],
                "bt_horizon_days": [20, 30, 30],
            }
        )
        summary = summarize_backtest(results)
        portfolio = summary[summary["rank"] == "PORTFOLIO"].iloc[0]
        self.assertEqual(portfolio["count"], 2)
        self.assertEqual(portfolio["recommendation_count"], 3)
        self.assertEqual(portfolio["avg_return_pct"], 7.5)
        self.assertEqual(portfolio["avg_alpha_pct"], 5.0)
        self.assertEqual(portfolio["holding_policy"], "adaptive_20-30d")

    def test_real_spy_buy_hold_uses_one_start_and_end_trade(self):
        results = pd.DataFrame(
            {
                "as_of_date": ["2025-01-01", "2025-01-02", "2025-01-02"],
                "bt_exit_date": ["2025-01-03"] * 3,
                "bt_return_pct": [0.0, 10.0, 20.0],
                "bt_alpha_pct": [0.0, 5.0, 15.0],
            }
        )
        spy = pd.Series(
            [100.0, 110.0, 120.0],
            index=pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]),
        )
        summary = summarize_backtest(
            results, spy, entry_slippage_bps=0, exit_slippage_bps=0
        )
        spy_row = summary[summary["rank"] == "SPY_BUY_HOLD"].iloc[0]
        self.assertEqual(spy_row["count"], 1)
        self.assertEqual(spy_row["avg_return_pct"], 20.0)

    def test_spy_buy_hold_quarantines_unverified_zero_quote(self):
        results = pd.DataFrame(
            {
                "bt_entry_date": ["2025-01-01"],
                "bt_exit_date": ["2025-01-02"],
                "bt_return_pct": [0.0],
                "bt_alpha_pct": [0.0],
            }
        )
        spy = pd.Series(
            [100.0, 0.0],
            index=pd.to_datetime(["2025-01-01", "2025-01-02"]),
        )
        summary = summarize_backtest(
            results, spy, entry_slippage_bps=0, exit_slippage_bps=0
        )
        self.assertNotIn("SPY_BUY_HOLD", summary["rank"].values)
        self.assertEqual(summary.attrs["spy_benchmark_status"], "omitted")
        self.assertEqual(summary.attrs["spy_benchmark_reason"], "spy_exit_nonpositive")

    def test_repeated_spy_windows_are_not_called_buy_hold(self):
        results = pd.DataFrame(
            {
                "as_of_date": ["A", "B", "B", "B", "B", "B"],
                "bt_return_pct": [0.0, 10.0, 10.0, 10.0, 10.0, 10.0],
                "bt_alpha_pct": [0.0] * 6,
                "bt_spy_return_pct": [0.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            }
        )
        summary = summarize_backtest(results)
        self.assertNotIn("SPY_BUY_HOLD", summary["rank"].values)
        portfolio = summary.iloc[0]
        self.assertEqual(portfolio["avg_return_pct"], 5.0)

    def test_empty_results_returns_empty(self):
        results = pd.DataFrame({"bt_return_pct": [], "bt_alpha_pct": []})
        self.assertTrue(summarize_backtest(results).empty)

    def test_all_nan_returns_dropped(self):
        results = pd.DataFrame(
            {"bt_return_pct": [10.0, None], "bt_alpha_pct": [5.0, None]}
        )
        summary = summarize_backtest(results)
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary.iloc[0]["count"], 1)

    def test_spy_buy_hold_omits_unbounded_price_window_with_reason(self):
        results = pd.DataFrame(
            {
                "bt_entry_date": ["2025-01-01"],
                "bt_exit_date": ["2025-01-31"],
                "bt_return_pct": [1.0],
                "bt_alpha_pct": [0.0],
            }
        )
        spy = pd.Series(
            [100.0, 110.0],
            index=pd.to_datetime(["2025-01-10", "2025-01-20"]),
        )
        summary = summarize_backtest(results, spy)
        self.assertNotIn("SPY_BUY_HOLD", summary["rank"].values)
        self.assertEqual(summary.attrs["spy_benchmark_status"], "omitted")
        self.assertEqual(
            summary.attrs["spy_benchmark_reason"], "spy_entry_outside_boundary"
        )


class TestBacktestCorrectnessRegressions(unittest.TestCase):
    """Regression tests for the four backtest-correctness bugs fixed in
    backtest/prices.py, backtest/evaluate.py, and backtest/summary.py.
    Each test proves one specific invariant holds after the fix."""

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _make_prices(dates, tickers_data: dict) -> pd.DataFrame:
        return pd.DataFrame(tickers_data, index=dates)

    @staticmethod
    def _single_rec(ticker="AAPL"):
        return pd.DataFrame(
            {
                "rank": [1],
                "ticker": [ticker],
                "signal_score": [50.0],
                "num_buyers": [2],
            }
        )

    # ---- Bug 1a: default use_dip_entry=False, entry = as_of price

    def test_default_entry_is_next_session_not_same_close_or_dip_search(self):
        as_of = pd.Timestamp("2025-01-02")
        dates = pd.date_range("2024-12-01", "2025-06-01", freq="D")
        prices_arr = [100.0 if d <= as_of else 80.0 for d in dates]
        spy_arr = [400.0 + i * 0.01 for i in range(len(dates))]
        prices = self._make_prices(dates, {"AAPL": prices_arr, "SPY": spy_arr})

        result = evaluate_backtest(self._single_rec(), prices, as_of, horizon=90)

        self.assertFalse(result.dropna(subset=["bt_return_pct"]).empty)
        self.assertEqual(result.iloc[0]["bt_entry_date"], date(2025, 1, 3))
        self.assertAlmostEqual(result.iloc[0]["bt_entry_price"], 80.0, places=1)
        self.assertEqual(result.iloc[0]["bt_entry_delay"], 1)

    def test_dip_entry_skips_position_when_no_dip_occurs(self):
        """Bug 1b: when use_dip_entry=True and no pullback occurs within
        max_wait_days, the position must NOT be taken (no fallback fill)."""
        as_of = pd.Timestamp("2025-01-02")
        dates = pd.date_range("2024-12-01", "2025-06-01", freq="D")
        # Price rises monotonically — no 5% pullback will occur
        prices_arr = [100.0 + i * 0.3 for i in range(len(dates))]
        spy_arr = [400.0 + i * 0.01 for i in range(len(dates))]
        prices = self._make_prices(dates, {"AAPL": prices_arr, "SPY": spy_arr})
        recs = self._single_rec()

        result = evaluate_backtest(
            recs,
            prices,
            as_of,
            horizon=90,
            use_dip_entry=True,
            pullback_pct=0.05,
            max_wait_days=10,
        )
        # No dip → position not taken → bt_return_pct is NaN
        self.assertTrue(
            result.dropna(subset=["bt_return_pct"]).empty,
            "no dip → position must not be taken (no fallback fill)",
        )

    def test_spy_window_aligned_with_dip_entry_date(self):
        """Bug 1b: when use_dip_entry=True the SPY return must cover the same
        calendar window as the position [dip_date, dip_date + horizon], not
        the shifted [as_of, as_of + horizon] window."""
        as_of = pd.Timestamp("2025-01-01")
        horizon = 30
        # Dip occurs on Jan 6 (5 calendar days after Jan 1).
        dip_date = pd.Timestamp("2025-01-06")
        dates = pd.date_range("2024-12-01", "2025-06-01", freq="D")
        prices_arr = [100.0 if d < dip_date else 94.0 for d in dates]  # 6% drop
        # Make SPY non-flat so the two windows give different returns.
        spy_arr = [400.0 + i * 0.5 for i in range(len(dates))]
        prices = self._make_prices(dates, {"AAPL": prices_arr, "SPY": spy_arr})
        recs = self._single_rec()

        result_dip = evaluate_backtest(
            recs,
            prices,
            as_of,
            horizon=horizon,
            use_dip_entry=True,
            pullback_pct=0.05,
            max_wait_days=10,
        )
        valid = result_dip.dropna(subset=["bt_return_pct"])
        # Finding 4 fix: assertFalse instead of skipTest so regressions are visible.
        self.assertFalse(
            valid.empty, "dip should have been triggered by the 6% price drop on Jan 6"
        )

        # entry_delay should be 5 calendar days (Jan 1 → Jan 6)
        self.assertEqual(valid.iloc[0]["bt_entry_delay"], 5)

        # SPY return in the dip window must differ from the as_of window
        # because SPY rises 0.5/day and 5 days of shift matters.
        spy_from_as_of = round(
            (
                spy_arr[dates.get_loc(as_of + pd.Timedelta(days=horizon))]
                / spy_arr[dates.get_loc(as_of)]
                - 1
            )
            * 100,
            2,
        )
        spy_from_dip = valid.iloc[0]["bt_spy_return_pct"]
        # The two windows must be different (SPY is non-flat)
        self.assertNotAlmostEqual(
            spy_from_dip,
            spy_from_as_of,
            places=1,
            msg="SPY return in dip window must differ from as_of window when entry is delayed",
        )

    # ---- Bug 1c: calendar-day delay, not array-row count

    def test_dip_entry_delay_uses_calendar_days_not_trading_rows(self):
        """Bug 1c: entry_delay must be calendar days, not the array-row offset.
        A dip on Monday after a Friday as_of spans 3 calendar days but only
        1 trading row (no weekend rows in the price array)."""
        as_of = pd.Timestamp("2025-01-03")  # Friday
        dip_date = pd.Timestamp("2025-01-06")  # Monday: 3 calendar days, 1 trading row
        # Use only business-day prices so the array index gap is 1 but calendar gap is 3
        bdays = pd.date_range("2024-12-02", "2025-06-01", freq="B")
        prices_arr = [90.0 if d >= dip_date else 100.0 for d in bdays]  # 10% drop
        spy_arr = [400.0 + i * 0.1 for i in range(len(bdays))]
        prices = self._make_prices(bdays, {"AAPL": prices_arr, "SPY": spy_arr})
        recs = self._single_rec()

        result = evaluate_backtest(
            recs,
            prices,
            as_of,
            horizon=90,
            use_dip_entry=True,
            pullback_pct=0.05,
            max_wait_days=10,
        )
        valid = result.dropna(subset=["bt_return_pct"])
        self.assertFalse(
            valid.empty, "dip should have been triggered by the 10% price drop on Jan 6"
        )
        entry_delay = valid.iloc[0]["bt_entry_delay"]
        # Old buggy code: would return 1 (array-row index hits[0])
        # Correct code:   must return 3 (calendar days Jan 3 → Jan 6)
        self.assertEqual(
            entry_delay,
            3,
            "entry_delay must be calendar days (3), not trading-row count (1)",
        )

    # ---- Bug 2: survivorship — delisted ticker included at last price

    def test_no_price_at_all_is_skipped_and_counted(self):
        """Bug 2: a ticker with no price data at all (column absent) must be
        skipped and counted in n_no_price, not n_delisted."""
        as_of = pd.Timestamp("2025-01-01")
        dates = pd.date_range("2024-12-01", "2025-06-01", freq="D")
        spy_arr = [400.0 + i * 0.01 for i in range(len(dates))]
        # AAPL column is absent — no data at all
        prices = self._make_prices(dates, {"SPY": spy_arr})
        recs = self._single_rec("AAPL")

        result = evaluate_backtest(recs, prices, as_of, horizon=90)
        self.assertTrue(
            result.dropna(subset=["bt_return_pct"]).empty,
            "no-price ticker must be skipped",
        )
        self.assertEqual(result.attrs.get("n_no_price", 0), 1)
        self.assertEqual(result.attrs.get("n_delisted", 0), 0)

    def test_truncated_ticker_does_not_invent_last_quote_return(self):
        as_of = pd.Timestamp("2025-01-01")
        dates = pd.date_range("2024-12-01", "2025-06-01", freq="D")
        last_day = pd.Timestamp("2025-01-20")
        n_alive = len(pd.date_range("2024-12-01", last_day, freq="D"))
        aapl_arr = [100.0 + i * 0.1 for i in range(n_alive)] + [
            float("nan")
        ] * (len(dates) - n_alive)
        prices = self._make_prices(
            dates,
            {"AAPL": aapl_arr, "SPY": [400.0 + i * 0.02 for i in range(len(dates))]},
        )

        result = evaluate_backtest(self._single_rec("AAPL"), prices, as_of, horizon=90)
        row = result.iloc[0]

        self.assertTrue(pd.isna(row["bt_return_pct"]))
        self.assertEqual(row["bt_coverage"], "unavailable")
        self.assertFalse(row["bt_delisted"])
        self.assertEqual(result.attrs["n_unavailable"], 1)

    def test_duplicate_ticker_in_recs_produces_two_rows_not_four(self):
        """Bug 3: two recommendations for the same ticker must produce exactly
        two output rows, not 2×2=4 (fan-out from ticker-based merge)."""
        as_of = pd.Timestamp("2025-01-01")
        dates = pd.date_range("2024-12-01", "2025-06-01", freq="D")
        n = len(dates)
        prices = self._make_prices(
            dates,
            {
                "AAPL": [100.0 + i * 0.1 for i in range(n)],
                "SPY": [400.0 + i * 0.02 for i in range(n)],
            },
        )
        # Two separate recommendations for the same ticker
        recs = pd.DataFrame(
            {
                "rank": [1, 2],
                "ticker": ["AAPL", "AAPL"],
                "signal_score": [50.0, 30.0],
                "num_buyers": [3, 2],
            }
        )

        result = evaluate_backtest(recs, prices, as_of, horizon=90)
        self.assertEqual(
            len(result),
            2,
            "duplicate-ticker recommendations must not fan-out: expected 2 rows not 4",
        )
        self.assertEqual(list(result["rank"]), [1, 2])
        # Both rows should have the same bt_entry_price (same ticker, same as_of)
        self.assertEqual(
            result.iloc[0]["bt_entry_price"], result.iloc[1]["bt_entry_price"]
        )

    # ---- Bug 4: SPY baseline in summary, coverage counts

    def test_summary_requires_real_prices_for_spy_buy_hold(self):
        results = pd.DataFrame(
            {
                "as_of_date": ["2025-01-01", "2025-01-02", "2025-01-02"],
                "bt_return_pct": [10.0, -5.0, 20.0],
                "bt_alpha_pct": [5.0, -8.0, 15.0],
                "bt_spy_return_pct": [4.0, 2.0, 3.0],
            }
        )
        summary = summarize_backtest(results)
        self.assertNotIn("SPY_BUY_HOLD", summary["rank"].values)

    def test_summary_propagates_coverage_counts_from_attrs(self):
        """Bug 4: summarize_backtest must propagate n_no_price and n_delisted
        from result.attrs so coverage gaps are visible to callers."""
        results = pd.DataFrame(
            {
                "rank": [1],
                "bt_return_pct": [10.0],
                "bt_alpha_pct": [5.0],
                "bt_spy_return_pct": [3.0],
            }
        )
        results.attrs["n_no_price"] = 5
        results.attrs["n_delisted"] = 2
        results.attrs["n_unavailable"] = 3
        summary = summarize_backtest(results)
        self.assertEqual(summary.attrs.get("n_no_price"), 5)
        self.assertEqual(summary.attrs.get("n_delisted"), 2)
        self.assertEqual(summary.attrs.get("n_unavailable"), 3)

    def test_coverage_counts_flow_end_to_end(self):
        """Unavailable outcomes remain distinct from absent entry prices."""
        as_of = pd.Timestamp("2025-01-01")
        dates = pd.date_range("2024-12-01", "2025-06-01", freq="D")
        n = len(dates)
        n_alive = len(pd.date_range("2024-12-01", "2025-01-20", freq="D"))
        prices = self._make_prices(
            dates,
            {
                "GOOG": [200.0 + i * 0.1 for i in range(n)],
                "AAPL": [100.0] * n_alive + [float("nan")] * (n - n_alive),
                "SPY": [400.0 + i * 0.02 for i in range(n)],
            },
        )
        recs = pd.DataFrame(
            {
                "rank": [1, 2, 3],
                "ticker": ["GOOG", "AAPL", "MSFT"],
                "signal_score": [50.0, 40.0, 30.0],
                "num_buyers": [3, 2, 2],
            }
        )

        ev = evaluate_backtest(recs, prices, as_of, horizon=90)
        summary = summarize_backtest(ev)

        self.assertEqual(summary.attrs.get("n_no_price"), 1)
        self.assertEqual(summary.attrs.get("n_delisted"), 0)
        self.assertEqual(ev.attrs.get("n_unavailable"), 1)

    def test_ticker_without_exact_entry_is_unavailable(self):
        """A stale pre-decision quote cannot establish an executable entry."""
        as_of = pd.Timestamp("2025-01-15")
        # Ticker last traded Dec 31 — 15 days before as_of, within the 30-day
        # entry staleness window so the entry lookup succeeds.  The exit date
        # is 90 days later (~April 15).  The fallback exit price is also Dec 31
        # (same stale price), so entry ≈ exit ≈ pure slippage ≈ -0.2%.
        last_trade = pd.Timestamp("2024-12-31")
        dates = pd.date_range("2024-12-01", "2025-06-01", freq="D")
        n_alive = len(pd.date_range("2024-12-01", last_trade, freq="D"))
        aapl_arr = [100.0] * n_alive + [float("nan")] * (len(dates) - n_alive)
        spy_arr = [400.0 + i * 0.02 for i in range(len(dates))]
        prices = self._make_prices(dates, {"AAPL": aapl_arr, "SPY": spy_arr})
        recs = self._single_rec("AAPL")

        result = evaluate_backtest(recs, prices, as_of, horizon=90)
        # Ticker was already dead at recommendation time → must NOT be included
        self.assertTrue(
            result.dropna(subset=["bt_return_pct"]).empty,
            "ticker delisted before as_of must not be included (not tradeable at rec time)",
        )
        self.assertEqual(result.iloc[0]["bt_coverage"], "unavailable")
        self.assertEqual(result.attrs.get("n_unavailable", 0), 1)
        self.assertEqual(result.attrs.get("n_no_price", 0), 0)
        self.assertEqual(result.attrs.get("n_delisted", 0), 0)

    # ---- Finding 1: multi-date concat must preserve summed coverage counts

    def test_multi_date_concat_preserves_unavailable_coverage_counts(self):
        dates = pd.date_range("2024-12-01", "2025-09-01", freq="D")
        n = len(dates)
        n_alive = len(pd.date_range("2024-12-01", "2025-02-01", freq="D"))
        prices = self._make_prices(
            dates,
            {
                "AAPL": [100.0 + i * 0.05 for i in range(n)],
                "GOOG": [200.0] * n_alive + [float("nan")] * (n - n_alive),
                "SPY": [400.0 + i * 0.02 for i in range(n)],
            },
        )
        recs1 = pd.DataFrame(
            {"rank": [1, 2], "ticker": ["AAPL", "MSFT"], "signal_score": [50.0, 40.0]}
        )
        recs2 = pd.DataFrame(
            {"rank": [1, 2], "ticker": ["AAPL", "GOOG"], "signal_score": [50.0, 40.0]}
        )
        ev1 = evaluate_backtest(recs1, prices, pd.Timestamp("2025-01-01"), horizon=90)
        ev2 = evaluate_backtest(recs2, prices, pd.Timestamp("2025-01-15"), horizon=90)

        self.assertEqual(ev1.attrs["n_no_price"], 1)
        self.assertEqual(ev2.attrs["n_unavailable"], 1)
        self.assertEqual(ev2.attrs["n_delisted"], 0)


class TestDatabaseDateRange(DatabaseTestCase):
    def test_filters_by_date_range(self):
        df = pd.DataFrame(
            {
                "doc_id": ["d1", "d2", "d3"],
                "member": ["Alice", "Bob", "Charlie"],
                "ticker": ["AAPL", "GOOG", "MSFT"],
                "transaction_date": pd.to_datetime(
                    ["2024-01-01", "2024-06-01", "2024-12-01"]
                ),
                "disclosure_date": pd.to_datetime(
                    ["2024-01-05", "2024-06-05", "2024-12-05"]
                ),
                "transaction_type": ["Purchase", "Purchase", "Purchase"],
            }
        )
        self.db.upsert_transactions(df, source="house_pdf")

        result = self.db.get_transactions_by_date_range(
            date(2024, 3, 1), date(2024, 10, 1)
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["ticker"], "GOOG")

    def test_excludes_transaction_date_after_disclosure(self):
        df = pd.DataFrame(
            {
                "doc_id": ["good", "bad"],
                "member": ["Alice", "Bob"],
                "ticker": ["AAPL", "GOOG"],
                "transaction_date": pd.to_datetime(["2024-06-01", "2024-12-01"]),
                "disclosure_date": pd.to_datetime(["2024-06-05", "2024-06-05"]),
                "transaction_type": ["Purchase", "Purchase"],
            }
        )
        self.db.upsert_transactions(df, source="house_pdf")

        result = self.db.get_transactions_by_date_range(
            date(2024, 1, 1), date(2024, 12, 31)
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["ticker"], "AAPL")

    def test_empty_range_returns_empty(self):
        df = pd.DataFrame(
            {
                "doc_id": ["d1"],
                "member": ["Alice"],
                "ticker": ["AAPL"],
                "transaction_date": pd.to_datetime(["2024-06-01"]),
                "disclosure_date": pd.to_datetime(["2024-06-05"]),
                "transaction_type": ["Purchase"],
            }
        )
        self.db.upsert_transactions(df, source="house_pdf")

        result = self.db.get_transactions_by_date_range(
            date(2025, 1, 1), date(2025, 6, 1)
        )
        self.assertTrue(result.empty)


class TestTrainingLookbackDays(unittest.TestCase):
    """Tests that training_lookback_days enforces a rolling window on training signals."""

    def setUp(self):
        self.as_of = pd.Timestamp("2024-06-15")
        horizon = 90
        elapsed_cutoff = self.as_of - pd.Timedelta(days=horizon)

        # Old signals from 2022 — far outside a 365-day lookback from 2024-06-15
        # Uses same members (Alpha, Beta) as recent signals and transactions
        # so they affect member_rankings when included.
        self.old_signals = _make_signals(
            [
                {
                    "member": "Alpha",
                    "ticker": "OLDT",
                    "disclosure_date": "2022-06-01",
                    "signal_type": "Purchase",
                    "horizon_days": 90,
                    "entry_price": 50.0,
                    "decayed_return_pct": 30.0,
                    "peak_potential_pct": 40.0,
                    "spy_alpha_pct": 20.0,
                    "total_return_pct": 35.0,
                    "total_spy_alpha_pct": 25.0,
                },
                {
                    "member": "Alpha",
                    "ticker": "OLDT2",
                    "disclosure_date": "2022-09-01",
                    "signal_type": "Purchase",
                    "horizon_days": 90,
                    "entry_price": 60.0,
                    "decayed_return_pct": 25.0,
                    "peak_potential_pct": 35.0,
                    "spy_alpha_pct": 18.0,
                    "total_return_pct": 28.0,
                    "total_spy_alpha_pct": 20.0,
                },
            ]
        )

        # Recent signals from 2024 — within a 365-day lookback from 2024-06-15
        self.recent_signals = _make_signals(
            [
                {
                    "member": "Alpha",
                    "ticker": "RECN",
                    "disclosure_date": elapsed_cutoff - pd.Timedelta(days=10),
                    "signal_type": "Purchase",
                    "horizon_days": 90,
                    "entry_price": 100.0,
                    "decayed_return_pct": 15.0,
                    "peak_potential_pct": 25.0,
                    "spy_alpha_pct": 10.0,
                    "total_return_pct": 18.0,
                    "total_spy_alpha_pct": 12.0,
                },
                {
                    "member": "Beta",
                    "ticker": "RECN2",
                    "disclosure_date": elapsed_cutoff - pd.Timedelta(days=5),
                    "signal_type": "Purchase",
                    "horizon_days": 90,
                    "entry_price": 80.0,
                    "decayed_return_pct": 12.0,
                    "peak_potential_pct": 20.0,
                    "spy_alpha_pct": 8.0,
                    "total_return_pct": 14.0,
                    "total_spy_alpha_pct": 9.0,
                },
            ]
        )

        self.all_signals = pd.concat(
            [self.old_signals, self.recent_signals], ignore_index=True
        )

        # Recent transactions that create candidate tickers — uses same members
        self.recent_transactions = _make_transactions(
            [
                {
                    "member": "Alpha",
                    "ticker": "CAND",
                    "transaction_date": "2024-06-01",
                    "disclosure_date": "2024-06-05",
                    "transaction_type": "Purchase",
                },
                {
                    "member": "Beta",
                    "ticker": "CAND",
                    "transaction_date": "2024-06-02",
                    "disclosure_date": "2024-06-06",
                    "transaction_type": "Purchase",
                },
            ]
        )

    def test_without_lookback_uses_all_loaded_signals(self):
        """Without training_lookback_days, old signals contribute to rankings."""
        recs = backtest_recommendations(
            self.all_signals,
            self.recent_transactions,
            self.as_of,
            horizon=90,
            lookback_days=60,
            min_buyers=2,
            top_n=10,
            threshold=5.0,
        )
        self.assertFalse(recs.empty)
        # OldMember signals are included in training → they affect member_rankings
        # The result should have a non-zero signal_score shaped by all training data
        score_without = recs.iloc[0]["signal_score"]
        self.assertIsNotNone(score_without)

    def test_with_lookback_excludes_old_signals(self):
        """With training_lookback_days=365, signals from 2022 are excluded."""
        recs = backtest_recommendations(
            self.all_signals,
            self.recent_transactions,
            self.as_of,
            horizon=90,
            lookback_days=60,
            min_buyers=2,
            top_n=10,
            threshold=5.0,
            training_lookback_days=365,
        )
        self.assertFalse(recs.empty)
        score_with = recs.iloc[0]["signal_score"]
        self.assertIsNotNone(score_with)

    def test_lookback_changes_member_rankings(self):
        """Scores differ when old signals are excluded vs included."""
        recs_without = backtest_recommendations(
            self.all_signals,
            self.recent_transactions,
            self.as_of,
            horizon=90,
            lookback_days=60,
            min_buyers=2,
            top_n=10,
            threshold=5.0,
        )
        recs_with = backtest_recommendations(
            self.all_signals,
            self.recent_transactions,
            self.as_of,
            horizon=90,
            lookback_days=60,
            min_buyers=2,
            top_n=10,
            threshold=5.0,
            training_lookback_days=365,
        )
        self.assertFalse(recs_without.empty)
        self.assertFalse(recs_with.empty)
        # The signal_score should differ because member rankings change
        # when old signals are filtered out
        self.assertNotEqual(
            recs_without.iloc[0]["signal_score"],
            recs_with.iloc[0]["signal_score"],
            "signal_score should differ when training_lookback_days filters old signals",
        )

    def test_lookback_only_recent_signals_unchanged(self):
        """When all signals are within the lookback, results match no-lookback."""
        recs_no_lookback = backtest_recommendations(
            self.recent_signals,
            self.recent_transactions,
            self.as_of,
            horizon=90,
            lookback_days=60,
            min_buyers=2,
            top_n=10,
            threshold=5.0,
        )
        recs_with_lookback = backtest_recommendations(
            self.recent_signals,
            self.recent_transactions,
            self.as_of,
            horizon=90,
            lookback_days=60,
            min_buyers=2,
            top_n=10,
            threshold=5.0,
            training_lookback_days=365,
        )
        pd.testing.assert_frame_equal(recs_no_lookback, recs_with_lookback)

    def test_lookback_none_has_no_effect(self):
        """training_lookback_days=None behaves like no filter."""
        recs_none = backtest_recommendations(
            self.all_signals,
            self.recent_transactions,
            self.as_of,
            horizon=90,
            lookback_days=60,
            min_buyers=2,
            top_n=10,
            threshold=5.0,
            training_lookback_days=None,
        )
        recs_default = backtest_recommendations(
            self.all_signals,
            self.recent_transactions,
            self.as_of,
            horizon=90,
            lookback_days=60,
            min_buyers=2,
            top_n=10,
            threshold=5.0,
        )
        pd.testing.assert_frame_equal(recs_none, recs_default)



class TestEquityEligibilityCanaries(unittest.TestCase):
    def test_options_and_known_non_equities_are_rejected(self):
        rows = pd.DataFrame(
            {
                "member": ["A", "B", "C", "D", "E"],
                "ticker": ["AMZN", "MATT", "ALLI", "ARLP", "AAPL"],
                "disclosure_date": pd.to_datetime(["2025-01-14"] * 5),
                "instrument_type": ["option", "stock", "stock", "stock", "stock"],
                "asset_description": [
                    "Amazon (AMZN) [OP]",
                    "Matthews International Mutual Fund [OT]",
                    "Alliant Holdings, LP [OL]",
                    "Alliance Resource Partners (ARLP) [ST]",
                    "Apple (AAPL) [ST]",
                ],
            }
        )
        filtered = _filter_equity_rows(rows)
        self.assertEqual(set(filtered["ticker"]), {"ARLP", "AAPL"})
        # 20034095/20034670 stale ALLI is also quarantined when descriptions are absent.
        stale = pd.DataFrame(
            {
                "member": ["A", "B"],
                "ticker": ["ALLI", "AAPL"],
                "instrument_type": ["stock", "stock"],
                "ticker_origin": ["official", "official"],
                "disclosure_date": pd.to_datetime(["2026-03-02", "2026-03-02"]),
            }
        )
        self.assertEqual(_candidate_tickers(stale, 1), ["AAPL"])

    def test_unverified_fund_and_missing_metadata_are_ineligible(self):
        rows = pd.DataFrame(
            {
                "member": ["A", "B", "C"],
                "ticker": ["VFINX", "NOTREAL", "TECH"],
                "instrument_type": ["stock", "stock", "stock"],
                "ticker_origin": ["official", None, "official"],
                "asset_description": [
                    "Vanguard 500 Index Fund",
                    None,
                    "Bio-Techne Corporation Common Stock [ST]",
                ],
            }
        )
        self.assertEqual(_filter_equity_rows(rows)["ticker"].tolist(), ["TECH"])

    def test_aliases_wait_for_date_aware_price_mapping(self):
        rows = pd.DataFrame(
            {
                "member": ["A"],
                "ticker": ["FB"],
                "instrument_type": ["stock"],
                "ticker_origin": ["official"],
                "asset_description": ["Meta Platforms Common Stock [ST]"],
                "transaction_date": pd.to_datetime(["2023-01-01"]),
            }
        )
        self.assertEqual(_candidate_tickers(rows, 1), [])

    def test_unknown_instrument_abstains_when_column_is_present(self):
        rows = pd.DataFrame(
            {
                "member": ["A", "B"],
                "ticker": ["AAPL", "MSFT"],
                "instrument_type": [None, "stock"],
                "ticker_origin": ["official", None],
            }
        )
        self.assertTrue(_filter_equity_rows(rows).empty)
