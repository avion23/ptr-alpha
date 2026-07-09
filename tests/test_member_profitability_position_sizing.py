"""Smoke tests for member_profitability.position_sizing module."""
import unittest

import pandas as pd


class TestPositionSizingImports(unittest.TestCase):

    def test_module_imports(self):
        import member_profitability.position_sizing
        self.assertTrue(callable(member_profitability.position_sizing.position_sizing_grid_search))


class TestSummarizeGridResult(unittest.TestCase):

    def test_zero_picks_returns_zeroed(self):
        from member_profitability.position_sizing import _summarize_grid_result
        out = _summarize_grid_result(
            window_returns=[], window_wins=0, window_total=0,
            top_n=5, min_buyers=2,
        )
        self.assertEqual(out["total_picks"], 0)
        self.assertEqual(out["avg_spy_alpha_pct"], 0.0)
        self.assertEqual(out["sharpe_proxy"], 0.0)
        self.assertEqual(out["win_rate_pct"], 0)

    def test_single_positive_pick(self):
        from member_profitability.position_sizing import _summarize_grid_result
        out = _summarize_grid_result(
            window_returns=[1.5], window_wins=1, window_total=1,
            top_n=5, min_buyers=2,
        )
        self.assertEqual(out["total_picks"], 1)
        self.assertEqual(out["avg_spy_alpha_pct"], 1.5)
        self.assertEqual(out["win_rate_pct"], 100.0)

    def test_sharpe_zero_for_single_pick(self):
        from member_profitability.position_sizing import _summarize_grid_result
        out = _summarize_grid_result(
            window_returns=[2.0], window_wins=1, window_total=1,
            top_n=5, min_buyers=2,
        )
        # std=0 -> sharpe=0 (avoids divide-by-zero)
        self.assertEqual(out["sharpe_proxy"], 0.0)

    def test_evaluate_grid_counts_picks_and_wins(self):
        from unittest.mock import patch

        from member_profitability.position_sizing import _evaluate_grid

        ticker_scores = [
            {"ticker": "A", "composite_score": 2.0, "test_returns": [3.0, 1.0]},
            {"ticker": "B", "composite_score": 1.0, "test_returns": [-2.0]},
        ]
        nonempty = pd.DataFrame({"signal_type": ["Purchase"]})
        with (
            patch(
                "member_profitability.position_sizing._slice_window",
                return_value=(nonempty, nonempty),
            ),
            patch(
                "member_profitability.position_sizing._rank_train",
                return_value=pd.DataFrame({"member": ["m"], "shrunk_alpha": [1.0]}),
            ),
            patch(
                "member_profitability.position_sizing._score_test_tickers",
                return_value=ticker_scores,
            ),
        ):
            returns, wins, total = _evaluate_grid(pd.DataFrame(), [{}], top_n=2, min_buyers=1)

        self.assertEqual(returns, [2.0, -2.0])
        self.assertEqual(wins, 1)
        self.assertEqual(total, 2)


class TestScoreTestTickers(unittest.TestCase):

    def test_empty_test_sigs_returns_empty(self):
        from member_profitability.position_sizing import _score_test_tickers
        test_sigs = pd.DataFrame(columns=["ticker", "member", "signal_type", "spy_alpha_pct"])
        train_rankings = pd.DataFrame(columns=["member", "shrunk_alpha"])
        result = _score_test_tickers(test_sigs, train_rankings, min_buyers=2)
        self.assertEqual(result, [])

    def test_filters_below_min_buyers(self):
        from member_profitability.position_sizing import _score_test_tickers
        test_sigs = pd.DataFrame({
            "ticker": ["A", "B"],
            "member": ["m1", "m1"],   # single buyer for each ticker
            "signal_type": ["Purchase", "Purchase"],
            "spy_alpha_pct": [2.0, 3.0],
        })
        train_rankings = pd.DataFrame(columns=["member", "shrunk_alpha"])
        # min_buyers=2 -> tickers with only 1 buyer should be filtered
        result = _score_test_tickers(test_sigs, train_rankings, min_buyers=2)
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
