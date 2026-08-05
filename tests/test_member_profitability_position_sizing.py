"""Smoke tests for member_profitability.position_sizing module."""
import unittest





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





if __name__ == "__main__":
    unittest.main()
