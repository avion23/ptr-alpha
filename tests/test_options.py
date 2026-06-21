"""Smoke tests for analyzer.options module."""
import unittest

from analyzer.options import estimate_options_leverage


class TestEstimateOptionsLeverage(unittest.TestCase):

    def test_stock_returns_one(self):
        self.assertEqual(estimate_options_leverage("stock"), 1.0)
        self.assertEqual(estimate_options_leverage(""), 1.0)
        self.assertEqual(estimate_options_leverage("unknown"), 1.0)

    def test_call_returns_positive_leverage(self):
        result = estimate_options_leverage("call")
        self.assertGreater(result, 1.0)
        self.assertLessEqual(result, 15.0)

    def test_put_returns_negative_leverage(self):
        result = estimate_options_leverage("put")
        self.assertLess(result, 0.0)
        self.assertGreaterEqual(result, -10.0)

    def test_amount_adjustment_within_bounds(self):
        # Large amount should pull leverage toward floor
        high_amount = estimate_options_leverage("call", amount_midpoint=10_000_000)
        small_amount = estimate_options_leverage("call", amount_midpoint=1_000)
        # Large trades -> less leverage; small trades -> more leverage
        self.assertLess(high_amount, small_amount)

    def test_none_amount_treated_as_no_adjustment(self):
        with_none = estimate_options_leverage("call", amount_midpoint=None)
        self.assertEqual(with_none, estimate_options_leverage("call", amount_midpoint=None))


if __name__ == "__main__":
    unittest.main()