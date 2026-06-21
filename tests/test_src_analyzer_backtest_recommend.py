"""Smoke tests for analyzer.backtest.recommend submodule."""
import unittest

import pandas as pd


class TestRecommendImports(unittest.TestCase):

    def test_module_imports(self):
        import analyzer.backtest.recommend
        self.assertTrue(callable(analyzer.backtest.recommend.backtest_recommendations))

    def test_private_helpers_callable(self):
        import analyzer.backtest.recommend as r
        self.assertTrue(callable(r._build_member_rankings))
        self.assertTrue(callable(r._candidate_tickers))
        self.assertTrue(callable(r._score_and_rank))
        self.assertTrue(callable(r._build_metadata_maps))
        self.assertTrue(callable(r._score_one_ticker))
        self.assertTrue(callable(r._apply_features_to_row))


class TestCandidateTickers(unittest.TestCase):

    def test_returns_only_min_buyers(self):
        from analyzer.backtest.recommend import _candidate_tickers
        recent = pd.DataFrame({
            "ticker": ["A", "A", "B", "C", "C", "C"],
            "member": ["m1", "m2", "m1", "m1", "m2", "m3"],
        })
        result = _candidate_tickers(recent, min_buyers=2)
        self.assertEqual(set(result), {"A", "C"})

    def test_empty_returns_empty_list(self):
        from analyzer.backtest.recommend import _candidate_tickers
        result = _candidate_tickers(pd.DataFrame(columns=["ticker", "member"]), min_buyers=2)
        self.assertEqual(result, [])


class TestBuildMetadataMaps(unittest.TestCase):

    def test_no_prices_returns_empty_maps(self):
        from analyzer.backtest.recommend import _build_metadata_maps
        recent = pd.DataFrame({"ticker": ["A", "B"], "member": ["m1", "m2"]})
        inst_map, amt_map = _build_metadata_maps(recent, has_prices=False)
        self.assertEqual(inst_map, {})
        self.assertEqual(amt_map, {})

    def test_with_prices_maps_instrument_and_amount(self):
        from analyzer.backtest.recommend import _build_metadata_maps
        recent = pd.DataFrame({
            "ticker": ["A", "B", "A"],
            "member": ["m1", "m2", "m3"],
            "instrument_type": ["stock", "option", "stock"],
            "amount_midpoint": [50000, 25000, 75000],
        })
        inst_map, amt_map = _build_metadata_maps(recent, has_prices=True)
        # drop_duplicates keeps the first occurrence per ticker
        self.assertEqual(inst_map, {"A": "stock", "B": "option"})
        self.assertEqual(amt_map, {"A": 50000, "B": 25000})


if __name__ == "__main__":
    unittest.main()

