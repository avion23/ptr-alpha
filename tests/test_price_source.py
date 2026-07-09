"""Regression tests for cache/fetch price preparation."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import pandas as pd

from analyzer.price_source import YFinancePriceSource


class TestReadOnlyPriceMerge(unittest.TestCase):
    def test_fetched_prices_fill_and_override_partial_cached_column(self):
        source = object.__new__(YFinancePriceSource)
        source.db = SimpleNamespace(is_read_only=True)
        source._download_yfinance = Mock(return_value=pd.DataFrame(
            {"Close": [101.0, 102.0, 103.0]},
            index=pd.date_range("2024-01-01", periods=3, freq="D"),
        ))

        cached = pd.DataFrame(
            {"AAPL": [100.0]},
            index=pd.to_datetime(["2024-01-01"]),
        )
        prices = source._fetch_and_merge_prices(
            ["AAPL"],
            {"AAPL": "AAPL"},
            {},
            cached,
            pd.Timestamp("2024-01-01").date(),
            pd.Timestamp("2024-01-04").date(),
            ["AAPL"],
        )

        self.assertEqual(prices.loc[pd.Timestamp("2024-01-01"), "AAPL"], 101.0)
        self.assertEqual(prices.loc[pd.Timestamp("2024-01-03"), "AAPL"], 103.0)


if __name__ == "__main__":
    unittest.main()
