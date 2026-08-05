"""Regression tests for cache/fetch price preparation."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import pandas as pd

from analyzer.database import Database
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
            list(pd.date_range("2024-01-01", "2024-01-03")),
        )

        self.assertEqual(prices.loc[pd.Timestamp("2024-01-01"), "AAPL"], 101.0)
        self.assertEqual(prices.loc[pd.Timestamp("2024-01-03"), "AAPL"], 103.0)


def test_recent_gap_fetches_only_missing_span(tmp_path, monkeypatch):
    db = Database(tmp_path / "prices.duckdb")
    try:
        trading_dates = pd.to_datetime(
            ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05",
             "2024-01-08", "2024-01-09", "2024-01-10"]
        )
        db.upsert_prices(
            pd.DataFrame(
                {
                    "AAPL": [100.0, 101.0, 102.0, 103.0, 104.0, None, None],
                    "SPY": range(len(trading_dates)),
                },
                index=trading_dates,
            )
        )

        source = object.__new__(YFinancePriceSource)
        source.db = db
        calls = []

        def fake_download(tickers, start, end):
            calls.append((tickers, start, end))
            return pd.DataFrame(
                {"Close": [105.0, 106.0]},
                index=pd.to_datetime(["2024-01-09", "2024-01-10"]),
            )

        monkeypatch.setattr(source, "_download_yfinance", fake_download)

        prices = source.get_prices(
            ["AAPL"], pd.Timestamp("2024-01-01").date(),
            pd.Timestamp("2024-01-10").date()
        )

        assert calls == [
            (["AAPL"], pd.Timestamp("2024-01-09").date(),
             pd.Timestamp("2024-01-11").date())
        ]
        assert prices.loc[pd.Timestamp("2024-01-10"), "AAPL"] == 106.0
    finally:
        db.close()


if __name__ == "__main__":
    unittest.main()
