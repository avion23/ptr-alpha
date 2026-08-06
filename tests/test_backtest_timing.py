import unittest

import numpy as np
import pandas as pd

from analyzer.backtest.filters import _filter_recent_trades
from analyzer.analysis import evaluate_backtest
from analyzer.options import estimate_options_leverage


class TestBacktestTiming(unittest.TestCase):

    def test_recent_trade_window_includes_lower_bound_and_excludes_as_of(self):
        as_of = pd.Timestamp("2025-01-10")
        transactions = pd.DataFrame({
            "disclosure_date": pd.to_datetime([
                "2024-12-11",  # lookback lower bound
                "2024-12-12",
                "2025-01-10",  # exact as_of
            ]),
            "transaction_type": ["Purchase"] * 3,
        })

        result = _filter_recent_trades(transactions, lookback_days=30, as_of_iso=as_of.isoformat())

        self.assertEqual(
            list(result["disclosure_date"]),
            [pd.Timestamp("2024-12-11"), pd.Timestamp("2024-12-12")],
        )

    def test_ordinary_entry_and_spy_use_prior_close(self):
        dates = pd.date_range("2025-01-01", "2025-01-06", freq="D")
        prices = pd.DataFrame({
            "AAPL": [90.0, 100.0, 200.0, 210.0, 220.0, 230.0],
            "SPY": [300.0, 320.0, 600.0, 660.0, 700.0, 740.0],
        }, index=dates)

        result = evaluate_backtest(
            pd.DataFrame({"ticker": ["AAPL"]}),
            prices,
            pd.Timestamp("2025-01-03"),
            horizon=1,
            entry_slippage_bps=0,
            exit_slippage_bps=0,
        )
        row = result.iloc[0]

        self.assertEqual(row["bt_entry_price"], 100.0)
        self.assertAlmostEqual(row["bt_spy_return_pct"], 106.25, places=2)

    def test_dip_entry_keeps_future_fill_timing(self):
        dates = pd.date_range("2025-01-02", "2025-01-10", freq="B")
        prices = pd.DataFrame({
            "AAPL": [90.0, 100.0, 94.0, 95.0, 96.0, 97.0, 98.0],
            "SPY": [390.0, 400.0, 500.0, 501.0, 502.0, 503.0, 504.0],
        }, index=dates)

        result = evaluate_backtest(
            pd.DataFrame({"ticker": ["AAPL"]}),
            prices,
            pd.Timestamp("2025-01-03"),
            horizon=1,
            use_dip_entry=True,
            entry_slippage_bps=0,
            exit_slippage_bps=0,
        )
        row = result.iloc[0]

        self.assertEqual(row["bt_entry_price"], 94.0)
        self.assertEqual(row["bt_entry_delay"], 3)
        self.assertAlmostEqual(row["bt_spy_return_pct"], 0.2, places=2)

    def test_option_bases_and_amount_clamp(self):
        self.assertEqual(estimate_options_leverage("call"), 4.0)
        self.assertEqual(estimate_options_leverage("put"), -2.0)
        self.assertEqual(estimate_options_leverage("call", 10_000_000), 2.8)
        self.assertEqual(estimate_options_leverage("put", 10_000_000), -1.4)

    def test_stale_delisting_keeps_last_price_but_forces_loss(self):
        dates = pd.date_range("2024-12-01", "2025-02-10", freq="D")
        last_trade = pd.Timestamp("2025-01-10")
        prices = pd.DataFrame({
            "AAPL": [100.0 if day <= last_trade else np.nan for day in dates],
            "SPY": [400.0 + i for i in range(len(dates))],
        }, index=dates)

        result = evaluate_backtest(
            pd.DataFrame({"ticker": ["AAPL"]}),
            prices,
            pd.Timestamp("2025-01-01"),
            horizon=40,
            entry_slippage_bps=0,
            exit_slippage_bps=0,
        )
        row = result.iloc[0]

        self.assertEqual(row["bt_exit_price"], 100.0)
        self.assertTrue(row["bt_delisted"])
        self.assertEqual(row["bt_raw_return_pct"], 0.0)
        self.assertEqual(row["bt_return_pct"], -100.0)
        self.assertEqual(row["bt_alpha_pct"], round(-100.0 - row["bt_spy_return_pct"], 2))

    def test_exactly_25_days_stale_keeps_price_derived_return(self):
        dates = pd.date_range("2024-12-01", "2025-02-10", freq="D")
        last_trade = pd.Timestamp("2025-01-16")
        prices = pd.DataFrame({
            "AAPL": [100.0 if day < pd.Timestamp("2025-01-01") else
                     110.0 if day <= last_trade else np.nan for day in dates],
            "SPY": [400.0 + i for i in range(len(dates))],
        }, index=dates)

        result = evaluate_backtest(
            pd.DataFrame({"ticker": ["AAPL"]}),
            prices,
            pd.Timestamp("2025-01-01"),
            horizon=40,
            entry_slippage_bps=0,
            exit_slippage_bps=0,
        )
        row = result.iloc[0]

        self.assertEqual(row["bt_exit_price"], 110.0)
        self.assertFalse(row["bt_delisted"])
        self.assertEqual(row["bt_return_pct"], 10.0)


if __name__ == "__main__":
    unittest.main()
