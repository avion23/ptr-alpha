import unittest

import numpy as np
import pandas as pd

from analyzer.analysis import evaluate_backtest
from analyzer.options import UnsupportedOptionPricingError, estimate_options_leverage


class TestBacktestTiming(unittest.TestCase):
    def test_ordinary_entry_and_spy_use_as_of_close(self):
        as_of = pd.Timestamp("2025-01-03")
        dates = pd.date_range("2025-01-01", "2025-01-06", freq="D")
        prices = pd.DataFrame(
            {
                "AAPL": [90.0, 100.0, 200.0, 210.0, 220.0, 230.0],
                "SPY": [300.0, 320.0, 600.0, 660.0, 700.0, 740.0],
            },
            index=dates,
        )

        result = evaluate_backtest(
            pd.DataFrame({"ticker": ["AAPL"]}),
            prices,
            as_of,
            horizon=1,
            entry_slippage_bps=0,
            exit_slippage_bps=0,
        )
        row = result.iloc[0]

        self.assertEqual(row["bt_entry_price"], 200.0)
        self.assertEqual(row["bt_spy_return_pct"], 10.0)

    def test_one_day_holding_period_uses_as_of_and_next_day_prices(self):
        as_of = pd.Timestamp("2025-01-03")
        entry_date = pd.Timestamp("2025-01-03")
        exit_date = pd.Timestamp("2025-01-04")
        prices = pd.DataFrame(
            {
                "AAPL": [200.0, 210.0],
                "SPY": [600.0, 660.0],
            },
            index=pd.DatetimeIndex([entry_date, exit_date]),
        )

        result = evaluate_backtest(
            pd.DataFrame({"ticker": ["AAPL"]}),
            prices,
            as_of,
            horizon=1,
            entry_slippage_bps=0,
            exit_slippage_bps=0,
        )
        row = result.iloc[0]

        self.assertEqual(entry_date, as_of)
        self.assertEqual(exit_date - entry_date, pd.Timedelta(days=1))
        self.assertEqual(row["bt_entry_price"], prices.loc[as_of, "AAPL"])
        self.assertEqual(
            row["bt_exit_price"], prices.loc[as_of + pd.Timedelta(days=1), "AAPL"]
        )
        self.assertEqual(row["bt_raw_return_pct"], 5.0)

    def test_dip_entry_keeps_future_fill_timing(self):
        dates = pd.date_range("2025-01-02", "2025-01-10", freq="B")
        prices = pd.DataFrame(
            {
                "AAPL": [90.0, 100.0, 94.0, 95.0, 96.0, 97.0, 98.0],
                "SPY": [390.0, 400.0, 500.0, 501.0, 502.0, 503.0, 504.0],
            },
            index=dates,
        )

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

    def test_option_returns_require_contract_prices(self):
        for instrument in ("call", "put", "option"):
            with self.subTest(instrument=instrument):
                with self.assertRaises(UnsupportedOptionPricingError):
                    estimate_options_leverage(instrument, 10_000_000)

    def test_stale_exit_keeps_last_price_and_sets_coverage_flags(self):
        dates = pd.date_range("2024-12-01", "2025-02-10", freq="D")
        last_trade = pd.Timestamp("2025-01-10")
        prices = pd.DataFrame(
            {
                "AAPL": [100.0 if day <= last_trade else np.nan for day in dates],
                "SPY": [400.0 + i for i in range(len(dates))],
            },
            index=dates,
        )

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
        self.assertEqual(row["bt_return_pct"], 0.0)
        self.assertEqual(row["bt_alpha_pct"], round(-row["bt_spy_return_pct"], 2))
        self.assertEqual(row["bt_coverage"], "stale")
        self.assertTrue(row["bt_stale_exit"])

    def test_truncated_history_keeps_price_derived_return(self):
        dates = pd.date_range("2024-12-01", "2025-02-10", freq="D")
        as_of = pd.Timestamp("2025-01-01")
        last_trade = pd.Timestamp("2025-01-10")
        prices = pd.DataFrame(
            {
                "AAPL": [
                    100.0 if day <= as_of else 130.0 if day <= last_trade else np.nan
                    for day in dates
                ],
                "SPY": [400.0 + i for i in range(len(dates))],
            },
            index=dates,
        )

        result = evaluate_backtest(
            pd.DataFrame({"ticker": ["AAPL"]}),
            prices,
            as_of,
            horizon=40,
            entry_slippage_bps=0,
            exit_slippage_bps=0,
        )
        row = result.iloc[0]

        self.assertEqual(row["bt_entry_price"], 100.0)
        self.assertEqual(row["bt_exit_price"], 130.0)
        self.assertEqual(row["bt_raw_return_pct"], 30.0)
        self.assertEqual(row["bt_return_pct"], 30.0)
        self.assertEqual(row["bt_coverage"], "stale")
        self.assertTrue(row["bt_stale_exit"])
        self.assertTrue(row["bt_delisted"])

    def test_evaluator_abstains_from_underlying_based_option_returns(self):
        dates = pd.date_range("2024-12-01", "2025-02-10", freq="D")
        prices = pd.DataFrame(
            {
                "CALL": [100.0] * len(dates),
                "SPY": [400.0 + i for i in range(len(dates))],
            },
            index=dates,
        )
        recommendations = pd.DataFrame(
            {"ticker": ["CALL"], "instrument_type": ["call"]}
        )

        with self.assertRaisesRegex(UnsupportedOptionPricingError, "contract prices"):
            evaluate_backtest(
                recommendations,
                prices,
                pd.Timestamp("2025-01-01"),
                horizon=40,
                entry_slippage_bps=0,
                exit_slippage_bps=0,
            )

    def test_exactly_25_days_stale_keeps_price_derived_return(self):
        dates = pd.date_range("2024-12-01", "2025-02-10", freq="D")
        last_trade = pd.Timestamp("2025-01-16")
        prices = pd.DataFrame(
            {
                "AAPL": [
                    100.0
                    if day <= pd.Timestamp("2025-01-01")
                    else 110.0
                    if day <= last_trade
                    else np.nan
                    for day in dates
                ],
                "SPY": [400.0 + i for i in range(len(dates))],
            },
            index=dates,
        )

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
        self.assertEqual(row["bt_coverage"], "complete")
        self.assertFalse(row["bt_stale_exit"])


if __name__ == "__main__":
    unittest.main()
