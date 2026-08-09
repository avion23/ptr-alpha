import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from analyzer.analysis import calculate_signal_potential, evaluate_backtest
from analyzer.backtest.curves import _build_curves_for_rows
from analyzer.backtest.filters import _filter_ticker_perf, _filter_training
from analyzer.backtest.prices import (
    _aligned_price_at_or_before_arrays,
    _next_tradable_price_arrays,
)
from analyzer.options import UnsupportedOptionPricingError, estimate_options_leverage
from analyzer.pipeline import _entry_prices_from_matrix
from analyzer.price_source import _validate_and_log_prices
from analyzer.signals import _price_arrays


class TestBacktestTiming(unittest.TestCase):
    def test_ordinary_entry_uses_next_session_and_aligned_spy_dates(self):
        as_of = pd.Timestamp("2025-01-03")
        dates = pd.date_range("2025-01-01", "2025-01-07", freq="D")
        prices = pd.DataFrame(
            {
                "AAPL": [90.0, 100.0, 200.0, 210.0, 220.0, 230.0, 240.0],
                "SPY": [300.0, 320.0, 600.0, 660.0, 700.0, 740.0, 760.0],
            },
            index=dates,
        )

        row = evaluate_backtest(
            pd.DataFrame({"ticker": ["AAPL"]}),
            prices,
            as_of,
            horizon=1,
            entry_slippage_bps=0,
            exit_slippage_bps=0,
        ).iloc[0]

        self.assertEqual(row["bt_entry_date"], pd.Timestamp("2025-01-06").date())
        self.assertEqual(row["bt_entry_price"], 230.0)
        self.assertEqual(row["bt_exit_date"], pd.Timestamp("2025-01-07").date())
        self.assertEqual(row["bt_exit_price"], 240.0)
        self.assertEqual(row["bt_spy_return_pct"], round((760 / 740 - 1) * 100, 2))

    def test_horizon_starts_at_actual_entry_session(self):
        as_of = pd.Timestamp("2025-01-03")
        dates = pd.date_range("2025-01-03", "2025-01-07", freq="D")
        prices = pd.DataFrame(
            {
                "AAPL": [200.0, 210.0, 220.0, 230.0, 240.0],
                "SPY": [600.0, 660.0, 700.0, 740.0, 760.0],
            },
            index=dates,
        )

        row = evaluate_backtest(
            pd.DataFrame({"ticker": ["AAPL"]}),
            prices,
            as_of,
            horizon=1,
            entry_slippage_bps=0,
            exit_slippage_bps=0,
        ).iloc[0]

        self.assertEqual(row["bt_entry_date"], pd.Timestamp("2025-01-06").date())
        self.assertEqual(row["bt_exit_date"], pd.Timestamp("2025-01-07").date())
        self.assertEqual(row["bt_raw_return_pct"], round((240 / 230 - 1) * 100, 2))

    def test_dip_entry_keeps_future_fill_timing(self):
        dates = pd.date_range("2025-01-02", "2025-01-10", freq="B")
        prices = pd.DataFrame(
            {
                "AAPL": [90.0, 100.0, 94.0, 95.0, 96.0, 97.0, 98.0],
                "SPY": [390.0, 400.0, 500.0, 501.0, 502.0, 503.0, 504.0],
            },
            index=dates,
        )

        row = evaluate_backtest(
            pd.DataFrame({"ticker": ["AAPL"]}),
            prices,
            pd.Timestamp("2025-01-03"),
            horizon=1,
            use_dip_entry=True,
            entry_slippage_bps=0,
            exit_slippage_bps=0,
        ).iloc[0]

        self.assertEqual(row["bt_entry_price"], 94.0)
        self.assertEqual(row["bt_entry_delay"], 3)
        self.assertEqual(row["bt_entry_date"], pd.Timestamp("2025-01-06").date())
        self.assertAlmostEqual(row["bt_spy_return_pct"], 0.2, places=2)

    def test_option_returns_require_contract_prices(self):
        for instrument in ("call", "put", "option"):
            with self.subTest(instrument=instrument):
                with self.assertRaises(UnsupportedOptionPricingError):
                    estimate_options_leverage(instrument, 10_000_000)


    def test_truncated_history_is_unavailable_not_an_invented_delisting_return(self):
        dates = pd.date_range("2024-12-01", "2025-02-15", freq="D")
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

        self.assertTrue(np.isnan(row["bt_exit_price"]))
        self.assertTrue(np.isnan(row["bt_return_pct"]))
        self.assertFalse(row["bt_delisted"])
        self.assertEqual(row["bt_coverage"], "unavailable")
        self.assertTrue(row["bt_stale_exit"])
        self.assertEqual(result.attrs["n_unavailable"], 1)

    def test_weekend_exit_uses_bounded_prior_common_session(self):
        dates = pd.bdate_range("2025-01-02", "2025-01-10")
        prices = pd.DataFrame(
            {"AAPL": np.arange(len(dates)) + 100.0, "SPY": np.arange(len(dates)) + 400.0},
            index=dates,
        )

        row = evaluate_backtest(
            pd.DataFrame({"ticker": ["AAPL"]}),
            prices,
            pd.Timestamp("2025-01-02"),
            horizon=2,
            entry_slippage_bps=0,
            exit_slippage_bps=0,
        ).iloc[0]

        self.assertEqual(row["bt_entry_date"], pd.Timestamp("2025-01-03").date())
        self.assertEqual(row["bt_exit_date"], pd.Timestamp("2025-01-03").date())
        self.assertEqual(row["bt_coverage"], "complete")

    def test_missing_spy_on_security_exit_is_unavailable(self):
        dates = pd.date_range("2025-01-01", "2025-01-04", freq="D")
        prices = pd.DataFrame(
            {"AAPL": [100.0, 101.0, 102.0, 103.0], "SPY": [400.0, 401.0, np.nan, 403.0]},
            index=dates,
        )

        row = evaluate_backtest(
            pd.DataFrame({"ticker": ["AAPL"]}),
            prices,
            pd.Timestamp("2025-01-01"),
            horizon=1,
            entry_slippage_bps=0,
            exit_slippage_bps=0,
        ).iloc[0]

        self.assertEqual(row["bt_coverage"], "unavailable")
        self.assertEqual(row["bt_unavailable_reason"], "benchmark_quote_unavailable")
        self.assertFalse(row["bt_stale_exit"])
        self.assertTrue(np.isnan(row["bt_alpha_pct"]))

    def test_evaluator_isolates_underlying_based_option_returns(self):
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

        row = evaluate_backtest(
            recommendations,
            prices,
            pd.Timestamp("2025-01-01"),
            horizon=40,
            entry_slippage_bps=0,
            exit_slippage_bps=0,
        ).iloc[0]

        self.assertEqual(row["bt_coverage"], "unavailable")
        self.assertEqual(
            row["bt_unavailable_reason"], "unsupported_instrument_pricing"
        )
        self.assertTrue(np.isnan(row["bt_return_pct"]))

    def test_exactly_25_days_stale_is_unavailable(self):
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

        self.assertTrue(np.isnan(row["bt_exit_price"]))
        self.assertFalse(row["bt_delisted"])
        self.assertTrue(np.isnan(row["bt_return_pct"]))
        self.assertEqual(row["bt_coverage"], "unavailable")
        self.assertTrue(row["bt_stale_exit"])

    def test_training_maturity_uses_actual_label_window_end(self):
        signals = pd.DataFrame(
            {
                "horizon_days": [30],
                "disclosure_date": [pd.Timestamp("2024-01-01")],
                "label_window_end": [pd.Timestamp("2024-02-05")],
                "window_complete": [True],
                "total_spy_alpha_pct": [1.0],
            }
        )

        early = _filter_training(signals, 30, "2024-02-01", None)
        mature = _filter_training(signals, 30, "2024-02-06", None)
        ticker_early = _filter_ticker_perf(signals, 30, "2024-02-01")

        self.assertTrue(early.empty)
        self.assertTrue(ticker_early.empty)
        self.assertEqual(len(mature), 1)

    def test_sale_peak_uses_executable_next_session_basis(self):
        dates = pd.bdate_range("2024-01-08", "2024-01-12")
        prices = pd.DataFrame(
            {
                "SALE": [190.0, 200.0, 150.0, 180.0, 180.0],
                "SPY": [400.0] * len(dates),
            },
            index=dates,
        )
        entries = pd.DataFrame(
            {
                "member": ["Alice"],
                "ticker": ["SALE"],
                "disclosure_date": [pd.Timestamp("2024-01-08")],
                "transaction_type": ["Sale"],
                "entry_price": [100.0],
            }
        )

        row = calculate_signal_potential(entries, prices, [2]).iloc[0]

        self.assertEqual(row["label_entry_date"], pd.Timestamp("2024-01-09"))
        self.assertAlmostEqual(row["peak_potential_pct"], (200 / 150 - 1) * 100)

    def test_fresh_matrix_constructs_entry_without_database_cache(self):
        transactions = pd.DataFrame(
            {
                "member": ["Alice"],
                "ticker": ["AAPL"],
                "transaction_date": [pd.Timestamp("2024-01-02")],
                "disclosure_date": [pd.Timestamp("2024-01-02")],
                "transaction_type": ["Purchase"],
            }
        )
        prices = pd.DataFrame(
            {"AAPL": [999.0], "SPY": [400.0]},
            index=pd.DatetimeIndex(["2024-01-03"]),
        )

        entries = _entry_prices_from_matrix(transactions, prices)

        self.assertEqual(len(entries), 1)
        self.assertEqual(entries.iloc[0]["entry_price"], 999.0)
        self.assertEqual(entries.iloc[0]["entry_price_date"], pd.Timestamp("2024-01-03"))

    def test_six_day_early_quote_is_unavailable(self):
        dates = pd.bdate_range("2025-01-02", "2025-01-14")
        aapl = pd.Series(np.nan, index=dates)
        aapl.loc["2025-01-03"] = 100.0
        aapl.loc["2025-01-07"] = 110.0
        prices = pd.DataFrame({"AAPL": aapl, "SPY": 400.0}, index=dates)

        row = evaluate_backtest(
            pd.DataFrame({"ticker": ["AAPL"]}),
            prices,
            pd.Timestamp("2025-01-02"),
            horizon=10,
            entry_slippage_bps=0,
            exit_slippage_bps=0,
        ).iloc[0]

        self.assertEqual(row["bt_coverage"], "unavailable")
        self.assertTrue(np.isnan(row["bt_return_pct"]))

    def test_invalid_exact_entry_does_not_shift_to_later_quote(self):
        dates = pd.bdate_range("2025-01-02", "2025-01-07")
        prices = pd.DataFrame(
            {"AAPL": [90.0, 0.0, 100.0, 110.0], "SPY": [390.0, 400.0, 410.0, 420.0]},
            index=dates,
        )

        result = evaluate_backtest(
            pd.DataFrame({"ticker": ["AAPL"]}),
            prices,
            pd.Timestamp("2025-01-02"),
            horizon=1,
            entry_slippage_bps=0,
            exit_slippage_bps=0,
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["bt_coverage"], "unavailable")
        self.assertTrue(np.isnan(result.iloc[0]["bt_entry_price"]))
        self.assertEqual(result.attrs["n_unavailable"], 1)

    def test_nonpositive_exact_exit_is_unavailable(self):
        dates = pd.bdate_range("2025-01-02", "2025-01-06")
        prices = pd.DataFrame(
            {"AAPL": [90.0, 100.0, 0.0], "SPY": [390.0, 400.0, 410.0]},
            index=dates,
        )

        row = evaluate_backtest(
            pd.DataFrame({"ticker": ["AAPL"]}),
            prices,
            pd.Timestamp("2025-01-02"),
            horizon=3,
            entry_slippage_bps=0,
            exit_slippage_bps=0,
        ).iloc[0]

        self.assertEqual(row["bt_coverage"], "unavailable")
        self.assertTrue(np.isnan(row["bt_exit_price"]))
        self.assertTrue(np.isnan(row["bt_raw_return_pct"]))
        self.assertEqual(row["bt_unavailable_reason"], "exit_quote_unavailable")
        self.assertTrue(row["bt_stale_exit"])

    def test_fail_closed_acquisition_entry_and_execution_integration(self):
        transactions = pd.DataFrame(
            {
                "member": ["Alice"],
                "ticker": ["AAPL"],
                "transaction_date": [pd.Timestamp("2025-01-02")],
                "disclosure_date": [pd.Timestamp("2025-01-02")],
                "transaction_type": ["Purchase"],
            }
        )
        dates = pd.bdate_range("2025-01-02", "2025-01-06")

        for invalid_entry in (0.0, np.nan):
            with self.subTest(endpoint="entry", value=invalid_entry):
                acquired = _validate_and_log_prices(
                    pd.DataFrame(
                        {
                            "AAPL": [90.0, invalid_entry, 110.0],
                            "SPY": [390.0, 400.0, 410.0],
                        },
                        index=dates,
                    ),
                    ["AAPL", "SPY"],
                )
                entries = _entry_prices_from_matrix(transactions, acquired)
                evaluated = evaluate_backtest(
                    pd.DataFrame({"ticker": ["AAPL"]}),
                    acquired,
                    pd.Timestamp("2025-01-02"),
                    horizon=3,
                    entry_slippage_bps=0,
                    exit_slippage_bps=0,
                )
                self.assertTrue(entries.empty)
                self.assertEqual(len(evaluated), 1)
                self.assertEqual(evaluated.iloc[0]["bt_coverage"], "unavailable")
                self.assertTrue(np.isnan(evaluated.iloc[0]["bt_entry_price"]))
                self.assertFalse(evaluated.iloc[0]["bt_stale_exit"])
                self.assertEqual(evaluated.attrs["n_unavailable"], 1)

        for invalid_exit in (0.0, np.nan):
            with self.subTest(endpoint="exit", value=invalid_exit):
                acquired = _validate_and_log_prices(
                    pd.DataFrame(
                        {
                            "AAPL": [90.0, 100.0, invalid_exit],
                            "SPY": [390.0, 400.0, 410.0],
                        },
                        index=dates,
                    ),
                    ["AAPL", "SPY"],
                )
                entries = _entry_prices_from_matrix(transactions, acquired)
                row = evaluate_backtest(
                    pd.DataFrame({"ticker": ["AAPL"]}),
                    acquired,
                    pd.Timestamp("2025-01-02"),
                    horizon=3,
                    entry_slippage_bps=0,
                    exit_slippage_bps=0,
                ).iloc[0]
                self.assertEqual(len(entries), 1)
                self.assertEqual(row["bt_coverage"], "unavailable")
                self.assertTrue(np.isnan(row["bt_return_pct"]))

    def test_mixed_stock_option_and_missing_instrument_are_isolated(self):
        dates = pd.bdate_range("2025-01-02", "2025-01-06")
        prices = pd.DataFrame(
            {
                "AAPL": [90.0, 100.0, 110.0],
                "MSFT": [190.0, 200.0, 220.0],
                "NVDA": [290.0, 300.0, 330.0],
                "SPY": [390.0, 400.0, 410.0],
            },
            index=dates,
        )
        recommendations = pd.DataFrame(
            {
                "ticker": ["AAPL", "MSFT", "NVDA"],
                "instrument_type": ["stock", "call", None],
            }
        )

        result = evaluate_backtest(
            recommendations,
            prices,
            pd.Timestamp("2025-01-02"),
            horizon=3,
            entry_slippage_bps=0,
            exit_slippage_bps=0,
        ).set_index("ticker")

        self.assertEqual(result.loc["AAPL", "bt_coverage"], "complete")
        self.assertEqual(result.loc["AAPL", "bt_return_pct"], 10.0)
        for ticker in ("MSFT", "NVDA"):
            self.assertEqual(result.loc[ticker, "bt_coverage"], "unavailable")
            self.assertEqual(
                result.loc[ticker, "bt_unavailable_reason"],
                "unsupported_instrument_pricing",
            )
            self.assertTrue(np.isnan(result.loc[ticker, "bt_return_pct"]))
            self.assertFalse(result.loc[ticker, "bt_stale_exit"])
        self.assertEqual(result.attrs["n_unavailable"], 2)

    def test_stock_pricing_programming_error_propagates(self):
        dates = pd.bdate_range("2025-01-02", "2025-01-06")
        prices = pd.DataFrame(
            {"AAPL": [90.0, 100.0, 110.0], "SPY": [390.0, 400.0, 410.0]},
            index=dates,
        )
        recommendations = pd.DataFrame(
            {"ticker": ["AAPL"], "instrument_type": ["stock"]}
        )

        with patch(
            "analyzer.options.estimate_options_leverage",
            side_effect=ValueError("programming error"),
        ):
            with self.assertRaisesRegex(ValueError, "programming error"):
                evaluate_backtest(
                    recommendations,
                    prices,
                    pd.Timestamp("2025-01-02"),
                    horizon=3,
                    entry_slippage_bps=0,
                    exit_slippage_bps=0,
                )

    def test_timezone_aware_daily_index_preserves_calendar_dates(self):
        dates = pd.bdate_range("2025-01-02", "2025-01-07", tz="America/New_York")
        prices = pd.DataFrame(
            {"AAPL": [90.0, 100.0, 110.0, 120.0], "SPY": [390.0, 400.0, 410.0, 420.0]},
            index=dates,
        )

        row = evaluate_backtest(
            pd.DataFrame({"ticker": ["AAPL"]}),
            prices,
            pd.Timestamp("2025-01-02", tz="America/New_York"),
            horizon=1,
            entry_slippage_bps=0,
            exit_slippage_bps=0,
        ).iloc[0]

        self.assertEqual(row["bt_entry_date"], pd.Timestamp("2025-01-03").date())
        self.assertEqual(row["bt_exit_date"], pd.Timestamp("2025-01-03").date())


if __name__ == "__main__":
    unittest.main()


class TestAlignedPriceHelpers(unittest.TestCase):
    def test_next_session_returns_execution_date_and_wait(self):
        prices = pd.DataFrame(
            {"A": [100.0, 120.0]},
            index=pd.to_datetime(["2025-01-03", "2025-01-06"]),
        )
        idx_ns, vals = _price_arrays(prices, "A")
        execution = _next_tradable_price_arrays(
            idx_ns, vals, pd.Timestamp("2025-01-05"), max_wait_days=3
        )
        self.assertEqual(execution.price, 120.0)
        self.assertEqual(execution.date, pd.Timestamp("2025-01-06"))
        self.assertEqual(execution.staleness_days, 1)

    def test_aligned_prior_rejects_stale_and_nonpositive(self):
        prices = pd.DataFrame(
            {"A": [100.0, 0.0]},
            index=pd.to_datetime(["2025-01-01", "2025-01-03"]),
        )
        idx_ns, vals = _price_arrays(prices, "A")
        self.assertIsNone(
            _aligned_price_at_or_before_arrays(
                idx_ns, vals, pd.Timestamp("2025-01-10"), max_staleness_days=5
            )
        )
        aligned = _aligned_price_at_or_before_arrays(
            idx_ns, vals, pd.Timestamp("2025-01-03"), max_staleness_days=3
        )
        self.assertEqual(aligned.price, 100.0)
        self.assertEqual(aligned.date, pd.Timestamp("2025-01-01"))
        self.assertEqual(aligned.staleness_days, 2)

    def test_ou_curve_enters_after_weekend_disclosure(self):
        rows = pd.DataFrame(
            {
                "ticker": ["A"],
                "disclosure_date": [pd.Timestamp("2025-01-05")],
                "entry_price": [100.0],
            }
        )
        prices = pd.DataFrame(
            {"A": [100.0, 120.0, 132.0, 144.0]},
            index=pd.to_datetime(
                ["2025-01-03", "2025-01-06", "2025-01-07", "2025-01-08"]
            ),
        )
        curves = _build_curves_for_rows(rows, prices, horizon=3)
        self.assertEqual(len(curves), 1)
        np.testing.assert_allclose(curves[0], [0.0, 0.1, 0.2])
