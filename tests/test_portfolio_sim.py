"""Tests for portfolio simulation module."""

import unittest
from datetime import date

import pandas as pd

from analyzer.portfolio_sim import (
    PortfolioConfig,
    PortfolioSimulator,
)


def _make_prices(tickers, start, end, base_prices=None, daily_drift=0.0):
    """Create a synthetic prices DataFrame."""
    dates = pd.date_range(start, end, freq="D")
    if base_prices is None:
        base_prices = {t: 100.0 + i * 10 for i, t in enumerate(tickers)}
    data = {}
    for ticker in tickers:
        base = base_prices.get(ticker, 100.0)
        data[ticker] = [base * (1 + daily_drift * i) for i in range(len(dates))]
    return pd.DataFrame(data, index=dates)


def _make_recs(tickers, as_of_date, scores=None):
    """Create a synthetic recommendations DataFrame."""
    if scores is None:
        scores = list(range(len(tickers), 0, -1))
    return pd.DataFrame(
        {
            "rank": range(1, len(tickers) + 1),
            "ticker": tickers,
            "signal_score": scores,
            "num_buyers": [3] * len(tickers),
            "sector": ["Technology"] * len(tickers),
            "as_of_date": pd.Timestamp(as_of_date),
        }
    )


class TestPositionEntry(unittest.TestCase):
    def test_respects_max_positions(self):
        cfg = PortfolioConfig(
            initial_capital=100000,
            max_positions=2,
            hold_period_days=365,
            entry_slippage_pct=0.0,
            exit_slippage_pct=0.0,
        )
        sim = PortfolioSimulator(cfg)
        prices = _make_prices(["A", "B", "C"], "2024-01-01", "2024-01-10")
        recs = _make_recs(["A", "B", "C"], "2024-01-01")
        sim.run(recs, prices, date(2024, 1, 1), date(2024, 1, 2))
        self.assertLessEqual(len(sim.positions), 2)

    def test_position_sizing_uses_equal_slot_fraction(self):
        cfg = PortfolioConfig(
            initial_capital=10000,
            max_positions=10,
            hold_period_days=365,
            entry_slippage_pct=0.0,
            exit_slippage_pct=0.0,
        )
        sim = PortfolioSimulator(cfg)
        prices = _make_prices(
            ["A"], "2024-01-01", "2024-01-10", base_prices={"A": 100.0}
        )
        recs = _make_recs(["A"], "2024-01-01")
        sim.run(recs, prices, date(2024, 1, 1), date(2024, 1, 2))
        self.assertEqual(len(sim.positions), 1)
        pos = sim.positions[0]
        self.assertEqual(pos.cost, cfg.initial_capital / cfg.max_positions)

    def test_repeated_runs_reset_mutable_state(self):
        cfg = PortfolioConfig(
            initial_capital=1000,
            max_positions=1,
            hold_period_days=2,
            entry_slippage_pct=0.0,
            exit_slippage_pct=0.0,
        )
        prices = _make_prices(["A"], "2024-01-01", "2024-01-10")
        recommendations = _make_recs(["A"], "2024-01-01")
        sim = PortfolioSimulator(cfg)

        first = sim.run(recommendations, prices, date(2024, 1, 1), date(2024, 1, 10))
        first_closed = list(sim.closed_positions)
        first_gross = sim.gross_traded_notional
        second = sim.run(recommendations, prices, date(2024, 1, 1), date(2024, 1, 10))

        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(sim.closed_positions, first_closed)
        self.assertEqual(sim.gross_traded_notional, first_gross)
        self.assertEqual(len(sim.snapshots), len(first))

    def test_run_does_not_mutate_recommendations(self):
        cfg = PortfolioConfig(
            initial_capital=1000,
            max_positions=1,
        )
        prices = _make_prices(["A"], "2024-01-01", "2024-01-03")
        recommendations = _make_recs(["A"], "2024-01-01")
        original = recommendations.copy(deep=True)

        PortfolioSimulator(cfg).run(
            recommendations, prices, date(2024, 1, 1), date(2024, 1, 3)
        )

        pd.testing.assert_frame_equal(recommendations, original)


class TestExitAfterHoldPeriod(unittest.TestCase):
    def test_exit_after_hold_period(self):
        cfg = PortfolioConfig(
            initial_capital=20000,
            max_positions=5,
            hold_period_days=30,
            entry_slippage_pct=0.0,
            exit_slippage_pct=0.0,
        )
        sim = PortfolioSimulator(cfg)
        prices = _make_prices(
            ["AAPL"], "2024-01-01", "2024-03-01", base_prices={"AAPL": 150.0}
        )
        recs = _make_recs(["AAPL"], "2024-01-01")

        sim.run(recs, prices, date(2024, 1, 1), date(2024, 2, 15))

        # After 30+ days the position should be closed
        self.assertEqual(len(sim.positions), 0)
        self.assertGreater(len(sim.closed_positions), 0)
        self.assertEqual(sim.closed_positions[0]["ticker"], "AAPL")


class TestCashFlows(unittest.TestCase):
    def test_cash_decreases_on_entry(self):
        cfg = PortfolioConfig(
            initial_capital=10000,
            max_positions=5,
            hold_period_days=365,
            entry_slippage_pct=0.0,
            exit_slippage_pct=0.0,
        )
        sim = PortfolioSimulator(cfg)
        initial_cash = sim.cash
        prices = _make_prices(
            ["A"], "2024-01-01", "2024-01-10", base_prices={"A": 100.0}
        )
        recs = _make_recs(["A"], "2024-01-01")
        sim.run(recs, prices, date(2024, 1, 1), date(2024, 1, 2))
        self.assertLess(sim.cash, initial_cash)

    def test_cash_increases_on_exit(self):
        cfg = PortfolioConfig(
            initial_capital=10000,
            max_positions=5,
            hold_period_days=5,
            entry_slippage_pct=0.0,
            exit_slippage_pct=0.0,
        )
        sim = PortfolioSimulator(cfg)
        prices = _make_prices(
            ["A"], "2024-01-01", "2024-02-01", base_prices={"A": 100.0}
        )
        recs = _make_recs(["A"], "2024-01-01")
        sim.run(recs, prices, date(2024, 1, 1), date(2024, 1, 15))
        # After exit, cash should have increased from the entry deduction
        self.assertEqual(len(sim.positions), 0)
        self.assertGreater(sim.cash, 0)
        # Cash should be roughly initial minus slippage costs
        self.assertGreater(sim.cash, cfg.initial_capital * 0.9)


class TestSlippage(unittest.TestCase):
    def test_entry_slippage_applied(self):
        cfg_no_slip = PortfolioConfig(
            initial_capital=10000,
            max_positions=5,
            hold_period_days=365,
            entry_slippage_pct=0.0,
            exit_slippage_pct=0.0,
        )
        cfg_slip = PortfolioConfig(
            initial_capital=10000,
            max_positions=5,
            hold_period_days=365,
            entry_slippage_pct=0.01,
            exit_slippage_pct=0.0,
        )
        prices = _make_prices(
            ["A"], "2024-01-01", "2024-01-10", base_prices={"A": 100.0}
        )

        sim_no = PortfolioSimulator(cfg_no_slip)
        sim_no.run(
            _make_recs(["A"], "2024-01-01"), prices, date(2024, 1, 1), date(2024, 1, 2)
        )

        sim_slip = PortfolioSimulator(cfg_slip)
        sim_slip.run(
            _make_recs(["A"], "2024-01-01"), prices, date(2024, 1, 1), date(2024, 1, 2)
        )

        self.assertEqual(len(sim_no.positions), 1)
        self.assertEqual(len(sim_slip.positions), 1)
        self.assertGreater(
            sim_slip.positions[0].entry_price, sim_no.positions[0].entry_price
        )

    def test_exit_slippage_applied(self):
        cfg = PortfolioConfig(
            initial_capital=10000,
            max_positions=5,
            hold_period_days=5,
            entry_slippage_pct=0.0,
            exit_slippage_pct=0.05,
        )
        sim = PortfolioSimulator(cfg)
        prices = _make_prices(
            ["A"], "2024-01-01", "2024-02-01", base_prices={"A": 100.0}
        )
        sim.run(
            _make_recs(["A"], "2024-01-01"), prices, date(2024, 1, 1), date(2024, 1, 15)
        )
        self.assertEqual(len(sim.closed_positions), 1)
        exit_price = sim.closed_positions[0]["exit_price"]
        self.assertLess(exit_price, 100.0)


class TestSharpeRatio(unittest.TestCase):
    def test_sharpe_ratio_computed(self):
        cfg = PortfolioConfig(
            initial_capital=20000,
            max_positions=5,
            hold_period_days=5,
            entry_slippage_pct=0.0,
            exit_slippage_pct=0.0,
        )
        sim = PortfolioSimulator(cfg)
        prices = _make_prices(
            ["A"],
            "2024-01-01",
            "2024-06-01",
            base_prices={"A": 100.0},
            daily_drift=0.001,
        )
        sim.run(
            _make_recs(["A"], "2024-01-01"), prices, date(2024, 1, 1), date(2024, 5, 1)
        )
        metrics = sim.compute_metrics(prices)
        self.assertIn("sharpe_ratio", metrics)
        self.assertIsInstance(metrics["sharpe_ratio"], float)


class TestComputeMetrics(unittest.TestCase):
    def test_spy_comparison_present(self):
        cfg = PortfolioConfig(
            initial_capital=20000,
            max_positions=5,
            hold_period_days=30,
            entry_slippage_pct=0.0,
            exit_slippage_pct=0.0,
        )
        sim = PortfolioSimulator(cfg)
        prices = _make_prices(
            ["A", "SPY"],
            "2024-01-01",
            "2024-03-01",
            base_prices={"A": 100.0, "SPY": 400.0},
        )
        sim.run(
            _make_recs(["A"], "2024-01-01"), prices, date(2024, 1, 1), date(2024, 2, 15)
        )
        metrics = sim.compute_metrics(prices)
        self.assertIsNotNone(metrics["spy_return_pct"])


class TestOverlappingPositions(unittest.TestCase):
    def test_overlapping_positions_tracked(self):
        cfg = PortfolioConfig(
            initial_capital=100000,
            max_positions=5,
            hold_period_days=60,
            entry_slippage_pct=0.0,
            exit_slippage_pct=0.0,
        )
        sim = PortfolioSimulator(cfg)
        prices = _make_prices(
            ["A", "B", "C", "D", "E"],
            "2024-01-01",
            "2024-06-01",
            base_prices={"A": 100, "B": 100, "C": 100, "D": 100, "E": 100},
        )
        # Create recommendations for multiple dates to generate overlap
        all_recs = []
        for i, d in enumerate(pd.date_range("2024-01-01", "2024-02-01", freq="14D")):
            tickers = [["A", "B"], ["C", "D"], ["E", "A"]][i % 3]
            recs = _make_recs(tickers, d, scores=[10 - j for j in range(len(tickers))])
            all_recs.append(recs)
        combined = pd.concat(all_recs, ignore_index=True)

        sim.run(combined, prices, date(2024, 1, 1), date(2024, 3, 1))
        metrics = sim.compute_metrics(prices)
        self.assertGreaterEqual(metrics["max_concurrent_positions"], 1)


class TestDrawdownFromInitialCapital(unittest.TestCase):
    """Regression: max_drawdown must anchor to initial_capital, not just
    to the first snapshot's post-trade value."""

    def test_drawdown_captures_first_period_loss(self):
        cfg = PortfolioConfig(
            initial_capital=10000,
            max_positions=1,
            hold_period_days=365,
            entry_slippage_pct=0.01,
            exit_slippage_pct=0.0,
        )
        sim = PortfolioSimulator(cfg)
        # Flat prices with entry slippage make the first recorded snapshot the
        # trough relative to initial capital; later snapshots are not lower.
        prices = _make_prices(
            ["A"],
            "2024-01-01",
            "2024-01-10",
            base_prices={"A": 100.0},
            daily_drift=0.0,
        )
        sim.run(
            _make_recs(["A"], "2024-01-01"), prices, date(2024, 1, 1), date(2024, 1, 5)
        )
        metrics = sim.compute_metrics(prices)
        # The pre-fix code reported 0% drawdown because the peak only tracked
        # post-entry equity; anchoring to initial capital captures the first
        # snapshot drawdown caused by entry slippage.
        self.assertAlmostEqual(metrics["max_drawdown_pct"], -0.99, places=2)


class TestCausalExecutionScenarios(unittest.TestCase):
    def test_weekend_signal_executes_monday_not_friday(self):
        cfg = PortfolioConfig(
            initial_capital=1000,
            max_positions=1,
            entry_slippage_pct=0.0,
            exit_slippage_pct=0.0,
        )
        prices = pd.DataFrame(
            {"A": [100.0, 120.0]},
            index=pd.to_datetime(["2024-01-05", "2024-01-08"]),
        )
        recs = _make_recs(["A"], "2024-01-07")
        sim = PortfolioSimulator(cfg)
        sim.run(recs, prices, date(2024, 1, 7), date(2024, 1, 8))
        self.assertEqual(sim.positions[0].entry_date, date(2024, 1, 8))
        self.assertEqual(sim.positions[0].entry_price, 120.0)

    def test_missing_exact_next_session_is_rejected(self):
        cfg = PortfolioConfig()
        prices = pd.DataFrame({"A": [100.0]}, index=pd.to_datetime(["2024-01-05"]))
        sim = PortfolioSimulator(cfg)
        sim.run(
            _make_recs(["A"], "2024-01-07"),
            prices,
            date(2024, 1, 7),
            date(2024, 1, 10),
        )
        self.assertFalse(sim.positions)
        self.assertEqual(sim.rejected_orders[0]["reason"], "no_next_tradable_session")

    def test_weekend_horizon_exits_on_prior_nyse_session(self):
        cfg = PortfolioConfig(
            initial_capital=1000,
            max_positions=1,
            hold_period_days=5,
            entry_slippage_pct=0.0,
            exit_slippage_pct=0.0,
        )
        prices = pd.DataFrame(
            {"A": [100.0, 101.0, 102.0, 103.0, 104.0]},
            index=pd.to_datetime(
                ["2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11", "2024-01-12"]
            ),
        )
        sim = PortfolioSimulator(cfg)
        sim.run(
            _make_recs(["A"], "2024-01-07"),
            prices,
            date(2024, 1, 7),
            date(2024, 1, 14),
        )
        self.assertEqual(sim.closed_positions[0]["entry_date"], date(2024, 1, 8))
        self.assertEqual(sim.closed_positions[0]["exit_date"], date(2024, 1, 12))

    def test_hand_ledger_shared_cash_and_gross_turnover(self):
        cfg = PortfolioConfig(
            initial_capital=1000,
            max_positions=1,
            hold_period_days=2,
            entry_slippage_pct=0.0,
            exit_slippage_pct=0.0,
        )
        prices = pd.DataFrame(
            {"A": [110.0, 120.0, 130.0]},
            index=pd.to_datetime(["2024-01-08", "2024-01-09", "2024-01-10"]),
        )
        sim = PortfolioSimulator(cfg)
        sim.run(
            _make_recs(["A"], "2024-01-07"),
            prices,
            date(2024, 1, 7),
            date(2024, 1, 10),
        )
        self.assertEqual(sim.closed_positions[0]["shares"], 9)
        self.assertEqual(sim.cash, 1180.0)
        self.assertEqual(sim.gross_traded_notional, 2160.0)
        metrics = sim.compute_metrics(prices)
        self.assertEqual(metrics["open_dollar_exposure"], 0.0)
        self.assertEqual(metrics["open_positions"], [])
        self.assertEqual(metrics["gross_traded_notional"], 2160.0)

    def test_open_ledger_marks_liquidation_cost(self):
        cfg = PortfolioConfig(
            initial_capital=1000,
            max_positions=1,
            hold_period_days=30,
            entry_slippage_pct=0.0,
            exit_slippage_pct=0.10,
        )
        prices = pd.DataFrame({"A": [100.0]}, index=pd.to_datetime(["2024-01-02"]))
        sim = PortfolioSimulator(cfg)
        sim.run(
            _make_recs(["A"], "2024-01-01"),
            prices,
            date(2024, 1, 1),
            date(2024, 1, 2),
        )
        metrics = sim.compute_metrics(prices)
        self.assertEqual(metrics["open_dollar_exposure"], 900.0)
        self.assertEqual(metrics["open_positions"][0]["liquidation_value"], 900.0)
        self.assertEqual(metrics["total_return_pct"], -10.0)

    def test_sector_metadata_is_not_required(self):
        cfg = PortfolioConfig()
        prices = pd.DataFrame({"A": [100.0]}, index=pd.to_datetime(["2024-01-02"]))
        recs = _make_recs(["A"], "2024-01-01").drop(columns=["sector"])
        sim = PortfolioSimulator(cfg)
        sim.run(recs, prices, date(2024, 1, 1), date(2024, 1, 2))
        self.assertEqual([position.ticker for position in sim.positions], ["A"])

    def test_recommendations_off_rebalance_cadence_are_not_traded(self):
        cfg = PortfolioConfig(
            initial_capital=1000,
            max_positions=2,
            rebalance_freq_days=2,
        )
        recs = pd.concat(
            [_make_recs(["A"], "2024-01-01"), _make_recs(["B"], "2024-01-02")],
            ignore_index=True,
        )
        prices = pd.DataFrame(
            {"A": [100.0, 100.0], "B": [100.0, 100.0]},
            index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
        )
        sim = PortfolioSimulator(cfg)
        sim.run(recs, prices, date(2024, 1, 1), date(2024, 1, 3))
        self.assertEqual([p.ticker for p in sim.positions], ["A"])

    def test_unverified_zero_quote_makes_exit_and_valuation_unavailable(self):
        cfg = PortfolioConfig(
            initial_capital=1000,
            max_positions=1,
            hold_period_days=1,
            entry_slippage_pct=0.0,
            exit_slippage_pct=0.0,
        )
        prices = pd.DataFrame(
            {"A": [100.0, 0.0]},
            index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
        )
        sim = PortfolioSimulator(cfg)
        sim.run(
            _make_recs(["A"], "2024-01-01"),
            prices,
            date(2024, 1, 1),
            date(2024, 1, 3),
        )
        self.assertEqual(sim.closed_positions, [])
        metrics = sim.compute_metrics(prices)
        self.assertEqual(metrics["valuation_status"], "unavailable")
        self.assertIsNone(metrics["total_return_pct"])
        self.assertIsNone(metrics["open_positions"][0]["mark_price"])

    def test_missing_exact_exit_session_does_not_shift_trade_later(self):
        cfg = PortfolioConfig(
            initial_capital=1000,
            max_positions=1,
            hold_period_days=1,
            entry_slippage_pct=0.0,
            exit_slippage_pct=0.0,
        )
        prices = pd.DataFrame(
            {"A": [100.0, 80.0]},
            index=pd.to_datetime(["2024-01-02", "2024-01-10"]),
        )
        sim = PortfolioSimulator(cfg)
        sim.run(
            _make_recs(["A"], "2024-01-01"),
            prices,
            date(2024, 1, 1),
            date(2024, 1, 10),
        )
        self.assertEqual(sim.closed_positions, [])
        metrics = sim.compute_metrics(prices)
        self.assertEqual(metrics["valuation_status"], "unavailable")
        self.assertEqual(metrics["valuation_reason"], "unbounded_open_position_mark")
        self.assertIsNone(metrics["total_return_pct"])
        self.assertEqual(metrics["unresolved_expired_positions"], 1)

    def test_unresolved_final_mark_abstains_from_return(self):
        cfg = PortfolioConfig(
            initial_capital=1000,
            max_positions=1,
            hold_period_days=1,
            entry_slippage_pct=0.0,
            exit_slippage_pct=0.0,
        )
        prices = pd.DataFrame(
            {"A": [100.0, 80.0]},
            index=pd.to_datetime(["2024-01-02", "2024-01-10"]),
        )
        sim = PortfolioSimulator(cfg)
        sim.run(
            _make_recs(["A"], "2024-01-01"),
            prices,
            date(2024, 1, 1),
            date(2024, 1, 5),
        )
        metrics = sim.compute_metrics(prices)
        self.assertEqual(metrics["valuation_status"], "unavailable")
        self.assertIsNone(metrics["total_return_pct"])
        self.assertIsNone(metrics["open_dollar_exposure"])
        self.assertEqual(
            metrics["open_positions"][0]["state"],
            "exit_unresolved_valuation_unavailable",
        )
        self.assertIsNone(metrics["open_positions"][0]["liquidation_value"])

    def test_compute_metrics_requires_prices_for_open_positions(self):
        cfg = PortfolioConfig(
            initial_capital=1000,
            max_positions=1,
        )
        prices = pd.DataFrame({"A": [100.0]}, index=pd.to_datetime(["2024-01-02"]))
        sim = PortfolioSimulator(cfg)
        sim.run(
            _make_recs(["A"], "2024-01-01"),
            prices,
            date(2024, 1, 1),
            date(2024, 1, 2),
        )
        with self.assertRaisesRegex(ValueError, "prices_df is required"):
            sim.compute_metrics()


if __name__ == "__main__":
    unittest.main()
