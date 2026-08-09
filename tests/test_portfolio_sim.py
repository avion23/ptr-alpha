"""Tests for portfolio simulation module."""

import unittest
from datetime import date
from unittest.mock import patch

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
    @patch.object(PortfolioSimulator, "_get_sector", return_value="Technology")
    def test_respects_max_positions(self, _mock_sector):
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

    @patch.object(PortfolioSimulator, "_get_sector", return_value="Technology")
    def test_position_sizing_respects_max_position_pct(self, _mock_sector):
        cfg = PortfolioConfig(
            initial_capital=10000,
            max_positions=10,
            max_position_pct=0.25,
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
        max_value = cfg.initial_capital * cfg.max_position_pct
        self.assertLessEqual(pos.cost, max_value + 1.0)


class TestSectorConstraint(unittest.TestCase):
    def test_sector_constraint_respected(self):
        cfg = PortfolioConfig(
            initial_capital=100000,
            max_positions=10,
            max_sector_pct=0.40,
            hold_period_days=365,
            entry_slippage_pct=0.0,
            exit_slippage_pct=0.0,
        )
        sim = PortfolioSimulator(cfg)

        sector_map = {"A": "Tech", "B": "Tech", "C": "Tech", "D": "Finance"}
        with patch.object(
            sim,
            "_get_sector",
            side_effect=lambda t, rec=None: sector_map.get(t, "Unknown"),
        ):
            prices = _make_prices(
                ["A", "B", "C", "D"],
                "2024-01-01",
                "2024-01-10",
                base_prices={"A": 100, "B": 100, "C": 100, "D": 100},
            )
            recs = _make_recs(
                ["A", "B", "C", "D"], "2024-01-01", scores=[40, 30, 20, 10]
            )
            sim.run(recs, prices, date(2024, 1, 1), date(2024, 1, 2))

        # At least one tech should be excluded due to sector cap
        tech_count = sum(1 for p in sim.positions if p.sector == "Tech")
        total = len(sim.positions)
        if total > 0:
            tech_pct = tech_count / cfg.max_positions
            # Sector exposure should not exceed limit
            self.assertLessEqual(tech_pct, cfg.max_sector_pct + 0.01)


class TestExitAfterHoldPeriod(unittest.TestCase):
    @patch.object(PortfolioSimulator, "_get_sector", return_value="Technology")
    def test_exit_after_hold_period(self, _mock_sector):
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
    @patch.object(PortfolioSimulator, "_get_sector", return_value="Technology")
    def test_cash_decreases_on_entry(self, _mock_sector):
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

    @patch.object(PortfolioSimulator, "_get_sector", return_value="Technology")
    def test_cash_increases_on_exit(self, _mock_sector):
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
    @patch.object(PortfolioSimulator, "_get_sector", return_value="Technology")
    def test_entry_slippage_applied(self, _mock_sector):
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

    @patch.object(PortfolioSimulator, "_get_sector", return_value="Technology")
    def test_exit_slippage_applied(self, _mock_sector):
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
    @patch.object(PortfolioSimulator, "_get_sector", return_value="Technology")
    def test_sharpe_ratio_computed(self, _mock_sector):
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
    @patch.object(PortfolioSimulator, "_get_sector", return_value="Technology")
    def test_spy_comparison_present(self, _mock_sector):
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
    @patch.object(PortfolioSimulator, "_get_sector", return_value="Technology")
    def test_overlapping_positions_tracked(self, _mock_sector):
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

    @patch.object(PortfolioSimulator, "_get_sector", return_value="Technology")
    def test_drawdown_captures_first_period_loss(self, _mock_sector):
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
        self.assertAlmostEqual(metrics["max_drawdown_pct"], -0.24, places=2)


class TestCausalExecutionScenarios(unittest.TestCase):
    def test_weekend_signal_executes_monday_not_friday(self):
        cfg = PortfolioConfig(
            initial_capital=1000,
            max_positions=1,
            max_position_pct=1.0,
            max_sector_pct=1.0,
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

    def test_stale_or_missing_next_session_is_rejected(self):
        cfg = PortfolioConfig(max_execution_wait_days=3)
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

    def test_hand_ledger_shared_cash_and_gross_turnover(self):
        cfg = PortfolioConfig(
            initial_capital=1000,
            max_positions=1,
            max_position_pct=1.0,
            max_sector_pct=1.0,
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
            max_position_pct=1.0,
            max_sector_pct=1.0,
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

    def test_missing_stored_sector_fails_loudly(self):
        cfg = PortfolioConfig()
        prices = pd.DataFrame({"A": [100.0]}, index=pd.to_datetime(["2024-01-02"]))
        recs = _make_recs(["A"], "2024-01-01").drop(columns=["sector"])
        with self.assertRaisesRegex(ValueError, "Missing stored sector"):
            PortfolioSimulator(cfg).run(
                recs, prices, date(2024, 1, 1), date(2024, 1, 2)
            )

    def test_recommendations_off_rebalance_cadence_are_not_traded(self):
        cfg = PortfolioConfig(
            initial_capital=1000,
            max_positions=2,
            max_position_pct=0.5,
            max_sector_pct=1.0,
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


if __name__ == "__main__":
    unittest.main()
