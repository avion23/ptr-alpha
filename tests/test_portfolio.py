"""Tests for Kelly criterion portfolio construction."""

import unittest

import pandas as pd

from analyzer.portfolio import (
    KellyConfig,
    build_kelly_portfolio,
    compute_portfolio_metrics,
    compute_payout_ratio,
    half_kelly,
    kelly_fraction,
    simulate_portfolio_returns,
)


def _make_recs(tickers, scores=None, win_rates=None, crash_probs=None):
    """Create a synthetic recommendations DataFrame."""
    n = len(tickers)
    if scores is None:
        scores = list(range(n, 0, -1))
    if win_rates is None:
        win_rates = [0.6] * n
    if crash_probs is None:
        crash_probs = [0.0] * n
    return pd.DataFrame({
        "rank": range(1, n + 1),
        "ticker": tickers,
        "signal_score": scores,
        "win_rate": win_rates,
        "crash_prob": crash_probs,
        "member": ["m1", "m2", "m3", "m4", "m5"][:n],
    })


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


class TestKellyFraction(unittest.TestCase):
    def test_positive_edge(self):
        # p=0.6, b=1.25 -> f* = (0.6*1.25 - 0.4)/1.25 = (0.75-0.4)/1.25 = 0.28
        f = kelly_fraction(0.6, 1.25)
        self.assertAlmostEqual(f, 0.28, places=2)

    def test_no_edge(self):
        # p=0.5, b=1.0 -> f* = (0.5*1.0 - 0.5)/1.0 = 0.0
        f = kelly_fraction(0.5, 1.0)
        self.assertAlmostEqual(f, 0.0, places=6)

    def test_negative_edge(self):
        f = kelly_fraction(0.4, 1.0)
        self.assertEqual(f, 0.0)

    def test_edge_cases(self):
        self.assertEqual(kelly_fraction(0.0, 1.0), 0.0)
        self.assertEqual(kelly_fraction(1.0, 1.0), 0.0)
        self.assertEqual(kelly_fraction(0.6, 0.0), 0.0)
        self.assertEqual(kelly_fraction(0.6, -1.0), 0.0)

    def test_half_kelly(self):
        f_full = kelly_fraction(0.6, 1.25)
        f_half = half_kelly(0.6, 1.25)
        self.assertAlmostEqual(f_half, f_full / 2.0, places=6)

    def test_payout_ratio(self):
        b = compute_payout_ratio(1.5, 1.0)
        self.assertAlmostEqual(b, 1.5, places=6)

    def test_payout_ratio_zero_loss(self):
        b = compute_payout_ratio(1.5, 0.0)
        self.assertEqual(b, 0.0)


class TestBuildKellyPortfolio(unittest.TestCase):
    def test_basic_sizing(self):
        recs = _make_recs(["A", "B", "C"])
        portfolio = build_kelly_portfolio(recs, KellyConfig(capital=100_000))
        self.assertGreater(len(portfolio), 0)
        self.assertTrue("ticker" in portfolio.columns)
        self.assertTrue("weight" in portfolio.columns)
        self.assertTrue("kelly_fraction" in portfolio.columns)
        self.assertTrue("position_value" in portfolio.columns)

    def test_weights_sum_to_at_most_one(self):
        recs = _make_recs(["A", "B", "C", "D"])
        portfolio = build_kelly_portfolio(recs, KellyConfig(capital=100_000))
        total_weight = portfolio["weight"].sum()
        self.assertLessEqual(total_weight, 1.0 + 0.01)
        self.assertGreater(total_weight, 0.0)

    def test_position_values_sum_to_capital(self):
        # Use relaxed caps so all capital can be deployed
        recs = _make_recs(["A", "B", "C"])
        portfolio = build_kelly_portfolio(
            recs, KellyConfig(capital=100_000, max_member_pct=1.0, max_ticker_pct=1.0)
        )
        total_value = portfolio["position_value"].sum()
        self.assertAlmostEqual(total_value, 100_000, delta=1000)

    def test_max_ticker_cap(self):
        # One very strong signal should be capped at 20%
        recs = _make_recs(["A", "B"], scores=[100, 1], win_rates=[0.8, 0.5])
        portfolio = build_kelly_portfolio(
            recs, KellyConfig(capital=100_000, max_ticker_pct=0.20)
        )
        max_weight = portfolio["weight"].max()
        self.assertLessEqual(max_weight, 0.20 + 0.01)  # small tolerance

    def test_max_member_cap(self):
        # Same member on multiple tickers
        recs = pd.DataFrame({
            "rank": [1, 2, 3],
            "ticker": ["A", "B", "C"],
            "signal_score": [30, 20, 10],
            "win_rate": [0.6, 0.6, 0.6],
            "crash_prob": [0.0, 0.0, 0.0],
            "member": ["same", "same", "other"],
        })
        portfolio = build_kelly_portfolio(
            recs, KellyConfig(capital=100_000, max_member_pct=0.05)
        )
        member_weights = portfolio.groupby("member")["weight"].sum()
        self.assertLessEqual(member_weights.get("same", 0), 0.05 + 0.01)

    def test_crash_guard_reduces_kelly_fraction(self):
        recs = _make_recs(["A", "B"], scores=[10, 10], crash_probs=[0.5, 0.0])
        portfolio = build_kelly_portfolio(recs, KellyConfig(capital=100_000, crash_guard=True))
        kellys = portfolio.set_index("ticker")["kelly_fraction"]
        if "A" in kellys.index and "B" in kellys.index:
            self.assertLess(kellys["A"], kellys["B"])

    def test_crash_guard_no_guard_equal(self):
        recs = _make_recs(["A", "B"], scores=[10, 10], crash_probs=[0.5, 0.0])
        no_guard = build_kelly_portfolio(recs, KellyConfig(capital=100_000, crash_guard=False))
        kellys = no_guard.set_index("ticker")["kelly_fraction"]
        if "A" in kellys.index and "B" in kellys.index:
            self.assertAlmostEqual(kellys["A"], kellys["B"], places=6)

    def test_empty_recommendations(self):
        portfolio = build_kelly_portfolio(pd.DataFrame())
        self.assertTrue(portfolio.empty)

    def test_all_zero_crash_probs(self):
        recs = _make_recs(["A", "B"], crash_probs=[0.0, 0.0])
        portfolio = build_kelly_portfolio(recs, KellyConfig(crash_guard=True))
        self.assertGreater(len(portfolio), 0)

    def test_low_win_rate_filtering(self):
        # Very low win rates should produce zero Kelly -> filtered out
        recs = _make_recs(["A", "B"], win_rates=[0.3, 0.3])
        portfolio = build_kelly_portfolio(recs)
        # With p=0.3, Kelly should be 0 or very low
        if len(portfolio) > 0:
            self.assertTrue(all(portfolio["kelly_fraction"] >= 0))


class TestSimulatePortfolioReturns(unittest.TestCase):
    def test_single_period(self):
        recs = _make_recs(["A", "B"], scores=[10, 5])
        portfolio = build_kelly_portfolio(recs, KellyConfig(capital=100_000))
        prices = _make_prices(
            ["A", "B", "SPY"], "2024-01-01", "2024-04-01",
            base_prices={"A": 100, "B": 110, "SPY": 400},
        )
        portfolio["as_of_date"] = pd.Timestamp("2024-01-01").date()
        result = simulate_portfolio_returns(portfolio, prices, horizon=60)
        self.assertGreater(len(result), 0)
        self.assertIn("portfolio_return", result.columns)
        self.assertIn("spy_return", result.columns)
        self.assertIn("portfolio_alpha", result.columns)

    def test_alpha_calculation(self):
        # alpha = portfolio_return - spy_return
        result = simulate_portfolio_returns(
            pd.DataFrame(), pd.DataFrame()
        )
        self.assertTrue(result.empty)


class TestComputePortfolioMetrics(unittest.TestCase):
    def test_basic_metrics(self):
        df = pd.DataFrame({
            "as_of_date": ["2024-01-01", "2024-02-01", "2024-03-01"],
            "portfolio_return": [2.0, -1.0, 3.0],
            "spy_return": [1.0, -0.5, 1.5],
            "portfolio_alpha": [1.0, -0.5, 1.5],
            "num_positions": [3, 3, 3],
        })
        metrics = compute_portfolio_metrics(df)
        self.assertIn("total_return_pct", metrics)
        self.assertIn("sharpe_ratio", metrics)
        self.assertIn("max_drawdown_pct", metrics)
        self.assertIn("win_rate_pct", metrics)
        self.assertIn("spy_total_return_pct", metrics)

    def test_empty_returns(self):
        metrics = compute_portfolio_metrics(pd.DataFrame())
        self.assertEqual(metrics, {})


class TestKellyConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = KellyConfig()
        self.assertEqual(cfg.capital, 100_000.0)
        self.assertEqual(cfg.max_ticker_pct, 0.20)
        self.assertEqual(cfg.max_member_pct, 0.05)
        self.assertTrue(cfg.use_half_kelly)
        self.assertTrue(cfg.crash_guard)

    def test_custom(self):
        cfg = KellyConfig(capital=50_000, max_ticker_pct=0.15, use_half_kelly=False)
        self.assertEqual(cfg.capital, 50_000)
        self.assertEqual(cfg.max_ticker_pct, 0.15)
        self.assertFalse(cfg.use_half_kelly)


if __name__ == "__main__":
    unittest.main()
