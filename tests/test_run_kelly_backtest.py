"""Tests for scripts.run_kelly_backtest module."""
import unittest


class TestRunKellyBacktest(unittest.TestCase):

    def test_module_imports(self):
        from scripts import run_kelly_backtest
        assert callable(run_kelly_backtest.main)

    def test_imports_analyzer_portfolio_functions(self):
        """Verify the portfolio functions the script depends on are importable."""
        from analyzer.portfolio import (
            KellyConfig,
            build_portfolios_from_backtest,
            compute_portfolio_metrics,
            simulate_portfolio_returns,
        )
        assert callable(build_portfolios_from_backtest)
        assert callable(compute_portfolio_metrics)
        assert callable(simulate_portfolio_returns)
        # KellyConfig should be constructible with expected fields
        cfg = KellyConfig(
            capital=100_000,
            max_ticker_pct=0.20,
            max_member_pct=0.05,
            total_exposure_pct=1.00,
            use_half_kelly=True,
            crash_guard=True,
        )
        self.assertEqual(cfg.capital, 100_000)

    def test_imports_analysis_module(self):
        """Verify the analysis module the script depends on is importable."""
        from analyzer import analysis
        assert callable(analysis.calculate_signal_potential)


if __name__ == "__main__":
    unittest.main()
