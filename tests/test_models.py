"""Smoke tests for analyzer.models module."""
import unittest

from analyzer.models import (
    DownloadResult,
    DownloadStatus,
    FilingType,
    TransactionType,
)


class TestTransactionType(unittest.TestCase):

    def test_values(self):
        self.assertEqual(TransactionType.PURCHASE.value, "Purchase")
        self.assertEqual(TransactionType.SALE.value, "Sale")

    def test_str_enum_is_str(self):
        # StrEnum members are also strings.
        self.assertEqual(TransactionType.PURCHASE, "Purchase")


class TestFilingType(unittest.TestCase):

    def test_values(self):
        self.assertEqual(FilingType.PTR.value, "P")
        self.assertEqual(FilingType.AMENDMENT.value, "A")


class TestDownloadStatus(unittest.TestCase):

    def test_values(self):
        self.assertEqual(DownloadStatus.SUCCESS.value, "success")
        self.assertEqual(DownloadStatus.SKIPPED.value, "skipped")
        self.assertEqual(DownloadStatus.FAILED.value, "failed")
        self.assertEqual(DownloadStatus.ERROR.value, "error")


class TestDownloadResult(unittest.TestCase):

    def test_defaults(self):
        r = DownloadResult(doc_id="doc-1", status=DownloadStatus.SUCCESS)
        self.assertEqual(r.doc_id, "doc-1")
        self.assertEqual(r.status, DownloadStatus.SUCCESS)
        self.assertEqual(r.error_message, "")
        self.assertEqual(r.status_code, 0)

    def test_overrides(self):
        r = DownloadResult(
            doc_id="doc-2",
            status=DownloadStatus.FAILED,
            error_message="network error",
            status_code=503,
        )
        self.assertEqual(r.status, DownloadStatus.FAILED)
        self.assertEqual(r.error_message, "network error")
        self.assertEqual(r.status_code, 503)


class TestFrozenDataclasses(unittest.TestCase):
    """Tests for frozen dataclass behavior."""

    def test_analysis_params_is_frozen(self):
        from analyzer.pipeline import AnalysisParams
        p = AnalysisParams(year=2024, horizons=(90,), threshold=5.0)
        with self.assertRaises(AttributeError):
            p.year = 2025

    def test_backtest_params_is_frozen(self):
        from datetime import date
        from analyzer.pipeline import BacktestParams
        p = BacktestParams(start_date=date(2024, 1, 1), end_date=date(2024, 12, 31))
        with self.assertRaises(AttributeError):
            p.horizon = 60

    def test_analysis_params_horizons_is_tuple(self):
        from analyzer.pipeline import AnalysisParams
        p = AnalysisParams(year=2024, horizons=(90,), threshold=5.0)
        self.assertIsInstance(p.horizons, tuple)

    def test_ticker_scoring_params_is_frozen(self):
        from analyzer.pipeline import TickerScoringParams
        p = TickerScoringParams(year=2024, horizons=(90,))
        with self.assertRaises(AttributeError):
            p.year = 2025

    def test_ticker_scoring_params_horizons_is_tuple(self):
        from analyzer.pipeline import TickerScoringParams
        p = TickerScoringParams(year=2024, horizons=(90,))
        self.assertIsInstance(p.horizons, tuple)

    def test_price_snapshot_unresolved_tickers_is_tuple(self):
        from analyzer.price_snapshot import PriceSnapshot
        p = PriceSnapshot(
            snapshot_id="test-id",
            created_at="2024-01-01",
            git_sha="abc123",
            yfinance_version="0.2.0",
            python_version="3.11.0",
            requested_tickers=10,
            resolved_tickers=8,
            unresolved_tickers=("AAPL", "MSFT"),
            price_rows=100,
            first_date="2024-01-01",
            last_date="2024-12-31",
        )
        self.assertIsInstance(p.unresolved_tickers, tuple)

    def test_price_snapshot_is_frozen(self):
        from analyzer.price_snapshot import PriceSnapshot
        p = PriceSnapshot(
            snapshot_id="test-id",
            created_at="2024-01-01",
            git_sha="abc123",
            yfinance_version="0.2.0",
            python_version="3.11.0",
            requested_tickers=10,
            resolved_tickers=8,
            unresolved_tickers=("AAPL", "MSFT"),
            price_rows=100,
            first_date="2024-01-01",
            last_date="2024-12-31",
        )
        with self.assertRaises(AttributeError):
            p.snapshot_id = "new-id"

    def test_step_result_is_frozen(self):
        from analyzer.exceptions import StepResult
        r = StepResult(success=True)
        with self.assertRaises(AttributeError):
            r.success = False

    def test_kelly_config_is_frozen(self):
        from analyzer.portfolio.kelly import KellyConfig
        c = KellyConfig()
        with self.assertRaises(AttributeError):
            c.capital = 200000.0

    def test_portfolio_config_is_frozen(self):
        from analyzer.portfolio_sim import PortfolioConfig
        c = PortfolioConfig()
        with self.assertRaises(AttributeError):
            c.initial_capital = 50000.0

    def test_portfolio_position_is_frozen(self):
        from datetime import date
        from analyzer.portfolio_sim import PortfolioPosition
        p = PortfolioPosition(
            ticker="AAPL",
            entry_date=date(2024, 1, 1),
            entry_price=150.0,
            shares=10,
            cost=1500.0,
            sector="Technology",
            signal_score=0.8,
            rank=1,
        )
        with self.assertRaises(AttributeError):
            p.ticker = "MSFT"

    def test_ticker_resolution_is_frozen(self):
        from analyzer.ticker_resolver import TickerResolution
        r = TickerResolution(
            raw_ticker="FB",
            price_symbol="META",
            status="renamed",
            confidence=1.0,
            notes="Renamed",
        )
        with self.assertRaises(AttributeError):
            r.raw_ticker = "META"

    def test_snooping_report_is_frozen(self):
        from analyzer.snooping import SnoopingReport
        r = SnoopingReport(
            n_tests=100,
            alpha_slope=1.5,
            overall_alpha=2.0,
            sharpe=1.2,
            n_observations=50,
            dates_evaluated=40,
            t_statistic=2.5,
            p_value_raw=0.01,
            bonferroni_threshold=0.0005,
            p_value_bonferroni=0.01,
            significant_bonferroni=True,
            bh_rejected=True,
            bh_adjusted_alpha=0.05,
            dsr=0.95,
            significant_dsr=True,
            min_years=2.0,
        )
        with self.assertRaises(AttributeError):
            r.n_tests = 200

    def test_sweep_result_is_frozen(self):
        from analyzer.validation import SweepResult
        r = SweepResult(
            horizon=90,
            frequency_days=30,
            training_lookback_days=365,
            min_buyers=3,
            top_n=5,
            decay_lambda=0.005,
            bayes_prior_strength=20,
        )
        with self.assertRaises(AttributeError):
            r.horizon = 120

    def test_ou_params_is_frozen(self):
        from analyzer.return_process import OUParams
        p = OUParams(theta=0.05, mu=0.1, sigma=0.02)
        with self.assertRaises(AttributeError):
            p.theta = 0.1

    def test_ou_posterior_is_frozen(self):
        from analyzer.return_process import OUPosterior
        p = OUPosterior(
            mu_mean=0.1,
            mu_var=0.01,
            theta=0.05,
            sigma2_ou=0.001,
            n_obs=10,
        )
        with self.assertRaises(AttributeError):
            p.mu_mean = 0.2

    def test_signal_features_is_frozen(self):
        from datetime import date
        from analyzer.signal_features import SignalFeatures
        f = SignalFeatures(
            ticker="AAPL",
            disclosure_date=date(2024, 1, 15),
            lag_days=10,
            pre_disclosure_return=0.05,
            pre_disclosure_alpha=0.02,
            max_drawdown_to_entry=0.03,
            volatility_20d=0.25,
            drawdown_from_ath=0.10,
            days_since_ipo=3650,
            n_buyers_30d=5,
        )
        with self.assertRaises(AttributeError):
            f.ticker = "MSFT"

    def test_crash_hazard_is_frozen(self):
        from analyzer.signal_features import CrashHazard
        h = CrashHazard(
            crash_prob=0.15,
            expected_return=-0.02,
            var_95=-0.15,
            cvar_95=-0.25,
        )
        with self.assertRaises(AttributeError):
            h.crash_prob = 0.5


if __name__ == "__main__":
    unittest.main()