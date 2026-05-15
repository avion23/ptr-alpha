import unittest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from analyzer.pipeline import (
    AnalysisParams,
    TickerAnalysisParams,
    TickerScoringParams,
    _prepare_analysis_data,
    _save_results,
    _analyze_by_sector,
    pipeline_step,
)
from analyzer.exceptions import AnalyzerError, DataSourceError


class TestAnalysisParams(unittest.TestCase):

    def test_defaults(self):
        p = AnalysisParams(source="house", year=2024, horizons=[90], threshold=5.0)
        self.assertIsNone(p.member_filter)
        self.assertIsNone(p.top_n)
        self.assertFalse(p.show_signals)

    def test_with_all_params(self):
        p = AnalysisParams(source="house", year=2024, horizons=[90], threshold=5.0, member_filter="test", top_n=10, show_signals=True)
        self.assertEqual(p.member_filter, "test")
        self.assertEqual(p.top_n, 10)
        self.assertTrue(p.show_signals)


class TestTickerScoringParams(unittest.TestCase):

    def test_defaults(self):
        p = TickerScoringParams(year=2024, horizons=[90])
        self.assertEqual(p.threshold, 5.0)
        self.assertEqual(p.days_back, 28)
        self.assertEqual(p.min_buyers, 2)
        self.assertEqual(p.top_n, 15)

    def test_custom_values(self):
        p = TickerScoringParams(year=2024, horizons=[30, 60], threshold=10.0, days_back=14, min_buyers=3, top_n=5)
        self.assertEqual(p.threshold, 10.0)
        self.assertEqual(p.days_back, 14)
        self.assertEqual(p.min_buyers, 3)
        self.assertEqual(p.top_n, 5)


class TestPipelineStep(unittest.TestCase):

    def test_wraps_success(self):
        @pipeline_step
        def success_fn():
            return 42

        result = success_fn()
        self.assertEqual(result, 42)

    def test_wraps_analyzer_error_returns_false(self):
        @pipeline_step
        def failing_fn():
            raise AnalyzerError("boom")

        result = failing_fn()
        self.assertFalse(result)

    def test_wraps_data_source_error_returns_false(self):
        @pipeline_step
        def failing_fn():
            raise DataSourceError("no data")

        result = failing_fn()
        self.assertFalse(result)

    def test_non_analyzer_error_propagates(self):
        @pipeline_step
        def bad_fn():
            raise ValueError("raw error")

        with self.assertRaises(ValueError):
            bad_fn()


class TestTickerAnalysisParams(unittest.TestCase):

    def test_constructs_with_defaults(self):
        p = TickerAnalysisParams(ticker="AAPL", year=2024)
        self.assertEqual(p.ticker, "AAPL")
        self.assertEqual(p.year, 2024)
        self.assertEqual(p.horizon, 90)
        self.assertEqual(p.threshold, 5.0)

    def test_constructs_with_all_fields(self):
        p = TickerAnalysisParams(ticker="GOOGL", year=2025, horizon=30, threshold=10.0)
        self.assertEqual(p.ticker, "GOOGL")
        self.assertEqual(p.year, 2025)
        self.assertEqual(p.horizon, 30)
        self.assertEqual(p.threshold, 10.0)


class TestPrepareAnalysisData(unittest.TestCase):

    @patch("analyzer.pipeline.analysis.calculate_signal_potential")
    def test_prepare_analysis_data(self, mock_calc_signals):
        mock_transactions = pd.DataFrame({
            "member": ["Alice"],
            "ticker": ["AAPL"],
            "disclosure_date": pd.to_datetime(["2024-01-15"]),
            "transaction_type": ["Purchase"],
        })

        mock_prices = pd.DataFrame({
            "AAPL": [100.0],
            "SPY": [200.0],
        }, index=pd.to_datetime(["2024-01-15"]))

        mock_entry_prices = pd.DataFrame({
            "member": ["Alice"],
            "ticker": ["AAPL"],
            "disclosure_date": pd.to_datetime(["2024-01-15"]),
            "transaction_type": ["Purchase"],
            "entry_price": [100.0],
        })

        mock_signals = pd.DataFrame({
            "member": ["Alice"],
            "ticker": ["AAPL"],
            "signal_type": ["Purchase"],
            "horizon_days": [90],
        })

        mock_calc_signals.return_value = mock_signals

        mock_tx_source = MagicMock()
        mock_tx_source.get_transactions.return_value = mock_transactions
        mock_tx_source.db.get_entry_prices.return_value = mock_entry_prices

        mock_price_source = MagicMock()
        mock_price_source.get_prices.return_value = mock_prices

        trades, prices, signals = _prepare_analysis_data(mock_tx_source, mock_price_source, 2024, [90])

        mock_tx_source.get_transactions.assert_called_once_with(2024)
        mock_tx_source.db.get_entry_prices.assert_called_once()
        mock_price_source.get_prices.assert_called_once()

        self.assertEqual(len(trades), 1)
        self.assertEqual(len(signals), 1)
        mock_calc_signals.assert_called_once_with(mock_entry_prices, mock_prices, [90])

    @patch("analyzer.pipeline.analysis.calculate_signal_potential")
    def test_prepare_analysis_data_empty_trades_raises(self, mock_calc_signals):
        mock_tx_source = MagicMock()
        mock_tx_source.get_transactions.return_value = pd.DataFrame()

        mock_price_source = MagicMock()

        with self.assertRaises(DataSourceError):
            _prepare_analysis_data(mock_tx_source, mock_price_source, 2024, [90])

        mock_calc_signals.assert_not_called()


class TestSaveResults(unittest.TestCase):

    def test_console_output_no_exception(self):
        table = pd.DataFrame({
            "member": ["Alice"],
            "avg_spy_alpha_pct": [10.5],
            "bayes_win_prob": [0.8],
            "hit_rate_pct": [70.0],
            "sharpe_ratio": [1.2],
            "bayes_factor": [2.1],
            "purchase_trades": [5],
        })

        try:
            _save_results(table, "console", None, False, Path("/tmp"))
        except Exception as e:
            self.fail(f"_save_results raised {e} unexpectedly")

    def test_csv_output_creates_file(self):
        table = pd.DataFrame({
            "member": ["Alice"],
            "avg_spy_alpha_pct": [10.5],
            "bayes_win_prob": [0.8],
            "hit_rate_pct": [70.0],
            "sharpe_ratio": [1.2],
            "bayes_factor": [2.1],
            "purchase_trades": [5],
        })

        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            _save_results(table, "csv", None, False, data_dir)

            filepath = data_dir / "member_rankings.csv"
            self.assertTrue(filepath.exists())

            saved = pd.read_csv(filepath)
            self.assertEqual(len(saved), 1)
            self.assertIn("member", saved.columns)

    def test_csv_output_member_filter_filename(self):
        table = pd.DataFrame({
            "ticker": ["AAPL"],
            "spy_alpha_pct": [10.0],
        })

        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            _save_results(table, "csv", "John Doe", True, data_dir)

            filepath = data_dir / "john_doe_signals.csv"
            self.assertTrue(filepath.exists())

    def test_csv_output_show_signals_filename(self):
        table = pd.DataFrame({
            "member": ["Alice"],
            "spy_alpha_pct": [10.0],
        })

        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            _save_results(table, "csv", None, True, data_dir)

            filepath = data_dir / "top_signals.csv"
            self.assertTrue(filepath.exists())

    def test_fallback_columns(self):
        table = pd.DataFrame({
            "col_a": [1],
            "col_b": [2],
        })

        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            _save_results(table, "csv", None, False, data_dir)

            saved = pd.read_csv(data_dir / "member_rankings.csv")
            self.assertIn("col_a", saved.columns)
            self.assertIn("col_b", saved.columns)

    def test_sales_columns_included_in_display(self):
        table = pd.DataFrame({
            "member": ["Alice"],
            "avg_loss_avoided_pct": [12.5],
            "median_loss_avoided_pct": [10.0],
            "sale_trades": [3],
            "sharpe_ratio": [1.1],
            "bayes_win_prob": [0.75],
            "bayes_factor": [2.0],
            "avg_spy_alpha_pct": [8.0],
        })

        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            _save_results(table, "csv", None, False, data_dir)

            saved = pd.read_csv(data_dir / "member_rankings.csv")
            self.assertIn("avg_loss_avoided_pct", saved.columns)
            self.assertIn("median_loss_avoided_pct", saved.columns)
            self.assertIn("sale_trades", saved.columns)
            self.assertIn("avg_spy_alpha_pct", saved.columns)

    def test_sales_columns_excluded_ranking_columns(self):
        table = pd.DataFrame({
            "member": ["Alice"],
            "avg_loss_avoided_pct": [12.5],
            "median_loss_avoided_pct": [10.0],
            "sale_trades": [3],
            "sharpe_ratio": [1.1],
            "bayes_win_prob": [0.75],
            "bayes_factor": [2.0],
            "avg_spy_alpha_pct": [8.0],
        })

        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            _save_results(table, "csv", None, False, data_dir)

            saved = pd.read_csv(data_dir / "member_rankings.csv")
            self.assertNotIn("purchase_trades", saved.columns)
            self.assertNotIn("hit_rate_pct", saved.columns)


class TestAnalyzeBySector(unittest.TestCase):

    @patch("analyzer.pipeline.analysis.rank_members")
    @patch("analyzer.pipeline._load_sector_data")
    def test_returns_dataframe_with_sectors(self, mock_load_sector, mock_rank):
        mock_load_sector.return_value = pd.DataFrame({
            "ticker": ["AAPL", "GOOGL"],
            "sector": ["Technology", "Technology"],
        })

        mock_rank.return_value = pd.DataFrame({
            "member": ["Alice", "Bob"],
            "avg_spy_alpha_pct": [15.0, 10.0],
            "purchase_trades": [5, 3],
        })

        trades = pd.DataFrame({
            "ticker": ["AAPL", "GOOGL"],
            "disclosure_date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "transaction_type": ["Purchase", "Purchase"],
        })

        signals = pd.DataFrame({
            "ticker": ["AAPL", "AAPL", "AAPL", "GOOGL", "GOOGL", "GOOGL"],
            "member": ["Alice", "Bob", "Charlie", "Alice", "Bob", "Charlie"],
            "signal_type": ["Purchase"] * 6,
            "horizon_days": [90] * 6,
            "decayed_return_pct": [10.0, 8.0, 5.0, 7.0, 6.0, 4.0],
            "peak_potential_pct": [20.0, 15.0, 10.0, 12.0, 11.0, 8.0],
        })

        result = _analyze_by_sector(trades, signals, [90])

        self.assertIsNotNone(result)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("sector", result.columns)
        self.assertIn("top_member", result.columns)
        self.assertGreaterEqual(len(result), 1)

    @patch("analyzer.pipeline._load_sector_data")
    def test_returns_none_when_no_sector_data(self, mock_load_sector):
        mock_load_sector.return_value = pd.DataFrame(columns=["ticker", "sector"])

        trades = pd.DataFrame({
            "ticker": ["AAPL"],
            "disclosure_date": pd.to_datetime(["2024-01-01"]),
        })

        signals = pd.DataFrame({
            "ticker": ["AAPL"],
            "signal_type": ["Purchase"],
            "horizon_days": [90],
        })

        result = _analyze_by_sector(trades, signals, [90])
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
