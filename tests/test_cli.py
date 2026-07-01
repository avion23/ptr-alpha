"""Smoke tests for analyzer.cli module."""
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from typer.testing import CliRunner

from analyzer.cli import app, setup_logging
from analyzer.pipeline import AnalysisParams, BacktestParams


class TestSetupLogging(unittest.TestCase):

    def test_setup_logging_no_error(self):
        setup_logging(verbose=False)
        setup_logging(verbose=True)


class TestCliApp(unittest.TestCase):

    def setUp(self):
        self.runner = CliRunner()

    def test_app_help(self):
        result = self.runner.invoke(app, ["--help"])
        self.assertEqual(result.exit_code, 0)
        # Help should mention the analyzer commands
        self.assertIn("analyze", result.stdout)
        self.assertIn("fetch", result.stdout)

    def test_analyze_help(self):
        result = self.runner.invoke(app, ["analyze", "--help"])
        self.assertEqual(result.exit_code, 0)

    def test_fetch_help(self):
        result = self.runner.invoke(app, ["fetch", "--help"])
        self.assertEqual(result.exit_code, 0)

    def test_parse_help(self):
        result = self.runner.invoke(app, ["parse", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("--gemini-ocr", result.output)

    def test_analyze_invalid_mode(self):
        result = self.runner.invoke(app, ["analyze", "--mode", "invalid_mode"])
        self.assertNotEqual(result.exit_code, 0)

    def test_analyze_help_shows_valid_modes(self):
        result = self.runner.invoke(app, ["analyze", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("ranks", result.output)

    def test_backtest_cli_defaults_match_dataclass_defaults(self):
        captured = {}

        def fake_pipeline(params, tx_source, price_source, data_path):
            captured["params"] = params
            return True

        with patch("analyzer.cli.run_backtest_pipeline", side_effect=fake_pipeline), \
             patch("analyzer.cli.get_context", return_value=MagicMock()):
            result = self.runner.invoke(app, ["backtest", "--start", "2024-01-01", "--end", "2024-02-01"])

        self.assertEqual(result.exit_code, 0, result.output)
        params = captured.get("params")
        self.assertIsNotNone(params)
        for field_name in (
            "horizon",
            "lookback_days",
            "training_lookback_days",
            "min_buyers",
            "top_n",
            "threshold",
            "frequency_days",
        ):
            self.assertEqual(
                getattr(params, field_name),
                BacktestParams.__dataclass_fields__[field_name].default,
                field_name,
            )

    def test_parse_fails_when_pipeline_fails_even_if_ocr_inserts_rows(self):
        mock_ctx = MagicMock()
        mock_ctx.settings.data.data_dir = "data"

        with patch("analyzer.cli.get_context", return_value=mock_ctx), \
             patch("analyzer.cli.run_parse_pipeline", return_value=False), \
             patch("scripts.ocr_zero_rows.run_gemini_ocr_for_year", return_value=3):
            result = self.runner.invoke(app, ["parse", "--gemini-ocr"])

        self.assertEqual(result.exit_code, 1, result.output)

    def test_ticker_csv_output_warns_not_supported(self):
        mock_ctx = MagicMock()
        mock_ctx.transaction_source.db.conn.execute.return_value.fetchone.return_value = (None,)

        with patch("analyzer.cli.get_context", return_value=mock_ctx), \
             patch("analyzer.cli.run_ticker_analysis", return_value=True):
            result = self.runner.invoke(app, ["analyze", "--ticker", "AAPL", "--output", "csv"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("CSV output is not supported for --ticker analysis", result.output)

    def test_tickers_mode_csv_output_warns_not_supported(self):
        mock_ctx = MagicMock()
        mock_ctx.transaction_source.db.conn.execute.return_value.fetchone.return_value = (None,)

        with patch("analyzer.cli.get_context", return_value=mock_ctx), \
             patch("analyzer.cli.run_recent_ticker_scoring", return_value=True):
            result = self.runner.invoke(app, ["analyze", "--mode", "tickers", "--output", "csv"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("CSV output is not supported for --mode tickers", result.output)


class TestAnalyzeParamsMapping(unittest.TestCase):
    """Regression tests for BUG 1: AnalysisParams positional-arg mismatch in CLI."""

    def setUp(self):
        self.runner = CliRunner()

    def _invoke_analyze(self, extra_args=None):
        """Invoke `analyze` with run_analysis_pipeline stubbed out.

        Returns the AnalysisParams instance that was passed to the stub.
        """
        captured = {}

        def fake_pipeline(params, tx_source, price_source, data_path, output_format):
            captured["params"] = params
            return True

        # Also stub get_context so we don't need a real database.
        mock_ctx = MagicMock()
        mock_ctx.transaction_source.db.conn.execute.return_value.fetchone.return_value = (None,)

        with patch("analyzer.cli.run_analysis_pipeline", side_effect=fake_pipeline), \
             patch("analyzer.cli.get_context", return_value=mock_ctx):
            args = ["analyze"] + (extra_args or [])
            result = self.runner.invoke(app, args)

        return result, captured.get("params")

    def test_default_member_filter_is_none(self):
        """member_filter must be None when --member is not supplied."""
        result, params = self._invoke_analyze()
        self.assertIsNotNone(params, f"pipeline was not called; exit={result.exit_code}, output={result.output}")
        self.assertIsNone(params.member_filter,
                          f"member_filter landed in wrong slot; got {params.member_filter!r}")

    def test_top_n_is_correct(self):
        """top_n must be the integer passed via --top-n, not the member string."""
        result, params = self._invoke_analyze(["--top-n", "20"])
        self.assertIsNotNone(params, f"pipeline was not called; exit={result.exit_code}, output={result.output}")
        self.assertEqual(params.top_n, 20,
                         f"top_n landed in wrong slot; got {params.top_n!r}")
        self.assertIsNone(params.member_filter,
                          f"member_filter must be None, got {params.member_filter!r}")

    def test_member_filter_set_when_member_supplied(self):
        """When --member is given, member_filter must carry that value."""
        result, params = self._invoke_analyze(["--mode", "member", "--member", "Nancy Pelosi"])
        self.assertIsNotNone(params, f"pipeline was not called; exit={result.exit_code}, output={result.output}")
        self.assertEqual(params.member_filter, "Nancy Pelosi",
                         f"member_filter was not set; got {params.member_filter!r}")

    def test_show_signals_false_for_ranks_mode(self):
        """show_signals must be False for default ranks mode."""
        result, params = self._invoke_analyze(["--mode", "ranks"])
        self.assertIsNotNone(params, f"pipeline was not called; exit={result.exit_code}, output={result.output}")
        self.assertFalse(params.show_signals,
                         f"show_signals must be False for ranks mode, got {params.show_signals!r}")

    def test_show_signals_true_for_signals_mode(self):
        """show_signals must be True for signals mode."""
        result, params = self._invoke_analyze(["--mode", "signals"])
        self.assertIsNotNone(params, f"pipeline was not called; exit={result.exit_code}, output={result.output}")
        self.assertTrue(params.show_signals,
                        f"show_signals must be True for signals mode, got {params.show_signals!r}")

    def test_source_field_unaffected(self):
        """source must keep its default value 'house' regardless of --member."""
        result, params = self._invoke_analyze(["--member", "some member"])
        self.assertIsNotNone(params)
        self.assertEqual(params.source, "house",
                         f"source field was corrupted; got {params.source!r}")

    def test_freshness_check_uses_disclosure_date(self):
        captured_sql = {}

        def fake_execute(sql):
            captured_sql["sql"] = sql
            result = MagicMock()
            result.fetchone.return_value = (None,)
            return result

        mock_ctx = MagicMock()
        mock_ctx.transaction_source.db.conn.execute.side_effect = fake_execute

        with patch("analyzer.cli.run_analysis_pipeline", return_value=True), \
             patch("analyzer.cli.get_context", return_value=mock_ctx):
            result = self.runner.invoke(app, ["analyze"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("MAX(disclosure_date)", captured_sql["sql"])
        self.assertNotIn("transaction_date", captured_sql["sql"])


if __name__ == "__main__":
    unittest.main()
