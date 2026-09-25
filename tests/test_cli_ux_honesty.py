"""Regression tests for CLI UX honesty: horizons and ticker csv output.

No database is opened: every rejection happens before get_context.
"""

import unittest
from unittest.mock import patch

from typer.testing import CliRunner

from analyzer.cli import app


class TestAnalyzeHorizonsHonesty(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()

    def test_multiple_horizons_rejected_before_db_open(self):
        with patch("analyzer.cli.get_context") as context:
            result = self.runner.invoke(
                app, ["analyze", "--horizons", "30", "--horizons", "90"]
            )
        self.assertEqual(result.exit_code, 1, result.output)
        self.assertIn("single value", result.output)
        context.assert_not_called()

    def test_single_horizon_still_accepted(self):
        with (
            patch("analyzer.cli.get_context", return_value=None),
            patch("analyzer.cli._run_analysis_mode") as run_analysis,
        ):
            result = self.runner.invoke(app, ["analyze", "--horizons", "30"])
        self.assertEqual(result.exit_code, 0, result.output)
        run_analysis.assert_called_once()


class TestTickerCsvFailsLoudly(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()

    def test_tickers_mode_csv_rejected_before_db_open(self):
        with patch("analyzer.cli.get_context") as context:
            result = self.runner.invoke(
                app, ["analyze", "--mode", "tickers", "--output", "csv"]
            )
        self.assertEqual(result.exit_code, 1, result.output)
        self.assertIn("not supported", result.output)
        self.assertNotIn("WARNING", result.output)
        context.assert_not_called()

    def test_single_ticker_csv_rejected_before_db_open(self):
        with patch("analyzer.cli.get_context") as context:
            result = self.runner.invoke(
                app, ["analyze", "--ticker", "AAPL", "--output", "csv"]
            )
        self.assertEqual(result.exit_code, 1, result.output)
        self.assertIn("not supported", result.output)
        self.assertNotIn("WARNING", result.output)
        context.assert_not_called()

    def test_tickers_help_documents_fixed_lookback(self):
        result = self.runner.invoke(app, ["analyze", "--help"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("28-day", result.output)


if __name__ == "__main__":
    unittest.main()
