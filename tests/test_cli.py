"""Smoke tests for analyzer.cli module."""
import unittest
from unittest.mock import patch, MagicMock

from typer.testing import CliRunner

from analyzer.cli import app
from analyzer.exceptions import StepResult




class TestCliApp(unittest.TestCase):

    def setUp(self):
        self.runner = CliRunner()





    def test_analyze_invalid_mode(self):
        result = self.runner.invoke(app, ["analyze", "--mode", "invalid_mode"])
        self.assertNotEqual(result.exit_code, 0)


    def test_analyze_rejects_invalid_numeric_and_output_options_before_db_open(self):
        cases = [
            ["--horizons", "0"],
            ["--days-back", "0"],
            ["--min-buyers", "0"],
            ["--top-n", "0"],
            ["--output", "json"],
        ]
        for args in cases:
            with self.subTest(args=args), patch("analyzer.cli.get_context") as context:
                result = self.runner.invoke(app, ["analyze", *args])
                self.assertEqual(result.exit_code, 1, result.output)
                self.assertIn("Error:", result.output)
                context.assert_not_called()

    def test_backtest_rejects_nonpositive_windows_before_db_open(self):
        for option in ("--horizon", "--lookback-days", "--training-lookback-days",
                       "--min-buyers", "--top-n", "--frequency-days"):
            with self.subTest(option=option), patch("analyzer.cli.get_context") as context:
                result = self.runner.invoke(app, [
                    "backtest", "--start", "2024-01-01", "--end", "2024-02-01",
                    option, "0",
                ])
                self.assertEqual(result.exit_code, 1, result.output)
                context.assert_not_called()

    def test_portfolio_rejects_nonpositive_constraints_before_db_open(self):
        for option in ("--horizon", "--lookback-days", "--training-lookback-days",
                       "--min-buyers", "--top-n", "--frequency-days",
                       "--initial-capital", "--max-positions", "--hold-days"):
            with self.subTest(option=option), patch("analyzer.cli.get_context") as context:
                result = self.runner.invoke(app, [
                    "portfolio", "--start", "2024-01-01", "--end", "2024-02-01",
                    option, "0",
                ])
                self.assertEqual(result.exit_code, 1, result.output)
                context.assert_not_called()

    def test_parse_fails_when_pipeline_fails_even_if_ocr_inserts_rows(self):
        mock_ctx = MagicMock()
        mock_ctx.settings.data.data_dir = "data"

        with patch("analyzer.cli.get_context", return_value=mock_ctx), \
             patch("analyzer.cli.run_parse_pipeline", return_value=StepResult(success=False)), \
             patch("scripts.ocr_zero_rows.run_gemini_ocr_for_year", return_value=3):
            result = self.runner.invoke(app, ["parse", "--gemini-ocr"])

        self.assertEqual(result.exit_code, 1, result.output)


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
            return StepResult(success=True)

        # Also stub get_context so we don't need a real database.
        mock_ctx = MagicMock()
        mock_ctx.transaction_source.db.conn.execute.return_value.fetchone.return_value = (None,)

        with patch("analyzer.cli.run_analysis_pipeline", side_effect=fake_pipeline), \
             patch("analyzer.cli.get_context", return_value=mock_ctx):
            args = ["analyze"] + (extra_args or [])
            result = self.runner.invoke(app, args)

        return result, captured.get("params")


if __name__ == "__main__":
    unittest.main()
