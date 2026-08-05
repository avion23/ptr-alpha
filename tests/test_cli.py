"""Smoke tests for analyzer.cli module."""
import importlib
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from typer.testing import CliRunner

from analyzer.cli import app, _print_portfolio_metrics
from analyzer.exceptions import StepResult

REPO_ROOT = Path(__file__).resolve().parents[1]




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

    def test_portfolio_metrics_do_not_report_zero_performance_without_closed_trades(self):
        metrics = {
            "total_return_pct": 1.0, "annualized_return_pct": 2.0,
            "sharpe_ratio": 0.5, "max_drawdown_pct": -1.0,
            "win_rate_pct": 0.0, "avg_holding_days": 0.0,
            "turnover_rate": 0.0, "max_concurrent_positions": 2,
            "total_closed_trades": 0, "spy_return_pct": None,
            "sector_concentration": {},
        }
        with patch("builtins.print") as printer:
            _print_portfolio_metrics(metrics)
        output = "\n".join(call.args[0] for call in printer.call_args_list)
        self.assertIn("N/A (no closed trades)", output)
        self.assertNotIn("Win rate:           0.0%", output)


    def test_parse_fails_when_pipeline_fails_even_if_ocr_inserts_rows(self):
        mock_ctx = MagicMock()
        mock_ctx.settings.data.data_dir = "data"

        with patch("analyzer.cli.get_context", return_value=mock_ctx), \
             patch("analyzer.cli.run_parse_pipeline", return_value=StepResult(success=False)), \
             patch("scripts.ocr_zero_rows.run_gemini_ocr_for_year", return_value=3):
            result = self.runner.invoke(app, ["parse", "--gemini-ocr"])

        self.assertEqual(result.exit_code, 1, result.output)

    def test_gemini_ocr_scripts_package_is_installed_with_cli(self):
        """Regression: `ptr-alpha parse --gemini-ocr` imports
        `scripts.ocr_zero_rows.run_gemini_ocr_for_year` at runtime. When the
        package is installed via `pip install .` (no repo root on sys.path),
        this import only resolves if `scripts` is declared as an installable
        package in pyproject.toml AND has an `__init__.py`. A namespace
        package (no __init__.py) is NOT installed by setuptools and the CLI
        crashes with ModuleNotFoundError. This test pins both requirements.
        """
        # 1. scripts must be a real (regular) package, not a namespace pkg.
        scripts_pkg = importlib.import_module("scripts")
        self.assertIsNotNone(
            getattr(scripts_pkg, "__file__", None),
            "scripts/ must have an __init__.py so setuptools ships it; "
            "a namespace package is not installed and breaks `ptr-alpha --gemini-ocr`.",
        )

        # 2. the symbol the CLI imports must resolve.
        from scripts.ocr_zero_rows import run_gemini_ocr_for_year
        self.assertTrue(callable(run_gemini_ocr_for_year))

        # 3. pyproject.toml must declare scripts in package discovery.
        pyproject = (REPO_ROOT / "pyproject.toml").read_text()
        self.assertIn("[tool.setuptools.packages.find]", pyproject)
        self.assertIn("scripts*", pyproject)







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
