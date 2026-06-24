"""Smoke tests for analyzer.cli module."""
import unittest

from typer.testing import CliRunner

from analyzer.cli import app, setup_logging


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


if __name__ == "__main__":
    unittest.main()