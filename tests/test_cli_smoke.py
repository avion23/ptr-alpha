import unittest

from typer.testing import CliRunner

from analyzer.cli import app

runner = CliRunner()


class TestCliSmoke(unittest.TestCase):
    def test_help_lists_commands(self):
        result = runner.invoke(app, ["--help"])
        self.assertEqual(result.exit_code, 0)
        for command in (
            "health",
            "analyze",
            "fetch",
            "parse",
            "backtest",
            "refresh",
            "fetch-senate-efd",
        ):
            self.assertIn(command, result.output)

    def test_each_command_help(self):
        for command in (
            "health",
            "analyze",
            "fetch",
            "parse",
            "backtest",
            "refresh",
            "fetch-senate-efd",
        ):
            result = runner.invoke(app, [command, "--help"])
            self.assertEqual(result.exit_code, 0, command)
            self.assertIn("Usage", result.output)


if __name__ == "__main__":
    unittest.main()
