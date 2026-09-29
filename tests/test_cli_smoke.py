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
            "fetch-form4",
            "fetch-13f",
            "follow-member",
            "setups",
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
            "fetch-form4",
            "fetch-13f",
            "follow-member",
            "setups",
        ):
            result = runner.invoke(app, [command, "--help"])
            self.assertEqual(result.exit_code, 0, command)
            self.assertIn("Usage", result.output)

    def test_setups_help_and_missing_component_are_clean(self):
        help_result = runner.invoke(app, ["setups", "--help"])
        self.assertEqual(help_result.exit_code, 0)
        self.assertIn("--member", help_result.output)
        self.assertIn("--ticker", help_result.output)

        result = runner.invoke(app, ["setups", "--member", "Bogus Member"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("No candidates.", result.output)
        self.assertNotIn("Traceback", result.output)


if __name__ == "__main__":
    unittest.main()
