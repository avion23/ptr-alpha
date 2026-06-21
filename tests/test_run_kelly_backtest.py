"""Smoke tests for scripts.run_kelly_backtest module."""
import unittest


class TestRunKellyBacktest(unittest.TestCase):

    def test_module_imports(self):
        from scripts import run_kelly_backtest
        assert callable(run_kelly_backtest.main)


if __name__ == "__main__":
    unittest.main()