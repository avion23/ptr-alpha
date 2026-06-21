"""Smoke tests for member_profitability.data module."""
import unittest


class TestDataImports(unittest.TestCase):

    def test_module_imports(self):
        import member_profitability.data
        self.assertTrue(callable(member_profitability.data.load_transactions_and_prices))
        self.assertTrue(callable(member_profitability.data.compute_signals))
        self.assertTrue(callable(member_profitability.data.print_loaded_data))


class TestPrintLoadedData(unittest.TestCase):

    def test_does_not_raise(self):
        import io
        import contextlib
        import time

        import pandas as pd

        from member_profitability.data import print_loaded_data

        all_tx = pd.DataFrame({
            "ticker": ["A", "B"],
            "member": ["m1", "m2"],
            "disclosure_date": pd.to_datetime(["2024-01-01", "2024-02-01"]),
        })
        all_tickers = ["A", "B", "SPY"]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            print_loaded_data(t0=time.time(), all_tx=all_tx, all_tickers=all_tickers)
        out = buf.getvalue()
        self.assertIn("Data loaded", out)
        self.assertIn("Transactions: 2", out)
        self.assertIn("Tickers: 3", out)


if __name__ == "__main__":
    unittest.main()
