"""Tests for the reconciliation-only Capitol Trades script."""

import unittest
from unittest.mock import MagicMock, patch

import pandas as pd


class TestFetchCapitolTrades(unittest.TestCase):
    @patch("scripts.fetch_capitol_trades.CapitolTradesSource")
    def test_fetch_routes_through_core_client(self, source_cls):
        from scripts import fetch_capitol_trades

        expected = pd.DataFrame([{"member": "Test", "chamber": "house"}])
        source = MagicMock()
        source.fetch_all_trades.return_value = expected
        source_cls.return_value.__enter__.return_value = source

        result = fetch_capitol_trades.fetch_all_trades()

        self.assertIs(result, expected)
        source_cls.assert_called_once_with(data_dir="data", read_only=True)
        source.fetch_all_trades.assert_called_once_with()

    @patch("scripts.fetch_capitol_trades.fetch_all_trades")
    def test_main_reports_reconciliation_only_and_does_not_save(self, fetch):
        from scripts import fetch_capitol_trades

        fetch.return_value = pd.DataFrame(
            [
                {
                    "member": "Test Member",
                    "ticker": "AAPL",
                    "transaction_date": pd.Timestamp("2025-01-02"),
                    "chamber": "house",
                    "transaction_type": "Purchase",
                }
            ]
        )

        with patch("builtins.print") as printer:
            fetch_capitol_trades.main()

        output = "\n".join(
            " ".join(map(str, call.args)) for call in printer.call_args_list
        )
        self.assertIn("Reconciliation only", output)
        self.assertIn("no canonical transactions were saved", output)


if __name__ == "__main__":
    unittest.main()
