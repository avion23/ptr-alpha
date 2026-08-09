"""Tests for the reconciliation-only Capitol Trades script."""

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd


class TestFetchCapitolTrades(unittest.TestCase):
    @patch("scripts.fetch_capitol_trades.CapitolTradesSource")
    def test_fetch_routes_core_and_requires_manifest(self, source_cls):
        from scripts import fetch_capitol_trades

        expected = pd.DataFrame([{"member": "Test", "chamber": "house"}])
        source = MagicMock()
        source.fetch_all_trades.return_value = expected
        source_cls.return_value.__enter__.return_value = source

        result = fetch_capitol_trades.fetch_all_trades(
            generation="run-1", output=Path("artifact.json")
        )

        self.assertIs(result, expected)
        source_cls.assert_called_once_with(
            data_dir="data", read_only=True, generation="run-1"
        )
        source.fetch_all_trades.assert_called_once_with()
        source.write_reconciliation_artifact.assert_called_once_with(
            expected, Path("artifact.json")
        )

    def test_main_requires_output_and_generation(self):
        from scripts import fetch_capitol_trades

        with self.assertRaises(SystemExit) as raised:
            fetch_capitol_trades.main([])
        self.assertEqual(raised.exception.code, 2)

    @patch("scripts.fetch_capitol_trades.fetch_all_trades")
    def test_main_reports_artifact_and_no_canonical_save(self, fetch):
        from scripts import fetch_capitol_trades

        fetch.return_value = pd.DataFrame(
            [
                {
                    "member": "Test Member",
                    "ticker": "AAPL",
                    "transaction_date": pd.Timestamp("2025-01-02"),
                    "chamber": "house",
                }
            ]
        )
        with patch("builtins.print") as printer:
            fetch_capitol_trades.main(
                ["--output", "artifact.json", "--generation", "run-1"]
            )

        fetch.assert_called_once_with(generation="run-1", output=Path("artifact.json"))
        output = "\n".join(
            " ".join(map(str, call.args)) for call in printer.call_args_list
        )
        self.assertIn("Reconciliation artifact: artifact.json", output)
        self.assertIn("No canonical transactions were saved", output)


if __name__ == "__main__":
    unittest.main()
