"""Tests for the fail-closed reconciliation-only Capitol Trades staging script."""

import io
import json
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import pandas as pd
import requests

from analyzer.capitol_trades import CapitolTradesError


class TestFetchCapitolRecon(unittest.TestCase):
    @patch("scripts.fetch_capitol_recon.CapitolTradesSource")
    def test_fetch_routes_core_and_requires_manifest(self, source_cls):
        from scripts import fetch_capitol_recon

        expected = pd.DataFrame([{"member": "Test", "chamber": "house"}])
        source = MagicMock()
        source.fetch_all_trades.return_value = expected
        source_cls.return_value.__enter__.return_value = source

        result = fetch_capitol_recon.fetch_all_trades(
            generation="run-1", output=Path("artifact.json")
        )

        self.assertIs(result, expected)
        source_cls.assert_called_once_with(
            data_dir="data", read_only=True, generation="run-1"
        )
        source.fetch_all_trades.assert_called_once_with()
        source.write_reconciliation_artifact.assert_called_once_with(
            Path("artifact.json")
        )

    def test_main_requires_output_and_generation(self):
        from scripts import fetch_capitol_recon

        with self.assertRaises(SystemExit) as raised:
            fetch_capitol_recon.main([])
        self.assertEqual(raised.exception.code, 2)

    def test_default_retry_marker_derived_from_output(self):
        from scripts import fetch_capitol_recon

        self.assertEqual(
            fetch_capitol_recon.default_retry_marker(Path("out/artifact.json")),
            Path("out/artifact.json.retry.json"),
        )

    def test_write_retry_marker_records_generation_and_failure(self):
        from scripts import fetch_capitol_recon

        with TemporaryDirectory() as tmp:
            marker = Path(tmp) / "capitol.retry.json"
            output = Path(tmp) / "capitol.json"
            fetch_capitol_recon.write_retry_marker(
                generation="run-1",
                output=output,
                retry_marker=marker,
                failure_reason="API request failed: 503 Server Error",
            )
            payload = json.loads(marker.read_text())
            self.assertEqual(payload["marker_type"], "capitol_trades_scheduled_retry")
            self.assertEqual(payload["schema_version"], 1)
            self.assertEqual(payload["ingestion_generation"], "run-1")
            self.assertEqual(payload["target_output"], str(output))
            self.assertIn("503", payload["failure_reason"])
            self.assertIn("created_at_utc", payload)

    @patch("scripts.fetch_capitol_recon.fetch_all_trades")
    def test_main_success_reports_artifact_and_clears_stale_marker(self, fetch):
        from scripts import fetch_capitol_recon

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
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "artifact.json"
            marker = Path(tmp) / "artifact.json.retry.json"
            marker.write_text("stale")
            with patch("builtins.print") as printer:
                fetch_capitol_recon.main(
                    [
                        "--output",
                        str(output),
                        "--generation",
                        "run-1",
                        "--retry-marker",
                        str(marker),
                    ]
                )

        fetch.assert_called_once_with(generation="run-1", output=output)
        self.assertFalse(marker.exists())
        printed = "\n".join(
            " ".join(map(str, call.args)) for call in printer.call_args_list
        )
        self.assertIn("Reconciliation artifact:", printed)
        self.assertIn(str(output), printed)
        self.assertIn("No canonical transactions were saved", printed)

    def test_main_503_fails_closed_no_artifact_and_writes_retry_marker(self):
        from scripts import fetch_capitol_recon

        response = requests.Response()
        response.status_code = 503
        response.reason = "Service Unavailable"
        response.url = "https://trades.telep.io/api/trades"

        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "capitol.json"
            marker = Path(tmp) / "capitol.retry.json"
            stderr = io.StringIO()
            with patch("requests.Session.get", return_value=response):
                with redirect_stderr(stderr):
                    with self.assertRaises(SystemExit) as raised:
                        fetch_capitol_recon.main(
                            [
                                "--output",
                                str(output),
                                "--generation",
                                "run-1",
                                "--retry-marker",
                                str(marker),
                            ]
                        )
            self.assertEqual(raised.exception.code, 1)
            self.assertIn(
                "FAILED (no reconciliation artifact written)", stderr.getvalue()
            )
            self.assertIn("Scheduled retry marker written", stderr.getvalue())
            self.assertFalse(output.exists())
            self.assertTrue(marker.exists())
            payload = json.loads(marker.read_text())
            self.assertEqual(payload["marker_type"], "capitol_trades_scheduled_retry")
            self.assertEqual(payload["ingestion_generation"], "run-1")
            self.assertEqual(payload["target_output"], str(output))
            self.assertIn("503", payload["failure_reason"])

    @patch("scripts.fetch_capitol_recon.fetch_all_trades")
    def test_main_failure_with_existing_artifact_skips_retry_marker(self, fetch):
        from scripts import fetch_capitol_recon

        fetch.side_effect = CapitolTradesError(
            "Refusing to overwrite reconciliation artifact"
        )
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "capitol.json"
            marker = Path(tmp) / "capitol.retry.json"
            output.write_text("{}")
            with self.assertRaises(SystemExit) as raised:
                fetch_capitol_recon.main(
                    [
                        "--output",
                        str(output),
                        "--generation",
                        "run-1",
                        "--retry-marker",
                        str(marker),
                    ]
                )
            self.assertEqual(raised.exception.code, 1)
            self.assertTrue(output.exists())
            self.assertFalse(marker.exists())


if __name__ == "__main__":
    unittest.main()
