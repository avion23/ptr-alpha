"""Smoke tests for analyzer.cli module."""

import unittest
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
from typer.testing import CliRunner

from analyzer.cli import _check_data_freshness, _load_sector_map, app
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
        for option in (
            "--horizon",
            "--lookback-days",
            "--training-lookback-days",
            "--min-buyers",
            "--top-n",
            "--frequency-days",
        ):
            with (
                self.subTest(option=option),
                patch("analyzer.cli.get_context") as context,
            ):
                result = self.runner.invoke(
                    app,
                    [
                        "backtest",
                        "--start",
                        "2024-01-01",
                        "--end",
                        "2024-02-01",
                        option,
                        "0",
                    ],
                )
                self.assertEqual(result.exit_code, 1, result.output)
                context.assert_not_called()

    def test_portfolio_rejects_nonpositive_constraints_before_db_open(self):
        for option in (
            "--horizon",
            "--lookback-days",
            "--training-lookback-days",
            "--min-buyers",
            "--top-n",
            "--frequency-days",
            "--initial-capital",
            "--max-positions",
            "--hold-days",
        ):
            with (
                self.subTest(option=option),
                patch("analyzer.cli.get_context") as context,
            ):
                result = self.runner.invoke(
                    app,
                    [
                        "portfolio",
                        "--start",
                        "2024-01-01",
                        "--end",
                        "2024-02-01",
                        option,
                        "0",
                    ],
                )
                self.assertEqual(result.exit_code, 1, result.output)
                context.assert_not_called()

    def test_fetch_capitol_requires_artifact_and_generation(self):
        with patch("analyzer.capitol_trades.CapitolTradesSource") as source:
            result = self.runner.invoke(app, ["fetch-capitol", "--all"])
        self.assertNotEqual(result.exit_code, 0)
        source.assert_not_called()

    def test_fetch_capitol_writes_reconciliation_artifact_without_database_save(self):
        frame = pd.DataFrame([{"member": "Test", "ticker": "AAPL"}])
        capitol = MagicMock()
        capitol.fetch_all_trades.return_value = frame
        with patch(
            "analyzer.capitol_trades.CapitolTradesSource", return_value=capitol
        ) as source:
            result = self.runner.invoke(
                app,
                [
                    "fetch-capitol",
                    "--all",
                    "--output",
                    "capitol.json",
                    "--generation",
                    "run-1",
                ],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        source.assert_called_once_with(
            data_dir="data", read_only=True, generation="run-1"
        )
        capitol.fetch_all_trades.assert_called_once_with(None, None, None)
        capitol.write_reconciliation_artifact.assert_called_once_with(
            Path("capitol.json")
        )
        capitol.save_to_db.assert_not_called()
        self.assertIn("No canonical transactions were saved", result.output)

    def test_official_refresh_never_constructs_capitol_source(self):
        mock_ctx = MagicMock()
        fetchone = mock_ctx.transaction_source.db.conn.execute.return_value.fetchone
        fetchone.side_effect = [
            (10,),
            (10,),
            (10,),
            (10,),
            (date(2026, 8, 1), date(2026, 8, 2), 0),
        ]
        mock_ctx.transaction_source.fetch_and_cache_pdfs.return_value = MagicMock(
            archive_year=2026,
            metadata_count=10,
            ptr_count=10,
            valid_pdf_count=10,
            downloaded_count=0,
            skipped_count=10,
            orphan_pdf_count=0,
            removed_doc_count=0,
            quarantined_pdf_count=0,
            generation_id="generation-2026",
            generation_status="acquired",
        )
        mock_ctx.transaction_source.db.get_unresolved_house_doc_ids.return_value = []
        mock_ctx.transaction_source.db.get_latest_house_generation.return_value = (
            "generation-2026"
        )
        with (
            patch("analyzer.cli.get_context", return_value=mock_ctx),
            patch(
                "analyzer.cli.run_parse_pipeline",
                return_value=StepResult(success=True),
            ),
            patch("analyzer.capitol_trades.CapitolTradesSource") as capitol,
        ):
            result = self.runner.invoke(app, ["refresh", "--year", "2026"])

        self.assertEqual(result.exit_code, 0, result.output)
        capitol.assert_not_called()
        self.assertIn("Excluding Capitol Trades from official refresh", result.output)

    def test_portfolio_requires_sector_map_before_db_open(self):
        with patch("analyzer.cli.get_context") as context:
            result = self.runner.invoke(
                app,
                [
                    "portfolio",
                    "--start",
                    "2024-01-01",
                    "--end",
                    "2024-02-01",
                ],
            )
        self.assertEqual(result.exit_code, 1, result.output)
        self.assertIn("--sector-map is required", result.output)
        context.assert_not_called()

    def test_sector_map_loader_validates_deterministic_json(self):
        with self.runner.isolated_filesystem():
            Path("sectors.json").write_text('{"AAPL": "Technology"}')
            self.assertEqual(_load_sector_map("sectors.json"), {"AAPL": "Technology"})
            Path("bad.json").write_text('{"AAPL": ""}')
            with self.assertRaisesRegex(ValueError, "blank ticker or sector"):
                _load_sector_map("bad.json")

    def test_portfolio_fails_before_simulation_when_sector_ticker_missing(self):
        mock_ctx = MagicMock()
        mock_ctx.transaction_source.db.get_transactions_by_date_range.return_value = (
            pd.DataFrame({"ticker": ["B"]})
        )
        recommendations = pd.DataFrame(
            {
                "ticker": ["B"],
                "signal_score": [1.0],
                "as_of_date": [pd.Timestamp("2024-01-01")],
            }
        )
        with self.runner.isolated_filesystem():
            Path("sectors.json").write_text('{"A": "Technology"}')
            with (
                patch("analyzer.cli.get_context", return_value=mock_ctx),
                patch(
                    "analyzer.cli._load_portfolio_inputs",
                    return_value=(
                        pd.DataFrame(),
                        pd.DataFrame(),
                        pd.DataFrame(),
                        recommendations,
                    ),
                ),
                patch("analyzer.portfolio_sim.PortfolioSimulator") as simulator,
            ):
                result = self.runner.invoke(
                    app,
                    [
                        "portfolio",
                        "--start",
                        "2024-01-01",
                        "--end",
                        "2024-02-01",
                        "--sector-map",
                        "sectors.json",
                    ],
                )
        self.assertEqual(result.exit_code, 1, result.output)
        self.assertIn("sector map is missing 1 recommended ticker", result.output)
        simulator.assert_not_called()


    def test_parse_fails_when_pipeline_fails_even_if_ocr_inserts_rows(self):
        mock_ctx = MagicMock()
        mock_ctx.settings.data.data_dir = "data"

        with (
            patch("analyzer.cli.get_context", return_value=mock_ctx),
            patch(
                "analyzer.cli.run_parse_pipeline",
                return_value=StepResult(success=False),
            ),
            patch("scripts.ocr_zero_rows.run_gemini_ocr_for_year", return_value=3),
        ):
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
        mock_ctx.transaction_source.db.conn.execute.return_value.fetchone.return_value = (
            None,
        )

        with (
            patch("analyzer.cli.run_analysis_pipeline", side_effect=fake_pipeline),
            patch("analyzer.cli.get_context", return_value=mock_ctx),
        ):
            args = ["analyze"] + (extra_args or [])
            result = self.runner.invoke(app, args)

        return result, captured.get("params")


if __name__ == "__main__":
    unittest.main()



def test_data_freshness_reads_only_canonical_scope(capsys):
    connection = MagicMock()
    connection.execute.return_value.fetchone.return_value = (date.today(),)
    context = SimpleNamespace(
        transaction_source=SimpleNamespace(
            db=SimpleNamespace(conn=connection)
        )
    )

    _check_data_freshness(context)

    query = connection.execute.call_args.args[0]
    assert "canonical_transactions" in query
    assert "WARNING" not in capsys.readouterr().err



def test_reconcile_blotter_reads_only_canonical_scope(monkeypatch):
    from scripts import reconcile_blotter

    connection = MagicMock()
    connection.execute.return_value.fetchall.return_value = []
    monkeypatch.setattr(reconcile_blotter.duckdb, "connect", lambda *a, **k: connection)

    assert reconcile_blotter.get_congressional("AAPL") == []
    query = connection.execute.call_args.args[0]
    assert "canonical_transactions" in query
    connection.close.assert_called_once()
