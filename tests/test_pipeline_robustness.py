from unittest.mock import MagicMock, patch

import pandas as pd
from typer.testing import CliRunner

from analyzer.cli import app
from analyzer.database import Database
from analyzer.exceptions import StepResult


def test_refresh_summary_is_scoped_to_requested_year(tmp_path):
    db = Database(tmp_path / "multi-year.duckdb")
    db.upsert_transactions(pd.DataFrame([
        {
            "doc_id": "2024-valid", "member": "A", "ticker": "AAA",
            "transaction_date": "2024-12-30", "disclosure_date": "2024-12-31",
            "transaction_type": "Purchase",
        },
        {
            "doc_id": "2024-invalid", "member": "B", "ticker": "BBB",
            "transaction_date": "2025-01-02", "disclosure_date": "2024-12-29",
            "transaction_type": "Sale",
        },
        {
            "doc_id": "2025-valid", "member": "C", "ticker": "CCC",
            "transaction_date": "2025-12-30", "disclosure_date": "2025-12-31",
            "transaction_type": "Purchase",
        },
    ]), source="house_pdf")
    ctx = MagicMock()
    ctx.transaction_source.db = db

    try:
        with patch("analyzer.cli.get_context", return_value=ctx), \
             patch("analyzer.cli.run_fetch_pipeline", return_value=StepResult(success=True)), \
             patch("analyzer.cli.run_parse_pipeline", return_value=StepResult(success=True)):
            result = CliRunner().invoke(
                app, ["refresh", "--year", "2024", "--skip-capitol"]
            )
    finally:
        db.close()

    assert result.exit_code == 0, result.output
    assert "Latest transaction date: 2024-12-30" in result.output
    assert "Latest disclosure date: 2024-12-31" in result.output
    assert "Excluded from analyses: 1 transaction(s)" in result.output
    assert "2025-12-30" not in result.output
    assert "2025-12-31" not in result.output
