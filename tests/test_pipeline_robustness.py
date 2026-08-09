from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
from typer.testing import CliRunner

from analyzer.cli import app
from analyzer.database import Database
from analyzer.download import HouseFetchSummary
from analyzer.exceptions import DataResult, DataSourceError, StepResult
from analyzer.pipeline import (
    TickerAnalysisParams,
    TickerScoringParams,
    run_recent_ticker_scoring,
    run_ticker_analysis,
)


def test_refresh_summary_is_scoped_to_requested_year(tmp_path):
    db = Database(tmp_path / "multi-year.duckdb")
    db.upsert_transactions(
        pd.DataFrame(
            [
                {
                    "doc_id": "2024-valid",
                    "member": "A",
                    "ticker": "AAA",
                    "transaction_date": "2024-12-30",
                    "disclosure_date": "2024-12-31",
                    "transaction_type": "Purchase",
                },
                {
                    "doc_id": "2024-invalid",
                    "member": "B",
                    "ticker": "BBB",
                    "transaction_date": "2025-01-02",
                    "disclosure_date": "2024-12-29",
                    "transaction_type": "Sale",
                },
                {
                    "doc_id": "2025-valid",
                    "member": "C",
                    "ticker": "CCC",
                    "transaction_date": "2025-12-30",
                    "disclosure_date": "2025-12-31",
                    "transaction_type": "Purchase",
                },
            ]
        ),
        source="house_pdf",
    )
    ctx = MagicMock()
    ctx.transaction_source.db = db
    ctx.transaction_source.fetch_and_cache_pdfs.return_value = HouseFetchSummary(
        archive_year=2024,
        metadata_count=10,
        ptr_count=3,
        valid_pdf_count=3,
        downloaded_count=0,
        skipped_count=3,
        orphan_pdf_count=0,
    )

    try:
        with (
            patch("analyzer.cli.get_context", return_value=ctx),
            patch(
                "analyzer.cli.run_parse_pipeline", return_value=StepResult(success=True)
            ),
        ):
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


def _consensus_test_trades() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "member": ["Alice", "Carol", "Future Bob"],
            "ticker": ["AAPL", "AAPL", "AAPL"],
            "transaction_date": pd.to_datetime(
                ["2025-05-19", "2025-05-24", "2025-06-04"]
            ),
            "disclosure_date": pd.to_datetime(
                ["2025-05-20", "2025-05-25", "2025-06-05"]
            ),
            "transaction_type": ["Purchase", "Purchase", "Purchase"],
        }
    )


def test_ticker_analysis_uses_real_consensus_at_explicit_cutoff():
    as_of = date(2025, 6, 1)
    with (
        patch(
            "analyzer.pipeline.prepare_analysis_data",
            return_value=(_consensus_test_trades(), pd.DataFrame(), pd.DataFrame()),
        ),
        patch(
            "analyzer.pipeline.analysis.rank_members",
            side_effect=AssertionError("pipeline must not rank member history"),
        ),
        patch(
            "analyzer.member_ranking.buyer_scoring.rank_members",
            side_effect=AssertionError("scorer must not rank member history"),
        ),
    ):
        result = run_ticker_analysis(
            TickerAnalysisParams(ticker="AAPL", year=2025, as_of_date=as_of),
            MagicMock(),
            MagicMock(),
        )

    assert result.success
    assert result.data["buyers"]["member"].tolist() == ["Alice", "Carol"]
    assert result.data["score"].iloc[0]["num_buyers"] == 2
    assert result.data["score"].iloc[0]["scoring_mode"] == "consensus"
    assert result.data["score"].iloc[0]["signal_score_raw"] > 0


def test_recent_ticker_scoring_uses_real_consensus_without_rankings():
    as_of = date(2025, 6, 1)
    with (
        patch(
            "analyzer.pipeline.prepare_live_analysis_data",
            return_value=(_consensus_test_trades(), pd.DataFrame(), pd.DataFrame()),
        ),
        patch(
            "analyzer.pipeline.analysis.rank_members",
            side_effect=AssertionError("pipeline must not rank member history"),
        ),
        patch(
            "analyzer.member_ranking.buyer_scoring.rank_members",
            side_effect=AssertionError("scorer must not rank member history"),
        ),
    ):
        result = run_recent_ticker_scoring(
            MagicMock(),
            MagicMock(),
            TickerScoringParams(
                year=2025,
                horizons=(90,),
                as_of_date=as_of,
                days_back=28,
                min_buyers=2,
            ),
        )

    assert result.success
    scored = result.data["result"].iloc[0]
    assert scored["ticker"] == "AAPL"
    assert scored["num_buyers"] == 2
    assert scored["scoring_mode"] == "consensus"
    assert scored["signal_score_raw"] > 0


def test_cli_as_of_reaches_single_ticker_analysis_params():
    captured = []

    def fake_run(params, transaction_source, price_source):
        captured.append(params)
        return DataResult(
            success=True,
            data={
                "ticker": params.ticker,
                "buyers": pd.DataFrame(),
                "score": pd.DataFrame({"signal_score": [1.0]}),
            },
        )

    with (
        patch("analyzer.cli.get_context", return_value=MagicMock()),
        patch("analyzer.cli._check_data_freshness"),
        patch("analyzer.cli.run_ticker_analysis", side_effect=fake_run),
    ):
        result = CliRunner().invoke(
            app,
            [
                "analyze",
                "--ticker",
                "AAPL",
                "--year",
                "2025",
                "--as-of",
                "2025-06-01",
            ],
        )

    assert result.exit_code == 0, result.output
    assert len(captured) == 1
    assert captured[0].as_of_date == date(2025, 6, 1)

def test_refresh_stops_before_parse_and_backup_when_house_fetch_is_incomplete(
    tmp_path,
):
    db = Database(tmp_path / "incomplete.duckdb")
    ctx = MagicMock()
    ctx.transaction_source.db = db
    ctx.transaction_source.fetch_and_cache_pdfs.side_effect = DataSourceError(
        "Incomplete House archive 2026: 1/2 valid PTR PDFs; missing 1: 2002 (HTTP 503)"
    )

    try:
        with (
            patch("analyzer.cli.get_context", return_value=ctx),
            patch("analyzer.cli.run_parse_pipeline") as parse_pipeline,
            patch("analyzer.capitol_trades.CapitolTradesSource") as capitol_source,
        ):
            result = CliRunner().invoke(app, ["refresh", "--year", "2026"])
    finally:
        db.close()

    assert result.exit_code == 1, result.output
    assert "missing 1: 2002 (HTTP 503)" in result.output
    parse_pipeline.assert_not_called()
    capitol_source.assert_not_called()



def test_full_history_refresh_fetches_every_archive_before_parse(tmp_path):
    db = Database(tmp_path / "full-history.duckdb")
    ctx = MagicMock()
    ctx.transaction_source.db = db

    def summary(archive_year, **_kwargs):
        return HouseFetchSummary(
            archive_year=archive_year,
            metadata_count=1,
            ptr_count=1,
            valid_pdf_count=1,
            downloaded_count=0,
            skipped_count=1,
            orphan_pdf_count=0,
        )

    ctx.transaction_source.fetch_and_cache_pdfs.side_effect = summary
    try:
        with (
            patch("analyzer.cli.get_context", return_value=ctx),
            patch(
                "analyzer.cli.run_parse_pipeline", return_value=StepResult(success=True)
            ) as parse_pipeline,
        ):
            result = CliRunner().invoke(
                app, ["refresh", "--all-years", "--skip-capitol"]
            )
    finally:
        db.close()

    assert result.exit_code == 0, result.output
    fetched_years = [call.args[0] for call in ctx.transaction_source.fetch_and_cache_pdfs.call_args_list]
    parsed_years = [call.args[1] for call in parse_pipeline.call_args_list]
    assert fetched_years == list(range(2015, date.today().year + 1))
    assert parsed_years == fetched_years
    assert all(
        call.kwargs["refresh_metadata"]
        for call in ctx.transaction_source.fetch_and_cache_pdfs.call_args_list
    )
