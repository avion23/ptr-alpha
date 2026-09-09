from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
from typer.testing import CliRunner

from analyzer.cli import app
from analyzer.database import Database
from analyzer.download import HouseFetchSummary
from analyzer.exceptions import DataResult, DataSourceError, StepResult
from analyzer.pipeline import (
    BacktestParams,
    TickerAnalysisParams,
    TickerScoringParams,
    run_backtest_pipeline,
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
    db.conn.execute(
        """
        INSERT INTO house_archive_generations (
            archive_year, generation_id, metadata_sha256,
            metadata_count, ptr_count, parse_status
        ) VALUES (2024, 'test-2024', 'sha', 0, 0, 'incomplete')
        """
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
            "instrument_type": ["stock", "stock", "stock"],
            "raw_asset_class": ["ST", "ST", "ST"],
            "ticker_origin": ["official", "official", "official"],
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
        transaction_source = MagicMock()
        transaction_source.db.get_transactions.return_value = _consensus_test_trades()
        result = run_ticker_analysis(
            TickerAnalysisParams(
                ticker="AAPL", year=2025, min_buyers=2, as_of_date=as_of
            ),
            transaction_source,
        )

    assert result.success
    assert result.data["buyers"]["member"].tolist() == ["Alice", "Carol"]
    assert result.data["score"].iloc[0]["num_buyers"] == 2
    assert result.data["score"].iloc[0]["scoring_mode"] == "consensus"
    assert result.data["score"].iloc[0]["signal_score_raw"] > 0


def test_single_ticker_rejects_prelisting_reused_symbol_rows():
    rows = pd.DataFrame(
        {
            "member": ["Early Buyer"],
            "ticker": ["SPCX"],
            "transaction_date": pd.to_datetime(["2026-06-11"]),
            "disclosure_date": pd.to_datetime(["2026-06-11"]),
            "transaction_type": ["Purchase"],
            "instrument_type": ["stock"],
            "source": ["house_pdf"],
        }
    )
    transaction_source = MagicMock()
    transaction_source.db.get_transactions.return_value = rows

    result = run_ticker_analysis(
        TickerAnalysisParams(
            ticker="SPCX",
            year=2026,
            days_back=28,
            min_buyers=1,
            as_of_date=date(2026, 6, 11),
        ),
        transaction_source,
    )

    assert result.success
    assert result.data["buyers"].empty
    assert result.data["score"].iloc[0]["signal_score"] == 0.0


def test_recent_ticker_scoring_uses_real_consensus_without_rankings():
    as_of = date(2025, 6, 1)
    transaction_source = MagicMock()
    transaction_source.db.get_transactions_by_date_range.return_value = _consensus_test_trades()
    with (
        patch(
            "analyzer.pipeline.prepare_live_consensus_data",
            return_value=_consensus_test_trades(),
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
            transaction_source,
            MagicMock(),
            TickerScoringParams(
                year=2025,
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


def test_recent_ticker_scoring_consensus_is_transaction_only():
    as_of = date(2025, 6, 1)
    transaction_source = MagicMock()
    transaction_source.db.get_transactions_by_date_range.return_value = (
        _consensus_test_trades()
    )
    price_source = MagicMock()
    with (
        patch(
            "analyzer.pipeline.analysis.calculate_signal_potential",
            side_effect=AssertionError("live consensus must not build labels"),
        ),
        patch(
            "analyzer.pipeline.analysis.rank_members",
            side_effect=AssertionError("live consensus must not rank history"),
        ),
        patch(
            "analyzer.member_ranking.buyer_scoring.rank_members",
            side_effect=AssertionError("live consensus must not rank history"),
        ),
    ):
        result = run_recent_ticker_scoring(
            transaction_source,
            price_source,
            TickerScoringParams(
                year=2025,
                horizons=(90,),
                as_of_date=as_of,
                days_back=28,
                min_buyers=2,
                top_n=1,
                training_lookback_days=365,
            ),
        )

    assert result.success
    assert result.data["result"]["ticker"].tolist() == ["AAPL"]
    assert result.data["top_n"] == 1
    assert result.data["days_back"] == 28
    assert result.data["min_buyers"] == 2
    assert result.data["as_of_date"] == as_of
    transaction_source.db.get_transactions_by_date_range.assert_called_once_with(
        pd.Timestamp("2024-03-03"),
        pd.Timestamp(as_of),
    )
    price_source.get_prices.assert_not_called()
    transaction_source.db.get_entry_prices.assert_not_called()


def test_recent_ticker_scoring_filters_rejected_symbols_before_candidate_gate():
    as_of = date(2025, 6, 1)
    rejected = pd.DataFrame(
        {
            "member": ["Invalid Buyer One", "Invalid Buyer Two"],
            "ticker": ["SP", "SP"],
            "transaction_date": pd.to_datetime(["2025-05-26", "2025-05-27"]),
            "disclosure_date": pd.to_datetime(["2025-05-28", "2025-05-29"]),
            "transaction_type": ["Purchase", "Purchase"],
        }
    )
    trades = pd.concat([_consensus_test_trades(), rejected], ignore_index=True)

    with patch(
        "analyzer.pipeline.prepare_live_consensus_data",
        return_value=trades,
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
    assert result.data["result"]["ticker"].tolist() == ["AAPL"]


def test_cli_as_of_reaches_single_ticker_analysis_params():
    captured = []

    def fake_run(params, transaction_source):
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
        db.conn.execute(
            """
            INSERT INTO house_archive_generations (
                archive_year, generation_id, metadata_sha256,
                metadata_count, ptr_count, parse_status
            ) VALUES (?, ?, 'sha', 0, 0, 'incomplete')
            """,
            [archive_year, f"test-{archive_year}"],
        )
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
    fetched_years = [
        call.args[0]
        for call in ctx.transaction_source.fetch_and_cache_pdfs.call_args_list
    ]
    parsed_years = [call.args[1] for call in parse_pipeline.call_args_list]
    assert fetched_years == list(range(2015, date.today().year + 1))
    assert parsed_years == fetched_years
    assert all(
        call.kwargs["refresh_metadata"]
        for call in ctx.transaction_source.fetch_and_cache_pdfs.call_args_list
    )


def test_backtest_pipeline_emits_real_spy_buy_hold_row(tmp_path):
    transactions = pd.DataFrame(
        {
            "member": ["Alice"],
            "ticker": ["AAPL"],
            "transaction_date": pd.to_datetime(["2024-12-01"]),
            "disclosure_date": pd.to_datetime(["2024-12-02"]),
            "transaction_type": ["Purchase"],
        }
    )
    index = pd.date_range("2024-11-01", "2025-01-20", freq="D")
    prices = pd.DataFrame(
        {
            "AAPL": range(100, 100 + len(index)),
            "SPY": range(400, 400 + len(index)),
        },
        index=index,
    )
    evaluated = pd.DataFrame(
        {
            "rank": [1],
            "ticker": ["AAPL"],
            "bt_return_pct": [10.0],
            "bt_alpha_pct": [5.0],
            "bt_raw_return_pct": [10.0],
            "bt_entry_date": [date(2025, 1, 3)],
            "bt_exit_date": [date(2025, 1, 10)],
            "bt_leverage": [1.0],
        }
    )
    transaction_source = MagicMock()
    transaction_source.db.get_transactions_by_date_range.return_value = transactions
    price_source = MagicMock()
    price_source.get_prices.return_value = prices

    with (
        patch("analyzer.pipeline.create_snapshot", return_value=MagicMock()),
        patch(
            "analyzer.pipeline.analysis.backtest_recommendations",
            return_value=pd.DataFrame({"ticker": ["AAPL"]}),
        ),
        patch(
            "analyzer.pipeline.analysis.evaluate_backtest",
            return_value=evaluated,
        ),
    ):
        result = run_backtest_pipeline(
            BacktestParams(
                start_date=date(2025, 1, 2),
                end_date=date(2025, 1, 2),
                horizon=7,
                frequency_days=1,
            ),
            transaction_source,
            price_source,
        )

    assert result.success
    summary = result.data["summary"]
    assert "SPY_BUY_HOLD" in summary["rank"].tolist()
    assert summary.attrs["spy_benchmark_status"] == "available"
    assert summary.attrs["spy_benchmark_reason"] is None


def test_backtest_pipeline_keeps_supported_no_recommendation_dates_as_cash(tmp_path):
    transactions = pd.DataFrame(
        {
            "member": ["Alice"],
            "ticker": ["AAPL"],
            "transaction_date": pd.to_datetime(["2024-12-01"]),
            "disclosure_date": pd.to_datetime(["2024-12-02"]),
            "transaction_type": ["Purchase"],
        }
    )
    index = pd.date_range("2024-11-01", "2025-01-20", freq="D")
    prices = pd.DataFrame(
        {
            "AAPL": range(100, 100 + len(index)),
            "SPY": range(400, 400 + len(index)),
        },
        index=index,
    )
    evaluated = pd.DataFrame(
        {
            "rank": [1],
            "ticker": ["AAPL"],
            "bt_return_pct": [10.0],
            "bt_alpha_pct": [5.0],
            "bt_raw_return_pct": [10.0],
            "bt_entry_date": [date(2025, 1, 3)],
            "bt_exit_date": [date(2025, 1, 10)],
            "bt_leverage": [1.0],
        }
    )
    transaction_source = MagicMock()
    transaction_source.db.get_transactions_by_date_range.return_value = transactions
    price_source = MagicMock()
    price_source.get_prices.return_value = prices

    with (
        patch("analyzer.pipeline.create_snapshot", return_value=MagicMock()),
        patch("analyzer.pipeline.save_snapshot"),
        patch(
            "analyzer.pipeline._entry_prices_from_matrix",
            return_value=pd.DataFrame({"entry_price": [100.0]}),
        ),
        patch(
            "analyzer.pipeline.analysis.calculate_signal_potential",
            return_value=pd.DataFrame({"member": ["Alice"]}),
        ),
        patch(
            "analyzer.pipeline.analysis.backtest_recommendations",
            side_effect=[pd.DataFrame({"ticker": ["AAPL"]}), pd.DataFrame()],
        ),
        patch("analyzer.pipeline.analysis.evaluate_backtest", return_value=evaluated),
        patch("analyzer.pipeline._benchmark_return", side_effect=[4.0, 5.0]),
    ):
        result = run_backtest_pipeline(
            BacktestParams(
                start_date=date(2025, 1, 2),
                end_date=date(2025, 1, 3),
                horizon=7,
                frequency_days=1,
            ),
            transaction_source,
            price_source,
            data_dir=tmp_path,
        )

    assert result.success
    assert result.data["evaluable_dates"] == 2
    assert result.data["total_as_of_dates"] == 2

    combined = result.data["combined"]
    assert set(combined["as_of_date"]) == {date(2025, 1, 2), date(2025, 1, 3)}
    cash = combined[combined["status"] == "cash"].iloc[0]
    assert cash["recommendation_count"] == 0
    assert cash["bt_return_pct"] == 0.0
    assert cash["spy_return_pct"] == 5.0
    assert cash["net_alpha_pct"] == -5.0

    observations = result.data["date_observations"]
    assert len(observations) == 2
    no_trade = observations[observations["status"] == "cash"].iloc[0]
    assert no_trade["recommendation_count"] == 0
    assert no_trade["strategy_return_pct"] == 0.0
    assert no_trade["net_alpha_pct"] == -5.0
    portfolio = result.data["summary"]
    assert portfolio.loc[portfolio["rank"] == "PORTFOLIO", "count"].iloc[0] == 2
