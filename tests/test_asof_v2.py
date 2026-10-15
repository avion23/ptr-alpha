import ast
import importlib.util
from pathlib import Path
import subprocess
from types import SimpleNamespace
from datetime import date

import pandas as pd
import pytest

from analyzer import actors, database, pipeline, positions
from analyzer.database import Database


def seed(db, *, doc="delayed", promoted="2026-08-01", accepted="2026-10-01",
         created="2026-08-02", disclosure="2026-08-03", ticker="JPM"):
    year = int(disclosure[:4])
    db.conn.execute("INSERT INTO house_archive_generations "
                    "(archive_year, generation_id, promoted_at) VALUES (?, ?, ?)",
                    [year, doc, promoted])
    db.conn.execute("INSERT INTO house_generation_metadata "
                    "(archive_year, generation_id, doc_id, filing_type) VALUES (?, ?, ?, 'P')",
                    [year, doc, doc])
    db.conn.execute("INSERT INTO house_pdf_artifacts "
                    "(archive_year, generation_id, doc_id, artifact_sha256) VALUES (?, ?, ?, 'sha')",
                    [year, doc, doc])
    row = dict(doc_id=doc, member="Jane Doe", ticker=ticker,
               transaction_date=disclosure, disclosure_date=disclosure,
               transaction_type="Purchase", chamber="house", source_record_id=doc,
               source_row_id="row1", official_filing_date=disclosure,
               ingestion_generation=doc, artifact_sha256="sha")
    db.upsert_transactions(pd.DataFrame([row]), source="house_pdf")
    db.conn.execute("UPDATE transactions SET created_at = ? WHERE doc_id = ?", [created, doc])
    db.parse_runs.upsert(doc_id=doc, year=year, parser_version="pdftotext", status="success",
                         engines_attempted="pdftotext", raw_row_count=1, transaction_count=1,
                         artifact_sha256="sha", ingestion_generation=doc)
    db.conn.execute("UPDATE pdf_parse_runs SET parsed_at = ? WHERE doc_id = ?", [accepted, doc])
    return row


@pytest.mark.parametrize("read_only", [False, True])
def test_delayed_parse_and_completion_fallback(tmp_path, read_only):
    path = tmp_path / "db.duckdb"
    db = Database(path)
    seed(db)
    db.close()
    db = Database(path, read_only=read_only)
    try:
        # Production metadata, incomplete promoted archive, immutable row and terminal run.
        assert len(db.get_transactions(2026)) == 1
        with db.canonical_as_of(date(2026, 9, 21)):
            assert db.get_transactions(2026).empty
        with db.canonical_as_of(date(2026, 10, 1)):
            assert len(db.get_transactions(2026)) == 1
    finally:
        db.close()
    if read_only:
        return
    db = Database(path)
    try:
        db.conn.execute("DELETE FROM pdf_parse_runs")
        db.conn.execute("UPDATE house_archive_generations SET parse_status='complete', completed_at='2026-10-02'")
        with db.canonical_as_of(date(2026, 10, 1)):
            assert db.get_transactions(2026).empty
        with db.canonical_as_of(date(2026, 10, 2)):
            assert len(db.get_transactions(2026)) == 1
        db.conn.execute("UPDATE house_archive_generations SET completed_at=NULL")
        with db.canonical_as_of(date(2026, 10, 3)):
            assert db.get_transactions(2026).empty  # no invented completion date
    finally:
        db.close()


def test_null_promotion_is_unknown_not_always_available(tmp_path):
    db = Database(tmp_path / "db.duckdb")
    try:
        seed(db, promoted=None)
        assert len(db.get_transactions(2026)) == 1  # unchanged live policy
        with db.canonical_as_of(date(2026, 10, 15)):
            assert db.get_transactions(2026).empty
    finally:
        db.close()


def test_replacement_preserves_accepted_rowset_and_terminal_history(tmp_path):
    db = Database(tmp_path / "db.duckdb")
    try:
        row = seed(db, accepted="2026-08-04")
        row["ticker"] = "MSFT"
        db.replace_transactions_for_docs(
            pd.DataFrame([row]), source="house_pdf", attempted_doc_ids=["delayed"],
            replacement_doc_ids=["delayed"], ingestion_generation="delayed",
            parse_runs=[dict(doc_id="delayed", year=2026, parser_version="pdftotext",
                             status="success", engines_attempted="pdftotext", raw_row_count=1,
                             transaction_count=1, artifact_sha256="sha", ingestion_generation="delayed")],
        )
        assert db.get_transactions(2026).ticker.tolist() == ["MSFT"]
        with db.canonical_as_of(date(2026, 9, 21)):
            assert db.get_transactions(2026).ticker.tolist() == ["JPM"]
        assert db.conn.execute("SELECT COUNT(*) FROM parse_run_versions").fetchone()[0] == 1
        assert db.conn.execute("SELECT COUNT(*) FROM transaction_versions").fetchone()[0] == 1
    finally:
        db.close()


def test_same_count_old_run_cannot_accept_new_rows(tmp_path):
    db = Database(tmp_path / "db.duckdb")
    try:
        seed(db, accepted="2026-08-04", created="2026-10-01")
        with db.canonical_as_of(date(2026, 10, 2)):
            assert db.get_transactions(2026).empty
    finally:
        db.close()


def test_one_parser_cannot_accept_another_parsers_delayed_rowset(tmp_path):
    db = Database(tmp_path / "db.duckdb")
    try:
        row = seed(db, accepted="2026-08-04")
        row.update(ticker="MSFT", source_row_id="ocr-row")
        db.upsert_transactions(pd.DataFrame([row]), source="gemini_ocr")
        db.conn.execute("UPDATE transactions SET created_at='2026-08-02' WHERE source='gemini_ocr'")
        db.parse_runs.upsert(doc_id="delayed", year=2026, parser_version="gemini-v1", status="success",
                             engines_attempted="gemini/v1", raw_row_count=1, transaction_count=1,
                             artifact_sha256="sha", ingestion_generation="delayed")
        db.conn.execute("UPDATE pdf_parse_runs SET parsed_at='2026-10-01' WHERE parser_version='gemini-v1'")
        with db.canonical_as_of(date(2026, 9, 21)):
            assert db.get_transactions(2026).ticker.tolist() == ["JPM"]
        with db.canonical_as_of(date(2026, 10, 1)):
            assert set(db.get_transactions(2026).ticker) == {"JPM", "MSFT"}
    finally:
        db.close()


def test_live_bytes_equal_independent_main_implementation(tmp_path):
    db = Database(tmp_path / "db.duckdb")
    try:
        seed(db)
        seed(db, doc="unknown", promoted=None, ticker="MSFT")
        actual = db.conn.execute("SELECT * FROM canonical_transactions ORDER BY id").fetchdf().to_csv(index=False).encode()
        # Compile the ORIGINAL main implementation, not a second call to patched SQL.
        source = subprocess.check_output(["git", "show", "main:src/analyzer/database.py"], text=True)
        tree = ast.parse(source)
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Database")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_init_canonical_transactions_view")
        constant = next(n for n in tree.body if isinstance(n, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == "_HOUSE_PARSE_SOURCE" for t in n.targets))
        namespace = vars(database).copy()
        exec(compile(ast.fix_missing_locations(ast.Module(body=[constant, method], type_ignores=[])), "main-oracle", "exec"), namespace)
        namespace[method.name](db)
        expected = db.conn.execute("SELECT * FROM canonical_transactions ORDER BY id").fetchdf().to_csv(index=False).encode()
        assert actual == expected
    finally:
        db.close()


@pytest.mark.parametrize("consumer", ["analysis", "consensus", "freshness", "holdings", "skill", "targeted_skill"])
def test_omitted_dated_consumer_reads_historical_rows(tmp_path, monkeypatch, consumer):
    db = Database(tmp_path / "db.duckdb")
    seed(db)
    cutoff = date(2026, 9, 21)
    source = SimpleNamespace(db=db)
    try:
        if consumer == "analysis":
            with pytest.raises(pipeline.DataSourceError, match="No trading data"):
                pipeline.prepare_analysis_data(source, None, 2026, (30,), cutoff)
        elif consumer == "consensus":
            assert pipeline.prepare_live_consensus_data(source, cutoff, days_back=None).empty
        elif consumer == "freshness":
            result = pipeline.source_freshness_metrics(source, cutoff)
            assert result.recent_disclosures.sum() == 0
            assert result.latest_disclosure_date.isna().all()
        elif consumer == "holdings":
            original = positions.load_member_trades

            def checked(*args, **kwargs):
                assert db._canonical_as_of == cutoff
                result = original(*args, **kwargs)
                assert result.empty
                return result
            monkeypatch.setattr(positions, "load_member_trades", checked)
            assert positions.holdings_candidates(db, "Jane Doe", cutoff).empty
        else:
            original = db.get_transactions_by_date_range

            def checked(*args, **kwargs):
                assert db._canonical_as_of == cutoff
                result = original(*args, **kwargs)
                assert result.empty
                return result
            monkeypatch.setattr(db, "get_transactions_by_date_range", checked)
            ids = {"congress:Jane Doe"} if consumer == "targeted_skill" else None
            assert actors.load_skill_table(db, cutoff, ids).empty
        assert len(db.get_transactions(2026)) == 1
    finally:
        db.close()


def load_script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_skill_cutoff_is_disclosure_cutoff_not_acceptance_cutoff(tmp_path, monkeypatch):
    script = load_script("backtest_assumptions")
    db = Database(tmp_path / "db.duckdb")
    try:
        seed(db, promoted="2022-12-01", created="2022-12-02", accepted="2023-06-01",
             disclosure="2022-12-03")
        seed(db, doc="new-disclosure", promoted="2023-01-01", created="2023-01-02",
             accepted="2023-01-03", disclosure="2023-01-03", ticker="MSFT")
        monkeypatch.setattr(script, "next_session_close", lambda *args: (date(2022, 12, 5), 100))
        monkeypatch.setattr(script, "fwd_excess", lambda *args: (0, 0, 5.0))
        assert script.skill_episodes(db, date(2023, 1, 15)) == []
        later = script.skill_episodes(db, date(2023, 7, 15))
        assert [r["ticker"] for r in later] == ["JPM"]
    finally:
        db.close()


@pytest.mark.parametrize("consumer", ["backtest", "portfolio", "kelly"])
def test_historical_simulators_load_each_decision_not_end_date(tmp_path, monkeypatch, consumer):
    from analyzer import cli
    import typer

    db = Database(tmp_path / "db.duckdb")
    seed(db, promoted="2023-04-21", created="2023-04-22", accepted="2023-06-01",
         disclosure="2023-04-20")
    dates = pd.bdate_range("2022-01-01", "2026-01-01")
    prices = pd.DataFrame({"JPM": 100.0, "SPY": 100.0}, index=dates)
    monkeypatch.setattr(db, "get_prices", lambda *args, **kwargs: prices)
    source = SimpleNamespace(db=db)
    calls = []

    def recommendations(frame, decision, **kwargs):
        decision = pd.Timestamp(decision)
        assert frame.ticker.tolist() == (["JPM"] if decision >= pd.Timestamp("2023-06-01") else [])
        calls.append(decision)
        return pd.DataFrame()

    monkeypatch.setattr(pipeline.analysis, "backtest_recommendations", recommendations)
    try:
        if consumer == "backtest":
            monkeypatch.setattr(pipeline, "create_snapshot", lambda *args, **kwargs: None)
            monkeypatch.setattr(pipeline, "_benchmark_return", lambda *args: None)
            params = pipeline.BacktestParams(date(2023, 5, 1), date(2023, 7, 1), frequency_days=61)
            pipeline.run_backtest_pipeline(params, source, SimpleNamespace(get_prices=lambda *args: prices))
        elif consumer == "portfolio":
            with pytest.raises(typer.Exit):
                cli._load_portfolio_inputs(SimpleNamespace(transaction_source=source), None,
                                           date(2023, 7, 1), 28, 3, 5, 61, date(2023, 5, 1))
        else:
            script = load_script("run_kelly_backtest")

            def portfolios(frame, decisions, **kwargs):
                assert len(decisions) == 1
                decision = decisions[0]
                assert db._canonical_as_of == pd.Timestamp(decision).date()
                return recommendations(frame, decision)
            monkeypatch.setattr(script, "build_portfolios_from_backtest", portfolios)
            assert script._run(source, pd.DataFrame()) == 2
        assert min(calls) < pd.Timestamp("2023-06-01") < max(calls)
    finally:
        db.close()
