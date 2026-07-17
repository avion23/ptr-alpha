import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest
from typer.testing import CliRunner

from analyzer import datasources
from analyzer import parser_cascade
from analyzer import download
from analyzer.cli import app
from analyzer.database import Database
from analyzer.exceptions import StepResult
from analyzer.exceptions import ParsingError
from analyzer.models import FilingType
from analyzer.exceptions import DataSourceError
from analyzer.models import DownloadResult, DownloadStatus


def _tx(transaction_date="2024-01-02", amount_midpoint=1000.0):
    return {
        "ticker": "AAPL",
        "transaction_type": "P",
        "transaction_date": transaction_date,
        "owner_code": "SP",
        "amount_raw": "$1,001 - $15,000",
        "amount_midpoint": amount_midpoint,
        "instrument_type": None,
        "strike_price": None,
        "expiry_date": None,
        "asset_description": "Apple Inc.",
    }


def test_is_valid_pdf_requires_header_and_content(tmp_path):
    valid = tmp_path / "valid.pdf"
    valid.write_bytes(b"%PDF-1.7\nbody")
    empty = tmp_path / "empty.pdf"
    empty.write_bytes(b"")
    html = tmp_path / "html.pdf"
    html.write_bytes(b"<html>error</html>")

    assert datasources._is_valid_pdf(valid)
    assert not datasources._is_valid_pdf(empty)
    assert not datasources._is_valid_pdf(html)


def test_result_quality_counts_rows_with_date_and_amount():
    assert datasources._result_quality([]) == 0.0
    assert datasources._result_quality([_tx(), _tx(amount_midpoint=None)]) == 0.5


def test_parse_worker_prefers_high_quality_later_text_engine(monkeypatch, tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.7\nbody")
    low_quality = [_tx(transaction_date=None, amount_midpoint=None)]
    high_quality = [_tx() for _ in range(10)]

    monkeypatch.setattr(parser_cascade, "_try_pdfplumber", lambda path: low_quality)
    monkeypatch.setattr(parser_cascade, "_try_camelot_lattice", lambda path: [])
    monkeypatch.setattr(parser_cascade, "_try_camelot_stream", lambda path: [])
    monkeypatch.setattr(parser_cascade, "_try_pdftotext", lambda path: high_quality)
    monkeypatch.setattr(parser_cascade, "_try_docling", lambda path: (_ for _ in ()).throw(AssertionError("docling should not run")))
    monkeypatch.setattr(parser_cascade, "_try_tesseract", lambda path: (_ for _ in ()).throw(AssertionError("ocr should not run")))

    _, transactions, engines_attempted = parser_cascade._parse_pdf_worker(pdf_path)

    assert transactions == high_quality
    assert engines_attempted == [
        "pdfplumber",
        "lattice",
        "stream",
        "pdftotext",
        "won:pdftotext",
    ]


def test_reparse_all_filters_ptr_filings_unconditionally(tmp_path):
    from scripts import reparse_all

    data_dir = tmp_path / "data"
    (data_dir / "2024" / "pdfs").mkdir(parents=True)
    settings = MagicMock()
    settings.data.data_dir = str(data_dir)

    metadata = pd.DataFrame(
        {
            "DocID": ["p1", "a1"],
            "FilingType": [FilingType.PTR.value, FilingType.AMENDMENT.value],
            "First": ["A", "B"],
            "Last": ["One", "Two"],
            "FilingDate": ["2024-01-01", "2024-01-02"],
        }
    )
    source = MagicMock()
    source.fetch_metadata.return_value = metadata

    captured = {}

    def fake_filter(ptrs, pdf_dir):
        captured["ptrs"] = ptrs.copy()
        return [], pd.DataFrame()

    with patch.object(reparse_all, "HouseTransactionSource", return_value=source), \
         patch.object(reparse_all, "_filter_existing_pdfs", side_effect=fake_filter):
        assert reparse_all.parse_year(2024, MagicMock(), settings) == 0

    assert captured["ptrs"]["DocID"].tolist() == ["p1"]


def test_save_parse_results_warns_when_zero_row_doc_retains_db_rows(caplog, tmp_path):
    source = download.HouseTransactionSource.__new__(download.HouseTransactionSource)
    source.db = MagicMock()
    source.db.count_transactions_for_docs.return_value = {"123": 3}
    pdf_path = tmp_path / "123.pdf"

    with patch.object(download, "consolidate_transactions", return_value=pd.DataFrame()), \
         caplog.at_level("WARNING", logger="analyzer.download"):
        source._save_parse_results(2024, [(pdf_path, [], ["pdfplumber"])], {})

    source.db.count_transactions_for_docs.assert_called_once_with(["123"])
    source.db.parse_runs.upsert.assert_called_once_with(
        doc_id="123",
        year=2024,
        parser_version="v3",
        status="zero_rows",
        engines_attempted="pdfplumber",
        raw_row_count=0,
        transaction_count=0,
        _in_transaction=True,
    )
    assert "1 docs parsed to zero rows but retain 3 existing DB rows (stale?): 123" in caplog.text


def test_parse_cached_pdfs_skips_cached_and_force_reparses(tmp_path):
    year = 2026
    pdf_dir = tmp_path / str(year) / "pdfs"
    pdf_dir.mkdir(parents=True)
    for doc_id in ("1", "2"):
        (pdf_dir / f"{doc_id}.pdf").write_bytes(b"%PDF-1.7\nbody")

    source = download.HouseTransactionSource.__new__(download.HouseTransactionSource)
    source.data_dir = tmp_path
    source.parallel_workers = 1
    source.db = Database(tmp_path / "test.duckdb")
    source.fetch_metadata = MagicMock(return_value=pd.DataFrame({
        "DocID": ["1", "2"],
        "FilingType": [FilingType.PTR.value, FilingType.PTR.value],
        "First": ["A", "B"],
        "Last": ["One", "Two"],
        "FilingDate": ["2026-01-01", "2026-01-02"],
    }))
    parsed_batches = []

    class FakePool:
        def __init__(self, workers):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

        def map(self, worker, paths):
            parsed_batches.append([path.stem for path in paths])
            return [(path, [_tx()], ["pdfplumber"]) for path in paths]

    try:
        with patch.object(download, "Pool", FakePool):
            source.parse_cached_pdfs(year)
            source.parse_cached_pdfs(year)
            source.parse_cached_pdfs(year, force=True)
    finally:
        source.db.close()

    assert parsed_batches == [["1", "2"], ["1", "2"]]


def test_save_parse_results_does_not_mark_success_before_replacement(tmp_path):
    source = download.HouseTransactionSource.__new__(download.HouseTransactionSource)
    source.db = MagicMock()
    source.db.count_transactions_for_docs.return_value = {}
    source.db.replace_transactions_for_docs.side_effect = OSError("disk full")
    pdf_path = tmp_path / "123.pdf"
    consolidated = pd.DataFrame({"doc_id": ["123"]})

    with patch.object(download, "consolidate_transactions", return_value=consolidated), \
         patch.object(download, "preserve_existing_fields", return_value=consolidated), \
         pytest.raises(OSError, match="disk full"):
        source._save_parse_results(2024, [(pdf_path, [_tx()], ["pdfplumber"])], {})

    source.db.upsert_parse_run.assert_not_called()
    source.db.replace_transactions_for_docs.assert_called_once_with(
        consolidated,
        source="house_pdf",
        parse_runs=[{
            "doc_id": "123",
            "year": 2024,
            "parser_version": "v3",
            "status": "success",
            "engines_attempted": "pdfplumber",
            "raw_row_count": 0,
            "transaction_count": 1,
        }],
    )


def test_save_parse_results_audits_consolidated_rows(tmp_path):
    source = download.HouseTransactionSource.__new__(download.HouseTransactionSource)
    source.db = MagicMock()
    source.db.count_transactions_for_docs.return_value = {"dropped": 2}
    kept_pdf = tmp_path / "kept.pdf"
    dropped_pdf = tmp_path / "dropped.pdf"
    consolidated = pd.DataFrame({"doc_id": ["kept"]})

    with patch.object(download, "consolidate_transactions", return_value=consolidated), \
         patch.object(download, "preserve_existing_fields", return_value=consolidated):
        source._save_parse_results(
            2024,
            [
                (kept_pdf, [_tx(), _tx()], ["pdfplumber"]),
                (dropped_pdf, [_tx()], ["pdftotext"]),
            ],
            {},
        )

    source.db.count_transactions_for_docs.assert_called_once_with(["dropped"])
    parse_runs = source.db.replace_transactions_for_docs.call_args.kwargs["parse_runs"]
    assert parse_runs == [
        {
            "doc_id": "kept", "year": 2024, "parser_version": "v3",
            "status": "success", "engines_attempted": "pdfplumber",
            "raw_row_count": 0, "transaction_count": 1,
        },
        {
            "doc_id": "dropped", "year": 2024, "parser_version": "v3",
            "status": "zero_rows", "engines_attempted": "pdftotext",
            "raw_row_count": 0, "transaction_count": 0,
        },
    ]


def test_failed_metadata_refresh_does_not_clear_cached_rows():
    source = download.HouseTransactionSource.__new__(download.HouseTransactionSource)
    source.db = MagicMock()
    source.metadata_url_template = "https://example.test/{year}.zip"
    source._download_and_upsert_metadata = MagicMock(side_effect=OSError("bad zip"))

    with pytest.raises(DataSourceError):
        source.fetch_metadata(2024, refresh=True)

    source.db.clear_metadata.assert_not_called()
    source._download_and_upsert_metadata.assert_called_once_with(
        2024, "https://example.test/2024.zip", replace=True,
    )


def test_metadata_zip_accepts_uppercase_txt_and_windows_1252(tmp_path):
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        archive.writestr("2024FD.TXT", "DocID\tFirst\tLast\tFilingDate\n1\tAda\tLovelace\t2024-01-01")
    assert download._read_first_text_from_zip(buf.getvalue(), 2024).startswith("DocID")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        archive.writestr(
            "2024FD.txt",
            b"DocID\tFirst\tLast\tFilingDate\n1\tRen\xe9\tDoe\t2024-01-01",
        )
    assert "René" in download._read_first_text_from_zip(buf.getvalue(), 2024)


def test_metadata_zip_selects_table_not_readme_and_rejects_ambiguity():
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        archive.writestr("README.txt", "instructions")
        archive.writestr("2024FD.txt", "DocID\tFirst\tLast\tFilingDate\n1\tA\tOne\t2024-01-01")
    assert download._read_first_text_from_zip(buf.getvalue(), 2024).startswith("DocID")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        header = "DocID\tFirst\tLast\tFilingDate\n"
        archive.writestr("one.txt", header + "1\tA\tOne\t2024-01-01")
        archive.writestr("two.txt", header + "2\tB\tTwo\t2024-01-02")
    with pytest.raises(ParsingError, match="Ambiguous metadata ZIP"):
        download._read_first_text_from_zip(buf.getvalue(), 2024)


def test_filter_existing_pdfs_excludes_invalid_files(tmp_path):
    valid = tmp_path / "1.pdf"
    invalid = tmp_path / "2.pdf"
    valid.write_bytes(b"%PDF-1.7\nbody")
    invalid.write_bytes(b"<html>error</html>")
    ptrs = pd.DataFrame({"DocID": ["1", "2"]})

    paths, docs = download._filter_existing_pdfs(ptrs, tmp_path)

    assert paths == [valid]
    assert docs["DocID"].tolist() == ["1"]


def test_member_lookup_rejects_ambiguous_duplicate_doc_id():
    docs = pd.DataFrame({
        "DocID": ["1", "1"], "First": ["A", "B"], "Last": ["One", "Two"],
        "FilingDate": ["2024-01-01", "2024-01-01"],
    })
    with pytest.raises(ParsingError, match="member attribution ambiguous"):
        download._build_member_lookup(docs)


def test_member_lookup_allows_identical_duplicate_doc_id():
    docs = pd.DataFrame({
        "DocID": ["1", "1"], "First": ["A", "A"], "Last": ["One", "One"],
        "FilingDate": ["2024-01-01", "2024-01-01"],
    })
    assert download._build_member_lookup(docs)["1"]["First"] == "A"


def test_pdf_batch_failure_is_reported_to_caller(tmp_path):
    source = download.HouseTransactionSource.__new__(download.HouseTransactionSource)
    source.fetch_metadata = MagicMock(return_value=pd.DataFrame({
        "DocID": ["1"], "FilingType": [FilingType.PTR.value],
    }))
    source.data_dir = tmp_path
    source.pdf_url_template = "https://example.test/{year}/{doc_id}.pdf"
    source._download_pdf_async = AsyncMock(return_value=DownloadResult(
        doc_id="1", status=DownloadStatus.FAILED, error_message="HTTP 500",
    ))

    with pytest.raises(DataSourceError, match="Failed to download 1 of 1"):
        asyncio.run(source._fetch_and_cache_pdfs_async(2024))


def _refresh_context():
    ctx = MagicMock()
    execute = ctx.transaction_source.db.conn.execute
    execute.side_effect = [
        MagicMock(fetchone=MagicMock(return_value=(10,))),
        MagicMock(fetchone=MagicMock(return_value=(10,))),
        MagicMock(fetchone=MagicMock(return_value=("2024-01-02",))),
    ]
    return ctx


def test_refresh_exits_one_when_fetch_pipeline_returns_false():
    runner = CliRunner()
    with patch("analyzer.cli.get_context", return_value=_refresh_context()), \
         patch("analyzer.cli.run_fetch_pipeline", return_value=StepResult(success=False)), \
         patch("analyzer.cli.run_parse_pipeline", return_value=StepResult(success=True)):
        result = runner.invoke(app, ["refresh", "--skip-capitol"])

    assert result.exit_code == 1, result.output
    assert "FAILED steps: fetch" in result.output


def test_refresh_exits_zero_when_all_steps_succeed():
    runner = CliRunner()
    with patch("analyzer.cli.get_context", return_value=_refresh_context()), \
         patch("analyzer.cli.run_fetch_pipeline", return_value=StepResult(success=True)), \
         patch("analyzer.cli.run_parse_pipeline", return_value=StepResult(success=True)):
        result = runner.invoke(app, ["refresh", "--skip-capitol"])

    assert result.exit_code == 0, result.output
    assert "FAILED steps:" not in result.output
