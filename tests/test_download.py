import hashlib
import zipfile
from contextlib import nullcontext
from datetime import date
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from analyzer.database import Database
from analyzer.download import HouseTransactionSource
from analyzer.exceptions import DataSourceError
from analyzer.models import DownloadResult, DownloadStatus
from analyzer.settings import Settings


def _source(tmp_path):
    settings = Settings()
    settings.data.data_dir = str(tmp_path)
    db = Database(tmp_path / "test.duckdb")
    return HouseTransactionSource(settings, db=db), db


def _metadata(*doc_ids):
    return pd.DataFrame(
        {
            "DocID": list(doc_ids),
            "ArchiveYear": [2021] * len(doc_ids),
            "First": ["First"] * len(doc_ids),
            "Last": ["Last"] * len(doc_ids),
            "FilingDate": pd.to_datetime(["2022-01-04"] * len(doc_ids)),
            "FilingType": ["P"] * len(doc_ids),
        }
    )


def test_cross_year_ptr_uses_archive_year_for_url_and_file_path(tmp_path):
    source, db = _source(tmp_path)
    source.fetch_metadata = MagicMock(return_value=_metadata("8218519"))
    captured = {}

    async def download(_session, doc_id, pdf_path, url):
        captured.update(doc_id=doc_id, pdf_path=pdf_path, url=url)
        pdf_path.write_bytes(b"%PDF-test")
        return DownloadResult(doc_id=doc_id, status=DownloadStatus.SUCCESS)

    source._download_pdf_async = download
    try:
        summary = source.fetch_and_cache_pdfs(2021, refresh_metadata=False)
    finally:
        source.close()
        db.close()

    assert captured["doc_id"] == "8218519"
    assert captured["pdf_path"] == tmp_path / "2021" / "pdfs" / "8218519.pdf"
    assert captured["url"].endswith("/ptr-pdfs/2021/8218519.pdf")
    assert summary.ptr_count == summary.valid_pdf_count == 1


def test_incomplete_pdf_batch_reports_every_missing_doc_id(tmp_path):
    source, db = _source(tmp_path)
    source.fetch_metadata = MagicMock(
        return_value=_metadata("present", "missing-a", "missing-b")
    )

    async def download(_session, doc_id, pdf_path, _url):
        if doc_id == "present":
            pdf_path.write_bytes(b"%PDF-test")
            return DownloadResult(doc_id=doc_id, status=DownloadStatus.SUCCESS)
        return DownloadResult(
            doc_id=doc_id,
            status=DownloadStatus.FAILED,
            status_code=503,
            error_message="HTTP 503",
        )

    source._download_pdf_async = download
    try:
        with pytest.raises(DataSourceError) as exc_info:
            source.fetch_and_cache_pdfs(2021, refresh_metadata=False)
    finally:
        source.close()
        db.close()

    message = str(exc_info.value)
    assert "1/3 valid PTR PDFs" in message
    assert "missing 2: missing-a (HTTP 503), missing-b (HTTP 503)" in message


def test_current_archive_fetch_forces_fresh_metadata(tmp_path):
    source, db = _source(tmp_path)
    source.fetch_metadata = MagicMock(return_value=_metadata())
    current_year = date.today().year

    try:
        source.fetch_and_cache_pdfs(current_year)
    finally:
        source.close()
        db.close()

    source.fetch_metadata.assert_called_once_with(current_year, refresh=True)



def test_forced_metadata_refresh_bypasses_http_cache_and_persists_archive(tmp_path):
    source, db = _source(tmp_path)
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(
            "2021FD.txt",
            "First\tLast\tFilingDate\tFilingType\tDocID\n"
            "Michael T.\tMcCaul\t01/04/2022\tP\t8218519\n",
        )
    source.session.cache_disabled = MagicMock(return_value=nullcontext())
    source.session.get = MagicMock(
        return_value=SimpleNamespace(status_code=200, content=buffer.getvalue())
    )

    try:
        metadata = source.fetch_metadata(2021, refresh=True)
    finally:
        source.close()
        db.close()

    source.session.cache_disabled.assert_called_once_with()
    assert metadata[["DocID", "ArchiveYear"]].to_dict("records") == [
        {"DocID": "8218519", "ArchiveYear": 2021}
    ]



def test_parse_persistence_records_house_provenance_and_artifact_hash(tmp_path):
    source, db = _source(tmp_path)
    pdf_path = tmp_path / "2021" / "pdfs" / "doc.pdf"
    pdf_path.parent.mkdir(parents=True)
    pdf_bytes = b"%PDF-test-artifact"
    pdf_path.write_bytes(pdf_bytes)
    result = (
        pdf_path,
        [
            {
                "transaction_date": date(2021, 1, 2),
                "ticker": "AAPL",
                "transaction_type": "Purchase",
                "owner_code": "SP",
                "amount_raw": "$1,001 - $15,000",
                "amount_midpoint": 8000.5,
                "instrument_type": "stock",
                "strike_price": None,
                "expiry_date": None,
                "asset_description": "Apple Inc.",
            }
        ],
        ["pdfplumber", "won:pdfplumber"],
    )
    member_lookup = {
        "doc": {
            "First": "Jane",
            "Last": "Doe",
            "FilingDate": pd.Timestamp("2021-01-03"),
        }
    }

    try:
        source._save_parse_results(2021, [result], member_lookup)
        stored = db.get_transactions_for_doc("doc").iloc[0]
        parse_run = db.conn.execute(
            """
            SELECT parser_version, status, raw_row_count, transaction_count
            FROM pdf_parse_runs WHERE doc_id = 'doc'
            """
        ).fetchone()
    finally:
        source.close()
        db.close()

    assert stored["chamber"] == "house"
    assert stored["source_record_id"] == "doc"
    assert stored["official_filing_date"].date() == date(2021, 1, 3)
    assert stored["artifact_sha256"] == hashlib.sha256(pdf_bytes).hexdigest()
    assert parse_run == ("v4-deterministic", "success", 1, 1)
