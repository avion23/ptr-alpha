import hashlib
import os
import zipfile
from contextlib import nullcontext
from datetime import date, datetime
from io import BytesIO
from pathlib import Path
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


def _acquired(*doc_ids: str):
    metadata = _metadata(*doc_ids).rename(
        columns={
            "DocID": "doc_id",
            "First": "first_name",
            "Last": "last_name",
            "FilingDate": "filing_date",
            "FilingType": "filing_type",
        }
    )
    metadata["archive_year"] = 2021
    metadata["fetched_at"] = datetime.now()
    return metadata, {
        "metadata_sha256": "metadata-sha",
        "metadata_http_status": 200,
        "metadata_etag": None,
        "metadata_last_modified": None,
    }


def test_cross_year_ptr_uses_archive_year_for_url_and_file_path(tmp_path):
    source, db = _source(tmp_path)
    source._acquire_metadata_archive = MagicMock(return_value=_acquired("8218519"))
    captured = {}

    async def download(_session, doc_id, pdf_path, url):
        captured.update(doc_id=doc_id, pdf_path=pdf_path, url=url)
        pdf_path.write_bytes(b"%PDF-test\n%%EOF")
        return DownloadResult(doc_id=doc_id, status=DownloadStatus.SUCCESS)

    source._download_pdf_async = download
    try:
        summary = source.fetch_and_cache_pdfs(2021, refresh_metadata=False)
    finally:
        source.close()
        db.close()

    assert captured["doc_id"] == "8218519"
    assert captured["pdf_path"].name == "8218519.pdf"
    assert ".staging/house/2021" in str(captured["pdf_path"])
    assert (tmp_path / "2021" / "pdfs" / "8218519.pdf").exists()
    assert captured["url"].endswith("/ptr-pdfs/2021/8218519.pdf")
    assert summary.ptr_count == summary.valid_pdf_count == 1


def test_incomplete_pdf_batch_reports_every_missing_doc_id(tmp_path):
    source, db = _source(tmp_path)
    source._acquire_metadata_archive = MagicMock(
        return_value=_acquired("present", "missing-a", "missing-b")
    )

    async def download(_session, doc_id, pdf_path, _url):
        if doc_id == "present":
            pdf_path.write_bytes(b"%PDF-test\n%%EOF")
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
    source._acquire_metadata_archive = MagicMock(return_value=_acquired())
    current_year = date.today().year

    try:
        source.fetch_and_cache_pdfs(current_year)
    finally:
        source.close()
        db.close()

    source._acquire_metadata_archive.assert_called_once_with(
        current_year, bypass_cache=True
    )



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
    pdf_bytes = b"%PDF-test-artifact\n%%EOF"
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
    assert stored["ingestion_generation"] == "legacy-untracked-2021"
    assert stored["official_filing_date"].date() == date(2021, 1, 3)
    assert stored["artifact_sha256"] == hashlib.sha256(pdf_bytes).hexdigest()
    assert parse_run == ("v4-deterministic", "success", 1, 1)



def test_zero_output_parse_preserves_stale_house_and_ocr_rows(tmp_path):
    source, db = _source(tmp_path)
    pdf_path = tmp_path / "2021" / "pdfs" / "zero.pdf"
    pdf_path.parent.mkdir(parents=True)
    pdf_path.write_bytes(b"%PDF-zero\n%%EOF")
    base = {
        "doc_id": "zero",
        "member": "Jane Doe",
        "transaction_date": date(2021, 1, 2),
        "disclosure_date": date(2021, 1, 3),
        "transaction_type": "Purchase",
    }
    db.upsert_transactions(
        pd.DataFrame([{**base, "ticker": "AAPL"}]),
        source="gemini_ocr",
    )
    db.upsert_transactions(
        pd.DataFrame([{**base, "ticker": "MSFT"}]),
        source="house_pdf",
    )
    member_lookup = {
        "zero": {
            "First": "Jane",
            "Last": "Doe",
            "FilingDate": pd.Timestamp("2021-01-03"),
        }
    }

    try:
        source._save_parse_results(
            2021,
            [(pdf_path, [], ["pdfplumber", "pdftotext"])],
            member_lookup,
        )
        rows = db.conn.execute(
            """
            SELECT ticker, source FROM transactions
            WHERE doc_id = 'zero' ORDER BY ticker
            """
        ).fetchall()
        parse_run = db.conn.execute(
            """
            SELECT status, raw_row_count, transaction_count
            FROM pdf_parse_runs
            WHERE doc_id = 'zero' AND parser_version = 'v4-deterministic'
            """
        ).fetchone()
    finally:
        source.close()
        db.close()

    assert rows == [("AAPL", "gemini_ocr"), ("MSFT", "house_pdf")]
    assert parse_run == ("zero_rows", 0, 0)



def test_staged_failure_leaves_prior_generation_untouched(tmp_path):
    source, db = _source(tmp_path)
    old_metadata, _ = _acquired("old")
    db.replace_metadata(2021, old_metadata)
    canonical = tmp_path / "2021" / "pdfs" / "old.pdf"
    canonical.parent.mkdir(parents=True)
    canonical.write_bytes(b"%PDF-old\n%%EOF")
    source._acquire_metadata_archive = MagicMock(
        return_value=_acquired("new-a", "new-b")
    )

    async def download(_session, doc_id, pdf_path, _url):
        if doc_id == "new-a":
            pdf_path.write_bytes(b"%PDF-new\n%%EOF")
            return DownloadResult(doc_id=doc_id, status=DownloadStatus.SUCCESS)
        return DownloadResult(
            doc_id=doc_id,
            status=DownloadStatus.FAILED,
            error_message="HTTP 503",
        )

    source._download_pdf_async = download
    try:
        with pytest.raises(DataSourceError):
            source.fetch_and_cache_pdfs(2021, refresh_metadata=True)
        assert db.get_metadata(2021)["DocID"].tolist() == ["old"]
        assert canonical.read_bytes() == b"%PDF-old\n%%EOF"
        assert not (tmp_path / "2021" / "pdfs" / "new-a.pdf").exists()
    finally:
        source.close()
        db.close()


def test_removed_house_rows_remain_until_new_generation_activates(tmp_path):
    source, db = _source(tmp_path)
    old_metadata, _ = _acquired("removed")
    db.replace_metadata(2021, old_metadata)
    old_pdf = tmp_path / "2021" / "pdfs" / "removed.pdf"
    old_pdf.parent.mkdir(parents=True)
    old_pdf.write_bytes(b"%PDF-removed\n%%EOF")
    db.upsert_transactions(
        pd.DataFrame(
            [{
                "doc_id": "removed",
                "member": "Jane Doe",
                "ticker": "AAPL",
                "transaction_date": date(2021, 1, 2),
                "disclosure_date": date(2021, 1, 3),
                "transaction_type": "Purchase",
            }]
        ),
        source="house_pdf",
    )
    source._acquire_metadata_archive = MagicMock(return_value=_acquired("current"))

    async def download(_session, doc_id, pdf_path, _url):
        pdf_path.write_bytes(b"%PDF-current\n%%EOF")
        return DownloadResult(doc_id=doc_id, status=DownloadStatus.SUCCESS)

    source._download_pdf_async = download
    try:
        summary = source.fetch_and_cache_pdfs(2021, refresh_metadata=True)
        audit = db.conn.execute(
            """
            SELECT reason, removed_house_rows, quarantine_path
            FROM house_archive_quarantine WHERE doc_id = 'removed'
            """
        ).fetchone()
        transaction_audit = db.conn.execute(
            """
            SELECT transaction_json FROM house_transaction_quarantine
            WHERE doc_id = 'removed'
            """
        ).fetchone()[0]
        assert db.count_transactions_for_docs(["removed"]) == {"removed": 1}
        assert '"ticker":"AAPL"' in transaction_audit
        current_sha = db.get_house_artifact_hashes(2021)["current"]
        db.upsert_parse_run(
            doc_id="current",
            year=2021,
            parser_version="verified",
            status="no_txs",
            engines_attempted="verified",
            raw_row_count=0,
            transaction_count=0,
            artifact_sha256=current_sha,
            ingestion_generation=db.get_latest_house_generation(2021),
        )
        db.mark_house_generation_parse_complete(2021)
        assert db.count_transactions_for_docs(["removed"]) == {}
        assert not old_pdf.exists()
        assert Path(audit[2]).read_bytes() == b"%PDF-removed\n%%EOF"
    finally:
        source.close()
        db.close()

    assert summary.removed_doc_count == 1
    assert summary.quarantined_pdf_count == 1
    assert audit[:2] == ("removed_from_authoritative_archive", 1)



def test_second_directory_rename_failure_restores_prior_canonical(tmp_path, monkeypatch):
    source, db = _source(tmp_path)
    old_metadata, _ = _acquired("old")
    db.replace_metadata(2021, old_metadata)
    old_pdf = tmp_path / "2021" / "pdfs" / "old.pdf"
    old_pdf.parent.mkdir(parents=True)
    old_pdf.write_bytes(b"%PDF-old\n%%EOF")
    source._acquire_metadata_archive = MagicMock(return_value=_acquired("new"))

    async def download(_session, doc_id, pdf_path, _url):
        pdf_path.write_bytes(b"%PDF-new\n%%EOF")
        return DownloadResult(doc_id=doc_id, status=DownloadStatus.SUCCESS)

    source._download_pdf_async = download
    real_replace = os.replace
    directory_renames = 0

    def fail_second_directory_rename(src, dst):
        nonlocal directory_renames
        if Path(src).is_dir():
            directory_renames += 1
            if directory_renames == 2:
                raise OSError("injected promotion rename failure")
        return real_replace(src, dst)

    monkeypatch.setattr("analyzer.download.os.replace", fail_second_directory_rename)
    try:
        with pytest.raises(OSError, match="injected promotion rename failure"):
            source.fetch_and_cache_pdfs(2021, refresh_metadata=True)
        assert old_pdf.read_bytes() == b"%PDF-old\n%%EOF"
        assert db.get_metadata(2021)["DocID"].tolist() == ["old"]
    finally:
        source.close()
        db.close()
