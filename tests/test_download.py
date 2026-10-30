import hashlib
import os
import signal
import zipfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from datetime import date, datetime
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from analyzer.persistence.database import Database
from analyzer.download import HouseTransactionSource, PARSE_VERSION
from analyzer.exceptions import DataSourceError
from analyzer.models import DownloadResult, DownloadStatus
from analyzer.settings import Settings

pytestmark = pytest.mark.slow


def _source(tmp_path, *, db_name="test.duckdb"):
    settings = Settings()
    settings.data.data_dir = str(tmp_path)
    db = Database(tmp_path / db_name)
    return HouseTransactionSource(settings, db=db), db


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    import socket

    def refused(*args, **kwargs):
        raise AssertionError("House download tests must not access the network")

    monkeypatch.setattr(socket.socket, "connect", refused)
    monkeypatch.setattr(socket, "create_connection", refused)


def _pipeline_source(tmp_path, monkeypatch, docs, *, year=2021, filing_date="2022-01-04"):
    from analyzer.ocr import common, coordinator

    source, db = _source(tmp_path, db_name="congress.duckdb")
    source.parallel_workers = 2
    db.conn.execute("INSERT INTO house_archive_generations (archive_year, generation_id, metadata_count, ptr_count) VALUES (?, 'captured-g1', ?, ?)", [year, len(docs), len(docs)])
    for doc in docs:
        pdf = tmp_path / str(year) / "pdfs" / f"{doc}.pdf"
        pdf.parent.mkdir(parents=True, exist_ok=True)
        pdf.write_bytes(b"%PDF-" + doc.encode() + b"\n%%EOF")
        digest = hashlib.sha256(pdf.read_bytes()).hexdigest()
        db.conn.execute("INSERT INTO metadata (doc_id, archive_year, first_name, last_name, filing_date, filing_type) VALUES (?, ?, 'First', 'Last', ?, 'P')", [doc, year, filing_date])
        db.conn.execute("INSERT INTO house_generation_metadata (archive_year, generation_id, doc_id, first_name, last_name, filing_date, filing_type) VALUES (?, 'captured-g1', ?, 'First', 'Last', ?, 'P')", [year, doc, filing_date])
        db.conn.execute("INSERT INTO house_pdf_artifacts (archive_year, generation_id, doc_id, artifact_sha256) VALUES (?, 'captured-g1', ?, ?)", [year, doc, digest])
    monkeypatch.setattr(common, "pdf_page_count", lambda _: 1)
    monkeypatch.setattr(coordinator, "ProcessPoolExecutor", lambda slots, **kwargs: ThreadPoolExecutor(slots))
    return source, db


def _local_transaction(**overrides):
    return {"ticker": "AAPL", "transaction_type": "Purchase", "transaction_date": "2021-01-02",
            "notification_date": "2021-01-03", "owner_code": "JT", "amount_raw": "$1,001 - $15,000",
            "amount_midpoint": 8000.5, "instrument_type": "stock", "asset_description": "Original stock [ST]",
            "source_row_id": "r1", **overrides}


def _pipeline_doc(path):
    return path.read_bytes().splitlines()[0].decode().removeprefix("%PDF-")


def _pipeline_backend(monkeypatch, backend):
    from analyzer.ocr import workers
    from analyzer.parsing import cascade

    run = workers.subprocess.run

    def text_source(command, *args, **kwargs):
        if command[0] == "pdftotext":
            return SimpleNamespace(returncode=0, stdout=Path(command[-2]).read_text(), stderr="")
        return run(command, *args, **kwargs)

    monkeypatch.setattr(cascade, "parse_pdf_worker", backend)
    monkeypatch.setattr(workers.subprocess, "run", text_source)


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


def test_identical_ptr_refresh_skips_promote_but_metadata_correction_promotes(
    tmp_path,
):
    source, db = _source(tmp_path)
    provenance = {
        "metadata_sha256": "metadata-sha",
        "metadata_http_status": 200,
        "metadata_etag": None,
        "metadata_last_modified": None,
    }

    def archive(first_name, fetched_at):
        metadata, _ = _acquired("ptr")
        non_ptr = metadata.iloc[[0]].copy()
        non_ptr["doc_id"] = "non-ptr"
        non_ptr["filing_type"] = "F"
        metadata = pd.concat([metadata, non_ptr], ignore_index=True)
        metadata.loc[metadata["doc_id"] == "ptr", "first_name"] = first_name
        metadata["fetched_at"] = pd.Timestamp(fetched_at)
        return metadata, provenance

    source._acquire_metadata_archive = MagicMock(
        side_effect=[
            archive("First", "2021-01-01"),
            archive("First", "2022-01-01"),
            archive("Corrected", "2023-01-01"),
        ]
    )
    pdf_bytes = b"%PDF-identical\n%%EOF"

    async def download(_session, doc_id, pdf_path, _url):
        pdf_path.write_bytes(pdf_bytes)
        return DownloadResult(doc_id=doc_id, status=DownloadStatus.SUCCESS)

    source._download_pdf_async = download
    promote = MagicMock(wraps=db.promote_house_archive)
    db.promote_house_archive = promote
    try:
        first = source.fetch_and_cache_pdfs(2021, refresh_metadata=True)
        unchanged = source.fetch_and_cache_pdfs(2021, refresh_metadata=True)
        corrected = source.fetch_and_cache_pdfs(2021, refresh_metadata=True)
        stored = db.get_metadata(2021)
    finally:
        source.close()
        db.close()

    assert first.generation_status == "incomplete"
    assert unchanged.generation_status == "unchanged"
    assert corrected.generation_status == "incomplete"
    assert promote.call_count == 2
    assert stored.loc[stored["DocID"] == "ptr", "First"].item() == "Corrected"


@pytest.mark.parametrize("canonical_state", ["missing", "corrupt"])
def test_identical_staged_pdf_repairs_invalid_canonical_before_skip(
    tmp_path, canonical_state
):
    source, db = _source(tmp_path)
    source._acquire_metadata_archive = MagicMock(return_value=_acquired("ptr"))
    pdf_bytes = b"%PDF-valid-replacement\n%%EOF"

    async def download(_session, doc_id, pdf_path, _url):
        pdf_path.write_bytes(pdf_bytes)
        return DownloadResult(doc_id=doc_id, status=DownloadStatus.SUCCESS)

    source._download_pdf_async = download
    canonical = tmp_path / "2021" / "pdfs" / "ptr.pdf"
    try:
        source.fetch_and_cache_pdfs(2021, refresh_metadata=True)
        if canonical_state == "missing":
            canonical.unlink()
        else:
            canonical.write_bytes(b"%PDF-corrupt")

        summary = source.fetch_and_cache_pdfs(2021, refresh_metadata=True)
        assert summary.generation_status != "unchanged"
        assert canonical.read_bytes() == pdf_bytes
    finally:
        source.close()
        db.close()



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
                "source_row_id": "pdfplumber:r1",
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
        source._save_parse_results(
            2021, [result], member_lookup, "legacy-untracked-2021"
        )
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
    assert parse_run == ("v7-deterministic-row-context", "success", 1, 1)



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
            "legacy-untracked-2021",
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
            WHERE doc_id = 'zero' AND parser_version = 'v7-deterministic-row-context'
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
        db.mark_house_generation_parse_complete(
            2021, db.get_latest_house_generation(2021)
        )
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



def test_parse_cached_pdfs_binds_save_to_captured_generation(tmp_path, monkeypatch):
    from analyzer.ocr import workers

    source, db = _pipeline_source(tmp_path, monkeypatch, ["bound"])
    captured = []
    claim = db.work.claim

    def advance_generation(task_id, **kwargs):
        work = claim(task_id, **kwargs)
        captured.append(work.generation_id)
        if work.generation_id == "captured-g1":
            db.conn.execute("INSERT INTO house_archive_generations (archive_year, generation_id, metadata_count, ptr_count, promoted_at) VALUES (2021, 'replacement-g2', 1, 1, '2099-01-01')")
            db.conn.execute("INSERT INTO house_generation_metadata SELECT archive_year, 'replacement-g2', doc_id, first_name, last_name, filing_date, filing_type, fetched_at FROM house_generation_metadata WHERE generation_id='captured-g1'")
            db.conn.execute("INSERT INTO house_pdf_artifacts (archive_year, generation_id, doc_id, artifact_sha256) SELECT archive_year, 'replacement-g2', doc_id, artifact_sha256 FROM house_pdf_artifacts WHERE generation_id='captured-g1'")
        return work

    monkeypatch.setattr(db.work, "claim", advance_generation)
    _pipeline_backend(monkeypatch, lambda path: (path, [_local_transaction()], ["pdftotext"]))
    try:
        source.parse_cached_pdfs(2021, force=True)
        assert captured == ["captured-g1"]
        assert db.conn.execute("SELECT COUNT(*) FROM transactions").fetchone() == (0,)
        assert db.work.status()["by_state"] == {"blocked": 1}
        source.parse_cached_pdfs(2021)
        assert captured == ["captured-g1", "replacement-g2"]
        assert db.conn.execute("SELECT ingestion_generation FROM transactions").fetchall() == [("replacement-g2",)]
        bundles = [workers.load_result(path) for path in (tmp_path / "pipeline" / "results").glob("*.json")]
        assert {bundle.work.generation_id for bundle in bundles} == {"captured-g1", "replacement-g2"}
    finally:
        source.close()
        db.close()


@pytest.mark.parametrize("opt_in,force", [
    (None, False),
    ("2026", False),
    ("2025", False),
    ("2026", True),
    ("invalid", True),
])
def test_pipeline_cache_follows_bound_readiness_not_retired_legacy_environment(
    tmp_path, monkeypatch, opt_in, force
):
    from analyzer.download import validated_pdf_sha256
    from analyzer.ocr.workers import extraction_recipe
    from analyzer.ocr.contracts import ExtractionLane
    from analyzer.parsing.metadata import consolidate_transactions

    monkeypatch.delenv("PTR_REUSE_LEGACY_V5_CACHE_YEAR", raising=False)
    if opt_in is not None:
        monkeypatch.setenv("PTR_REUSE_LEGACY_V5_CACHE_YEAR", opt_in)
    docs = ("legacy", "current", "fresh", "changed", "oldgen", "failed")
    source, db = _pipeline_source(tmp_path, monkeypatch, docs, year=2026, filing_date="2026-10-20")
    generation = "captured-g1"
    pdf_dir = tmp_path / "2026" / "pdfs"
    transaction = dict(
        ticker="AAPL", transaction_type="Purchase", transaction_date="09/29/2026",
        owner_code="JT", amount_raw="$1,001 - $15,000", amount_midpoint=8000.5,
        asset_description="Original stock [ST]", instrument_type="stock", source_row_id="r1",
    )
    parsed = []

    def worker(path):
        parsed.append(_pipeline_doc(path))
        return path, [{**transaction, "notification_date": "09/30/2026"}], ["pdfplumber"]

    _pipeline_backend(monkeypatch, worker)
    lookup = {doc: dict(First="First", Last="Last", FilingDate="2026-10-20") for doc in docs}
    try:
        # Seed genuine legacy versions and values, never label old results as v6.
        for doc in ("legacy", "current", "changed", "oldgen", "failed"):
            artifact = validated_pdf_sha256(pdf_dir / f"{doc}.pdf")
            if doc == "changed":
                artifact = "f" * 64
            bound_generation = "old-generation" if doc == "oldgen" else generation
            frame = consolidate_transactions({Path(f"{doc}.pdf"): [transaction]}, lookup)
            frame["chamber"] = "house"
            frame["official_filing_date"] = pd.Timestamp("2026-10-20")
            frame["source_record_id"] = doc
            frame["ingestion_generation"] = bound_generation
            frame["artifact_sha256"] = artifact
            db.upsert_transactions(frame, source="house_pdf")
            db.upsert_parse_run(
                doc_id=doc, year=2026,
                parser_version=PARSE_VERSION if doc == "current" else "v5-deterministic",
                status="error" if doc == "failed" else "success", engines_attempted="pdfplumber",
                raw_row_count=1, transaction_count=1,
                artifact_sha256=artifact, ingestion_generation=bound_generation,
            )
        before_rows = db.conn.execute("SELECT * FROM transactions WHERE doc_id = 'legacy'").fetchall()
        before_runs = db.conn.execute("SELECT * FROM pdf_parse_runs WHERE doc_id = 'legacy'").fetchall()
        assert set(db.get_unresolved_house_doc_ids(2026, generation)) == {"fresh", "changed", "oldgen", "failed"}
        source.parse_cached_pdfs(2026, force=force)
        expected = set(docs) if force else set(docs) - {"legacy", "current"}
        assert set(parsed) == expected
        assert db.conn.execute("SELECT COUNT(*) FROM extraction_attempts").fetchone() == (len(expected),)
        if not force:
            assert db.conn.execute("SELECT * FROM transactions WHERE doc_id = 'legacy'").fetchall() == before_rows
            assert db.conn.execute("SELECT * FROM pdf_parse_runs WHERE doc_id = 'legacy'").fetchall() == before_runs
        recipe = extraction_recipe(ExtractionLane.LOCAL)
        for doc in expected:
            assert db.conn.execute(
                "SELECT status FROM pdf_parse_runs WHERE doc_id = ? AND parser_version = ?",
                [doc, recipe],
            ).fetchall() == [("success",)]
            stored = db.get_transactions_for_doc(doc)
            assert stored["notification_date"].eq(pd.Timestamp("2026-09-30")).any()
        source.parse_cached_pdfs(2026)
        assert len(parsed) == len(expected)  # Receipts and canonical binding prevent another parse.
    finally:
        source.close()
        db.close()


@pytest.mark.parametrize("value", ["", "1", "true", "2026,2025", "0000"])
def test_retired_legacy_environment_cannot_change_the_coordinator_contract(monkeypatch, tmp_path, value):
    from analyzer.ocr import coordinator

    monkeypatch.setenv("PTR_REUSE_LEGACY_V5_CACHE_YEAR", value)
    source = SimpleNamespace(db=MagicMock(), data_dir=tmp_path, parallel_workers=2)
    run = MagicMock(return_value={"unresolved": 0})
    monkeypatch.setattr(coordinator, "run_pipeline", run)
    HouseTransactionSource.parse_cached_pdfs(source, 2026)
    run.assert_called_once_with(tmp_path, [2026], local_workers=2, refresh_recipe=False, database=source.db)


def test_parse_cached_pdfs_persists_large_year_in_resumable_batches(
    tmp_path, monkeypatch
):
    stems = [f"doc{i:02d}" for i in range(17)]
    source, db = _pipeline_source(tmp_path, monkeypatch, stems)
    commits = MagicMock(wraps=db.replace_transactions_for_docs)
    monkeypatch.setattr(db, "replace_transactions_for_docs", commits)
    parsed = []

    def backend(path):
        parsed.append(_pipeline_doc(path))
        return path, [_local_transaction()], ["pdftotext"]

    _pipeline_backend(monkeypatch, backend)
    try:
        source.parse_cached_pdfs(2021, force=True)
        assert sorted(parsed) == stems
        assert commits.call_count == len(stems)
        for call in commits.call_args_list:
            assert len(call.args[0]) == 1
            assert len(call.kwargs["attempted_doc_ids"]) == 1
            assert db.work.is_finished(call.kwargs["receipt"])
        assert db.work.status()["by_state"] == {"accepted": 17}
        assert db.conn.execute("SELECT COUNT(*) FROM transactions").fetchone() == (17,)
        source.parse_cached_pdfs(2021)
        assert len(parsed) == 17
        assert db.conn.execute("SELECT COUNT(*) FROM extraction_attempts").fetchone() == (17,)
    finally:
        source.close()
        db.close()


def test_parse_cached_pdfs_isolates_per_pdf_cascade_failures(
    tmp_path, monkeypatch
):
    from analyzer.ocr import workers
    from analyzer.parsing.cascade import ParserCascadeError

    source, db = _pipeline_source(tmp_path, monkeypatch, ["goodone", "badone"])
    parsed = []

    def fake_worker(path):
        parsed.append(_pipeline_doc(path))
        if _pipeline_doc(path) == "badone":
            raise ParserCascadeError(f"{path}: unresolved parser completeness: boom")
        return path, [_local_transaction()], ["pdfplumber", "won:pdfplumber"]

    _pipeline_backend(monkeypatch, fake_worker)
    try:
        source.parse_cached_pdfs(2021, force=True)
        assert sorted(parsed) == ["badone", "goodone"]
        assert db.conn.execute("SELECT doc_id FROM canonical_transactions").fetchall() == [("goodone",)]
        assert db.work.status()["by_state"] == {"accepted": 1, "needs_review": 1}
        bundles = {bundle.work.doc_id: bundle for bundle in (
            workers.load_result(path) for path in (tmp_path / "pipeline" / "results").glob("*.json"))}
        assert bundles["badone"].evidence["failure"]["type"] == "ParserCascadeError"
        assert "boom" in bundles["badone"].evidence["failure"]["message"]
        assert bundles["badone"].raw_response == "%PDF-badone\n%%EOF"
        source.parse_cached_pdfs(2021)
        assert len(parsed) == 2
    finally:
        source.close()
        db.close()


def test_failed_reparse_preserves_terminal_run_rows_and_generation_audit(
    tmp_path, monkeypatch
):
    from analyzer.ocr import workers
    from analyzer.parsing.cascade import ParserCascadeError
    from scripts.audit_generation import (
        check_house_generation_activation,
        check_parse_counts_match_persisted,
    )

    source, db = _pipeline_source(tmp_path, monkeypatch, ["stable"], filing_date="2021-01-03")
    generation = "captured-g1"
    pdf_path = tmp_path / "2021" / "pdfs" / "stable.pdf"
    artifact_sha = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    db.conn.execute("UPDATE house_archive_generations SET parse_status='complete'")
    db.upsert_transactions(
        pd.DataFrame(
            [{
                "doc_id": "stable",
                "member": "First Last",
                "ticker": "AAPL",
                "transaction_date": date(2021, 1, 2),
                "disclosure_date": date(2021, 1, 3),
                "official_filing_date": date(2021, 1, 3),
                "transaction_type": "Purchase",
                "chamber": "house",
                "source_record_id": "stable",
                "source_row_id": "stable:r1",
                "ingestion_generation": generation,
                "artifact_sha256": artifact_sha,
            }]
        ),
        source="house_pdf",
    )
    db.upsert_parse_run(
        doc_id="stable",
        year=2021,
        parser_version=PARSE_VERSION,
        status="success",
        engines_attempted="pdfplumber",
        raw_row_count=1,
        transaction_count=1,
        artifact_sha256=artifact_sha,
        ingestion_generation=generation,
    )
    db.conn.execute(
        """
        INSERT INTO source_reports (
            ingestion_generation, source, chamber, source_record_id,
            report_path, member, official_filing_date, outcome,
            artifact_sha256, landing_sha256, error_message,
            raw_row_count, accepted_row_count, rejected_row_count
        ) VALUES (?, 'house_pdf', 'house', 'stable', '2021/pdfs/stable.pdf',
                  'First Last', '2021-01-03', 'parsed', ?, ?, NULL, 1, 1, 0)
        """,
        [generation, artifact_sha, artifact_sha],
    )

    def failed_worker(path):
        raise ParserCascadeError(f"{path}: unresolved parser completeness: boom")

    _pipeline_backend(monkeypatch, failed_worker)
    before_rows = db.conn.execute("SELECT * FROM transactions").fetchall()
    before_versions = db.conn.execute("SELECT * FROM transaction_versions").fetchall()
    try:
        source.parse_cached_pdfs(2021, force=True)
        assert db.work.status()["by_state"] == {"needs_review": 1}
        assert db.conn.execute("SELECT * FROM transactions").fetchall() == before_rows
        assert db.conn.execute("SELECT * FROM transaction_versions").fetchall() == before_versions
        bundle = workers.load_result(next((tmp_path / "pipeline" / "results").glob("*.json")))
        assert "boom" in bundle.evidence["failure"]["message"]
        runs = db.conn.execute(
            """
            SELECT status, transaction_count
            FROM pdf_parse_runs
            WHERE doc_id = 'stable'
              AND parser_version = 'v7-deterministic-row-context'
              AND artifact_sha256 = ?
              AND ingestion_generation = ?
            """,
            [artifact_sha, generation],
        ).fetchall()
        rows = db.conn.execute(
            """
            SELECT ticker, ingestion_generation, artifact_sha256
            FROM canonical_transactions
            WHERE doc_id = 'stable'
            """
        ).fetchall()
        unresolved = db.get_unresolved_house_doc_ids(2021, generation)
        parse_audit = check_parse_counts_match_persisted(db.conn)
        generation_audit = check_house_generation_activation(db.conn)
    finally:
        source.close()
        db.close()

    assert runs == [("success", 1)]
    assert rows == [("AAPL", generation, artifact_sha)]
    assert unresolved == []
    assert parse_audit.passed, parse_audit.violations
    assert generation_audit.passed, generation_audit.violations


def test_engine_error_detail_only_flags_dedicated_sentinel():
    from analyzer.download import engine_error_detail

    transient = ["pdfplumber", "error:lattice", "won:pdftotext"]
    assert engine_error_detail(transient) is None
    sentinel = ["__parse_failed__:ParserCascadeError:boom"]
    assert engine_error_detail(sentinel) == "ParserCascadeError:boom"


def test_parse_cached_pdfs_saves_docs_with_transient_engine_errors(
    tmp_path, monkeypatch
):
    from analyzer.ocr import workers

    source, db = _pipeline_source(tmp_path, monkeypatch, ["mixedone"])

    def fake_worker(path):
        return path, [_local_transaction()], ["lattice", "error:lattice", "won:pdftotext"]

    _pipeline_backend(monkeypatch, fake_worker)
    try:
        source.parse_cached_pdfs(2021, force=True)
        assert db.work.status()["by_state"] == {"accepted": 1}
        assert db.conn.execute("SELECT doc_id FROM canonical_transactions").fetchall() == [("mixedone",)]
        bundle = workers.load_result(next((tmp_path / "pipeline" / "results").glob("*.json")))
        assert bundle.outcome == "candidate"
        assert bundle.evidence["engines"] == ["lattice", "error:lattice", "won:pdftotext"]
        assert "failure" not in bundle.evidence
    finally:
        source.close()
        db.close()


def test_local_notification_before_transaction_is_reviewable_without_guessing_or_redispatch(tmp_path, monkeypatch):
    from analyzer.ocr import coordinator, workers

    source, db = _pipeline_source(tmp_path, monkeypatch, ["badnf"], filing_date="2021-01-03")
    pdf = tmp_path / "2021" / "pdfs" / "badnf.pdf"
    raw_source = "%PDF-badnf\nTX: 01/02/2021\nNF: 01/01/2021\n%%EOF"
    pdf.write_text(raw_source)
    db.conn.execute("UPDATE house_pdf_artifacts SET artifact_sha256=? WHERE doc_id='badnf'", [hashlib.sha256(pdf.read_bytes()).hexdigest()])
    row = _local_transaction(notification_date="2021-01-01", raw_cells={"TX": "01/02/2021", "NF": "01/01/2021"})
    calls = []

    def backend(path):
        calls.append(_pipeline_doc(path))
        return path, [row], ["pdftotext"]

    _pipeline_backend(monkeypatch, backend)
    try:
        source.parse_cached_pdfs(2021, force=True)
        bundle = workers.load_result(next((tmp_path / "pipeline" / "results").glob("*.json")))
        assert bundle.raw_response == raw_source
        assert bundle.rows == [row]
        assert bundle.rows[0]["raw_cells"]["NF"] == "01/01/2021"
        assert db.work.status()["by_state"] == {"needs_review": 1}
        assert db.get_unresolved_house_doc_ids(2021, "captured-g1") == ["badnf"]
        assert db.conn.execute("SELECT COUNT(*) FROM transactions").fetchone() == (0,)
        assert db.conn.execute("SELECT COUNT(*) FROM transaction_versions").fetchone() == (0,)
        status = coordinator.pipeline_status(tmp_path, database=db)
        assert status["documents"]["by_state"] == {"needs_review": 1}
        source.parse_cached_pdfs(2021)
        assert calls == ["badnf"]
        assert db.conn.execute("SELECT COUNT(*) FROM extraction_attempts").fetchone() == (1,)
        assert db.conn.execute("SELECT state, finished_at IS NOT NULL FROM extraction_results").fetchall() == [("needs_review", True)]
    finally:
        source.close()
        db.close()



def test_house_get_transactions_includes_pdf_and_gemini_but_not_capitol(tmp_path):
    source, db = _source(tmp_path)
    base = {
        "member": "Jane Doe",
        "transaction_date": date(2024, 1, 2),
        "disclosure_date": date(2024, 1, 3),
        "transaction_type": "Purchase",
    }
    for doc_id, ticker, row_source in (
        ("pdf", "PDF", "house_pdf"),
        ("ocr", "OCR", "gemini_ocr"),
        ("capitol", "CAP", "capitol_trades"),
    ):
        db.upsert_transactions(
            pd.DataFrame([{**base, "doc_id": doc_id, "ticker": ticker}]),
            source=row_source,
        )

    try:
        transactions = source.get_transactions(2024)
    finally:
        source.close()
        db.close()

    assert set(transactions["ticker"]) == {"PDF", "OCR"}



def test_house_get_transactions_returns_empty_for_capitol_only_year(tmp_path):
    source, db = _source(tmp_path)
    db.upsert_transactions(
        pd.DataFrame(
            [{
                "doc_id": "capitol-only",
                "member": "Jane Doe",
                "ticker": "CAP",
                "transaction_date": date(2024, 1, 2),
                "disclosure_date": date(2024, 1, 3),
                "transaction_type": "Purchase",
            }]
        ),
        source="capitol_trades",
    )

    try:
        transactions = source.get_transactions(2024)
    finally:
        source.close()
        db.close()
    assert transactions.empty
    assert "ticker" in transactions.columns



def test_house_get_transactions_returns_empty_for_ineligible_house_rows(tmp_path):
    source, db = _source(tmp_path)
    db.upsert_transactions(
        pd.DataFrame(
            [{
                "doc_id": "invalid-house-only",
                "member": "Jane Doe",
                "ticker": "BAD",
                "transaction_date": date(2021, 1, 4),
                "disclosure_date": date(2021, 1, 3),
                "transaction_type": "Purchase",
            }]
        ),
        source="house_pdf",
    )

    try:
        transactions = source.get_transactions(2021)
    finally:
        source.close()
        db.close()
    assert transactions.empty
    assert "ticker" in transactions.columns


def test_parse_budget_exceeded_is_runtime_error():
    from analyzer.download import ParseBudgetExceeded

    assert issubclass(ParseBudgetExceeded, RuntimeError)


def test_skip_stems_from_env_parses_comma_separated_values():
    from analyzer.download import _skip_stems_from_env

    assert _skip_stems_from_env("20004297, 20012345,,  20099888 ") == frozenset(
        {"20004297", "20012345", "20099888"}
    )
    assert _skip_stems_from_env(None) == frozenset()
    assert _skip_stems_from_env("") == frozenset()


def test_tolerant_parse_watchdog_budget_fires_and_restores_state(
    tmp_path, monkeypatch
):
    import time as time_module

    from analyzer import download as download_module

    pdf_path = tmp_path / "slow.pdf"
    pdf_path.write_bytes(b"%PDF-slow\n%%EOF")
    prior_handler = signal.getsignal(signal.SIGALRM)
    monkeypatch.setattr(download_module, "_PARSE_DOC_BUDGET_SECONDS", 0.05)

    def slow_worker(_path):
        time_module.sleep(0.5)
        return _path, [], ["pdfplumber"]

    monkeypatch.setattr(download_module, "parse_pdf_worker", slow_worker)

    start = time_module.monotonic()
    result = download_module.tolerant_parse_pdf_worker(pdf_path)
    elapsed = time_module.monotonic() - start

    out_path, transactions, engines = result
    assert out_path == pdf_path
    assert transactions == []
    assert engines == [
        f"__parse_failed__:parse budget "
        f"{download_module._PARSE_DOC_BUDGET_SECONDS}s exceeded"
    ]
    assert elapsed < 0.4, f"watchdog did not interrupt the slow worker ({elapsed}s)"
    assert signal.getsignal(signal.SIGALRM) is prior_handler
    assert signal.setitimer(signal.ITIMER_REAL, 0) == (0.0, 0.0)


def test_tolerant_worker_keeps_backend_timeout_as_parse_failure(tmp_path, monkeypatch):
    from analyzer import download as download_module
    from analyzer.parsing import cascade as parser_cascade

    pdf_path = tmp_path / "backend-timeout.pdf"
    pdf_path.write_bytes(b"%PDF-backend-timeout\n%%EOF")

    def raise_timeout(_path):
        raise parser_cascade.ParseBudgetExceeded("backend timeout")

    monkeypatch.setattr(
        parser_cascade, "extract_tables_with_pdfplumber", raise_timeout
    )

    out_path, transactions, engines = download_module.tolerant_parse_pdf_worker(
        pdf_path
    )

    assert out_path == pdf_path
    assert transactions == []
    assert engines == [
        f"__parse_failed__:parse budget "
        f"{download_module._PARSE_DOC_BUDGET_SECONDS}s exceeded"
    ]


def test_tolerant_parse_watchdog_absent_outside_main_thread(tmp_path, monkeypatch):
    import threading
    import time as time_module

    from analyzer import download as download_module

    pdf_path = tmp_path / "threaded.pdf"
    pdf_path.write_bytes(b"%PDF-threaded\n%%EOF")
    monkeypatch.setattr(download_module, "_PARSE_DOC_BUDGET_SECONDS", 0.01)
    outcome = {}

    def slow_worker(path):
        time_module.sleep(0.1)
        return path, [], ["pdfplumber", "won:pdfplumber"]

    monkeypatch.setattr(download_module, "parse_pdf_worker", slow_worker)

    def run():
        outcome["result"] = download_module.tolerant_parse_pdf_worker(pdf_path)
        outcome["handler"] = signal.getsignal(signal.SIGALRM)

    thread = threading.Thread(target=run)
    thread.start()
    thread.join(timeout=5)
    assert not thread.is_alive()

    out_path, transactions, engines = outcome["result"]
    assert out_path == pdf_path
    assert transactions == []
    assert engines == ["pdfplumber", "won:pdfplumber"]
    assert outcome["handler"] == signal.SIG_DFL


def test_tolerant_parse_skips_docs_via_ptr_skip_docs(tmp_path, monkeypatch):
    from analyzer import download as download_module

    skipped_path = tmp_path / "20004297.pdf"
    normal_path = tmp_path / "20011111.pdf"
    monkeypatch.setenv("PTR_SKIP_DOCS", "20004297")

    def explode_if_called(_path):
        raise AssertionError("parse_pdf_worker must not run for skipped docs")

    monkeypatch.setattr(download_module, "parse_pdf_worker", explode_if_called)

    out_path, transactions, engines = download_module.tolerant_parse_pdf_worker(
        skipped_path
    )
    assert out_path == skipped_path
    assert transactions == []
    assert engines == ["__parse_failed__:skipped via PTR_SKIP_DOCS"]

    monkeypatch.setattr(
        download_module,
        "parse_pdf_worker",
        lambda path: (path, [], ["won:pdfplumber"]),
    )
    out_path, transactions, engines = download_module.tolerant_parse_pdf_worker(
        normal_path
    )
    assert engines == ["won:pdfplumber"]
