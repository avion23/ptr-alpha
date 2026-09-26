from multiprocessing.pool import ThreadPool
from pathlib import Path
import time

import pandas as pd
import pytest

from analyzer.download import _validated_pdf_sha256
from analyzer.parsing.metadata import consolidate_transactions
from .test_download import _source, _metadata


def _tx(day="2021-01-02", row="r1"):
    return dict(transaction_date=day, source_row_id=row, ticker="AAPL",
                transaction_type="Purchase", asset_description="Apple")


LOOKUP = {"doc": dict(First="Jane", Last="Doe", FilingDate="2021-01-03")}


@pytest.mark.parametrize("partial", [True, False])
def test_dropped_rows_fail_closed(tmp_path, partial):
    source, db = _source(tmp_path)
    path = tmp_path / "doc.pdf"
    path.write_bytes(b"%PDF-test\n%%EOF")
    old = consolidate_transactions({path: [_tx()]}, LOOKUP)
    old["ticker"] = "MSFT"
    db.upsert_transactions(old, source="house_pdf")
    rows = ([_tx()] if partial else []) + [_tx("invalid", "r2")]
    try:
        source._save_parse_results(2021, [(path, rows, ["pdfplumber"])], LOOKUP, "legacy-untracked-2021")
        run = db.conn.execute("SELECT status, raw_row_count FROM pdf_parse_runs WHERE doc_id='doc'").fetchone()
        assert run == ("error", len(rows))
        assert db.get_transactions_for_doc("doc")["ticker"].tolist() == ["MSFT"]
        artifact_hash = _validated_pdf_sha256(path)
        assert artifact_hash is not None
        assert not db.parse_runs.get_cached_doc_ids(
            year=2021, parser_version="v5-deterministic",
            artifact_hashes={"doc": artifact_hash},
            ingestion_generation="legacy-untracked-2021",
        )
    finally:
        source.close()
        db.close()


@pytest.mark.parametrize("dates", [("01/02/2021", "2021-01-02"), ("2021-01-02", "01/02/2021")])
def test_mixed_dates_survive(dates):
    lookup = {str(i): dict(First="Jane", Last="Doe", FilingDate=day) for i, day in enumerate(dates)}
    result = consolidate_transactions({Path(f"{i}.pdf"): [_tx(day)] for i, day in enumerate(dates)}, lookup)
    assert len(result) == 2
    assert result.transaction_date.eq(pd.Timestamp("2021-01-02")).all()
    assert result.disclosure_date.eq(pd.Timestamp("2021-01-02")).all()


def test_completed_pdf_checkpoint_precedes_stalled_batch(tmp_path, monkeypatch):
    source, db = _source(tmp_path)
    source.parallel_workers = 2
    pdf_dir = tmp_path / "2021" / "pdfs"
    pdf_dir.mkdir(parents=True)
    for doc in ("slow", "fast"):
        (pdf_dir / f"{doc}.pdf").write_bytes(b"%PDF-test\n%%EOF")
    monkeypatch.setattr(source, "fetch_metadata", lambda year: _metadata("slow", "fast"))
    monkeypatch.setattr(db, "get_latest_house_generation", lambda year: "legacy-untracked-2021")
    monkeypatch.setattr("analyzer.download.Pool", ThreadPool)

    def worker(path):
        if path.stem == "slow":
            time.sleep(0.5)
            raise RuntimeError("interrupted batch")
        return path, [_tx()], ["pdfplumber"]

    monkeypatch.setattr("analyzer.download._tolerant_parse_pdf_worker", worker)
    try:
        with pytest.raises(RuntimeError, match="interrupted batch"):
            source.parse_cached_pdfs(2021, force=True)
        assert len(db.get_transactions_for_doc("fast")) == 1
        assert db.conn.execute("SELECT status FROM pdf_parse_runs WHERE doc_id='fast'").fetchone() == ("success",)
    finally:
        source.close()
        db.close()
