import hashlib
import os
from types import SimpleNamespace

import pandas as pd


def test_parse_year_uses_house_generation_replacement_api(monkeypatch, tmp_path):
    previous = os.environ.get("PTR_SKIP_DOCLING")
    try:
        from scripts import reparse_all
    finally:
        if previous is None:
            os.environ.pop("PTR_SKIP_DOCLING", None)
        else:
            os.environ["PTR_SKIP_DOCLING"] = previous

    pdf_dir = tmp_path / "2026" / "pdfs"
    pdf_dir.mkdir(parents=True)
    success_path = pdf_dir / "success.pdf"
    zero_path = pdf_dir / "ambiguous-zero.pdf"
    success_path.write_bytes(b"%PDF-success-generation")
    zero_path.write_bytes(b"%PDF-zero-generation")

    metadata = pd.DataFrame({"FilingType": ["P", "P"]})

    class FakeSource:
        def __init__(self, _settings):
            pass

        def fetch_metadata(self, _year):
            return metadata

        def close(self):
            pass

    transactions = [
        {
            "ticker": "NVDA",
            "transaction_type": "Purchase",
            "transaction_date": "06/26/2026",
            "owner_code": None,
            "amount_raw": "$1,001 - $15,000",
            "amount_midpoint": 8000.5,
            "instrument_type": "stock",
            "asset_description": "NVIDIA Corporation (NVDA)",
            "source_row_id": "pdfplumber:p1:l25",
        }
    ]
    results = [
        (success_path, transactions, ["pdfplumber", "won:pdfplumber"]),
        (zero_path, [], ["pdfplumber", "pdftotext", "ocr"]),
    ]

    class FakePool:
        def __init__(self, _workers):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def map(self, worker, paths):
            assert worker is reparse_all._parse_pdf_worker
            assert paths == [success_path, zero_path]
            return results

    class FakeDatabase:
        def __init__(self):
            self.replacement = None

        def get_latest_house_generation(self, archive_year):
            assert archive_year == 2026
            return "acquired-generation-2026"

        def replace_transactions_for_docs(self, df, **kwargs):
            self.replacement = (df.copy(), kwargs)

    db = FakeDatabase()
    settings = SimpleNamespace(
        data=SimpleNamespace(data_dir=str(tmp_path), get_workers=lambda: 1)
    )
    monkeypatch.setattr(reparse_all, "HouseTransactionSource", FakeSource)
    monkeypatch.setattr(
        reparse_all,
        "_filter_existing_pdfs",
        lambda _ptrs, _pdf_dir: ([success_path, zero_path], pd.DataFrame()),
    )
    monkeypatch.setattr(
        reparse_all,
        "_build_member_lookup",
        lambda _docs: {
            "success": {
                "First": "Cleo",
                "Last": "Fields",
                "FilingDate": pd.Timestamp("2026-07-16"),
            },
            "ambiguous-zero": {
                "First": "Zero",
                "Last": "Candidate",
                "FilingDate": pd.Timestamp("2026-07-16"),
            },
        },
    )
    monkeypatch.setattr(reparse_all, "Pool", FakePool)
    monkeypatch.setattr(reparse_all, "preserve_existing_fields", lambda df, _db: df)
    monkeypatch.setattr(
        reparse_all,
        "_persisted_house_generation_counts",
        lambda _db, doc_ids, generation: {
            "success": 1,
            "ambiguous-zero": 4,
        },
    )

    persisted = reparse_all.parse_year(2026, db, settings)

    assert persisted == 1
    stored_df, kwargs = db.replacement
    assert stored_df["doc_id"].tolist() == ["success"]
    assert stored_df["ingestion_generation"].tolist() == ["acquired-generation-2026"]
    assert stored_df["artifact_sha256"].tolist() == [
        hashlib.sha256(success_path.read_bytes()).hexdigest()
    ]
    assert kwargs["attempted_doc_ids"] == ["success", "ambiguous-zero"]
    assert kwargs["replacement_doc_ids"] == ["success"]
    assert kwargs["ingestion_generation"] == "acquired-generation-2026"
    parse_runs = {run["doc_id"]: run for run in kwargs["parse_runs"]}
    assert (
        parse_runs["success"]["artifact_sha256"]
        == hashlib.sha256(success_path.read_bytes()).hexdigest()
    )
    assert (
        parse_runs["ambiguous-zero"]["artifact_sha256"]
        == hashlib.sha256(zero_path.read_bytes()).hexdigest()
    )
    assert parse_runs["success"]["status"] == "success"
    assert parse_runs["ambiguous-zero"]["status"] == "zero_rows"
