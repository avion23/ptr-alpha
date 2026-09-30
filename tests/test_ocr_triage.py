import hashlib
import json
import shutil
import subprocess

import pytest

from analyzer.database import Database
from scripts import ocr_triage


@pytest.mark.parametrize(
    ("page_count", "text", "dpi", "expected"),
    [
        (2, "native transaction text " * 30, 150, ocr_triage.TEXT_MISSED),
        (2, "", 180, ocr_triage.LOW_RESOLUTION),
        (2, "", 300, ocr_triage.SCANNED_IMAGE),
    ],
)
def test_probe_and_classifier_use_poppler_signals(
    monkeypatch, tmp_path, page_count, text, dpi, expected
):
    pdf = tmp_path / "fixture.pdf"
    pdf.touch()

    def run(command, _timeout):
        if command[0] == "pdfinfo":
            output = f"Pages:          {page_count}\nPage size:      612 x 792 pts\n"
        elif command[0] == "pdftotext":
            output = text
        else:
            output = (
                "page num type width height color comp bpc enc interp object ID "
                "x-ppi y-ppi size ratio\n"
                f"1 0 image 2400 3300 rgb 3 8 jpeg no 12 0 {dpi} {dpi} 1.2M 4%\n"
                f"2 0 image 2400 3300 rgb 3 8 jpeg no 13 0 {dpi} {dpi} 1.2M 4%\n"
            )
        return subprocess.CompletedProcess(command, 0, output, "")

    monkeypatch.setattr(ocr_triage, "_run", run)
    probe = ocr_triage.probe_pdf(pdf)
    assert probe.page_count == page_count
    assert probe.text_char_count == len(text.strip())
    assert probe.dpi_estimate == dpi
    assert ocr_triage.classify_probe(probe) == expected


def test_priority_is_member_prominence_times_filing_recency(tmp_path):
    items = [
        {
            "doc_id": "active-old",
            "year": 2026,
            "pdf_path": "2026/pdfs/active-old.pdf",
            "filing_date": "2026-09-01",
            "member_prominence": 80,
        },
        {
            "doc_id": "prominent-recent",
            "year": 2026,
            "pdf_path": "2026/pdfs/prominent-recent.pdf",
            "filing_date": "2026-09-03",
            "member_prominence": 50,
        },
        {
            "doc_id": "new-low",
            "year": 2026,
            "pdf_path": "2026/pdfs/new-low.pdf",
            "filing_date": "2026-09-02",
            "member_prominence": 5,
        },
    ]

    ranked = ocr_triage.prioritize_queue(items)

    assert [item["doc_id"] for item in ranked] == [
        "prominent-recent",
        "active-old",
        "new-low",
    ]
    assert [item["priority_score"] for item in ranked] == [150, 80, 10]
    queue_path = tmp_path / "queue.json"
    ocr_triage._queue_work_items(ranked, queue_path)
    assert json.loads(queue_path.read_text()) == [
        ["prominent-recent", 2026, "2026/pdfs/prominent-recent.pdf"],
        ["active-old", 2026, "2026/pdfs/active-old.pdf"],
        ["new-low", 2026, "2026/pdfs/new-low.pdf"],
    ]


def _seed_unresolved_house_db(path):
    db = Database(path)
    db.conn.execute(
        "INSERT INTO house_archive_generations "
        "(archive_year, generation_id, parse_status) VALUES (2026, 'g1', 'incomplete')"
    )
    db.conn.execute(
        "INSERT INTO house_generation_metadata "
        "(archive_year, generation_id, doc_id, first_name, last_name, filing_date, filing_type) "
        "VALUES (2026, 'g1', '42', 'Ada', 'Lovelace', '2026-09-03', 'P')"
    )
    db.conn.execute(
        "INSERT INTO house_pdf_artifacts "
        "(archive_year, generation_id, doc_id, artifact_sha256) "
        "VALUES (2026, 'g1', '42', 'fixture-sha')"
    )
    db.conn.execute(
        "INSERT INTO transactions (doc_id, member, source, chamber) "
        "VALUES ('prior', 'Ada Lovelace', 'house_pdf', 'House')"
    )
    db.close()


def test_dry_run_reads_temp_database_without_writing_it(monkeypatch, tmp_path):
    seed = tmp_path / "seed.duckdb"
    _seed_unresolved_house_db(seed)
    data_dir = tmp_path / "data-copy"
    data_dir.mkdir()
    db_path = data_dir / "congress.duckdb"
    shutil.copy2(seed, db_path)
    pdf = data_dir / "2026" / "pdfs" / "42.pdf"
    pdf.parent.mkdir(parents=True)
    pdf.write_bytes(b"fixture PDF bytes")
    queue_path = tmp_path / "queue.json"
    before = hashlib.sha256(db_path.read_bytes()).digest()

    monkeypatch.setattr(
        ocr_triage,
        "probe_pdf",
        lambda _path: ocr_triage.PdfProbe(1, 0, 180, 1),
    )
    monkeypatch.setattr(
        ocr_triage,
        "_local_reparse",
        lambda _item: (False, {"status": "unresolved"}, []),
    )
    report = ocr_triage.run_triage(data_dir, queue_path=queue_path)

    assert report["unresolved"] == 1
    assert report["classifications"][ocr_triage.LOW_RESOLUTION] == 1
    assert report["attempted"] == 1
    assert hashlib.sha256(db_path.read_bytes()).digest() == before
    assert json.loads(queue_path.read_text()) == []


def test_local_reparse_preserves_amount_owner_and_row_identity(monkeypatch):
    from scripts import ocr_local_sweep

    recovered_rows = [
        {
            "asset_description": "Apple Inc. (AAPL)",
            "owner_code": "SP",
            "transaction_type": "Purchase",
            "transaction_date_raw": "09/01/2026",
            "notification_date_raw": "09/03/2026",
            "amount_raw": "B",
            "amount_midpoint": 32500,
            "source_row_id": "42:page:1:row:1",
        },
        {
            "asset_description": "Microsoft Corp. (MSFT)",
            "owner_code": "S",
            "transaction_type": "Sale",
            "transaction_date_raw": "09/02/2026",
            "notification_date_raw": "09/03/2026",
            "amount_raw": "C",
            "amount_midpoint": 75000,
            "source_row_id": "42:page:1:row:2",
        },
    ]
    monkeypatch.setattr(
        ocr_local_sweep,
        "process_document",
        lambda *_args: {
            "status": "resolved",
            "rows": recovered_rows,
            "artifact_sha256": "fixture-sha",
        },
    )

    succeeded, _result, transactions = ocr_triage._local_reparse(
        {"doc_id": "42", "year": 2026, "pdf_path": "42.pdf", "member": "Ada Lovelace"}
    )

    assert succeeded is True
    assert [
        (row["amount_raw"], row["amount_midpoint"], row["owner_code"], row["source_row_id"])
        for row in transactions
    ] == [
        ("B", 32500, "SP", "42:page:1:row:1"),
        ("C", 75000, "S", "42:page:1:row:2"),
    ]


def test_local_reparse_fails_when_any_recovered_row_is_dropped(monkeypatch):
    from scripts import ocr_local_sweep

    recovered_rows = [
        {"asset_description": "Apple", "source_row_id": "42:r1"},
        {"asset_description": "Microsoft", "source_row_id": "42:r2"},
    ]
    monkeypatch.setattr(
        ocr_local_sweep,
        "process_document",
        lambda *_args: {"status": "resolved", "rows": recovered_rows},
    )
    parsed_rows = iter(([{"transaction_type": "Purchase"}], []))
    monkeypatch.setattr(
        "analyzer.parsing.rows.parse_pdf_table",
        lambda _table: next(parsed_rows),
    )

    succeeded, _result, transactions = ocr_triage._local_reparse(
        {"doc_id": "42", "year": 2026, "pdf_path": "42.pdf", "member": "Ada Lovelace"}
    )

    assert len(transactions) == 1
    assert succeeded is False
