import io
import json
import zipfile

import pandas as pd

from scripts import audit_house_coverage


def _canned_archive(year: int) -> bytes:
    """Build an in-memory {year}FD.ZIP matching the modern House schema."""
    rows = [
        ["", "Foster", "Bill", "", "P", "IL11", str(year), "3/1/2026", "20010001"],
        ["", "Foster", "Bill", "", "A", "IL11", str(year), "4/1/2026", "20010002"],
        ["", "Lee", "Summer", "", "P", "CA12", str(year), "5/1/2026", "20010003"],
        ["", "Garcia", "Robert", "", "X", "CA27", str(year), "6/1/2026", "20010004"],
    ]
    header = (
        "Prefix\tLast\tFirst\tSuffix\tFilingType\tStateDst\tYear\tFilingDate\tDocID"
    )
    content = "\n".join([header, *["\t".join(row) for row in rows]])
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(f"{year}FD.txt", content)
    return buffer.getvalue()


def test_build_year_coverage_reconciles_expected_local_missing_orphans():
    metadata = pd.DataFrame(
        {
            "DocID": ["20010001", "20010002", "20010003"],
            "First": ["Bill", "Bill", "Summer"],
            "Last": ["Foster", "Foster", "Lee"],
            "FilingDate": pd.to_datetime(["2026-03-01", "2026-04-01", "2026-05-01"]),
            "FilingType": ["P", "A", "P"],
        }
    )
    local_ids = {"20010001", "99999999"}

    row = audit_house_coverage.build_year_coverage(metadata, local_ids, 2026)

    assert row["year"] == 2026
    assert row["metadata_records"] == 3
    assert row["ptr_expected"] == 2
    assert row["amendments"] == 1
    assert row["local_pdfs"] == 2
    assert row["covered"] == 1
    assert row["coverage_ratio"] == 0.5
    assert row["missing"] == 1
    assert row["missing_doc_ids"] == ["20010003"]
    assert row["orphans"] == 1
    assert row["orphan_doc_ids"] == ["99999999"]
    assert row["filing_type_breakdown"] == {"P": 2, "A": 1}


def test_scan_local_inventory_reads_year_pdf_dirs(tmp_path):
    for year, doc_ids in {
        "2021": {"111", "222"},
        "2026": {"333"},
    }.items():
        pdf_dir = tmp_path / year / "pdfs"
        pdf_dir.mkdir(parents=True)
        for doc_id in doc_ids:
            (pdf_dir / f"{doc_id}.pdf").write_bytes(b"%PDF")
    (tmp_path / "not-a-year" / "pdfs").mkdir(parents=True)
    (tmp_path / "not-a-year" / "pdfs" / "x.pdf").write_bytes(b"%PDF")

    inventory = audit_house_coverage.scan_local_inventory(tmp_path)

    assert inventory == {2021: {"111", "222"}, 2026: {"333"}}


def test_main_end_to_end_writes_coverage_json(tmp_path, monkeypatch):
    local_dir = tmp_path / "inventory"
    pdf_dir = local_dir / "2021" / "pdfs"
    pdf_dir.mkdir(parents=True)
    (pdf_dir / "20010001.pdf").write_bytes(b"%PDF")

    def fake_fetch(year, _session, _timeout):
        return 200, _canned_archive(year), "Mon, 01 Jan 2026 00:00:00 GMT"

    monkeypatch.setattr(audit_house_coverage, "fetch_metadata_archive", fake_fetch)
    output = tmp_path / "out" / "coverage.json"

    exit_code = audit_house_coverage.main(
        [
            str(local_dir),
            str(output),
            "--first-year",
            "2021",
            "--last-year",
            "2022",
        ]
    )

    assert exit_code == 0
    payload = json.loads(output.read_text())
    assert payload["audit"] == "house_metadata_coverage"
    assert payload["excluded_legacy"]["years"] == list(range(2008, 2015))
    assert payload["summary"]["years_audited"] == 2
    by_year = {row["year"]: row for row in payload["years"]}
    assert all(row["error"] is None for row in payload["years"])
    assert by_year[2021]["ptr_expected"] == 2
    assert by_year[2021]["local_pdfs"] == 1
    assert by_year[2021]["missing_doc_ids"] == ["20010003"]
    assert by_year[2021]["orphan_doc_ids"] == []
    assert by_year[2022]["local_pdfs"] == 0
    assert by_year[2022]["missing"] == 2


def test_main_fails_closed_on_archive_error(tmp_path, monkeypatch):
    def failing_fetch(year, _session, _timeout):
        raise RuntimeError(f"network down for {year}")

    monkeypatch.setattr(audit_house_coverage, "fetch_metadata_archive", failing_fetch)
    output = tmp_path / "coverage.json"

    exit_code = audit_house_coverage.main(
        [str(tmp_path / "missing-local"), str(output)]
    )

    assert exit_code == 1
    payload = json.loads(output.read_text())
    assert payload["summary"]["years_failed"] == 12
    assert all("network down" in row["error"] for row in payload["years"])
