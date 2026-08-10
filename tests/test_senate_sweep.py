"""Offline tests for the staged live Senate eFD sweep driver.

Covers artifact serialization, quarantine transaction rebuild, canary
verification and manifest SHA accounting without any network access.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date

import pandas as pd
import pytest

from analyzer.senate_efd import (
    SenateEFDSource,
    SenateReportFetchResult,
)
from analyzer.models import ReportOutcome, TickerOrigin
from scripts.senate_sweep import (
    GENERATION,
    inventory_records,
    rebuild_normalized_transactions,
    sha256_file,
    verify_canaries,
    write_inventory_jsonl,
    write_manifest,
    write_transactions_jsonl,
)

KATIE_REPORT_ID = "37900303-65bf-467d-962b-76555d510b28"
KATIE_REPORT_PATH = f"/search/view/ptr/{KATIE_REPORT_ID}/"
SCOTT_REPORT_ID = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
SCOTT_REPORT_PATH = f"/search/view/ptr/{SCOTT_REPORT_ID}/"


def _inventory_row(**overrides):
    row = {
        "chamber": "senate",
        "source_record_id": KATIE_REPORT_ID,
        "report_path": KATIE_REPORT_PATH,
        "member": "Katie Britt",
        "official_filing_date": pd.Timestamp("2026-01-29"),
        "outcome": ReportOutcome.PARSED.value,
        "artifact_sha256": "a" * 64,
        "landing_sha256": "b" * 64,
        "paper_artifact_sha256": None,
        "paper_artifact_url": None,
        "error_message": None,
        "raw_row_count": 1,
        "accepted_row_count": 1,
        "rejected_row_count": 0,
        "ingestion_generation": GENERATION,
    }
    row.update(overrides)
    return row


def _raw_trade(**overrides):
    trade = {
        "doc_id": KATIE_REPORT_ID,
        "source_record_id": KATIE_REPORT_ID,
        "source_row_id": "official:1",
        "source_report_path": KATIE_REPORT_PATH,
        "senator": "Katie Britt",
        "filed_date": pd.Timestamp("2026-01-29"),
        "official_filing_date": pd.Timestamp("2026-01-29"),
        "available_date": pd.Timestamp("2026-01-29"),
        "notification_date": "01/29/2026",
        "amends_source_record_id": None,
        "ticker": "JPM",
        "ticker_raw": "JPM",
        "ticker_candidate": None,
        "ticker_origin": TickerOrigin.OFFICIAL.value,
        "transaction_date": "01/28/2026",
        "type": "Sale (Full)",
        "transaction_subtype_raw": "Sale (Full)",
        "owner": "SP",
        "owner_raw": "Spouse",
        "amount_range": "$1,001 - $15,000",
        "amount_range_raw": "$1,001 - $15,000",
        "asset_name": "JPMorgan Chase & Co. Common Stock",
        "asset_type": "Stock",
        "ingestion_generation": GENERATION,
        "artifact_sha256": "a" * 64,
    }
    trade.update(overrides)
    return trade


def test_inventory_records_serialize_timestamps_as_iso_strings():
    records = inventory_records([_inventory_row()])
    assert records[0]["official_filing_date"] == "2026-01-29T00:00:00"
    assert records[0]["outcome"] == "parsed"
    assert records[0]["accepted_row_count"] == 1


def test_write_inventory_jsonl_roundtrip(tmp_path):
    rows = [
        _inventory_row(),
        _inventory_row(
            source_record_id=SCOTT_REPORT_ID,
            report_path=SCOTT_REPORT_PATH,
            member="Rick Scott",
            outcome=ReportOutcome.PAPER_ONLY.value,
            accepted_row_count=0,
            paper_artifact_sha256="c" * 64,
        ),
    ]
    target = tmp_path / "report_inventory.jsonl"
    write_inventory_jsonl(rows, target)
    loaded = [json.loads(line) for line in target.read_text().splitlines()]
    assert len(loaded) == 2
    assert loaded[0]["member"] == "Katie Britt"
    assert loaded[1]["outcome"] == "paper_only"


def test_write_transactions_jsonl_roundtrip(tmp_path):
    source = object.__new__(SenateEFDSource)
    source.ingestion_generation = GENERATION
    frame = source._normalize([_raw_trade()])
    assert len(frame) == 1

    target = tmp_path / "transactions.jsonl"
    write_transactions_jsonl(frame, target)
    loaded = [json.loads(line) for line in target.read_text().splitlines()]
    assert len(loaded) == 1
    row = loaded[0]
    assert row["chamber"] == "senate"
    assert row["member_key"] == "KATIE BRITT"
    assert row["ticker"] == "JPM"
    assert row["transaction_type"] == "Sale"
    assert row["transaction_date"] == "2026-01-28T00:00:00"
    assert row["official_filing_date"] == "2026-01-29T00:00:00"
    assert row["notification_date"] == "2026-01-29T00:00:00"
    assert row["artifact_sha256"] == "a" * 64
    assert set(row) == set(
        [
            "doc_id",
            "chamber",
            "source_record_id",
            "source_row_id",
            "source_report_path",
            "member",
            "member_key",
            "chamber_member_key",
            "ticker",
            "raw_ticker",
            "ticker_candidate",
            "ticker_origin",
            "transaction_date",
            "disclosure_date",
            "official_filing_date",
            "available_date",
            "notification_date",
            "transaction_type",
            "raw_transaction_subtype",
            "owner_code",
            "raw_owner",
            "amount_raw",
            "amount_midpoint",
            "instrument_type",
            "raw_asset_class",
            "strike_price",
            "expiry_date",
            "asset_description",
            "raw_asset_description",
            "amends_source_record_id",
            "ingestion_generation",
            "artifact_sha256",
        ]
    )


def test_rebuild_normalized_transactions_uses_accepted_normalization():
    source = object.__new__(SenateEFDSource)
    source.ingestion_generation = GENERATION
    source.staged_results = {
        KATIE_REPORT_PATH: SenateReportFetchResult(
            outcome=ReportOutcome.PARSED,
            transactions=tuple([_raw_trade()]),
            landing_sha256="b" * 64,
        )
    }
    inventory = [_inventory_row()]
    rebuilt = rebuild_normalized_transactions(source, inventory)
    assert len(rebuilt) == 1
    row = rebuilt.iloc[0]
    assert row["chamber_member_key"] == "senate:KATIE BRITT"
    assert row["ticker"] == "JPM"
    assert row["transaction_type"] == "Sale"
    assert row["ingestion_generation"] == GENERATION


def test_rebuild_normalized_transactions_skips_non_parsed_reports():
    source = object.__new__(SenateEFDSource)
    source.ingestion_generation = GENERATION
    source.staged_results = {}
    inventory = [
        _inventory_row(outcome=ReportOutcome.UNAVAILABLE.value, accepted_row_count=0),
        _inventory_row(
            source_record_id=SCOTT_REPORT_ID,
            report_path=SCOTT_REPORT_PATH,
            member="Rick Scott",
            outcome=ReportOutcome.FAILED.value,
            accepted_row_count=0,
        ),
    ]
    rebuilt = rebuild_normalized_transactions(source, inventory)
    assert len(rebuilt) == 0


def test_verify_canaries_passes_for_britt_one_and_scott_twelve():
    inventory = [
        _inventory_row(accepted_row_count=1),
        _inventory_row(
            source_record_id=SCOTT_REPORT_ID,
            report_path=SCOTT_REPORT_PATH,
            member="Rick Scott",
            accepted_row_count=12,
        ),
    ]
    result = verify_canaries(inventory)
    assert result["passed"] is True
    assert result["failures"] == []
    assert result["results"]["KATIE BRITT"]["accepted_row_counts"] == [1]
    assert result["results"]["RICK SCOTT"]["accepted_row_counts"] == [12]


def test_verify_canaries_fails_on_wrong_count():
    inventory = [
        _inventory_row(accepted_row_count=1),
        _inventory_row(
            source_record_id=SCOTT_REPORT_ID,
            report_path=SCOTT_REPORT_PATH,
            member="Rick Scott",
            accepted_row_count=11,
        ),
    ]
    result = verify_canaries(inventory)
    assert result["passed"] is False
    assert any("RICK SCOTT" in failure for failure in result["failures"])


def test_verify_canaries_fails_on_non_parsed_canary():
    inventory = [
        _inventory_row(accepted_row_count=1),
        _inventory_row(
            source_record_id=SCOTT_REPORT_ID,
            report_path=SCOTT_REPORT_PATH,
            member="Rick Scott",
            outcome=ReportOutcome.FAILED.value,
            accepted_row_count=0,
        ),
    ]
    result = verify_canaries(inventory)
    assert result["passed"] is False
    assert any("RICK SCOTT" in failure for failure in result["failures"])


def test_write_manifest_records_artifact_shas(tmp_path):
    inventory = [_inventory_row()]
    pd.DataFrame(
        columns=[
            "doc_id",
            "chamber",
            "source_record_id",
            "source_row_id",
            "source_report_path",
            "member",
            "member_key",
            "chamber_member_key",
            "ticker",
            "raw_ticker",
            "ticker_candidate",
            "ticker_origin",
            "transaction_date",
            "disclosure_date",
            "official_filing_date",
            "available_date",
            "notification_date",
            "transaction_type",
            "raw_transaction_subtype",
            "owner_code",
            "raw_owner",
            "amount_raw",
            "amount_midpoint",
            "instrument_type",
            "raw_asset_class",
            "strike_price",
            "expiry_date",
            "asset_description",
            "raw_asset_description",
            "amends_source_record_id",
            "ingestion_generation",
            "artifact_sha256",
        ]
    )
    gen_dir = tmp_path / "gen"
    gen_dir.mkdir()
    write_inventory_jsonl(inventory, gen_dir / "report_inventory.jsonl")
    tx_file = gen_dir / "transactions.jsonl"
    tx_file.write_text("")

    manifest_path = write_manifest(
        gen_dir,
        generation=GENERATION,
        start_date=pd.Timestamp("2025-08-09").date(),
        end_date=pd.Timestamp("2026-08-09").date(),
        inventory=inventory,
        summary={
            "found": 1,
            "parsed": 1,
            "paper_only": 0,
            "unavailable": 0,
            "failed": 0,
        },
        transactions_file=tx_file,
        papers=[],
        canaries={"results": {}, "failures": [], "passed": True},
        quarantine=None,
    )
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "complete"
    expected_sha, expected_size = sha256_file(gen_dir / "report_inventory.jsonl")
    assert manifest["artifacts"]["report_inventory.jsonl"]["sha256"] == expected_sha
    assert manifest["artifacts"]["report_inventory.jsonl"]["bytes"] == expected_size
    assert manifest["outcome_counts"]["parsed"] == 1
    assert manifest["window"]["start_date"] == "2025-08-09"


def test_write_manifest_quarantined_status(tmp_path):
    gen_dir = tmp_path / "gen2"
    gen_dir.mkdir()
    inventory = [
        _inventory_row(outcome=ReportOutcome.UNAVAILABLE.value, accepted_row_count=0)
    ]
    write_inventory_jsonl(inventory, gen_dir / "report_inventory.jsonl")
    (gen_dir / "transactions.jsonl").write_text("")
    quarantine = {
        "error": "Senate refresh incomplete",
        "missing": {"unavailable": 1, "failed": 0},
    }
    manifest_path = write_manifest(
        gen_dir,
        generation=GENERATION,
        start_date=pd.Timestamp("2025-08-09").date(),
        end_date=pd.Timestamp("2026-08-09").date(),
        inventory=inventory,
        summary={
            "found": 1,
            "parsed": 0,
            "paper_only": 0,
            "unavailable": 1,
            "failed": 0,
        },
        transactions_file=gen_dir / "transactions.jsonl",
        papers=[],
        canaries={"results": {}, "failures": [], "passed": True},
        quarantine=quarantine,
    )
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "quarantined"
    assert manifest["quarantine"]["missing"]["unavailable"] == 1


def test_verify_canaries_passes_with_multiple_reports_per_member():
    inventory = [
        _inventory_row(accepted_row_count=22),  # Britt non-canary filing
        _inventory_row(accepted_row_count=1),  # Britt canary filing
        _inventory_row(
            source_record_id=SCOTT_REPORT_ID,
            report_path=SCOTT_REPORT_PATH,
            member="Rick Scott",
            accepted_row_count=17,
        ),
        _inventory_row(
            source_record_id="dddddddd-eeee-ffff-0000-111111111111",
            report_path="/search/view/ptr/dddddddd-eeee-ffff-0000-111111111111/",
            member="Rick Scott",
            accepted_row_count=12,  # Scott canary filing
        ),
    ]
    result = verify_canaries(inventory)
    assert result["passed"] is True
    assert (
        result["results"]["KATIE BRITT"]["canary_matches"][0]["accepted_row_count"] == 1
    )
    scott_matches = result["results"]["RICK SCOTT"]["canary_matches"]
    assert [m["accepted_row_count"] for m in scott_matches] == [12]


def test_stage_paper_artifacts_writes_and_verifies_bytes(tmp_path):

    from scripts.senate_sweep import stage_paper_artifacts

    pdf_bytes = b"%PDF-1.4\nfake paper filing\n%%EOF"

    class FakeSource:
        paper_bytes = {"https://efdsearch.senate.gov/media/fake.pdf": pdf_bytes}

        def _request_with_retry(self, method, url, **kwargs):  # pragma: no cover
            raise AssertionError("should use retained bytes")

    inventory = [
        _inventory_row(
            outcome=ReportOutcome.PAPER_ONLY.value,
            accepted_row_count=0,
            paper_artifact_sha256=hashlib.sha256(pdf_bytes).hexdigest(),
            paper_artifact_url="https://efdsearch.senate.gov/media/fake.pdf",
        )
    ]
    staged = stage_paper_artifacts(FakeSource(), inventory, tmp_path / "papers")
    assert len(staged) == 1
    assert staged[0]["sha256"] == hashlib.sha256(pdf_bytes).hexdigest()
    target = tmp_path / "papers" / f"{KATIE_REPORT_ID}.pdf"
    assert target.read_bytes() == pdf_bytes


def test_stage_paper_artifacts_fails_closed_on_sha_mismatch(tmp_path):

    from scripts.senate_sweep import SenateSweepError, stage_paper_artifacts

    pdf_bytes = b"%PDF-1.4\nfake\n%%EOF"

    class FakeSource:
        paper_bytes = {"https://efdsearch.senate.gov/media/fake.pdf": pdf_bytes}

    inventory = [
        _inventory_row(
            outcome=ReportOutcome.PAPER_ONLY.value,
            accepted_row_count=0,
            paper_artifact_sha256="0" * 64,
            paper_artifact_url="https://efdsearch.senate.gov/media/fake.pdf",
        )
    ]
    with pytest.raises(SenateSweepError, match="SHA mismatch"):
        stage_paper_artifacts(FakeSource(), inventory, tmp_path / "papers")


def test_verify_canaries_any_requires_parsed_report_per_member():
    from scripts.senate_sweep import _CANARY_EXPECTED_ANY

    inventory = [
        _inventory_row(accepted_row_count=22),
        _inventory_row(
            source_record_id=SCOTT_REPORT_ID,
            report_path=SCOTT_REPORT_PATH,
            member="Rick Scott",
            accepted_row_count=17,
        ),
    ]
    result = verify_canaries(inventory, expected=_CANARY_EXPECTED_ANY)
    assert result["passed"] is True
    assert result["results"]["KATIE BRITT"]["parsed_reports"] == 1
    assert result["results"]["RICK SCOTT"]["parsed_reports"] == 1


def test_verify_canaries_any_fails_when_member_missing():
    from scripts.senate_sweep import _CANARY_EXPECTED_ANY

    inventory = [_inventory_row(accepted_row_count=1)]
    result = verify_canaries(inventory, expected=_CANARY_EXPECTED_ANY)
    assert result["passed"] is False
    assert any("RICK SCOTT" in failure for failure in result["failures"])


def _synthetic_chunk(tmp_path, name, rows, tx_rows, window, quarantine=None):
    chunk = tmp_path / name
    chunk.mkdir(parents=True)
    from scripts.senate_sweep import write_inventory_jsonl, write_manifest

    write_inventory_jsonl(rows, chunk / "report_inventory.jsonl")
    tx_file = chunk / "transactions.jsonl"
    tx_file.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in tx_rows))
    write_manifest(
        chunk,
        generation=name,
        start_date=window[0],
        end_date=window[1],
        inventory=rows,
        summary={
            "found": len(rows),
            "parsed": sum(1 for r in rows if r["outcome"] == "parsed"),
            "paper_only": 0,
            "unavailable": 0,
            "failed": 0,
        },
        transactions_file=tx_file,
        papers=[],
        canaries={"results": {}, "failures": [], "passed": True},
        quarantine=quarantine,
    )
    return chunk


def test_merge_window_combines_chunks_and_verifies_canaries(tmp_path):
    from scripts.senate_sweep import merge_window

    d = tmp_path / "window"
    d.mkdir()
    row1 = _inventory_row()  # Britt parsed, 1 tx
    row2 = _inventory_row(
        source_record_id=SCOTT_REPORT_ID,
        report_path=SCOTT_REPORT_PATH,
        member="Rick Scott",
        accepted_row_count=12,
        official_filing_date=pd.Timestamp("2024-03-15"),
    )
    row3 = _inventory_row(
        source_record_id="ffffffff-1111-2222-3333-444444444444",
        report_path="/search/view/ptr/ffffffff-1111-2222-3333-444444444444/",
        member="Thomas Tuberville",
        accepted_row_count=3,
        official_filing_date=pd.Timestamp("2023-06-01"),
    )
    _synthetic_chunk(
        d, "chunk-2023", [row3], [], (date(2023, 1, 1), date(2023, 12, 31))
    )
    _synthetic_chunk(
        d,
        "chunk-2024",
        [row1, row2],
        [],
        (date(2024, 1, 1), date(2024, 12, 31)),
    )
    exit_code = merge_window(d)
    assert exit_code == 0
    manifest = json.loads((d / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["summary"]["found"] == 3
    assert manifest["summary"]["parsed"] == 3
    assert manifest["window"] == {
        "start_date": "2023-01-01",
        "end_date": "2024-12-31",
    }
    assert manifest["canaries"]["passed"] is True
    merged_inv = [
        json.loads(line)
        for line in (d / "report_inventory.jsonl").read_text().splitlines()
    ]
    assert len(merged_inv) == 3
    assert "chunk-2023" in json.loads((d / "chunks.json").read_text())["chunks"]


def test_merge_window_dedupes_and_quarantines_when_chunk_quarantined(tmp_path):
    from scripts.senate_sweep import merge_window

    d = tmp_path / "window2"
    d.mkdir()
    dup = _inventory_row()
    dup["official_filing_date"] = pd.Timestamp("2024-01-05")
    _synthetic_chunk(
        d,
        "chunk-2024a",
        [dup],
        [],
        (date(2024, 1, 1), date(2024, 12, 31)),
    )
    _synthetic_chunk(
        d,
        "chunk-2024b",
        [dict(dup)],  # same report_path -> dedupe
        [],
        (date(2024, 1, 1), date(2024, 12, 31)),
        quarantine={
            "error": "Senate refresh incomplete",
            "missing": {"unavailable": 1, "failed": 0},
        },
    )
    exit_code = merge_window(d)
    assert exit_code == 2
    manifest = json.loads((d / "manifest.json").read_text())
    assert manifest["status"] == "quarantined"
    assert manifest["summary"]["found"] == 1  # deduped
    assert manifest["quarantine"]["missing"]["unavailable"] == 1
