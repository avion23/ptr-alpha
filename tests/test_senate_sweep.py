"""Offline tests for the staged live Senate eFD sweep driver.

Covers artifact serialization, quarantine transaction rebuild, canary
verification and manifest SHA accounting without any network access.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timezone
from functools import partial
from types import SimpleNamespace

import pandas as pd
import pytest

from analyzer.senate_efd import (
    SenateEFDSource,
    SenateReportFetchResult,
)
from analyzer.models import ReportOutcome
from scripts.senate_sweep import (
    GENERATION,
    inventory_records,
    rebuild_normalized_transactions,
    run_sweep,
    sha256_file,
    verify_canaries,
    write_inventory_jsonl,
    write_manifest,
    write_transactions_jsonl,
)
from .conftest import make_raw_trade

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


_raw_trade = partial(make_raw_trade, GENERATION)


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


def _install_fake_sweep_source(monkeypatch, *, ticker="JPM", canaries_pass=True):
    from scripts import senate_sweep

    instances = []

    class FakeSource:
        def __init__(self, *, ingestion_generation, **_kwargs):
            instances.append(ingestion_generation)
            self.ingestion_generation = ingestion_generation
            scott_count = 12 if canaries_pass else 11
            self.report_inventory = [
                _inventory_row(ingestion_generation=ingestion_generation),
                _inventory_row(
                    source_record_id=SCOTT_REPORT_ID,
                    report_path=SCOTT_REPORT_PATH,
                    member="Rick Scott",
                    raw_row_count=scott_count,
                    accepted_row_count=scott_count,
                    ingestion_generation=ingestion_generation,
                ),
            ]
            self.paper_bytes = {}
            self.last_refresh_summary = SimpleNamespace(
                found=2, parsed=2, paper_only=0, unavailable=0, failed=0
            )
            self.staged_results = {}

        def fetch_all_trades(self, *_args):
            accepted = object.__new__(SenateEFDSource)
            accepted.ingestion_generation = self.ingestion_generation
            trades = [make_raw_trade(self.ingestion_generation, ticker=ticker)]
            scott_count = 12 if canaries_pass else 11
            trades.extend(
                make_raw_trade(
                    self.ingestion_generation,
                    ticker=ticker,
                    doc_id=SCOTT_REPORT_ID,
                    source_record_id=SCOTT_REPORT_ID,
                    source_row_id=f"official:{row_id}",
                    source_report_path=SCOTT_REPORT_PATH,
                    senator="Rick Scott",
                )
                for row_id in range(1, scott_count + 1)
            )
            return accepted._normalize(
                trades
            )

        def close(self):
            pass

    monkeypatch.setattr(senate_sweep, "_StagingSenateSource", FakeSource)
    return instances


def _bundle_bytes(bundle):
    return {
        path.relative_to(bundle).as_posix(): path.read_bytes()
        for path in bundle.rglob("*")
        if path.is_file()
    }


def test_sweep_publishes_fresh_immutable_bundle_and_consumer_accepts_hashes(
    tmp_path, monkeypatch, capsys
):
    from scripts import rebuild_staged, senate_sweep
    from scripts.senate_sweep import SenateSweepError

    initial_calls = _install_fake_sweep_source(monkeypatch)
    root = tmp_path / "senate"
    old_id = "gen-old"
    assert run_sweep(old_id, root, date(2026, 1, 1), date(2026, 1, 31), "exact") == 0
    assert initial_calls == [old_id]
    old_bundle = root / old_id
    original_bytes = _bundle_bytes(old_bundle)
    manifest = json.loads((old_bundle / "manifest.json").read_text())
    assert manifest["generation"] == old_id
    assert ".senate-sweep-" not in json.dumps(manifest)
    transaction = json.loads(
        (old_bundle / "transactions.jsonl").read_text().splitlines()[0]
    )
    assert transaction["ingestion_generation"] == old_id
    inventory = json.loads(
        (old_bundle / "report_inventory.jsonl").read_text().splitlines()[0]
    )
    assert inventory["ingestion_generation"] == old_id
    verification = rebuild_staged._verify_artifact_files(manifest, old_bundle)
    assert verification["mismatches"] == []
    assert verification["files_checked"] == 2

    class FakeDatabase:
        def persist_source_refresh(self, **kwargs):
            self.kwargs = kwargs
            return len(kwargs["transactions"])

    db = FakeDatabase()
    ingested = rebuild_staged._ingest_senate(old_bundle, manifest, db)
    assert ingested["inserted_transactions"] == 13
    assert db.kwargs["ingestion_generation"] == old_id

    # A changed producer cannot rebind the completed ID; refusal occurs before
    # the producer is instantiated, leaving every prior byte and receipt intact.
    changed_calls = _install_fake_sweep_source(monkeypatch, ticker="ABC")
    with pytest.raises(SenateSweepError, match="already exists.*fresh --generation"):
        run_sweep(old_id, root, date(2026, 1, 1), date(2026, 1, 31), "none")
    assert changed_calls == []
    assert _bundle_bytes(old_bundle) == original_bytes
    assert senate_sweep.main(
        [
            "--generation",
            old_id,
            "--staging-root",
            str(root),
            "--start",
            "2026-01-01",
            "--end",
            "2026-01-31",
        ]
    ) == 1
    assert "choose a fresh --generation" in capsys.readouterr().err

    # A staging fault in a separate attempt also cannot damage the old bundle.
    monkeypatch.setattr(
        senate_sweep,
        "stage_paper_artifacts",
        lambda *_args: (_ for _ in ()).throw(OSError("injected staging failure")),
    )
    with pytest.raises(OSError, match="injected staging failure"):
        run_sweep("gen-failed", root, date(2026, 1, 1), date(2026, 1, 31), "exact")
    assert not (root / "gen-failed").exists()
    assert _bundle_bytes(old_bundle) == original_bytes

    # Changed content can be published and consumed under a fresh attempt ID.
    monkeypatch.undo()
    fresh_calls = _install_fake_sweep_source(monkeypatch, ticker="ABC")
    new_id = "gen-new"
    assert run_sweep(new_id, root, date(2026, 1, 1), date(2026, 1, 31), "exact") == 0
    assert fresh_calls == [new_id]
    new_bundle = root / new_id
    new_manifest = json.loads((new_bundle / "manifest.json").read_text())
    assert new_manifest["generation"] == new_id
    assert (
        json.loads(
            (new_bundle / "transactions.jsonl").read_text().splitlines()[0]
        )["ticker"]
        == "ABC"
    )
    assert (
        rebuild_staged._verify_artifact_files(new_manifest, new_bundle)["mismatches"]
        == []
    )
    assert _bundle_bytes(old_bundle) == original_bytes
    assert not list(root.glob(".senate-sweep-*"))


def test_failed_canary_bundle_never_reaches_persistence(tmp_path, monkeypatch):
    from scripts import rebuild_staged

    _install_fake_sweep_source(monkeypatch, canaries_pass=False)
    root = tmp_path / "senate"
    generation = "gen-failed-canary"
    assert run_sweep(
        generation, root, date(2026, 1, 1), date(2026, 1, 31), "exact"
    ) == 2
    bundle = root / generation
    bundle_manifest = json.loads((bundle / "manifest.json").read_text())
    assert bundle_manifest["status"] == "canary_failed"
    assert bundle_manifest["canary_expect"] == "exact"
    assert bundle_manifest["canaries"]["passed"] is False

    class FakeDatabase:
        writes = 0

        def persist_source_refresh(self, **_kwargs):
            self.writes += 1

    db = FakeDatabase()
    with pytest.raises(ValueError, match="status is not complete"):
        rebuild_staged._ingest_senate(bundle, bundle_manifest, db)
    assert db.writes == 0


def test_cli_default_generation_is_a_current_attempt_id(tmp_path, monkeypatch):
    from scripts import senate_sweep

    captured = {}

    def fake_run(generation, *_args, **_kwargs):
        captured["generation"] = generation
        return 0

    monkeypatch.setattr(senate_sweep, "run_sweep", fake_run)
    assert senate_sweep.main(
        [
            "--staging-root",
            str(tmp_path),
            "--start",
            "2026-01-01",
            "--end",
            "2026-01-02",
        ]
    ) == 0
    generation = captured["generation"]
    today = datetime.now(timezone.utc).strftime("%Y%m%dT")
    assert generation.startswith(f"gen-live-{today}")
    assert generation != GENERATION


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


def _synthetic_chunk(
    tmp_path, name, rows, tx_rows, window, quarantine=None, generation=None
):
    chunk = tmp_path / name
    chunk.mkdir(parents=True)
    from scripts.senate_sweep import write_inventory_jsonl, write_manifest

    write_inventory_jsonl(rows, chunk / "report_inventory.jsonl")
    tx_file = chunk / "transactions.jsonl"
    tx_file.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in tx_rows))
    write_manifest(
        chunk,
        generation=generation or name,
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


def test_merge_window_rejects_unsupported_multi_generation_without_mutation(tmp_path):
    from scripts.senate_sweep import SenateSweepError, merge_window

    d = tmp_path / "window-input"
    d.mkdir()
    row1 = _inventory_row(ingestion_generation="chunk-2024")
    row2 = _inventory_row(
        source_record_id=SCOTT_REPORT_ID,
        report_path=SCOTT_REPORT_PATH,
        member="Rick Scott",
        accepted_row_count=12,
        official_filing_date=pd.Timestamp("2024-03-15"),
        ingestion_generation="chunk-2024",
    )
    row3 = _inventory_row(
        source_record_id="ffffffff-1111-2222-3333-444444444444",
        report_path="/search/view/ptr/ffffffff-1111-2222-3333-444444444444/",
        member="Thomas Tuberville",
        accepted_row_count=3,
        official_filing_date=pd.Timestamp("2023-06-01"),
        ingestion_generation="chunk-2023",
    )
    paper_id = "bbbbbbbb-cccc-dddd-eeee-ffffffffffff"
    paper_bytes = b"%PDF-1.4\nsynthetic merged paper\n%%EOF"
    paper_url = "https://efdsearch.senate.gov/media/merged.pdf"
    paper_row = _inventory_row(
        source_record_id=paper_id,
        report_path=f"/search/view/ptr/{paper_id}/",
        member="Jon Ossoff",
        outcome=ReportOutcome.PAPER_ONLY.value,
        accepted_row_count=0,
        official_filing_date=pd.Timestamp("2024-06-01"),
        paper_artifact_sha256=hashlib.sha256(paper_bytes).hexdigest(),
        paper_artifact_url=paper_url,
        ingestion_generation="chunk-2024",
    )
    _synthetic_chunk(
        d, "chunk-2023", [row3], [], (date(2023, 1, 1), date(2023, 12, 31))
    )
    _synthetic_chunk(
        d,
        "chunk-2024",
        [row1, row2, paper_row],
        [
            {
                "source_record_id": KATIE_REPORT_ID,
                "source_row_id": "official:1",
                "ingestion_generation": "chunk-2024",
            }
        ],
        (date(2024, 1, 1), date(2024, 12, 31)),
    )
    source_paper = d / "chunk-2024" / "papers" / f"{paper_id}.pdf"
    source_paper.parent.mkdir()
    source_paper.write_bytes(paper_bytes)
    generation = "window-merged"
    merged = tmp_path / generation
    source_bytes = _bundle_bytes(d)
    with pytest.raises(SenateSweepError, match="unsupported multi-generation"):
        merge_window(d, generation)
    assert not merged.exists()
    assert _bundle_bytes(d) == source_bytes
    assert not list(tmp_path.glob(".senate-merge-*"))


def test_merge_window_dedupes_and_quarantines_when_chunk_quarantined(tmp_path):
    from scripts.senate_sweep import merge_window

    d = tmp_path / "window2"
    d.mkdir()
    dup = _inventory_row()
    dup["official_filing_date"] = pd.Timestamp("2024-01-05")
    dup["ingestion_generation"] = "window-quarantined"
    _synthetic_chunk(
        d,
        "chunk-2024a",
        [dup],
        [],
        (date(2024, 1, 1), date(2024, 12, 31)),
        generation="window-quarantined",
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
        generation="window-quarantined",
    )
    generation = "window-quarantined"
    exit_code = merge_window(d, generation)
    merged = tmp_path / generation
    assert exit_code == 2
    manifest = json.loads((merged / "manifest.json").read_text())
    assert manifest["status"] == "quarantined"
    assert manifest["summary"]["found"] == 1  # deduped
    assert manifest["quarantine"]["missing"]["unavailable"] == 1
    from scripts import rebuild_staged

    assert rebuild_staged._verify_artifact_files(manifest, merged)["mismatches"] == []


@pytest.mark.parametrize("duplicate_kind", ["report", "transaction"])
def test_merge_window_rejects_conflicting_duplicates_without_mutation(
    tmp_path, duplicate_kind
):
    from scripts.senate_sweep import SenateSweepError, merge_window

    root = tmp_path / "conflicts"
    root.mkdir()
    generation = "shared-generation"
    row = _inventory_row(ingestion_generation=generation)
    changed = dict(row, accepted_row_count=2)
    first_tx = {
        "source_record_id": KATIE_REPORT_ID,
        "source_row_id": "official:1",
        "ingestion_generation": generation,
        "ticker": "JPM",
    }
    second_tx = dict(first_tx, ticker="ABC")
    _synthetic_chunk(
        root,
        "chunk-a",
        [row],
        [first_tx] if duplicate_kind == "transaction" else [],
        (date(2024, 1, 1), date(2024, 12, 31)),
        generation=generation,
    )
    _synthetic_chunk(
        root,
        "chunk-b",
        [changed if duplicate_kind == "report" else dict(row)],
        [second_tx] if duplicate_kind == "transaction" else [],
        (date(2024, 1, 1), date(2024, 12, 31)),
        generation=generation,
    )
    before = _bundle_bytes(root)
    message = (
        "conflicting duplicate Senate report_path"
        if duplicate_kind == "report"
        else "conflicting duplicate Senate transaction key"
    )
    with pytest.raises(SenateSweepError, match=message):
        merge_window(root, generation)
    assert not (tmp_path / generation).exists()
    assert _bundle_bytes(root) == before
