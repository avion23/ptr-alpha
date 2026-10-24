"""Regression tests for the staged rebuild parser boundary."""

import hashlib
import importlib
import json
from datetime import date
from types import SimpleNamespace
from typing import cast

import pandas as pd
import pytest


def _senate_test_artifacts(track_dir):
    return {
        name: {
            "file": name,
            "sha256": hashlib.sha256((track_dir / name).read_bytes()).hexdigest(),
            "bytes": (track_dir / name).stat().st_size,
        }
        for name in ("report_inventory.jsonl", "transactions.jsonl")
    }


def test_rebuild_import_does_not_set_docling_environment(monkeypatch):
    from scripts import rebuild_staged

    monkeypatch.delenv("PTR_SKIP_DOCLING", raising=False)
    importlib.reload(rebuild_staged)
    assert "PTR_SKIP_DOCLING" not in rebuild_staged.os.environ


def test_house_parse_scopes_local_ocr_default_to_operation(monkeypatch, tmp_path):
    from scripts import rebuild_staged

    staging = tmp_path / "staging"
    staging.mkdir()
    rebuild_staged._save_manifest(
        staging,
        {"house": {"2026": {"parse_status": "incomplete"}}},
    )
    observed = []

    class FakeDatabase:
        def __init__(self, *_args, **_kwargs):
            pass

        def close(self):
            pass

    monkeypatch.setattr(rebuild_staged, "Database", FakeDatabase)

    def parse(_staging, _db, _year):
        observed.append(rebuild_staged.os.environ.get("PTR_SKIP_DOCLING"))
        return {"attempted": 1}

    monkeypatch.setattr(rebuild_staged, "_parse_house_year_tolerant", parse)
    monkeypatch.setattr(rebuild_staged, "_refresh_house_completion", lambda *_args: ([], 1))
    monkeypatch.setenv("PTR_SKIP_DOCLING", "0")
    rebuild_staged.house_parse(
        rebuild_staged.argparse.Namespace(staging=str(staging), years=[2026], force=False)
    )
    assert observed == ["1"]
    assert rebuild_staged.os.environ["PTR_SKIP_DOCLING"] == "0"


def test_shared_verification_phase_keeps_staged_check_contract(tmp_path):
    from analyzer.persistence.database import Database
    from scripts import rebuild_staged

    db = Database(tmp_path / "empty.duckdb")
    try:
        checks = {}
        rebuild_staged._verify_shared_database_audits(db, checks)
        assert list(checks) == [
            "house_success_run_counts_equal_persisted_rows",
            "source_report_reconciliation_consistent",
            "chronology_valid",
            "no_source_identity_duplicates",
            "source_identity_complete",
        ]
        assert all(result["passed"] for result in checks.values())
    finally:
        db.close()


@pytest.mark.parametrize("bad_row", ["invalid_date", "missing_metadata"])
def test_house_persistence_rejects_consolidation_loss_in_both_paths(tmp_path, bad_row):
    from analyzer.download import HouseTransactionSource, validated_pdf_sha256
    from scripts import rebuild_staged

    pdf = tmp_path / "loss.pdf"
    pdf.write_bytes(b"%PDF-test\n%%EOF")
    row = dict(
        ticker="AAPL", transaction_type="Purchase", transaction_date="2026-01-02",
        source_row_id="pdftotext:l1", asset_description="Apple Inc. (AAPL)",
    )
    rows = [row, dict(row, transaction_date="invalid", source_row_id="pdftotext:l2")]
    metadata = {"loss": {"First": "Test", "Last": "Member", "FilingDate": "2026-01-03"}}
    if bad_row == "missing_metadata":
        metadata = {}
    results = [(pdf, rows, ["won:pdftotext"])]
    captured = []

    class FakeDatabase:
        def count_transactions_for_docs(self, _docs):
            return {}

        def replace_transactions_for_docs(self, frame, **kwargs):
            captured.append((frame.copy(), kwargs))
            return SimpleNamespace(
                by_doc_total={}, by_doc_source={}, total_current_rows=0, total_raw_rows=0
            )

    db = FakeDatabase()
    source = object.__new__(HouseTransactionSource)
    source.db = db
    source._save_parse_results(2026, results, metadata, "generation")
    rebuild_staged._persist_house_parse_batch(
        db, year=2026, ingestion_generation="generation", member_lookup=metadata,
        artifact_hashes={"loss": validated_pdf_sha256(pdf)}, results=results,
    )
    assert len(captured) == 2
    for frame, kwargs in captured:
        assert frame.empty
        assert kwargs["attempted_doc_ids"] == ["loss"]
        assert kwargs["replacement_doc_ids"] == []
        run, = kwargs["parse_runs"]
        assert run["status"] == "error"
        assert run["raw_row_count"] == 2
        assert run["transaction_count"] == 0
        assert run["error_message"] == "rows lost during consolidation"
        assert run["artifact_sha256"] == validated_pdf_sha256(pdf)
        assert run["ingestion_generation"] == "generation"
    assert captured[0][1] == captured[1][1]


def test_house_parse_source_recognizes_all_validated_llm_backends():
    from scripts import rebuild_staged

    assert rebuild_staged._house_parse_source(
        "v9-ling-3.0-flash-vl", "openrouter/inclusionai/ling-3.0-flash-vl:free"
    ) == "gemini_ocr"
    assert rebuild_staged._house_parse_source(
        "v7-gpt-5.6-sol", "gpt-5.6-sol"
    ) == "gemini_ocr"
    assert rebuild_staged._house_parse_source(
        "v4-deterministic", "pdfplumber"
    ) == "house_pdf"


def test_required_house_scope_matches_production_inputs_not_full_archive():
    from scripts import rebuild_staged

    required = set(rebuild_staged.REQUIRED_HOUSE_YEARS)
    assert set(range(2021, 2026)).issubset(required)
    assert date.today().year in required
    assert max(2015, date.today().year - 1) in required
    assert 2015 not in required
    assert required.issubset(set(rebuild_staged.HOUSE_YEARS))


def test_senate_ingest_rejects_generation_provenance_rewrite(tmp_path):
    from scripts import rebuild_staged
    from scripts.senate_sweep import verify_canaries

    generation = "manifest-generation"
    inventory = [
        {
            "member": "Katie Britt",
            "outcome": "parsed",
            "accepted_row_count": 1,
            "source_record_id": "britt-report",
            "official_filing_date": "2026-01-01",
            "ingestion_generation": generation,
        },
        {
            "member": "Rick Scott",
            "outcome": "parsed",
            "accepted_row_count": 12,
            "source_record_id": "scott-report",
            "official_filing_date": "2026-01-02",
            "ingestion_generation": generation,
        },
    ]
    (tmp_path / "transactions.jsonl").write_text(
        "".join(
            json.dumps({"ingestion_generation": child}) + "\n"
            for child in ("child-a", "child-b")
        )
    )
    (tmp_path / "report_inventory.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in inventory)
    )
    (tmp_path / "chunks.json").write_text("{}\n")

    class FakeDatabase:
        writes = 0

        def persist_source_refresh(self, **_kwargs):
            self.writes += 1

    db = FakeDatabase()
    with pytest.raises(ValueError, match="unsupported multi-generation"):
        rebuild_staged._ingest_senate(
            tmp_path,
            {
                "generation": generation,
                "status": "complete",
                "canary_expect": "exact",
                "canaries": verify_canaries(inventory),
                "artifacts": _senate_test_artifacts(tmp_path),
            },
            cast(rebuild_staged.Database, db),
        )
    assert db.writes == 0


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_mode",
        "no_gate",
        "missing_canaries",
        "missing_passed",
        "missing_failures",
        "failed",
        "fake_pass",
    ],
)
def test_senate_ingest_rejects_missing_or_failed_canary_contract(tmp_path, mutation):
    from scripts import rebuild_staged
    from scripts.senate_sweep import verify_canaries

    generation = "manifest-generation"
    inventory = [
        {
            "member": "Katie Britt",
            "outcome": "parsed",
            "accepted_row_count": 1,
            "source_record_id": "britt-report",
            "official_filing_date": "2026-01-01",
            "ingestion_generation": generation,
        },
        {
            "member": "Rick Scott",
            "outcome": "parsed",
            "accepted_row_count": 12,
            "source_record_id": "scott-report",
            "official_filing_date": "2026-01-02",
            "ingestion_generation": generation,
        },
    ]
    (tmp_path / "report_inventory.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in inventory)
    )
    (tmp_path / "transactions.jsonl").write_text(
        json.dumps({"ingestion_generation": generation}) + "\n"
    )
    manifest = {
        "generation": generation,
        "status": "complete",
        "canary_expect": "exact",
        "canaries": verify_canaries(inventory),
        "artifacts": _senate_test_artifacts(tmp_path),
    }
    if mutation == "missing_mode":
        manifest.pop("canary_expect")
    elif mutation == "no_gate":
        manifest["canary_expect"] = "none"
    elif mutation == "missing_canaries":
        manifest.pop("canaries")
    elif mutation == "missing_passed":
        manifest["canaries"].pop("passed")
    elif mutation == "missing_failures":
        manifest["canaries"].pop("failures")
    elif mutation == "failed":
        manifest["canaries"]["passed"] = False
    else:
        manifest["canaries"]["results"] = {}

    class FakeDatabase:
        writes = 0

        def persist_source_refresh(self, **_kwargs):
            self.writes += 1

    db = FakeDatabase()
    with pytest.raises(ValueError, match="canary"):
        rebuild_staged._ingest_senate(
            tmp_path, manifest, cast(rebuild_staged.Database, db)
        )
    assert db.writes == 0


@pytest.mark.parametrize("failure", ["missing", "malformed"])
def test_senate_ingest_rejects_bad_artifact_claims_before_persistence(
    tmp_path, failure
):
    from scripts import rebuild_staged
    from scripts.senate_sweep import verify_canaries

    generation = "manifest-generation"
    inventory = [
        {
            "member": "Katie Britt",
            "outcome": "parsed",
            "accepted_row_count": 1,
            "source_record_id": "britt-report",
            "official_filing_date": "2026-01-01",
            "ingestion_generation": generation,
        },
        {
            "member": "Rick Scott",
            "outcome": "parsed",
            "accepted_row_count": 12,
            "source_record_id": "scott-report",
            "official_filing_date": "2026-01-02",
            "ingestion_generation": generation,
        },
    ]
    (tmp_path / "report_inventory.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in inventory)
    )
    (tmp_path / "transactions.jsonl").write_text(
        json.dumps({"ingestion_generation": generation}) + "\n"
    )
    original_bytes = {
        name: (tmp_path / name).read_bytes()
        for name in ("report_inventory.jsonl", "transactions.jsonl")
    }

    class FakeDatabase:
        writes = 0

        def persist_source_refresh(self, **_kwargs):
            self.writes += 1

    db = FakeDatabase()
    artifacts = _senate_test_artifacts(tmp_path)
    if failure == "missing":
        artifacts.pop("transactions.jsonl")
        error = "missing required Senate hash claim"
    else:
        artifacts["transactions.jsonl"]["sha256"] = "not-a-sha256"
        error = "malformed Senate artifact hash claim"
    canaries = verify_canaries(
        inventory,
        expected={"KATIE BRITT": None, "RICK SCOTT": None},
    )
    with pytest.raises(ValueError, match=error):
        rebuild_staged._ingest_senate(
            tmp_path,
            {
                "generation": generation,
                "status": "complete",
                "canary_expect": "any",
                "canaries": canaries,
                "artifacts": artifacts,
            },
            cast(rebuild_staged.Database, db),
        )
    assert db.writes == 0
    assert original_bytes == {
        name: (tmp_path / name).read_bytes() for name in original_bytes
    }


def test_text_only_pass_accepts_only_text_proven_complete_coverage(monkeypatch):
    from scripts import rebuild_staged

    row = {
        "asset_description": "Apple Inc. (AAPL)",
        "transaction_type": "Purchase",
        "transaction_date": pd.Timestamp("2026-01-02"),
        "amount_raw": "$1,001 - $15,000",
        "owner_code": "JT",
        "notification_date": pd.Timestamp("2026-01-03"),
    }
    monkeypatch.setattr(
        rebuild_staged._parser_cascade, "_try_pdfplumber", lambda _path: [row]
    )
    monkeypatch.setattr(
        rebuild_staged._parser_cascade, "_try_camelot_lattice", lambda _path: []
    )
    monkeypatch.setattr(
        rebuild_staged._parser_cascade, "_try_camelot_stream", lambda _path: []
    )
    monkeypatch.setattr(
        rebuild_staged._parser_cascade, "_try_pdftotext", lambda _path: [dict(row)]
    )
    # Fixture paths are not real PDFs; the page-coverage gate is covered by
    # cascade regression tests, so pin it here to test engine agreement only.
    monkeypatch.setattr(
        rebuild_staged._parser_cascade,
        "_has_document_text_coverage",
        lambda _path: True,
    )
    _, rows, engines = rebuild_staged._parser_cascade._parse_text_only_worker(
        rebuild_staged.Path("matching.pdf")
    )
    assert rows == [row]
    assert "won:pdftotext" in engines

    conflicting = dict(row)
    conflicting["transaction_date"] = pd.Timestamp("2026-01-04")
    monkeypatch.setattr(
        rebuild_staged._parser_cascade,
        "_try_pdftotext",
        lambda _path: [conflicting],
    )
    with pytest.raises(rebuild_staged.ParserCascadeError):
        rebuild_staged._parser_cascade._parse_text_only_worker(
            rebuild_staged.Path("conflicting.pdf")
        )


def test_house_parse_keeps_winning_fallback_rows_and_quarantines_total_failure(
    monkeypatch, tmp_path
):
    from scripts import rebuild_staged

    fallback_path = tmp_path / "fallback.pdf"
    failed_path = tmp_path / "failed.pdf"
    fallback_path.write_bytes(b"%PDF-fallback\n%%EOF")
    failed_path.write_bytes(b"%PDF-failed\n%%EOF")

    class FakeSource:
        def fetch_metadata(self, _year):
            return pd.DataFrame(
                {
                    "DocID": ["fallback", "failed"],
                    "FilingType": ["P", "P"],
                }
            )

        def close(self):
            pass

    class FakeQuery:
        def __init__(self, rows=()):
            self.rows = list(rows)

        def fetchall(self):
            return self.rows

    class FakeConnection:
        def execute(self, sql, *_args):
            if "SELECT DISTINCT doc_id" in sql:
                return FakeQuery([("failed",)])
            return FakeQuery()

    class FakeParseRuns:
        def get_cached_doc_ids(self, **_kwargs):
            return set()

    class FakeDatabase:
        def __init__(self):
            self.conn = FakeConnection()
            self.parse_runs = FakeParseRuns()
            self.replacements = []

        def get_latest_house_generation(self, year):
            assert year == 2026
            return "generation-2026"

        def get_unresolved_house_doc_ids(self, year, generation):
            assert year == 2026
            assert generation == "generation-2026"
            return ["fallback", "failed"]

        def replace_transactions_for_docs(self, dataframe, **kwargs):
            self.replacements.append((dataframe.copy(), kwargs))
            by_doc_total = {
                doc_id: int((dataframe.get("doc_id") == doc_id).sum())
                if "doc_id" in dataframe
                else 0
                for doc_id in kwargs["attempted_doc_ids"]
            }
            return SimpleNamespace(by_doc_total=by_doc_total)

    class FakePool:
        def __init__(self, _workers):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def map(self, worker, paths, chunksize=1):
            raise AssertionError("readiness preflight must not gate parsing")

        def imap_unordered(self, worker, paths, chunksize=1):
            assert chunksize == 1
            for path in paths:
                yield worker(path)

    fallback_rows = [
        {
            "doc_id": "fallback",
            "transaction_date": pd.Timestamp("2026-06-20"),
            "disclosure_date": pd.Timestamp("2026-06-26"),
            "source_row_id": "pdftotext:r1",
            "asset_description": "Apple Inc. (AAPL)",
        }
    ]

    def fake_parser(path):
        if path == fallback_path:
            return path, fallback_rows, ["lattice", "error:lattice", "won:pdftotext"]
        raise rebuild_staged.ParserCascadeError("unresolved parser completeness")

    def fake_consolidate(pdf_transactions, _member_lookup):
        if fallback_path in pdf_transactions:
            assert pdf_transactions[fallback_path] == fallback_rows
            return pd.DataFrame(fallback_rows)
        assert pdf_transactions == {failed_path: []}
        return pd.DataFrame()

    fake_database = FakeDatabase()
    monkeypatch.setattr(rebuild_staged, "_house_source", lambda _staging: FakeSource())
    monkeypatch.setattr(
        rebuild_staged,
        "filter_existing_pdfs",
        lambda _ptrs, _pdf_dir: (
            [failed_path, fallback_path],
            pd.DataFrame({"DocID": ["failed", "fallback"]}),
        ),
    )
    monkeypatch.setattr(rebuild_staged, "build_member_lookup", lambda _docs: {})
    monkeypatch.setattr(
        rebuild_staged,
        "_settings_for",
        lambda _staging: SimpleNamespace(
            data=SimpleNamespace(get_workers=lambda: 1)
        ),
    )
    monkeypatch.setattr(rebuild_staged, "Pool", FakePool)
    monkeypatch.setattr(
        rebuild_staged,
        "_tolerant_text_parse_worker",
        lambda path: (
            path,
            [],
            ["__parse_failed__:text engines did not prove complete coverage"],
        ),
    )
    monkeypatch.setattr(rebuild_staged, "_parse_pdf_worker", fake_parser)
    from analyzer import download
    monkeypatch.setattr(download, "consolidate_transactions", fake_consolidate)
    monkeypatch.setattr(
        rebuild_staged, "preserve_existing_fields", lambda dataframe, _db: dataframe
    )

    result = rebuild_staged._parse_house_year_tolerant(
        tmp_path, cast(rebuild_staged.Database, fake_database), 2026
    )

    assert result["parse_run_statuses"] == {"success": 1, "error": 1}
    assert len(fake_database.replacements) == 2
    stored_df, fallback_kwargs = fake_database.replacements[0]
    assert stored_df["doc_id"].tolist() == ["fallback"]
    assert fallback_kwargs["attempted_doc_ids"] == ["fallback"]
    assert fallback_kwargs["replacement_doc_ids"] == ["fallback"]
    failed_df, failed_kwargs = fake_database.replacements[1]
    assert failed_df.empty
    assert failed_kwargs["attempted_doc_ids"] == ["failed"]
    assert failed_kwargs["replacement_doc_ids"] == []

    parse_runs = {
        run["doc_id"]: run
        for _, kwargs in fake_database.replacements
        for run in kwargs["parse_runs"]
    }
    assert parse_runs["fallback"]["status"] == "success"
    assert parse_runs["fallback"]["error_message"] is None
    assert parse_runs["fallback"]["engines_attempted"] == (
        "lattice,error:lattice,won:pdftotext"
    )
    assert parse_runs["failed"]["status"] == "error"
    assert parse_runs["failed"]["error_message"] == "unresolved parser completeness"
    assert parse_runs["failed"]["engines_attempted"] == "cascade-failed"


def test_house_parse_streams_text_results_before_ocr_tail_and_persists_tail_immediately(
    monkeypatch, tmp_path
):
    from scripts import rebuild_staged

    stems = ["fast2", "defer", "retry", "fast1", "terminal"]
    paths = [tmp_path / f"{stem}.pdf" for stem in stems]
    for path in paths:
        path.write_bytes(b"%PDF-test\n%%EOF")
    existing_docs = pd.DataFrame({"DocID": [path.stem for path in paths]})

    source = SimpleNamespace(
        fetch_metadata=lambda _year: pd.DataFrame(
            {
                "DocID": [path.stem for path in paths],
                "FilingType": ["P"] * len(paths),
            }
        ),
        close=lambda: None,
    )
    db = SimpleNamespace(
        conn=SimpleNamespace(
            execute=lambda sql, *_args: SimpleNamespace(
                fetchall=lambda: (
                    [("retry",)] if "SELECT DISTINCT doc_id" in sql else []
                )
            )
        ),
        parse_runs=SimpleNamespace(get_cached_doc_ids=lambda **_kwargs: set()),
        get_latest_house_generation=lambda _year: "generation",
        get_unresolved_house_doc_ids=lambda _year, _generation: [
            "fast2", "defer", "retry", "fast1"
        ],
    )

    text_requested = []
    full_requested = []
    persisted = []

    class FakePool:
        def __init__(self, _workers):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def imap_unordered(self, worker, items, chunksize=1):
            assert chunksize == 1
            for path in items:
                yield worker(path)

    def fake_text_worker(path):
        text_requested.append(path.stem)
        if path.stem == "defer":
            return path, [], ["__parse_failed__:text pass deferred"]
        return path, [{"doc_id": path.stem}], ["won:pdftotext"]

    def fake_full_worker(path):
        full_requested.append(path.stem)
        return path, [{"doc_id": path.stem}], ["won:reconciled_complete_ocr"]

    def fake_persist(_db, **kwargs):
        batch = [path.stem for path, *_ in kwargs["results"]]
        persisted.append(batch)
        count = len(batch)
        return {
            "attempted": count,
            "parse_run_statuses": {"success": count},
            "persisted_transactions": count,
        }

    monkeypatch.setattr(rebuild_staged, "_house_source", lambda _staging: source)
    monkeypatch.setattr(
        rebuild_staged,
        "filter_existing_pdfs",
        lambda _ptrs, _pdf_dir: (paths, existing_docs),
    )
    monkeypatch.setattr(rebuild_staged, "build_member_lookup", lambda _docs: {})
    monkeypatch.setattr(
        rebuild_staged,
        "_settings_for",
        lambda _staging: SimpleNamespace(
            data=SimpleNamespace(get_workers=lambda: 2)
        ),
    )
    monkeypatch.setattr(rebuild_staged, "Pool", FakePool)
    page_counts = {
        "fast2": 1,
        "retry": 1,
        "terminal": 1,
        "fast1": 2,
        "defer": 3,
    }
    monkeypatch.setattr(
        rebuild_staged,
        "_pdf_page_count_hint",
        lambda path: page_counts[path.stem],
    )
    monkeypatch.setattr(rebuild_staged, "_tolerant_text_parse_worker", fake_text_worker)
    monkeypatch.setattr(rebuild_staged, "_tolerant_parse_worker", fake_full_worker)
    monkeypatch.setattr(rebuild_staged, "_persist_house_parse_batch", fake_persist)

    result = rebuild_staged._parse_house_year_tolerant(
        tmp_path, cast(rebuild_staged.Database, db), 2026
    )

    assert text_requested == ["fast2", "fast1", "defer"]
    assert full_requested == ["defer", "retry"]
    assert persisted == [["fast2", "fast1"], ["defer"], ["retry"]]
    assert result["attempted"] == 4


def test_cached_gemini_ingest_requires_exact_artifact_and_preserves_model_identity(
    monkeypatch, tmp_path
):
    from analyzer.persistence.database import Database
    from scripts import rebuild_staged
    from analyzer.ocr import common as gemini_ocr_common

    staging = tmp_path / "stage"
    pdf_dir = staging / "2024" / "pdfs"
    cache_dir = tmp_path / "cache"
    pdf_dir.mkdir(parents=True)
    cache_dir.mkdir()
    pdf_path = pdf_dir / "cache-doc.pdf"
    pdf_path.write_bytes(b"%PDF-test-cache\n%%EOF")
    artifact_sha = rebuild_staged._sha256_file(pdf_path)

    db = Database(staging / "congress.duckdb")
    db.conn.execute(
        """
        INSERT INTO house_archive_generations (
            archive_year, generation_id, metadata_count, ptr_count, parse_status
        ) VALUES (2024, 'cache-generation', 1, 1, 'incomplete')
        """
    )
    db.conn.execute(
        """
        INSERT INTO metadata (
            doc_id, archive_year, first_name, last_name,
            filing_date, filing_type, fetched_at
        ) VALUES (
            'cache-doc', 2024, 'Jane', 'Doe',
            '2024-01-20', 'P', CURRENT_TIMESTAMP
        )
        """
    )
    db.conn.execute(
        """
        INSERT INTO house_generation_metadata (
            archive_year, generation_id, doc_id, first_name, last_name,
            filing_date, filing_type, fetched_at
        ) VALUES (
            2024, 'cache-generation', 'cache-doc', 'Jane', 'Doe',
            '2024-01-20', 'P', CURRENT_TIMESTAMP
        )
        """
    )
    db.conn.execute(
        """
        INSERT INTO house_pdf_artifacts (
            archive_year, doc_id, generation_id, artifact_sha256
        ) VALUES (2024, 'cache-doc', 'cache-generation', ?)
        """,
        [artifact_sha],
    )
    envelope = {
        "cache_envelope_version": gemini_ocr_common.CACHE_ENVELOPE_VERSION,
        "output_schema_version": gemini_ocr_common.OUTPUT_SCHEMA_VERSION,
        "doc_id": "cache-doc",
        "pdf_sha256": artifact_sha,
        "pdf_page_count": 1,
        "model": "gemini/test-model",
        "prompt_sha256": gemini_ocr_common.PROMPT_SHA256,
        "parser_version": "v-test-gemini",
        "output": (
            "MEMBER: Jane Doe\nPAGES: 1\nPAGE: 1\n"
            "Apple Inc. (AAPL) | Purchase | 01/15/24 | 01/16/24 | A\n"
        ),
    }
    (cache_dir / "cache-doc.json").write_text(json.dumps(envelope))
    monkeypatch.setattr(gemini_ocr_common, "pdf_page_count", lambda _path: 1)

    try:
        result = rebuild_staged._ingest_cached_gemini_year(
            staging, db, 2024, cache_dir
        )
        assert [row["doc_id"] for row in result["accepted"]] == ["cache-doc"]
        assert result["rejected"] == []
        assert db.conn.execute(
            """
            SELECT source, ticker, transaction_type, ingestion_generation
            FROM transactions WHERE doc_id = 'cache-doc'
            """
        ).fetchall() == [
            ("gemini_ocr", "AAPL", "Purchase", "cache-generation")
        ]
        assert db.conn.execute(
            """
            SELECT parser_version, engines_attempted, status, transaction_count
            FROM pdf_parse_runs WHERE doc_id = 'cache-doc'
            """
        ).fetchall() == [
            ("v-test-gemini", "gemini/test-model", "success", 1)
        ]
    finally:
        db.close()


def test_repair_audit_gaps_reconciles_parser_source_family_and_retires_orphan_rows(tmp_path):
    from analyzer.persistence.database import Database
    from scripts import rebuild_staged

    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "manifest.json").write_text(
        json.dumps(
            {
                "generation": "test-generation",
                "house": {
                    "2024": {
                        "generation_id": "house-generation",
                        "ptr_count": 2,
                    }
                },
            }
        )
    )
    db = Database(staging / "congress.duckdb")
    db.conn.execute(
        """
        INSERT INTO house_archive_generations (
            archive_year, generation_id, metadata_count, ptr_count, parse_status
        ) VALUES (2024, 'house-generation', 2, 2, 'incomplete')
        """
    )
    db.conn.executemany(
        """
        INSERT INTO house_generation_metadata (
            archive_year, generation_id, doc_id, first_name, last_name,
            filing_date, filing_type
        ) VALUES (2024, 'house-generation', ?, 'Jane', 'Doe', '2024-01-20', 'P')
        """,
        [("gemini-good",), ("gemini-error",)],
    )
    db.conn.executemany(
        """
        INSERT INTO house_pdf_artifacts (
            archive_year, generation_id, doc_id, artifact_sha256
        ) VALUES (2024, 'house-generation', ?, ?)
        """,
        [("gemini-good", "a" * 64), ("gemini-error", "b" * 64)],
    )
    db.conn.executemany(
        """
        INSERT INTO transactions (
            doc_id, member, ticker, transaction_date, disclosure_date,
            transaction_type, source, chamber, source_record_id, source_row_id,
            official_filing_date, ingestion_generation, artifact_sha256
        ) VALUES (?, 'Jane Doe', 'AAPL', '2024-01-15', '2024-01-20',
                  'Purchase', 'gemini_ocr', 'House', ?, ?, '2024-01-20',
                  'house-generation', ?)
        """,
        [
            ("gemini-good", "gemini-good", "gemini-good:r1", "a" * 64),
            ("gemini-error", "gemini-error", "gemini-error:r1", "b" * 64),
        ],
    )
    db.upsert_parse_run(
        doc_id="gemini-good",
        year=2024,
        parser_version="v5-gemini-validated",
        status="success",
        engines_attempted="gemini/model",
        raw_row_count=1,
        transaction_count=1,
        artifact_sha256="a" * 64,
        ingestion_generation="house-generation",
    )
    db.upsert_parse_run(
        doc_id="gemini-error",
        year=2024,
        parser_version="v5-gemini-validated",
        status="error",
        engines_attempted="gemini/model",
        raw_row_count=1,
        transaction_count=0,
        error_message="failed semantic validation",
        artifact_sha256="b" * 64,
        ingestion_generation="house-generation",
    )
    db.close()

    rebuild_staged.repair_audit_gaps(SimpleNamespace(staging=str(staging)))

    db = Database(staging / "congress.duckdb", read_only=True)
    try:
        assert db.conn.execute(
            "SELECT status, transaction_count FROM pdf_parse_runs "
            "WHERE doc_id='gemini-good'"
        ).fetchone() == ("success", 1)
        assert db.conn.execute(
            "SELECT COUNT(*) FROM transactions WHERE doc_id='gemini-good'"
        ).fetchone()[0] == 1
        assert db.conn.execute(
            "SELECT COUNT(*) FROM transactions WHERE doc_id='gemini-error'"
        ).fetchone()[0] == 0
    finally:
        db.close()


def _seed_house_inventory_database(tmp_path):
    from analyzer.persistence.database import Database

    db = Database(tmp_path / "house-inventory.duckdb")
    db.conn.execute(
        """
        INSERT INTO house_archive_generations (
            archive_year, generation_id, metadata_count, ptr_count
        ) VALUES (2026, 'house-generation', 3, 3)
        """
    )
    db.conn.executemany(
        """
        INSERT INTO house_generation_metadata (
            archive_year, generation_id, doc_id, first_name, last_name,
            filing_date, filing_type
        ) VALUES (2026, 'house-generation', ?, ?, ?, '2026-01-03', 'P')
        """,
        [("deterministic", "Jane", "Doe"), ("ocr", "John", "Doe"), ("empty", "Alex", "Doe")],
    )
    db.conn.executemany(
        """
        INSERT INTO house_pdf_artifacts (
            archive_year, generation_id, doc_id, artifact_sha256
        ) VALUES (2026, 'house-generation', ?, ?)
        """,
        [("deterministic", "a" * 64), ("ocr", "b" * 64), ("empty", "c" * 64)],
    )
    db.conn.execute(
        """
        INSERT INTO transactions (
            doc_id, member, ticker, transaction_date, disclosure_date,
            transaction_type, source, chamber, source_record_id, source_row_id,
            official_filing_date, ingestion_generation, artifact_sha256
        ) VALUES (
            'deterministic', 'Jane Doe', 'AAPL', '2026-01-01', '2026-01-03',
            'Purchase', 'house_pdf', 'house', 'deterministic', 'pdf:r1',
            '2026-01-03', 'house-generation', ?
        )
        """,
        ["a" * 64],
    )
    db.conn.execute(
        """
        INSERT INTO transactions (
            doc_id, member, ticker, transaction_date, disclosure_date,
            transaction_type, source, chamber, source_record_id, source_row_id,
            official_filing_date, ingestion_generation, artifact_sha256
        ) VALUES (
            'ocr', 'John Doe', 'MSFT', '2026-01-01', '2026-01-03',
            'Purchase', 'gemini_ocr', 'House', 'ocr', 'ocr:r1',
            '2026-01-03', 'house-generation', ?
        )
        """,
        ["b" * 64],
    )
    db.upsert_parse_run(
        doc_id="deterministic", year=2026, parser_version="v5-deterministic",
        status="success", engines_attempted="pdf", raw_row_count=1, transaction_count=1,
        artifact_sha256="a" * 64, ingestion_generation="house-generation",
    )
    db.upsert_parse_run(
        doc_id="ocr", year=2026, parser_version="v5-gemini-validated",
        status="success", engines_attempted="ocr", raw_row_count=1, transaction_count=1,
        artifact_sha256="b" * 64, ingestion_generation="house-generation",
    )
    db.upsert_parse_run(
        doc_id="empty", year=2026, parser_version="v5-deterministic",
        status="no_txs", engines_attempted="pdf", raw_row_count=0, transaction_count=0,
        artifact_sha256="c" * 64, ingestion_generation="house-generation",
    )
    return db


def test_house_inventory_rows_include_mixed_sources_and_no_txs(tmp_path):
    from scripts import rebuild_staged

    db = _seed_house_inventory_database(tmp_path)
    try:
        rows = rebuild_staged._house_inventory_rows(
            db, 2026, "house-generation"
        )
    finally:
        db.close()

    assert [row["source_record_id"] for row in rows] == [
        "deterministic", "empty", "ocr"
    ]
    by_id = {row["source_record_id"]: row for row in rows}
    assert by_id["deterministic"]["source"] == "house_pdf"
    assert by_id["deterministic"]["outcome"] == "parsed"
    assert by_id["ocr"]["source"] == "gemini_ocr"
    assert by_id["ocr"]["outcome"] == "parsed"
    assert by_id["empty"]["source"] == "house_pdf"
    assert by_id["empty"]["outcome"] == "no_txs"
    assert by_id["empty"]["accepted_row_count"] == 0


def test_house_inventory_rows_reject_ambiguous_terminal_runs(tmp_path):
    from scripts import rebuild_staged

    db = _seed_house_inventory_database(tmp_path)
    try:
        db.upsert_parse_run(
            doc_id="ocr", year=2026, parser_version="v5-gemini-alt",
            status="success", engines_attempted="pdf", raw_row_count=1, transaction_count=1,
            artifact_sha256="b" * 64, ingestion_generation="house-generation",
        )
        with pytest.raises(RuntimeError, match="ambiguous terminal parse runs"):
            rebuild_staged._house_inventory_rows(db, 2026, "house-generation")
    finally:
        db.close()


def test_house_inventory_rows_reject_stale_source_bindings(tmp_path):
    from scripts import rebuild_staged

    db = _seed_house_inventory_database(tmp_path)
    try:
        db.conn.execute(
            """
            INSERT INTO transactions (
                doc_id, member, ticker, transaction_date, disclosure_date,
                transaction_type, source, chamber, source_record_id, source_row_id,
                official_filing_date, ingestion_generation, artifact_sha256
            ) VALUES (
                'ocr', 'John Doe', 'TSLA', '2026-01-01', '2026-01-03',
                'Purchase', 'house_pdf', 'house', 'ocr', 'stale:r1',
                '2026-01-03', 'house-generation', ?
            )
            """,
            ["b" * 64],
        )
        with pytest.raises(RuntimeError, match="stale source/artifact bindings"):
            rebuild_staged._house_inventory_rows(db, 2026, "house-generation")
    finally:
        db.close()


def test_refresh_house_completion_replaces_each_source_inventory(tmp_path):
    from scripts import rebuild_staged

    db = _seed_house_inventory_database(tmp_path)
    try:
        house = {
            "generation_id": "house-generation",
            "ptr_count": 3,
        }
        unresolved, report_count = rebuild_staged._refresh_house_completion(
            db, house, 2026
        )
        assert unresolved == []
        assert report_count == 3
        assert house["parse_status"] == "complete"
        assert db.conn.execute(
            """
            SELECT parse_status FROM house_archive_generations
            WHERE archive_year = 2026 AND generation_id = 'house-generation'
            """
        ).fetchone()[0] == "complete"
        assert db.source_reports.reconcile(
            "house-generation", "house_pdf", "house"
        ) == {
            "found": 2,
            "parsed": 1,
            "no_txs": 1,
            "paper_only": 0,
            "unavailable": 0,
            "failed": 0,
        }
        assert db.source_reports.reconcile(
            "house-generation", "gemini_ocr", "house"
        ) == {
            "found": 1,
            "parsed": 1,
            "no_txs": 0,
            "paper_only": 0,
            "unavailable": 0,
            "failed": 0,
        }
    finally:
        db.close()


def _seed_semantically_stale_generation_database(tmp_path):
    from analyzer.persistence.database import Database

    db = Database(tmp_path / "canonical-semantic.duckdb")
    generations = [
        ("g1", "artifact-g1", "2026-07-01 00:00:00", None),
        ("g2", "artifact-g2", "2026-07-02 00:00:00", date(2026, 1, 1)),
    ]
    for generation, artifact_sha, promoted_at, notification_date in generations:
        db.conn.execute(
            """
            INSERT INTO house_archive_generations (
                archive_year, generation_id, metadata_sha256,
                metadata_count, ptr_count, parse_status, promoted_at
            ) VALUES (2026, ?, 'metadata', 1, 1, 'complete', ?)
            """,
            [generation, promoted_at],
        )
        db.conn.execute(
            """
            INSERT INTO house_generation_metadata (
                archive_year, generation_id, doc_id, first_name, last_name,
                filing_date, filing_type, fetched_at
            ) VALUES (
                2026, ?, 'semantic-doc', 'Jane', 'Doe',
                '2026-01-03', 'P', '2026-01-04'
            )
            """,
            [generation],
        )
        db.conn.execute(
            """
            INSERT INTO house_pdf_artifacts (
                archive_year, doc_id, generation_id, artifact_sha256
            ) VALUES (2026, 'semantic-doc', ?, ?)
            """,
            [generation, artifact_sha],
        )
        db.upsert_transactions(
            pd.DataFrame(
                [
                    {
                        "doc_id": "semantic-doc",
                        "member": "Jane Doe",
                        "ticker": "OLD" if generation == "g1" else "NEW",
                        "transaction_date": date(2026, 1, 2),
                        "disclosure_date": date(2026, 1, 3),
                        "notification_date": notification_date,
                        "transaction_type": "Purchase",
                        "chamber": "house",
                        "source_record_id": "semantic-doc",
                        "source_row_id": f"{generation}:r1",
                        "official_filing_date": date(2026, 1, 3),
                        "ingestion_generation": generation,
                        "artifact_sha256": artifact_sha,
                    }
                ]
            ),
            source="house_pdf",
        )
        db.upsert_parse_run(
            doc_id="semantic-doc",
            year=2026,
            parser_version="v4-deterministic",
            status="success",
            engines_attempted="pdfplumber",
            raw_row_count=1,
            transaction_count=1,
            artifact_sha256=artifact_sha,
            ingestion_generation=generation,
        )
    return db


def test_canonical_view_check_uses_latest_semantically_accepted_generation(tmp_path):
    from scripts import rebuild_staged

    db = _seed_semantically_stale_generation_database(tmp_path)
    try:
        assert rebuild_staged._latest_accepted_house_generations(db) == {2026: "g1"}
        assert rebuild_staged._canonical_view_diff_counts(db) == (0, 0)

        db.conn.execute(
            """
            CREATE OR REPLACE TEMP VIEW canonical_transactions AS
            SELECT * FROM transactions
            """
        )
        assert rebuild_staged._canonical_view_diff_counts(db) == (1, 0)

        db.conn.execute(
            """
            CREATE OR REPLACE TEMP VIEW canonical_transactions AS
            SELECT * FROM transactions WHERE FALSE
            """
        )
        assert rebuild_staged._canonical_view_diff_counts(db) == (0, 1)
    finally:
        db.close()


def test_refresh_house_completion_demotes_db_generation_on_unresolved_doc(tmp_path):
    from scripts import rebuild_staged

    db = _seed_house_inventory_database(tmp_path)
    try:
        initial_house = {
            "generation_id": "house-generation",
            "ptr_count": 3,
        }
        initial_unresolved, initial_report_count = rebuild_staged._refresh_house_completion(
            db, initial_house, 2026
        )
        assert initial_unresolved == []
        assert initial_report_count == 3
        assert db.source_reports.reconcile(
            "house-generation", "house_pdf", "house"
        )["found"] == 2
        assert db.source_reports.reconcile(
            "house-generation", "gemini_ocr", "house"
        )["found"] == 1
        db.conn.execute(
            """
            UPDATE transactions
            SET notification_date = '2025-12-31'
            WHERE doc_id = 'ocr' AND ingestion_generation = 'house-generation'
            """
        )
        house = {
            "generation_id": "house-generation",
            "ptr_count": 3,
            "parse_status": "complete",
        }
        unresolved, report_count = rebuild_staged._refresh_house_completion(
            db, house, 2026
        )

        assert unresolved == ["ocr"]
        assert report_count is None
        assert house["parse_status"] == "incomplete"
        assert db.conn.execute(
            """
            SELECT parse_status FROM house_archive_generations
            WHERE archive_year = 2026 AND generation_id = 'house-generation'
            """
        ).fetchone()[0] == "incomplete"
        assert db.source_reports.reconcile(
            "house-generation", "house_pdf", "house"
        )["found"] == 0
        assert db.source_reports.reconcile(
            "house-generation", "gemini_ocr", "house"
        )["found"] == 0
    finally:
        db.close()


def test_ocr_consumer_uses_only_manifest_bound_verified_current_pairs(tmp_path, monkeypatch):
    from scripts import ocr_local_sweep, rebuild_staged, scan_ocr
    monkeypatch.setattr(scan_ocr, "_tesseract_version", lambda: "test")

    doc_id = "consumer-bound"
    root = tmp_path / "ocr"
    ocr_local_sweep.stage_document(
        root,
        {
            "doc_id": doc_id,
            "year": 2015,
            "status": "resolved",
            "artifact_sha256": "source-hash",
            "page_count": 1,
            "covered_pages": [1],
            "uncovered_pages": [],
            "row_count": 1,
            "rows": [
                {
                    "doc_id": doc_id,
                    "source_record_id": doc_id,
                    "source_row_id": f"{doc_id}:page:1:row:1",
                    "artifact_sha256": "source-hash",
                    "asset_description": "Bound row",
                }
            ],
            "reasons": [],
            "engines": [],
            "canary": None,
        },
    )
    ocr_local_sweep.write_manifest(root, kind="sweep", data_dir=root)
    manifest = json.loads((root / "manifest.json").read_text())
    envelope = json.loads((root / "docs" / f"{doc_id}.json").read_text())
    rows_path = root / envelope["rows_file"]
    # A preserved prior attempt is not part of the active docs/manifest inventory.
    (root / "rows" / f"{doc_id}.archived.jsonl").write_bytes(rows_path.read_bytes())

    inventory = rebuild_staged._validated_ocr_inventory(root, manifest)
    assert len(inventory) == 1
    assert inventory[0]["doc_id"] == doc_id
    assert inventory[0]["rows"][0]["asset_description"] == "Bound row"

    # A legacy fixed filename is accepted only when this envelope and manifest
    # bind it; the previous hash-addressed attempt remains inactive.
    hashed_rows_file = envelope["rows_file"]
    legacy_rows_path = root / "rows" / f"{doc_id}.jsonl"
    legacy_rows_path.write_bytes(rows_path.read_bytes())
    envelope["rows_file"] = f"rows/{doc_id}.jsonl"
    envelope_path = root / "docs" / f"{doc_id}.json"
    envelope_path.write_text(json.dumps(envelope, indent=2, sort_keys=True) + "\n")
    hashes = manifest["staged_files_sha256"]
    hashes.pop(hashed_rows_file)
    hashes[envelope["rows_file"]] = ocr_local_sweep.sha256_file(legacy_rows_path)
    hashes[f"docs/{doc_id}.json"] = ocr_local_sweep.sha256_file(envelope_path)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    legacy_inventory = rebuild_staged._validated_ocr_inventory(root, manifest)
    assert legacy_inventory[0]["rows"] == inventory[0]["rows"]

    # scan_ocr uses the same publication/consumer contract and root lock.
    scan_root = tmp_path / "scan-ocr"
    scan_doc = "scan-consumer"
    scan_ocr.write_staging(
        scan_root,
        {
            scan_doc: {
                "doc_id": scan_doc,
                "artifact_sha256": "scan-source",
                "page_count": 1,
                "rows": [
                    scan_ocr.OcrRow(
                        page_number=1,
                        row_index=1,
                        asset_description="Scan row",
                        transaction_type="Purchase",
                        transaction_date_raw="3-31-26",
                        notification_date_raw=None,
                    )
                ],
                "resolved": True,
                "status": "resolved",
                "reasons": [],
                "uncovered_pages": [],
            }
        },
        data_dir=tmp_path,
        year=2026,
    )
    scan_manifest = json.loads((scan_root / "manifest.json").read_text())
    scan_inventory = rebuild_staged._validated_ocr_inventory(scan_root, scan_manifest)
    assert scan_inventory[0]["rows"][0]["doc_id"] == scan_doc

    legacy_rows_path.write_text(
        '{"doc_id":"consumer-bound","asset_description":"foreign"}\n'
    )
    with pytest.raises(ValueError, match="rows_file_sha256"):
        rebuild_staged._ingest_ocr(root, manifest, SimpleNamespace())


@pytest.mark.parametrize(
    ("staged_sha", "target_sha", "error"),
    [
        ("a" * 64, "b" * 64, "does not match staged database artifact"),
        (None, "a" * 64, "lacks a valid PDF hash"),
        ("a" * 64, None, "lacks a valid PDF hash"),
    ],
)
def test_ocr_ingest_binds_whole_rowset_to_db_artifact_before_writes(
    tmp_path, staged_sha, target_sha, error
):
    from scripts import ocr_local_sweep, rebuild_staged

    doc_id = "ocr-proof"
    root = tmp_path / "ocr-proof"
    rows = [
        {
            "doc_id": doc_id,
            "source_record_id": doc_id,
            "source_row_id": f"{doc_id}:page:1:row:{index}",
            "artifact_sha256": staged_sha,
            "ingestion_generation": "producer-attempt-generation",
            "asset_description": f"Asset {index}",
            "transaction_type": "Purchase",
            "transaction_date": f"2015-01-0{index}",
        }
        for index in (1, 2)
    ]
    ocr_local_sweep.stage_document(
        root,
        {
            "doc_id": doc_id,
            "year": 2015,
            "status": "resolved",
            "artifact_sha256": staged_sha,
            "page_count": 1,
            "covered_pages": [1],
            "uncovered_pages": [],
            "row_count": len(rows),
            "rows": rows,
            "reasons": [],
            "engines": [],
            "canary": None,
        },
    )
    ocr_local_sweep.write_manifest(root, kind="sweep", data_dir=root)
    manifest = json.loads((root / "manifest.json").read_text())
    original_bytes = {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }

    class FakeConnection:
        def __init__(self):
            self.target_sha = target_sha

        def execute(self, query, _params):
            if "FROM house_generation_metadata" in query:
                row = (
                    2015,
                    "archive-generation",
                    "Test",
                    "Member",
                    "2015-02-01",
                    self.target_sha,
                )
            else:
                assert "FROM pdf_parse_runs" in query
                row = (0,)
            return SimpleNamespace(fetchone=lambda: row)

    class FakeDatabase:
        def __init__(self):
            self.conn = FakeConnection()
            self.writes = []

        def replace_transactions_for_docs(self, frame, **kwargs):
            self.writes.append((frame.copy(), kwargs))
            return SimpleNamespace(by_doc_total={doc_id: len(frame)})

    db = FakeDatabase()
    with pytest.raises(ValueError, match=error):
        rebuild_staged._ingest_ocr(root, manifest, db)
    assert db.writes == []
    assert original_bytes == {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }

    if staged_sha is not None:
        # Matching staged and archived hashes retain the existing archive
        # generation mapping and row count; producer attempt bytes stay intact.
        db.conn.target_sha = staged_sha
        result = rebuild_staged._ingest_ocr(root, manifest, db)
        assert result["ingested_docs"] == [doc_id]
        assert result["ingested_rows"] == 2
        assert len(db.writes) == 1
        frame, kwargs = db.writes[0]
        assert len(frame) == 2
        assert set(frame["artifact_sha256"]) == {staged_sha}
        assert set(frame["ingestion_generation"]) == {"archive-generation"}
        assert kwargs["parse_runs"][0]["artifact_sha256"] == staged_sha
        assert kwargs["parse_runs"][0]["ingestion_generation"] == "archive-generation"
        assert original_bytes == {
            path.relative_to(root): path.read_bytes()
            for path in root.rglob("*")
            if path.is_file()
        }
