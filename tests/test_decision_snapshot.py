import json
import logging
import os
import subprocess
import tempfile
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from typer.testing import CliRunner

from analyzer import decision_snapshot as receipts
from analyzer.setups import score


DAY = date(2024, 6, 1)


@pytest.fixture
def inputs():
    row = dict(
        ticker="AAPL",
        actor_id="congress:ALICE",
        kind="congress",
        source="house_pdf",
        source_record_id="filing-123",
        entry_ref=100.0,
        disclosure_date=date(2024, 5, 30),
        as_of=DAY,
        corroboration=False,
        position_evidence=True,
        blocked_reason=float("nan"),
    )
    fund = dict(
        row,
        actor_id="manager:FUND A",
        source="13f",
        source_record_id="13f-456",
        corroboration=True,
    )
    return (
        pd.DataFrame([row, fund]),
        {"congress:ALICE": 1.0},
        {"AAPL": (100.0, DAY)},
        {"manager:FUND A"},
    )


def _receipt_dir(tmp_path):
    return tmp_path / "recommendations" / str(DAY)


def _receipts(tmp_path):
    return sorted(_receipt_dir(tmp_path).glob("*.json"))


def test_record_idempotent_and_replay(tmp_path, inputs):
    candidates, weights, quotes, watchlist = inputs
    ranked, blocked = score(candidates, weights, quotes, watchlist=watchlist)
    path = receipts.record_decision(tmp_path, DAY, *inputs, ranked, blocked)
    original, stat = path.read_bytes(), path.stat()
    # Identical rerun: same content-addressed id, a true no-op.
    assert receipts.record_decision(tmp_path, DAY, *inputs, ranked, blocked) == path
    assert path.read_bytes() == original
    assert path.stat().st_mtime_ns == stat.st_mtime_ns
    assert receipts.verify_decision(path) == {"agreement": True, "mismatches": []}
    manifest = json.loads(original)
    winner = manifest["ranked"][0]
    assert winner["rank"] == 1 and winner["score"] == 1.5
    assert winner["members"] == [
        {
            "member": "congress:ALICE",
            "entry_ref": 100.0,
            "weight": 1.0,
            "contribution": 1.0,
        }
    ]
    assert winner["quote_date"] == DAY.isoformat()
    assert winner["quote_value"] == 100.0 and winner["corroborated_13f"]
    assert len(manifest["input_rows_hash"]) == 64
    assert manifest["price_snapshot"]["snapshot_id"].endswith(
        manifest["price_snapshot"]["value_hash"]
    )
    # decision_id is the hash of the canonical payload EXCLUDING manifest_hash.
    without_digest = json.loads(original)
    digest = without_digest.pop("manifest_hash")
    assert digest == receipts._hash(without_digest)
    assert path.stem == digest
    # The publication is anchored in the append-only ledger.
    index = receipts.date_index(tmp_path, DAY)
    assert index == [path.stem]


def test_same_day_distinct_decisions_both_recorded(tmp_path, inputs, caplog):
    """Reviewer reproduction: a second same-day decision must get its own
    receipt and ledger line, never be swallowed as a conflict, and an
    initial empty decision must not block later receipts."""
    candidates, weights, quotes, watchlist = inputs
    ranked, blocked = score(candidates, weights, quotes, watchlist=watchlist)
    with caplog.at_level(logging.INFO, logger="analyzer.decision_snapshot"):
        empty = receipts.record_decision(tmp_path, DAY, pd.DataFrame(), {}, {}, set())
        first = receipts.record_decision(tmp_path, DAY, *inputs, ranked, blocked)
        other = receipts.record_decision(
            tmp_path, DAY, candidates, {"congress:ALICE": 2.0}, quotes, watchlist
        )
    paths = {empty, first, other}
    assert len(paths) == 3 and len(_receipts(tmp_path)) == 3
    for path in paths:
        assert receipts.verify_decision(path)["agreement"], path
    # Every distinct decision is individually visible in the ledger index.
    assert receipts.date_index(tmp_path, DAY) == [empty.stem, first.stem, other.stem]
    lines = (tmp_path / "recommendations" / "receipts.log").read_text().splitlines()
    assert len(lines) == 3
    # The differing decision was RECORDED (logged), not swallowed.
    assert "distinct decisions now recorded" in caplog.text


def _attacker_write(path, text):
    """An owner can chmod the 0444 receipt back to writable, then edit it."""
    os.chmod(path, 0o644)
    path.write_text(text)


def test_coherent_replacement_attack_is_rejected(tmp_path, inputs):
    """Reviewer reproduction: regenerate derived fields + internal hashes and
    os.replace the receipt. verify_decision must FAIL: the rewritten payload
    matches no ledger entry, and the ledger chain still anchors the original.
    """
    candidates, weights, quotes, watchlist = inputs
    path = receipts.record_decision(tmp_path, DAY, *inputs)
    assert receipts.verify_decision(path)["agreement"]
    original_digest = path.stem

    manifest = json.loads(path.read_text())
    # The attacker changes a frozen quote 100 -> 101 ...
    manifest["inputs"]["quotes"]["AAPL"] = [101.0, DAY.isoformat()]
    # ... regenerates every derived field and internal hash ...
    manifest["input_rows_hash"] = receipts._hash(manifest["inputs"]["candidates"])
    manifest["price_snapshot"] = receipts._price_reference(
        manifest["inputs"], date.fromisoformat(manifest["decision_date"])
    )
    manifest.update(receipts._derive(manifest["inputs"]))
    manifest.pop("manifest_hash")
    manifest["manifest_hash"] = receipts._hash(manifest)
    forged = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    # ... and publishes with an atomic replace.
    os.chmod(path, 0o644)
    fd, staged = tempfile.mkstemp(dir=path.parent)
    with os.fdopen(fd, "w") as stream:
        stream.write(forged)
    os.replace(staged, path)

    result = receipts.verify_decision(path)
    assert result["agreement"] is False
    assert "receipt_not_in_ledger" in result["mismatches"]
    # The ledger still anchors the ORIGINAL decision history; the forged
    # digest matches no ledger entry anywhere.
    index = receipts.date_index(tmp_path, DAY)
    assert original_digest in index and manifest["manifest_hash"] not in index


def test_ledger_missing_or_corrupt_fails_closed(tmp_path, inputs):
    path = receipts.record_decision(tmp_path, DAY, *inputs)
    ledger = tmp_path / "recommendations" / "receipts.log"
    backup = ledger.read_bytes()

    ledger.unlink()
    assert receipts.verify_decision(path)["agreement"] is False
    assert any(
        mismatch.startswith("ledger") or mismatch == "receipt_not_in_ledger"
        for mismatch in receipts.verify_decision(path)["mismatches"]
    )

    ledger.write_bytes(backup)
    lines = ledger.read_text().splitlines()
    entry = json.loads(lines[0])
    entry["manifest_hash"] = "0" * 64  # flip one field: chain history rewritten
    lines[0] = json.dumps(entry, sort_keys=True, separators=(",", ":"))
    ledger.write_text("\n".join(lines) + "\n")
    tampered = receipts.verify_decision(path)
    assert tampered["agreement"] is False
    assert any(m.startswith("ledger") for m in tampered["mismatches"])


def test_ledger_chain_tamper_fails_closed_for_recording_too(
    tmp_path, inputs, caplog
):
    candidates, weights, quotes, watchlist = inputs
    receipts.record_decision(tmp_path, DAY, *inputs)
    ledger = tmp_path / "recommendations" / "receipts.log"
    lines = ledger.read_text().splitlines()
    entry = json.loads(lines[0])
    entry["decision_id"] = "f" * 64
    lines[0] = json.dumps(entry, sort_keys=True, separators=(",", ":"))
    ledger.write_text("\n".join(lines) + "\n")

    with caplog.at_level(logging.ERROR, logger="analyzer.decision_snapshot"):
        assert receipts.record_decision_safely(
            tmp_path, DAY, *inputs
        ) is None
    assert "Decision receipt was not recorded" in caplog.text
    assert len(ledger.read_text().splitlines()) == 1  # nothing appended


def test_tampering_is_detected_even_with_recomputed_hash(tmp_path, inputs):
    path = receipts.record_decision(tmp_path, DAY, *inputs)
    manifest = json.loads(path.read_text())
    manifest["ranked"][0]["score"] += 1
    _attacker_write(path, json.dumps(manifest))
    # The stale digest still matches its ledger entry, but not the content.
    assert set(receipts.verify_decision(path)["mismatches"]) == {
        "manifest_hash",
        "ranked",
    }
    manifest.pop("manifest_hash")
    manifest["manifest_hash"] = receipts._hash(manifest)
    _attacker_write(path, json.dumps(manifest))
    # The recomputed digest matches no ledger entry: fail closed.
    mismatches = receipts.verify_decision(path)["mismatches"]
    assert "ranked" in mismatches and "receipt_not_in_ledger" in mismatches


def test_receipt_files_are_read_only_after_publish(tmp_path, inputs):
    path = receipts.record_decision(tmp_path, DAY, *inputs)
    assert path.stat().st_mode & 0o777 == 0o444


def test_cli_recording_is_byte_identical_and_failure_is_nonfatal(
    tmp_path, monkeypatch, inputs, caplog
):
    from analyzer import cli, pipeline, positions

    candidates, weights, quotes, watchlist = inputs
    connection = SimpleNamespace(
        execute=lambda *a: SimpleNamespace(fetchone=lambda: (100.0, DAY))
    )
    monkeypatch.setattr(
        cli,
        "_open_database",
        lambda *a, **kw: SimpleNamespace(conn=connection, close=lambda: None),
    )
    monkeypatch.setattr(pipeline, "eligible_events", lambda *a: candidates)
    monkeypatch.setattr(
        positions, "holdings_candidates", lambda *a, **kw: pd.DataFrame()
    )
    monkeypatch.setattr(positions, "coalesce_actor_variants", lambda events, _: events)
    monkeypatch.setattr(positions, "gate_closed_events", lambda events, _: events)
    monkeypatch.setattr(cli, "weights_for_candidates", lambda *a: weights)
    from analyzer import manager_watchlist

    monkeypatch.setattr(manager_watchlist, "watchlist_actor_ids", lambda: watchlist)
    args = [
        "setups",
        "--ticker",
        "AAPL",
        "--as-of",
        DAY.isoformat(),
        "--data-dir",
        str(tmp_path),
    ]
    enabled = CliRunner().invoke(cli.app, args)
    assert enabled.exit_code == 0, enabled.output
    receipts_written = _receipts(tmp_path)
    assert len(receipts_written) == 1
    assert receipts.verify_decision(receipts_written[0])["agreement"]
    with monkeypatch.context() as failure:
        def fail(*a, **kw):
            raise OSError("disk full")

        failure.setattr(receipts, "record_decision", fail)
        failed_write = CliRunner().invoke(cli.app, args)
        assert failed_write.exit_code == 0
        assert failed_write.output.encode() == enabled.output.encode()
        assert "disk full" in caplog.text
    monkeypatch.setattr(receipts, "record_decision_safely", lambda *a, **kw: None)
    disabled = CliRunner().invoke(cli.app, args)
    assert disabled.exit_code == 0
    assert enabled.output.encode() == disabled.output.encode()


def test_write_failure_logged_without_raising(tmp_path, inputs, monkeypatch, caplog):
    def fail(*a, **kw):
        raise OSError("disk full")

    monkeypatch.setattr(receipts.os, "link", fail)
    assert receipts.record_decision_safely(tmp_path, DAY, *inputs) is None
    assert "Decision receipt was not recorded" in caplog.text
    assert "disk full" in caplog.text
    assert not list(tmp_path.rglob(".decision-*"))


def test_empty_decision_is_recorded(tmp_path):
    path = receipts.record_decision(tmp_path, DAY, pd.DataFrame(), {}, {}, set())
    assert json.loads(path.read_text())["ranked"] == []
    assert receipts.verify_decision(path)["agreement"]


# ---------------------------------------------------------------------------
# Reviewer reproduction: input filing identities must survive into receipts
# ---------------------------------------------------------------------------


def test_eligible_events_preserve_filing_identity():
    from analyzer.pipeline import eligible_events

    def trades(record_id):
        return pd.DataFrame(
            [
                {
                    "member": "Jane Doe",
                    "ticker": "AAPL",
                    "transaction_date": pd.Timestamp("2025-05-20"),
                    "disclosure_date": pd.Timestamp("2025-05-30"),
                    "transaction_type": "Purchase",
                    "instrument_type": "stock",
                    "source": "form4",
                    "amount_midpoint": 1500.0,
                    "asset_description": "Common Stock; 10 shares @ $150",
                    "source_record_id": record_id,
                    "source_row_id": f"row-{record_id}",
                }
            ]
        )

    db = SimpleNamespace(
        get_transactions_by_date_range=lambda *a: trades("filing-A"),
        conn=None,
    )
    events = eligible_events(db, pd.Timestamp("2025-06-01"), days_back=None)
    assert events["source_record_id"].tolist() == ["filing-A"]
    assert events["source_row_id"].tolist() == ["row-filing-A"]

    db_swap = SimpleNamespace(
        get_transactions_by_date_range=lambda *a: trades("filing-B"),
        conn=None,
    )
    events_b = eligible_events(db_swap, pd.Timestamp("2025-06-01"), days_back=None)
    # Old behavior: frames byte-identical despite different filing identity.
    assert not events.equals(events_b)
    assert events_b["source_record_id"].tolist() == ["filing-B"]


def test_legacy_producer_without_ids_records_explicit_null():
    from analyzer.pipeline import eligible_events

    trades = pd.DataFrame(
        [
            {
                "member": "Jane Doe",
                "ticker": "AAPL",
                "transaction_date": pd.Timestamp("2025-05-20"),
                "disclosure_date": pd.Timestamp("2025-05-30"),
                "transaction_type": "Purchase",
                "instrument_type": "stock",
                "source": "gemini_ocr",
            }
        ]
    )
    db = SimpleNamespace(get_transactions_by_date_range=lambda *a: trades, conn=None)
    events = eligible_events(db, pd.Timestamp("2025-06-01"), days_back=None)
    assert events["source_record_id"].isna().all()
    assert events["source_row_id"].isna().all()


def test_holdings_candidates_track_contributing_filing_lots():
    import duckdb

    def db(record_id):
        conn = duckdb.connect()
        frame = pd.DataFrame(
            [
                {
                    "member": "M",
                    "ticker": "T",
                    "transaction_type": "Purchase",
                    "transaction_date": date(2025, 1, 14),
                    "disclosure_date": date(2025, 1, 20),
                    "amount_midpoint": 1000.0,
                    "source": "form4",
                    "source_record_id": record_id,
                }
            ]
        )
        prices = pd.DataFrame(
            [("T", date(2025, 1, 14), 100.0), ("T", date(2026, 9, 21), 120.0)],
            columns=["ticker", "date", "close"],
        )
        conn.register("t", frame)
        conn.register("p", prices)
        conn.execute("CREATE TABLE canonical_transactions AS SELECT * FROM t")
        conn.execute("CREATE TABLE prices AS SELECT * FROM p")
        return SimpleNamespace(conn=conn), conn

    holdings_a, conn_a = db("filing-A")
    holdings_b, conn_b = db("filing-B")
    try:
        from analyzer.positions import holdings_candidates

        candidates_a = holdings_candidates(holdings_a, "M", date(2026, 9, 21))
        candidates_b = holdings_candidates(holdings_b, "M", date(2026, 9, 21))
    finally:
        conn_a.close()
        conn_b.close()

    # Scoring is unchanged by the identity swap ...
    ranked_a, _ = score(candidates_a, {"officer:M": 1.0}, {"T": 120.0})
    ranked_b, _ = score(candidates_b, {"officer:M": 1.0}, {"T": 120.0})
    assert ranked_a.to_csv(index=False) == ranked_b.to_csv(index=False)
    # ... but provenance is not: the recorded bytes MUST change.
    assert candidates_a["filing_refs"].tolist() != candidates_b["filing_refs"].tolist()
    assert candidates_a["filing_refs"].iloc[0] == [
        {"source": "form4", "source_record_id": "filing-A", "source_row_id": None}
    ]
    assert candidates_b["filing_refs"].iloc[0][0]["source_record_id"] == "filing-B"


def test_filing_identity_change_changes_receipt_bytes(tmp_path, inputs):
    """Reviewer reproduction: swapping Form 4 filing identity filing-A ->
    filing-B must change the receipt bytes while keeping ranking identical."""
    candidates, weights, quotes, watchlist = inputs
    candidates["source_record_id"] = "filing-A"
    ranked, blocked = score(candidates, weights, quotes, watchlist=watchlist)
    first = receipts.record_decision(tmp_path, DAY, *inputs, ranked, blocked)
    ranked_bytes = json.loads(first.read_text())["ranked"]

    candidates["source_record_id"] = "filing-B"
    ranked_b, blocked_b = score(candidates, weights, quotes, watchlist=watchlist)
    second = receipts.record_decision(tmp_path, DAY, candidates, weights, quotes,
                                      watchlist, ranked_b, blocked_b)
    assert first != second
    assert first.read_bytes() != second.read_bytes()
    # The swap does NOT change the decision: byte-identical ranking output.
    assert json.loads(second.read_text())["ranked"] == ranked_bytes
    assert receipts.verify_decision(first)["agreement"]
    assert receipts.verify_decision(second)["agreement"]
    assert sorted(receipts.date_index(tmp_path, DAY)) == sorted(
        [first.stem, second.stem]
    )


# ---------------------------------------------------------------------------
# Reviewer reproduction: analyze boundaries must write receipts
# ---------------------------------------------------------------------------


@pytest.fixture
def analyze_ctx(tmp_path, monkeypatch):
    from analyzer import cli

    settings = SimpleNamespace(data=SimpleNamespace(data_dir=str(tmp_path)))
    monkeypatch.setattr(
        cli,
        "get_context",
        lambda *a, **kw: SimpleNamespace(settings=settings, transaction_source=None),
    )
    return cli


def test_analyze_ticker_path_records_receipt(analyze_ctx, tmp_path, monkeypatch):
    from analyzer.exceptions import DataResult

    cli = analyze_ctx
    score_frame = pd.DataFrame(
        [{"ticker": "SPCX", "num_buyers": 3, "signal_score": 3.0}]
    )
    monkeypatch.setattr(
        cli,
        "run_ticker_analysis",
        lambda params, source: DataResult(
            success=True,
            data={
                "buyers": pd.DataFrame(
                    [{"member": "A", "num_purchases": 1}]
                ),
                "score": score_frame,
                "ticker": "SPCX",
            },
        ),
    )
    result = CliRunner().invoke(
        cli.app,
        ["analyze", "--ticker", "SPCX", "--year", "2024", "--as-of", "2024-06-01",
         "--data-dir", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output
    written = sorted((tmp_path / "recommendations").rglob("*.json"))
    assert len(written) == 1
    manifest = json.loads(written[0].read_text())
    assert manifest["kind"] == "ticker_consensus"
    assert manifest["scope"]["ticker"] == "SPCX"
    assert receipts.verify_decision(written[0])["agreement"]
    assert receipts.date_index(tmp_path, "2024-06-01") == [written[0].stem]


def test_analyze_tickers_mode_records_receipt(analyze_ctx, tmp_path, monkeypatch):
    from analyzer.exceptions import DataResult

    cli = analyze_ctx
    result_frame = pd.DataFrame(
        [{"ticker": "SPCX", "num_buyers": 3, "signal_score": 3.0}]
    )
    monkeypatch.setattr(
        cli,
        "run_recent_ticker_scoring",
        lambda source, params: DataResult(
            success=True,
            data={
                "result": result_frame,
                "top_n": params.top_n,
                "days_back": params.days_back,
                "min_buyers": params.min_buyers,
                "as_of_date": DAY,
            },
        ),
    )
    result = CliRunner().invoke(
        cli.app,
        ["analyze", "--mode", "tickers", "--year", "2024", "--as-of",
         "2024-06-01", "--data-dir", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output
    written = sorted((tmp_path / "recommendations").rglob("*.json"))
    assert len(written) == 1
    manifest = json.loads(written[0].read_text())
    assert manifest["kind"] == "ticker_consensus"
    assert manifest["outputs"]["result"]["rows"][0][0] == "SPCX"
    assert receipts.verify_decision(written[0])["agreement"]


def test_analyze_receipt_failure_is_nonfatal(analyze_ctx, tmp_path, monkeypatch, caplog):
    from analyzer.exceptions import DataResult

    cli = analyze_ctx
    monkeypatch.setattr(
        cli,
        "run_recent_ticker_scoring",
        lambda source, params: DataResult(
            success=True,
            data={
                "result": pd.DataFrame([{"ticker": "SPCX", "signal_score": 3.0}]),
                "top_n": 20,
                "days_back": 28,
                "min_buyers": 3,
                "as_of_date": DAY,
            },
        ),
    )

    def fail(*a, **kw):
        raise OSError("disk full")

    monkeypatch.setattr(receipts, "record_ticker_decision", fail)
    result = CliRunner().invoke(
        cli.app,
        ["analyze", "--mode", "tickers", "--year", "2024", "--as-of",
         "2024-06-01", "--data-dir", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output
    assert "disk full" in caplog.text


# ---------------------------------------------------------------------------
# Reviewer reproduction: recorded git SHA must be the executing package's
# ---------------------------------------------------------------------------


def _git(directory, *args):
    return subprocess.run(
        ["git", "-C", str(directory), *args],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def test_git_sha_resolves_executing_package_not_cwd(tmp_path, monkeypatch, inputs):
    """Run branch code with CWD pointed at a different repository ('main'):
    the recorded SHA must be the executing package's revision."""
    other_repo = tmp_path / "main"
    other_repo.mkdir()
    _git(other_repo, "init", "-q")
    (other_repo / "file.txt").write_text("main checkout")
    _git(other_repo, "add", ".")
    _git(other_repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "x")
    cwd_sha = _git(other_repo, "rev-parse", "HEAD")

    from analyzer import price_snapshot

    package_dir = Path(price_snapshot.__file__).resolve().parent
    package_sha = _git(package_dir, "rev-parse", "HEAD")
    assert package_sha != cwd_sha

    monkeypatch.chdir(other_repo)
    assert price_snapshot._get_git_sha() == package_sha

    candidates, weights, quotes, watchlist = inputs
    path = receipts.record_decision(tmp_path, DAY, *inputs)
    assert json.loads(path.read_text())["git_sha"] == package_sha


def test_git_sha_fails_closed_or_uses_env(monkeypatch):
    from analyzer import price_snapshot

    def unresolvable(*a, **kw):
        raise FileNotFoundError("no git")

    monkeypatch.setattr(price_snapshot.subprocess, "run", unresolvable)
    monkeypatch.setenv("GIT_SHA", "abc123")
    assert price_snapshot._get_git_sha() == "abc123"
    monkeypatch.delenv("GIT_SHA")
    # Never a guess: unresolvable provenance is None, not a wrong value.
    assert price_snapshot._get_git_sha() is None
