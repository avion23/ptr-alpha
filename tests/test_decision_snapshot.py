from datetime import date
import json
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
        source_id="filing-123",
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
        source_id="13f-456",
        corroboration=True,
    )
    return (
        pd.DataFrame([row, fund]),
        {"congress:ALICE": 1.0},
        {"AAPL": (100.0, DAY)},
        {"manager:FUND A"},
    )


def test_record_idempotent_and_replay(tmp_path, inputs):
    candidates, weights, quotes, watchlist = inputs
    ranked, blocked = score(candidates, weights, quotes, watchlist=watchlist)
    path = receipts.record_decision(tmp_path, DAY, *inputs, ranked, blocked)
    original, stat = path.read_bytes(), path.stat()
    receipts.record_decision(tmp_path, DAY, *inputs, ranked, blocked)
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


def test_conflicting_score_preserves_original(tmp_path, inputs):
    candidates, weights, quotes, watchlist = inputs
    ranked, blocked = score(candidates, weights, quotes, watchlist=watchlist)
    path = receipts.record_decision(tmp_path, DAY, *inputs, ranked, blocked)
    original = path.read_bytes()
    ranked.loc[0, "score"] += 1
    with pytest.raises(receipts.DecisionConflict):
        receipts.record_decision(tmp_path, DAY, *inputs, ranked, blocked)
    with pytest.raises(receipts.DecisionConflict, match="decision conflict"):
        receipts.record_decision(
            tmp_path, DAY, candidates, {"congress:ALICE": 2.0}, quotes, watchlist
        )
    assert path.read_bytes() == original


def test_tampering_is_detected_even_with_recomputed_hash(tmp_path, inputs):
    path = receipts.record_decision(tmp_path, DAY, *inputs)
    manifest = json.loads(path.read_text())
    manifest["ranked"][0]["score"] += 1
    path.write_text(json.dumps(manifest))
    assert set(receipts.verify_decision(path)["mismatches"]) == {
        "manifest_hash",
        "ranked",
    }
    manifest.pop("manifest_hash")
    manifest["manifest_hash"] = receipts._hash(manifest)
    path.write_text(json.dumps(manifest))
    assert receipts.verify_decision(path) == {
        "agreement": False,
        "mismatches": ["ranked"],
    }


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
    path = tmp_path / "recommendations" / str(DAY) / "decision.json"
    assert receipts.verify_decision(path)["agreement"]
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
