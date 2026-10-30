import json
import logging
import multiprocessing
import os
import subprocess
import tempfile
import time
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from typer.testing import CliRunner

from analyzer import decision_snapshot as receipts
from analyzer.setups import score


DAY = date(2024, 6, 1)
_CONTROL = "enable_13f_corroboration"


def _derive_recipe(enabled=False):
    return {
        "scorer": {
            "parameters": {
                "entry_tol": 1.05,
                "event_half_life_days": 90,
                _CONTROL: enabled,
            }
        }
    }


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
        holding_state="held",
        shares_held=100,
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


def _publish_mutated_receipt(tmp_path, inputs, monkeypatch, mutate, *, enabled=False):
    path = receipts.record_decision(
        tmp_path,
        DAY,
        *inputs,
        enable_13f_corroboration=enabled,
    )
    manifest = json.loads(path.read_text())
    manifest.pop("manifest_hash")
    mutate(manifest)
    with monkeypatch.context() as patcher:
        patcher.setattr(receipts, "_verify_published", lambda receipt: receipt)
        return receipts._publish(tmp_path, manifest)


def test_record_idempotent_and_replay(tmp_path, inputs, monkeypatch):
    candidates, weights, quotes, watchlist = inputs
    monkeypatch.setenv("INSIDER_ENABLE_13F_CORROBORATION", "1")
    ranked, blocked = score(candidates, weights, quotes, watchlist=watchlist)
    path = receipts.record_decision(tmp_path, DAY, *inputs, ranked, blocked)
    original, stat = path.read_bytes(), path.stat()
    # Identical rerun: same content-addressed id, a true no-op.
    assert receipts.record_decision(tmp_path, DAY, *inputs, ranked, blocked) == path
    assert path.read_bytes() == original
    assert path.stat().st_mtime_ns == stat.st_mtime_ns
    verification = receipts.verify_decision(path)
    assert verification["agreement"]
    assert verification["integrity"] == {"agreement": True, "mismatches": []}
    assert verification["replay"] == {"status": "pass", "mismatches": []}
    manifest = json.loads(original)
    assert manifest["schema_version"] == 3
    assert manifest["recipe"]["scorer"] == {
        "name": "analyzer.setups.score",
        "version": "3",
        "parameters": {
            "entry_tol": 1.05,
            "event_half_life_days": 90,
            _CONTROL: True,
        },
    }
    assert manifest["recipe"]["parser"] == {
        "name": "analyzer.decision_snapshot",
        "version": "1",
    }
    assert manifest["recipe"]["frame"]["row_count"] == len(candidates)
    assert manifest["recipe"]["price_snapshot_id"] == manifest["price_snapshot"]["snapshot_id"]
    assert manifest["inputs"]["ex_dividends"] is None
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


@pytest.mark.parametrize("enabled", [False, True])
def test_v3_control_is_explicit_and_environment_independent(
    tmp_path, inputs, monkeypatch, enabled
):
    candidates, weights, quotes, watchlist = inputs
    results = []
    for environment in ("0" if enabled else "1", "1" if enabled else "0"):
        monkeypatch.setenv("INSIDER_ENABLE_13F_CORROBORATION", environment)
        audit = {}
        ranked, blocked = score(
            candidates,
            weights,
            quotes,
            watchlist=watchlist,
            enable_13f_corroboration=enabled,
            audit=audit,
        )
        path = receipts.record_decision(
            tmp_path,
            DAY,
            *inputs,
            ranked,
            blocked,
            enable_13f_corroboration=enabled,
        )
        results.append(
            (
                ranked.to_dict("records"),
                blocked.to_dict("records"),
                audit,
                path.read_bytes(),
                (tmp_path / "recommendations" / receipts.LEDGER_NAME).read_bytes(),
            )
        )
    assert results[0] == results[1]
    manifest = json.loads(_receipts(tmp_path)[0].read_text())
    assert manifest["recipe"]["scorer"]["parameters"][_CONTROL] is enabled


def test_record_resolves_omitted_control_once_and_rejects_nonbool(
    tmp_path, inputs, monkeypatch
):
    from analyzer import positions

    calls = []
    original = positions.is_13f_corroboration_enabled

    def capture(value):
        calls.append(value)
        return original(value)

    monkeypatch.setattr(positions, "is_13f_corroboration_enabled", capture)
    monkeypatch.setenv("INSIDER_ENABLE_13F_CORROBORATION", "1")
    path = receipts.record_decision(tmp_path, DAY, *inputs)

    assert calls == [None]
    assert json.loads(path.read_text())["recipe"]["scorer"]["parameters"][
        _CONTROL
    ] is True
    with pytest.raises(TypeError, match="must be a bool"):
        receipts.record_decision(
            tmp_path,
            DAY,
            *inputs,
            enable_13f_corroboration=1,
        )
    assert calls == [None, 1]
    assert len(_receipts(tmp_path)) == 1


def test_record_rejects_frozen_outputs_from_a_different_control(inputs, tmp_path):
    candidates, weights, quotes, watchlist = inputs
    ranked, blocked = score(
        candidates,
        weights,
        quotes,
        watchlist=watchlist,
        enable_13f_corroboration=False,
    )

    with pytest.raises(receipts.DecisionConflict, match="do not reproduce"):
        receipts.record_decision(
            tmp_path,
            DAY,
            *inputs,
            ranked,
            blocked,
            enable_13f_corroboration=True,
        )
    assert not _receipts(tmp_path)


@pytest.mark.parametrize("provided", ["ranked", "blocked"])
def test_record_rejects_half_output_pair_before_publication(tmp_path, inputs, provided):
    output_pair = {"ranked": None, "blocked": None}
    output_pair[provided] = pd.DataFrame()

    with pytest.raises(ValueError, match="ranked and blocked must be provided together"):
        receipts.record_decision(
            tmp_path,
            DAY,
            *inputs,
            output_pair["ranked"],
            output_pair["blocked"],
        )

    assert list(tmp_path.iterdir()) == []
    assert not (tmp_path / "recommendations" / receipts.LEDGER_NAME).exists()


@pytest.mark.parametrize("invalid", ["missing", None, 1, "false"])
def test_derive_requires_recipe_control_before_calling_score(
    inputs, monkeypatch, invalid
):
    candidates, weights, quotes, watchlist = inputs
    frozen = receipts._freeze(
        {
            "candidates": receipts._frame(candidates),
            "weights": weights,
            "quotes": quotes,
            "watchlist": sorted(watchlist),
        }
    )
    recipe = _derive_recipe()
    if invalid == "missing":
        recipe["scorer"]["parameters"].pop(_CONTROL)
    else:
        recipe["scorer"]["parameters"][_CONTROL] = invalid
    monkeypatch.setattr(receipts, "score", lambda *a, **k: pytest.fail("score called"))

    with pytest.raises(ValueError, match=_CONTROL):
        receipts._derive(frozen, recipe)


def test_dividend_input_is_frozen_and_replayed_exactly(tmp_path, inputs):
    candidates, weights, quotes, watchlist = inputs
    dividends = {"AAPL": [date(2024, 5, 30)]}
    ranked, blocked = score(
        candidates, weights, quotes, watchlist=watchlist, ex_dividends=dividends
    )
    path = receipts.record_decision(
        tmp_path,
        DAY,
        *inputs,
        ranked,
        blocked,
        ex_dividends=dividends,
        dividend_source="test.dividends",
    )
    manifest = json.loads(path.read_text())
    assert manifest["inputs"]["ex_dividends"] == {
        "AAPL": ["2024-05-30"]
    }
    assert manifest["recipe"]["dividends"] == {
        "source": "test.dividends",
        "input_hash": receipts._hash(manifest["inputs"]["ex_dividends"]),
    }
    assert manifest["ranked"] == []
    assert manifest["blocked"]

    recorded = receipts.verify_decision(path)
    assert recorded["agreement"]
    assert recorded["replay"]["status"] == "pass"
    # Diagnostic replay against a mutated live dividend snapshot disagrees,
    # while the default replay remains anchored to the published input above.
    mutated = receipts.verify_decision(path, current_dividends={})
    assert mutated["integrity"]["agreement"]
    assert mutated["agreement"] is False
    assert mutated["replay"]["status"] == "fail"
    assert {"ranked", "blocked"}.intersection(mutated["replay"]["mismatches"])
    assert receipts.verify_decision(path)["replay"]["status"] == "pass"


def test_historical_receipt_is_integrity_only_and_not_rewritten(tmp_path, inputs):
    candidates, weights, quotes, watchlist = inputs
    frozen = receipts._freeze(
        {
            "candidates": receipts._frame(candidates),
            "weights": weights,
            "quotes": quotes,
            "watchlist": sorted(watchlist),
        }
    )
    outputs = receipts._derive(frozen, _derive_recipe())
    if outputs["ranked"]:
        outputs["ranked"][0]["reasons"] = ["reason text as originally published"]
    manifest = {
        "schema_version": 2,
        "kind": "setups",
        "decision_date": DAY.isoformat(),
        "git_sha": None,
        "scope": {},
        "inputs": frozen,
        "input_rows_hash": receipts._hash(frozen["candidates"]),
        "price_snapshot": receipts._price_reference(frozen, DAY),
        **outputs,
    }
    path = receipts._publish(tmp_path, manifest)
    historical_bytes = path.read_bytes()
    result = receipts.verify_decision(path)
    assert result["agreement"]
    assert result["integrity"] == {"agreement": True, "mismatches": []}
    assert result["replay"] == {"status": "pre-recipe anchor", "mismatches": []}
    assert path.read_bytes() == historical_bytes


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


# ---------------------------------------------------------------------------
# Reviewer reproduction: concurrent writers must not read the same ledger tail
# ---------------------------------------------------------------------------


def _concurrent_publisher(data_dir, start, result_path, writer):
    """Child body: publish one distinct decision once both writers are ready."""
    try:
        start.wait(timeout=30)
        receipts.record_ticker_decision(
            data_dir,
            DAY,
            {"writer": writer},
            {"verdict": writer},
            scope={"writer": writer},
        )
    except Exception as exc:
        Path(result_path).write_text(f"{type(exc).__name__}: {exc}")
    else:
        Path(result_path).write_text("published")


def test_concurrent_publications_serialize_the_ledger(tmp_path, monkeypatch):
    """Two writers racing one ledger must not both append to the same tail.

    O_APPEND serializes byte placement, not predecessor selection: without the
    interprocess lock both children read the same tail, both link to it, and
    one publication fails verification on a broken chain. The ledger load is
    slowed here so the overlap the unlocked read-append cycle lost to luck
    becomes deterministic.
    """
    original_load = receipts._load_ledger

    def slow_load(recommendations):
        time.sleep(0.3)
        return original_load(recommendations)

    context = multiprocessing.get_context("fork")
    start = context.Barrier(2, timeout=30)
    processes = [
        context.Process(
            target=_concurrent_publisher,
            args=(tmp_path, start, str(tmp_path / f"writer-{i}.txt"), writer),
        )
        for i, writer in enumerate(("alpha", "beta"))
    ]
    with monkeypatch.context() as patcher:
        patcher.setattr(receipts, "_load_ledger", slow_load)
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=120)
    assert [process.exitcode for process in processes] == [0, 0]
    outcomes = sorted(
        (tmp_path / f"writer-{i}.txt").read_text() for i in range(len(processes))
    )
    # Neither publication failed: the race that broke the chain is gone.
    assert outcomes == ["published", "published"]

    recommendations = tmp_path / "recommendations"
    assert receipts._ledger_lock_path(recommendations).exists()
    paths = _receipts(tmp_path)
    assert len(paths) == 2
    for path in paths:
        assert receipts.verify_decision(path)["agreement"], path
    lines = (recommendations / receipts.LEDGER_NAME).read_text().splitlines()
    assert len(lines) == 2
    entries = [json.loads(line) for line in lines]
    assert entries[0]["prev_line_hash"] == receipts.GENESIS_HASH
    assert entries[1]["prev_line_hash"] == receipts._line_hash(entries[0])
    # Every receipt is anchored by exactly its own ledger line.
    assert {entry["decision_id"] for entry in entries} == {
        path.stem for path in paths
    }
    # The lock is released, so a later sequential publication still works.
    third = receipts.record_ticker_decision(
        tmp_path,
        DAY,
        {"writer": "gamma"},
        {"verdict": "gamma"},
        scope={"writer": "gamma"},
    )
    assert receipts.verify_decision(third)["agreement"]
    assert len((recommendations / receipts.LEDGER_NAME).read_text().splitlines()) == 3


def _attacker_write(path, text):
    """An owner can chmod the 0444 receipt back to writable, then edit it."""
    os.chmod(path, 0o644)
    path.write_text(text)


@pytest.mark.parametrize("version", ["1", "2"])
@pytest.mark.parametrize(
    ("control_state", "base_enabled"),
    [
        ("absent", False),
        ("null", False),
        ("nonboolean", False),
        ("explicit-false", False),
        ("contradictory", True),
    ],
)
@pytest.mark.parametrize("environment", ["0", "1"])
def test_v1_v2_control_metadata_is_unsupported_without_derive(
    tmp_path, inputs, monkeypatch, version, control_state, base_enabled, environment
):
    def mutate(manifest):
        scorer = manifest["recipe"]["scorer"]
        scorer["version"] = version
        parameters = scorer["parameters"]
        if control_state == "absent":
            parameters.pop(_CONTROL)
        elif control_state == "null":
            parameters[_CONTROL] = None
        elif control_state == "nonboolean":
            parameters[_CONTROL] = "true"
        elif control_state == "contradictory":
            parameters[_CONTROL] = False

    path = _publish_mutated_receipt(
        tmp_path, inputs, monkeypatch, mutate, enabled=base_enabled
    )
    monkeypatch.setenv("INSIDER_ENABLE_13F_CORROBORATION", environment)
    monkeypatch.setattr(
        receipts, "_derive", lambda *a, **k: pytest.fail("legacy derive called")
    )

    result = receipts.verify_decision(path)

    assert result["integrity"] == {"agreement": True, "mismatches": []}
    assert result["agreement"] is False
    assert result["replay"]["status"] == "unsupported-version"
    assert "recipe.scorer_version" in result["replay"]["mismatches"]


@pytest.mark.parametrize("version", ["1", "2"])
@pytest.mark.parametrize("audit_state", ["missing", "malformed", "conflicting"])
def test_v1_v2_audit_variants_are_unsupported_without_derive(
    tmp_path, inputs, monkeypatch, version, audit_state
):
    def mutate(manifest):
        manifest["recipe"]["scorer"]["version"] = version
        if audit_state == "missing":
            manifest.pop("eligibility")
        elif audit_state == "malformed":
            manifest["eligibility"] = "not-an-audit-map"
        else:
            manifest["eligibility"] = {
                "AAPL": {"corroborated_13f": True, "ranking_gate": False}
            }

    path = _publish_mutated_receipt(tmp_path, inputs, monkeypatch, mutate)
    monkeypatch.setenv("INSIDER_ENABLE_13F_CORROBORATION", "1")
    monkeypatch.setattr(
        receipts, "_derive", lambda *a, **k: pytest.fail("legacy derive called")
    )

    result = receipts.verify_decision(path)

    assert result["integrity"]["agreement"] is True
    assert result["replay"]["status"] == "unsupported-version"
    assert "recipe.scorer_version" in result["replay"]["mismatches"]


@pytest.mark.parametrize(
    ("component", "field", "value", "mismatch"),
    [
        ("scorer", "name", "other.scorer", "recipe.scorer_version"),
        ("scorer", "version", "2", "recipe.scorer_version"),
        ("scorer", "version", ["3"], "recipe.scorer_version"),
        ("parser", "name", "other.parser", "recipe.parser_version"),
        ("parser", "version", "2", "recipe.parser_version"),
    ],
)
def test_unsupported_recipe_identity_stops_before_derive(
    tmp_path, inputs, monkeypatch, component, field, value, mismatch
):
    def mutate(manifest):
        manifest["recipe"][component][field] = value

    path = _publish_mutated_receipt(tmp_path, inputs, monkeypatch, mutate)
    monkeypatch.setattr(
        receipts, "_derive", lambda *a, **k: pytest.fail("unsupported derive called")
    )

    result = receipts.verify_decision(path)

    assert result["integrity"]["agreement"] is True
    assert result["replay"]["status"] == "unsupported-version"
    assert mismatch in result["replay"]["mismatches"]


@pytest.mark.parametrize("bad_control", ["missing", None, 1, "false"])
@pytest.mark.parametrize("environment", ["0", "1"])
def test_v3_malformed_control_fails_before_score(
    tmp_path, inputs, monkeypatch, bad_control, environment
):
    def mutate(manifest):
        parameters = manifest["recipe"]["scorer"]["parameters"]
        if bad_control == "missing":
            parameters.pop(_CONTROL)
        else:
            parameters[_CONTROL] = bad_control

    path = _publish_mutated_receipt(tmp_path, inputs, monkeypatch, mutate)
    monkeypatch.setenv("INSIDER_ENABLE_13F_CORROBORATION", environment)
    monkeypatch.setattr(receipts, "score", lambda *a, **k: pytest.fail("score called"))

    result = receipts.verify_decision(path)

    assert result["integrity"]["agreement"] is True
    assert result["agreement"] is False
    assert result["replay"]["status"] == "fail"
    assert any(_CONTROL in item for item in result["replay"]["mismatches"])


def test_integrity_failure_precedes_unsupported_recipe_and_skips_derive(
    tmp_path, inputs, monkeypatch
):
    path = receipts.record_decision(
        tmp_path, DAY, *inputs, enable_13f_corroboration=False
    )
    manifest = json.loads(path.read_text())
    manifest["recipe"]["scorer"]["version"] = "1"
    _attacker_write(path, json.dumps(manifest))
    monkeypatch.setattr(
        receipts, "_derive", lambda *a, **k: pytest.fail("integrity-invalid derive called")
    )

    result = receipts.verify_decision(path)

    assert result["integrity"]["agreement"] is False
    assert "manifest_hash" in result["integrity"]["mismatches"]
    assert result["replay"]["status"] == "not-run"
    assert "recipe.scorer_version" not in result["replay"]["mismatches"]


def test_frozen_anchor_failure_stops_before_derive(tmp_path, inputs, monkeypatch):
    def mutate(manifest):
        manifest["recipe"]["frame"]["input_hash"] = "0" * 64

    path = _publish_mutated_receipt(tmp_path, inputs, monkeypatch, mutate)
    monkeypatch.setattr(
        receipts, "_derive", lambda *a, **k: pytest.fail("bad-anchor derive called")
    )

    result = receipts.verify_decision(path)

    assert result["integrity"]["agreement"] is True
    assert result["replay"]["status"] == "fail"
    assert "recipe.frame" in result["replay"]["mismatches"]


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
    manifest.update(receipts._derive(manifest["inputs"], manifest["recipe"]))
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
        with pytest.raises(receipts.LedgerError):
            receipts.record_decision_safely(tmp_path, DAY, *inputs)
    assert "Decision receipt was not recorded" in caplog.text
    assert len(ledger.read_text().splitlines()) == 1  # nothing appended


def test_tampering_is_detected_even_with_recomputed_hash(tmp_path, inputs):
    path = receipts.record_decision(tmp_path, DAY, *inputs)
    manifest = json.loads(path.read_text())
    manifest["ranked"][0]["score"] += 1
    _attacker_write(path, json.dumps(manifest))
    # The stale digest still matches its ledger entry, but not the content.
    stale = receipts.verify_decision(path)
    assert stale["agreement"] is False
    assert stale["integrity"]["mismatches"] == ["manifest_hash"]
    assert stale["replay"]["status"] == "not-run"
    manifest.pop("manifest_hash")
    manifest["manifest_hash"] = receipts._hash(manifest)
    _attacker_write(path, json.dumps(manifest))
    # The recomputed digest matches no ledger entry: fail closed.
    unanchored = receipts.verify_decision(path)
    assert unanchored["agreement"] is False
    assert "receipt_not_in_ledger" in unanchored["integrity"]["mismatches"]
    assert unanchored["replay"]["status"] == "not-run"


def test_receipt_files_are_read_only_after_publish(tmp_path, inputs):
    path = receipts.record_decision(tmp_path, DAY, *inputs)
    assert path.stat().st_mode & 0o777 == 0o444


def test_cli_recording_is_byte_identical_and_failure_is_terminal(
    tmp_path, monkeypatch, inputs, caplog
):
    from analyzer import cli, pipeline, positions

    candidates, weights, quotes, watchlist = inputs
    connection = SimpleNamespace(
        execute=lambda *a, **k: SimpleNamespace(fetchone=lambda: (100.0, DAY), fetchdf=lambda: pd.DataFrame([{"ticker": "AAPL", "close": 100.0, "date": pd.Timestamp(DAY)}]), fetchall=lambda: [])
    )
    monkeypatch.setattr(
        cli,
        "Database",
        lambda *a, **kw: SimpleNamespace(conn=connection, close=lambda: None),
    )
    monkeypatch.setattr(pipeline, "eligible_events", lambda *a: candidates)
    monkeypatch.setattr(
        positions, "holdings_candidates", lambda *a, **kw: pd.DataFrame()
    )
    monkeypatch.setattr(
        positions, "holdings_candidates_batch", lambda *a, **kw: {}
    )
    monkeypatch.setattr(positions, "coalesce_actor_variants", lambda events, _: events)
    monkeypatch.setattr(
        positions, "gate_closed_events", lambda events, _, **kwargs: events
    )
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
        assert failed_write.exit_code == 1
        assert "Error: setup ranking failed: disk full" in failed_write.output
        assert "=== Ranked setups ===" not in failed_write.output
        assert "disk full" in caplog.text


@pytest.mark.parametrize("enabled", [False, True])
def test_cli_setups_passes_one_resolved_control_to_every_normal_path(
    tmp_path, monkeypatch, inputs, enabled
):
    from analyzer import cli, manager_watchlist, pipeline, positions, setups

    candidates, weights, quotes, watchlist = inputs
    resolutions = []
    calls = []
    resolve = positions.is_13f_corroboration_enabled

    def capture_resolution(value):
        resolutions.append(value)
        return resolve(value)

    monkeypatch.setattr(positions, "is_13f_corroboration_enabled", capture_resolution)
    monkeypatch.setenv("INSIDER_ENABLE_13F_CORROBORATION", "1" if enabled else "0")
    connection = SimpleNamespace(
        execute=lambda *a, **k: SimpleNamespace(
            fetchone=lambda: (100.0, DAY),
            fetchdf=lambda: pd.DataFrame(
                [{"ticker": "AAPL", "close": 100.0, "date": pd.Timestamp(DAY)}]
            ),
            fetchall=lambda: [],
        )
    )
    monkeypatch.setattr(
        cli,
        "Database",
        lambda *a, **kw: SimpleNamespace(conn=connection, close=lambda: None),
    )
    monkeypatch.setattr(pipeline, "eligible_events", lambda *a: candidates.copy())

    def selected(*args, **kwargs):
        calls.append(("selected", kwargs["enable_13f_corroboration"]))
        return pd.DataFrame()

    def batched(*args, **kwargs):
        calls.append(("batched", kwargs["enable_13f_corroboration"]))
        return {}

    gate = positions.gate_closed_events

    def gated(*args, **kwargs):
        calls.append(("gate", kwargs["enable_13f_corroboration"]))
        return gate(*args, **kwargs)

    scorer = setups.score

    def scored(*args, **kwargs):
        calls.append(("score", kwargs["enable_13f_corroboration"]))
        return scorer(*args, **kwargs)

    recorder = receipts.record_decision_safely

    def recorded(*args, **kwargs):
        calls.append(("receipt", kwargs["enable_13f_corroboration"]))
        return recorder(*args, **kwargs)

    monkeypatch.setattr(positions, "holdings_candidates", selected)
    monkeypatch.setattr(positions, "holdings_candidates_batch", batched)
    monkeypatch.setattr(positions, "gate_closed_events", gated)
    monkeypatch.setattr(setups, "score", scored)
    monkeypatch.setattr(receipts, "record_decision_safely", recorded)
    monkeypatch.setattr(manager_watchlist, "watchlist_actor_ids", lambda: watchlist)

    result = CliRunner().invoke(
        cli.app,
        [
            "setups",
            "--member",
            "Alice",
            "--ticker",
            "AAPL",
            "--as-of",
            DAY.isoformat(),
            "--data-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls == [(name, enabled) for name in (
        "selected", "batched", "gate", "score", "receipt"
    )]
    assert resolutions.count(None) == 1
    assert resolutions == [None, enabled, enabled]
    manifest = json.loads(_receipts(tmp_path)[0].read_text())
    assert manifest["recipe"]["scorer"]["parameters"][_CONTROL] is enabled


@pytest.mark.parametrize("branch", ["no-frames", "filtered-empty"])
@pytest.mark.parametrize("enabled", [False, True])
def test_cli_empty_setups_paths_record_resolved_control(
    tmp_path, monkeypatch, branch, enabled
):
    from analyzer import cli, manager_watchlist, pipeline, positions

    resolutions = []
    calls = []
    resolve = positions.is_13f_corroboration_enabled

    def capture_resolution(value):
        resolutions.append(value)
        return resolve(value)

    monkeypatch.setattr(positions, "is_13f_corroboration_enabled", capture_resolution)
    monkeypatch.setenv("INSIDER_ENABLE_13F_CORROBORATION", "1" if enabled else "0")
    connection = SimpleNamespace(
        execute=lambda *a, **k: SimpleNamespace(
            fetchone=lambda: (100.0, DAY),
            fetchdf=lambda: pd.DataFrame(),
            fetchall=lambda: [],
        )
    )
    monkeypatch.setattr(
        cli,
        "Database",
        lambda *a, **kw: SimpleNamespace(conn=connection, close=lambda: None),
    )
    monkeypatch.setattr(pipeline, "eligible_events", lambda *a: pd.DataFrame())

    candidate = pd.DataFrame(
        [
            {
                "ticker": "MSFT",
                "actor_id": "congress:ALICE",
                "kind": "congress",
                "source": "house_pdf",
                "entry_ref": 100.0,
                "event_date": DAY,
                "disclosure_date": DAY,
                "corroboration": False,
                "position_evidence": True,
                "as_of": DAY,
                "filing_refs": [],
            }
        ]
    )

    def selected(*args, **kwargs):
        calls.append(("selected", kwargs["enable_13f_corroboration"]))
        return candidate if branch == "filtered-empty" else pd.DataFrame()

    recorder = receipts.record_decision_safely

    def recorded(*args, **kwargs):
        calls.append(("receipt", kwargs["enable_13f_corroboration"]))
        return recorder(*args, **kwargs)

    monkeypatch.setattr(positions, "holdings_candidates", selected)
    monkeypatch.setattr(receipts, "record_decision_safely", recorded)
    monkeypatch.setattr(manager_watchlist, "watchlist_actor_ids", lambda: set())

    result = CliRunner().invoke(
        cli.app,
        [
            "setups",
            "--member",
            "Alice",
            "--ticker",
            "AAPL",
            "--as-of",
            DAY.isoformat(),
            "--data-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls == [("selected", enabled), ("receipt", enabled)]
    assert resolutions == [None, enabled]
    manifest = json.loads(_receipts(tmp_path)[0].read_text())
    assert manifest["recipe"]["scorer"]["parameters"][_CONTROL] is enabled


def test_unscoped_setups_keep_the_full_event_universe(
    tmp_path, monkeypatch, inputs
):
    """Regression: with no --ticker/--member, `setups` used to discard every
    purchase event (cli.py `events.iloc[0:0]`), silently serving an empty
    ranking that contradicted its docstring ("purchase events, all official
    sources"). The unscoped universe must survive to scoring."""
    from analyzer import cli, pipeline, positions
    from analyzer import manager_watchlist

    candidates, weights, quotes, watchlist = inputs
    connection = SimpleNamespace(
        execute=lambda *a, **k: SimpleNamespace(fetchone=lambda: (100.0, DAY), fetchdf=lambda: pd.DataFrame([{"ticker": "AAPL", "close": 100.0, "date": pd.Timestamp(DAY)}]), fetchall=lambda: [])
    )
    monkeypatch.setattr(
        cli,
        "Database",
        lambda *a, **kw: SimpleNamespace(conn=connection, close=lambda: None),
    )
    monkeypatch.setattr(pipeline, "eligible_events", lambda *a: candidates)
    monkeypatch.setattr(
        positions, "holdings_candidates", lambda *a, **kw: pd.DataFrame()
    )
    monkeypatch.setattr(
        positions, "holdings_candidates_batch", lambda *a, **kw: {}
    )
    monkeypatch.setattr(positions, "coalesce_actor_variants", lambda events, _: events)
    monkeypatch.setattr(
        positions, "gate_closed_events", lambda events, _, **kwargs: events
    )
    monkeypatch.setattr(cli, "weights_for_candidates", lambda *a: weights)
    monkeypatch.setattr(manager_watchlist, "watchlist_actor_ids", lambda: watchlist)

    result = CliRunner().invoke(
        cli.app,
        ["setups", "--as-of", DAY.isoformat(), "--data-dir", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output
    receipts_written = _receipts(tmp_path)
    assert len(receipts_written) == 1
    manifest = json.loads(receipts_written[0].read_text())
    # The unscoped universe reached scoring: candidate rows and a ranked
    # winner are present, and the receipt verifies.
    assert len(manifest["inputs"]["candidates"]["rows"]) == len(candidates)
    assert manifest["ranked"] and manifest["ranked"][0]["ticker"] == "AAPL"
    assert receipts.verify_decision(receipts_written[0])["agreement"]


def test_scoped_setups_with_no_holdings_return_empty_not_the_universe(
    tmp_path, monkeypatch, inputs
):
    """Counterpart: an explicitly-scoped request (--member) that resolves to
    no holdings must return an empty decision, NOT silently expand to the
    full event universe (a silent-wrong-answer of the same class)."""
    from analyzer import cli, pipeline, positions
    from analyzer import manager_watchlist

    candidates, weights, quotes, watchlist = inputs
    connection = SimpleNamespace(
        execute=lambda *a, **k: SimpleNamespace(fetchone=lambda: (100.0, DAY), fetchdf=lambda: pd.DataFrame([{"ticker": "AAPL", "close": 100.0, "date": pd.Timestamp(DAY)}]), fetchall=lambda: [])
    )
    monkeypatch.setattr(
        cli,
        "Database",
        lambda *a, **kw: SimpleNamespace(conn=connection, close=lambda: None),
    )
    monkeypatch.setattr(pipeline, "eligible_events", lambda *a: candidates)
    monkeypatch.setattr(
        positions, "holdings_candidates", lambda *a, **kw: pd.DataFrame()
    )
    monkeypatch.setattr(
        positions, "holdings_candidates_batch", lambda *a, **kw: {}
    )
    monkeypatch.setattr(positions, "coalesce_actor_variants", lambda events, _: events)
    monkeypatch.setattr(
        positions, "gate_closed_events", lambda events, _, **kwargs: events
    )
    monkeypatch.setattr(cli, "weights_for_candidates", lambda *a: weights)
    monkeypatch.setattr(manager_watchlist, "watchlist_actor_ids", lambda: watchlist)

    result = CliRunner().invoke(
        cli.app,
        [
            "setups",
            "--member",
            "Bogus Member",
            "--as-of",
            DAY.isoformat(),
            "--data-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    manifest = json.loads(_receipts(tmp_path)[0].read_text())
    assert manifest["ranked"] == []
    assert manifest["inputs"]["candidates"]["rows"] == []


def test_no_receipt_flag_writes_nothing_but_keeps_output(
    tmp_path, monkeypatch, inputs
):
    """--no-receipt is how the suite avoids polluting the production oracle:
    byte-identical output, zero receipt files, zero ledger lines."""
    from analyzer import cli, pipeline, positions
    from analyzer import manager_watchlist

    candidates, weights, quotes, watchlist = inputs
    connection = SimpleNamespace(
        execute=lambda *a, **k: SimpleNamespace(fetchone=lambda: (100.0, DAY), fetchdf=lambda: pd.DataFrame([{"ticker": "AAPL", "close": 100.0, "date": pd.Timestamp(DAY)}]), fetchall=lambda: [])
    )
    monkeypatch.setattr(
        cli,
        "Database",
        lambda *a, **kw: SimpleNamespace(conn=connection, close=lambda: None),
    )
    monkeypatch.setattr(pipeline, "eligible_events", lambda *a: candidates)
    monkeypatch.setattr(
        positions, "holdings_candidates", lambda *a, **kw: pd.DataFrame()
    )
    monkeypatch.setattr(
        positions, "holdings_candidates_batch", lambda *a, **kw: {}
    )
    monkeypatch.setattr(positions, "coalesce_actor_variants", lambda events, _: events)
    monkeypatch.setattr(
        positions, "gate_closed_events", lambda events, _, **kwargs: events
    )
    monkeypatch.setattr(cli, "weights_for_candidates", lambda *a: weights)
    monkeypatch.setattr(manager_watchlist, "watchlist_actor_ids", lambda: watchlist)

    base = [
        "setups",
        "--ticker",
        "AAPL",
        "--as-of",
        DAY.isoformat(),
        "--data-dir",
        str(tmp_path),
    ]
    with_receipt = CliRunner().invoke(cli.app, base)
    assert with_receipt.exit_code == 0, with_receipt.output
    assert len(_receipts(tmp_path)) == 1
    without_receipt = CliRunner().invoke(cli.app, base + ["--no-receipt"])
    assert without_receipt.exit_code == 0, without_receipt.output
    assert without_receipt.output == with_receipt.output
    assert len(_receipts(tmp_path)) == 1  # nothing new written


def test_write_failure_logged_and_propagated(tmp_path, inputs, monkeypatch, caplog):
    def fail(*a, **kw):
        raise OSError("disk full")

    monkeypatch.setattr(receipts.os, "link", fail)
    with pytest.raises(OSError, match="disk full"):
        receipts.record_decision_safely(tmp_path, DAY, *inputs)
    assert "Decision receipt was not recorded" in caplog.text
    assert "disk full" in caplog.text
    assert not list(tmp_path.rglob(".decision-*"))


def test_decision_conflict_is_terminal_in_safe_wrapper(monkeypatch, tmp_path, inputs):
    def conflict(*a, **kw):
        raise receipts.DecisionConflict("receipt collision")

    monkeypatch.setattr(receipts, "record_decision", conflict)
    with pytest.raises(receipts.DecisionConflict, match="receipt collision"):
        receipts.record_decision_safely(tmp_path, DAY, *inputs)


def test_empty_decision_is_recorded(tmp_path):
    path = receipts.record_decision(tmp_path, DAY, pd.DataFrame(), {}, {}, set())
    assert json.loads(path.read_text())["ranked"] == []
    assert receipts.verify_decision(path)["agreement"]


# ---------------------------------------------------------------------------
# Reviewer reproduction: input filing identities must survive into receipts
# ---------------------------------------------------------------------------


def test_eligible_events_preserve_filing_identity():
    from unittest.mock import patch

    from analyzer.persistence.database import Database
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

    with Database(":memory:") as db:
        with patch.object(
            db, "get_transactions_by_date_range", return_value=trades("filing-A")
        ):
            events = eligible_events(db, pd.Timestamp("2025-06-01"), days_back=None)
        with patch.object(
            db, "get_transactions_by_date_range", return_value=trades("filing-B")
        ):
            events_b = eligible_events(db, pd.Timestamp("2025-06-01"), days_back=None)

    assert events["source_record_id"].tolist() == ["filing-A"]
    assert events["source_row_id"].tolist() == ["row-filing-A"]
    # Old behavior: frames byte-identical despite different filing identity.
    assert not events.equals(events_b)
    assert events_b["source_record_id"].tolist() == ["filing-B"]


def test_legacy_producer_without_ids_records_explicit_null():
    from unittest.mock import patch

    from analyzer.persistence.database import Database
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
    with Database(":memory:") as db:
        with patch.object(db, "get_transactions_by_date_range", return_value=trades):
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


def test_analyze_receipt_failure_is_terminal(analyze_ctx, tmp_path, monkeypatch, caplog):
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
    assert result.exit_code != 0
    assert "Recommendation:" not in result.output
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
    assert price_snapshot.get_git_sha() == package_sha

    candidates, weights, quotes, watchlist = inputs
    path = receipts.record_decision(tmp_path, DAY, *inputs)
    assert json.loads(path.read_text())["git_sha"] == package_sha


def test_git_sha_fails_closed_or_uses_env(monkeypatch):
    from analyzer import price_snapshot

    def unresolvable(*a, **kw):
        raise FileNotFoundError("no git")

    monkeypatch.setattr(price_snapshot.subprocess, "run", unresolvable)
    monkeypatch.setenv("GIT_SHA", "abc123")
    assert price_snapshot.get_git_sha() == "abc123"
    monkeypatch.delenv("GIT_SHA")
    # Never a guess: unresolvable provenance is None, not a wrong value.
    assert price_snapshot.get_git_sha() is None
