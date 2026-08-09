"""Tests for the frozen, exactly-once, capital-constrained evaluation contract."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

import scripts.frozen_validation as fv
from analyzer.exceptions import DatabaseError
from analyzer.validation import (
    EvaluationAlreadyConsumedError,
    _canonical_ledger_path,
    _validate_ledger,
)

from tests.test_validation_harness import build_fixture_db


def _freeze(tmp_path: Path, monkeypatch, grid=None) -> Path:
    """Freeze a manifest to a temp path, optionally with a reduced grid."""
    if grid is not None:
        monkeypatch.setattr(fv, "GRID", grid)
    manifest_path = tmp_path / "manifest.json"
    fv.freeze_manifest(manifest_path)
    return manifest_path


def _single_trial_grid():
    return {
        "horizon": [60],
        "frequency_days": [30],
        "training_lookback_days": [365],
        "min_buyers": [2],
        "top_n": [5],
        "decay_lambda": [0.005],
        "bayes_prior_strength": [20],
        "scoring_mode": ["consensus"],
    }


class TestFreeze:
    def test_freeze_records_config_code_git_dependency_hashes(self, tmp_path, monkeypatch):
        manifest_path = _freeze(tmp_path, monkeypatch)
        manifest = json.loads(manifest_path.read_text())
        assert manifest["schema_version"] == 1
        assert (
            manifest["evidence_class"]
            == "retrospective_previously_used_not_fresh_oos"
        )
        assert manifest["verdict_policy"]["top_level_verdict"] == "not_established"
        assert manifest["phases"]["locked_final"]["consumed"] is False
        assert (
            manifest["config"]["phases"]["test"]["status"]
            == "retrospective_diagnostics_only"
        )
        hashes = manifest["hashes"]
        for key in (
            "config_sha256",
            "code_sha256",
            "harness_sha256",
            "git_revision",
            "git_diff_sha256",
            "dependency_sha256",
        ):
            assert hashes[key], f"missing frozen hash {key}"
        assert len(hashes["config_sha256"]) == 64
        assert len(hashes["code_sha256"]) == 64
        assert hashes["database_sha256"] is None
        assert hashes["value_snapshot_sha256"] is None
        assert manifest["data_hashes_recorded_at_evaluation"] is True
        assert manifest["config"]["scoring_modes"] == ["consensus"]
        assert manifest["config"]["grid"]["scoring_mode"] == ["consensus"]
        # The recorded config hash must match the frozen config payload.
        assert hashes["config_sha256"] == fv._sha256_json(fv.config_payload())

    def test_frozen_state_verifies_clean(self, tmp_path, monkeypatch):
        manifest_path = _freeze(tmp_path, monkeypatch, grid=_single_trial_grid())
        manifest = json.loads(manifest_path.read_text())
        ok, reasons = fv.verify_frozen_state(manifest)
        assert ok, reasons

    def test_verify_fails_closed_on_config_drift(self, tmp_path, monkeypatch):
        manifest_path = _freeze(tmp_path, monkeypatch)
        manifest = json.loads(manifest_path.read_text())
        manifest["config"]["alpha"] = 0.99
        ok, reasons = fv.verify_frozen_state(manifest)
        assert not ok
        assert any("config_sha256 mismatch" in reason for reason in reasons)

    def test_verify_fails_closed_on_code_drift(self, tmp_path, monkeypatch):
        manifest_path = _freeze(tmp_path, monkeypatch)
        manifest = json.loads(manifest_path.read_text())
        monkeypatch.setattr(
            "analyzer.validation._code_hash", lambda: "f" * 64
        )
        ok, reasons = fv.verify_frozen_state(manifest)
        assert not ok
        assert any("code_sha256 mismatch" in reason for reason in reasons)

    def test_verify_fails_closed_on_git_drift(self, tmp_path, monkeypatch):
        manifest_path = _freeze(tmp_path, monkeypatch)
        manifest = json.loads(manifest_path.read_text())
        monkeypatch.setattr(
            "analyzer.validation._git_state",
            lambda: {
                "revision": "deadbeef" * 5,
                "dirty": False,
                "diff_sha256": "0" * 64,
            },
        )
        ok, reasons = fv.verify_frozen_state(manifest)
        assert not ok
        assert any("git_revision mismatch" in reason for reason in reasons)

    def test_verify_fails_closed_on_dependency_drift(self, tmp_path, monkeypatch):
        manifest_path = _freeze(tmp_path, monkeypatch)
        manifest = json.loads(manifest_path.read_text())
        monkeypatch.setattr(
            "analyzer.validation._dependency_version", lambda name: "0.0.0-drifted"
        )
        ok, reasons = fv.verify_frozen_state(manifest)
        assert not ok
        assert any("dependency_sha256 mismatch" in reason for reason in reasons)


class TestEvaluate:
    def _fixture_db(self, tmp_path: Path) -> Path:
        # Transactions and prices must cover the frozen windows
        # (train 2022-01-01..2023-12-31, test 2024-01-01..2025-06-30).
        return build_fixture_db(
            tmp_path,
            tx_end=date(2025, 2, 1),
            price_end="2025-07-31",
        )

    def test_evaluate_runs_exactly_once_and_writes_report(
        self, tmp_path, monkeypatch
    ):
        db_path = self._fixture_db(tmp_path)
        manifest_path = _freeze(tmp_path, monkeypatch, grid=_single_trial_grid())
        report_path = tmp_path / "report.json"

        report = fv.evaluate_manifest(db_path, report_path, manifest_path)
        assert report["verdict"] == "not_established"
        assert (
            report["evidence_class"]
            == "retrospective_previously_used_not_fresh_oos"
        )
        assert report["verification"]["ok"] is True
        assert report_path.exists()
        # The validation output carries its own full runtime manifest with the
        # recorded data/value hashes.
        validation = report["validation"]
        assert validation["manifest"]["hashes"]["database_sha256"]
        assert validation["manifest"]["hashes"]["value_snapshot_sha256"]
        assert (
            validation["manifest"]["phases"]["test"]["evidence_class"]
            == "retrospective_previously_used_not_fresh_oos"
        )
        assert validation["manifest"]["phases"]["locked_final"]["consumed"] is False
        # No profitability wording anywhere in the report.
        for key, value in report.items():
            if isinstance(value, str):
                assert "profit" not in value.lower(), key
        # The ledger recorded exactly one reservation + completion.
        ledger_path = _canonical_ledger_path(db_path)
        ledger = json.loads(ledger_path.read_text())
        _validate_ledger(ledger)
        assert sum(e["event_type"] == "reservation" for e in ledger["events"]) == 1
        assert sum(e["event_type"] == "completion" for e in ledger["events"]) == 1

        # Exactly-once: a second evaluation of the same staged DB is refused.
        with pytest.raises(EvaluationAlreadyConsumedError):
            fv.evaluate_manifest(db_path, tmp_path / "report2.json", manifest_path)
        ledger_after = json.loads(ledger_path.read_text())
        _validate_ledger(ledger_after)
        assert (
            sum(e["event_type"] == "reservation" for e in ledger_after["events"]) == 1
        )

    def test_evaluate_refuses_state_mismatch_without_touching_ledger(
        self, tmp_path, monkeypatch
    ):
        db_path = self._fixture_db(tmp_path)
        manifest_path = _freeze(tmp_path, monkeypatch, grid=_single_trial_grid())
        manifest = json.loads(manifest_path.read_text())
        manifest["config"]["n_permutations"] = 42
        manifest_path.write_text(json.dumps(manifest))

        with pytest.raises(fv.FrozenStateMismatchError):
            fv.evaluate_manifest(db_path, tmp_path / "report.json", manifest_path)
        ledger_path = _canonical_ledger_path(db_path)
        assert not ledger_path.exists(), (
            "a refused evaluation must not write any ledger event"
        )

    def test_evaluate_requires_existing_staged_database(self, tmp_path, monkeypatch):
        manifest_path = _freeze(tmp_path, monkeypatch, grid=_single_trial_grid())
        with pytest.raises(FileNotFoundError, match="staged database not found"):
            fv.evaluate_manifest(tmp_path / "missing.duckdb", None, manifest_path)

    def test_evaluate_requires_transactions_in_frozen_window(
        self, tmp_path, monkeypatch
    ):
        manifest_path = _freeze(tmp_path, monkeypatch, grid=_single_trial_grid())
        empty_db = tmp_path / "empty.duckdb"
        empty_db.write_bytes(b"not a real database")
        with pytest.raises(DatabaseError):
            fv.evaluate_manifest(empty_db, None, manifest_path)


class TestCli:
    def test_freeze_command_writes_canonical_manifest(self, tmp_path, monkeypatch):
        target = tmp_path / "canonical.json"
        monkeypatch.setattr(fv, "FROZEN_MANIFEST_PATH", target)
        assert fv.main(["freeze"]) == 0
        assert target.exists()
        manifest = json.loads(target.read_text())
        assert manifest["evaluation"]["staged_db"]["status"] == (
            "awaiting_staging_confirmation_and_root_authorization"
        )
