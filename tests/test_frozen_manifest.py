"""Behavioral tests for predeclared retrospective validation."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

import scripts.frozen_validation as fv
from analyzer.database import DatabaseError
from analyzer.member_ranking.buyer_scoring import CONSENSUS_LOOKBACK_DAYS

from tests.test_validation_harness import build_fixture_db


def _freeze(tmp_path: Path, monkeypatch, grid=None) -> Path:
    if grid is not None:
        monkeypatch.setattr(fv, "GRID", grid)
    manifest_path = tmp_path / "manifest.json"
    fv.freeze_manifest(manifest_path)
    return manifest_path


def _single_trial_grid() -> dict[str, list]:
    return {
        "horizon": [60],
        "frequency_days": [30],
        "lookback_days": [CONSENSUS_LOOKBACK_DAYS],
        "min_buyers": [2],
        "top_n": [5],
    }


class TestFreeze:
    def test_freeze_records_only_predeclared_experiment_inputs(
        self, tmp_path, monkeypatch
    ):
        manifest_path = _freeze(tmp_path, monkeypatch)
        manifest = json.loads(manifest_path.read_text())

        assert manifest["schema_version"] == 2
        assert (
            manifest["evidence_class"]
            == "retrospective_previously_used_not_fresh_oos"
        )
        config = manifest["config"]
        assert config["phases"]["locked_final"]["status"] == (
            "locked_not_queried_or_evaluated"
        )
        assert config["phases"]["test"]["status"] == "retrospective_diagnostics_only"
        assert config["grid"]["lookback_days"] == [CONSENSUS_LOOKBACK_DAYS]
        assert "scoring_mode" not in config["grid"]
        for irrelevant in (
            "training_lookback_days",
            "decay_lambda",
            "bayes_prior_strength",
            "threshold",
        ):
            assert irrelevant not in config["grid"]
        assert "hashes" not in manifest
        assert "evaluation" not in manifest

    def test_grid_variant_is_recorded_before_evaluation(self, tmp_path):
        grid = _single_trial_grid()
        grid["min_buyers"] = [1]
        manifest = fv.freeze_manifest(
            tmp_path / "variant.json",
            grid=grid,
            grid_decision="sparse official-source diagnostic",
        )
        assert manifest["config"]["grid"]["min_buyers"] == [1]
        assert manifest["config"]["grid_decision"] == (
            "sparse official-source diagnostic"
        )

    def test_manifest_rejects_invalid_schema(self):
        with pytest.raises(fv.FrozenManifestError, match="schema"):
            fv._manifest_config({"schema_version": 1, "config": {}})

    def test_manifest_rejects_test_window_inside_locked_final(self):
        manifest = fv.freeze_manifest(Path("/tmp/unused-predeclared-manifest.json"))
        manifest["config"]["phases"]["test"]["boundary"] = [
            "2025-12-01",
            "2026-02-01",
        ]
        with pytest.raises(fv.FrozenManifestError, match="locked final phase"):
            fv._manifest_config(manifest)


class TestEvaluate:
    def _fixture_db(self, tmp_path: Path) -> Path:
        return build_fixture_db(
            tmp_path,
            tx_end=date(2025, 2, 1),
            price_end="2025-07-31",
        )

    def test_evaluate_runs_real_validation_and_writes_report(
        self, tmp_path, monkeypatch
    ):
        db_path = self._fixture_db(tmp_path)
        manifest_path = _freeze(tmp_path, monkeypatch, grid=_single_trial_grid())
        report_path = tmp_path / "report.json"

        report = fv.evaluate_manifest(db_path, report_path, manifest_path)

        assert report["schema_version"] == 2
        assert report["verdict"] == "not_established"
        assert (
            report["evidence_class"]
            == "retrospective_previously_used_not_fresh_oos"
        )
        assert report_path.exists()
        assert report["validation"]["status"] in {
            "retrospective_positive_result",
            "retrospective_failed_result",
            "no_deployable_config",
        }
        assert "hashes" not in report["predeclared_manifest"]
        assert "verification" not in report
        assert not (tmp_path / ".ptr-alpha-evaluation-ledger-v2.json").exists()

    def test_evaluate_requires_existing_staged_database(self, tmp_path, monkeypatch):
        manifest_path = _freeze(tmp_path, monkeypatch, grid=_single_trial_grid())
        with pytest.raises(FileNotFoundError, match="staged database not found"):
            fv.evaluate_manifest(tmp_path / "missing.duckdb", None, manifest_path)

    def test_evaluate_requires_readable_database(self, tmp_path, monkeypatch):
        manifest_path = _freeze(tmp_path, monkeypatch, grid=_single_trial_grid())
        empty_db = tmp_path / "empty.duckdb"
        empty_db.write_bytes(b"not a real database")
        with pytest.raises(DatabaseError):
            fv.evaluate_manifest(empty_db, None, manifest_path)


class TestCli:
    def test_freeze_command_writes_predeclared_manifest(self, tmp_path, monkeypatch):
        target = tmp_path / "predeclared.json"
        monkeypatch.setattr(fv, "FROZEN_MANIFEST_PATH", target)
        assert fv.main(["freeze"]) == 0
        manifest = json.loads(target.read_text())
        assert manifest["schema_version"] == 2
        assert manifest["config"]["grid"]["lookback_days"] == [
            CONSENSUS_LOOKBACK_DAYS
        ]
