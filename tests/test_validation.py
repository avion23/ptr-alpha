"""Scenario tests for purged, fail-closed validation."""

from __future__ import annotations

import hashlib
import inspect
import math
from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd
import pytest

from analyzer.pipeline import BacktestParams
from analyzer.validation import (
    LOCKED_FINAL_START,
    MIN_RELEASE_PERMUTATIONS,
    PRIMARY_METRIC,
    EvaluationAlreadyConsumedError,
    EvaluationLedgerIntegrityError,
    _backtest_core,
    _build_manifest,
    _canonical_ledger_path,
    _complete_evaluation,
    _hash_untracked_path,
    _member_family_sha256,
    _member_identity_permutations,
    _phase_end,
    _reserve_evaluation,
    _run_member_identity_control,
    _validate_ledger,
    newey_west_tstat,
    permute_signal_member_labels,
    run_validation,
    select_config,
)


def _series(values, start="2020-01-01"):
    return pd.Series(
        values, index=pd.date_range(start, periods=len(values), freq="D"), dtype=float
    )


def _selection_frame(
    series_by_trial: dict[int, pd.Series], slopes=None
) -> pd.DataFrame:
    rows = []
    slopes = slopes or [0.0] * len(series_by_trial)
    for trial_id, values in series_by_trial.items():
        statistic = newey_west_tstat(values, lag=0)
        p_value = (
            float(__import__("scipy").stats.norm.sf(statistic))
            if math.isfinite(statistic)
            else (0.0 if statistic > 0 else 1.0)
        )
        rows.append(
            {
                "trial_id": trial_id,
                "horizon": 60,
                "frequency_days": 30,
                "training_lookback_days": 365,
                "min_buyers": 2,
                "top_n": 5,
                "decay_lambda": 0.005,
                "bayes_prior_strength": 20.0,
                "scoring_mode": "consensus",
                "total_recs": 100,
                "dates_evaluated": len(values),
                "overall_alpha": float(values.mean()),
                "overall_return": max(float(values.mean()), 0.0),
                "alpha_slope": slopes[trial_id],
                "nw_lag": 0,
                "nw_tstat": statistic,
                "p_value": p_value,
                "min_sample_ok": True,
            }
        )
    return pd.DataFrame(rows)


def _with_series(frame: pd.DataFrame, series: dict[int, pd.Series]) -> pd.DataFrame:
    frame = frame.copy()
    frame.attrs["series_by_trial"] = series
    return frame


class TestNeweyWest:
    def test_zero_alpha_canary_is_exactly_null(self):
        values = _series(np.zeros(40))
        assert newey_west_tstat(values, lag=5) == 0.0

    def test_lag_zero_matches_biased_plain_tstat(self):
        values = _series(np.arange(1.0, 9.0))
        expected = values.mean() / (np.std(values) / np.sqrt(len(values)))
        assert newey_west_tstat(values, lag=0) == pytest.approx(expected)

    def test_lag_is_capped(self):
        values = _series([1.0, 2.0, 4.0, 8.0])
        assert newey_west_tstat(values, 99) == newey_west_tstat(values, 3)


class TestCorrectedSelection:
    def test_all_zero_canary_has_no_deployable_config(self):
        null = {0: _series(np.zeros(60))}
        result = select_config(
            _selection_frame(null),
            series_by_trial=null,
            n_permutations=MIN_RELEASE_PERMUTATIONS,
        )
        assert result["deployable_config"] is None
        assert result["n_survivors"] == 0
        assert result["failure_reason"] == "no_dependence_safe_survivor"

    def test_insufficient_null_count_fails_closed(self):
        strong = {0: _series(2.0 + np.random.default_rng(2).normal(0, 0.1, 120))}
        result = select_config(
            _selection_frame(strong),
            series_by_trial=strong,
            n_permutations=99,
        )
        assert result["deployable_config"] is None
        assert (
            result["failure_reason"]
            == "insufficient_bootstrap_count_or_family_resolution"
        )
        assert result["bootstrap"]["release_ready"] is False

    def test_missing_or_incomplete_null_series_fails_closed(self):
        series = {0: _series(np.ones(30)), 1: _series(np.ones(30))}
        frame = _selection_frame(series)
        missing = select_config(frame, n_permutations=999)
        incomplete = select_config(
            frame, series_by_trial={0: series[0]}, n_permutations=999
        )
        assert missing["failure_reason"] == "missing_bootstrap_series"
        assert incomplete["failure_reason"] == "incomplete_bootstrap_series"
        assert missing["deployable_config"] is None
        assert incomplete["deployable_config"] is None

    def test_primary_mean_selects_not_rank_slope(self):
        rng = np.random.default_rng(4)
        series = {
            0: _series(2.0 + rng.normal(0, 0.2, 180)),
            1: _series(1.0 + rng.normal(0, 0.2, 180), start="2020-01-01"),
        }
        frame = _selection_frame(series, slopes=[-1000.0, 1000.0])
        result = select_config(
            frame,
            series_by_trial=series,
            n_permutations=999,
            permutation_seed=7,
        )
        assert result["deployable_config"] is None
        assert result["statistical_candidate"]["trial_id"] == 0
        assert result["primary_metric"] == PRIMARY_METRIC

    def test_block_permuted_null_does_not_survive(self):
        blocks = np.tile(np.concatenate([np.ones(5), -np.ones(5)]), 20)
        null = {0: _series(blocks), 1: _series(-blocks)}
        result = select_config(
            _selection_frame(null),
            series_by_trial=null,
            n_permutations=999,
            permutation_seed=11,
        )
        assert result["n_survivors"] == 0
        assert result["deployable_config"] is None

    def test_no_survivor_is_descriptive_only_not_a_fallback(self):
        rng = np.random.default_rng(9)
        null = {0: _series(rng.normal(0, 1, 80)), 1: _series(rng.normal(0, 1, 80))}
        frame = _selection_frame(null, slopes=[1.0, 9999.0])
        result = select_config(
            frame,
            series_by_trial=null,
            n_permutations=9990,
        )
        assert result["deployable_config"] is None
        assert result["descriptive_best"]["label"] == "descriptive_only_not_deployable"


class TestMemberIdentityGate:
    def test_statistical_candidate_cannot_deploy_without_member_control(self):
        rng = np.random.default_rng(41)
        series = {0: _series(2.0 + rng.normal(0, 0.1, 180))}
        result = select_config(
            _selection_frame(series),
            series_by_trial=series,
            n_permutations=999,
        )
        assert result["statistical_candidate"] is not None
        assert result["deployable_config"] is None
        assert result["failure_reason"] == "member_identity_control_required_or_failed"

    def test_member_control_below_release_count_fails_closed(self):
        rng = np.random.default_rng(42)
        series = {0: _series(2.0 + rng.normal(0, 0.1, 180))}
        result = select_config(
            _selection_frame(series),
            series_by_trial=series,
            n_permutations=999,
            member_control={
                "status": "completed",
                "n_permutations": 998,
                "release_ready": False,
                "max_stat_p_value": 0.001,
            },
        )
        assert result["deployable_config"] is None
        assert (
            result["member_identity_control"]["status"] == "invalid_non_runner_control"
        )

    def test_descriptive_scoring_modes_are_never_deployment_candidates(self):
        series = {0: _series(np.full(180, 2.0))}
        frame = _selection_frame(series)
        frame["scoring_mode"] = "shrunk_alpha"
        result = select_config(frame, series_by_trial=series, n_permutations=999)
        assert result["statistical_candidate"] is None
        assert result["deployable_config"] is None

    def test_identity_free_exemption_payload_is_never_accepted(self):
        series = {0: _series(np.full(180, 2.0))}
        frame = _selection_frame(series)
        frame["scoring_mode"] = "identity_free_consensus"
        result = select_config(
            frame,
            series_by_trial=series,
            n_permutations=999,
            member_control={
                "exempt": True,
                "invariance_proof_sha256": "a" * 64,
                "release_ready": True,
                "max_stat_p_value": 0.0,
            },
        )
        assert result["deployable_config"] is None
        assert (
            result["member_identity_control"]["status"] == "invalid_non_runner_control"
        )

    def test_production_member_control_derives_executed_family_and_statistic(
        self, monkeypatch
    ):
        signal = pd.DataFrame({"member": ["A", "B"], "value": [1.0, 2.0]})
        monkeypatch.setattr(
            "analyzer.validation.analysis.calculate_signal_potential",
            lambda *args, **kwargs: signal,
        )
        series = {0: _series(np.full(20, 3.0))}
        baseline = _with_series(_selection_frame(series), series)
        calls = []

        def fake_sweep(*args, **kwargs):
            permuted = kwargs["signals_by_horizon"]
            calls.append(permuted)
            labels = permuted[(60, 0.005)]["member"].tolist()
            result = baseline.copy()
            if labels != ["A", "B"]:
                result["nw_tstat"] = 0.0
            result.attrs["series_by_trial"] = series
            return result

        monkeypatch.setattr("analyzer.validation.sweep_configs", fake_sweep)
        result = _run_member_identity_control(
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            {"horizon": [60], "decay_lambda": [0.005]},
            date(2022, 1, 1),
            date(2023, 1, 1),
            observed_trial_id=0,
            n_permutations=3,
            seed=10,
        )
        assert len(calls) == 2  # baseline plus swap; identity reuses baseline
        assert result.status == "completed"
        assert result.exact_enumeration is True
        assert result.permutation_group_size == 2
        assert result.p_value_resolution == 0.5
        assert result.max_stat_p_value >= 0.5
        assert result.observed_statistic == baseline.iloc[0]["nw_tstat"]
        assert result.family_sha256 == _member_family_sha256(baseline, series)
        assert result.release_ready is True

    def test_large_group_samples_unique_uniform_permutations(self):
        members = list("ABCDEFG")
        permutations, group_size, exact = _member_identity_permutations(
            members, 999, seed=8
        )
        assert group_size == math.factorial(7)
        assert exact is False
        assert len(permutations) == len(set(permutations)) == 999
        assert all(set(value) == set(members) for value in permutations)
        assert any(
            source == target
            for value in permutations
            for source, target in zip(members, value)
        )

    def test_runtime_budget_fails_closed_without_duplicate_draw_claims(
        self, monkeypatch
    ):
        signal = pd.DataFrame({"member": ["A", "B"], "value": [1.0, 2.0]})
        monkeypatch.setattr(
            "analyzer.validation.analysis.calculate_signal_potential",
            lambda *args, **kwargs: signal,
        )
        series = {0: _series(np.ones(20))}
        baseline = _with_series(_selection_frame(series), series)
        monkeypatch.setattr(
            "analyzer.validation.sweep_configs", lambda *args, **kwargs: baseline
        )
        result = _run_member_identity_control(
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            {"horizon": [60], "decay_lambda": [0.005]},
            date(2022, 1, 1),
            date(2023, 1, 1),
            observed_trial_id=0,
            n_permutations=999,
            seed=10,
            runtime_budget_seconds=0.0,
        )
        assert result.status == "infeasible_runtime_budget"
        assert result.evaluated_permutations == 0
        assert result.release_ready is False
        assert result.max_stat_p_value == 1.0

    def test_only_bound_unedited_runner_result_can_unlock_deployment(self, monkeypatch):
        signal = pd.DataFrame({"member": list("ABCDEFG"), "value": np.arange(7.0)})
        monkeypatch.setattr(
            "analyzer.validation.analysis.calculate_signal_potential",
            lambda *args, **kwargs: signal,
        )
        series = {0: _series(np.full(180, 2.0))}
        baseline = _with_series(_selection_frame(series), series)
        calls = 0

        def fake_sweep(*args, **kwargs):
            nonlocal calls
            calls += 1
            result = baseline.copy()
            if calls > 1:
                result["nw_tstat"] = 0.0
            result.attrs["series_by_trial"] = series
            return result

        monkeypatch.setattr("analyzer.validation.sweep_configs", fake_sweep)
        control = _run_member_identity_control(
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            {"horizon": [60], "decay_lambda": [0.005]},
            date(2022, 1, 1),
            date(2023, 1, 1),
            observed_trial_id=0,
            n_permutations=999,
            seed=1,
        )
        result = select_config(baseline, n_permutations=999, member_control=control)
        assert control.evaluated_permutations == 999
        assert control.release_ready is True
        assert result["deployable_config"] is not None

        edited = replace(control, max_stat_p_value=0.0)
        edited_result = select_config(
            baseline, n_permutations=999, member_control=edited
        )
        assert edited_result["deployable_config"] is None

        statistic_edited = replace(control, observed_statistic=0.0)
        statistic_result = select_config(
            baseline, n_permutations=999, member_control=statistic_edited
        )
        assert statistic_result["deployable_config"] is None

    def test_runner_result_must_match_exact_executed_family(self, monkeypatch):
        signal = pd.DataFrame({"member": ["A", "B"], "value": [1.0, 2.0]})
        monkeypatch.setattr(
            "analyzer.validation.analysis.calculate_signal_potential",
            lambda *args, **kwargs: signal,
        )
        executed_series = {0: _series(np.full(180, 2.0))}
        executed = _with_series(_selection_frame(executed_series), executed_series)
        monkeypatch.setattr(
            "analyzer.validation.sweep_configs", lambda *args, **kwargs: executed
        )
        control = _run_member_identity_control(
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            {"horizon": [60], "decay_lambda": [0.005]},
            date(2022, 1, 1),
            date(2023, 1, 1),
            observed_trial_id=0,
            n_permutations=999,
            seed=1,
        )
        changed_series = {0: _series(np.full(180, 3.0))}
        changed = _with_series(_selection_frame(changed_series), changed_series)
        result = select_config(changed, n_permutations=999, member_control=control)
        assert result["statistical_candidate"] is not None
        assert result["deployable_config"] is None

    def test_short_series_cannot_fall_back_to_asymptotic_reward(self):
        short = {0: _series([2.0, 2.1, 1.9])}
        frame = _selection_frame(short)
        frame["horizon"] = 120
        frame["frequency_days"] = 30  # block length four; needs eight observations
        result = select_config(frame, series_by_trial=short, n_permutations=999)
        assert result["deployable_config"] is None
        assert result["failure_reason"] == "bootstrap_sample_too_small"
        assert "at least 8" in result["bootstrap"]["error"]


class TestExecutionSupport:
    def test_frequency_support_and_no_trade_cash_use_identical_spy_dates(
        self, monkeypatch
    ):
        evaluation_calls = []

        def fake_recommendations(*args, **kwargs):
            assert len(args) == 2
            assert kwargs["scoring_mode"] == "consensus"
            as_of = pd.Timestamp(kwargs["as_of_date"])
            if as_of.day != 16:
                return pd.DataFrame()
            return pd.DataFrame(
                [
                    {
                        "rank": 1,
                        "ticker": "AAA",
                        "signal_score": 3.0,
                        "optimal_horizon": 120,
                    }
                ]
            )

        def fake_evaluate(recommendations, prices, as_of, horizon):
            evaluation_calls.append(
                (recommendations.copy(), pd.Timestamp(as_of), horizon)
            )
            result = recommendations.copy()
            if result.iloc[0]["ticker"] == "SPY":
                result["bt_return_pct"] = 1.0
                result["bt_spy_return_pct"] = 1.0
                result["bt_alpha_pct"] = 0.0
            else:
                assert "optimal_horizon" not in result.columns
                assert horizon == 60
                result["bt_return_pct"] = 2.0
                result["bt_spy_return_pct"] = 1.0
                result["bt_alpha_pct"] = 1.0
            return result

        monkeypatch.setattr(
            "analyzer.validation.analysis.backtest_recommendations",
            fake_recommendations,
        )
        monkeypatch.setattr(
            "analyzer.validation.analysis.evaluate_backtest", fake_evaluate
        )
        params = BacktestParams(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
            horizon=60,
            frequency_days=15,
            training_lookback_days=365,
            min_buyers=2,
            top_n=3,
        )
        result, primary = _backtest_core(
            pd.DataFrame(),
            pd.DataFrame({"SPY": [1.0]}),
            params,
            pd.DataFrame(),
            20,
            0.005,
        )
        assert list(primary) == pytest.approx([-1.0, 1.0, -1.0])
        assert result.scoring_mode == "consensus"
        assert (
            inspect.signature(_backtest_core).parameters["scoring_mode"].default
            == "consensus"
        )
        assert result.scheduled_dates == 3
        assert result.benchmark_dates == 3
        assert result.dates_evaluated == 3
        assert result.no_trade_dates == 2
        assert result.coverage_pct == 100.0
        assert result.overall_return == pytest.approx(2.0 / 3.0, abs=1e-4)
        assert result.overall_spy_return == 1.0
        assert result.overall_alpha == pytest.approx(-1.0 / 3.0, abs=1e-4)
        assert math.isnan(result.rank5_alpha)
        assert math.isnan(result.alpha_slope)
        assert (
            len(evaluation_calls) == 4
        )  # one SPY check per date plus one strategy trade


class TestPurgeAndManifest:
    def test_purge_uses_max_executable_holding(self):
        assert _phase_end(date(2023, 12, 31), 120, 0) == date(2023, 9, 2)
        assert _phase_end(date(2023, 12, 31), 120, 10) == date(2023, 8, 23)

    def test_locked_final_phase_is_rejected_before_database_open(self, tmp_path):
        with pytest.raises(ValueError, match="locked final phase"):
            run_validation(
                tmp_path / "missing.duckdb",
                date(2022, 1, 1),
                date(2023, 12, 31),
                date(2024, 1, 1),
                LOCKED_FINAL_START,
                {"horizon": [60]},
            )

    def test_manifest_hashes_and_locks_final_without_consuming_it(
        self, tmp_path, monkeypatch
    ):
        database = tmp_path / "db.duckdb"
        database.write_bytes(b"known database bytes")
        frame = pd.DataFrame(
            {"x": [1, 2]}, index=pd.date_range("2024-01-01", periods=2)
        )
        monkeypatch.setattr("analyzer.validation._code_hash", lambda: "c" * 64)
        monkeypatch.setattr(
            "analyzer.validation._git_state",
            lambda: {"revision": "git-known", "dirty": False, "diff_sha256": "d" * 64},
        )
        manifest = _build_manifest(
            database,
            frame,
            frame,
            frame,
            {"horizon": [60], "frequency_days": [30]},
            date(2022, 1, 1),
            date(2023, 12, 31),
            date(2023, 11, 1),
            date(2024, 1, 1),
            date(2025, 6, 30),
            date(2025, 5, 1),
            60,
            999,
            7,
            999,
            0.05,
        )
        assert manifest["phases"]["locked_final"] == {
            "start": "2026-01-01",
            "end": None,
            "status": "locked_not_queried_or_evaluated",
            "value_rows_queried": False,
            "whole_database_file_hashed_for_provenance": True,
            "consumed": False,
        }
        assert manifest["phases"]["train"]["outcomes_end_by"] == "2023-12-31"
        assert (
            manifest["phases"]["test"]["evidence_class"]
            == "retrospective_previously_used_not_fresh_oos"
        )
        assert manifest["hashes"]["code_sha256"] == "c" * 64
        assert manifest["hashes"]["git_revision"] == "git-known"
        assert manifest["hashes"]["git_diff_sha256"] == "d" * 64
        assert manifest["git"]["dirty"] is False
        for key in [
            "database_sha256",
            "value_snapshot_sha256",
            "config_sha256",
            "dependency_sha256",
        ]:
            assert len(manifest["hashes"][key]) == 64
        assert manifest["n_trials"] == 1


class TestEvaluationConsumptionLedger:
    def test_reservation_is_durable_and_refuses_repeat_or_alternate_grid(
        self, tmp_path
    ):
        ledger = tmp_path / "ledger.json"
        manifest = {
            "hashes": {
                "database_sha256": "a" * 64,
                "value_snapshot_sha256": "b" * 64,
            }
        }
        first = _reserve_evaluation(
            ledger,
            manifest,
            {"horizon": 60},
            {"horizon": [60]},
            date(2024, 1, 1),
            date(2025, 6, 30),
        )
        assert ledger.exists()
        payload = __import__("json").loads(ledger.read_text())
        assert payload["events"][0]["status"] == "reserved_consumed"
        _validate_ledger(payload)
        with pytest.raises(EvaluationAlreadyConsumedError, match="alternate"):
            _reserve_evaluation(
                ledger,
                manifest,
                {"horizon": 90},
                {"horizon": [90]},
                date(2025, 1, 1),
                date(2025, 12, 31),
            )
        _complete_evaluation(ledger, first, "completed_retrospective")
        payload = __import__("json").loads(ledger.read_text())
        assert payload["events"][-1]["status"] == "completed_retrospective"
        assert (
            payload["events"][-1]["previous_sha256"]
            == payload["events"][0]["event_sha256"]
        )
        _validate_ledger(payload)

    def test_prior_v1_ledger_requires_explicit_archive_or_migration(self, tmp_path):
        legacy = tmp_path / "validation_evaluation_ledger.json"
        legacy.write_text(
            __import__("json").dumps(
                {
                    "schema_version": 1,
                    "evaluations": [{"window": ["2024-01-01", "2025-06-30"]}],
                }
            )
        )
        manifest = {
            "hashes": {
                "database_sha256": "a" * 64,
                "value_snapshot_sha256": "b" * 64,
            }
        }
        with pytest.raises(EvaluationLedgerIntegrityError, match="archive or migrate"):
            _reserve_evaluation(
                tmp_path / ".ptr-alpha-evaluation-ledger-v2.json",
                manifest,
                {},
                {},
                date(2024, 1, 1),
                date(2025, 6, 30),
            )

    def test_hash_chain_detects_local_tampering(self, tmp_path):
        ledger = tmp_path / "ledger.json"
        manifest = {
            "hashes": {
                "database_sha256": "a" * 64,
                "value_snapshot_sha256": "b" * 64,
            }
        }
        _reserve_evaluation(
            ledger, manifest, {}, {}, date(2024, 1, 1), date(2024, 12, 31)
        )
        payload = __import__("json").loads(ledger.read_text())
        payload["events"][0]["status"] = "rewritten"
        with pytest.raises(EvaluationLedgerIntegrityError, match="hash"):
            _validate_ledger(payload)
        assert "local attacker" in payload["local_tamper_limitation"]

    def test_public_runner_has_only_canonical_ledger_path(self, tmp_path):
        assert (
            "evaluation_ledger_path" not in inspect.signature(run_validation).parameters
        )
        db_path = tmp_path / "data" / "database.duckdb"
        assert _canonical_ledger_path(db_path) == (
            db_path.parent.resolve() / ".ptr-alpha-evaluation-ledger-v2.json"
        )

    def test_untracked_directory_hash_recurses_into_file_contents(self, tmp_path):
        directory = tmp_path / "untracked"
        directory.mkdir()
        nested = directory / "nested"
        nested.mkdir()
        value = nested / "value.txt"
        value.write_text("first")
        first = hashlib.sha256()
        _hash_untracked_path(first, tmp_path, directory)
        value.write_text("second")
        second = hashlib.sha256()
        _hash_untracked_path(second, tmp_path, directory)
        assert first.hexdigest() != second.hexdigest()

    def test_untracked_hash_records_nested_directory_symlinks(self, tmp_path):
        directory = tmp_path / "untracked"
        directory.mkdir()
        first_target = tmp_path / "first-target"
        second_target = tmp_path / "second-target"
        first_target.mkdir()
        second_target.mkdir()
        link = directory / "nested-link"
        link.symlink_to(first_target, target_is_directory=True)
        first = hashlib.sha256()
        _hash_untracked_path(first, tmp_path, directory)
        link.unlink()
        link.symlink_to(second_target, target_is_directory=True)
        second = hashlib.sha256()
        _hash_untracked_path(second, tmp_path, directory)
        assert first.hexdigest() != second.hexdigest()


class TestMemberPermutationCanary:
    def test_member_label_permutation_is_bijective_and_preserves_values(self):
        original = {
            (60, 0.005): pd.DataFrame(
                {
                    "member": ["A", "A", "B", "C"],
                    "ticker": ["X", "Y", "Z", "Q"],
                    "outcome": [1.0, 2.0, 3.0, 4.0],
                }
            ),
            (90, 0.005): pd.DataFrame(
                {"member": ["A", "B", "C"], "outcome": [5.0, 6.0, 7.0]}
            ),
        }
        permuted = permute_signal_member_labels(original, seed=3)
        assert sorted(permuted[(60, 0.005)]["member"].value_counts()) == [1, 1, 2]
        assert permuted[(60, 0.005)]["outcome"].tolist() == [1.0, 2.0, 3.0, 4.0]
        # The full permutation group is valid: fixed points are not excluded.
        identity = permute_signal_member_labels(original, permutation=("A", "B", "C"))
        assert (
            identity[(60, 0.005)]["member"].tolist()
            == original[(60, 0.005)]["member"].tolist()
        )
        assert original[(60, 0.005)]["member"].tolist() == ["A", "A", "B", "C"]


def test_legacy_sweep_refuses_winner_claims(capsys):
    import sweep

    with pytest.raises(SystemExit) as exc:
        sweep.main()
    assert exc.value.code == 2
    error = capsys.readouterr().err
    assert "disabled" in error
    assert "no in-sample winner" in error
