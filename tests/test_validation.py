"""Scenario tests for purged, fail-closed validation."""

from __future__ import annotations

import math
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
    _backtest_core,
    _build_manifest,
    _complete_evaluation,
    _phase_end,
    _reserve_evaluation,
    _run_member_identity_control,
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
                "scoring_mode": "shrunk_alpha",
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
        assert result["failure_reason"] == "insufficient_bootstrap_count"
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
            member_control={
                "status": "completed",
                "n_permutations": 999,
                "release_ready": True,
                "max_stat_p_value": 0.001,
            },
        )
        assert result["deployable_config"] is not None
        assert result["deployable_config"]["trial_id"] == 0
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
        assert result["member_identity_control"]["n_permutations"] == 998

    def test_production_member_control_runs_full_family_and_records_runtime(
        self, monkeypatch
    ):
        signal = pd.DataFrame({"member": ["A", "B"], "value": [1.0, 2.0]})
        monkeypatch.setattr(
            "analyzer.validation.analysis.calculate_signal_potential",
            lambda *args, **kwargs: signal,
        )
        calls = []

        def fake_sweep(*args, **kwargs):
            calls.append(kwargs["signals_by_horizon"])
            return pd.DataFrame(
                [{"min_sample_ok": True, "nw_tstat": float(len(calls))}]
            )

        monkeypatch.setattr("analyzer.validation.sweep_configs", fake_sweep)
        result = _run_member_identity_control(
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            {
                "horizon": [60],
                "decay_lambda": [0.005],
                "frequency_days": [30],
            },
            date(2022, 1, 1),
            date(2023, 1, 1),
            observed_statistic=2.5,
            n_permutations=3,
            seed=10,
        )
        assert len(calls) == 3
        assert result["status"] == "completed"
        assert result["n_permutations"] == 3
        assert result["release_ready"] is False
        assert result["runtime_seconds"] >= 0

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
            as_of = pd.Timestamp(args[2])
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
        assert payload["evaluations"][0]["status"] == "reserved_consumed"
        with pytest.raises(EvaluationAlreadyConsumedError, match="alternate"):
            _reserve_evaluation(
                ledger,
                manifest,
                {"horizon": 90},
                {"horizon": [90]},
                date(2024, 1, 1),
                date(2025, 6, 30),
            )
        _complete_evaluation(ledger, first, "completed_retrospective")
        payload = __import__("json").loads(ledger.read_text())
        assert payload["evaluations"][0]["status"] == "completed_retrospective"


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
        assert all(
            changed != source
            for changed, source in zip(
                permuted[(60, 0.005)]["member"],
                original[(60, 0.005)]["member"],
            )
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
