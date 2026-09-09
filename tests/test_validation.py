"""Scenario tests for purged, fail-closed validation."""

from __future__ import annotations

import inspect
import math
from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd
import pytest

from analyzer.cli import _validation_grid
from analyzer.experiments.family import FAMILY_PROVENANCE, build_family
from analyzer.exceptions import AnalysisError
from analyzer.pipeline import BacktestParams
from analyzer.member_ranking.buyer_scoring import (
    CONSENSUS_SCORER_PROVENANCE,
    score_ticker_by_buyers,
)
from analyzer.member_ranking.lookups import _compute_alpha_for_scoring_mode
from analyzer.validation import (
    LOCKED_FINAL_START,
    MIN_RELEASE_PERMUTATIONS,
    PRIMARY_METRIC,
    _backtest_core,
    _build_manifest,
    _effective_validation_grid,
    _phase_end,
    _run_identity_invariant_control,
    _run_validation_with_db,
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
                "scorer_provenance": CONSENSUS_SCORER_PROVENANCE,
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
        null = {
            0: _series(rng.normal(-2, 1, 80)),
            1: _series(rng.normal(-2, 1, 80)),
        }
        frame = _selection_frame(null, slopes=[1.0, 9999.0])
        result = select_config(
            frame,
            series_by_trial=null,
            n_permutations=9990,
        )
        assert result["deployable_config"] is None
        assert result["descriptive_best"]["label"] == "descriptive_only_not_deployable"


class TestMemberIdentityGate:
    def test_consensus_statistical_survivor_deploys_without_identity_record(self):
        rng = np.random.default_rng(41)
        series = {0: _series(2.0 + rng.normal(0, 0.1, 180))}
        result = select_config(
            _selection_frame(series),
            series_by_trial=series,
            n_permutations=999,
        )
        assert result["statistical_candidate"] is not None
        assert result["deployable_config"] is not None
        assert result["failure_reason"] is None
        assert result["member_identity_control"]["gating"] is False

    def test_caller_supplied_member_control_is_rejected(self):
        series = {0: _series(np.full(180, 2.0))}
        with pytest.raises(TypeError, match="unexpected keyword"):
            select_config(
                _selection_frame(series),
                series_by_trial=series,
                n_permutations=999,
                member_control={"release_ready": True},
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
        with pytest.raises(TypeError, match="unexpected keyword"):
            select_config(
                _selection_frame(series),
                series_by_trial=series,
                n_permutations=999,
                member_control={"exempt": True},
            )

    def test_forged_identity_diagnostic_cannot_change_deployment(self):
        series = {0: _series(np.full(180, 2.0))}
        baseline = _with_series(_selection_frame(series), series)
        selection_before = select_config(baseline, n_permutations=999)
        assert selection_before["deployable_config"] is not None

        control = _run_identity_invariant_control(baseline, 0)
        forged = replace(
            control, method="forged_significant_relabel_test", max_stat_p_value=0.0
        )
        selection_after = select_config(baseline, n_permutations=999)
        assert (
            selection_after["deployable_config"]
            == selection_before["deployable_config"]
        )
        with pytest.raises(TypeError, match="unexpected keyword"):
            select_config(baseline, n_permutations=999, member_control=forged)

    def test_short_series_cannot_fall_back_to_asymptotic_reward(self):
        short = {0: _series([2.0, 2.1, 1.9])}
        frame = _selection_frame(short)
        frame["horizon"] = 120
        frame["frequency_days"] = 30  # block length four; needs eight observations
        result = select_config(frame, series_by_trial=short, n_permutations=999)
        assert result["deployable_config"] is None
        assert result["failure_reason"] == "bootstrap_sample_too_small"
        assert "at least 8" in result["bootstrap"]["error"]


class TestConsensusProductionScoring:
    def test_consensus_is_distinct_from_member_alpha_and_rejects_alpha_path(self):
        transactions = pd.DataFrame(
            {
                "member": ["Alice", "Bob"],
                "ticker": ["AAPL", "AAPL"],
                "transaction_date": pd.to_datetime(["2024-05-09", "2024-05-11"]),
                "disclosure_date": pd.to_datetime(["2024-05-10", "2024-05-12"]),
                "transaction_type": ["Purchase", "Purchase"],
            }
        )
        as_of = pd.Timestamp("2024-05-20")
        consensus = score_ticker_by_buyers(
            "AAPL", transactions, min_buyers=1, as_of_date=as_of
        )
        ranking_dicts = {
            "mode": "shrunk_alpha",
            "alpha": {"ALICE": 10.0, "BOB": 20.0},
            "trades": {"ALICE": 5, "BOB": 5},
            "prob": {},
            "has_shrunk": True,
        }
        descriptive = score_ticker_by_buyers(
            "AAPL",
            transactions,
            signals_df=pd.DataFrame({"value": [1.0]}),
            member_rankings=pd.DataFrame({"value": [1.0]}),
            min_buyers=1,
            _ranking_dicts=ranking_dicts,
            scoring_mode="shrunk_alpha",
            as_of_date=as_of,
        )
        assert (
            consensus.iloc[0]["signal_score_raw"]
            != descriptive.iloc[0]["signal_score_raw"]
        )
        assert consensus.iloc[0]["scorer_provenance"] == CONSENSUS_SCORER_PROVENANCE
        with pytest.raises(AnalysisError, match="identity-free"):
            _compute_alpha_for_scoring_mode(
                pd.DataFrame(
                    {
                        "member": ["Alice"],
                        "shrunk_alpha": [1.0],
                        "purchase_trades": [1],
                    }
                ),
                "shrunk_alpha",
                "consensus",
            )

    def test_consensus_reports_non_gating_identity_invariance_diagnostic(self):
        series = {0: _series(np.full(180, 2.0))}
        baseline = _with_series(_selection_frame(series), series)
        selection = select_config(baseline, n_permutations=999)
        assert selection["deployable_config"] is not None

        control = _run_identity_invariant_control(baseline, 0)
        assert control.status == "identity_invariant"
        assert control.method == "identity_invariant_by_consensus_scorer_contract_v1"
        assert control.gating is False
        assert control.evaluated_permutations == 0
        assert control.max_stat_p_value == 1.0
        assert (
            select_config(baseline, n_permutations=999)["deployable_config"]
            == selection["deployable_config"]
        )


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
                        "scorer_provenance": CONSENSUS_SCORER_PROVENANCE,
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
                assert result["optimal_horizon"].iloc[0] == 120
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


class TestFailureFamilies:
    def _params(self):
        return BacktestParams(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 1),
            horizon=60,
            lookback_days=60,
            training_lookback_days=365,
            min_buyers=2,
            top_n=5,
            frequency_days=30,
        )

    def test_recommendation_exception_is_trial_failure_not_cash(self, monkeypatch):
        monkeypatch.setattr("analyzer.validation._benchmark_return", lambda *args: 1.0)
        monkeypatch.setattr(
            "analyzer.validation.analysis.backtest_recommendations",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                RuntimeError("broken recommender")
            ),
        )
        result, series = _backtest_core(
            pd.DataFrame(),
            pd.DataFrame(),
            self._params(),
            pd.DataFrame(),
            20.0,
            0.005,
        )
        assert result.status == "failed"
        assert result.failure_reason == "recommendation_exception"
        assert result.failure_count == 1
        assert result.no_trade_dates == 0
        assert series.empty
        assert result.failure_records[0]["error_type"] == "RuntimeError"

    def test_empty_recommendations_remain_a_legitimate_no_trade(self, monkeypatch):
        monkeypatch.setattr("analyzer.validation._benchmark_return", lambda *args: 1.0)
        monkeypatch.setattr(
            "analyzer.validation.analysis.backtest_recommendations",
            lambda *args, **kwargs: pd.DataFrame(),
        )
        result, series = _backtest_core(
            pd.DataFrame(),
            pd.DataFrame(),
            self._params(),
            pd.DataFrame(),
            20.0,
            0.005,
        )
        assert result.status == "completed"
        assert result.failure_count == 0
        assert result.no_trade_dates == 1
        assert series.tolist() == pytest.approx([-1.0])

    def test_evaluation_exception_is_trial_failure_not_cash(self, monkeypatch):
        monkeypatch.setattr("analyzer.validation._benchmark_return", lambda *args: 1.0)
        monkeypatch.setattr(
            "analyzer.validation.analysis.backtest_recommendations",
            lambda *args, **kwargs: pd.DataFrame(
                [
                    {
                        "rank": 1,
                        "ticker": "AAA",
                        "signal_score": 1.0,
                        "scorer_provenance": CONSENSUS_SCORER_PROVENANCE,
                    }
                ]
            ),
        )
        monkeypatch.setattr(
            "analyzer.validation.analysis.evaluate_backtest",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                RuntimeError("broken evaluator")
            ),
        )
        result, series = _backtest_core(
            pd.DataFrame(),
            pd.DataFrame(),
            self._params(),
            pd.DataFrame(),
            20.0,
            0.005,
        )
        assert result.status == "failed"
        assert result.failure_reason == "evaluation_exception"
        assert result.failure_count == 1
        assert result.no_trade_dates == 0
        assert series.empty
        assert result.failure_records[0]["stage"] == "evaluation"

    def test_failed_trial_fails_the_whole_family_and_cannot_deploy(self):
        series = {0: _series(np.full(180, 2.0)), 1: _series(np.full(180, 3.0))}
        frame = _with_series(_selection_frame(series), series)
        frame["trial_failed"] = [True, False]
        frame["trial_status"] = ["failed", "completed"]
        frame["failure_reason"] = ["recommendation_exception", None]
        frame["failure_count"] = [1, 0]
        frame["failure_records"] = [
            '[{"stage":"recommendation","reason":"recommendation_exception"}]',
            "[]",
        ]
        result = select_config(frame, n_permutations=999)
        assert result["deployable_config"] is None
        assert result["statistical_candidate"] is None
        assert result["failure_reason"] == "family_trial_failure"
        assert result["family_failure"]["status"] == "failed"
        assert result["family_failure"]["failed_trial_count"] == 1

    def test_failed_family_stops_before_test_window(self, tmp_path, monkeypatch):
        series = {0: _series(np.full(180, 2.0))}
        frame = _with_series(_selection_frame(series), series)
        frame["trial_failed"] = True
        frame["trial_status"] = "failed"
        frame["failure_reason"] = "recommendation_exception"
        frame["failure_count"] = 1
        frame["failure_records"] = '[{"stage":"recommendation"}]'
        monkeypatch.setattr(
            "analyzer.validation.sweep_configs", lambda *args, **kwargs: frame
        )
        transaction_queries = []

        class EmptyDb:
            def get_transactions_by_date_range(self, *args):
                transaction_queries.append(args)
                return pd.DataFrame({"ticker": pd.Series(dtype=str)})

            def get_prices(self, *args):
                return pd.DataFrame()

            def get_entry_prices(self, *args):
                raise AssertionError("consensus validation must not read entry prices")

        database = tmp_path / "db.duckdb"
        database.write_bytes(b"db")
        output = _run_validation_with_db(
            EmptyDb(),
            database,
            date(2022, 1, 1),
            date(2023, 1, 1),
            date(2022, 12, 1),
            date(2023, 2, 1),
            date(2024, 1, 1),
            date(2023, 11, 1),
            {"horizon": [60], "frequency_days": [30]},
            max_holding=60,
            n_permutations=999,
            permutation_seed=0,
            alpha=0.05,
            out_path=None,
        )
        assert output["selected_config"] is None
        assert output["correction"]["failure_reason"] == "family_trial_failure"
        assert transaction_queries == [
            (pd.Timestamp("2021-12-04"), pd.Timestamp("2022-12-01"))
        ]


class TestCanonicalFamilyMetadata:
    def test_reordered_parameter_or_value_grid_changes_identity(self):
        grid = {"horizon": [60, 90], "top_n": [5, 10]}
        same = build_family(grid)
        repeated = build_family({"horizon": [60, 90], "top_n": [5, 10]})
        reordered_parameters = build_family(
            {"top_n": [5, 10], "horizon": [60, 90]}
        )
        reordered_values = build_family(
            {"horizon": [90, 60], "top_n": [5, 10]}
        )
        assert same.family_sha256 == repeated.family_sha256
        assert [trial.trial_id for trial in same.trials] == [0, 1, 2, 3]
        assert same.family_sha256 != reordered_parameters.family_sha256
        assert same.family_sha256 != reordered_values.family_sha256
        assert same.family_size == 4
        assert same.metadata()["family_provenance"] == FAMILY_PROVENANCE
        assert same.metadata()["family_size"] == len(same.trials)
        assert same.metadata()["parameter_order"] == ["horizon", "top_n"]


class TestPurgeAndManifest:
    def test_purge_uses_exact_next_session_execution_window(self):
        assert _phase_end(date(2023, 12, 31), 120) == date(2023, 8, 31)

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

    def test_manifest_records_statistical_evidence_without_execution_receipts(self):
        frame = pd.DataFrame(
            {"x": [1, 2]}, index=pd.date_range("2024-01-01", periods=2)
        )
        manifest = _build_manifest(
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
            0.05,
        )
        assert manifest["phases"]["locked_final"] == {
            "start": "2026-01-01",
            "end": None,
            "status": "locked_not_queried_or_evaluated",
            "value_rows_queried": False,
            "consumed": False,
        }
        assert manifest["phases"]["train"]["outcomes_end_by"] == "2023-12-31"
        assert (
            manifest["phases"]["test"]["evidence_class"]
            == "retrospective_previously_used_not_fresh_oos"
        )
        assert manifest["n_trials"] == 1
        assert manifest["family"]["family_size"] == 1
        assert manifest["family"]["family_provenance"] == FAMILY_PROVENANCE
        assert manifest["coverage_input"]["transactions"] == 2
        assert "hashes" not in manifest
        assert "git" not in manifest
        assert "dependencies" not in manifest
        assert "evaluation_ledger" not in manifest


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


def test_cli_validation_grid_counts_are_exact():
    assert math.prod(len(values) for values in _validation_grid(False).values()) == 18
    assert math.prod(len(values) for values in _validation_grid(True).values()) == 36
    assert _validation_grid(False)["scoring_mode"] == ["consensus"]
    assert _validation_grid(False)["lookback_days"] == [28]
    assert "training_lookback_days" not in _validation_grid(True)
    assert "decay_lambda" not in _validation_grid(True)
    assert "bayes_prior_strength" not in _validation_grid(True)


def test_consensus_family_discards_nonoperative_dimensions():
    effective = _effective_validation_grid(
        {
            "horizon": [60],
            "frequency_days": [30],
            "lookback_days": [28],
            "training_lookback_days": [180, 365],
            "min_buyers": [3],
            "top_n": [5],
            "threshold": [1.0, 5.0],
            "decay_lambda": [0.001, 0.02],
            "bayes_prior_strength": [5, 50],
            "scoring_mode": ["consensus"],
        }
    )

    assert effective == {
        "horizon": [60],
        "frequency_days": [30],
        "lookback_days": [28],
        "min_buyers": [3],
        "top_n": [5],
        "scoring_mode": ["consensus"],
    }
