from __future__ import annotations

import json
from unittest.mock import patch

import pandas as pd
import pytest

from analyzer.member_names import canonical_member_key
from optimize_profit import main
from optimize_profit.metrics import summarize_walk_forward
from optimize_profit.scoring import make_shuffled_scorer, score_constant
from optimize_profit.walk_forward import (
    _build_custom_ranking_dicts,
    _score_candidates,
    run_walk_forward,
)


def _ranking() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "member": ["Nancy P. Pelosi", "John Q. Public", "Alice Example"],
            "purchase_trades": [8, 4, 2],
            "bayes_win_prob": [0.7, 0.5, 0.4],
            "shrunk_alpha": [3.0, 1.0, -1.0],
        }
    )


def _period(date: str, *, status: str = "ready", reason: str | None = None) -> dict:
    return {
        "as_of_ts": pd.Timestamp(date),
        "horizon": 90,
        "status": status,
        "reason": reason,
        "candidate_tickers": {1: ["AAA"]},
    }


def _metric_run(dates=("2024-07-01",)) -> dict:
    support = list(dates)
    metrics = {
        "total_return_pct": 0.0,
        "spy_total_return_pct": 1.0,
        "excess_total_return_pct": -1.0,
        "mean_alpha_pct": -1.0,
        "sharpe": 0.0,
        "alpha_sharpe": -1.0,
        "terminal_observation_drawdown_pct": 0.0,
        "win_rate_pct": 0.0,
        "n_periods": len(support),
        "n_cash_periods": len(support),
        "avg_positions": 0.0,
        "requested_periods": len(support),
        "coverage_pct": 100.0,
        "support_sha256": "support-hash",
    }
    return {
        **metrics,
        "support_dates": support,
        "period_results": [],
        "rejection_ledger": [],
    }


def test_custom_ranking_dicts_add_canonical_member_aliases():
    lookups = _build_custom_ranking_dicts(
        _ranking(), lambda frame: dict(zip(frame["member"], [9.0, 2.0, 1.0]))
    )

    assert lookups["alpha"][canonical_member_key("Nancy P. Pelosi")] == 9.0
    assert lookups["trades"][canonical_member_key("John Q. Public")] == 4
    assert lookups["prob"][canonical_member_key("Alice Example")] == 0.4


def test_non_overlapping_schedule_is_enforced():
    periods = {"a": _period("2024-01-01"), "b": _period("2024-01-31")}

    with pytest.raises(ValueError, match="Overlapping bankroll periods"):
        run_walk_forward(
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            periods,
            score_constant,
            top_n=3,
            min_buyers=1,
            allocation="equal",
        )


def test_rejected_period_is_cash_with_spy_opportunity_alpha_and_full_support():
    periods = {
        "a": _period("2024-01-01", status="rejected", reason="empty_training"),
        "b": _period("2024-04-01", status="rejected", reason="no_candidates"),
    }
    with patch("optimize_profit.walk_forward._spy_period_return", return_value=5.0):
        result = run_walk_forward(
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            periods,
            score_constant,
            top_n=3,
            min_buyers=1,
            allocation="equal",
        )

    assert result["coverage_pct"] == 100.0
    assert result["n_periods"] == result["requested_periods"] == 2
    assert result["n_cash_periods"] == 2
    assert result["total_return_pct"] == 0.0
    assert result["mean_alpha_pct"] == -5.0
    assert {row["status"] for row in result["period_results"]} == {"cash"}


def test_terminal_observation_drawdown_is_labeled_and_cannot_stop_trading():
    summary = summarize_walk_forward(
        [{"portfolio_return_pct": -10.0, "spy_return_pct": 0.0, "n_positions": 1}]
    )
    assert summary["terminal_observation_drawdown_pct"] == -10.0
    assert "max_drawdown_pct" not in summary

    with pytest.raises(ValueError, match="daily NAV is required"):
        run_walk_forward(
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            {"a": _period("2024-01-01")},
            score_constant,
            top_n=3,
            min_buyers=1,
            allocation="equal",
            max_dd_pct=5,
        )


def test_candidate_failures_propagate_instead_of_being_swallowed():
    with patch(
        "optimize_profit.walk_forward.score_ticker_by_buyers",
        side_effect=RuntimeError("canary failure"),
    ):
        with pytest.raises(RuntimeError, match="canary failure"):
            _score_candidates(
                ["AAA"],
                pd.DataFrame({"ticker": ["AAA"]}),
                pd.DataFrame({"signal": [1]}),
                90,
                5.0,
                _ranking(),
                1,
                pd.DataFrame(),
                {"alpha": {}, "trades": {}, "prob": {}},
            )


def test_each_decay_is_passed_to_signal_computation():
    calls = []

    def fake_signals(entries, prices, horizons, decay_lambda):
        calls.append(decay_lambda)
        return pd.DataFrame({"decay": [decay_lambda]})

    def fake_precompute(signals, *args, **kwargs):
        return {"decay": float(signals["decay"].iloc[0])}

    with (
        patch.object(
            main.analysis, "calculate_signal_potential", side_effect=fake_signals
        ),
        patch.object(main, "precompute_walk_forward_data", side_effect=fake_precompute),
    ):
        datasets = main._build_decay_datasets(
            pd.DataFrame({"entry": [1]}),
            pd.DataFrame({"transaction": [1]}),
            pd.DataFrame({"price": [1]}),
            pd.DatetimeIndex(["2024-01-01"]),
        )

    assert calls == list(main.PARAM_GRID["decay_lambda"])
    assert datasets == {decay: {"decay": decay} for decay in calls}


def test_phase_embargo_precommits_future_final_without_consuming_it():
    selection, retrospective, final = main._phase_dates()

    assert retrospective.min() == main.RETROSPECTIVE_START
    assert final.min() == main.FINAL_TEST_START
    assert selection.max() + pd.Timedelta(days=main.HORIZON) <= retrospective.min()
    assert retrospective.max() + pd.Timedelta(days=main.HORIZON) <= final.min()
    assert final.min() > main.RETROSPECTIVE_END

    with pytest.raises(ValueError, match="Horizon embargo violated"):
        main._assert_phase_embargo(
            pd.DatetimeIndex(["2025-06-26"]),
            pd.DatetimeIndex(["2025-07-01"]),
            "retrospective",
            "final",
        )


def test_mismatched_or_incomplete_support_fails_closed():
    complete = _metric_run(("2024-07-01", "2024-09-29"))
    mismatch = _metric_run(("2024-07-01", "2024-10-01"))
    incomplete = _metric_run(("2024-07-01", "2024-09-29"))
    incomplete["coverage_pct"] = 50.0

    with pytest.raises(RuntimeError, match="support differs"):
        main._assert_identical_support("validation", [complete, mismatch])
    with pytest.raises(RuntimeError, match="lacks 100%"):
        main._assert_identical_support("validation", [complete, incomplete])


def test_family_gate_is_bonferroni_not_bh():
    trials = pd.DataFrame(
        [
            {
                "alpha_sharpe": 2.0,
                "mean_alpha_pct": 1.0,
                "bonferroni_significant": False,
            },
            {
                "alpha_sharpe": 1.0,
                "mean_alpha_pct": 0.5,
                "bonferroni_significant": True,
            },
        ]
    )
    selected, passed = main._select_frozen_config(trials)
    assert passed is True
    assert selected["alpha_sharpe"] == 1.0


def test_constant_and_period_aligned_shuffled_scorers_are_reproducible():
    ranking = _ranking()
    assert set(score_constant(ranking).values()) == {1.0}

    def base(frame):
        return dict(zip(frame["member"], frame["shrunk_alpha"]))

    first = make_shuffled_scorer(base, seed=7)(ranking)
    second = make_shuffled_scorer(base, seed=7)(ranking)

    assert first == second
    assert sorted(first.values()) == sorted(ranking["shrunk_alpha"].tolist())
    assert main.MIN_NULL_PERMUTATIONS >= 999


def test_small_null_count_is_labeled_diagnostic_only():
    retrospective = {
        **_metric_run(),
        "mean_alpha_pct": 1.0,
        "total_return_pct": 2.0,
        "alpha_sharpe": 1.0,
    }
    passive = {**_metric_run(), "total_return_pct": 1.0}
    constant = {**_metric_run(), "alpha_sharpe": 0.0}
    nulls = pd.DataFrame(
        {
            "phase": ["retrospective_validation"],
            "alpha_sharpe": [0.1],
        }
    )
    with patch.object(main, "NULL_PERMUTATIONS", 99):
        passed, reasons = main._assess_retrospective(
            pd.Series(), True, retrospective, passive, constant, nulls, 0.01
        )
    assert passed is False
    assert any("diagnostic only" in reason for reason in reasons)


def test_manifest_labels_reused_history_and_locks_unconsumed_final(tmp_path):
    db = tmp_path / "db.duckdb"
    db.write_bytes(b"database-canary")
    run = _metric_run()
    selected = pd.Series(
        {
            "trial_id": 1,
            "scoring_fn": "shrunk_alpha",
            "top_n": 3,
            "min_buyers": 2,
            "allocation": "equal",
            "decay_lambda": 0.003,
            "alpha_p_value": 1.0,
            "bonferroni_significant": False,
            **{key: run[key] for key in main.METRIC_KEYS},
        }
    )
    trials = pd.DataFrame([selected])
    null_df = pd.DataFrame(
        [
            {
                "phase": "selection",
                "support_dates": run["support_dates"],
                **{key: run[key] for key in main.METRIC_KEYS},
            }
        ]
    )
    artifact = main._persist_artifacts(
        output_root=tmp_path / "runs",
        db_path=db,
        config=main._manifest_config(db),
        trials_df=trials,
        selected=selected,
        selected_selection_run=run,
        selection_periods=[],
        selection_rejections=[],
        retrospective_run=run,
        selection_constant=run,
        retrospective_constant=run,
        selection_spy=run,
        retrospective_spy=run,
        null_df=null_df,
        null_periods=[],
        null_rejections=[],
        family_gate_passed=False,
        null_empirical_p=1.0,
        retrospective_passed=False,
        reasons=["canary"],
    )
    manifest = json.loads((artifact / "manifest.json").read_text())

    assert manifest["phase_labels"]["validation"].startswith("retrospective")
    assert "untouched" not in json.dumps(manifest).lower()
    assert manifest["final_test"]["status"] == "locked_not_evaluated"
    assert manifest["final_test"]["consumed"] is False
    assert manifest["final_test"]["start"] == "2026-01-01"
    assert manifest["source_aggregate_sha256"]
    assert any(key.startswith("src/analyzer/") for key in manifest["source_sha256"])
    assert any(key.startswith("optimize_profit/") for key in manifest["source_sha256"])
    assert manifest["runtime"]["python"]
    assert manifest["runtime"]["dependencies"]["pandas"]
    assert "dirty" in manifest["git"]
    assert manifest["retrospective_db_sha256"]
    assert manifest["artifact_sha256"]
    assert not (artifact / "final_evaluation.json").exists()
