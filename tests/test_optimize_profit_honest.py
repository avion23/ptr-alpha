from __future__ import annotations

import json
import hashlib
import threading
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
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
    lock = json.loads(main._canonical_final_lock_path().read_text())
    assert (
        final.tolist()
        == pd.DatetimeIndex(pd.to_datetime(lock["decision_dates"])).tolist()
    )
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
    assert manifest["final_test"]["status"] == "repository_lock_not_evaluated"
    assert manifest["final_test"]["consumed"] is False
    lock = json.loads(main._canonical_final_lock_path().read_text())
    assert manifest["final_test"]["start"] == "2026-10-01"
    assert manifest["final_test"]["scheduled_as_of_dates"] == lock["decision_dates"]
    assert manifest["final_test"]["analytics_queried"] is False
    assert manifest["final_test"]["database_whole_file_hashed"] is True
    assert manifest["source_aggregate_sha256"]
    assert any(key.startswith("src/analyzer/") for key in manifest["source_sha256"])
    assert any(key.startswith("optimize_profit/") for key in manifest["source_sha256"])
    assert manifest["runtime"]["python"]
    assert manifest["runtime"]["dependencies"]["pandas"]
    assert "dirty" in manifest["git"]
    assert manifest["retrospective_db_sha256"]
    assert manifest["artifact_sha256"]
    assert not (artifact / "final_evaluation.json").exists()


def test_repository_final_lock_has_eight_post_precommit_mature_observations():
    lock_path = main._canonical_final_lock_path()
    lock = json.loads(lock_path.read_text())
    dates = pd.DatetimeIndex(pd.to_datetime(lock["decision_dates"]))

    assert len(dates) >= 8
    assert dates.is_unique
    assert dates.min() > pd.Timestamp("2026-08-09")
    assert (dates[1:] - dates[:-1]).min() >= pd.Timedelta(days=main.HORIZON)
    assert dates.max() + pd.Timedelta(days=main.HORIZON) == pd.Timestamp(
        lock["required_price_through"]
    )
    assert lock["minimum_final_observations"] >= 8
    assert lock["final_null_permutations"] >= 999


def test_final_lock_tamper_and_noncanonical_path_fail_closed(tmp_path):
    canonical = main._canonical_final_lock_path()
    original_sha = main._sha256_file(canonical)
    tampered = json.loads(canonical.read_text())
    tampered["locked_config"]["top_n"] = 5
    path = tmp_path / "final_lock.json"
    path.write_text(json.dumps(tampered))

    with pytest.raises(RuntimeError, match="Only repository lock"):
        main._load_and_verify_repository_lock(path)
    with (
        patch.object(main, "_canonical_final_lock_path", return_value=path),
        patch.object(
            main,
            "_load_final_seal",
            return_value={
                "schema_version": 1,
                "lock_sha256": original_sha,
                "lock_commit": "canary",
            },
        ),
    ):
        with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
            main._load_and_verify_repository_lock(path)


def test_final_runtime_mismatch_fails_before_git_or_database():
    lock_path = main._canonical_final_lock_path()
    lock_sha = main._sha256_file(lock_path)
    lock = json.loads(lock_path.read_text())
    wrong_runtime = {**lock["runtime_fingerprint"], "python_version": "0.0"}
    with (
        patch.object(
            main,
            "_load_final_seal",
            return_value={
                "schema_version": 1,
                "lock_sha256": lock_sha,
                "lock_commit": "canary",
            },
        ),
        patch.object(main, "_locked_runtime_fingerprint", return_value=wrong_runtime),
        patch.object(main, "_lock_blob_sha256", return_value=lock_sha),
        patch.object(
            main,
            "_source_hashes",
            return_value=(
                lock["sealed_source_sha256"],
                lock["sealed_source_aggregate_sha256"],
            ),
        ),
    ):
        with pytest.raises(
            RuntimeError, match="Runtime, platform, architecture, BLAS, or dependencies"
        ):
            main._load_and_verify_repository_lock(lock_path)


def test_final_maturity_fails_before_consumption_reservation():
    lock = json.loads(main._canonical_final_lock_path().read_text())
    with (
        patch.object(
            main, "_load_and_verify_repository_lock", return_value=(lock, "sha")
        ),
        patch.object(main, "_reserve_final_consumption") as reserve,
    ):
        with pytest.raises(RuntimeError, match="immature"):
            main.evaluate_locked_final(
                main._canonical_final_lock_path(), Path("missing")
            )
    reserve.assert_not_called()


def test_final_endpoint_coverage_requires_each_ticker_and_spy_exactly():
    index = pd.DatetimeIndex(["2029-01-02", "2029-04-02"])
    complete = pd.DataFrame({"SPY": [100.0, 110.0], "AAA": [50.0, 55.0]}, index=index)
    returns, spy_return, endpoint_rows = main._strict_endpoint_returns(
        ["AAA"], complete, pd.Timestamp("2029-01-01"), 90
    )
    assert len(returns) == 1
    assert spy_return > 0
    assert endpoint_rows == [
        {
            "entry_date": "2029-01-02",
            "exit_date": "2029-04-02",
            "ticker_entry_price": 50.0,
            "ticker_exit_price": 55.0,
            "spy_entry_price": 100.0,
            "spy_exit_price": 110.0,
        }
    ]

    missing = complete.copy()
    missing.loc[pd.Timestamp("2029-04-02"), "AAA"] = float("nan")
    with pytest.raises(RuntimeError, match="invalid endpoint prices"):
        main._strict_endpoint_returns(["AAA"], missing, pd.Timestamp("2029-01-01"), 90)


def test_append_only_consumption_reservation_is_atomic_and_irreversible(tmp_path):
    ledger = tmp_path / "consumption.jsonl"
    barrier = threading.Barrier(2)

    def attempt_reservation():
        barrier.wait()
        try:
            return ("reserved", main._reserve_final_consumption("lock-sha"))
        except RuntimeError as exc:
            return ("rejected", str(exc))

    with (
        patch.object(main, "_canonical_consumption_ledger_path", return_value=ledger),
        patch.object(main, "_git_state", return_value={"commit": "canary"}),
        patch.object(main, "_repository_has_reservation_object", return_value=False),
        patch.object(main, "_consumption_anchor_commit", return_value=None),
        patch.object(
            main,
            "_anchored_ledger_bytes",
            side_effect=lambda _: ledger.read_bytes(),
        ),
        patch.object(main, "_commit_consumption_anchor"),
        ThreadPoolExecutor(max_workers=2) as pool,
    ):
        outcomes = list(pool.map(lambda _: attempt_reservation(), range(2)))
        reservation = next(value for status, value in outcomes if status == "reserved")
        assert sum(status == "reserved" for status, _ in outcomes) == 1
        assert sum(status == "rejected" for status, _ in outcomes) == 1
        main._append_consumption_event(
            "lock-sha", reservation, "failed", {"error": "canary"}
        )

    events = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert [event["event"] for event in events] == ["reserved", "failed"]
    assert all(event["reservation_id"] == reservation for event in events)


def test_final_claim_requires_power_null_constant_and_multiplicity_gates():
    lock = json.loads(main._canonical_final_lock_path().read_text())
    periods = [
        {
            "as_of_date": "2027-01-01",
            "portfolio_return_pct": 2.0,
            "spy_return_pct": 1.0,
            "n_positions": 3,
        }
    ] * 7
    strategy = {
        **_metric_run(tuple(str(i) for i in range(7))),
        "n_periods": 7,
        "requested_periods": 7,
        "mean_alpha_pct": 1.0,
        "total_return_pct": 10.0,
        "spy_total_return_pct": 5.0,
        "alpha_sharpe": 1.0,
        "period_results": periods,
        "position_results": [],
    }
    constant = {**strategy, "alpha_sharpe": 0.0}
    family = {
        "strategy": strategy,
        "constant": constant,
        "null_permutations": 999,
        "null_empirical_p_value": 0.01,
    }
    result = main._final_claim_result(lock, "sha", "db", {"python": "canary"}, family)

    assert result["claim_gates"]["adequate_observations"] is False
    assert result["claim_gates"]["pre_final_bonferroni_gate"] is False
    assert result["verdict"] == "no_validated_profit_claim"
    assert result["runtime"] == {"python": "canary"}


def test_sealed_source_includes_exact_main_and_semantic_constants():
    lock = json.loads(main._canonical_final_lock_path().read_text())
    sources, aggregate = main._source_hashes()

    if lock["sealed_source_sha256"] == sources:
        assert lock["sealed_source_aggregate_sha256"] == aggregate
    else:
        seal = main._load_final_seal()
        locked_sources = {}
        for source_path in lock["sealed_source_sha256"]:
            blob = subprocess.run(
                ["git", "show", f"{seal['lock_commit']}:{source_path}"],
                cwd=main._repo_root(),
                check=True,
                capture_output=True,
            ).stdout
            locked_sources[source_path] = hashlib.sha256(blob).hexdigest()
        assert lock["sealed_source_sha256"] == locked_sources
        source_order = sorted(
            name for name in locked_sources if name.startswith("src/analyzer/")
        ) + sorted(
            name for name in locked_sources if name.startswith("optimize_profit/")
        )
        locked_aggregate = hashlib.sha256(
            "".join(
                f"{name}:{locked_sources[name]}\n" for name in source_order
            ).encode()
        ).hexdigest()
        assert lock["sealed_source_aggregate_sha256"] == locked_aggregate

    assert "optimize_profit/main.py" in lock["sealed_source_sha256"]
    assert lock["semantic_constants"] == main._semantic_constants()
    assert lock["runtime_fingerprint"] == main._locked_runtime_fingerprint()


def _init_git_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Canary"], cwd=path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "canary@example.test"], cwd=path, check=True
    )
    (path / "base.txt").write_text("base\n")
    subprocess.run(["git", "add", "base.txt"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=path, check=True)


def test_reservation_bytes_are_anchored_and_survive_ref_and_ledger_delete(tmp_path):
    _init_git_repo(tmp_path)
    ledger = tmp_path / "data" / "optimize_profit_final_consumption.jsonl"
    lock_sha = "a" * 64
    with (
        patch.object(main, "_repo_root", return_value=tmp_path),
        patch.object(main, "_canonical_consumption_ledger_path", return_value=ledger),
    ):
        reservation = main._reserve_final_consumption(lock_sha)
        anchor = main._consumption_anchor_commit(lock_sha)
        assert (
            anchor
            == subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=tmp_path,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        assert main._git_state()["dirty"] is False
        anchored_bytes = main._anchored_ledger_bytes(lock_sha)
        assert anchored_bytes == ledger.read_bytes()
        event = json.loads(anchored_bytes)
        assert event["reservation_id"] == reservation
        assert event["previous_ledger_sha256"] == hashlib.sha256(b"").hexdigest()

        forged = dict(event)
        forged["reserved_at_utc"] = "2099-01-01T00:00:00+00:00"
        forged.pop("event_sha256")
        forged["event_sha256"] = main._canonical_json_sha256(forged)
        ledger.write_bytes(main._serialize_consumption_event(forged))
        with pytest.raises(RuntimeError, match="differs byte-for-byte"):
            main._append_consumption_event(
                lock_sha, reservation, "failed", {"error": "forged timestamp"}
            )

        ledger.write_bytes(anchored_bytes)
        subprocess.run(
            ["git", "update-ref", "-d", main._consumption_ref(lock_sha)],
            cwd=tmp_path,
            check=True,
        )
        ledger.unlink()
        assert main._consumption_anchor_commit(lock_sha) is None
        assert main._repository_has_reservation_object(lock_sha) is True
        with pytest.raises(RuntimeError, match="reservation object in Git storage"):
            main._reserve_final_consumption(lock_sha)


def test_dangling_reservation_commit_is_found_by_git_fsck(tmp_path):
    _init_git_repo(tmp_path)
    lock_sha = "b" * 64
    reservation = main._hashed_consumption_event(
        {
            "event": "reserved",
            "lock_sha256": lock_sha,
            "reservation_id": "dangling-canary",
            "reserved_at_utc": "2026-08-09T00:00:00+00:00",
            "git_commit_before_reservation": "canary",
        },
        b"",
    )
    raw = main._serialize_consumption_event(reservation)
    blob = (
        subprocess.run(
            ["git", "hash-object", "-w", "--stdin"],
            cwd=tmp_path,
            input=raw,
            check=True,
            capture_output=True,
        )
        .stdout.decode()
        .strip()
    )
    data_tree = subprocess.run(
        ["git", "mktree"],
        cwd=tmp_path,
        input=f"100644 blob {blob}\toptimize_profit_final_consumption.jsonl\n",
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    root_tree = subprocess.run(
        ["git", "mktree"],
        cwd=tmp_path,
        input=f"040000 tree {data_tree}\tdata\n",
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dangling_commit = subprocess.run(
        ["git", "commit-tree", root_tree],
        cwd=tmp_path,
        input="unreferenced reservation canary\n",
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert (
        subprocess.run(
            ["git", "branch", "--contains", dangling_commit],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        == ""
    )

    ledger = tmp_path / "data" / "optimize_profit_final_consumption.jsonl"
    with (
        patch.object(main, "_repo_root", return_value=tmp_path),
        patch.object(main, "_canonical_consumption_ledger_path", return_value=ledger),
    ):
        assert main._consumption_anchor_commit(lock_sha) is None
        assert main._repository_has_reservation_object(lock_sha) is True
        with pytest.raises(RuntimeError, match="reservation object in Git storage"):
            main._reserve_final_consumption(lock_sha)


def test_coordinated_lock_and_seal_tamper_is_rejected_by_git_history(tmp_path):
    _init_git_repo(tmp_path)
    optimize = tmp_path / "optimize_profit"
    optimize.mkdir()
    lock_path = optimize / "final_lock.json"
    seal_path = optimize / "final_lock.sha256"
    lock_path.write_text("{}\n")
    subprocess.run(
        ["git", "add", str(lock_path.relative_to(tmp_path))], cwd=tmp_path, check=True
    )
    subprocess.run(["git", "commit", "-qm", "lock"], cwd=tmp_path, check=True)
    lock_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    seal_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "lock_sha256": main._sha256_file(lock_path),
                "lock_commit": lock_commit,
            },
            sort_keys=True,
        )
        + "\n"
    )
    subprocess.run(
        ["git", "add", str(seal_path.relative_to(tmp_path))], cwd=tmp_path, check=True
    )
    subprocess.run(["git", "commit", "-qm", "seal"], cwd=tmp_path, check=True)
    with (
        patch.object(main, "_repo_root", return_value=tmp_path),
        patch.object(main, "_canonical_final_seal_path", return_value=seal_path),
    ):
        assert main._load_final_seal()["lock_commit"] == lock_commit
        lock_path.write_text('{"tampered":true}\n')
        tampered = json.loads(seal_path.read_text())
        tampered["lock_sha256"] = main._sha256_file(lock_path)
        seal_path.write_text(json.dumps(tampered, sort_keys=True) + "\n")
        subprocess.run(["git", "add", "optimize_profit"], cwd=tmp_path, check=True)
        subprocess.run(
            ["git", "commit", "-qm", "coordinated tamper"], cwd=tmp_path, check=True
        )
        with pytest.raises(RuntimeError, match="added once and never modified"):
            main._load_final_seal()
