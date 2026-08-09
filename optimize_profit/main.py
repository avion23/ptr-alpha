"""Retrospective optimization that locks, but never consumes, a final test."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import itertools
import json
import os
import platform
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from analyzer import analysis
from analyzer.backtest import evaluate_backtest
from analyzer.database import Database

from optimize_profit.metrics import summarize_walk_forward
from optimize_profit.precompute import precompute_walk_forward_data
from optimize_profit.reporting import (
    print_retrospective,
    print_selection,
    print_verdict,
)
from optimize_profit.scoring import (
    SCORING_FUNCTIONS,
    make_shuffled_scorer,
    score_constant,
)
from optimize_profit.walk_forward import (
    _build_custom_ranking_dicts,
    _portfolio_return,
    _portfolio_weights,
    _score_candidates,
    run_walk_forward,
)

TX_START = pd.Timestamp("2021-10-07")
SELECTION_START = pd.Timestamp("2022-01-01")
RETROSPECTIVE_START = pd.Timestamp("2024-07-01")
RETROSPECTIVE_END = pd.Timestamp("2025-06-30")
HORIZON = 90
REBALANCE_DAYS = HORIZON
LOOKBACK_DAYS = 60
TRAINING_LOOKBACK_DAYS = 365
MIN_NULL_PERMUTATIONS = 999
SELECTION_FAMILY_ALPHA = 0.05
SELECTION_EMPIRICAL_ALPHA = 0.01
RETROSPECTIVE_ALPHA = 0.10
FINAL_NULL_ALPHA = 0.05
FINAL_ALPHA = 0.05
ENDPOINT_MAX_DELAY_DAYS = 7
BUY_SLIPPAGE_FACTOR = 1.001
SELL_SLIPPAGE_FACTOR = 0.999
NULL_PERMUTATIONS = int(
    os.environ.get("OPTIMIZE_PROFIT_NULL_PERMUTATIONS", MIN_NULL_PERMUTATIONS)
)

PARAM_GRID = {
    "scoring_fn": tuple(SCORING_FUNCTIONS),
    "top_n": (3, 5),
    "min_buyers": (1, 2, 3),
    "allocation": ("equal", "signal"),
    "decay_lambda": (0.003, 0.005),
}

METRIC_KEYS = (
    "total_return_pct",
    "spy_total_return_pct",
    "excess_total_return_pct",
    "mean_alpha_pct",
    "sharpe",
    "alpha_sharpe",
    "terminal_observation_drawdown_pct",
    "win_rate_pct",
    "n_periods",
    "n_cash_periods",
    "avg_positions",
    "requested_periods",
    "coverage_pct",
    "support_sha256",
)

DEPENDENCIES = (
    "pandas",
    "numpy",
    "scipy",
    "duckdb",
    "pydantic",
    "requests",
    "yfinance",
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluate-final",
        type=Path,
        help="Consume a locked manifest exactly once after the staged DB is complete",
    )
    args = parser.parse_args(argv)
    db_path = Path(os.environ.get("OPTIMIZE_PROFIT_DB", "data/congress.duckdb"))
    output_root = Path(
        os.environ.get("OPTIMIZE_PROFIT_OUTPUT", "data/optimize_profit_runs")
    )
    if args.evaluate_final is not None:
        evaluate_locked_final(args.evaluate_final, db_path)
        return
    run_retrospective(db_path, output_root)


def run_retrospective(db_path: Path, output_root: Path) -> Path:
    config = _manifest_config(db_path)
    selection_dates, retrospective_dates, final_dates = _phase_dates()
    retrospective_price_end = RETROSPECTIVE_END + pd.Timedelta(days=HORIZON + 7)

    with Database(db_path, read_only=True) as db:
        entry_prices, transactions, prices = _load_data_range(
            db, RETROSPECTIVE_END, retrospective_price_end
        )

    datasets = _build_decay_datasets(
        entry_prices,
        transactions,
        prices,
        selection_dates.append(retrospective_dates),
    )
    selection_sets = {
        decay: _subset_periods(precomputed, selection_dates)
        for decay, precomputed in datasets.items()
    }
    retrospective_sets = {
        decay: _subset_periods(precomputed, retrospective_dates)
        for decay, precomputed in datasets.items()
    }

    trials_df, selection_periods, selection_rejections = _run_selection_sweep(
        entry_prices, transactions, prices, selection_sets
    )
    selected, family_gate_passed = _select_frozen_config(trials_df)
    selected_params = _params_from_row(selected)
    decay = selected_params["decay_lambda"]

    selection_run = _run_config(
        entry_prices,
        transactions,
        prices,
        selection_sets[decay],
        selected_params,
    )
    retrospective_run = _run_config(
        entry_prices,
        transactions,
        prices,
        retrospective_sets[decay],
        selected_params,
    )
    selection_constant = _run_canary(
        entry_prices,
        transactions,
        prices,
        selection_sets[decay],
        selected_params,
        score_constant,
    )
    retrospective_constant = _run_canary(
        entry_prices,
        transactions,
        prices,
        retrospective_sets[decay],
        selected_params,
        score_constant,
    )
    selection_spy = _run_passive_benchmark(prices, selection_sets[decay])
    retrospective_spy = _run_passive_benchmark(prices, retrospective_sets[decay])
    null_df, null_periods, null_rejections = _run_shuffled_canaries(
        entry_prices,
        transactions,
        prices,
        selection_sets[decay],
        retrospective_sets[decay],
        selected_params,
    )

    _assert_identical_support(
        "selection",
        [selection_run, selection_spy, selection_constant]
        + _phase_null_runs(null_df, "selection", selection_sets[decay]),
    )
    _assert_identical_support(
        "retrospective_validation",
        [retrospective_run, retrospective_spy, retrospective_constant]
        + _phase_null_runs(
            null_df, "retrospective_validation", retrospective_sets[decay]
        ),
    )
    null_empirical_p = _empirical_null_p_value(selected, null_df)
    retrospective_passed, reasons = _assess_retrospective(
        selected,
        family_gate_passed,
        retrospective_run,
        retrospective_spy,
        retrospective_constant,
        null_df,
        null_empirical_p,
    )

    artifact_dir = _persist_artifacts(
        output_root=output_root,
        db_path=db_path,
        config=config,
        trials_df=trials_df,
        selected=selected,
        selected_selection_run=selection_run,
        selection_periods=selection_periods,
        selection_rejections=selection_rejections,
        retrospective_run=retrospective_run,
        selection_constant=selection_constant,
        retrospective_constant=retrospective_constant,
        selection_spy=selection_spy,
        retrospective_spy=retrospective_spy,
        null_df=null_df,
        null_periods=null_periods,
        null_rejections=null_rejections,
        family_gate_passed=family_gate_passed,
        null_empirical_p=null_empirical_p,
        retrospective_passed=retrospective_passed,
        reasons=reasons,
    )

    print_selection(selected, len(trials_df), null_empirical_p)
    print_retrospective(retrospective_run, retrospective_spy, retrospective_constant)
    print_verdict(retrospective_passed, reasons, artifact_dir, final_dates.min().date())
    return artifact_dir


def _phase_dates() -> tuple[pd.DatetimeIndex, pd.DatetimeIndex, pd.DatetimeIndex]:
    selection_last_as_of = RETROSPECTIVE_START - pd.Timedelta(days=HORIZON)
    selection = pd.date_range(
        SELECTION_START, selection_last_as_of, freq=f"{REBALANCE_DAYS}D"
    )
    retrospective = pd.date_range(
        RETROSPECTIVE_START, RETROSPECTIVE_END, freq=f"{REBALANCE_DAYS}D"
    )
    lock = _read_repository_lock()
    final = pd.DatetimeIndex(pd.to_datetime(lock["decision_dates"]))
    _assert_phase_embargo(selection, retrospective, "selection", "retrospective")
    _assert_phase_embargo(retrospective, final, "retrospective", "final_test")
    return selection, retrospective, final


def _assert_phase_embargo(left, right, left_label: str, right_label: str) -> None:
    if left.empty or right.empty:
        raise ValueError(f"Empty phase: {left_label} or {right_label}")
    outcome_complete = pd.Timestamp(left.max()) + pd.Timedelta(days=HORIZON)
    if outcome_complete > pd.Timestamp(right.min()):
        raise ValueError(
            f"Horizon embargo violated: {left_label} outcome completes "
            f"{outcome_complete.date()} after {right_label} starts {right.min().date()}"
        )


def _load_data_range(db: Database, tx_end, price_end):
    transactions = db.get_transactions_by_date_range(TX_START, tx_end)
    tickers = sorted(
        set(
            ticker
            for ticker in transactions["ticker"].dropna().unique()
            if isinstance(ticker, str)
        )
        | {"SPY"}
    )
    prices = db.get_prices(tickers, TX_START, price_end)
    entry_prices = db.get_entry_prices(tickers, TX_START, tx_end)
    return entry_prices, transactions, prices


def _build_decay_datasets(entry_prices, transactions, prices, as_of_dates):
    datasets = {}
    for decay in PARAM_GRID["decay_lambda"]:
        signals = analysis.calculate_signal_potential(
            entry_prices, prices, [HORIZON], decay_lambda=decay
        )
        datasets[decay] = precompute_walk_forward_data(
            signals,
            transactions,
            prices,
            as_of_dates,
            HORIZON,
            lookback_days=LOOKBACK_DAYS,
            training_lookback_days=TRAINING_LOOKBACK_DAYS,
            min_buyers_list=PARAM_GRID["min_buyers"],
        )
    return datasets


def _subset_periods(precomputed, scheduled_dates):
    wanted = {pd.Timestamp(date) for date in scheduled_dates}
    return {
        key: data
        for key, data in precomputed.items()
        if pd.Timestamp(data["as_of_ts"]) in wanted
    }


def _run_selection_sweep(entry_prices, transactions, prices, selection_sets):
    keys = tuple(PARAM_GRID)
    rows = []
    all_periods = []
    all_rejections = []
    expected_support = None
    combinations = itertools.product(*(PARAM_GRID[key] for key in keys))
    for trial_id, values in enumerate(combinations, start=1):
        params = dict(zip(keys, values))
        run = _run_config(
            entry_prices,
            transactions,
            prices,
            selection_sets[params["decay_lambda"]],
            params,
        )
        expected_support = _assert_support_matches(
            expected_support, run, f"selection trial {trial_id}"
        )
        p_value = _alpha_p_value(run["period_results"])
        rows.append(
            {
                "trial_id": trial_id,
                **params,
                **{key: run[key] for key in METRIC_KEYS},
                "alpha_p_value": p_value,
            }
        )
        all_periods.extend(
            {"trial_id": trial_id, "phase": "selection", **period}
            for period in run["period_results"]
        )
        all_rejections.extend(
            {"trial_id": trial_id, "phase": "selection", **rejection}
            for rejection in run["rejection_ledger"]
        )
    trials = pd.DataFrame(rows)
    threshold = SELECTION_FAMILY_ALPHA / len(trials)
    trials["bonferroni_threshold"] = threshold
    trials["bonferroni_significant"] = trials["alpha_p_value"] <= threshold
    return trials, all_periods, all_rejections


def _run_config(entry_prices, transactions, prices, precomputed, params):
    return run_walk_forward(
        entry_prices,
        transactions,
        prices,
        precomputed,
        scoring_fn=SCORING_FUNCTIONS[params["scoring_fn"]],
        top_n=int(params["top_n"]),
        min_buyers=int(params["min_buyers"]),
        allocation=str(params["allocation"]),
    )


def _run_canary(entry_prices, transactions, prices, periods, params, scorer):
    return run_walk_forward(
        entry_prices,
        transactions,
        prices,
        periods,
        scoring_fn=scorer,
        top_n=int(params["top_n"]),
        min_buyers=int(params["min_buyers"]),
        allocation=str(params["allocation"]),
    )


def _select_frozen_config(trials: pd.DataFrame) -> tuple[pd.Series, bool]:
    eligible = trials[trials["bonferroni_significant"] & (trials["mean_alpha_pct"] > 0)]
    family_gate_passed = not eligible.empty
    pool = eligible if family_gate_passed else trials
    selected = pool.sort_values(
        ["alpha_sharpe", "mean_alpha_pct"], ascending=False
    ).iloc[0]
    return selected, family_gate_passed


def _params_from_row(row: pd.Series) -> dict:
    return {
        "scoring_fn": str(row["scoring_fn"]),
        "top_n": int(row["top_n"]),
        "min_buyers": int(row["min_buyers"]),
        "allocation": str(row["allocation"]),
        "decay_lambda": float(row["decay_lambda"]),
    }


def _run_passive_benchmark(prices: pd.DataFrame, periods: dict) -> dict:
    returns = []
    for data in periods.values():
        as_of = data["as_of_ts"]
        evaluated = evaluate_backtest(
            pd.DataFrame({"ticker": ["SPY"], "signal_score": [1.0]}),
            prices,
            as_of,
            int(data["horizon"]),
        ).dropna(subset=["bt_return_pct"])
        if evaluated.empty:
            raise RuntimeError(f"SPY support unavailable for scheduled date {as_of}")
        value = float(evaluated["bt_return_pct"].iloc[0])
        returns.append(
            {
                "as_of_date": as_of.date(),
                "portfolio_return_pct": value,
                "spy_return_pct": value,
                "n_positions": 1,
                "status": "passive_spy",
                "reason": None,
            }
        )
    metrics = summarize_walk_forward(returns, periods_per_year=365.0 / HORIZON)
    support = [row["as_of_date"].isoformat() for row in returns]
    return {
        **metrics,
        "period_results": returns,
        "rejection_ledger": [],
        "requested_periods": len(periods),
        "coverage_pct": 100.0,
        "support_dates": support,
        "support_sha256": hashlib.sha256("|".join(support).encode()).hexdigest(),
    }


def _run_shuffled_canaries(
    entry_prices,
    transactions,
    prices,
    selection_periods,
    retrospective_periods,
    params,
):
    rows = []
    period_rows = []
    rejection_rows = []
    base = SCORING_FUNCTIONS[params["scoring_fn"]]
    for seed in range(NULL_PERMUTATIONS):
        scorer = make_shuffled_scorer(base, seed)
        for phase, periods in (
            ("selection", selection_periods),
            ("retrospective_validation", retrospective_periods),
        ):
            run = _run_canary(
                entry_prices, transactions, prices, periods, params, scorer
            )
            rows.append(
                {
                    "diagnostic_type": "period_aligned_member_permutation",
                    "phase": phase,
                    "seed": seed,
                    **{key: run[key] for key in METRIC_KEYS},
                    "support_dates": run["support_dates"],
                    "alpha_p_value": _alpha_p_value(run["period_results"]),
                }
            )
            period_rows.extend(
                {
                    "diagnostic_type": "period_aligned_member_permutation",
                    "phase": phase,
                    "seed": seed,
                    **row,
                }
                for row in run["period_results"]
            )
            rejection_rows.extend(
                {
                    "diagnostic_type": "period_aligned_member_permutation",
                    "phase": phase,
                    "seed": seed,
                    **row,
                }
                for row in run["rejection_ledger"]
            )
    return pd.DataFrame(rows), period_rows, rejection_rows


def _phase_null_runs(null_df, phase: str, periods: dict) -> list[dict]:
    expected_dates = [data["as_of_ts"].date().isoformat() for data in periods.values()]
    return [
        {
            "support_dates": dates,
            "coverage_pct": float(row.coverage_pct),
            "n_periods": int(row.n_periods),
            "requested_periods": int(row.requested_periods),
        }
        for row in null_df[null_df["phase"] == phase].itertuples()
        for dates in [
            row.support_dates if isinstance(row.support_dates, list) else expected_dates
        ]
    ]


def _assert_support_matches(expected, run: dict, label: str):
    if run["coverage_pct"] != 100.0 or run["n_periods"] != run["requested_periods"]:
        raise RuntimeError(f"{label} lacks 100% scheduled support")
    support = tuple(run["support_dates"])
    if expected is not None and support != expected:
        raise RuntimeError(f"{label} support differs from the family support")
    return support


def _assert_identical_support(label: str, runs: list[dict]) -> None:
    expected = None
    for index, run in enumerate(runs):
        expected = _assert_support_matches(expected, run, f"{label} run {index}")


def _alpha_p_value(period_results: list[dict]) -> float:
    if len(period_results) < 2:
        return 1.0
    alpha = np.asarray(
        [row["portfolio_return_pct"] - row["spy_return_pct"] for row in period_results],
        dtype=float,
    )
    if np.std(alpha, ddof=1) == 0:
        return 0.0 if alpha.mean() > 0 else 1.0
    return float(stats.ttest_1samp(alpha, 0.0, alternative="greater").pvalue)


def _empirical_null_p_value(selected, null_df) -> float:
    null_selection = null_df[null_df["phase"] == "selection"]["alpha_sharpe"]
    exceedances = int((null_selection >= float(selected["alpha_sharpe"])).sum())
    return (1 + exceedances) / (1 + len(null_selection))


def _assess_retrospective(
    selected,
    family_gate_passed,
    retrospective,
    retrospective_spy,
    retrospective_constant,
    null_df,
    null_empirical_p,
):
    reasons = []
    if not family_gate_passed:
        reasons.append("No selection trial survived the Bonferroni family gate")
    if NULL_PERMUTATIONS < MIN_NULL_PERMUTATIONS:
        reasons.append(
            f"Only {NULL_PERMUTATIONS} null permutations; result is diagnostic only"
        )
    if null_empirical_p > SELECTION_EMPIRICAL_ALPHA:
        reasons.append("Selection empirical permutation p-value exceeds 0.01")
    if retrospective["mean_alpha_pct"] <= 0:
        reasons.append("Retrospective mean opportunity alpha is not positive")
    if retrospective["total_return_pct"] <= retrospective_spy["total_return_pct"]:
        reasons.append("Frozen strategy did not beat passive SPY retrospectively")
    if retrospective["alpha_sharpe"] <= retrospective_constant["alpha_sharpe"]:
        reasons.append("Frozen strategy did not beat the constant-score canary")
    shuffled = null_df[null_df["phase"] == "retrospective_validation"]["alpha_sharpe"]
    if not shuffled.empty and retrospective["alpha_sharpe"] <= shuffled.quantile(0.99):
        reasons.append("Frozen strategy did not exceed the 99th percentile null scorer")
    if _alpha_p_value(retrospective["period_results"]) > RETROSPECTIVE_ALPHA:
        reasons.append("Retrospective one-sided alpha p-value exceeds 0.10")
    return not reasons, reasons


def _persist_artifacts(**kwargs) -> Path:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    artifact_dir = kwargs["output_root"] / run_id
    artifact_dir.mkdir(parents=True, exist_ok=False)

    kwargs["trials_df"].to_csv(artifact_dir / "selection_trials.csv", index=False)
    pd.DataFrame(kwargs["selection_periods"]).to_csv(
        artifact_dir / "selection_periods.csv", index=False
    )
    pd.DataFrame(kwargs["selection_rejections"]).to_csv(
        artifact_dir / "selection_rejections.csv", index=False
    )
    pd.DataFrame(kwargs["retrospective_run"]["period_results"]).to_csv(
        artifact_dir / "retrospective_validation_periods.csv", index=False
    )
    pd.DataFrame(kwargs["retrospective_run"]["rejection_ledger"]).to_csv(
        artifact_dir / "retrospective_validation_rejections.csv", index=False
    )
    kwargs["null_df"].drop(columns=["support_dates"]).to_csv(
        artifact_dir / "null_diagnostics.csv", index=False
    )
    pd.DataFrame(kwargs["null_periods"]).to_csv(
        artifact_dir / "null_periods.csv", index=False
    )
    pd.DataFrame(kwargs["null_rejections"]).to_csv(
        artifact_dir / "null_rejections.csv", index=False
    )
    _benchmark_rows(kwargs).to_csv(artifact_dir / "benchmarks.csv", index=False)

    config_json = json.dumps(kwargs["config"], sort_keys=True, separators=(",", ":"))
    source_hashes, source_aggregate = _source_hashes()
    git_state = _git_state()
    repository_lock_path = _canonical_final_lock_path()
    repository_lock = json.loads(repository_lock_path.read_text())
    repository_lock_sha = _sha256_file(repository_lock_path)
    selected = kwargs["selected"]
    manifest = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase_labels": {
            "selection": "retrospective_selection",
            "validation": "retrospective_validation_reused_history",
            "final": "locked_final_test_not_evaluated",
        },
        "verdict": "no_final_out_of_sample_profit_claim",
        "retrospective_gates_passed": kwargs["retrospective_passed"],
        "retrospective_reasons": kwargs["reasons"],
        "family_gate": {
            "method": "bonferroni",
            "n_trials": len(kwargs["trials_df"]),
            "threshold": SELECTION_FAMILY_ALPHA / len(kwargs["trials_df"]),
            "passed": kwargs["family_gate_passed"],
        },
        "null_test": {
            "method": "period_aligned_member_score_permutation",
            "n_permutations": NULL_PERMUTATIONS,
            "minimum_confirmatory_permutations": MIN_NULL_PERMUTATIONS,
            "empirical_p_value": kwargs["null_empirical_p"],
            "confirmatory": NULL_PERMUTATIONS >= MIN_NULL_PERMUTATIONS,
        },
        "selected_trial_id": int(selected["trial_id"]),
        "locked_config": _params_from_row(selected),
        "selection_metrics": {key: _json_value(selected[key]) for key in METRIC_KEYS},
        "retrospective_metrics": {
            key: _json_value(kwargs["retrospective_run"][key]) for key in METRIC_KEYS
        },
        "final_test": {
            "status": "repository_lock_not_evaluated",
            "repository_lock": str(repository_lock_path.relative_to(_repo_root())),
            "repository_lock_sha256": repository_lock_sha,
            "start": repository_lock["test_start"],
            "end": repository_lock["test_end"],
            "scheduled_as_of_dates": repository_lock["decision_dates"],
            "required_price_through": repository_lock["required_price_through"],
            "horizon_days": repository_lock["horizon_days"],
            "analytics_queried": False,
            "database_whole_file_hashed": True,
            "consumed": False,
        },
        "config": kwargs["config"],
        "config_sha256": hashlib.sha256(config_json.encode()).hexdigest(),
        "retrospective_db_sha256": _sha256_file(kwargs["db_path"]),
        "source_sha256": source_hashes,
        "source_aggregate_sha256": source_aggregate,
        "runtime": _runtime_manifest(),
        "git": git_state,
    }
    artifact_hashes = {
        path.name: _sha256_file(path) for path in sorted(artifact_dir.glob("*.csv"))
    }
    manifest["artifact_sha256"] = artifact_hashes
    (artifact_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return artifact_dir


def _benchmark_rows(kwargs) -> pd.DataFrame:
    rows = []
    for name in (
        "selection_constant",
        "retrospective_constant",
        "selection_spy",
        "retrospective_spy",
    ):
        phase, kind = name.split("_", 1)
        run = kwargs[name]
        rows.append(
            {
                "diagnostic_type": kind,
                "phase": phase,
                **{key: run[key] for key in METRIC_KEYS},
                "alpha_p_value": _alpha_p_value(run["period_results"]),
            }
        )
    return pd.DataFrame(rows)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _canonical_final_lock_path() -> Path:
    return _repo_root() / "optimize_profit" / "final_lock.json"


def _canonical_final_seal_path() -> Path:
    return _repo_root() / "optimize_profit" / "final_lock.v3.sha256"


def _read_repository_lock() -> dict:
    return json.loads(_canonical_final_lock_path().read_text())


def _canonical_consumption_ledger_path() -> Path:
    return _repo_root() / "data" / "optimize_profit_final_consumption.jsonl"


def _canonical_json_sha256(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    raise TypeError(f"Cannot JSON encode {type(value).__name__}")


def _load_final_seal() -> dict:
    seal_path = _canonical_final_seal_path()
    if not seal_path.exists():
        raise RuntimeError("Repository final lock is not sealed")
    seal = json.loads(seal_path.read_text())
    if set(seal) != {"lock_commit", "lock_sha256", "schema_version"}:
        raise RuntimeError("Final lock seal has unexpected fields")
    repo = _repo_root()
    history = subprocess.run(
        ["git", "log", "--format=%H", "--", str(seal_path.relative_to(repo))],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    if len(history) != 1:
        raise RuntimeError("Final lock seal must be added once and never modified")
    sealed_blob = subprocess.run(
        ["git", "show", f"{history[0]}:{seal_path.relative_to(repo)}"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if sealed_blob != seal_path.read_text():
        raise RuntimeError(
            "Working final lock seal differs from its immutable Git blob"
        )
    return seal


def _lock_blob_sha256(lock_commit: str, path: Path) -> str:
    blob = subprocess.run(
        ["git", "show", f"{lock_commit}:{path.relative_to(_repo_root())}"],
        cwd=_repo_root(),
        check=True,
        capture_output=True,
    ).stdout
    return hashlib.sha256(blob).hexdigest()


def _load_and_verify_repository_lock(requested_path: Path) -> tuple[dict, str]:
    canonical = _canonical_final_lock_path().resolve()
    if requested_path.resolve() != canonical:
        raise RuntimeError(f"Only repository lock {canonical} is accepted")
    seal = _load_final_seal()
    lock_sha = _sha256_file(canonical)
    if lock_sha != seal["lock_sha256"]:
        raise RuntimeError("Repository final lock SHA-256 mismatch")
    repo = _repo_root()
    if _lock_blob_sha256(seal["lock_commit"], canonical) != lock_sha:
        raise RuntimeError("Final lock differs from the blob in its creation commit")
    lock = json.loads(canonical.read_text())
    if _canonical_json_sha256(lock["locked_config"]) != lock["config_sha256"]:
        raise RuntimeError("Locked strategy config hash mismatch")
    current_sources, aggregate = _source_hashes()
    if current_sources != lock["sealed_source_sha256"]:
        raise RuntimeError(
            "Current source, including the verifier, differs from the lock"
        )
    if aggregate != lock["sealed_source_aggregate_sha256"]:
        raise RuntimeError("Current source aggregate differs from the lock")
    if _semantic_constants() != lock["semantic_constants"]:
        raise RuntimeError("Semantic constants differ from the final lock")
    if _locked_runtime_fingerprint() != lock["runtime_fingerprint"]:
        raise RuntimeError(
            "Runtime, platform, architecture, BLAS, or dependencies differ"
        )
    git = _git_state()
    if git["dirty"]:
        raise RuntimeError("Final evaluation requires a clean worktree")
    for ancestor in (seal["lock_commit"],):
        descendant = subprocess.run(
            ["git", "merge-base", "--is-ancestor", ancestor, git["commit"]],
            cwd=repo,
            check=False,
        )
        if descendant.returncode != 0:
            raise RuntimeError("Current commit is not a descendant of the lock commit")
    return lock, lock_sha


def _consumption_ref(lock_sha: str) -> str:
    return f"refs/optimize-profit/final-consumption/{lock_sha}"


def _consumption_anchor_commit(lock_sha: str) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "--verify", _consumption_ref(lock_sha)],
        cwd=_repo_root(),
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _serialize_consumption_event(event: dict) -> bytes:
    return (json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _hashed_consumption_event(payload: dict, previous_ledger: bytes) -> dict:
    event = {
        **payload,
        "previous_ledger_sha256": hashlib.sha256(previous_ledger).hexdigest(),
    }
    event["event_sha256"] = _canonical_json_sha256(event)
    return event


def _parse_and_validate_consumption_ledger(raw: bytes) -> list[dict]:
    if raw and not raw.endswith(b"\n"):
        raise RuntimeError("Final consumption ledger lacks a terminal newline")
    prefix = b""
    events = []
    for line in raw.splitlines(keepends=True):
        try:
            event = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                "Final consumption ledger contains invalid JSON"
            ) from exc
        claimed = event.get("event_sha256")
        payload = {key: value for key, value in event.items() if key != "event_sha256"}
        if event.get("previous_ledger_sha256") != hashlib.sha256(prefix).hexdigest():
            raise RuntimeError("Final consumption ledger byte-digest chain is broken")
        if _canonical_json_sha256(payload) != claimed:
            raise RuntimeError("Final consumption event hash is invalid")
        events.append(event)
        prefix += line
    return events


def _blob_contains_reservation(raw: bytes, lock_sha: str) -> bool:
    try:
        events = _parse_and_validate_consumption_ledger(raw)
    except RuntimeError:
        return False
    return any(
        event.get("event") == "reserved" and event.get("lock_sha256") == lock_sha
        for event in events
    )


def _ledger_blob_at_commit(commit: str) -> bytes | None:
    ledger_path = _canonical_consumption_ledger_path().relative_to(_repo_root())
    result = subprocess.run(
        ["git", "show", f"{commit}:{ledger_path}"],
        cwd=_repo_root(),
        check=False,
        capture_output=True,
    )
    return result.stdout if result.returncode == 0 else None


def _repository_has_reservation_object(lock_sha: str) -> bool:
    """Search refs, reflogs, and unreachable Git objects for a reservation."""
    if _consumption_anchor_commit(lock_sha) is not None:
        return True
    repo = _repo_root()
    ledger_path = str(_canonical_consumption_ledger_path().relative_to(repo))
    reachable = subprocess.run(
        ["git", "log", "--all", "--reflog", "--format=%H", "--", ledger_path],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    checked_commits = set()
    for commit in reachable:
        checked_commits.add(commit)
        raw = _ledger_blob_at_commit(commit)
        if raw is not None and _blob_contains_reservation(raw, lock_sha):
            return True

    fsck = subprocess.run(
        ["git", "fsck", "--full", "--unreachable", "--no-reflogs"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    for line in fsck:
        parts = line.split()
        if len(parts) < 3 or parts[0] not in {"unreachable", "dangling"}:
            continue
        object_type, object_id = parts[1], parts[2]
        if object_type == "commit" and object_id not in checked_commits:
            raw = _ledger_blob_at_commit(object_id)
        elif object_type == "blob":
            raw_result = subprocess.run(
                ["git", "cat-file", "blob", object_id],
                cwd=repo,
                check=False,
                capture_output=True,
            )
            raw = raw_result.stdout if raw_result.returncode == 0 else None
        else:
            raw = None
        if raw is not None and _blob_contains_reservation(raw, lock_sha):
            return True
    return False


def _anchored_ledger_bytes(lock_sha: str) -> bytes:
    anchor = _consumption_anchor_commit(lock_sha)
    if anchor is None:
        raise RuntimeError("Final consumption Git anchor is missing")
    raw = _ledger_blob_at_commit(anchor)
    if raw is None:
        raise RuntimeError("Final consumption anchor has no ledger blob")
    _parse_and_validate_consumption_ledger(raw)
    return raw


def _commit_consumption_anchor(
    lock_sha: str, event: dict, extra_paths: tuple[Path, ...] = ()
) -> None:
    repo = _repo_root()
    ledger = _canonical_consumption_ledger_path()
    previous_anchor = _consumption_anchor_commit(lock_sha)
    paths = (ledger, *extra_paths)
    relative_paths = [str(path.resolve().relative_to(repo.resolve())) for path in paths]
    subprocess.run(
        ["git", "add", "--force", "--", *relative_paths], cwd=repo, check=True
    )
    subprocess.run(
        [
            "git",
            "commit",
            "--no-gpg-sign",
            "-m",
            f"final-test: {event['event']} {event['event_sha256']}",
            "--",
            *relative_paths,
        ],
        cwd=repo,
        check=True,
    )
    commit = _git_state()["commit"]
    old = previous_anchor or "0" * 40
    updated = subprocess.run(
        ["git", "update-ref", _consumption_ref(lock_sha), commit, old],
        cwd=repo,
        check=False,
    )
    if updated.returncode != 0:
        raise RuntimeError("Could not atomically advance final consumption Git anchor")
    state = _git_state()
    if state["dirty"]:
        raise RuntimeError("Consumption event commit did not leave a clean worktree")
    committed = subprocess.run(
        ["git", "show", f"{commit}:{relative_paths[0]}"],
        cwd=repo,
        check=True,
        capture_output=True,
    ).stdout
    if committed != ledger.read_bytes():
        raise RuntimeError("Committed consumption ledger differs byte-for-byte")


def _reserve_final_consumption(lock_sha: str) -> str:
    """Atomically append and Git-anchor reservation before any DB access."""
    if _repository_has_reservation_object(lock_sha):
        raise RuntimeError("Locked final test has a reservation object in Git storage")
    ledger = _canonical_consumption_ledger_path()
    ledger.parent.mkdir(parents=True, exist_ok=True)
    reservation_id = str(uuid.uuid4())
    with ledger.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        raw = handle.read()
        events = _parse_and_validate_consumption_ledger(raw)
        if events or _repository_has_reservation_object(lock_sha):
            raise RuntimeError(
                "Locked final test already has a consumption reservation"
            )
        event = _hashed_consumption_event(
            {
                "event": "reserved",
                "lock_sha256": lock_sha,
                "reservation_id": reservation_id,
                "reserved_at_utc": datetime.now(timezone.utc).isoformat(),
                "git_commit_before_reservation": _git_state()["commit"],
            },
            raw,
        )
        serialized = _serialize_consumption_event(event)
        handle.seek(0, os.SEEK_END)
        handle.write(serialized)
        handle.flush()
        os.fsync(handle.fileno())
        _commit_consumption_anchor(lock_sha, event)
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return reservation_id


def _append_consumption_event(
    lock_sha: str,
    reservation_id: str,
    event_name: str,
    payload: dict,
    extra_paths: tuple[Path, ...] = (),
) -> None:
    ledger = _canonical_consumption_ledger_path()
    with ledger.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        raw = handle.read()
        anchored = _anchored_ledger_bytes(lock_sha)
        if raw != anchored:
            raise RuntimeError(
                "Working consumption ledger differs byte-for-byte from its anchor"
            )
        events = _parse_and_validate_consumption_ledger(anchored)
        reserved = any(
            event.get("event") == "reserved"
            and event.get("lock_sha256") == lock_sha
            and event.get("reservation_id") == reservation_id
            for event in events
        )
        if not reserved:
            raise RuntimeError("Final consumption reservation is missing")
        event = _hashed_consumption_event(
            {
                "event": event_name,
                "lock_sha256": lock_sha,
                "reservation_id": reservation_id,
                "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                **payload,
            },
            anchored,
        )
        handle.seek(0, os.SEEK_END)
        handle.write(_serialize_consumption_event(event))
        handle.flush()
        os.fsync(handle.fileno())
        _commit_consumption_anchor(lock_sha, event, extra_paths)
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def evaluate_locked_final(lock_path: Path, db_path: Path) -> Path:
    """Consume the repository final lock exactly once after all outcomes mature."""
    lock, lock_sha = _load_and_verify_repository_lock(lock_path)
    maturity = pd.Timestamp(lock["required_price_through"])
    if pd.Timestamp.now(tz="UTC").tz_localize(None).normalize() < maturity:
        raise RuntimeError(f"Final outcomes are immature until {maturity.date()}")

    reservation_id = _reserve_final_consumption(lock_sha)
    runtime = _runtime_manifest()
    try:
        db_sha = _sha256_file(db_path)
        final_dates = pd.DatetimeIndex(pd.to_datetime(lock["decision_dates"]))
        with Database(db_path, read_only=True) as db:
            entry_prices, transactions, prices = _load_data_range(
                db, pd.Timestamp(lock["test_end"]), maturity
            )
        if prices.empty or pd.Timestamp(prices.index.max()) < maturity - pd.Timedelta(
            days=7
        ):
            raise RuntimeError(
                f"Final price data does not reach exact maturity {maturity.date()}"
            )
        chosen = lock["locked_config"]
        decay = float(chosen["decay_lambda"])
        signals = analysis.calculate_signal_potential(
            entry_prices, prices, [HORIZON], decay_lambda=decay
        )
        precomputed = precompute_walk_forward_data(
            signals,
            transactions,
            prices,
            final_dates,
            HORIZON,
            lookback_days=LOOKBACK_DAYS,
            training_lookback_days=TRAINING_LOOKBACK_DAYS,
            min_buyers_list=PARAM_GRID["min_buyers"],
        )
        family = _run_strict_final_family(
            entry_prices, transactions, prices, precomputed, chosen, lock
        )
        _, post_run_lock_sha = _load_and_verify_repository_lock(lock_path)
        if post_run_lock_sha != lock_sha:
            raise RuntimeError("Final lock changed during evaluation")
        if _sha256_file(db_path) != db_sha:
            raise RuntimeError("Database file changed during final evaluation")
        result = _final_claim_result(lock, lock_sha, db_sha, runtime, family)
        result["reservation_id"] = reservation_id
        result["consumption_ledger"] = str(_canonical_consumption_ledger_path())
        output = _repo_root() / "data" / "optimize_profit_final_result.json"
        if output.exists():
            raise RuntimeError("Canonical final result already exists")
        with output.open("x") as handle:
            json.dump(result, handle, indent=2, sort_keys=True, default=_json_default)
            handle.write("\n")
        _append_consumption_event(
            lock_sha,
            reservation_id,
            "completed",
            {"result_sha256": _sha256_file(output), "verdict": result["verdict"]},
            extra_paths=(output,),
        )
        return output
    except Exception as exc:
        _append_consumption_event(
            lock_sha,
            reservation_id,
            "failed",
            {"error_type": type(exc).__name__, "error": str(exc)},
        )
        raise


def _run_strict_final_family(
    entry_prices, transactions, prices, precomputed, params, lock
) -> dict:
    base = SCORING_FUNCTIONS[params["scoring_fn"]]
    strategy = _run_strict_final(precomputed, prices, base, params)
    constant = _run_strict_final(precomputed, prices, score_constant, params)
    null_metrics = []
    for seed in range(int(lock["final_null_permutations"])):
        scorer = make_shuffled_scorer(base, seed)
        null_metrics.append(_run_strict_final(precomputed, prices, scorer, params))
    _assert_identical_support("strict_final", [strategy, constant, *null_metrics])
    null_sharpes = np.asarray([run["alpha_sharpe"] for run in null_metrics])
    empirical_p = (1 + int((null_sharpes >= strategy["alpha_sharpe"]).sum())) / (
        1 + len(null_sharpes)
    )
    return {
        "strategy": strategy,
        "constant": constant,
        "null_alpha_sharpes": null_sharpes.tolist(),
        "null_empirical_p_value": empirical_p,
        "null_permutations": len(null_sharpes),
    }


def _run_strict_final(precomputed: dict, prices: pd.DataFrame, scorer, params) -> dict:
    all_returns = []
    position_rows = []
    top_n = int(params["top_n"])
    min_buyers = int(params["min_buyers"])
    for data in precomputed.values():
        if data.get("status") != "ready":
            raise RuntimeError(
                f"Final scheduled period {data['as_of_ts'].date()} is not ready: "
                f"{data.get('reason')}"
            )
        ranking_dicts = _build_custom_ranking_dicts(data["member_rankings"], scorer)
        candidates = data["candidate_tickers"].get(min_buyers, [])
        scored, _ = _score_candidates(
            candidates,
            data["recent_trades"],
            data["training"],
            data["horizon"],
            5.0,
            data["member_rankings"],
            min_buyers,
            data["ticker_perf_signals"],
            ranking_dicts,
        )
        selected = (
            pd.DataFrame(scored)
            .sort_values("signal_score", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )
        if len(selected) != top_n:
            raise RuntimeError(
                f"Final period {data['as_of_ts'].date()} has {len(selected)}/{top_n} positions"
            )
        weights = _portfolio_weights(
            selected["signal_score"].to_numpy(), str(params["allocation"])
        )
        ticker_returns, spy_return, endpoint_rows = _strict_endpoint_returns(
            selected["ticker"].astype(str).tolist(),
            prices,
            data["as_of_ts"],
            int(data["horizon"]),
        )
        portfolio_return = _portfolio_return(ticker_returns, weights) * 100
        all_returns.append(
            {
                "as_of_date": data["as_of_ts"].date().isoformat(),
                "portfolio_return_pct": portfolio_return,
                "spy_return_pct": spy_return,
                "n_positions": top_n,
            }
        )
        for ticker, ticker_return, weight, endpoints in zip(
            selected["ticker"], ticker_returns, weights, endpoint_rows
        ):
            position_rows.append(
                {
                    "as_of_date": data["as_of_ts"].date().isoformat(),
                    "ticker": str(ticker),
                    "weight": float(weight),
                    "return_pct": float(ticker_return),
                    **endpoints,
                }
            )
    metrics = summarize_walk_forward(all_returns, periods_per_year=365.0 / HORIZON)
    support = [row["as_of_date"] for row in all_returns]
    return {
        **metrics,
        "period_results": all_returns,
        "position_results": position_rows,
        "requested_periods": len(precomputed),
        "coverage_pct": 100.0,
        "support_dates": support,
        "support_sha256": hashlib.sha256("|".join(support).encode()).hexdigest(),
    }


def _strict_endpoint_returns(tickers, prices, as_of, horizon):
    spy = prices["SPY"].dropna()
    entry_target = pd.Timestamp(as_of)
    exit_target = entry_target + pd.Timedelta(days=horizon)
    entry_candidates = spy.index[spy.index >= entry_target]
    exit_candidates = spy.index[spy.index >= exit_target]
    if entry_candidates.empty or exit_candidates.empty:
        raise RuntimeError(f"SPY endpoints unavailable for {entry_target.date()}")
    entry_date = pd.Timestamp(entry_candidates[0])
    exit_date = pd.Timestamp(exit_candidates[0])
    if entry_date > entry_target + pd.Timedelta(days=ENDPOINT_MAX_DELAY_DAYS):
        raise RuntimeError(f"SPY entry endpoint too late for {entry_target.date()}")
    if exit_date > exit_target + pd.Timedelta(days=ENDPOINT_MAX_DELAY_DAYS):
        raise RuntimeError(f"SPY exit endpoint too late for {exit_target.date()}")
    if entry_date not in spy.index or exit_date not in spy.index:
        raise RuntimeError("Exact SPY endpoint coverage is missing")
    spy_entry = float(spy.loc[entry_date])
    spy_exit = float(spy.loc[exit_date])
    spy_return = (
        spy_exit * SELL_SLIPPAGE_FACTOR / (spy_entry * BUY_SLIPPAGE_FACTOR) - 1
    ) * 100

    returns = []
    endpoint_rows = []
    for ticker in tickers:
        if ticker not in prices.columns:
            raise RuntimeError(f"Final ticker {ticker} has no price column")
        series = prices[ticker]
        if entry_date not in series.index or exit_date not in series.index:
            raise RuntimeError(f"Final ticker {ticker} lacks exact endpoint dates")
        entry = series.loc[entry_date]
        exit_price = series.loc[exit_date]
        if pd.isna(entry) or pd.isna(exit_price) or entry <= 0 or exit_price <= 0:
            raise RuntimeError(f"Final ticker {ticker} has invalid endpoint prices")
        returns.append(
            (
                float(exit_price)
                * SELL_SLIPPAGE_FACTOR
                / (float(entry) * BUY_SLIPPAGE_FACTOR)
                - 1
            )
            * 100
        )
        endpoint_rows.append(
            {
                "entry_date": entry_date.date().isoformat(),
                "exit_date": exit_date.date().isoformat(),
                "ticker_entry_price": float(entry),
                "ticker_exit_price": float(exit_price),
                "spy_entry_price": spy_entry,
                "spy_exit_price": spy_exit,
            }
        )
    return np.asarray(returns), spy_return, endpoint_rows


def _final_claim_result(lock, lock_sha, db_sha, runtime, family) -> dict:
    strategy = family["strategy"]
    constant = family["constant"]
    gates = {
        "adequate_observations": strategy["n_periods"]
        >= int(lock["minimum_final_observations"]),
        "pre_final_bonferroni_gate": bool(
            lock["pre_final_evidence"]["bonferroni_passed"]
        ),
        "pre_final_permutation_gate": float(
            lock["pre_final_evidence"]["empirical_p_value"]
        )
        <= lock["claim_gates"]["pre_final_permutation_p_max"],
        "positive_final_alpha": strategy["mean_alpha_pct"] > 0,
        "positive_return_vs_spy": strategy["total_return_pct"]
        > strategy["spy_total_return_pct"],
        "beats_final_constant": strategy["alpha_sharpe"] > constant["alpha_sharpe"],
        "final_null_gate": family["null_empirical_p_value"]
        <= lock["claim_gates"]["final_null_p_max"],
        "final_alpha_inference": _alpha_p_value(strategy["period_results"])
        <= lock["claim_gates"]["final_alpha_p_max"],
        "exact_period_support": strategy["n_periods"] == strategy["requested_periods"],
        "exact_position_count": all(
            row["n_positions"] == int(lock["locked_config"]["top_n"])
            for row in strategy["period_results"]
        ),
    }
    current_sources, current_source_aggregate = _source_hashes()
    return {
        "lock_sha256": lock_sha,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "db_whole_file_sha256": db_sha,
        "locked_config_sha256": lock["config_sha256"],
        "locked_source_sha256": lock["sealed_source_sha256"],
        "locked_source_aggregate_sha256": lock["sealed_source_aggregate_sha256"],
        "evaluation_source_sha256": current_sources,
        "evaluation_source_aggregate_sha256": current_source_aggregate,
        "runtime": runtime,
        "runtime_fingerprint": _locked_runtime_fingerprint(),
        "git": _git_state(),
        "strategy_metrics": {key: strategy[key] for key in METRIC_KEYS},
        "constant_metrics": {key: constant[key] for key in METRIC_KEYS},
        "strategy_period_results": strategy["period_results"],
        "strategy_position_results": strategy["position_results"],
        "final_null_permutations": family["null_permutations"],
        "final_null_empirical_p_value": family["null_empirical_p_value"],
        "final_alpha_p_value": _alpha_p_value(strategy["period_results"]),
        "claim_gates": gates,
        "verdict": "final_out_of_sample_gates_passed"
        if all(gates.values())
        else "no_validated_profit_claim",
    }


def _semantic_constants() -> dict:
    return {
        "tx_start": TX_START.date().isoformat(),
        "selection_start": SELECTION_START.date().isoformat(),
        "retrospective_start": RETROSPECTIVE_START.date().isoformat(),
        "retrospective_end": RETROSPECTIVE_END.date().isoformat(),
        "horizon_days": HORIZON,
        "rebalance_days": REBALANCE_DAYS,
        "lookback_days": LOOKBACK_DAYS,
        "training_lookback_days": TRAINING_LOOKBACK_DAYS,
        "minimum_null_permutations": MIN_NULL_PERMUTATIONS,
        "selection_family_alpha": SELECTION_FAMILY_ALPHA,
        "selection_empirical_alpha": SELECTION_EMPIRICAL_ALPHA,
        "retrospective_alpha": RETROSPECTIVE_ALPHA,
        "final_null_alpha": FINAL_NULL_ALPHA,
        "final_alpha": FINAL_ALPHA,
        "endpoint_max_delay_days": ENDPOINT_MAX_DELAY_DAYS,
        "buy_slippage_factor": BUY_SLIPPAGE_FACTOR,
        "sell_slippage_factor": SELL_SLIPPAGE_FACTOR,
        "parameter_grid": {key: list(value) for key, value in PARAM_GRID.items()},
        "metric_keys": list(METRIC_KEYS),
        "dependencies": list(DEPENDENCIES),
    }


def _locked_runtime_fingerprint() -> dict:
    import numpy.__config__ as numpy_config

    runtime = _runtime_manifest()
    config = numpy_config.CONFIG
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "dependencies": runtime["dependencies"],
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "architecture": list(platform.architecture()),
            "byteorder": sys.byteorder,
        },
        "numpy_machine": config.get("Machine Information"),
        "blas": config.get("Build Dependencies", {}).get("blas"),
        "lapack": config.get("Build Dependencies", {}).get("lapack"),
        "simd": config.get("SIMD Extensions"),
    }


def _manifest_config(db_path: Path) -> dict:
    selection, retrospective, final = _phase_dates()
    return {
        "tx_start": TX_START.date().isoformat(),
        "selection_dates": [date.date().isoformat() for date in selection],
        "retrospective_validation_dates": [
            date.date().isoformat() for date in retrospective
        ],
        "locked_final_test_dates": [date.date().isoformat() for date in final],
        "horizon_days": HORIZON,
        "rebalance_days": REBALANCE_DAYS,
        "lookback_days": LOOKBACK_DAYS,
        "training_lookback_days": TRAINING_LOOKBACK_DAYS,
        "parameter_grid": {key: list(value) for key, value in PARAM_GRID.items()},
        "null_permutations": NULL_PERMUTATIONS,
        "db_path": str(db_path.resolve()),
    }


def _source_hashes() -> tuple[dict[str, str], str]:
    repo = Path(__file__).resolve().parents[1]
    paths = sorted((repo / "src" / "analyzer").rglob("*.py")) + sorted(
        (repo / "optimize_profit").rglob("*.py")
    )
    hashes = {str(path.relative_to(repo)): _sha256_file(path) for path in paths}
    aggregate = hashlib.sha256(
        "".join(f"{name}:{digest}\n" for name, digest in hashes.items()).encode()
    ).hexdigest()
    return hashes, aggregate


def _runtime_manifest() -> dict:
    versions = {}
    for package in DEPENDENCIES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "dependencies": versions,
    }


def _git_state() -> dict:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_repo_root(),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        cwd=_repo_root(),
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {
        "commit": commit,
        "dirty": bool(status),
        "status_porcelain": status.splitlines(),
        "status_sha256": hashlib.sha256(status.encode()).hexdigest(),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_value(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    return value


if __name__ == "__main__":
    main()
