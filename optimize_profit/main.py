"""Retrospective optimization that locks, but never consumes, a final test."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import itertools
import json
import os
import platform
import subprocess
import sys
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
from optimize_profit.walk_forward import run_walk_forward

TX_START = pd.Timestamp("2021-10-07")
SELECTION_START = pd.Timestamp("2022-01-01")
RETROSPECTIVE_START = pd.Timestamp("2024-07-01")
RETROSPECTIVE_END = pd.Timestamp("2025-06-30")
FINAL_TEST_START = pd.Timestamp("2026-01-01")
FINAL_TEST_END = pd.Timestamp("2026-06-30")
HORIZON = 90
REBALANCE_DAYS = HORIZON
LOOKBACK_DAYS = 60
TRAINING_LOOKBACK_DAYS = 365
MIN_NULL_PERMUTATIONS = 999
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
    selection_dates, retrospective_dates, _ = _phase_dates()
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
    print_verdict(retrospective_passed, reasons, artifact_dir, FINAL_TEST_START.date())
    return artifact_dir


def _phase_dates() -> tuple[pd.DatetimeIndex, pd.DatetimeIndex, pd.DatetimeIndex]:
    selection_last_as_of = RETROSPECTIVE_START - pd.Timedelta(days=HORIZON)
    selection = pd.date_range(
        SELECTION_START, selection_last_as_of, freq=f"{REBALANCE_DAYS}D"
    )
    retrospective = pd.date_range(
        RETROSPECTIVE_START, RETROSPECTIVE_END, freq=f"{REBALANCE_DAYS}D"
    )
    final = pd.date_range(FINAL_TEST_START, FINAL_TEST_END, freq=f"{REBALANCE_DAYS}D")
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
    threshold = 0.05 / len(trials)
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
    if null_empirical_p > 0.01:
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
    if _alpha_p_value(retrospective["period_results"]) > 0.10:
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
    _, _, final_dates = _phase_dates()
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
            "threshold": 0.05 / len(kwargs["trials_df"]),
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
            "status": "locked_not_evaluated",
            "start": FINAL_TEST_START.date().isoformat(),
            "end": FINAL_TEST_END.date().isoformat(),
            "scheduled_as_of_dates": [date.date().isoformat() for date in final_dates],
            "required_price_through": (
                pd.Timestamp(final_dates.max()) + pd.Timedelta(days=HORIZON)
            )
            .date()
            .isoformat(),
            "horizon_days": HORIZON,
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


def evaluate_locked_final(manifest_path: Path, db_path: Path) -> Path:
    """Consume a locked future test once; never called by retrospective runs."""
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text())
    final = manifest["final_test"]
    if final["status"] != "locked_not_evaluated" or final.get("consumed"):
        raise RuntimeError("Final test manifest is not locked and unconsumed")
    current_hashes, current_aggregate = _source_hashes()
    if current_aggregate != manifest["source_aggregate_sha256"]:
        raise RuntimeError("Source hash differs from the locked manifest")
    if current_hashes != manifest["source_sha256"]:
        raise RuntimeError("Source file set differs from the locked manifest")

    output = manifest_path.parent / "final_evaluation.json"
    if output.exists():
        raise RuntimeError("Locked final test was already consumed")

    final_dates = pd.DatetimeIndex(pd.to_datetime(final["scheduled_as_of_dates"]))
    required_price_through = pd.Timestamp(final["required_price_through"])
    with Database(db_path, read_only=True) as db:
        entry_prices, transactions, prices = _load_data_range(
            db, pd.Timestamp(final["end"]), required_price_through
        )
    if prices.empty or pd.Timestamp(
        prices.index.max()
    ) < required_price_through - pd.Timedelta(days=7):
        raise RuntimeError(
            f"Final price data does not reach required horizon {required_price_through.date()}"
        )
    chosen = manifest["locked_config"]
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
    strategy = _run_config(entry_prices, transactions, prices, precomputed, chosen)
    passive = _run_passive_benchmark(prices, precomputed)
    _assert_identical_support("locked_final_test", [strategy, passive])
    final_passed = bool(
        manifest["retrospective_gates_passed"]
        and strategy["mean_alpha_pct"] > 0
        and strategy["total_return_pct"] > passive["total_return_pct"]
        and _alpha_p_value(strategy["period_results"]) <= 0.10
    )
    final_periods_path = manifest_path.parent / "final_periods.csv"
    if final_periods_path.exists():
        raise RuntimeError("Locked final period artifact already exists")
    pd.DataFrame(strategy["period_results"]).to_csv(final_periods_path, index=False)
    result = {
        "locked_manifest_sha256": _sha256_file(manifest_path),
        "final_periods_sha256": _sha256_file(final_periods_path),
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "final_db_sha256": _sha256_file(db_path),
        "source_aggregate_sha256": current_aggregate,
        "support_sha256": strategy["support_sha256"],
        "strategy_metrics": {key: strategy[key] for key in METRIC_KEYS},
        "passive_metrics": {key: passive[key] for key in METRIC_KEYS},
        "alpha_p_value": _alpha_p_value(strategy["period_results"]),
        "verdict": "final_out_of_sample_gates_passed"
        if final_passed
        else "no_validated_profit_claim",
    }
    with output.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return output


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
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain=v1"],
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
