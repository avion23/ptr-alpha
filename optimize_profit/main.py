"""Chronological optimizer with an untouched holdout and immutable artifacts."""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from analyzer import analysis
from analyzer.database import Database
from analyzer.backtest import evaluate_backtest

from optimize_profit.metrics import summarize_walk_forward
from optimize_profit.precompute import precompute_walk_forward_data
from optimize_profit.reporting import print_holdout, print_selection, print_verdict
from optimize_profit.scoring import (
    SCORING_FUNCTIONS,
    make_shuffled_scorer,
    score_constant,
)
from optimize_profit.walk_forward import run_walk_forward

TX_START = pd.Timestamp("2021-10-07")
EVALUATION_START = pd.Timestamp("2022-01-01")
HOLDOUT_START = pd.Timestamp("2024-07-01")
EVALUATION_END = pd.Timestamp("2025-06-30")
HORIZON = 90
REBALANCE_DAYS = HORIZON  # one bankroll; no overlapping full-horizon vintages
LOOKBACK_DAYS = 60
TRAINING_LOOKBACK_DAYS = 365
MAX_DRAWDOWN_PCT = 30.0
SHUFFLED_CANARY_SEEDS = tuple(range(10))

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
    "max_drawdown_pct",
    "win_rate_pct",
    "n_periods",
    "avg_positions",
    "stopped_early",
    "requested_periods",
    "coverage_pct",
)


def main() -> None:
    db_path = Path(os.environ.get("OPTIMIZE_PROFIT_DB", "data/congress.duckdb"))
    output_root = Path(
        os.environ.get("OPTIMIZE_PROFIT_OUTPUT", "data/optimize_profit_runs")
    )
    config = _manifest_config(db_path)

    with Database(db_path, read_only=True) as db:
        entry_prices, transactions, prices = _load_data(db)

    datasets = _build_decay_datasets(entry_prices, transactions, prices)
    selection_sets = {
        decay: _subset_periods(precomputed, end=HOLDOUT_START, end_inclusive=False)
        for decay, precomputed in datasets.items()
    }
    holdout_sets = {
        decay: _subset_periods(precomputed, start=HOLDOUT_START)
        for decay, precomputed in datasets.items()
    }

    trials_df, selection_periods, selection_rejections = _run_selection_sweep(
        entry_prices, transactions, prices, selection_sets
    )
    selected, selection_has_corrected_signal = _select_frozen_config(trials_df)
    selected_params = _params_from_row(selected)

    holdout_run = _run_config(
        entry_prices,
        transactions,
        prices,
        holdout_sets[selected_params["decay_lambda"]],
        selected_params,
    )
    selection_constant = _run_canary(
        entry_prices,
        transactions,
        prices,
        selection_sets[selected_params["decay_lambda"]],
        selected_params,
        score_constant,
    )
    holdout_constant = _run_canary(
        entry_prices,
        transactions,
        prices,
        holdout_sets[selected_params["decay_lambda"]],
        selected_params,
        score_constant,
    )
    selection_spy = _run_passive_benchmark(
        prices, selection_sets[selected_params["decay_lambda"]]
    )
    holdout_spy = _run_passive_benchmark(
        prices, holdout_sets[selected_params["decay_lambda"]]
    )
    null_df, null_periods, null_rejections = _run_shuffled_canaries(
        entry_prices,
        transactions,
        prices,
        selection_sets[selected_params["decay_lambda"]],
        holdout_sets[selected_params["decay_lambda"]],
        selected_params,
    )

    robust, robustness_reasons = _assess_holdout_robustness(
        selected,
        selection_has_corrected_signal,
        holdout_run,
        holdout_spy,
        holdout_constant,
        null_df,
    )
    artifact_dir = _persist_artifacts(
        output_root=output_root,
        db_path=db_path,
        config=config,
        trials_df=trials_df,
        selected=selected,
        selection_periods=selection_periods,
        selection_rejections=selection_rejections,
        holdout_run=holdout_run,
        selection_constant=selection_constant,
        holdout_constant=holdout_constant,
        selection_spy=selection_spy,
        holdout_spy=holdout_spy,
        null_df=null_df,
        null_periods=null_periods,
        null_rejections=null_rejections,
        robust=robust,
        robustness_reasons=robustness_reasons,
    )

    print_selection(selected, len(trials_df))
    print_holdout(holdout_run, holdout_spy, holdout_constant)
    print_verdict(robust, robustness_reasons, artifact_dir)


def _load_data(db: Database) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    transactions = db.get_transactions_by_date_range(TX_START, EVALUATION_END)
    tickers = sorted(
        set(
            ticker
            for ticker in transactions["ticker"].dropna().unique()
            if isinstance(ticker, str)
        )
        | {"SPY"}
    )
    price_end = EVALUATION_END + pd.Timedelta(days=HORIZON + 40)
    prices = db.get_prices(tickers, TX_START, price_end)
    entry_prices = db.get_entry_prices(tickers, TX_START, price_end)
    return entry_prices, transactions, prices


def _build_decay_datasets(entry_prices, transactions, prices) -> dict[float, dict]:
    as_of_dates = pd.date_range(
        EVALUATION_START, EVALUATION_END, freq=f"{REBALANCE_DAYS}D"
    )
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


def _subset_periods(precomputed, start=None, end=None, end_inclusive=True):
    result = {}
    for key, data in precomputed.items():
        timestamp = pd.Timestamp(data["as_of_ts"])
        if start is not None and timestamp < pd.Timestamp(start):
            continue
        if end is not None:
            boundary = pd.Timestamp(end)
            if timestamp > boundary or (not end_inclusive and timestamp >= boundary):
                continue
        result[key] = data
    return result


def _run_selection_sweep(entry_prices, transactions, prices, selection_sets):
    keys = tuple(PARAM_GRID)
    rows = []
    all_periods = []
    all_rejections = []
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
    trials["bh_q_value"] = _bh_adjusted_pvalues(trials["alpha_p_value"].to_numpy())
    trials["bonferroni_significant"] = trials["alpha_p_value"] <= 0.05 / len(trials)
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
        max_dd_pct=MAX_DRAWDOWN_PCT,
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
        max_dd_pct=MAX_DRAWDOWN_PCT,
    )


def _select_frozen_config(trials: pd.DataFrame) -> tuple[pd.Series, bool]:
    eligible = trials[
        (trials["n_periods"] >= 6)
        & (~trials["stopped_early"])
        & (trials["mean_alpha_pct"] > 0)
        & (trials["bh_q_value"] <= 0.05)
    ]
    corrected_signal = not eligible.empty
    pool = eligible if corrected_signal else trials[trials["n_periods"] >= 6]
    if pool.empty:
        raise RuntimeError("No selection configuration has six covered periods")
    selected = pool.sort_values(
        ["alpha_sharpe", "mean_alpha_pct", "coverage_pct"],
        ascending=False,
    ).iloc[0]
    return selected, corrected_signal


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
    rejections = []
    for data in periods.values():
        as_of = data["as_of_ts"]
        evaluated = evaluate_backtest(
            pd.DataFrame({"ticker": ["SPY"], "signal_score": [1.0]}),
            prices,
            as_of,
            int(data["horizon"]),
        ).dropna(subset=["bt_return_pct"])
        if evaluated.empty:
            rejections.append(
                {"as_of_date": as_of.date(), "reason": "spy_price_unavailable"}
            )
            continue
        value = float(evaluated["bt_return_pct"].iloc[0])
        returns.append(
            {
                "as_of_date": as_of.date(),
                "portfolio_return_pct": value,
                "spy_return_pct": value,
                "n_positions": 1,
            }
        )
    metrics = summarize_walk_forward(returns, False, periods_per_year=365.0 / HORIZON)
    return {
        **metrics,
        "period_results": returns,
        "rejection_ledger": rejections,
        "requested_periods": len(periods),
        "coverage_pct": round(100 * len(returns) / len(periods), 1) if periods else 0.0,
    }


def _run_shuffled_canaries(
    entry_prices,
    transactions,
    prices,
    selection_periods,
    holdout_periods,
    params,
):
    rows = []
    period_rows = []
    rejection_rows = []
    base = SCORING_FUNCTIONS[params["scoring_fn"]]
    for seed in SHUFFLED_CANARY_SEEDS:
        scorer = make_shuffled_scorer(base, seed)
        for phase, periods in (
            ("selection", selection_periods),
            ("holdout", holdout_periods),
        ):
            run = _run_canary(
                entry_prices, transactions, prices, periods, params, scorer
            )
            rows.append(
                {
                    "diagnostic_type": "shuffled_scorer",
                    "phase": phase,
                    "seed": seed,
                    **{key: run[key] for key in METRIC_KEYS},
                    "alpha_p_value": _alpha_p_value(run["period_results"]),
                }
            )
            period_rows.extend(
                {
                    "diagnostic_type": "shuffled_scorer",
                    "phase": phase,
                    "seed": seed,
                    **row,
                }
                for row in run["period_results"]
            )
            rejection_rows.extend(
                {
                    "diagnostic_type": "shuffled_scorer",
                    "phase": phase,
                    "seed": seed,
                    **row,
                }
                for row in run["rejection_ledger"]
            )
    return pd.DataFrame(rows), period_rows, rejection_rows


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


def _bh_adjusted_pvalues(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0, 1)
    return result


def _assess_holdout_robustness(
    selected,
    selection_has_corrected_signal,
    holdout,
    holdout_spy,
    holdout_constant,
    null_df,
):
    reasons = []
    if not selection_has_corrected_signal:
        reasons.append("No selection trial survived Benjamini-Hochberg correction")
    if holdout["n_periods"] < 3:
        reasons.append("Fewer than three covered holdout periods")
    if holdout["mean_alpha_pct"] <= 0:
        reasons.append("Holdout mean alpha is not positive")
    if holdout["total_return_pct"] <= holdout_spy["total_return_pct"]:
        reasons.append("Frozen strategy did not beat passive SPY on holdout")
    if holdout["alpha_sharpe"] <= holdout_constant["alpha_sharpe"]:
        reasons.append("Frozen strategy did not beat the constant-score canary")
    shuffled_holdout = null_df[null_df["phase"] == "holdout"]["alpha_sharpe"]
    if not shuffled_holdout.empty and holdout[
        "alpha_sharpe"
    ] <= shuffled_holdout.quantile(0.95):
        reasons.append(
            "Frozen strategy did not exceed the 95th percentile shuffled scorer"
        )
    if _alpha_p_value(holdout["period_results"]) > 0.10:
        reasons.append("Holdout one-sided alpha p-value exceeds 0.10")
    if holdout["stopped_early"]:
        reasons.append("Holdout drawdown stop triggered")
    return not reasons, reasons


def _persist_artifacts(**kwargs) -> Path:
    output_root = kwargs["output_root"]
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    artifact_dir = output_root / run_id
    artifact_dir.mkdir(parents=True, exist_ok=False)

    trials_df = kwargs["trials_df"]
    selected = kwargs["selected"]
    trials_df.to_csv(artifact_dir / "selection_trials.csv", index=False)
    pd.DataFrame(kwargs["selection_periods"]).to_csv(
        artifact_dir / "selection_periods.csv", index=False
    )
    pd.DataFrame(kwargs["selection_rejections"]).to_csv(
        artifact_dir / "selection_rejections.csv", index=False
    )

    holdout_run = kwargs["holdout_run"]
    pd.DataFrame(holdout_run["period_results"]).to_csv(
        artifact_dir / "holdout_periods.csv", index=False
    )
    pd.DataFrame(holdout_run["rejection_ledger"]).to_csv(
        artifact_dir / "holdout_rejections.csv", index=False
    )

    diagnostics = kwargs["null_df"].copy()
    diagnostics = pd.concat(
        [
            trials_df.assign(diagnostic_type="selection_trial", phase="selection"),
            diagnostics,
            _benchmark_rows(kwargs),
        ],
        ignore_index=True,
        sort=False,
    )
    diagnostics.to_csv(
        artifact_dir / "multiplicity_and_null_diagnostics.csv", index=False
    )
    pd.DataFrame(kwargs["null_periods"]).to_csv(
        artifact_dir / "null_periods.csv", index=False
    )
    pd.DataFrame(kwargs["null_rejections"]).to_csv(
        artifact_dir / "null_rejections.csv", index=False
    )

    config_json = json.dumps(kwargs["config"], sort_keys=True, separators=(",", ":"))
    manifest = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": "holdout_robustness_passed"
        if kwargs["robust"]
        else "no_validated_profit_claim",
        "robustness_reasons": kwargs["robustness_reasons"],
        "selected_trial_id": int(selected["trial_id"]),
        "selected_config": _params_from_row(selected),
        "selection_metrics": {key: _json_value(selected[key]) for key in METRIC_KEYS},
        "holdout_metrics": {key: _json_value(holdout_run[key]) for key in METRIC_KEYS},
        "config": kwargs["config"],
        "config_sha256": hashlib.sha256(config_json.encode()).hexdigest(),
        "data_sha256": _sha256_file(kwargs["db_path"]),
        "code_sha256": _optimize_code_hash(),
        "git_commit": _git_commit(),
    }
    artifact_hashes = {}
    for path in sorted(artifact_dir.glob("*.csv")):
        artifact_hashes[path.name] = _sha256_file(path)
    manifest["artifact_sha256"] = artifact_hashes
    (artifact_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return artifact_dir


def _benchmark_rows(kwargs) -> pd.DataFrame:
    rows = []
    for name in (
        "selection_constant",
        "holdout_constant",
        "selection_spy",
        "holdout_spy",
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


def _manifest_config(db_path: Path) -> dict:
    return {
        "tx_start": TX_START.date().isoformat(),
        "evaluation_start": EVALUATION_START.date().isoformat(),
        "holdout_start": HOLDOUT_START.date().isoformat(),
        "evaluation_end": EVALUATION_END.date().isoformat(),
        "horizon_days": HORIZON,
        "rebalance_days": REBALANCE_DAYS,
        "lookback_days": LOOKBACK_DAYS,
        "training_lookback_days": TRAINING_LOOKBACK_DAYS,
        "max_drawdown_pct": MAX_DRAWDOWN_PCT,
        "parameter_grid": {key: list(value) for key, value in PARAM_GRID.items()},
        "shuffled_canary_seeds": list(SHUFFLED_CANARY_SEEDS),
        "db_path": str(db_path),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _optimize_code_hash() -> str:
    digest = hashlib.sha256()
    package = Path(__file__).resolve().parent
    for path in sorted(package.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def _json_value(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    return value


if __name__ == "__main__":
    main()
