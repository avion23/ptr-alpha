"""Predeclared retrospective validation for a staged congressional-trading database.

The ``freeze`` command writes the experiment configuration that will be used by
``evaluate``. The manifest is input, not authorization: evaluation does not
depend on code hashes, dependency fingerprints, filesystem locks, or an
exactly-once receipt. Statistical family controls and the locked post-2025
holdout boundary remain enforced by :mod:`analyzer.validation`.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from analyzer.member_ranking.buyer_scoring import (
    CONSENSUS_LOOKBACK_DAYS,
    _get_consensus_price_tickers,
)
from analyzer.portfolio_sim import PortfolioConfig, PortfolioSimulator
from analyzer.price_repository import next_nyse_session
from analyzer.validation import (
    LOCKED_FINAL_START,
    PRIMARY_METRIC,
    _effective_validation_grid,
    _phase_end,
    run_validation,
)

FROZEN_MANIFEST_PATH = (
    Path(__file__).resolve().parents[1] / "validation" / "phase2-evaluation-manifest.json"
)

TRAIN_START = date(2022, 1, 1)
TRAIN_END = date(2023, 12, 31)
TEST_START = date(2024, 1, 1)
TEST_END = date(2025, 6, 30)

GRID = {
    "horizon": [60, 90, 120],
    "frequency_days": [30],
    "lookback_days": [CONSENSUS_LOOKBACK_DAYS],
    "min_buyers": [2, 3, 5],
    "top_n": [3, 5],
}
ALPHA = 0.05
N_PERMUTATIONS = 999
PERMUTATION_SEED = 0

PORTFOLIO_CONFIG = {
    "initial_capital": 20000.0,
    "max_positions": 5,
    "rebalance_freq_days": 30,
    "hold_period_days": 120,
    "entry_slippage_pct": 0.001,
    "exit_slippage_pct": 0.001,
}


class FrozenManifestError(ValueError):
    """Raised when a predeclared validation manifest is malformed."""


def config_payload(
    grid: dict | None = None, grid_decision: str | None = None
) -> dict:
    """Return the complete predeclared experiment configuration."""
    payload = {
        "primary_metric": PRIMARY_METRIC,
        "phases": {
            "train": {
                "boundary": [str(TRAIN_START), str(TRAIN_END)],
                "outcomes_end_by": str(TRAIN_END),
            },
            "test": {
                "boundary": [str(TEST_START), str(TEST_END)],
                "evidence_class": "retrospective_previously_used_not_fresh_oos",
                "status": "retrospective_diagnostics_only",
            },
            "locked_final": {
                "start": str(LOCKED_FINAL_START),
                "status": "locked_not_queried_or_evaluated",
            },
        },
        "grid": grid if grid is not None else GRID,
        "alpha": ALPHA,
        "n_permutations": N_PERMUTATIONS,
        "permutation_seed": PERMUTATION_SEED,
        "portfolio": PORTFOLIO_CONFIG,
    }
    if grid_decision:
        payload["grid_decision"] = grid_decision
    return payload


def freeze_manifest(
    path: Path | None = None,
    *,
    grid: dict | None = None,
    grid_decision: str | None = None,
) -> dict:
    """Write a predeclared validation configuration."""
    path = path or FROZEN_MANIFEST_PATH
    manifest = {
        "schema_version": 2,
        "purpose": "Predeclared retrospective validation configuration",
        "evidence_class": "retrospective_previously_used_not_fresh_oos",
        "config": config_payload(grid, grid_decision),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def _manifest_config(manifest: dict) -> dict:
    """Validate and return the experiment config embedded in a manifest."""
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 2:
        raise FrozenManifestError("unsupported validation manifest schema")
    config = manifest.get("config")
    if not isinstance(config, dict):
        raise FrozenManifestError("validation manifest is missing config")
    phases = config.get("phases")
    grid = config.get("grid")
    if not isinstance(phases, dict) or not isinstance(grid, dict):
        raise FrozenManifestError("validation manifest requires phases and grid")
    try:
        config["grid"] = _effective_validation_grid(grid)
    except ValueError as exc:
        raise FrozenManifestError(str(exc)) from exc
    try:
        train_start, train_end = map(
            date.fromisoformat, phases["train"]["boundary"]
        )
        test_start, test_end = map(date.fromisoformat, phases["test"]["boundary"])
    except (KeyError, TypeError, ValueError) as exc:
        raise FrozenManifestError("validation phase boundaries are invalid") from exc
    if train_end < train_start or test_end < test_start or test_start <= train_end:
        raise FrozenManifestError("validation phase boundaries overlap or run backwards")
    if test_end >= LOCKED_FINAL_START:
        raise FrozenManifestError(
            f"test window enters locked final phase starting {LOCKED_FINAL_START}"
        )
    return config


def _portfolio_config(portfolio_cfg: dict | None = None) -> PortfolioConfig:
    cfg = portfolio_cfg or PORTFOLIO_CONFIG
    return PortfolioConfig(
        initial_capital=float(cfg["initial_capital"]),
        max_positions=int(cfg["max_positions"]),
        rebalance_freq_days=int(cfg["rebalance_freq_days"]),
        hold_period_days=int(cfg["hold_period_days"]),
        entry_slippage_pct=float(cfg["entry_slippage_pct"]),
        exit_slippage_pct=float(cfg["exit_slippage_pct"]),
    )


def _test_window_recommendations(
    all_tx: pd.DataFrame,
    config: dict,
    test_start: date,
    test_effective_end: date,
) -> pd.DataFrame:
    """Collect the exact consensus recommendations used by validation."""
    from analyzer.backtest.recommend import backtest_recommendations

    rows: list[pd.DataFrame] = []
    frequency_days = int(config.get("frequency_days", 30))
    lookback_days = int(config.get("lookback_days", CONSENSUS_LOOKBACK_DAYS))
    for as_of in pd.date_range(
        test_start, test_effective_end, freq=f"{frequency_days}D"
    ):
        recs = backtest_recommendations(
            all_tx,
            as_of_date=pd.Timestamp(as_of),
            lookback_days=lookback_days,
            min_buyers=int(config["min_buyers"]),
            top_n=int(config["top_n"]),
        )
        if recs.empty:
            continue
        recs = recs.copy()
        recs["as_of_date"] = pd.Timestamp(as_of)
        rows.append(recs)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _run_portfolio_evaluation(
    db,
    config: dict,
    test_start: date,
    test_effective_end: date,
    test_end: date,
    portfolio_cfg: dict | None = None,
) -> dict:
    """Run a capital-constrained portfolio on the selected test recommendations."""
    lookback_days = int(config.get("lookback_days", CONSENSUS_LOOKBACK_DAYS))
    tx_start = pd.Timestamp(test_start) - pd.Timedelta(days=lookback_days)
    all_tx = db.get_transactions_by_date_range(tx_start, pd.Timestamp(test_effective_end))
    if all_tx.empty:
        return {"status": "not_run_no_test_window_transactions"}

    recs = _test_window_recommendations(
        all_tx, config, test_start, test_effective_end
    )
    if recs.empty:
        return {"status": "not_run_no_test_window_recommendations"}

    tickers = sorted(set(_get_consensus_price_tickers(all_tx)) | {"SPY"})
    price_start = next_nyse_session(pd.Timestamp(test_start))
    prices = db.get_prices(tickers, price_start, pd.Timestamp(test_end))
    if prices.empty:
        return {"status": "not_run_no_test_window_prices"}

    sim = PortfolioSimulator(_portfolio_config(portfolio_cfg))
    results = sim.run(recs, prices, test_start, test_end)
    metrics = sim.compute_metrics(prices)
    metrics["recommendation_count"] = int(len(recs))
    metrics["snapshot_count"] = int(len(results))
    return {"status": "completed", "metrics": metrics}


def evaluate_manifest(
    db_path: str | Path,
    out_path: str | Path | None = None,
    manifest_path: Path | None = None,
) -> dict:
    """Evaluate a staged database under the predeclared manifest configuration."""
    manifest_path = manifest_path or FROZEN_MANIFEST_PATH
    manifest = json.loads(Path(manifest_path).read_text())
    cfg = _manifest_config(manifest)
    train_start, train_end = map(
        date.fromisoformat, cfg["phases"]["train"]["boundary"]
    )
    test_start, test_end = map(
        date.fromisoformat, cfg["phases"]["test"]["boundary"]
    )
    grid = dict(cfg["grid"])
    alpha = float(cfg["alpha"])
    n_permutations = int(cfg["n_permutations"])
    permutation_seed = int(cfg["permutation_seed"])
    portfolio_cfg = dict(cfg.get("portfolio", PORTFOLIO_CONFIG))

    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"staged database not found: {db_path}")

    from analyzer.database import Database

    max_lookback = max(
        int(value) for value in grid.get("lookback_days", [CONSENSUS_LOOKBACK_DAYS])
    )
    db = Database(db_path, read_only=True)
    try:
        available = db.get_transactions_by_date_range(
            pd.Timestamp(train_start) - pd.Timedelta(days=max_lookback),
            pd.Timestamp(test_end),
        )
        if available.empty:
            raise ValueError(
                f"staged database {db_path} has no transactions in the validation window"
            )
    finally:
        db.close()

    validation = run_validation(
        db_path=db_path,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        grid=grid,
        out_path=Path(out_path).with_name(Path(out_path).name + ".validation.json")
        if out_path is not None
        else None,
        n_permutations=n_permutations,
        permutation_seed=permutation_seed,
        alpha=alpha,
    )

    selected = validation.get("selected_config")
    if selected is None:
        portfolio = {
            "status": "not_run_no_deployable_config",
            "reason": "portfolio evaluation requires a corrected train survivor",
        }
    else:
        db = Database(db_path, read_only=True)
        try:
            test_effective_end = _phase_end(test_end, int(selected["horizon"]))
            portfolio = _run_portfolio_evaluation(
                db,
                selected,
                test_start,
                test_effective_end,
                test_end,
                portfolio_cfg,
            )
        finally:
            db.close()

    report = {
        "schema_version": 2,
        "evidence_class": "retrospective_previously_used_not_fresh_oos",
        "verdict": "not_established",
        "predeclared_manifest": manifest,
        "validation": validation,
        "portfolio": portfolio,
    }
    if out_path is not None:
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Predeclared retrospective validation of a staged database"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    freeze = sub.add_parser("freeze", help="write a predeclared evaluation manifest")
    freeze.add_argument("--manifest", default=str(FROZEN_MANIFEST_PATH))
    freeze.add_argument(
        "--min-buyers",
        default=None,
        help="comma-separated min_buyers values for a predeclared grid variant",
    )
    freeze.add_argument(
        "--grid-decision",
        default=None,
        help="reason recorded with a predeclared grid variant",
    )

    ev = sub.add_parser("evaluate", help="evaluate the staged database")
    ev.add_argument("--db", required=True, help="path to the staged congress.duckdb")
    ev.add_argument("--out", required=True, help="report JSON output path")
    ev.add_argument("--manifest", default=str(FROZEN_MANIFEST_PATH))

    args = parser.parse_args(argv)
    if args.command == "freeze":
        grid = None
        if args.min_buyers:
            base = {key: list(values) for key, values in GRID.items()}
            base["min_buyers"] = [int(value) for value in args.min_buyers.split(",")]
            grid = base
        freeze_manifest(
            Path(args.manifest), grid=grid, grid_decision=args.grid_decision
        )
        print(f"predeclared manifest written: {args.manifest}")
        return 0

    try:
        report = evaluate_manifest(args.db, args.out, Path(args.manifest))
    except FrozenManifestError as exc:
        print(f"invalid validation manifest: {exc}", file=sys.stderr)
        return 2
    print(f"report written: {args.out}")
    print(
        f"verdict: {report['verdict']} | "
        f"validation status: {report['validation']['status']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
