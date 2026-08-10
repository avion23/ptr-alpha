"""Frozen, exactly-once, capital-constrained validation of a staged database.

This module owns the frozen evaluation contract for the rebuilt staged
``congress.duckdb`` produced by the luna-rebuild sibling worktree:

* ``freeze``  - regenerate ``validation/phase2-evaluation-manifest.json`` and
  record the config/code/git/dependency hashes of the exact evaluation state.
* ``evaluate`` - verify the frozen state still matches the live environment
  (fail closed on any drift), then run the purged retrospective validation and
  the capital-constrained portfolio evaluation of the selected consensus
  configuration exactly once. Exactly-once is enforced by the canonical
  append-only evaluation ledger, which refuses any overlapping reservation.

No profitability claim is ever emitted: the test window is labeled
``retrospective_previously_used_not_fresh_oos``, the verdict is capped at
``not_fresh_oos_evidence`` and the top-level report verdict is
``not_established`` unless every frozen gate passes.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from analyzer.portfolio_sim import PortfolioConfig, PortfolioSimulator
from analyzer.validation import (
    LOCKED_FINAL_START,
    PRIMARY_METRIC,
    EvaluationAlreadyConsumedError,
    _code_hash,
    _dependency_version,
    _git_state,
    _phase_end,
    _sha256_file,
    _sha256_json,
    run_validation,
)

FROZEN_MANIFEST_PATH = (
    Path(__file__).resolve().parents[1] / "validation" / "phase2-evaluation-manifest.json"
)

TRAIN_START = date(2022, 1, 1)
TRAIN_END = date(2023, 12, 31)
TEST_START = date(2024, 1, 1)
TEST_END = date(2025, 6, 30)

# The fast consensus-only CLI grid (18 trials); the frozen evaluation always
# uses this exact grid so the rebuilt database is consumed exactly once.
GRID = {
    "horizon": [60, 90, 120],
    "frequency_days": [30],
    "training_lookback_days": [365],
    "min_buyers": [2, 3, 5],
    "top_n": [3, 5],
    "decay_lambda": [0.005],
    "bayes_prior_strength": [20],
    "scoring_mode": ["consensus"],
}
ALPHA = 0.05
N_PERMUTATIONS = 999
PERMUTATION_SEED = 0

PORTFOLIO_CONFIG = {
    "capital_constrained": True,
    "initial_capital": 20000.0,
    "max_positions": 5,
    "max_position_pct": 0.25,
    "max_sector_pct": 1.0,
    "rebalance_freq_days": 30,
    "hold_period_days": 120,
    "entry_slippage_pct": 0.001,
    "exit_slippage_pct": 0.001,
    "min_signal_score": 0.0,
    "max_price_staleness_days": 5,
    "max_execution_wait_days": 7,
    "sector_policy": (
        "deterministic_static_equity_mapping_no_live_calls; "
        "sector concentration is not a frozen gate"
    ),
    "valuation_gap_policy": (
        "risk metrics abstain (None) on any valuation gap; no fictional zero mark"
    ),
    "benchmark": "real SPY prices from the staged database",
}

DEPENDENCY_NAMES = ("numpy", "pandas", "scipy", "duckdb")


def config_payload(
    grid: dict | None = None, grid_decision: str | None = None
) -> dict:
    """The exact frozen evaluation configuration (grid, windows, gates).

    ``grid`` overrides the default module grid for diagnostic manifest
    variants (e.g. a data-sparse Senate-only min_buyers=1 freeze); the
    decision must be documented via ``grid_decision`` and made before the
    run. The default reproduces the canonical grid byte-for-byte.
    """
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
                "consumed": False,
            },
        },
        "grid": grid if grid is not None else GRID,
        "alpha": ALPHA,
        "n_permutations": N_PERMUTATIONS,
        "permutation_seed": PERMUTATION_SEED,
        "scoring_modes": ["consensus"],
        "portfolio": PORTFOLIO_CONFIG,
    }
    if grid_decision:
        payload["grid_decision"] = grid_decision
    return payload


def frozen_hashes(
    grid: dict | None = None, grid_decision: str | None = None
) -> dict:
    """Recorded hashes of the frozen evaluation state."""
    dependencies = {
        name: _dependency_version(name) for name in DEPENDENCY_NAMES
    }
    dependencies["python"] = platform.python_version()
    git_state = _git_state()
    return {
        "config_sha256": _sha256_json(config_payload(grid, grid_decision)),
        "code_sha256": _code_hash(),
        "harness_sha256": _sha256_file(Path(__file__)),
        "git_revision": git_state["revision"],
        "git_diff_sha256": git_state["diff_sha256"],
        "git_dirty": git_state["dirty"],
        "dependency_sha256": _sha256_json(dependencies),
        "database_sha256": None,
        "value_snapshot_sha256": None,
    }


def freeze_manifest(
    path: Path | None = None,
    *,
    grid: dict | None = None,
    grid_decision: str | None = None,
) -> dict:
    """Regenerate and persist a frozen evaluation manifest.

    Diagnostic variants pass an alternative ``grid`` (e.g. min_buyers=1 for
    data-sparse Senate-only coverage) plus a ``grid_decision`` note recorded
    in the manifest before any run. The canonical manifest is reproduced
    exactly when both are None.
    """
    path = path or FROZEN_MANIFEST_PATH
    manifest = {
        "schema_version": 1,
        "purpose": (
            "Frozen capital-constrained validation manifest for the rebuilt "
            "staged congress.duckdb; evaluated exactly once via the canonical "
            "append-only ledger."
        ),
        "freeze_policy": (
            "Regenerate only with scripts/frozen_validation.py freeze; a new "
            "freeze is a new manifest commit. The staged database evaluation "
            "is reserved exactly once in the ledger next to the database."
        ),
        "evidence_class": "retrospective_previously_used_not_fresh_oos",
        "verdict_policy": {
            "top_level_verdict": "not_established",
            "claim_rule": (
                "No profitability or fresh-OOS claim is emitted unless every "
                "frozen gate passes; retrospective evidence alone never "
                "establishes profitability."
            ),
        },
        "config": config_payload(grid, grid_decision),
        "hashes": frozen_hashes(grid, grid_decision),
        "data_hashes_recorded_at_evaluation": True,
        "variant": grid is not None,
        "evaluation": {
            "exactly_once": {
                "mechanism": (
                    "canonical append-only ledger refuses any overlapping "
                    "reservation (repeats and alternate configs/grids/windows)"
                ),
                "ledger_path": "<staged-db-dir>/.ptr-alpha-evaluation-ledger-v2.json",
            },
            "staged_db": {
                "producer": "luna-rebuild (.worktrees/luna-rebuild)",
                "status": "awaiting_staging_confirmation_and_root_authorization",
            },
            "runner": (
                "python3 scripts/frozen_validation.py evaluate "
                "--db <staged congress.duckdb> --out <report.json> "
                "[--manifest <manifest.json>]"
            ),
        },
        "dependencies": {
            name: _dependency_version(name) for name in DEPENDENCY_NAMES
        },
    }
    manifest["dependencies"]["python"] = platform.python_version()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def verify_frozen_state(manifest: dict) -> tuple[bool, list[str]]:
    """Return (ok, reasons) for the frozen manifest against the live state.

    The manifest is the frozen contract: the evaluation grid, windows, and
    gates are read from the manifest's embedded config, so the config is
    verified by self-consistency (embedded config hashes to the recorded
    config_sha256) rather than against module defaults. Fail closed on any
    drift of code, working-tree content, or dependencies. ``harness_sha256``,
    ``git_revision``, and ``git_dirty`` are recorded for provenance only:
    content is pinned by code/dependency hashes plus the working-tree diff,
    so a content-identical checkout at any later revision still verifies
    while any content drift fails closed. Data/database hashes are recorded
    at evaluation time and are not part of this check.
    """
    reasons: list[str] = []
    hashes = manifest.get("hashes", {})
    current = frozen_hashes()
    # Self-consistency: the hash of the manifest's embedded config must equal
    # the config hash the manifest records (tampering with either is caught).
    embedded_config_hash = _sha256_json(manifest.get("config", {}))
    if hashes.get("config_sha256") != embedded_config_hash:
        reasons.append(
            "config_sha256 self-consistency mismatch: embedded config does not "
            "hash to the recorded config_sha256"
        )
    for key in (
        "code_sha256",
        "git_diff_sha256",
        "dependency_sha256",
    ):
        if hashes.get(key) != current[key]:
            reasons.append(
                f"{key} mismatch: frozen={hashes.get(key)} live={current[key]}"
            )
    return (not reasons, reasons)


def _portfolio_config(
    portfolio_cfg: dict | None = None, sector_by_ticker: dict | None = None
) -> PortfolioConfig:
    cfg = portfolio_cfg or PORTFOLIO_CONFIG
    return PortfolioConfig(
        initial_capital=float(cfg["initial_capital"]),
        max_positions=int(cfg["max_positions"]),
        max_position_pct=float(cfg["max_position_pct"]),
        max_sector_pct=float(cfg["max_sector_pct"]),
        rebalance_freq_days=int(cfg["rebalance_freq_days"]),
        hold_period_days=int(cfg["hold_period_days"]),
        entry_slippage_pct=float(cfg["entry_slippage_pct"]),
        exit_slippage_pct=float(cfg["exit_slippage_pct"]),
        min_signal_score=float(cfg["min_signal_score"]),
        max_price_staleness_days=int(cfg["max_price_staleness_days"]),
        max_execution_wait_days=int(cfg["max_execution_wait_days"]),
        sector_by_ticker=sector_by_ticker or {},
    )


def _test_window_recommendations(
    all_tx: pd.DataFrame,
    prices: pd.DataFrame,
    entry_prices: pd.DataFrame,
    config: dict,
    test_start: date,
    test_effective_end: date,
) -> pd.DataFrame:
    """Collect the consensus recommendations of the frozen test window.

    Mirrors the validation pipeline's recommendation loop (same signal inputs,
    same frozen config) so the capital-constrained evaluation consumes exactly
    the recommendations the frozen retrospective evaluation sees.
    """
    from analyzer import analysis

    horizon = int(config["horizon"])
    signals = analysis.calculate_signal_potential(
        entry_prices,
        prices,
        [horizon],
        decay_lambda=float(config["decay_lambda"]),
    )
    rows: list[pd.DataFrame] = []
    frequency_days = int(config.get("frequency_days", 30))
    for as_of in pd.date_range(
        test_start, test_effective_end, freq=f"{frequency_days}D"
    ):
        recs = analysis.backtest_recommendations(
            signals,
            all_tx,
            as_of_date=as_of,
            horizon=horizon,
            lookback_days=60,
            min_buyers=int(config["min_buyers"]),
            top_n=int(config["top_n"]),
            threshold=float(config.get("threshold", 5.0)),
            prices_df=prices,
            training_lookback_days=int(config["training_lookback_days"]),
            scoring_mode="consensus",
            bayes_prior_strength=float(config["bayes_prior_strength"]),
        )
        if recs.empty:
            continue
        recs = recs.drop(columns=["optimal_horizon"], errors="ignore")
        recs["as_of_date"] = as_of
        rows.append(recs)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _run_portfolio_evaluation(
    db,
    db_path: Path,
    config: dict,
    test_start: date,
    test_effective_end: date,
    portfolio_cfg: dict | None = None,
) -> dict:
    """Run the frozen capital-constrained portfolio evaluation on the test window."""
    tx_end = pd.Timestamp(test_effective_end)
    price_end = pd.Timestamp(TEST_END)
    all_tx = db.get_transactions_by_date_range(
        pd.Timestamp("2021-10-07"), tx_end
    )
    tickers = sorted(set(all_tx["ticker"].dropna().astype(str)) | {"SPY"})
    prices = db.get_prices(tickers, pd.Timestamp("2021-10-07"), price_end)
    entry_prices = db.get_entry_prices(tickers, pd.Timestamp("2021-10-07"), price_end)

    recs = _test_window_recommendations(
        all_tx, prices, entry_prices, config, test_start, test_effective_end
    )
    if recs.empty:
        return {"status": "not_run_no_test_window_recommendations"}

    sector_by_ticker = {
        str(ticker): "Equity"
        for ticker in sorted(recs["ticker"].dropna().unique())
    }
    sim = PortfolioSimulator(_portfolio_config(portfolio_cfg, sector_by_ticker))
    results = sim.run(recs, prices, test_start, test_effective_end)
    metrics = sim.compute_metrics(prices)
    metrics["sector_by_ticker"] = sector_by_ticker
    metrics["recommendation_count"] = int(len(recs))
    metrics["snapshot_count"] = int(len(results))
    return {"status": "completed", "metrics": metrics}


def evaluate_manifest(
    db_path: str | Path,
    out_path: str | Path | None = None,
    manifest_path: Path | None = None,
) -> dict:
    """Verify the frozen state, then evaluate the staged database exactly once.

    The evaluation contract (grid, windows, gates, portfolio sizing) is read
    from the manifest's embedded config, so a diagnostic variant manifest
    drives its own frozen evaluation while the canonical manifest is
    untouched.
    """
    manifest_path = manifest_path or FROZEN_MANIFEST_PATH
    manifest = json.loads(Path(manifest_path).read_text())
    ok, reasons = verify_frozen_state(manifest)
    if not ok:
        raise FrozenStateMismatchError(reasons)

    cfg = manifest["config"]
    train_start, train_end = map(date.fromisoformat, cfg["phases"]["train"]["boundary"])
    test_start, test_end = map(date.fromisoformat, cfg["phases"]["test"]["boundary"])
    grid = dict(cfg["grid"])
    alpha = float(cfg["alpha"])
    n_permutations = int(cfg["n_permutations"])
    permutation_seed = int(cfg["permutation_seed"])
    portfolio_cfg = dict(cfg.get("portfolio", PORTFOLIO_CONFIG))

    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"staged database not found: {db_path}")

    from analyzer.database import Database

    db = Database(db_path, read_only=True)
    try:
        row_count = db.get_transactions_by_date_range(
            pd.Timestamp("2021-10-07"), pd.Timestamp(test_end)
        )
        if row_count.empty:
            raise ValueError(
                f"staged database {db_path} has no transactions in the frozen window"
            )
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
    finally:
        db.conn.close()

    portfolio: dict
    selected = validation.get("selected_config")
    if selected is None:
        portfolio = {
            "status": "not_run_no_deployable_config",
            "reason": (
                "capital-constrained portfolio evaluation requires a corrected "
                "train survivor"
            ),
        }
    else:
        db = Database(db_path, read_only=True)
        try:
            max_holding = int(selected["horizon"])
            test_effective_end = _phase_end(test_end, max_holding, 0)
            portfolio = _run_portfolio_evaluation(
                db, db_path, selected, test_start, test_effective_end, portfolio_cfg
            )
        finally:
            db.conn.close()

    report = {
        "schema_version": 1,
        "evidence_class": "retrospective_previously_used_not_fresh_oos",
        "verdict": "not_established",
        "frozen_manifest": manifest,
        "verification": {"ok": True, "checked_hashes": ["config_sha256", "code_sha256", "git_diff_sha256", "dependency_sha256"]},
        "validation": validation,
        "portfolio": portfolio,
    }
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(
            json.dumps(report, indent=2, sort_keys=True, default=str) + "\n"
        )
    return report


class FrozenStateMismatchError(RuntimeError):
    def __init__(self, reasons: list[str]):
        self.reasons = reasons
        super().__init__("frozen manifest verification failed: " + "; ".join(reasons))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Frozen exactly-once validation of the rebuilt staged database"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    freeze = sub.add_parser("freeze", help="regenerate a frozen evaluation manifest")
    freeze.add_argument("--manifest", default=str(FROZEN_MANIFEST_PATH))
    freeze.add_argument(
        "--min-buyers",
        default=None,
        help=(
            "comma-separated min_buyers values for a diagnostic grid variant "
            "(e.g. '1'); the canonical grid is used when omitted"
        ),
    )
    freeze.add_argument(
        "--grid-decision",
        default=None,
        help="documented pre-run decision recorded for a grid variant",
    )

    ev = sub.add_parser("evaluate", help="verify and evaluate the staged database")
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
        print(f"frozen manifest written: {args.manifest}")
        return 0
    try:
        report = evaluate_manifest(args.db, args.out, Path(args.manifest))
    except FrozenStateMismatchError as exc:
        print("frozen state mismatch; evaluation refused:", file=sys.stderr)
        for reason in exc.reasons:
            print(f"  - {reason}", file=sys.stderr)
        return 2
    except EvaluationAlreadyConsumedError as exc:
        print(f"evaluation already consumed: {exc}", file=sys.stderr)
        return 3
    print(f"report written: {args.out}")
    print(f"verdict: {report['verdict']} | validation status: {report['validation']['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
