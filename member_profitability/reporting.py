"""Honest serialization and qualified research summaries."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from member_profitability.analysis import summarize_combined_metrics
from member_profitability.config import (
    DATA_SCOPE,
    DECAY_LAMBDA,
    HORIZON,
    TARGET_RETURN_COLUMN,
    TEST_WINDOW_DAYS,
    TRAIN_WINDOW_DAYS,
)


def build_output_dict(
    sigs: pd.DataFrame,
    all_tx: pd.DataFrame,
    all_tickers: list[str],
    windows: list[dict],
    all_wf: pd.DataFrame,
    correlations: dict,
    tier_results: dict,
    trade_count_analysis: dict,
    position_research: dict,
    combined_results: dict,
    db_path: str | Path,
) -> dict:
    return {
        "analysis_config": _config_section(
            sigs, all_tx, all_tickers, windows, all_wf, db_path
        ),
        "spearman_correlations": correlations,
        "tier_analysis": tier_results,
        "trade_count_reliability": {str(key): value for key, value in trade_count_analysis.items()},
        "position_research": position_research,
        "combined_metrics": summarize_combined_metrics(combined_results),
        "recommendations": {},
        "profitability_claim": "not_established",
    }


def _config_section(
    sigs: pd.DataFrame,
    all_tx: pd.DataFrame,
    all_tickers: list[str],
    windows: list[dict],
    all_wf: pd.DataFrame,
    db_path: str | Path,
) -> dict:
    return {
        "database": str(Path(db_path).expanduser().resolve()),
        "data_scope": DATA_SCOPE,
        "data_scope_note": (
            "The input schema cannot prove chamber for every historical row; "
            "results are mixed/unclassified and must not be labeled House or Senate."
        ),
        "horizon": HORIZON,
        "target_return_column": TARGET_RETURN_COLUMN,
        "train_window_days": TRAIN_WINDOW_DAYS,
        "test_window_days": TEST_WINDOW_DAYS,
        "decay_lambda": DECAY_LAMBDA,
        "total_transactions": int(len(all_tx)),
        "unique_transaction_members": int(all_tx["member"].nunique()),
        "total_tickers": int(len(all_tickers)),
        "total_signals": int(len(sigs)),
        "complete_signals": int(sigs["window_complete"].fillna(False).sum()),
        "n_nonoverlapping_windows": int(len(windows)),
        "research_windows_analyzed": int(all_wf["window"].nunique()) if not all_wf.empty else 0,
        "unique_research_members": int(all_wf["member"].nunique()) if not all_wf.empty else 0,
        "member_window_observations": int(len(all_wf)),
    }


def best_predictors(
    correlations: dict,
    combined_results: dict,
    tier_results: dict,
    position_results,
) -> dict:
    """Summarize exploratory leaders without making a profit claim."""
    positive_metrics = [
        (name, data)
        for name, data in correlations.items()
        if data.get("n_windows", 0) > 0 and data.get("mean_spearman", 0.0) > 0
    ]
    leading_metric = (
        max(positive_metrics, key=lambda item: item[1]["mean_spearman"])
        if positive_metrics
        else (None, {})
    )
    combined_summary = summarize_combined_metrics(combined_results)
    positive_combined = [
        (name, data)
        for name, data in combined_summary.items()
        if data["mean_spearman"] > 0
    ]
    leading_combined = (
        max(positive_combined, key=lambda item: item[1]["mean_spearman"])
        if positive_combined
        else (None, {})
    )
    positive_tiers = [
        (name, data)
        for name, data in tier_results.items()
        if data.get("n_windows", 0) > 0 and data.get("alpha_lift", 0.0) > 0
    ]
    leading_tier = (
        max(positive_tiers, key=lambda item: item[1]["alpha_lift"])
        if positive_tiers
        else (None, {})
    )
    position = position_results if isinstance(position_results, dict) else {}
    findings = [
        "All metric comparisons are exploratory research-window results.",
        (
            "The last package window is selection-isolated within this run but is "
            "retrospective validation because this history was used by prior research."
        ),
        "No profitability claim is established by this analysis.",
    ]
    return {
        "best_single_predictor": _metric_entry(*leading_metric),
        "leading_combined_metric": _metric_entry(*leading_combined),
        "leading_positive_tier": _tier_entry(*leading_tier),
        "selected_position_candidate": position.get("selected_candidate"),
        "retrospective_validation_status": position.get(
            "retrospective_validation_status", "not_evaluated"
        ),
        "key_findings": findings,
    }


def _metric_entry(name: str | None, data: dict) -> dict | None:
    if name is None:
        return None
    return {
        "metric": name,
        "mean_spearman": data.get("mean_spearman", 0.0),
        "n_windows": data.get("n_windows", 0),
        "interpretation": "exploratory_positive_association",
    }


def _tier_entry(name: str | None, data: dict) -> dict | None:
    if name is None:
        return None
    return {
        "metric": name,
        "mean_window_lift_pct": data.get("alpha_lift", 0.0),
        "n_windows": data.get("n_windows", 0),
    }


def serialize_numpy(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp, np.datetime64)):
        return str(pd.Timestamp(obj))
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def write_output(output: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(output, handle, indent=2, default=serialize_numpy, allow_nan=False)
    print(f"Results written to: {path}")
