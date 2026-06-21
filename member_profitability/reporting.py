"""Output JSON serialization and recommendation generation for member profitability."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from member_profitability.config import (
    DECAY_LAMBDA,
    HORIZON,
    TEST_WINDOW_DAYS,
    TRAIN_WINDOW_DAYS,
)


def build_output_dict(
    sigs: pd.DataFrame,
    all_tx: pd.DataFrame,
    all_tickers: list[str],
    windows: list[dict],
    valid_windows: int,
    all_wf: pd.DataFrame,
    correlations: dict,
    tier_results: dict,
    trade_count_analysis: dict,
    position_results: list[dict],
    combined_results: dict,
) -> dict:
    """Assemble the output dict that will be JSON-serialized."""
    return {
        "analysis_config": _config_section(sigs, all_tx, all_tickers, windows, valid_windows, all_wf),
        "spearman_correlations": correlations,
        "tier_analysis": tier_results,
        "trade_count_reliability": _stringify_keys(trade_count_analysis),
        "position_sizing_grid": position_results,
        "combined_metrics": _summarize_combined(combined_results),
        "recommendations": {},
    }


def _config_section(
    sigs: pd.DataFrame,
    all_tx: pd.DataFrame,
    all_tickers: list[str],
    windows: list[dict],
    valid_windows: int,
    all_wf: pd.DataFrame,
) -> dict:
    return {
        "horizon": HORIZON,
        "train_window_days": TRAIN_WINDOW_DAYS,
        "test_window_days": TEST_WINDOW_DAYS,
        "decay_lambda": DECAY_LAMBDA,
        "total_transactions": len(all_tx),
        "total_tickers": len(all_tickers),
        "total_signals": len(sigs),
        "n_windows": len(windows),
        "valid_windows_analyzed": valid_windows,
        "total_window_observations": len(all_wf),
    }


def _stringify_keys(d: dict) -> dict:
    """JSON keys must be strings. Convert int trade-count thresholds to str."""
    return {str(k): v for k, v in d.items()}


def _summarize_combined(combined_results: dict) -> dict:
    summary: dict = {}
    for name, corrs in combined_results.items():
        if not corrs:
            continue
        summary[name] = {
            "mean_spearman": round(float(np.mean(corrs)), 4),
            "std_spearman": round(float(np.std(corrs)), 4) if len(corrs) > 1 else 0.0,
            "n_windows": len(corrs),
        }
    return summary


def best_predictors(
    correlations: dict,
    combined_results: dict,
    tier_results: dict,
    position_results: list[dict],
) -> dict:
    """Pick the best single metric, best combined, best tier, best grid."""
    best_metric_name, best_metric = _pick_best_metric(correlations)
    best_combined = _pick_best_combined(combined_results)
    best_tier_name, best_tier = _pick_best_tier(tier_results)
    best_grid = _pick_best_grid(position_results)

    findings = _build_findings(
        correlations, trade_count_lookup(), best_metric_name, best_metric,
        best_tier_name, best_tier, best_grid, best_combined,
    )

    return {
        "best_single_predictor": _single_predictor_entry(best_metric_name, best_metric),
        "best_combined_predictor": best_combined,
        "best_tier_metric": {
            "metric": best_tier_name,
            "alpha_lift": best_tier.get("alpha_lift", 0) if best_tier_name else 0,
        },
        "optimal_position_sizing": best_grid,
        "key_findings": findings,
    }


def _pick_best_metric(correlations: dict) -> tuple[str, dict]:
    return max(correlations.items(), key=lambda x: abs(x[1]["mean_spearman"]))


def _pick_best_combined(combined_results: dict) -> dict | None:
    if not combined_results:
        return None
    name, corrs = max(
        combined_results.items(),
        key=lambda x: abs(np.mean(x[1])) if x[1] else 0,
    )
    return {
        "name": name,
        "mean_spearman": round(float(np.mean(corrs)), 4),
    }


def _pick_best_tier(tier_results: dict) -> tuple[str | None, dict]:
    if not tier_results:
        return None, {}
    return max(tier_results.items(), key=lambda x: abs(x[1].get("alpha_lift", 0)))


def _pick_best_grid(position_results: list[dict]) -> dict:
    if not position_results:
        return {}
    return max(position_results, key=lambda x: x["sharpe_proxy"])


def trade_count_lookup() -> dict:
    """Empty placeholder for the findings builder."""
    return {}


def _build_findings(
    correlations: dict,
    _trade_count_analysis: dict,
    best_metric_name: str,
    best_metric: dict,
    best_tier_name: str | None,
    best_tier: dict,
    best_grid: dict,
    best_combined: dict | None,
) -> list[str]:
    findings: list[str] = []

    sorted_corrs = sorted(
        correlations.items(),
        key=lambda x: abs(x[1]["mean_spearman"]),
        reverse=True,
    )
    findings.append(
        f"Best single predictor of future alpha: '{sorted_corrs[0][0]}' "
        f"(Spearman={sorted_corrs[0][1]['mean_spearman']:+.4f})"
    )

    if best_grid:
        findings.append(
            f"Optimal position sizing: top_n={best_grid.get('top_n', 'N/A')}, "
            f"min_buyers={best_grid.get('min_buyers', 'N/A')} "
            f"(sharpe_proxy={best_grid.get('sharpe_proxy', 0):.4f})"
        )

    if best_tier_name:
        findings.append(
            f"Best tier separation: '{best_tier_name}' "
            f"(top10% vs bottom10% alpha lift={best_tier.get('alpha_lift', 0):+.4f}%)"
        )

    if best_combined:
        findings.append(
            f"Best combined predictor: '{best_combined['name']}' "
            f"(Spearman={best_combined['mean_spearman']:+.4f})"
        )

    return findings


def _single_predictor_entry(metric: str, data: dict) -> dict:
    corr = data["mean_spearman"]
    return {
        "metric": metric,
        "mean_spearman": corr,
        "interpretation": (
            "Strong positive" if corr > 0.2
            else "Moderate positive" if corr > 0.05
            else "Weak/no" if corr > -0.05
            else "Negative (avoid)"
        ),
    }


def serialize_numpy(obj):
    """JSON serializer that converts numpy types to native Python types."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def write_output(output: dict, path: Path = Path("data/member_analysis.json")) -> None:
    """Write the analysis output to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(output, f, indent=2, default=serialize_numpy)
    print(f"\nResults written to: {path}")
