"""Predictive-power analysis on non-overlapping member-period observations."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from member_profitability.config import (
    METRICS_TO_TEST,
    MIN_MEMBERS_FOR_CORR,
    MIN_MEMBERS_FOR_TIER,
    TEST_RETURN_COLUMN,
    TIER_FRACTION,
    TRADE_COUNT_THRESHOLDS,
)


def spearman_correlations_per_metric(all_wf: pd.DataFrame) -> dict:
    correlations: dict = {}
    for metric in METRICS_TO_TEST:
        window_corrs = _per_window_corrs(all_wf, metric)
        correlations[metric] = (
            _aggregate_window_corrs(window_corrs)
            if window_corrs
            else {"mean_spearman": 0.0, "n_windows": 0}
        )
    return correlations


def _per_window_corrs(all_wf: pd.DataFrame, metric: str) -> list[dict]:
    window_corrs: list[dict] = []
    for window_id in all_wf["window"].unique():
        subset = all_wf[all_wf["window"] == window_id]
        if len(subset) < MIN_MEMBERS_FOR_CORR:
            continue
        corr, p_value = stats.spearmanr(subset[metric], subset[TEST_RETURN_COLUMN])
        if not np.isnan(corr):
            window_corrs.append({"corr": float(corr), "pval": float(p_value)})
    return window_corrs


def _aggregate_window_corrs(window_corrs: list[dict]) -> dict:
    corrs = [row["corr"] for row in window_corrs]
    p_values = [row["pval"] for row in window_corrs]
    return {
        "mean_spearman": round(float(np.mean(corrs)), 4),
        "median_spearman": round(float(np.median(corrs)), 4),
        "std_spearman": round(float(np.std(corrs)), 4),
        "pct_significant": round(sum(p < 0.05 for p in p_values) / len(corrs) * 100, 1),
        "n_windows": len(corrs),
    }


def tier_analysis(all_wf: pd.DataFrame) -> dict:
    results: dict = {}
    for metric in METRICS_TO_TEST:
        result = _tier_lift_for_metric(all_wf, metric)
        results[metric] = result or {"alpha_lift": 0.0, "n_windows": 0}
    return results


def _tier_lift_for_metric(all_wf: pd.DataFrame, metric: str) -> dict | None:
    window_lifts: list[float] = []
    top_means: list[float] = []
    bottom_means: list[float] = []
    for window_id in all_wf["window"].unique():
        subset = all_wf[all_wf["window"] == window_id]
        if len(subset) < MIN_MEMBERS_FOR_TIER:
            continue
        tier_size = max(1, int(len(subset) * TIER_FRACTION))
        ordered = subset.sort_values(metric, ascending=False)
        top_mean = float(ordered.head(tier_size)[TEST_RETURN_COLUMN].mean())
        bottom_mean = float(ordered.tail(tier_size)[TEST_RETURN_COLUMN].mean())
        top_means.append(top_mean)
        bottom_means.append(bottom_mean)
        window_lifts.append(top_mean - bottom_mean)
    if not window_lifts:
        return None
    p_value = 1.0
    if len(window_lifts) >= 2 and float(np.std(window_lifts, ddof=1)) > 0:
        p_value = float(stats.ttest_1samp(window_lifts, 0.0).pvalue)
    lift = float(np.mean(window_lifts))
    return {
        "top_10pct_mean_excess_return": round(float(np.mean(top_means)), 4),
        "bottom_10pct_mean_excess_return": round(float(np.mean(bottom_means)), 4),
        "alpha_lift": round(lift, 4),
        "lift_p_value": round(p_value, 6),
        "n_windows": len(window_lifts),
        "top_outperforms": lift > 0,
    }


def trade_count_reliability(all_wf: pd.DataFrame) -> dict:
    return {
        threshold: _trade_count_for_threshold(all_wf, threshold)
        for threshold in TRADE_COUNT_THRESHOLDS
    }


def _trade_count_for_threshold(all_wf: pd.DataFrame, min_trades: int) -> dict:
    subset = all_wf[all_wf["purchase_trades"] >= min_trades]
    base = {
        "unique_members": int(subset["member"].nunique()),
        "member_window_observations": int(len(subset)),
        "mean_test_excess_return_pct": (
            round(float(subset[TEST_RETURN_COLUMN].mean()), 4) if not subset.empty else 0.0
        ),
    }
    correlations = _per_window_corrs(subset, "shrunk_excess_return_pct")
    if not correlations:
        return base
    values = [row["corr"] for row in correlations]
    base.update(
        {
            "mean_window_spearman": round(float(np.mean(values)), 4),
            "n_windows": len(values),
        }
    )
    return base


def combined_metrics_analysis(all_wf: pd.DataFrame) -> dict:
    results: dict = {"combined_quality": [], "evidence_weighted": []}
    for window_id in all_wf["window"].unique():
        subset = all_wf[all_wf["window"] == window_id].copy()
        if len(subset) < MIN_MEMBERS_FOR_CORR:
            continue
        subset["combined_quality"] = (
            subset["shrunk_excess_return_pct"].rank(pct=True) * 0.5
            + subset["bayes_positive_excess_prob"].rank(pct=True) * 0.3
            + subset["conviction_score"].rank(pct=True) * 0.2
        )
        subset["evidence_weighted"] = (
            subset["prob_positive_excess"].rank(pct=True) * 0.4
            + subset["purchase_trades"].rank(pct=True) * 0.3
            + subset["shrunk_excess_return_pct"].rank(pct=True) * 0.3
        )
        for name in results:
            corr, _ = stats.spearmanr(subset[name], subset[TEST_RETURN_COLUMN])
            if not np.isnan(corr):
                results[name].append(float(corr))
    return results


def summarize_combined_metrics(combined_results: dict) -> dict:
    return {
        name: {
            "mean_spearman": round(float(np.mean(correlations)), 4),
            "std_spearman": (
                round(float(np.std(correlations)), 4) if len(correlations) > 1 else 0.0
            ),
            "n_windows": len(correlations),
        }
        for name, correlations in combined_results.items()
        if correlations
    }
