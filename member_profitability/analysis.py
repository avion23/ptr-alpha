"""Predictive-power analysis: Spearman correlations, tier separation, trade-count thresholds.

Each function takes the walk-forward observations (`all_wf`) and returns
either a per-metric dict (for Spearman/tier) or a dict keyed by threshold
(for trade-count).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from member_profitability.config import (
    METRICS_TO_TEST,
    MIN_MEMBERS_FOR_CORR,
    MIN_MEMBERS_FOR_TIER,
    TIER_FRACTION,
    TRADE_COUNT_THRESHOLDS,
)


def spearman_correlations_per_metric(all_wf: pd.DataFrame) -> dict:
    """For each metric, compute Spearman correlation across windows and
    aggregate (mean, median, std, pct significant)."""
    correlations: dict = {}
    for metric in METRICS_TO_TEST:
        window_corrs = _per_window_corrs(all_wf, metric)
        if window_corrs:
            correlations[metric] = _aggregate_window_corrs(window_corrs)
        else:
            correlations[metric] = {"mean_spearman": 0.0, "n_windows": 0}
    return correlations


def _per_window_corrs(all_wf: pd.DataFrame, metric: str) -> list[dict]:
    """Compute Spearman correlation per window for one metric."""
    window_corrs: list[dict] = []
    for wi in all_wf["window"].unique():
        subset = all_wf[all_wf["window"] == wi].copy()
        if len(subset) < MIN_MEMBERS_FOR_CORR:
            continue
        subset["train_rank"] = subset[metric].rank(ascending=False)
        subset["test_rank"] = subset["test_alpha"].rank(ascending=False)
        corr, pval = stats.spearmanr(subset["train_rank"], subset["test_rank"])
        if not np.isnan(corr):
            window_corrs.append({"corr": corr, "pval": pval})
    return window_corrs


def _aggregate_window_corrs(window_corrs: list[dict]) -> dict:
    corrs = [c["corr"] for c in window_corrs]
    pvals = [c["pval"] for c in window_corrs]
    sig_count = sum(1 for p in pvals if p < 0.05)
    return {
        "mean_spearman": round(float(np.mean(corrs)), 4),
        "median_spearman": round(float(np.median(corrs)), 4),
        "std_spearman": round(float(np.std(corrs)), 4),
        "pct_significant": round(sig_count / len(corrs) * 100, 1),
        "n_windows": len(corrs),
    }


def tier_analysis(all_wf: pd.DataFrame) -> dict:
    """Top 10% vs bottom 10% alpha lift per metric.

    For each window and metric, split members into top/bottom tiers by the
    metric, then compare their test-period alpha. A positive lift means
    high-metric members actually outperformed low-metric members.
    """
    tier_results: dict = {}
    for metric in METRICS_TO_TEST:
        lift = _tier_lift_for_metric(all_wf, metric)
        if lift is not None:
            tier_results[metric] = lift
        else:
            tier_results[metric] = {"alpha_lift": 0.0, "n_observations": 0}
    return tier_results


def _tier_lift_for_metric(all_wf: pd.DataFrame, metric: str) -> dict | None:
    top_alphas: list[float] = []
    bottom_alphas: list[float] = []
    for wi in all_wf["window"].unique():
        subset = all_wf[all_wf["window"] == wi].copy()
        if len(subset) < MIN_MEMBERS_FOR_TIER:
            continue
        n_top = max(1, int(len(subset) * TIER_FRACTION))
        n_bottom = max(1, int(len(subset) * TIER_FRACTION))
        sorted_by_metric = subset.sort_values(metric, ascending=False)
        top_tier = sorted_by_metric.head(n_top)
        bottom_tier = sorted_by_metric.tail(n_bottom)
        top_alphas.extend(top_tier["test_alpha"].tolist())
        bottom_alphas.extend(bottom_tier["test_alpha"].tolist())

    if not top_alphas or not bottom_alphas:
        return None

    top_mean = float(np.mean(top_alphas))
    bottom_mean = float(np.mean(bottom_alphas))
    lift = top_mean - bottom_mean
    _, p_val = stats.ttest_ind(top_alphas, bottom_alphas)

    return {
        "top_10pct_mean_alpha": round(top_mean, 4),
        "bottom_10pct_mean_alpha": round(bottom_mean, 4),
        "alpha_lift": round(lift, 4),
        "lift_p_value": round(float(p_val), 6),
        "n_observations": len(top_alphas),
        "top_outperforms": top_mean > bottom_mean,
    }


def trade_count_reliability(all_wf: pd.DataFrame) -> dict:
    """For each min_trades threshold, report correlation of shrunk_alpha
    with test alpha and the mean test alpha. Higher min_trades means we're
    filtering out members with too little history."""
    trade_count_analysis: dict = {}
    for min_trades in TRADE_COUNT_THRESHOLDS:
        trade_count_analysis[min_trades] = _trade_count_for_threshold(all_wf, min_trades)
    return trade_count_analysis


def _trade_count_for_threshold(all_wf: pd.DataFrame, min_trades: int) -> dict:
    subset = all_wf[all_wf["purchase_trades"] >= min_trades]
    if len(subset) < MIN_MEMBERS_FOR_CORR:
        return {"n_members": len(subset), "mean_test_alpha": 0.0}

    corr, pval = stats.spearmanr(subset["shrunk_alpha"], subset["test_alpha"])
    return {
        "n_members": int(len(subset)),
        "mean_test_alpha": round(float(subset["test_alpha"].mean()), 4),
        "shrunk_alpha_corr": round(float(corr), 4) if not np.isnan(corr) else 0.0,
        "corr_p_value": round(float(pval), 6) if not np.isnan(pval) else 1.0,
    }


def combined_metrics_analysis(all_wf: pd.DataFrame) -> dict:
    """Try a few multi-metric combined scores and report their mean
    Spearman correlation with test alpha across windows."""
    combined_results: dict = {"combined_v1": [], "combined_v2": [], "trades_x_winprob": []}
    for wi in all_wf["window"].unique():
        subset = all_wf[all_wf["window"] == wi].copy()
        if len(subset) < MIN_MEMBERS_FOR_CORR:
            continue
        _compute_combined_for_window(subset, combined_results)
    return combined_results


def _compute_combined_for_window(subset: pd.DataFrame, combined_results: dict) -> None:
    subset["combined_v1"] = (
        subset["shrunk_alpha"].rank(pct=True) * 0.4
        + subset["bayes_win_prob"].rank(pct=True) * 0.2
        + subset["conviction_score"].rank(pct=True) * 0.2
        + subset["sharpe_ratio"].rank(pct=True) * 0.2
    )
    subset["combined_v2"] = (
        subset["prob_up_given_buy"].rank(pct=True) * 0.3
        + subset["conviction_score"].rank(pct=True) * 0.3
        + subset["purchase_trades"].rank(pct=True) * 0.2
        + subset["shrunk_alpha"].rank(pct=True) * 0.2
    )
    subset["trades_x_winprob"] = subset["purchase_trades"] * subset["prob_up_given_buy"]

    for combo_name in combined_results:
        corr, _ = stats.spearmanr(subset[combo_name], subset["test_alpha"])
        if not np.isnan(corr):
            combined_results[combo_name].append(corr)


def summarize_combined_metrics(combined_results: dict) -> dict:
    """Aggregate per-window combined correlations into a summary dict."""
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
