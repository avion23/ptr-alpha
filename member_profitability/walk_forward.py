"""Non-overlapping, point-in-time member research windows."""

from __future__ import annotations

import numpy as np
import pandas as pd

from analyzer.signals import _apply_quality_filter, _collapse_to_episodes, _get_horizon_data

from member_profitability.config import (
    BAYES_PRIOR_STRENGTH,
    HORIZON,
    METRICS_TO_TEST,
    MIN_MEMBERS_FOR_CORR,
    MIN_TEST_TRADES,
    TARGET_RETURN_COLUMN,
    TEST_WINDOW_DAYS,
    TRAIN_WINDOW_DAYS,
    WINDOW_SLIDE_DAYS,
)


def generate_windows(sigs: pd.DataFrame) -> list[dict]:
    """Build adjacent train/test windows with non-overlapping test periods."""
    if sigs.empty or sigs["disclosure_date"].dropna().empty:
        return []
    min_date = pd.Timestamp(sigs["disclosure_date"].dropna().min()).normalize()
    max_maturity = (
        pd.Timestamp(sigs["disclosure_date"].dropna().max()).normalize()
        + pd.Timedelta(days=HORIZON)
    )

    windows: list[dict] = []
    start = min_date
    while start + pd.Timedelta(days=TRAIN_WINDOW_DAYS + TEST_WINDOW_DAYS) <= max_maturity:
        train_end = start + pd.Timedelta(days=TRAIN_WINDOW_DAYS)
        test_end = train_end + pd.Timedelta(days=TEST_WINDOW_DAYS)
        windows.append(
            {
                "train_start": start,
                "train_end": train_end,
                "test_start": train_end,
                "test_end": test_end,
            }
        )
        start += pd.Timedelta(days=WINDOW_SLIDE_DAYS)
    return windows


def collect_window_results(sigs: pd.DataFrame, windows: list[dict]) -> pd.DataFrame:
    """Collect one member observation per non-overlapping test period."""
    all_window_results: list[pd.DataFrame] = []
    for wi, window in enumerate(windows):
        train_sigs, test_sigs = _slice_window(sigs, window)
        if train_sigs.empty or test_sigs.empty:
            continue
        train_rankings = _rank_train(train_sigs)
        if train_rankings.empty or len(train_rankings) < MIN_MEMBERS_FOR_CORR:
            continue
        test_outcomes = _compute_test_outcomes(test_sigs)
        if test_outcomes.empty:
            continue
        merged = _merge_train_test(train_rankings, test_outcomes)
        if merged.empty:
            continue
        all_window_results.append(_tag_window(merged, wi, window))

    if not all_window_results:
        return pd.DataFrame()
    return pd.concat(all_window_results, ignore_index=True)


def _slice_window(sigs: pd.DataFrame, window: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Slice labels that were fully known by their respective boundaries."""
    disclosure = pd.to_datetime(sigs["disclosure_date"])
    maturity = disclosure + pd.Timedelta(days=HORIZON)
    train = sigs[
        (disclosure >= window["train_start"])
        & (disclosure < window["train_end"])
        & (maturity <= window["train_end"])
    ].copy()
    test = sigs[
        (disclosure >= window["test_start"])
        & (disclosure < window["test_end"])
        & (maturity <= window["test_end"])
    ].copy()
    return train, test


def _rank_train(train_sigs: pd.DataFrame) -> pd.DataFrame:
    """Estimate comparable member statistics from endpoint excess returns."""
    purchases = _get_horizon_data(train_sigs, HORIZON, "Purchase")
    purchases = _apply_quality_filter(purchases)
    purchases = purchases[purchases[TARGET_RETURN_COLUMN].notna()]
    if purchases.empty:
        return pd.DataFrame()
    purchases = _collapse_to_episodes(purchases)
    purchases = purchases[purchases[TARGET_RETURN_COLUMN].notna()]
    if purchases.empty:
        return pd.DataFrame()

    target = purchases[TARGET_RETURN_COLUMN]
    global_mean = float(target.mean())
    global_positive_rate = float(np.clip((target > 0).mean(), 0.10, 0.90))
    grouped = purchases.groupby("member")
    aggregate = grouped[TARGET_RETURN_COLUMN].agg(
        purchase_trades="count",
        avg_excess_return_pct="mean",
        excess_return_std="std",
        excess_return_sum="sum",
    )
    positive = (target > 0).groupby(purchases["member"]).sum().reindex(aggregate.index)
    n = aggregate["purchase_trades"].astype(float)
    aggregate["prob_positive_excess"] = positive / n
    aggregate["bayes_positive_excess_prob"] = (
        global_positive_rate * BAYES_PRIOR_STRENGTH + positive
    ) / (BAYES_PRIOR_STRENGTH + n)
    aggregate["shrunk_excess_return_pct"] = (
        global_mean * BAYES_PRIOR_STRENGTH + aggregate["excess_return_sum"]
    ) / (BAYES_PRIOR_STRENGTH + n)
    aggregate["sharpe_excess_return"] = np.where(
        aggregate["excess_return_std"].fillna(0.0) > 0,
        aggregate["avg_excess_return_pct"] / aggregate["excess_return_std"],
        0.0,
    )
    aggregate["conviction_score"] = _conviction(grouped, aggregate.index)
    return (
        aggregate.reset_index()
        .drop(columns=["excess_return_std", "excess_return_sum"])
        .sort_values("shrunk_excess_return_pct", ascending=False)
        .reset_index(drop=True)
    )


def _conviction(grouped, member_index: pd.Index) -> np.ndarray:
    counts = grouped.size().reindex(member_index).to_numpy(dtype=float)
    count_score = np.minimum(counts / 10.0, 1.0)
    if "amount_midpoint" not in grouped.obj.columns:
        return count_score
    amount_count = grouped["amount_midpoint"].count().reindex(member_index).to_numpy()
    amount_mean = (
        grouped["amount_midpoint"].mean().reindex(member_index).fillna(0.0).to_numpy()
    )
    size_score = np.where(amount_count > 0, np.minimum(amount_mean / 50_000.0, 1.0), 0.0)
    return count_score * 0.6 + size_score * 0.4


def _compute_test_outcomes(test_sigs: pd.DataFrame) -> pd.DataFrame:
    purchases = _get_horizon_data(test_sigs, HORIZON, "Purchase")
    purchases = _apply_quality_filter(purchases)
    purchases = purchases[purchases[TARGET_RETURN_COLUMN].notna()]
    if purchases.empty:
        return pd.DataFrame()
    purchases = _collapse_to_episodes(purchases)
    return (
        purchases.groupby("member")[TARGET_RETURN_COLUMN]
        .agg(test_excess_return_pct="mean", test_trades="count", test_std="std")
        .reset_index()
    )


def _compute_test_alpha(test_sigs: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible alias for the endpoint-excess outcome table."""
    return _compute_test_outcomes(test_sigs)


def _merge_train_test(
    train_rankings: pd.DataFrame, test_outcomes: pd.DataFrame
) -> pd.DataFrame:
    merged = pd.merge(
        train_rankings[["member"] + METRICS_TO_TEST],
        test_outcomes,
        on="member",
        how="inner",
        validate="one_to_one",
    )
    merged = merged[merged["test_trades"] >= MIN_TEST_TRADES]
    if len(merged) < MIN_MEMBERS_FOR_CORR:
        return pd.DataFrame()
    return merged


def _tag_window(merged: pd.DataFrame, wi: int, window: dict) -> pd.DataFrame:
    result = merged.copy()
    result["window"] = wi
    for key in ("train_start", "train_end", "test_start", "test_end"):
        result[key] = window[key]
    return result
