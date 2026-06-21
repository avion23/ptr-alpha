"""Walk-forward analysis: rolling train/test windows + per-window metric collection.

Generates overlapping 6-month train / 6-month test windows over the
disclosure-date range, then for each window:
  1. Rank members on training data
  2. Compute test-period realized alpha per member
  3. Merge training metrics with test alpha
  4. Append the merged observations to the global results list
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analyzer.exceptions import AnalysisError
from analyzer.member_ranking import rank_members

from member_profitability.config import (
    HORIZON,
    METRICS_TO_TEST,
    MIN_MEMBERS_FOR_CORR,
    MIN_TEST_TRADES,
    TEST_WINDOW_DAYS,
    TRAIN_WINDOW_DAYS,
    WINDOW_SLIDE_DAYS,
)


def generate_windows(sigs: pd.DataFrame) -> list[dict]:
    """Build the rolling-window schedule over the signal disclosure range.

    Windows slide by WINDOW_SLIDE_DAYS (90d ≈ quarterly) so each window shares
    some data with its neighbours — gives more sample points for correlation
    stability estimation.
    """
    disc_dates = np.sort(sigs["disclosure_date"].dropna().unique())
    min_date = pd.Timestamp(disc_dates.min())
    max_date = pd.Timestamp(disc_dates.max())

    windows: list[dict] = []
    start = min_date
    while start + pd.Timedelta(days=TRAIN_WINDOW_DAYS + TEST_WINDOW_DAYS) <= max_date:
        train_end = start + pd.Timedelta(days=TRAIN_WINDOW_DAYS)
        test_end = train_end + pd.Timedelta(days=TEST_WINDOW_DAYS)
        windows.append({
            "train_start": start,
            "train_end": train_end,
            "test_start": train_end,
            "test_end": test_end,
        })
        start += pd.Timedelta(days=WINDOW_SLIDE_DAYS)
    return windows


def collect_window_results(sigs: pd.DataFrame, windows: list[dict]) -> pd.DataFrame:
    """For each window, compute the per-member train-metric + test-alpha table.

    Returns an empty DataFrame when no window has enough data.
    """
    all_window_results: list[pd.DataFrame] = []
    for wi, w in enumerate(windows):
        train_sigs, test_sigs = _slice_window(sigs, w)
        if train_sigs.empty or test_sigs.empty:
            continue

        train_rankings = _rank_train(train_sigs)
        if train_rankings.empty or len(train_rankings) < MIN_MEMBERS_FOR_CORR:
            continue

        test_alpha = _compute_test_alpha(test_sigs)
        if test_alpha.empty:
            continue

        merged = _merge_train_test(train_rankings, test_alpha)
        if merged.empty:
            continue

        merged = _tag_window(merged, wi, w)
        all_window_results.append(merged)

        if (wi + 1) % 5 == 0:
            print(f"  Window {wi+1}/{len(windows)}: {len(merged)} members with test data")

    if not all_window_results:
        return pd.DataFrame()
    return pd.concat(all_window_results, ignore_index=True)


def _slice_window(sigs: pd.DataFrame, w: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = sigs[
        (sigs["disclosure_date"] >= w["train_start"])
        & (sigs["disclosure_date"] < w["train_end"])
    ].copy()
    test = sigs[
        (sigs["disclosure_date"] >= w["test_start"])
        & (sigs["disclosure_date"] < w["test_end"])
    ].copy()
    return train, test


def _rank_train(train_sigs: pd.DataFrame) -> pd.DataFrame:
    try:
        return rank_members(train_sigs, HORIZON, threshold=5.0)
    except AnalysisError:
        return pd.DataFrame()


def _compute_test_alpha(test_sigs: pd.DataFrame) -> pd.DataFrame:
    test_purchases = test_sigs[test_sigs["signal_type"] == "Purchase"].copy()
    if test_purchases.empty:
        return pd.DataFrame()
    return (
        test_purchases.groupby("member")["spy_alpha_pct"]
        .agg(["mean", "count", "std"])
        .reset_index()
        .rename(columns={"mean": "test_alpha", "count": "test_trades", "std": "test_std"})
    )


def _merge_train_test(train_rankings: pd.DataFrame, test_alpha: pd.DataFrame) -> pd.DataFrame:
    merged = pd.merge(
        train_rankings[["member"] + METRICS_TO_TEST],
        test_alpha,
        on="member",
        how="inner",
    )
    merged = merged[merged["test_trades"] >= MIN_TEST_TRADES]
    if len(merged) < MIN_MEMBERS_FOR_CORR:
        return pd.DataFrame()
    return merged


def _tag_window(merged: pd.DataFrame, wi: int, w: dict) -> pd.DataFrame:
    merged = merged.copy()
    merged["window"] = wi
    merged["train_start"] = w["train_start"]
    merged["test_start"] = w["test_start"]
    return merged
