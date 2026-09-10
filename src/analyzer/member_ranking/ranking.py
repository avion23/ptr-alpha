"""Descriptive member ranking from completed public purchase episodes."""

from __future__ import annotations

import pandas as pd

from analyzer.exceptions import AnalysisError
from analyzer.member_ranking.bayes import normal_normal_posteriors
from analyzer.models import TransactionType
from analyzer.signals.filters import _collapse_to_episodes, _get_horizon_data


def rank_members(signal_df: pd.DataFrame, horizon: int = 90) -> pd.DataFrame:
    """Rank members by partially pooled endpoint SPY alpha.

    Each observation is one completed member/ticker/disclosure-date episode.
    Returns and hit rates use the exact executable endpoint labels; transaction
    size, owner, path-dependent peak return, decay weights, and pseudo-counts do
    not enter the member statistic.
    """
    if signal_df.empty:
        raise AnalysisError("Empty signals dataframe")

    purchases = _get_horizon_data(
        signal_df, horizon, TransactionType.PURCHASE.value
    ).copy()
    if purchases.empty:
        raise AnalysisError(f"No completed purchase signals found for horizon {horizon}")

    required = {"member", "ticker", "disclosure_date", "total_return_pct", "total_spy_alpha_pct"}
    missing = required - set(purchases.columns)
    if missing:
        raise AnalysisError(
            f"Member ranking requires endpoint outcome columns: {sorted(missing)}"
        )

    purchases = _collapse_to_episodes(purchases)
    purchases = purchases.dropna(subset=["member", "total_return_pct", "total_spy_alpha_pct"])
    if purchases.empty:
        raise AnalysisError("No complete endpoint purchase outcomes found")

    grouped = purchases.groupby("member", sort=True)
    episode_count = grouped.size().astype(int)
    avg_return = grouped["total_return_pct"].mean()
    avg_alpha = grouped["total_spy_alpha_pct"].mean()
    avg_spy_return = avg_return - avg_alpha
    positive_return_rate = grouped["total_return_pct"].apply(lambda values: float((values > 0).mean()))
    positive_alpha_rate = grouped["total_spy_alpha_pct"].apply(lambda values: float((values > 0).mean()))

    fit = normal_normal_posteriors(
        purchases["total_spy_alpha_pct"].to_numpy(dtype=float),
        purchases["member"].to_numpy(dtype=object),
    ).reindex(episode_count.index)

    result = pd.DataFrame(
        {
            "member": episode_count.index,
            "purchase_episodes": episode_count.to_numpy(),
            "avg_return_pct": avg_return.to_numpy(dtype=float),
            "avg_spy_return_pct": avg_spy_return.to_numpy(dtype=float),
            "avg_spy_alpha_pct": avg_alpha.to_numpy(dtype=float),
            "positive_return_rate": positive_return_rate.to_numpy(dtype=float),
            "positive_alpha_rate": positive_alpha_rate.to_numpy(dtype=float),
            "shrunk_alpha_pct": fit["posterior_mean"].to_numpy(dtype=float),
            "shrunk_alpha_std_pct": fit["posterior_std"].to_numpy(dtype=float),
            "alpha_shrinkage": fit["shrinkage"].to_numpy(dtype=float),
        }
    )
    return result.sort_values(
        ["shrunk_alpha_pct", "member"], ascending=[False, True]
    ).reset_index(drop=True)
