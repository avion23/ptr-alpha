"""Filters, episode collapsing, and dynamic-prior estimation.

`_get_horizon_data` and `_apply_quality_filter` are the cheap per-call
filters used by every downstream consumer. `_collapse_to_episodes` is
the expensive groupby that aggregates same-member/same-ticker signals
within a 14-day window into one row. `_compute_dynamic_prior` returns
the global up-rate for the dynamic-prior Bayesian shrinkage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analyzer.models import TransactionType

from analyzer.signals.constants import MIN_ENTRY_PRICE


def _get_horizon_data(
    signals_df: pd.DataFrame, horizon: int, transaction_type: str | None = None
) -> pd.DataFrame:
    mask = signals_df["horizon_days"] == horizon
    if transaction_type is not None:
        mask = mask & (signals_df["signal_type"] == transaction_type)
    return signals_df.loc[mask]


def _apply_quality_filter(signals_df: pd.DataFrame) -> pd.DataFrame:
    if "entry_price" not in signals_df.columns:
        return signals_df
    return signals_df[signals_df["entry_price"] >= MIN_ENTRY_PRICE]


def _compute_dynamic_prior(signals_df: pd.DataFrame, horizon: int) -> float:
    horizon_signals = _get_horizon_data(signals_df, horizon, TransactionType.PURCHASE.value)
    if horizon_signals.empty:
        return 0.50
    # Bug #2: NaN decayed_return_pct was previously treated as a loss because
    # (NaN > 0) evaluates to False in pandas.  Exclude NaN from both the
    # numerator (wins) and denominator (total) so missing price windows do not
    # bias the market-wide up-rate.
    valid = horizon_signals["decayed_return_pct"].dropna()
    if len(valid) == 0:
        return 0.50
    up_prob = (valid > 0).mean()
    return float(np.clip(up_prob, 0.10, 0.90))


def _assign_episode_ids(group_sorted: pd.DataFrame, max_gap_days: int) -> np.ndarray:
    dates = pd.to_datetime(group_sorted["disclosure_date"])
    if len(dates) <= 1:
        return np.zeros(len(dates), dtype=np.int64)
    gaps = dates.diff().dt.days.fillna(0).astype(int)
    return (gaps > max_gap_days).cumsum().values.astype(np.int64)


def _collapse_to_episodes(signals_df: pd.DataFrame, max_gap_days: int = 14) -> pd.DataFrame:
    """Collapse same-member/same-ticker/same-horizon/same-type signals that
    fall within `max_gap_days` of each other into a single weighted-average
    row. Used by `rank_members`/`_compute_member_stats` to deduplicate
    rapid-fire buy/sell activity into discrete trading episodes."""
    if signals_df.empty:
        return signals_df

    group_cols = ["member", "ticker", "horizon_days", "signal_type"]
    if not all(c in signals_df.columns for c in group_cols):
        return signals_df

    if "disclosure_date" not in signals_df.columns:
        return signals_df

    df = signals_df.sort_values(group_cols + ["disclosure_date"]).reset_index(drop=True)
    df = _assign_episode_column(df, group_cols, max_gap_days)
    df = _add_weight_column(df)

    existing_avg_cols = _get_existing_avg_cols(df)
    df = _add_weighted_columns(df, existing_avg_cols)

    return _aggregate_episodes(df, group_cols, existing_avg_cols, signals_df.columns)


def _assign_episode_column(df: pd.DataFrame, group_cols: list[str], max_gap_days: int) -> pd.DataFrame:
    dates = pd.to_datetime(df["disclosure_date"])
    gaps = dates.diff().dt.days.fillna(0).astype(np.int64)
    first_per_group = df.groupby(group_cols, sort=False).head(1).index
    gaps.loc[first_per_group] = 0
    df["_episode_id"] = (gaps > max_gap_days).cumsum().astype(np.int64)
    return df


def _add_weight_column(df: pd.DataFrame) -> pd.DataFrame:
    if "amount_midpoint" in df.columns:
        df["_weight"] = df["amount_midpoint"].fillna(1.0)
    else:
        df["_weight"] = 1.0
    return df


def _get_existing_avg_cols(df: pd.DataFrame) -> list[str]:
    avg_cols = [
        "decayed_return_pct", "spy_alpha_pct", "total_return_pct",
        "total_spy_alpha_pct", "peak_potential_pct",
    ]
    return [c for c in avg_cols if c in df.columns]


def _add_weighted_columns(df: pd.DataFrame, existing_avg_cols: list[str]) -> pd.DataFrame:
    for col in existing_avg_cols:
        non_nan = df[col].notna()
        df[f"_wp_{col}"] = np.where(non_nan, df[col] * df["_weight"], 0.0)
        df[f"_ws_{col}"] = np.where(non_nan, df["_weight"], 0.0)
    return df


def _aggregate_episodes(
    df: pd.DataFrame,
    group_cols: list[str],
    existing_avg_cols: list[str],
    orig_df_columns: pd.Index,
) -> pd.DataFrame:
    episode_key = group_cols + ["_episode_id"]
    agg_dict: dict = {
        "episode_count": ("_weight", "count"),
        "_weight_sum": ("_weight", "sum"),
    }
    for col, func in {"disclosure_date": "min", "entry_price": "first", "amount_midpoint": "sum"}.items():
        if col in df.columns:
            agg_dict[col] = (col, func)
    if "owner_code" in df.columns:
        # "first" instead of mode — O(N log N) per-group mode dominates cost.
        # Within episodes (same member/ticker, ≤14d gap), owner_code is
        # effectively constant, so first() is equivalent.
        agg_dict["owner_code"] = ("owner_code", "first")
    for col in existing_avg_cols:
        agg_dict[col] = (f"_wp_{col}", "sum")
        agg_dict[f"_ws_{col}"] = (f"_ws_{col}", "sum")

    collapsed = df.groupby(episode_key, sort=False).agg(**agg_dict).reset_index()

    for col in existing_avg_cols:
        ws = collapsed[f"_ws_{col}"]
        collapsed[col] = np.where(ws > 0, collapsed[col] / ws, np.nan)
        collapsed = collapsed.drop(columns=[f"_ws_{col}"])

    collapsed = collapsed.drop(columns=["_weight_sum", "_episode_id"])

    orig_cols = [c for c in orig_df_columns if c in collapsed.columns]
    return collapsed[orig_cols + ["episode_count"]]
