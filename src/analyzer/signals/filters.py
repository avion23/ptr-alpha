"""Completed-horizon filtering and public-event episode deduplication."""

from __future__ import annotations

import pandas as pd


def _get_horizon_data(
    signals_df: pd.DataFrame, horizon: int, transaction_type: str | None = None
) -> pd.DataFrame:
    mask = signals_df["horizon_days"] == horizon
    if transaction_type is not None:
        mask &= signals_df["signal_type"] == transaction_type
    if "window_complete" in signals_df.columns:
        mask &= signals_df["window_complete"].fillna(False).astype(bool)
    return signals_df.loc[mask]


def _collapse_to_episodes(signals_df: pd.DataFrame) -> pd.DataFrame:
    """Keep one observation per public member/ticker/disclosure event.

    Multiple source rows made public for the same member, ticker, date, horizon,
    and transaction type are one observable episode. Rows on different public
    dates remain separate observations; no arbitrary calendar-gap or trade-size
    weighting is applied.
    """
    if signals_df.empty:
        return signals_df

    keys = ["member", "ticker", "disclosure_date", "horizon_days", "signal_type"]
    if not all(column in signals_df.columns for column in keys):
        return signals_df

    frame = signals_df.copy()
    frame["disclosure_date"] = pd.to_datetime(
        frame["disclosure_date"], errors="coerce"
    ).dt.normalize()
    grouped = frame.groupby(keys, dropna=False, sort=False)
    counts = grouped["ticker"].transform("size")
    first = grouped.cumcount().eq(0)
    result = frame.loc[first].copy()
    result["episode_count"] = counts.loc[first].to_numpy(dtype=int)
    return result.reset_index(drop=True)
