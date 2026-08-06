"""Historical return-curve builders for OU parameter estimation.

`_build_ticker_curves` and `_build_global_curves` are memoized wrappers that
filter signals to a window, then delegate to the vectorized
`_build_curves_for_rows` which uses `searchsorted` on cached (idx_ns, vals)
arrays to avoid repeated pandas slicing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analyzer._memo import df_memoize
from analyzer.models import TransactionType
from analyzer.signals import _price_arrays


@df_memoize(copy=False)
def _build_ticker_curves(
    ticker: str,
    signals_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    as_of_date: pd.Timestamp,
    horizon: int,
) -> list:
    """Build historical return curves for a specific ticker's prior purchases.

    Returns a list of 1-D numpy arrays. Each array is the cumulative return
    curve r(t) = P(t)/P(entry) - 1 over [disclosure, disclosure+horizon].
    """
    if ticker not in prices_df.columns:
        return []

    cutoff = as_of_date - pd.Timedelta(days=horizon)
    eligible = signals_df[
        (signals_df["ticker"] == ticker)
        & (signals_df["horizon_days"] == horizon)
        & (signals_df["disclosure_date"] <= cutoff)
        & (signals_df["signal_type"] == TransactionType.PURCHASE.value)
    ]
    if eligible.empty:
        return []

    return _build_curves_for_rows(eligible, prices_df, horizon)


@df_memoize(copy=False)
def _build_global_curves(
    signals_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    as_of_date: pd.Timestamp,
    horizon: int,
) -> list:
    """Build historical return curves across ALL tickers' prior purchases.

    Used as a global prior when a ticker has no own disclosure history.
    """
    cutoff = as_of_date - pd.Timedelta(days=horizon)
    all_purchases = signals_df[
        (signals_df["horizon_days"] == horizon)
        & (signals_df["disclosure_date"] <= cutoff)
        & (signals_df["signal_type"] == TransactionType.PURCHASE.value)
    ]
    if all_purchases.empty:
        return []

    return _build_curves_for_rows(all_purchases, prices_df, horizon)


def _build_curves_for_rows(
    rows: pd.DataFrame,
    prices_df: pd.DataFrame,
    horizon: int,
) -> list:
    """Vectorized curve builder.

    Precomputes per-ticker (date_index_ns, price_values) once via the shared
    ``_price_arrays`` cache and uses searchsorted to slice windows instead of
    re-filtering pandas for every row.
    """
    price_cols = set(prices_df.columns)
    per_ticker: dict[str, tuple | None] = {}

    curves: list = []
    disclosures = rows["disclosure_date"].values
    entry_prices_arr = rows["entry_price"].values
    tickers = rows["ticker"].values

    horizon_ns = pd.Timedelta(days=horizon).value  # ns int

    # Pre-compute all disclosure timestamps as int64 ns to avoid
    # per-row pd.Timestamp() creation in the loop.
    disc_ns_all = np.empty(len(rows), dtype=np.int64)
    for i in range(len(rows)):
        disc_ns_all[i] = pd.Timestamp(disclosures[i]).value

    for i in range(len(rows)):
        entry_price = entry_prices_arr[i]
        if pd.isna(entry_price) or entry_price <= 0:
            continue
        tkr = tickers[i]
        if tkr not in price_cols:
            continue

        if tkr not in per_ticker:
            per_ticker[tkr] = _price_arrays(prices_df, tkr)

        cached = per_ticker[tkr]
        if cached is None:
            continue
        idx_ns, vals = cached
        if idx_ns is None:
            continue

        disc_ns = disc_ns_all[i]
        end_ns = disc_ns + horizon_ns

        lo = int(np.searchsorted(idx_ns, disc_ns, side="left"))
        hi = int(np.searchsorted(idx_ns, end_ns, side="right"))
        window = vals[lo:hi]
        if len(window) < 3:
            continue
        curves.append(window / entry_price - 1.0)

    return curves
