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
from analyzer.backtest.prices import (
    _aligned_price_at_or_before_arrays,
    _next_tradable_price_arrays,
)
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

    return _build_curves_for_rows(
        eligible, prices_df, horizon, available_through=as_of_date
    )


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

    return _build_curves_for_rows(
        all_purchases, prices_df, horizon, available_through=as_of_date
    )


def _build_curves_for_rows(
    rows: pd.DataFrame,
    prices_df: pd.DataFrame,
    horizon: int,
    *,
    available_through: pd.Timestamp | None = None,
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
    tickers = rows["ticker"].values
    horizon_ns = pd.Timedelta(days=horizon).value
    available_ns = (
        pd.Timestamp(available_through).normalize().value
        if available_through is not None
        else None
    )

    for i in range(len(rows)):
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

        # Disclosure metadata has no publication timestamp. Historical curves
        # are therefore entered on the first valid session strictly after the
        # filing date, never at a pre-filing or same-close repository price.
        entry = _next_tradable_price_arrays(
            idx_ns, vals, pd.Timestamp(disclosures[i]), max_wait_days=7
        )
        if entry is None:
            continue

        entry_ns = entry.date.value
        end_ns = entry_ns + horizon_ns
        if available_ns is not None and end_ns > available_ns:
            continue
        terminal = _aligned_price_at_or_before_arrays(
            idx_ns,
            vals,
            pd.Timestamp(end_ns),
            max_staleness_days=7,
            allow_zero=True,
        )
        if terminal is None:
            continue

        lo = int(np.searchsorted(idx_ns, entry_ns, side="left"))
        hi = int(np.searchsorted(idx_ns, end_ns, side="right"))
        window = vals[lo:hi]
        valid = np.isfinite(window) & (window >= 0)
        window = window[valid]
        if len(window) < 3:
            continue
        curves.append(window / entry.price - 1.0)

    return curves
