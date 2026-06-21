"""Price index + lookups: O(log N) binary search over per-ticker price arrays.

Module-level per-prices_df cache. Keyed by id(prices_df); cleaned up via a
weakref finalizer when the DataFrame is garbage-collected, so the cache
never returns data for an id that was reused by an unrelated DataFrame.
Stores per-ticker (sorted-non-NaN dates as int64 ns, values as np.ndarray)
tuples so price lookups become O(log N) via searchsorted instead of O(N) via
boolean masking + dropna.
"""

from __future__ import annotations

import weakref as _weakref

import numpy as np
import pandas as pd

from analyzer.signals.constants import _NS_PER_DAY


_PRICE_INDEX_DATA: dict[int, dict] = {}


def _price_index_for_df(prices_df: pd.DataFrame) -> dict:
    df_id = id(prices_df)
    by_ticker = _PRICE_INDEX_DATA.get(df_id)
    if by_ticker is None:
        by_ticker = {}
        _PRICE_INDEX_DATA[df_id] = by_ticker

        def _drop(_df_id=df_id):
            _PRICE_INDEX_DATA.pop(_df_id, None)

        try:
            _weakref.finalize(prices_df, _drop)
        except TypeError:
            # Object can't be weak-referenced; fall back to leaving the slot
            # in place (caller is responsible for clearing via _clear_price_index_cache).
            pass
    return by_ticker


def _price_arrays(prices_df: pd.DataFrame, ticker: str):
    """Return (dates_ns_int64, values_float64) for ``ticker`` in ``prices_df``,
    dropping NaNs. Result cached per (prices_df identity, ticker).

    Returns None if the ticker has no column or no non-NaN values. Dates are
    normalized to nanoseconds regardless of the source index's resolution
    (pandas 3+ defaults to microsecond resolution), so they compare directly
    against ``pd.Timestamp.value`` (always ns).
    """
    by_ticker = _price_index_for_df(prices_df)
    arrs = by_ticker.get(ticker, False)  # False = "not computed yet"
    if arrs is False:
        if ticker not in prices_df.columns:
            arrs = None
        else:
            s = prices_df[ticker].dropna()
            if s.empty:
                arrs = (None, None)
            else:
                idx = s.index
                # Normalize to nanoseconds. pandas 3+ exposes ``unit``;
                # pandas 2.x DatetimeIndex is always ns.
                unit = getattr(idx, "unit", "ns")
                if unit != "ns" and hasattr(idx, "as_unit"):
                    idx = idx.as_unit("ns")
                arrs = (
                    np.ascontiguousarray(idx.asi8, dtype=np.int64),
                    np.ascontiguousarray(s.values, dtype=np.float64),
                )
        by_ticker[ticker] = arrs
    return arrs


def _clear_price_index_cache() -> None:
    """Drop all cached price indices. Useful between unrelated runs."""
    _PRICE_INDEX_DATA.clear()


def _price_at_or_before(
    prices_df: pd.DataFrame,
    ticker: str,
    target_date: pd.Timestamp,
    max_staleness_days: int | None = None,
) -> float | None:
    arrs = _price_arrays(prices_df, ticker)
    if arrs is None:
        return None
    idx_ns, vals = arrs
    if idx_ns is None:
        return None
    target = pd.Timestamp(target_date).value
    # Rightmost position whose date <= target
    pos = int(np.searchsorted(idx_ns, target, side="right")) - 1
    if pos < 0:
        return None
    if max_staleness_days is not None:
        # Staleness in days: (target - found_date) ns / (1e9 * 86400)
        staleness_ns = target - int(idx_ns[pos])
        if staleness_ns > max_staleness_days * _NS_PER_DAY:
            return None
    return float(vals[pos])


def _price_at_or_near(
    prices_df: pd.DataFrame,
    ticker: str,
    target_date: pd.Timestamp,
    tolerance_days: int = 7,
) -> float | None:
    arrs = _price_arrays(prices_df, ticker)
    if arrs is None:
        return None
    idx_ns, vals = arrs
    if idx_ns is None:
        return None
    target = pd.Timestamp(target_date).value
    tol_ns = tolerance_days * _NS_PER_DAY
    lo = int(np.searchsorted(idx_ns, target - tol_ns, side="left"))
    hi = int(np.searchsorted(idx_ns, target + tol_ns, side="right"))
    if lo >= hi:
        return None
    window_dates = idx_ns[lo:hi]
    window_vals = vals[lo:hi]
    nearest = int(np.argmin(np.abs(window_dates - target)))
    return float(window_vals[nearest])


def _price_on_or_before(
    prices_df: pd.DataFrame,
    ticker: str,
    target_date: pd.Timestamp,
    max_staleness_days: int = 5,
) -> float | None:
    return _price_at_or_before(
        prices_df, ticker, target_date, max_staleness_days=max_staleness_days,
    )
