"""Price array helpers: dip-entry search and date-bounded price lookups.

All helpers accept pre-extracted ``(idx_ns, vals)`` numpy arrays so callers
can cache them once per ticker and reuse them across many lookups (avoids
repeated `_price_arrays` calls).

`_price_arrays` is read from `analyzer.signals` (the shared price cache).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analyzer.signals import _price_arrays


def _find_dip_entry(
    prices_df: pd.DataFrame,
    ticker: str,
    as_of_date: pd.Timestamp,
    pullback_pct: float = 0.05,
    max_wait_days: int = 10,
) -> tuple[float, int]:
    """Find dip entry price after as_of_date (which represents disclosure date in backtest).

    Returns (entry_price, delay_days). If no dip, returns (price_at_as_of, 0).
    """
    arrs = _price_arrays(prices_df, ticker)
    if arrs is None:
        return 0.0, 0
    idx_ns, vals = arrs
    if idx_ns is None:
        return 0.0, 0
    return _find_dip_entry_arrays(idx_ns, vals, as_of_date, pullback_pct, max_wait_days)


def _find_dip_entry_arrays(
    idx_ns,
    vals,
    as_of_date,
    pullback_pct: float = 0.05,
    max_wait_days: int = 10,
):
    """Find dip entry using pre-extracted price arrays (avoids repeated _price_arrays lookup)."""
    target_ns = pd.Timestamp(as_of_date).value
    window_end_ns = target_ns + max_wait_days * 86_400_000_000_000

    # First price >= as_of_date
    lo = int(np.searchsorted(idx_ns, target_ns, side="left"))
    if lo >= len(idx_ns):
        return 0.0, 0
    disc_price = float(vals[lo])
    if disc_price <= 0:
        return 0.0, 0

    # Window of prices within [as_of_date, as_of_date + max_wait_days]
    hi = int(np.searchsorted(idx_ns, window_end_ns, side="right"))
    window_vals = vals[lo:hi]
    if len(window_vals) == 0:
        return 0.0, 0

    target_price = disc_price * (1 - pullback_pct)
    hits = np.where(window_vals <= target_price)[0]
    if len(hits) > 0:
        return float(window_vals[hits[0]]), int(hits[0])

    return disc_price, 0


def _price_at_or_before_arrays(idx_ns, vals, target_date, max_staleness_days=None):
    """Price lookup using pre-extracted arrays."""
    target = pd.Timestamp(target_date).value
    pos = int(np.searchsorted(idx_ns, target, side="right")) - 1
    if pos < 0:
        return None
    if max_staleness_days is not None:
        staleness_ns = target - int(idx_ns[pos])
        if staleness_ns > max_staleness_days * 86_400_000_000_000:
            return None
    return float(vals[pos])


def _price_on_or_before_arrays(idx_ns, vals, target_date, max_staleness_days=5):
    """Price lookup using pre-extracted arrays."""
    return _price_at_or_before_arrays(idx_ns, vals, target_date, max_staleness_days=max_staleness_days)
