"""Price array helpers for date-bounded price lookups.

All helpers accept pre-extracted ``(idx_ns, vals)`` numpy arrays so callers
can cache them once per ticker and reuse them across many lookups (avoids
repeated `_price_arrays` calls).

`_price_arrays` is read from `analyzer.signals` (the shared price cache).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class AlignedPrice:
    """A positive finite market price aligned to a requested calendar date."""

    price: float
    date: pd.Timestamp
    staleness_days: int


def _valid_price_at(vals, pos: int) -> float | None:
    if pos < 0 or pos >= len(vals):
        return None
    price = float(vals[pos])
    if not np.isfinite(price) or price <= 0:
        return None
    return price


def _aligned_price_at_or_before_arrays(
    idx_ns, vals, target_date, max_staleness_days: int | None = None
) -> AlignedPrice | None:
    """Return the latest valid price at/before target with its real quote date."""
    target = pd.Timestamp(target_date).normalize()
    pos = int(np.searchsorted(idx_ns, target.value, side="right")) - 1
    if pos < 0:
        return None
    quote_date = pd.Timestamp(int(idx_ns[pos])).normalize()
    staleness_days = int((target - quote_date).days)
    if max_staleness_days is not None and staleness_days > max_staleness_days:
        return None
    price = _valid_price_at(vals, pos)
    if price is None:
        return None
    return AlignedPrice(price, quote_date, staleness_days)


def _aligned_price_on_or_after_arrays(
    idx_ns,
    vals,
    target_date,
    *,
    strictly_after: bool = False,
    max_wait_days: int | None = None,
) -> AlignedPrice | None:
    """Return the first valid price on/after target and its execution date.

    ``strictly_after`` is used for end-of-day signals that can only execute on
    the next tradable session.
    """
    target = pd.Timestamp(target_date).normalize()
    threshold = target + pd.Timedelta(days=1) if strictly_after else target
    pos = int(np.searchsorted(idx_ns, threshold.value, side="left"))
    while pos < len(idx_ns):
        quote_date = pd.Timestamp(int(idx_ns[pos])).normalize()
        wait_days = int((quote_date - target).days)
        if max_wait_days is not None and wait_days > max_wait_days:
            return None
        price = _valid_price_at(vals, pos)
        if price is not None:
            return AlignedPrice(price, quote_date, wait_days)
        pos += 1
    return None
