"""Price array helpers: dip-entry search and date-bounded price lookups.

All helpers accept pre-extracted ``(idx_ns, vals)`` numpy arrays so callers
can cache them once per ticker and reuse them across many lookups (avoids
repeated `_price_arrays` calls).

`_price_arrays` is read from `analyzer.signals` (the shared price cache).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from analyzer.signals import _price_arrays


@dataclass(frozen=True, slots=True)
class AlignedPrice:
    """A positive finite market price aligned to a requested calendar date."""

    price: float
    date: pd.Timestamp
    staleness_days: int


NS_PER_DAY = 86_400_000_000_000


def _find_dip_entry(
    prices_df: pd.DataFrame,
    ticker: str,
    as_of_date: pd.Timestamp,
    pullback_pct: float = 0.05,
    max_wait_days: int = 10,
) -> tuple[float, int]:
    """Find dip entry price after as_of_date (which represents disclosure date in backtest).

    Returns (entry_price, calendar_delay_days) when a dip is found.
    Returns (0.0, 0) when no dip occurs within max_wait_days — caller decides
    whether to skip the position or fall back.
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
    """Find dip entry using pre-extracted price arrays.

    Returns ``(dip_price, calendar_delay_days)`` when a dip of at least
    ``pullback_pct`` is found within ``max_wait_days`` calendar days after
    ``as_of_date``.

    Returns ``(0.0, 0)`` when no dip is found.  Callers that model a causal
    limit order (``use_dip_entry=True``) must treat a zero return as "no fill"
    and skip the position rather than falling back to the as-of price.

    Bug 1b fix: no automatic fallback to disc_price when no dip — eliminated
    lookahead that let the backtest "know" a dip would not occur.
    Bug 1c fix: delay is returned as calendar days (``(dip_ns - target_ns) //
    NS_PER_DAY``), not as an array-row index which undercounts over
    weekends/holidays.
    """
    target_ns = pd.Timestamp(as_of_date).value
    window_end_ns = target_ns + max_wait_days * NS_PER_DAY

    # First price on or after as_of_date
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
        # Bug 1c: compute actual calendar days between as_of and dip date,
        # not the array row index (which is shorter when weekends are absent).
        dip_idx = lo + int(hits[0])
        dip_ns = int(idx_ns[dip_idx])
        calendar_days = int((dip_ns - target_ns) // NS_PER_DAY)
        return float(window_vals[hits[0]]), calendar_days

    # No dip found — return sentinel so callers can skip the position
    return 0.0, 0


def _valid_price_at(vals, pos: int, *, allow_zero: bool = False) -> float | None:
    if pos < 0 or pos >= len(vals):
        return None
    price = float(vals[pos])
    if not np.isfinite(price) or price < 0 or (price == 0 and not allow_zero):
        return None
    return price


def _aligned_price_at_or_before_arrays(
    idx_ns,
    vals,
    target_date,
    max_staleness_days: int | None = None,
    *,
    allow_zero: bool = False,
) -> AlignedPrice | None:
    """Return the latest valid price at/before target with its real quote date."""
    target = pd.Timestamp(target_date).normalize()
    pos = int(np.searchsorted(idx_ns, target.value, side="right")) - 1
    while pos >= 0:
        price = _valid_price_at(vals, pos, allow_zero=allow_zero)
        quote_date = pd.Timestamp(int(idx_ns[pos])).normalize()
        staleness_days = int((target - quote_date).days)
        if max_staleness_days is not None and staleness_days > max_staleness_days:
            return None
        if price is not None:
            return AlignedPrice(price, quote_date, staleness_days)
        pos -= 1
    return None


def _aligned_price_on_or_after_arrays(
    idx_ns,
    vals,
    target_date,
    *,
    strictly_after: bool = False,
    max_wait_days: int | None = None,
    allow_zero: bool = False,
) -> AlignedPrice | None:
    """Return the first valid price on/after target and its execution date.

    ``strictly_after=True`` models an order created from end-of-session inputs:
    it cannot execute at the close used to create the signal and therefore waits
    for the next tradable session.
    """
    target = pd.Timestamp(target_date).normalize()
    threshold = target + pd.Timedelta(days=1) if strictly_after else target
    pos = int(np.searchsorted(idx_ns, threshold.value, side="left"))
    while pos < len(idx_ns):
        quote_date = pd.Timestamp(int(idx_ns[pos])).normalize()
        wait_days = int((quote_date - target).days)
        if max_wait_days is not None and wait_days > max_wait_days:
            return None
        price = _valid_price_at(vals, pos, allow_zero=allow_zero)
        if price is not None:
            return AlignedPrice(price, quote_date, wait_days)
        pos += 1
    return None


def _next_tradable_price_arrays(
    idx_ns, vals, signal_date, max_wait_days: int | None = 7
) -> AlignedPrice | None:
    """Return the first valid session strictly after an end-of-day signal."""
    return _aligned_price_on_or_after_arrays(
        idx_ns,
        vals,
        signal_date,
        strictly_after=True,
        max_wait_days=max_wait_days,
    )


def _price_at_or_before_arrays(idx_ns, vals, target_date, max_staleness_days=None):
    """Legacy scalar lookup; aligned execution callers use the strict helpers."""
    target = pd.Timestamp(target_date).value
    pos = int(np.searchsorted(idx_ns, target, side="right")) - 1
    if pos < 0:
        return None
    if max_staleness_days is not None:
        staleness_ns = target - int(idx_ns[pos])
        if staleness_ns > max_staleness_days * NS_PER_DAY:
            return None
    return float(vals[pos])


def _price_before_arrays(idx_ns, vals, target_date, max_staleness_days=None):
    """Legacy scalar lookup strictly before ``target_date``."""
    target = pd.Timestamp(target_date).value
    pos = int(np.searchsorted(idx_ns, target, side="left")) - 1
    if pos < 0:
        return None
    if max_staleness_days is not None:
        staleness_ns = target - int(idx_ns[pos])
        if staleness_ns > max_staleness_days * NS_PER_DAY:
            return None
    return float(vals[pos])


def _price_on_or_before_arrays(idx_ns, vals, target_date, max_staleness_days=5):
    """Backward-compatible scalar on-or-before lookup."""
    return _price_at_or_before_arrays(
        idx_ns, vals, target_date, max_staleness_days=max_staleness_days
    )
