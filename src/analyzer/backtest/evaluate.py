"""Evaluate backtest recommendations: simulate entry/exit prices and SPY alpha.

For each row in `recommendations`, simulates the buy at the (dip-aware)
entry price and sell at the optimal-horizon exit price, then computes the
return, SPY alpha, and options leverage.

Heavy preprocessing (SPY benchmark, per-ticker price caches, slippage
multipliers) is hoisted out of the per-row loop.

Bug fixes applied here (see prices.py for Bug 1b/1c low-level fixes):

  Bug 1a — default use_dip_entry changed from True to False: honest baseline
            uses the latest close at or before as_of with zero entry delay;
            the same-day close is used when available.

  Bug 1b — SPY benchmark aligned with actual entry/exit dates: when
            use_dip_entry=True the SPY window is [entry_date, exit_date], not
            the fixed [as_of, as_of+horizon] window. Positions where no dip
            occurs within max_wait_days are NOT taken (no fallback fill).

  Bug 2  — Survivorship bias removed: when a ticker's price history ends before
            the exit date, the last available price is used as exit price and
            the trade IS included. A `bt_delisted` flag marks rows whose last
            quote is after entry. Stale fallback coverage is reported through
            `bt_coverage` and `bt_stale_exit`. Tickers with no actionable price
            remain untradeable and count in result.attrs["n_no_price"].

  Bug 3  — Merge fan-out fixed: results are merged on a per-row index key
            (_bt_idx) rather than on ticker alone, so two recommendations for
            the same ticker produce exactly two output rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analyzer.backtest.prices import (
    _find_dip_entry_arrays,
    _price_at_or_before_arrays,
    _price_on_or_before_arrays,
)
from analyzer.signals import _price_arrays

NS_PER_DAY = 86_400_000_000_000
_EXIT_STALENESS_DAYS = 25

# Sentinel returned by _evaluate_one_recommendation when a ticker is
# discovered to have been delisted BEFORE the entry/as_of date — meaning the
# position was never actionable.  Distinct from None (other skips) so the
# caller can count it in n_no_price.
_UNTRADEABLE = object()

_EMPTY_BT_COLS = [
    "bt_entry_price", "bt_exit_price", "bt_raw_return_pct", "bt_return_pct",
    "bt_leverage", "bt_spy_return_pct", "bt_alpha_pct", "bt_entry_delay",
    "bt_delisted", "bt_coverage", "bt_stale_exit",
]


def evaluate_backtest(
    recommendations: pd.DataFrame,
    prices_df: pd.DataFrame,
    as_of_date: pd.Timestamp,
    horizon: int,
    max_staleness_days: int | None = 30,
    entry_slippage_bps: float = 10.0,
    exit_slippage_bps: float = 10.0,
    use_dip_entry: bool = False,   # Bug 1a: was True; False = honest baseline
    pullback_pct: float = 0.05,
    max_wait_days: int = 10,
) -> pd.DataFrame:
    if recommendations.empty:
        return recommendations

    entry_mult, exit_mult = _slippage_multipliers(entry_slippage_bps, exit_slippage_bps)

    horizons = (
        recommendations["optimal_horizon"].values
        if "optimal_horizon" in recommendations.columns
        else None
    )

    spy_arrs = _price_arrays(prices_df, "SPY")
    spy_ns, spy_vals = (spy_arrs if spy_arrs and spy_arrs[0] is not None else (None, None))

    spy_start, spy_entry_adj = _spy_start(spy_ns, spy_vals, as_of_date, entry_mult, max_staleness_days)
    spy_ends, spy_returns = _spy_returns_by_horizon(
        spy_ns, spy_vals, as_of_date, spy_start, spy_entry_adj, exit_mult, horizons, horizon,
    )

    tickers = recommendations["ticker"].tolist()
    ticker_horizons = (
        [int(h) for h in horizons]
        if horizons is not None
        else [horizon] * len(tickers)
    )

    price_cache = {t: _price_arrays(prices_df, t) for t in tickers}

    inst_type_arr, amount_arr = _extract_recommendation_arrays(recommendations)

    rows = []
    n_no_price = 0
    n_delisted = 0

    for i in range(len(recommendations)):
        ticker = recommendations["ticker"].iloc[i]
        t_horizon = ticker_horizons[i]

        # Check price availability before entering the per-rec helper so we can
        # count no-price skips separately (Bug 2: distinguish "no data" from
        # "data exists but stale/delisted").
        cached = price_cache.get(ticker)
        if cached is None or (isinstance(cached, tuple) and cached[0] is None):
            n_no_price += 1
            continue

        row = _evaluate_one_recommendation(
            ticker, i, t_horizon, cached,
            as_of_date, entry_mult, exit_mult,
            spy_ns, spy_vals,
            spy_start, spy_returns, spy_ends,
            use_dip_entry, pullback_pct, max_wait_days,
            max_staleness_days,
            inst_type_arr[i] if inst_type_arr is not None else None,
            amount_arr[i] if amount_arr is not None else None,
        )
        if row is _UNTRADEABLE:
            # Ticker was already delisted before entry — never actionable
            n_no_price += 1
        elif row is not None:
            if row.get("bt_delisted"):
                n_delisted += 1
            rows.append(row)

    if not rows:
        result = _empty_eval_joined(recommendations)
        result.attrs["n_no_price"] = n_no_price
        result.attrs["n_delisted"] = n_delisted
        return result

    eval_df = pd.DataFrame(rows)

    # Bug 3: merge on per-row index (_bt_idx) rather than ticker so that two
    # recommendations for the same ticker produce exactly two output rows,
    # not a cartesian product of N×N rows.
    recs = recommendations.copy().reset_index(drop=True)
    recs["_bt_idx"] = recs.index
    result = recs.merge(
        eval_df.drop(columns=["ticker"], errors="ignore"),
        on="_bt_idx",
        how="left",
    ).drop(columns=["_bt_idx"])

    result.attrs["n_no_price"] = n_no_price
    result.attrs["n_delisted"] = n_delisted
    return result


def _slippage_multipliers(entry_slippage_bps: float, exit_slippage_bps: float) -> tuple[float, float]:
    return (1.0 + entry_slippage_bps / 10000, 1.0 - exit_slippage_bps / 10000)


def _spy_start(spy_ns, spy_vals, as_of_date, entry_mult, max_staleness_days):
    if spy_ns is None:
        return None, 0.0
    start = _price_at_or_before_arrays(spy_ns, spy_vals, as_of_date, max_staleness_days)
    if not start:
        return None, 0.0
    return start, start * entry_mult


def _spy_returns_by_horizon(
    spy_ns, spy_vals, as_of_date, spy_start, spy_entry_adj,
    exit_mult, horizons, default_horizon,
):
    """Pre-compute SPY returns for every distinct horizon from as_of_date.

    These are the baseline SPY returns when entry_delay == 0.  When
    use_dip_entry=True and entry_delay > 0 the per-recommendation SPY return
    is computed separately by _compute_spy_return_shifted.
    """
    spy_ends: dict[int, float | None] = {}
    spy_returns: dict[int, float] = {}
    if not spy_start:
        return spy_ends, spy_returns
    horizon_iter = set(horizons) if horizons is not None else [default_horizon]
    for h in horizon_iter:
        spy_exit_ns = as_of_date.value + int(h) * NS_PER_DAY
        se = _price_on_or_before_arrays(spy_ns, spy_vals, pd.Timestamp(spy_exit_ns), max_staleness_days=30)
        spy_ends[h] = se
        if se:
            spy_exit_adj = se * exit_mult
            spy_returns[h] = round((spy_exit_adj / spy_entry_adj - 1) * 100, 2)
        else:
            spy_returns[h] = 0.0
    return spy_ends, spy_returns


def _compute_spy_return_shifted(
    spy_ns, spy_vals, as_of_ns, entry_delay, t_horizon,
    entry_mult, exit_mult, max_staleness_days,
):
    """Compute SPY return for the shifted window [as_of + entry_delay, …+ horizon].

    Bug 1b: when use_dip_entry=True the SPY benchmark must reflect the same
    calendar window as the actual ticker position, not the fixed as_of window.
    """
    if spy_ns is None:
        return None
    entry_ns = as_of_ns + entry_delay * NS_PER_DAY
    exit_ns = entry_ns + t_horizon * NS_PER_DAY
    spy_entry = _price_at_or_before_arrays(spy_ns, spy_vals, pd.Timestamp(entry_ns), max_staleness_days)
    if not spy_entry:
        return None
    spy_exit = _price_on_or_before_arrays(spy_ns, spy_vals, pd.Timestamp(exit_ns), max_staleness_days=30)
    if not spy_exit:
        return None
    return round((spy_exit * exit_mult / (spy_entry * entry_mult) - 1) * 100, 2)


def _extract_recommendation_arrays(recommendations: pd.DataFrame) -> tuple:
    """Pull pre-aligned numpy arrays out of the recommendations DataFrame to
    avoid per-row Series creation in the hot loop."""
    has_inst_type = "instrument_type" in recommendations.columns
    has_amount = "amount_midpoint" in recommendations.columns
    inst_type_arr = recommendations["instrument_type"].values if has_inst_type else None
    amount_arr = recommendations["amount_midpoint"].values if has_amount else None
    return inst_type_arr, amount_arr


def _empty_eval_joined(recommendations: pd.DataFrame) -> pd.DataFrame:
    """Return the original recommendations frame with empty bt_* columns."""
    recommendations = recommendations.copy()
    for col in _EMPTY_BT_COLS:
        recommendations[col] = None
    return recommendations


def _evaluate_one_recommendation(
    ticker, row_idx, t_horizon, cached,
    as_of_date, entry_mult, exit_mult,
    spy_ns, spy_vals,
    spy_start, spy_returns, spy_ends,
    use_dip_entry, pullback_pct, max_wait_days,
    max_staleness_days,
    inst_type_val, amount_val,
) -> dict | None:
    """Returns dict of bt_* fields for one ticker, or None to skip.

    ``row_idx`` is the position in the recommendations DataFrame; it is
    written to the dict as ``_bt_idx`` so the caller can do a 1:1 merge
    (Bug 3 fix).
    """
    # Caller already verified cached is a valid (idx_ns, vals) tuple.
    idx_ns, vals = cached

    entry, entry_delay = _resolve_entry(
        use_dip_entry, idx_ns, vals, as_of_date,
        pullback_pct, max_wait_days, max_staleness_days,
    )
    if not entry:
        return None

    as_of_ns = as_of_date.value
    # Bug 1c: entry_delay is now calendar days (fixed in prices.py), so this
    # arithmetic is correct even when weekends separate the dip from as_of.
    exit_ns = as_of_ns + (entry_delay + t_horizon) * NS_PER_DAY

    # Bug 2: when the exit price lookup fails the staleness check (ticker
    # delisted/suspended), fall back to the last available price rather than
    # silently dropping the trade from both numerator and denominator.
    exit_price = _price_on_or_before_arrays(
        idx_ns, vals, pd.Timestamp(exit_ns), max_staleness_days=_EXIT_STALENESS_DAYS,
    )
    is_delisted = False
    stale_exit = False
    if not exit_price:
        # Finding 2 fix: only treat as "delisted during hold" when the last
        # known price is STRICTLY AFTER the entry date.  If the last price
        # predates or equals the entry (ticker already dead at recommendation
        # time), the position was never actionable — return _UNTRADEABLE so
        # the caller can count it in n_no_price.
        entry_ns = as_of_ns + entry_delay * NS_PER_DAY
        fallback_pos = int(np.searchsorted(idx_ns, int(exit_ns), side="right")) - 1
        if fallback_pos >= 0 and int(idx_ns[fallback_pos]) > entry_ns:
            exit_price = float(vals[fallback_pos])
            is_delisted = True
            stale_exit = (
                int(exit_ns) - int(idx_ns[fallback_pos])
                > _EXIT_STALENESS_DAYS * NS_PER_DAY
            )
        else:
            # No usable price: either no data at all, or ticker already
            # delisted before/at entry — caller counts this in n_no_price.
            return _UNTRADEABLE

    # Bug 1b: SPY return must cover the same calendar window as the position.
    # When entry_delay > 0 (dip entry) the window is shifted; precomputed
    # spy_returns (anchored at as_of) would produce phantom alpha.
    if use_dip_entry and entry_delay > 0:
        spy_ret = _compute_spy_return_shifted(
            spy_ns, spy_vals, as_of_ns, entry_delay, t_horizon,
            entry_mult, exit_mult, max_staleness_days,
        )
        if spy_ret is None:
            return None  # SPY data unavailable for shifted window — skip
    else:
        spy_ret = spy_returns.get(t_horizon, 0.0)
        if not spy_start or (spy_ret == 0.0 and not spy_ends.get(t_horizon)):
            return None

    inst_type = str(inst_type_val) if inst_type_val is not None and pd.notna(inst_type_val) else "stock"
    amount = amount_val if amount_val is not None else None
    row = _bt_row(
        ticker, entry, entry_delay, exit_price, t_horizon,
        entry_mult, exit_mult, spy_ret, inst_type, amount,
    )
    row["_bt_idx"] = row_idx      # Bug 3: used for 1:1 merge
    row["bt_delisted"] = is_delisted  # Bug 2: flag for coverage reporting
    row["bt_coverage"] = "stale" if stale_exit else "complete"
    row["bt_stale_exit"] = stale_exit
    return row


def _resolve_entry(use_dip_entry, idx_ns, vals, as_of_date, pullback_pct, max_wait_days, max_staleness_days):
    """Compute entry price and entry delay (calendar days), with optional dip timing.

    Bug 1b: when use_dip_entry=True and no dip is found within max_wait_days,
    returns (None, 0) so the caller skips the position.  There is no fallback
    fill at a future known price.
    """
    if use_dip_entry:
        entry, entry_delay = _find_dip_entry_arrays(
            idx_ns, vals, as_of_date, pullback_pct, max_wait_days,
        )
        # (0.0, 0) from _find_dip_entry_arrays means "no dip found"
        if entry <= 0:
            return None, 0  # No fallback — position is not taken
    else:
        entry = _price_at_or_before_arrays(idx_ns, vals, as_of_date, max_staleness_days)
        entry_delay = 0
    return entry, entry_delay


def _bt_row(ticker, entry, entry_delay, exit_price, t_horizon, entry_mult, exit_mult, spy_ret, inst_type, amount):
    """Build the bt_* row dict after all inputs are resolved."""
    from analyzer.options import estimate_options_leverage

    entry_adj = entry * entry_mult
    exit_adj = exit_price * exit_mult
    return_pct = (exit_adj / entry_adj - 1) * 100
    leverage = estimate_options_leverage(inst_type, amount)
    leveraged_return_pct = return_pct * leverage
    alpha_pct = leveraged_return_pct - spy_ret

    return {
        "ticker": ticker,
        "bt_entry_price": round(entry, 2),
        "bt_exit_price": round(exit_price, 2),
        "bt_raw_return_pct": round(return_pct, 2),
        "bt_return_pct": round(leveraged_return_pct, 2),
        "bt_leverage": round(leverage, 2),
        "bt_spy_return_pct": spy_ret,
        "bt_alpha_pct": round(alpha_pct, 2),
        "bt_horizon_days": t_horizon,
        "bt_entry_delay": entry_delay,
        "bt_coverage": "complete",
        "bt_stale_exit": False,
    }
