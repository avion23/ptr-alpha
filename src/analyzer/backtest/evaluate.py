"""Evaluate backtest recommendations: simulate entry/exit prices and SPY alpha.

For each row in `recommendations`, simulates the buy at the (dip-aware)
entry price and sell at the optimal-horizon exit price, then computes the
return, SPY alpha, and options leverage.

Heavy preprocessing (SPY benchmark, per-ticker price caches, slippage
multipliers) is hoisted out of the per-row loop.

Bug fixes applied here (see prices.py for Bug 1b/1c low-level fixes):

  Timing — The ordinary baseline enters on the first session strictly after
           the decision. Dip orders also require a future fill and never fall
           back after observing that no dip occurred.

  Coverage — Entry occurs on the first tradable session after the decision.
             Security and SPY returns use identical entry/exit dates. Missing
             horizon outcomes remain unavailable; a last quote is never
             invented as a delisting or liquidation return.

  Bug 3  — Merge fan-out fixed: results are merged on a per-row index key
            (_bt_idx) rather than on ticker alone, so two recommendations for
            the same ticker produce exactly two output rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analyzer.backtest.prices import _find_dip_entry_arrays
from analyzer.price_repository import next_nyse_session, nyse_sessions, previous_nyse_session
from analyzer.signals import _price_arrays

NS_PER_DAY = 86_400_000_000_000
_ENTRY_STALENESS_DAYS = 7

# Sentinel returned by _evaluate_one_recommendation when a ticker is
# discovered to have been delisted BEFORE the entry/as_of date — meaning the
# position was never actionable.  Distinct from None (other skips) so the
# caller can count it in n_no_price.
_UNTRADEABLE = object()

_EMPTY_BT_COLS = [
    "bt_entry_price",
    "bt_entry_date",
    "bt_exit_price",
    "bt_exit_date",
    "bt_raw_return_pct",
    "bt_return_pct",
    "bt_leverage",
    "bt_spy_return_pct",
    "bt_alpha_pct",
    "bt_entry_delay",
    "bt_delisted",
    "bt_coverage",
    "bt_stale_exit",
]


def evaluate_backtest(
    recommendations: pd.DataFrame,
    prices_df: pd.DataFrame,
    as_of_date: pd.Timestamp,
    horizon: int,
    max_staleness_days: int | None = 30,
    entry_slippage_bps: float = 10.0,
    exit_slippage_bps: float = 10.0,
    use_dip_entry: bool = False,  # Bug 1a: was True; False = honest baseline
    pullback_pct: float = 0.05,
    max_wait_days: int = 10,
) -> pd.DataFrame:
    if recommendations.empty:
        return recommendations

    prices_df = _normalize_price_frame(prices_df)
    as_of_date = _normalize_date(as_of_date)
    entry_mult, exit_mult = _slippage_multipliers(entry_slippage_bps, exit_slippage_bps)

    horizons = (
        recommendations["optimal_horizon"].values
        if "optimal_horizon" in recommendations.columns
        else None
    )

    spy_arrs = _price_arrays(prices_df, "SPY")
    spy_ns, spy_vals = (
        spy_arrs if spy_arrs and spy_arrs[0] is not None else (None, None)
    )

    tickers = recommendations["ticker"].tolist()
    ticker_horizons = (
        [int(h) for h in horizons] if horizons is not None else [horizon] * len(tickers)
    )

    price_cache = {t: _price_arrays(prices_df, t) for t in tickers}

    inst_type_arr, amount_arr = _extract_recommendation_arrays(recommendations)

    rows = []
    n_no_price = 0
    n_delisted = 0
    n_unavailable = 0

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
            ticker,
            i,
            t_horizon,
            cached,
            as_of_date,
            entry_mult,
            exit_mult,
            spy_ns,
            spy_vals,
            use_dip_entry,
            pullback_pct,
            max_wait_days,
            max_staleness_days,
            inst_type_arr[i] if inst_type_arr is not None else None,
            amount_arr[i] if amount_arr is not None else None,
        )
        if row is _UNTRADEABLE:
            # Ticker was already delisted before entry — never actionable
            n_no_price += 1
        elif row is not None:
            if row.get("bt_coverage") == "unavailable":
                n_unavailable += 1
            if row.get("bt_delisted"):
                n_delisted += 1
            rows.append(row)

    if not rows:
        result = _empty_eval_joined(recommendations)
        result.attrs["n_no_price"] = n_no_price
        result.attrs["n_delisted"] = n_delisted
        result.attrs["n_unavailable"] = n_unavailable
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
    result.attrs["n_unavailable"] = n_unavailable
    return result


def _slippage_multipliers(
    entry_slippage_bps: float, exit_slippage_bps: float
) -> tuple[float, float]:
    return (1.0 + entry_slippage_bps / 10000, 1.0 - exit_slippage_bps / 10000)


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
    ticker,
    row_idx,
    t_horizon,
    cached,
    as_of_date,
    entry_mult,
    exit_mult,
    spy_ns,
    spy_vals,
    use_dip_entry,
    pullback_pct,
    max_wait_days,
    max_staleness_days,
    inst_type_val,
    amount_val,
) -> dict | None:
    """Evaluate one recommendation without guessing unavailable outcomes."""
    idx_ns, vals = cached
    entry, entry_delay, entry_date_ns = _resolve_entry(
        use_dip_entry,
        idx_ns,
        vals,
        as_of_date,
        pullback_pct,
        max_wait_days,
        max_staleness_days,
    )
    if not entry or entry_date_ns is None:
        return None if use_dip_entry else _UNTRADEABLE

    exit_target_ns = entry_date_ns + int(t_horizon) * NS_PER_DAY
    exit_date_ns = previous_nyse_session(pd.Timestamp(exit_target_ns)).value
    exit_pos = int(np.searchsorted(idx_ns, exit_date_ns, side="left"))
    exit_price = None
    if (
        exit_date_ns >= entry_date_ns
        and exit_pos < len(idx_ns)
        and int(idx_ns[exit_pos]) == exit_date_ns
    ):
        candidate = float(vals[exit_pos])
        # An observed zero is a realized total loss, not missing data.
        if np.isfinite(candidate) and candidate >= 0:
            exit_price = candidate

    inst_type = (
        str(inst_type_val)
        if inst_type_val is not None and pd.notna(inst_type_val)
        else "stock"
    )
    amount = amount_val if amount_val is not None else None
    if exit_price is None or exit_date_ns is None:
        return _unavailable_bt_row(
            ticker, row_idx, entry, entry_date_ns, entry_delay, inst_type, amount
        )

    # Benchmark the exact security holding endpoints. A nearest independent
    # SPY date would compare different holding periods and manufacture alpha.
    spy_entry = _price_at_exact_ns(spy_ns, spy_vals, entry_date_ns)
    spy_exit = _price_at_exact_ns(spy_ns, spy_vals, exit_date_ns)
    if not spy_entry or not spy_exit:
        return _unavailable_bt_row(
            ticker, row_idx, entry, entry_date_ns, entry_delay, inst_type, amount
        )
    spy_ret = round(
        (spy_exit * exit_mult / (spy_entry * entry_mult) - 1) * 100,
        2,
    )

    row = _bt_row(
        ticker,
        entry,
        entry_delay,
        exit_price,
        t_horizon,
        entry_mult,
        exit_mult,
        spy_ret,
        inst_type,
        amount,
    )
    row["_bt_idx"] = row_idx
    row["bt_entry_date"] = pd.Timestamp(entry_date_ns).date()
    row["bt_exit_date"] = pd.Timestamp(exit_date_ns).date()
    row["bt_delisted"] = False
    row["bt_coverage"] = "complete"
    row["bt_stale_exit"] = False
    return row


def _normalize_date(value) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tz is not None:
        timestamp = timestamp.tz_localize(None)
    return timestamp.normalize()


def _normalize_price_frame(prices: pd.DataFrame) -> pd.DataFrame:
    try:
        index = pd.DatetimeIndex(pd.to_datetime(prices.index))
    except (TypeError, ValueError) as exc:
        raise ValueError("Price index must contain valid dates") from exc
    if index.tz is not None:
        index = index.tz_localize(None)
    index = index.normalize()
    if index.has_duplicates:
        raise ValueError("Price index contains duplicate calendar dates")
    normalized = prices.copy()
    normalized.index = index
    return normalized.sort_index()


def _price_at_exact_ns(idx_ns, vals, target_ns):
    if idx_ns is None or vals is None:
        return None
    pos = int(np.searchsorted(idx_ns, target_ns, side="left"))
    if pos >= len(idx_ns) or int(idx_ns[pos]) != int(target_ns):
        return None
    value = float(vals[pos])
    return value if np.isfinite(value) and value > 0 else None


def _unavailable_bt_row(
    ticker, row_idx, entry, entry_date_ns, entry_delay, inst_type, amount
):
    from analyzer.options import estimate_options_leverage

    return {
        "_bt_idx": row_idx,
        "ticker": ticker,
        "bt_entry_price": round(float(entry), 2),
        "bt_entry_date": pd.Timestamp(entry_date_ns).date(),
        "bt_exit_price": np.nan,
        "bt_exit_date": None,
        "bt_raw_return_pct": np.nan,
        "bt_return_pct": np.nan,
        "bt_leverage": estimate_options_leverage(inst_type, amount),
        "bt_spy_return_pct": np.nan,
        "bt_alpha_pct": np.nan,
        "bt_entry_delay": entry_delay,
        "bt_delisted": False,
        "bt_coverage": "unavailable",
        "bt_stale_exit": True,
    }


def _resolve_entry(
    use_dip_entry,
    idx_ns,
    vals,
    as_of_date,
    pullback_pct,
    max_wait_days,
    max_staleness_days,
):
    """Return a fill strictly after the decision date."""
    as_of_ns = pd.Timestamp(as_of_date).value
    if use_dip_entry:
        entry, entry_delay = _find_dip_entry_arrays(
            idx_ns,
            vals,
            as_of_date,
            pullback_pct,
            max_wait_days,
        )
        if entry <= 0 or entry_delay <= 0:
            return None, 0, None
        entry_date_ns = as_of_ns + entry_delay * NS_PER_DAY
        return entry, entry_delay, entry_date_ns

    max_delay = _ENTRY_STALENESS_DAYS
    if max_staleness_days is not None:
        max_delay = min(max_delay, max_staleness_days)
    first_session = next_nyse_session(pd.Timestamp(as_of_ns))
    last_date = pd.Timestamp(as_of_ns + max_delay * NS_PER_DAY)
    valid_sessions = {
        session.value for session in nyse_sessions(first_session, last_date)
    }
    pos = int(np.searchsorted(idx_ns, first_session.value, side="left"))
    while pos < len(idx_ns):
        entry_date_ns = int(idx_ns[pos])
        delay = int((entry_date_ns - as_of_ns) // NS_PER_DAY)
        if delay > max_delay:
            return None, 0, None
        value = float(vals[pos])
        if entry_date_ns in valid_sessions and np.isfinite(value) and value > 0:
            return value, delay, entry_date_ns
        pos += 1
    return None, 0, None


def _bt_row(
    ticker,
    entry,
    entry_delay,
    exit_price,
    t_horizon,
    entry_mult,
    exit_mult,
    spy_ret,
    inst_type,
    amount,
):
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
