"""Evaluate backtest recommendations: simulate entry/exit prices and SPY alpha.

For each row in `recommendations`, simulates the buy at the (dip-aware)
entry price and sell at the optimal-horizon exit price, then computes the
return, SPY alpha, and options leverage.

Heavy preprocessing (SPY benchmark, per-ticker price caches, slippage
multipliers) is hoisted out of the per-row loop.
"""

from __future__ import annotations

import pandas as pd

from analyzer.backtest.prices import (
    _find_dip_entry_arrays,
    _price_at_or_before_arrays,
    _price_on_or_before_arrays,
)
from analyzer.signals import _price_arrays

NS_PER_DAY = 86_400_000_000_000

_EMPTY_BT_COLS = [
    "bt_entry_price", "bt_exit_price", "bt_raw_return_pct", "bt_return_pct",
    "bt_leverage", "bt_spy_return_pct", "bt_alpha_pct", "bt_entry_delay",
]


def evaluate_backtest(
    recommendations: pd.DataFrame,
    prices_df: pd.DataFrame,
    as_of_date: pd.Timestamp,
    horizon: int,
    max_staleness_days: int | None = 30,
    entry_slippage_bps: float = 10.0,
    exit_slippage_bps: float = 10.0,
    use_dip_entry: bool = True,
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
    for i in range(len(recommendations)):
        ticker = recommendations["ticker"].iloc[i]
        t_horizon = ticker_horizons[i]
        row = _evaluate_one_recommendation(
            ticker, t_horizon, price_cache.get(ticker),
            as_of_date, entry_mult, exit_mult,
            spy_start, spy_returns, spy_ends,
            use_dip_entry, pullback_pct, max_wait_days,
            max_staleness_days,
            inst_type_arr[i] if inst_type_arr is not None else None,
            amount_arr[i] if amount_arr is not None else None,
        )
        if row is not None:
            rows.append(row)

    if not rows:
        return _empty_eval_joined(recommendations)

    eval_df = pd.DataFrame(rows)
    return recommendations.merge(eval_df, on="ticker", how="left")


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
    """Pre-compute SPY returns for every distinct horizon (avoid re-lookup)."""
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
    ticker, t_horizon, cached,
    as_of_date, entry_mult, exit_mult,
    spy_start, spy_returns, spy_ends,
    use_dip_entry, pullback_pct, max_wait_days,
    max_staleness_days,
    inst_type_val, amount_val,
) -> dict | None:
    """Returns dict of bt_* fields for one ticker, or None to skip."""
    if cached is None:
        return None
    idx_ns, vals = cached
    if idx_ns is None:
        return None

    entry, entry_delay = _resolve_entry(
        use_dip_entry, idx_ns, vals, as_of_date,
        pullback_pct, max_wait_days, max_staleness_days,
    )
    if not entry:
        return None

    as_of_ns = as_of_date.value
    exit_ns = as_of_ns + (entry_delay + t_horizon) * NS_PER_DAY
    exit_price = _price_on_or_before_arrays(idx_ns, vals, pd.Timestamp(exit_ns), max_staleness_days=30)
    if not exit_price:
        return None

    spy_ret = spy_returns.get(t_horizon, 0.0)
    if not spy_start or (spy_ret == 0.0 and not spy_ends.get(t_horizon)):
        # Spy lookup failed for this horizon — skip rec
        return None

    inst_type = str(inst_type_val) if inst_type_val is not None and pd.notna(inst_type_val) else "stock"
    amount = amount_val if amount_val is not None else None
    return _bt_row(
        ticker, entry, entry_delay, exit_price, t_horizon,
        entry_mult, exit_mult, spy_ret, inst_type, amount,
    )


def _resolve_entry(use_dip_entry, idx_ns, vals, as_of_date, pullback_pct, max_wait_days, max_staleness_days):
    """Compute entry price and entry delay, with optional dip timing."""
    if use_dip_entry:
        entry, entry_delay = _find_dip_entry_arrays(
            idx_ns, vals, as_of_date, pullback_pct, max_wait_days,
        )
        if entry <= 0:
            entry = _price_at_or_before_arrays(idx_ns, vals, as_of_date, max_staleness_days)
            entry_delay = 0
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
    }
