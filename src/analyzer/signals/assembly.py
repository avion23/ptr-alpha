"""Result DataFrame assembly for calculate_signal_potential.

Takes the pre-computed numpy result arrays and metadata arrays, builds
the final column-wise DataFrame with optional columns, and applies the
percentage scaling (× 100) that downstream consumers expect.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


_OPTIONAL_COLS = ("owner_code", "amount_midpoint")

_FINAL_COLUMNS = [
    "member", "ticker", "disclosure_date", "signal_type", "horizon_days", "entry_price",
    "peak_potential_pct", "decayed_return_pct", "spy_alpha_pct", "total_return_pct",
    "total_spy_alpha_pct", "decayed_spy_return_pct",
]


def assemble_result_dataframe(signals: pd.DataFrame, metadata: dict, result_arrays: dict) -> pd.DataFrame:
    """Combine pre-computed arrays + metadata into the final signal DataFrame."""
    derived = _compute_derived_arrays(metadata, result_arrays)
    peak_potential = _compute_peak_potential(metadata, result_arrays)

    optional_columns = [col for col in _OPTIONAL_COLS if col in signals.columns]
    result_columns = _FINAL_COLUMNS + list(optional_columns)

    result_data = _build_result_data(
        signals, metadata, derived, peak_potential, optional_columns, result_arrays,
    )
    return pd.DataFrame(result_data)[result_columns]


def _compute_derived_arrays(metadata: dict, result_arrays: dict) -> dict:
    """Compute total_return, actual_spy_return, and decay-weighted SPY cum."""
    n = len(metadata["disc_ns"])
    r_disc_baseline = result_arrays["r_disc_baseline"]
    r_last_price = result_arrays["r_last_price"]
    r_spy_first = result_arrays["r_spy_first"]
    r_spy_last = result_arrays["r_spy_last"]
    r_spy_cum = result_arrays["r_spy_cum"]
    r_spy_wsum = result_arrays["r_spy_wsum"]

    valid_disc = (r_disc_baseline > 0) & np.isfinite(r_disc_baseline)
    # Bug #6: use NaN (not 0.0) for missing price windows so downstream
    # NaN-exclusion in aggregations (dynamic prior, hit rates) handles them.
    total_return = np.full(n, np.nan, dtype=np.float64)
    total_return[valid_disc] = r_last_price[valid_disc] / r_disc_baseline[valid_disc] - 1

    valid_spy = (r_spy_first > 0) & np.isfinite(r_spy_first)
    actual_spy_return = np.full(n, np.nan, dtype=np.float64)
    actual_spy_return[valid_spy] = r_spy_last[valid_spy] / r_spy_first[valid_spy] - 1

    # Bug #1: _populate_spy_arrays already stores the decay-weighted mean
    # (s_wr.sum() / s_ws) in r_spy_cum.  Dividing again by r_spy_wsum is a
    # double-division bug.  Use r_spy_cum directly.
    # Bug #6: missing SPY windows produce NaN instead of 0.0.
    spy_cum = np.where(r_spy_wsum > 0, r_spy_cum, np.nan)

    return {
        "total_return": total_return,
        "actual_spy_return": actual_spy_return,
        "spy_cum": spy_cum,
    }


def _compute_peak_potential(metadata: dict, result_arrays: dict) -> np.ndarray:
    """Peak potential % = (peak/trough - entry) * 100, per transaction type.

    Purchases use disclosure price as baseline and max price as upside.
    Sales use entry price as baseline and min trough as upside.
    """
    n = len(metadata["disc_ns"])
    r_disc_baseline = result_arrays["r_disc_baseline"]
    r_peak = result_arrays["r_peak"]
    r_trough = result_arrays["r_trough"]
    entry_prices_arr = metadata["entry_prices_arr"]
    txn_types = metadata["txn_types"]

    from analyzer.models import TransactionType
    valid_disc = (r_disc_baseline > 0) & np.isfinite(r_disc_baseline)
    is_purchase = txn_types == TransactionType.PURCHASE.value
    is_sale = txn_types == TransactionType.SALE.value
    purchase_mask = is_purchase & valid_disc
    sale_mask = is_sale & (r_trough > 0) & np.isfinite(r_trough)

    # Missing and not-yet-mature windows are unknown, not zero-return trades.
    peak_potential = np.full(n, np.nan, dtype=np.float64)
    peak_potential[purchase_mask] = (r_peak[purchase_mask] / r_disc_baseline[purchase_mask] - 1) * 100
    peak_potential[sale_mask] = (entry_prices_arr[sale_mask] / r_trough[sale_mask] - 1) * 100
    return peak_potential


def _build_result_data(
    signals: pd.DataFrame,
    metadata: dict,
    derived: dict,
    peak_potential: np.ndarray,
    optional_columns: list[str],
    result_arrays: dict,
) -> dict:
    r_decayed_ret = result_arrays["r_decayed_ret"]
    result_data = {
        "member": signals["member"].values,
        "ticker": metadata["ticker_arr"],
        "disclosure_date": signals["disclosure_date"].values,
        "signal_type": metadata["txn_types"],
        "horizon_days": metadata["horizon_days_arr"],
        "entry_price": metadata["entry_prices_arr"],
        "peak_potential_pct": peak_potential,
        "decayed_return_pct": r_decayed_ret * 100,
        "spy_alpha_pct": (r_decayed_ret - derived["spy_cum"]) * 100,
        "total_return_pct": derived["total_return"] * 100,
        "total_spy_alpha_pct": (
            derived["total_return"] - derived["actual_spy_return"]
        ) * 100,
        "decayed_spy_return_pct": derived["spy_cum"] * 100,
    }
    for col in optional_columns:
        result_data[col] = signals[col].values
    return result_data
