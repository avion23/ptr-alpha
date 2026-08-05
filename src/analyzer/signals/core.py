"""Core signal computation: vectorized per-ticker kernel + signal features.

The vectorized inner kernel `_compute_ticker_signals` mutates pre-allocated
result arrays for signals belonging to one ticker using numpy searchsorted
+ array ops (no pandas overhead). `calculate_signal_potential` is the main
public entry point that orchestrates the kernel across all tickers and
returns a tidy DataFrame of signals with metadata columns.

The data-oriented redesign replaces the merge-then-filter pattern
(75M+ intermediate rows → 3M filtered) with per-ticker searchsorted
lookups via the `_price_arrays` index. SPY log returns are pre-computed
once on the full Series instead of per-signal groupby shifts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analyzer.exceptions import AnalysisError
from analyzer.ticker_resolver import TickerResolver

from analyzer.signals import constants as _constants
from analyzer.signals.assembly import assemble_result_dataframe
from analyzer.signals.prices import _price_arrays


def _compute_ticker_signals(
    t_indices: np.ndarray,
    t_disc_ns: np.ndarray,
    t_end_ns: np.ndarray,
    dates_ns: np.ndarray,
    vals: np.ndarray,
    spy_dates_ns: np.ndarray | None,
    spy_vals: np.ndarray | None,
    spy_log_ret: np.ndarray | None,
    decay_lambda: float,
    r_peak: np.ndarray,
    r_trough: np.ndarray,
    r_decayed_ret: np.ndarray,
    r_disc_baseline: np.ndarray,
    r_last_price: np.ndarray,
    r_spy_cum: np.ndarray,
    r_spy_wsum: np.ndarray,
    r_spy_first: np.ndarray,
    r_spy_last: np.ndarray,
    r_window_complete: np.ndarray,
) -> None:
    """Mutate pre-allocated result arrays for signals belonging to one ticker."""
    n_signals = len(t_indices)
    if n_signals == 0:
        return

    # Vectorized searchsorted for all signals of this ticker at once
    t_lo = np.searchsorted(dates_ns, t_disc_ns, side="left")
    t_hi = np.searchsorted(dates_ns, t_end_ns, side="right")

    # A forward-return horizon is an outcome only after both the security and
    # benchmark datasets cover its end.  A seven-day tolerance accommodates
    # weekends/market holidays without treating a partially elapsed horizon
    # as a completed 30/90/180-day observation.
    coverage_tolerance_ns = 7 * _constants._NS_PER_DAY
    ticker_data_through = dates_ns[-1]
    has_benchmark = spy_dates_ns is not None and len(spy_dates_ns) > 0
    spy_data_through = spy_dates_ns[-1] if has_benchmark else 0
    today_ns = pd.Timestamp.now().normalize().value
    r_window_complete[t_indices] = (
        (t_end_ns <= today_ns)
        & (ticker_data_through >= t_end_ns - coverage_tolerance_ns)
        & ((not has_benchmark) | (spy_data_through >= t_end_ns - coverage_tolerance_ns))
    )

    spy_has = spy_dates_ns is not None and spy_vals is not None and spy_log_ret is not None

    for i in range(n_signals):
        # A forward-return label is valid only after the market benchmark has
        # reached the requested window end.  Without this guard, a 180-day
        # signal computed 20 days after disclosure was silently labelled with
        # a 20-day return, contaminating member rankings and validation.
        if not _market_window_is_complete(spy_dates_ns, t_end_ns[i]):
            continue
        lo = int(t_lo[i])
        hi = int(t_hi[i])
        if lo >= hi:
            continue

        idx = int(t_indices[i])
        disc = t_disc_ns[i]

        w_vals = vals[lo:hi]
        w_dates = dates_ns[lo:hi]
        n_w = len(w_vals)

        # Baseline, last, peak, trough
        r_disc_baseline[idx] = w_vals[0]
        r_last_price[idx] = w_vals[-1]
        r_peak[idx] = w_vals.max()
        r_trough[idx] = w_vals.min()

        # Days from disclosure (vectorized)
        days = ((w_dates - disc) // _constants._NS_PER_DAY).astype(np.float64)

        # Daily log returns (vectorized, handles zero prices)
        log_ret = np.zeros(n_w, dtype=np.float64)
        if n_w > 1:
            prev_vals = w_vals[:-1]
            valid = prev_vals > 0
            log_ret[1:][valid] = np.log(w_vals[1:][valid] / prev_vals[valid])

        # Mid-decay: weight by midpoint between consecutive days
        prev_days = np.empty(n_w, dtype=np.float64)
        prev_days[0] = 0.0
        prev_days[1:] = days[:-1]
        mid_d = np.exp(-decay_lambda * (days + prev_days) * 0.5)

        # Decay-weighted return normalized by total decay weight
        w_ret = log_ret * mid_d
        w_sum = mid_d.sum()
        if w_sum > 0:
            r_decayed_ret[idx] = w_ret.sum() / w_sum

        # SPY returns for the same window
        if spy_has:
            _populate_spy_arrays(
                idx, disc, t_end_ns[i], i,
                spy_dates_ns, spy_vals, spy_log_ret, decay_lambda,
                r_spy_cum, r_spy_wsum, r_spy_first, r_spy_last,
            )


def _market_window_is_complete(
    spy_dates_ns: np.ndarray | None,
    end_ns: int,
    max_staleness_days: int = 7,
) -> bool:
    """Return whether benchmark data reaches a signal's intended end date."""
    # Legacy/library callers may calculate absolute returns without a SPY
    # column.  Benchmark-relative metrics remain NaN there; maturity is
    # enforced when the production benchmark is present.
    if spy_dates_ns is None or len(spy_dates_ns) == 0:
        return True
    pos = int(np.searchsorted(spy_dates_ns, end_ns, side="right")) - 1
    if pos < 0:
        return False
    return int(end_ns) - int(spy_dates_ns[pos]) <= max_staleness_days * _constants._NS_PER_DAY


def _populate_spy_arrays(
    idx, disc, end_ns, i,
    spy_dates_ns, spy_vals, spy_log_ret, decay_lambda,
    r_spy_cum, r_spy_wsum, r_spy_first, r_spy_last,
) -> None:
    spy_lo = int(np.searchsorted(spy_dates_ns, disc, side="left"))
    spy_hi_end = int(np.searchsorted(spy_dates_ns, end_ns, side="right"))
    if spy_lo >= spy_hi_end:
        return

    sw_vals = spy_vals[spy_lo:spy_hi_end]
    sw_dates = spy_dates_ns[spy_lo:spy_hi_end]
    n_sw = len(sw_vals)

    r_spy_first[idx] = sw_vals[0]
    r_spy_last[idx] = sw_vals[-1]

    # Reuse pre-computed SPY log returns via index mapping
    spy_lr = np.zeros(n_sw, dtype=np.float64)
    if n_sw > 1:
        # Map window positions back to full SPY array indices
        spy_full_lo = spy_lo
        spy_lr[1:] = spy_log_ret[spy_full_lo + 1 : spy_full_lo + n_sw]

    s_days = ((sw_dates - disc) // _constants._NS_PER_DAY).astype(np.float64)
    s_prev = np.empty(n_sw, dtype=np.float64)
    s_prev[0] = 0.0
    s_prev[1:] = s_days[:-1]
    s_mid = np.exp(-decay_lambda * (s_days + s_prev) * 0.5)

    s_wr = spy_lr * s_mid
    s_ws = s_mid.sum()
    if s_ws > 0:
        r_spy_cum[idx] = s_wr.sum() / s_ws
        r_spy_wsum[idx] = s_ws


def calculate_signal_potential(
    entry_prices_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    horizons: list[int] | None = None,
    decay_lambda: float | None = None,
) -> pd.DataFrame:
    """Compute per-(transaction, horizon) signal potential metrics.

    Output columns:
      - peak_potential_pct   max upside from disclosure price
      - decayed_return_pct   midpoint-weighted log return
      - spy_alpha_pct       decayed_return - decayed_spy_return (× 100)
      - total_return_pct    last_price / disc_price - 1 (× 100)
      - total_spy_alpha_pct total_return - actual_spy_return (× 100)
      - decayed_spy_return_pct  SPY return on the same window
    """
    # Resolve at call time so a changed application default is observed. Sweeps
    # pass this argument explicitly and do not mutate shared module state.
    if horizons is None:
        horizons = [30, 60, 90, 180]
    if decay_lambda is None:
        decay_lambda = _constants.DECAY_LAMBDA
    _validate_inputs(entry_prices_df, prices_df)

    signals = entry_prices_df.copy()
    signals = _resolve_tickers(signals, prices_df)

    signals = _explode_by_horizon(signals, horizons)
    n = len(signals)

    metadata = _extract_metadata_arrays(signals)
    spy_dates_ns, spy_vals, spy_log_ret = _precompute_spy_log_returns(prices_df)
    result_arrays = _allocate_result_arrays(n)

    _compute_all_ticker_signals(
        signals, metadata, prices_df, decay_lambda,
        spy_dates_ns, spy_vals, spy_log_ret,
        result_arrays,
    )

    return _assemble_result_dataframe(signals, metadata, result_arrays)


# ── Pipeline helpers (private) ──────────────────────────────────────────

def _validate_inputs(entry_prices_df: pd.DataFrame, prices_df: pd.DataFrame) -> None:
    if entry_prices_df.empty:
        raise AnalysisError("Empty entry prices dataframe")
    if prices_df.empty:
        raise AnalysisError("Empty prices dataframe")

    required_cols = {"member", "ticker", "disclosure_date", "transaction_type", "entry_price"}
    if not required_cols.issubset(entry_prices_df.columns):
        raise AnalysisError(
            f"Missing columns in entry_prices: {required_cols - set(entry_prices_df.columns)}"
        )


def _resolve_tickers(signals: pd.DataFrame, prices_df: pd.DataFrame) -> pd.DataFrame:
    resolver = TickerResolver()
    price_tickers = set(prices_df.columns)
    raw_tickers = signals["ticker"].unique()
    for raw in raw_tickers:
        if raw not in price_tickers and raw != "SPY":
            resolved = resolver.resolve(raw)
            if resolved.price_symbol in price_tickers:
                signals.loc[signals["ticker"] == raw, "ticker"] = resolved.price_symbol
    if signals.empty:
        raise AnalysisError("No valid price matches found for transactions")
    return signals


def _explode_by_horizon(signals: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    signals = signals.assign(horizon_days=[horizons] * len(signals)).explode("horizon_days").reset_index(drop=True)
    signals["horizon_days"] = signals["horizon_days"].astype("int32")
    signals["window_end"] = signals["disclosure_date"] + pd.to_timedelta(signals["horizon_days"], unit="D")
    signals["signal_id"] = range(len(signals))
    return signals


def _extract_metadata_arrays(signals: pd.DataFrame) -> dict:
    """Convert text/timestamp/price columns to numpy arrays for the inner kernel.

    Dates are normalized to nanoseconds explicitly because pandas 3+ may store
    datetime64[us] internally, and `asi8` returns storage units (microseconds).
    `_price_arrays` already converts to ns via `as_unit("ns")`, so we must match.
    """
    disc_ns = pd.DatetimeIndex(signals["disclosure_date"]).as_unit("ns").asi8
    end_ns = pd.DatetimeIndex(signals["window_end"]).as_unit("ns").asi8
    # Object arrays avoid pandas StringDtype.__eq__ overhead
    return {
        "disc_ns": disc_ns,
        "end_ns": end_ns,
        "ticker_arr": signals["ticker"].to_numpy(dtype=object, na_value=None),
        "entry_prices_arr": signals["entry_price"].values,
        "txn_types": signals["transaction_type"].to_numpy(dtype=object, na_value=None),
        "horizon_days_arr": signals["horizon_days"].values,
    }


def _precompute_spy_log_returns(prices_df: pd.DataFrame) -> tuple:
    """Pre-compute SPY data once (vectorized log returns)."""
    spy_arrs = _price_arrays(prices_df, "SPY")
    spy_dates_ns = None
    spy_vals = None
    spy_log_ret = None
    if spy_arrs is not None and spy_arrs[0] is not None:
        spy_dates_ns, spy_vals = spy_arrs
        spy_log_ret = np.zeros(len(spy_vals), dtype=np.float64)
        if len(spy_vals) > 1:
            valid_spy = spy_vals[:-1] > 0
            spy_log_ret[1:][valid_spy] = np.log(spy_vals[1:][valid_spy] / spy_vals[:-1][valid_spy])
    return spy_dates_ns, spy_vals, spy_log_ret


def _allocate_result_arrays(n: int) -> dict:
    return {
        "r_peak": np.full(n, np.nan, dtype=np.float64),
        "r_trough": np.full(n, np.nan, dtype=np.float64),
        "r_decayed_ret": np.full(n, np.nan, dtype=np.float64),
        "r_disc_baseline": np.full(n, np.nan, dtype=np.float64),
        "r_last_price": np.full(n, np.nan, dtype=np.float64),
        "r_spy_cum": np.zeros(n, dtype=np.float64),
        "r_spy_wsum": np.zeros(n, dtype=np.float64),
        "r_spy_first": np.full(n, np.nan, dtype=np.float64),
        "r_spy_last": np.full(n, np.nan, dtype=np.float64),
        "r_window_complete": np.zeros(n, dtype=bool),
    }


def _compute_all_ticker_signals(
    signals, metadata, prices_df, decay_lambda,
    spy_dates_ns, spy_vals, spy_log_ret, result_arrays,
) -> None:
    """Process per-ticker (avoids 75M+ row merge)."""
    unique_tickers = np.unique(metadata["ticker_arr"])
    for ticker in unique_tickers:
        if ticker == "SPY":
            continue
        arrs = _price_arrays(prices_df, str(ticker))
        if arrs is None or arrs[0] is None:
            continue
        dates_ns, vals = arrs

        tmask = metadata["ticker_arr"] == ticker
        t_indices = np.where(tmask)[0]
        if len(t_indices) == 0:
            continue

        _compute_ticker_signals(
            t_indices,
            metadata["disc_ns"][t_indices],
            metadata["end_ns"][t_indices],
            dates_ns,
            vals,
            spy_dates_ns,
            spy_vals,
            spy_log_ret,
            decay_lambda,
            result_arrays["r_peak"],
            result_arrays["r_trough"],
            result_arrays["r_decayed_ret"],
            result_arrays["r_disc_baseline"],
            result_arrays["r_last_price"],
            result_arrays["r_spy_cum"],
            result_arrays["r_spy_wsum"],
            result_arrays["r_spy_first"],
            result_arrays["r_spy_last"],
            result_arrays["r_window_complete"],
        )


def _assemble_result_dataframe(signals: pd.DataFrame, metadata: dict, result_arrays: dict) -> pd.DataFrame:
    """Backward-compat thin wrapper that delegates to ``assembly.py``."""
    return assemble_result_dataframe(signals, metadata, result_arrays)


def compute_signal_potential_with_member_decay(
    entry_prices_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    horizons: list[int] | None = None,
    member_decay_map: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Compute signal potential with per-member decay rates.

    If member_decay_map is provided, each member's trades use their
    personal decay lambda instead of the global default.
    """
    if horizons is None:
        horizons = [30, 60, 90, 180]
    if member_decay_map is None or not member_decay_map:
        return calculate_signal_potential(entry_prices_df, prices_df, horizons)

    all_members = entry_prices_df["member"].unique()
    default_members = [m for m in all_members if m not in member_decay_map]
    custom_members = [m for m in all_members if m in member_decay_map]

    results = []
    if default_members:
        default_df = entry_prices_df[entry_prices_df["member"].isin(default_members)]
        if not default_df.empty:
            results.append(calculate_signal_potential(default_df, prices_df, horizons))

    for member in custom_members:
        member_df = entry_prices_df[entry_prices_df["member"] == member]
        if member_df.empty:
            continue
        member_lambda = member_decay_map[member]
        member_signals = calculate_signal_potential(
            member_df, prices_df, horizons, decay_lambda=member_lambda,
        )
        results.append(member_signals)

    if not results:
        return pd.DataFrame()

    return pd.concat(results, ignore_index=True)
