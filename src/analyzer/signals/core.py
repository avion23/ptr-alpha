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

import logging

import numpy as np
import pandas as pd

from analyzer.exceptions import AnalysisError
from analyzer.price_repository import next_nyse_session, previous_nyse_session
from analyzer.ticker_resolver import TickerResolver

from analyzer.signals import constants as _constants
from analyzer.signals.assembly import assemble_result_dataframe
from analyzer.signals.prices import _price_arrays

logger = logging.getLogger(__name__)


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
    r_entry_date: np.ndarray,
    r_exit_date: np.ndarray,
    r_label_window_end: np.ndarray,
) -> None:
    """Populate labels only for mature, endpoint-aligned price windows."""
    if len(t_indices) == 0:
        return

    today_ns = pd.Timestamp.now().normalize().value
    has_benchmark = (
        spy_dates_ns is not None
        and spy_vals is not None
        and spy_log_ret is not None
        and len(spy_dates_ns) > 0
    )

    for i, result_idx in enumerate(t_indices):
        disc_ns = int(t_disc_ns[i])
        horizon_ns = int(t_end_ns[i]) - disc_ns
        idx = int(result_idx)

        # The execution convention is explicit: enter on the next expected
        # NYSE session after the dated decision, then hold for the full calendar
        # horizon and exit on the expected NYSE session on or before that end.
        entry_date_ns = next_nyse_session(pd.Timestamp(disc_ns)).value
        entry_pos = int(np.searchsorted(dates_ns, entry_date_ns, side="left"))
        if entry_pos >= len(dates_ns) or int(dates_ns[entry_pos]) != entry_date_ns:
            continue
        r_entry_date[idx] = np.datetime64(entry_date_ns, "ns")

        intended_end_ns = entry_date_ns + horizon_ns
        r_label_window_end[idx] = np.datetime64(intended_end_ns, "ns")
        if intended_end_ns > today_ns:
            continue
        exit_date_ns = previous_nyse_session(pd.Timestamp(intended_end_ns)).value
        exit_pos = int(np.searchsorted(dates_ns, exit_date_ns, side="left"))
        if exit_pos >= len(dates_ns) or int(dates_ns[exit_pos]) != exit_date_ns:
            continue
        r_exit_date[idx] = np.datetime64(exit_date_ns, "ns")

        # Benchmark prices must exist on the security's actual entry and exit
        # sessions. Independent nearest-date lookups create phantom alpha.
        if has_benchmark:
            spy_entry_pos = int(np.searchsorted(spy_dates_ns, entry_date_ns, side="left"))
            spy_exit_pos = int(np.searchsorted(spy_dates_ns, exit_date_ns, side="left"))
            if (
                spy_entry_pos >= len(spy_dates_ns)
                or int(spy_dates_ns[spy_entry_pos]) != entry_date_ns
                or spy_exit_pos >= len(spy_dates_ns)
                or int(spy_dates_ns[spy_exit_pos]) != exit_date_ns
            ):
                continue

        r_window_complete[idx] = True
        w_vals = vals[entry_pos : exit_pos + 1]
        w_dates = dates_ns[entry_pos : exit_pos + 1]
        n_w = len(w_vals)

        r_disc_baseline[idx] = w_vals[0]
        r_last_price[idx] = w_vals[-1]
        r_peak[idx] = w_vals.max()
        r_trough[idx] = w_vals.min()

        days = ((w_dates - entry_date_ns) // _constants._NS_PER_DAY).astype(
            np.float64
        )
        log_ret = np.zeros(n_w, dtype=np.float64)
        if n_w > 1:
            log_ret[1:] = np.log(w_vals[1:] / w_vals[:-1])

        prev_days = np.empty(n_w, dtype=np.float64)
        prev_days[0] = 0.0
        prev_days[1:] = days[:-1]
        mid_d = np.exp(-decay_lambda * (days + prev_days) * 0.5)
        w_sum = mid_d.sum()
        if w_sum > 0:
            r_decayed_ret[idx] = (log_ret * mid_d).sum() / w_sum

        if has_benchmark:
            _populate_spy_arrays(
                idx,
                entry_date_ns,
                exit_date_ns,
                spy_dates_ns,
                spy_vals,
                spy_log_ret,
                decay_lambda,
                r_spy_cum,
                r_spy_wsum,
                r_spy_first,
                r_spy_last,
            )


def _populate_spy_arrays(
    idx,
    disc,
    end_ns,
    spy_dates_ns,
    spy_vals,
    spy_log_ret,
    decay_lambda,
    r_spy_cum,
    r_spy_wsum,
    r_spy_first,
    r_spy_last,
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
    prices_df = _normalize_price_frame(prices_df)

    signals = entry_prices_df.copy()
    signals = _resolve_tickers(signals, prices_df)

    signals = _explode_by_horizon(signals, horizons)
    n = len(signals)

    metadata = _extract_metadata_arrays(signals)
    spy_dates_ns, spy_vals, spy_log_ret = _precompute_spy_log_returns(prices_df)
    result_arrays = _allocate_result_arrays(n)

    _compute_all_ticker_signals(
        signals,
        metadata,
        prices_df,
        decay_lambda,
        spy_dates_ns,
        spy_vals,
        spy_log_ret,
        result_arrays,
    )

    return _assemble_result_dataframe(signals, metadata, result_arrays)


# ── Pipeline helpers (private) ──────────────────────────────────────────


def _normalize_price_frame(prices_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize daily price indexes without shifting their calendar dates."""
    try:
        index = pd.DatetimeIndex(pd.to_datetime(prices_df.index))
    except (TypeError, ValueError) as exc:
        raise AnalysisError("Price index must contain valid dates") from exc
    if index.tz is not None:
        index = index.tz_localize(None)
    index = index.normalize()
    if index.has_duplicates:
        raise AnalysisError("Price index contains duplicate calendar dates")
    normalized = prices_df.copy()
    normalized.index = index
    return normalized.sort_index()


def _validate_inputs(entry_prices_df: pd.DataFrame, prices_df: pd.DataFrame) -> None:
    if entry_prices_df.empty:
        raise AnalysisError("Empty entry prices dataframe")
    if prices_df.empty:
        raise AnalysisError("Empty prices dataframe")

    required_cols = {
        "member",
        "ticker",
        "disclosure_date",
        "transaction_type",
        "entry_price",
    }
    if not required_cols.issubset(entry_prices_df.columns):
        raise AnalysisError(
            f"Missing columns in entry_prices: {required_cols - set(entry_prices_df.columns)}"
        )


def _resolve_tickers(signals: pd.DataFrame, prices_df: pd.DataFrame) -> pd.DataFrame:
    """Resolve each row's raw ticker to its contemporaneous price symbol.

    Rename aliases (FB -> META, SQ -> XYZ, BLL -> BALL) map to different price
    series depending on the transaction date, so resolution is per-row using
    the transaction_date carried by entry prices. A rename alias without a
    transaction date fails explicit unverified and is dropped rather than
    silently priced under the raw symbol.
    """
    resolver = TickerResolver()
    rename_aliases = resolver.RENAME_MAP
    price_tickers = set(prices_df.columns)
    has_tx_date = "transaction_date" in signals.columns
    tx_dates = signals["transaction_date"] if has_tx_date else None

    keep_indices: list[int] = []
    resolved: list[object] = []
    for i in range(len(signals)):
        raw = signals["ticker"].iloc[i]
        if raw is None or pd.isna(raw):
            keep_indices.append(i)
            resolved.append(raw)
            continue
        normalized = str(raw).strip().upper()
        is_alias = normalized in rename_aliases
        if not is_alias and raw in price_tickers:
            # Prices stored under the raw ticker (class shares, pass-throughs)
            # are authoritative; the resolved symbol would miss the column.
            keep_indices.append(i)
            resolved.append(raw)
            continue
        tx_date = None
        if tx_dates is not None:
            candidate = tx_dates.iloc[i]
            if candidate is not None and not pd.isna(candidate):
                tx_date = candidate
        resolution = resolver.resolve(raw, tx_date)
        if is_alias and resolution.status == "unverified":
            logger.warning(
                "Dropping %s transaction without entry-price resolution: "
                "rename alias requires a transaction date (unverified without it)",
                raw,
            )
            continue
        keep_indices.append(i)
        resolved.append(resolution.price_symbol)

    signals = signals.iloc[keep_indices].copy()
    signals["ticker"] = resolved
    if signals.empty:
        raise AnalysisError("No valid price matches found for transactions")
    return signals


def _explode_by_horizon(signals: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    disclosure_dates = pd.DatetimeIndex(pd.to_datetime(signals["disclosure_date"]))
    if disclosure_dates.tz is not None:
        disclosure_dates = disclosure_dates.tz_localize(None)
    signals["disclosure_date"] = disclosure_dates.normalize()
    signals = (
        signals.assign(horizon_days=[horizons] * len(signals))
        .explode("horizon_days")
        .reset_index(drop=True)
    )
    signals["horizon_days"] = signals["horizon_days"].astype("int32")
    signals["window_end"] = signals["disclosure_date"] + pd.to_timedelta(
        signals["horizon_days"], unit="D"
    )
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
            spy_log_ret[1:][valid_spy] = np.log(
                spy_vals[1:][valid_spy] / spy_vals[:-1][valid_spy]
            )
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
        "r_entry_date": np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]"),
        "r_exit_date": np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]"),
        "r_label_window_end": np.full(
            n, np.datetime64("NaT"), dtype="datetime64[ns]"
        ),
    }


def _compute_all_ticker_signals(
    signals,
    metadata,
    prices_df,
    decay_lambda,
    spy_dates_ns,
    spy_vals,
    spy_log_ret,
    result_arrays,
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
            result_arrays["r_entry_date"],
            result_arrays["r_exit_date"],
            result_arrays["r_label_window_end"],
        )


def _assemble_result_dataframe(
    signals: pd.DataFrame, metadata: dict, result_arrays: dict
) -> pd.DataFrame:
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
            member_df,
            prices_df,
            horizons,
            decay_lambda=member_lambda,
        )
        results.append(member_signals)

    if not results:
        return pd.DataFrame()

    return pd.concat(results, ignore_index=True)
