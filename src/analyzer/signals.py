"""Signal generation and calculation.

Data-oriented redesign: replaces the merge-then-filter pattern
(75M+ intermediate rows → 3M filtered) with per-ticker searchsorted
lookups via the _price_arrays index. Pre-computes SPY log returns
once on the full Series instead of per-signal groupby shifts.
"""

from __future__ import annotations

import weakref as _weakref

import numpy as np
import pandas as pd

from analyzer.exceptions import AnalysisError
from analyzer.models import TransactionType
from analyzer.ticker_resolver import TickerResolver

DECAY_LAMBDA = 0.005
POSITION_SIZE_BASELINE = 10000.0
MAX_DISCLOSURE_METADATA_ADJUSTMENT = 0.15
BAYES_PRIOR_STRENGTH = 20.0
BUYER_RECENCY_DECAY = 0.03
TICKER_PERF_MIN_TRADES = 3

MIN_ENTRY_PRICE = 3.0
# Use pure SPY alpha as signal score — avoids double-counting stock return
CONVICTION_WEIGHT_ALPHA = 1.0
CONVICTION_WEIGHT_REALIZED = 0.0

_NS_PER_DAY = 86_400_000_000_000  # nanoseconds in a day


# Module-level per-prices_df index cache. Keyed by id(prices_df); cleaned up
# via a weakref finalizer when the DataFrame is garbage-collected, so the
# cache never returns data for an id that was reused by an unrelated DataFrame.
# Stores per-ticker (sorted-non-NaN dates as int64 ns, values as np.ndarray)
# tuples so price lookups become O(log N) via searchsorted instead of O(N) via
# boolean masking + dropna.

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
    prices_df: pd.DataFrame, ticker: str, target_date: pd.Timestamp,
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
    prices_df: pd.DataFrame, ticker: str, target_date: pd.Timestamp,
    max_staleness_days: int = 5,
) -> float | None:
    return _price_at_or_before(
        prices_df, ticker, target_date, max_staleness_days=max_staleness_days,
    )


def _get_horizon_data(
    signals_df: pd.DataFrame, horizon: int, transaction_type: str | None = None
) -> pd.DataFrame:
    mask = signals_df["horizon_days"] == horizon
    if transaction_type is not None:
        mask = mask & (signals_df["signal_type"] == transaction_type)
    return signals_df.loc[mask]


def _apply_quality_filter(signals_df: pd.DataFrame) -> pd.DataFrame:
    if "entry_price" not in signals_df.columns:
        return signals_df
    return signals_df[signals_df["entry_price"] >= MIN_ENTRY_PRICE]


def _compute_dynamic_prior(signals_df: pd.DataFrame, horizon: int) -> float:
    horizon_signals = _get_horizon_data(signals_df, horizon, TransactionType.PURCHASE.value)
    if horizon_signals.empty:
        return 0.50
    up_prob = (horizon_signals["decayed_return_pct"] > 0).mean()
    return float(np.clip(up_prob, 0.10, 0.90))


def _assign_episode_ids(group_sorted: pd.DataFrame, max_gap_days: int) -> np.ndarray:
    dates = pd.to_datetime(group_sorted["disclosure_date"])
    if len(dates) <= 1:
        return np.zeros(len(dates), dtype=np.int64)
    gaps = dates.diff().dt.days.fillna(0).astype(int)
    return (gaps > max_gap_days).cumsum().values.astype(np.int64)


def _collapse_to_episodes(signals_df: pd.DataFrame, max_gap_days: int = 14) -> pd.DataFrame:
    if signals_df.empty:
        return signals_df

    group_cols = ["member", "ticker", "horizon_days", "signal_type"]
    if not all(c in signals_df.columns for c in group_cols):
        return signals_df

    if "disclosure_date" not in signals_df.columns:
        return signals_df

    df = signals_df.sort_values(group_cols + ["disclosure_date"]).reset_index(drop=True)

    dates = pd.to_datetime(df["disclosure_date"])
    gaps = dates.diff().dt.days.fillna(0).astype(np.int64)
    first_per_group = df.groupby(group_cols, sort=False).head(1).index
    gaps.loc[first_per_group] = 0
    df["_episode_id"] = (gaps > max_gap_days).cumsum().astype(np.int64)

    if "amount_midpoint" in df.columns:
        df["_weight"] = df["amount_midpoint"].fillna(1.0)
    else:
        df["_weight"] = 1.0

    avg_cols = ["decayed_return_pct", "spy_alpha_pct", "total_return_pct", "total_spy_alpha_pct", "peak_potential_pct"]
    existing_avg_cols = [c for c in avg_cols if c in df.columns]

    for col in existing_avg_cols:
        non_nan = df[col].notna()
        df[f"_wp_{col}"] = np.where(non_nan, df[col] * df["_weight"], 0.0)
        df[f"_ws_{col}"] = np.where(non_nan, df["_weight"], 0.0)

    episode_key = group_cols + ["_episode_id"]

    agg_dict = {
        "episode_count": ("_weight", "count"),
        "_weight_sum": ("_weight", "sum"),
    }

    for col, func in {"disclosure_date": "min", "entry_price": "first", "amount_midpoint": "sum"}.items():
        if col in df.columns:
            agg_dict[col] = (col, func)

    if "owner_code" in df.columns:
        # "first" instead of mode — O(N log N) per-group mode dominates cost.
        # Within episodes (same member/ticker, ≤14d gap), owner_code is
        # effectively constant, so first() is equivalent.
        agg_dict["owner_code"] = ("owner_code", "first")

    for col in existing_avg_cols:
        agg_dict[col] = (f"_wp_{col}", "sum")
        agg_dict[f"_ws_{col}"] = (f"_ws_{col}", "sum")

    collapsed = df.groupby(episode_key, sort=False).agg(**agg_dict).reset_index()

    for col in existing_avg_cols:
        ws = collapsed[f"_ws_{col}"]
        collapsed[col] = np.where(ws > 0, collapsed[col] / ws, np.nan)
        collapsed = collapsed.drop(columns=[f"_ws_{col}"])

    collapsed = collapsed.drop(columns=["_weight_sum", "_episode_id"])

    orig_cols = [c for c in signals_df.columns if c in collapsed.columns]
    collapsed = collapsed[orig_cols + ["episode_count"]]

    return collapsed


def _get_top_signals(signals_df: pd.DataFrame, horizon: int = 90, top_n: int = 15) -> pd.DataFrame:
    top_data = _get_horizon_data(signals_df, horizon, TransactionType.PURCHASE.value)
    if top_data.empty:
        raise AnalysisError(f"No purchase signals found for horizon {horizon}")

    top_data = _apply_quality_filter(top_data)
    if top_data.empty:
        raise AnalysisError(f"No signals survived quality filter (min price ${MIN_ENTRY_PRICE})")

    top_data = top_data.copy()
    top_data["signal_score"] = (
        top_data["total_spy_alpha_pct"].fillna(0) * CONVICTION_WEIGHT_ALPHA
        + top_data["total_return_pct"].fillna(0) * CONVICTION_WEIGHT_REALIZED
    )

    return top_data.nlargest(top_n, "signal_score")[
        ["member", "ticker", "disclosure_date", "spy_alpha_pct", "peak_potential_pct",
         "total_return_pct", "total_spy_alpha_pct", "signal_score"]
    ]


def _get_member_signals(
    signals_df: pd.DataFrame, member: str, horizon: int = 90, top_n: int = 5
) -> pd.DataFrame:
    member_data = _get_horizon_data(signals_df, horizon)
    member_data = member_data[member_data["member"] == member]

    if member_data.empty:
        raise AnalysisError(f"No signals found for member {member} at horizon {horizon}")

    purchases = member_data[member_data["signal_type"] == TransactionType.PURCHASE.value]
    if purchases.empty:
        raise AnalysisError(f"No purchase signals for member {member} at horizon {horizon}")

    purchases = _apply_quality_filter(purchases)
    if purchases.empty:
        raise AnalysisError(f"No signals survived quality filter for {member}")

    purchases = purchases.copy()
    purchases["signal_score"] = (
        purchases["total_spy_alpha_pct"].fillna(0) * CONVICTION_WEIGHT_ALPHA
        + purchases["total_return_pct"].fillna(0) * CONVICTION_WEIGHT_REALIZED
    )

    return purchases.nlargest(top_n, "signal_score")[
        ["ticker", "disclosure_date", "spy_alpha_pct", "peak_potential_pct", "total_return_pct", "total_spy_alpha_pct", "signal_score"]
    ]


# ---------------------------------------------------------------------------
# Vectorized inner kernel: compute signal metrics for a single ticker's
# price window using numpy searchsorted + array ops (no pandas overhead).
# ---------------------------------------------------------------------------

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
) -> None:
    """Mutate pre-allocated result arrays for signals belonging to one ticker."""
    n_signals = len(t_indices)
    if n_signals == 0:
        return

    # Vectorized searchsorted for all signals of this ticker at once
    t_lo = np.searchsorted(dates_ns, t_disc_ns, side="left")
    t_hi = np.searchsorted(dates_ns, t_end_ns, side="right")

    spy_has = spy_dates_ns is not None and spy_vals is not None and spy_log_ret is not None

    for i in range(n_signals):
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
        days = ((w_dates - disc) // _NS_PER_DAY).astype(np.float64)

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
            spy_lo = int(np.searchsorted(spy_dates_ns, disc, side="left"))
            spy_hi_end = int(np.searchsorted(spy_dates_ns, t_end_ns[i], side="right"))
            if spy_lo < spy_hi_end:
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

                s_days = ((sw_dates - disc) // _NS_PER_DAY).astype(np.float64)
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
    horizons: list[int] = [30, 60, 90, 180],
    decay_lambda: float | None = None,
) -> pd.DataFrame:
    # Resolve at call time so callers that mutate the module global
    # (e.g. the parameter sweep) see the updated value. A default-arg
    # would freeze the value at function-definition time.
    if decay_lambda is None:
        decay_lambda = DECAY_LAMBDA
    if entry_prices_df.empty:
        raise AnalysisError("Empty entry prices dataframe")
    if prices_df.empty:
        raise AnalysisError("Empty prices dataframe")

    required_cols = {"member", "ticker", "disclosure_date", "transaction_type", "entry_price"}
    if not required_cols.issubset(entry_prices_df.columns):
        raise AnalysisError(f"Missing columns in entry_prices: {required_cols - set(entry_prices_df.columns)}")

    signals = entry_prices_df.copy()

    # Resolve tickers so lookups match prices_df columns
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

    # Explode across horizons
    signals = signals.assign(horizon_days=[horizons] * len(signals)).explode("horizon_days").reset_index(drop=True)
    signals["horizon_days"] = signals["horizon_days"].astype("int32")
    signals["window_end"] = signals["disclosure_date"] + pd.to_timedelta(signals["horizon_days"], unit="D")
    signals["signal_id"] = range(len(signals))

    n = len(signals)

    # --- Vectorized signal metadata (numpy arrays, no pandas overhead) ---
    # Convert to nanoseconds explicitly: pandas may use datetime64[us] internally,
    # and asi8 returns storage units (microseconds), not nanoseconds. price_arrays
    # already converts to ns via as_unit("ns"), so we must match.
    disc_ns = pd.DatetimeIndex(signals["disclosure_date"]).as_unit("ns").asi8
    end_ns = pd.DatetimeIndex(signals["window_end"]).as_unit("ns").asi8
    # Convert to numpy object arrays to avoid pandas StringDtype comparison overhead
    # (pandas StringDtype.__eq__ triggers _isna_string_dtype per element)
    ticker_arr = signals["ticker"].to_numpy(dtype=object, na_value=None)
    entry_prices_arr = signals["entry_price"].values
    txn_types = signals["transaction_type"].to_numpy(dtype=object, na_value=None)
    horizon_days_arr = signals["horizon_days"].values

    # --- Pre-compute SPY data once (vectorized log returns) ---
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

    # --- Pre-allocate result arrays ---
    r_peak = np.full(n, np.nan, dtype=np.float64)
    r_trough = np.full(n, np.nan, dtype=np.float64)
    r_decayed_ret = np.full(n, np.nan, dtype=np.float64)
    r_disc_baseline = np.full(n, np.nan, dtype=np.float64)
    r_last_price = np.full(n, np.nan, dtype=np.float64)
    r_spy_cum = np.zeros(n, dtype=np.float64)
    r_spy_wsum = np.zeros(n, dtype=np.float64)
    r_spy_first = np.full(n, np.nan, dtype=np.float64)
    r_spy_last = np.full(n, np.nan, dtype=np.float64)

    # --- Process per-ticker (avoids 75M+ row merge) ---
    unique_tickers = np.unique(ticker_arr)
    for ticker in unique_tickers:
        if ticker == "SPY":
            continue
        arrs = _price_arrays(prices_df, str(ticker))
        if arrs is None or arrs[0] is None:
            continue
        dates_ns, vals = arrs

        # Get indices of signals for this ticker
        tmask = ticker_arr == ticker
        t_indices = np.where(tmask)[0]
        if len(t_indices) == 0:
            continue

        _compute_ticker_signals(
            t_indices,
            disc_ns[t_indices],
            end_ns[t_indices],
            dates_ns,
            vals,
            spy_dates_ns,
            spy_vals,
            spy_log_ret,
            decay_lambda,
            r_peak, r_trough, r_decayed_ret, r_disc_baseline,
            r_last_price, r_spy_cum, r_spy_wsum, r_spy_first, r_spy_last,
        )

    # --- Derived columns (vectorized) ---
    valid_disc = (r_disc_baseline > 0) & np.isfinite(r_disc_baseline)
    total_return = np.zeros(n, dtype=np.float64)
    total_return[valid_disc] = r_last_price[valid_disc] / r_disc_baseline[valid_disc] - 1

    valid_spy = (r_spy_first > 0) & np.isfinite(r_spy_first)
    actual_spy_return = np.zeros(n, dtype=np.float64)
    actual_spy_return[valid_spy] = r_spy_last[valid_spy] / r_spy_first[valid_spy] - 1

    spy_cum = np.where(r_spy_wsum > 0, r_spy_cum / np.maximum(r_spy_wsum, 1e-15), 0.0)

    # Peak potential
    is_purchase = txn_types == TransactionType.PURCHASE.value
    purchase_mask = is_purchase & valid_disc
    sale_mask = ~is_purchase & (r_trough > 0) & np.isfinite(r_trough)

    peak_potential = np.zeros(n, dtype=np.float64)
    peak_potential[purchase_mask] = (r_peak[purchase_mask] / r_disc_baseline[purchase_mask] - 1) * 100
    peak_potential[sale_mask] = (entry_prices_arr[sale_mask] / r_trough[sale_mask] - 1) * 100

    # --- Assemble output DataFrame ---
    optional_columns = [col for col in ["owner_code", "amount_midpoint"] if col in signals.columns]
    result_columns = [
        "member", "ticker", "disclosure_date", "signal_type", "horizon_days", "entry_price",
        "peak_potential_pct", "decayed_return_pct", "spy_alpha_pct", "total_return_pct",
        "total_spy_alpha_pct", "decayed_spy_return_pct", *optional_columns,
    ]

    result_data = {
        "member": signals["member"].values,
        "ticker": ticker_arr,
        "disclosure_date": signals["disclosure_date"].values,
        "signal_type": txn_types,
        "horizon_days": horizon_days_arr,
        "entry_price": entry_prices_arr,
        "peak_potential_pct": peak_potential,
        "decayed_return_pct": r_decayed_ret * 100,
        "spy_alpha_pct": (r_decayed_ret - spy_cum) * 100,
        "total_return_pct": total_return * 100,
        "total_spy_alpha_pct": (total_return - actual_spy_return) * 100,
        "decayed_spy_return_pct": spy_cum * 100,
    }
    for col in optional_columns:
        result_data[col] = signals[col].values

    return pd.DataFrame(result_data)[result_columns]


def get_top_signals(signal_df: pd.DataFrame, horizon: int = 90, top_n: int = 15) -> pd.DataFrame:
    if signal_df.empty:
        raise AnalysisError("Empty signals dataframe")
    return _get_top_signals(signal_df, horizon, top_n)


def get_member_signals(signal_df: pd.DataFrame, member: str, horizon: int = 90, top_n: int = 5) -> pd.DataFrame:
    if signal_df.empty:
        raise AnalysisError("Empty signals dataframe")
    return _get_member_signals(signal_df, member, horizon, top_n)


def compute_signal_potential_with_member_decay(
    entry_prices_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    horizons: list[int] = [30, 60, 90, 180],
    member_decay_map: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Compute signal potential with per-member decay rates.

    If member_decay_map is provided, each member's trades use their
    personal decay lambda instead of the global default.
    """
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
