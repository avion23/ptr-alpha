"""Historical return-curve builders for OU parameter estimation.

`_build_ticker_curves` and `_build_global_curves` are memoized wrappers that
filter signals to a window, then delegate to the vectorized
`_build_curves_for_rows` which uses `searchsorted` on cached (idx_ns, vals)
arrays to avoid repeated pandas slicing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analyzer._memo import df_memoize
from analyzer.backtest.prices import _aligned_price_on_or_after_arrays
from analyzer.models import TransactionType
from analyzer.signals import _price_arrays


@df_memoize(copy=False)
def _build_ticker_curves(
    ticker: str,
    signals_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    as_of_date: pd.Timestamp,
    horizon: int,
) -> list:
    """Build historical return curves for a specific ticker's prior purchases.

    Returns a list of 1-D numpy arrays. Each array is the cumulative return
    curve r(t) = P(t)/P(entry) - 1 over [disclosure, disclosure+horizon].
    """
    if ticker not in prices_df.columns:
        return []

    base_mask = (
        (signals_df["ticker"] == ticker)
        & (signals_df["horizon_days"] == horizon)
        & (signals_df["signal_type"] == TransactionType.PURCHASE.value)
    )
    eligible = _eligible_historical_rows(signals_df, base_mask, as_of_date, horizon)
    if eligible.empty:
        return []

    return _build_curves_for_rows(
        eligible, prices_df, horizon, available_through=as_of_date
    )


@df_memoize(copy=False)
def _build_global_curves(
    signals_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    as_of_date: pd.Timestamp,
    horizon: int,
) -> list:
    """Build historical return curves across ALL tickers' prior purchases.

    Used as a global prior when a ticker has no own disclosure history.
    """
    base_mask = (signals_df["horizon_days"] == horizon) & (
        signals_df["signal_type"] == TransactionType.PURCHASE.value
    )
    all_purchases = _eligible_historical_rows(
        signals_df, base_mask, as_of_date, horizon
    )
    if all_purchases.empty:
        return []

    return _build_curves_for_rows(
        all_purchases, prices_df, horizon, available_through=as_of_date
    )


def _eligible_historical_rows(
    signals_df: pd.DataFrame,
    base_mask: pd.Series,
    as_of_date: pd.Timestamp,
    horizon: int,
) -> pd.DataFrame:
    """Select only labels known complete by the historical as-of date."""
    cutoff = pd.Timestamp(as_of_date).normalize() - pd.Timedelta(days=horizon)
    mask = base_mask & (pd.to_datetime(signals_df["disclosure_date"]) <= cutoff)
    if "window_complete" in signals_df.columns:
        mask &= signals_df["window_complete"].fillna(False).astype(bool)
    if "label_window_end" in signals_df.columns:
        label_end = pd.to_datetime(signals_df["label_window_end"], errors="coerce")
        mask &= label_end.notna() & (label_end <= pd.Timestamp(as_of_date).normalize())
    return signals_df[mask]


def _expected_exchange_session(
    sessions_ns: np.ndarray,
    target_date: pd.Timestamp,
    *,
    strictly_after: bool,
    max_wait_days: int = 7,
) -> pd.Timestamp | None:
    target = pd.Timestamp(target_date).normalize()
    threshold = target + pd.Timedelta(days=1) if strictly_after else target
    pos = int(np.searchsorted(sessions_ns, threshold.value, side="left"))
    if pos >= len(sessions_ns):
        return None
    session = pd.Timestamp(int(sessions_ns[pos])).normalize()
    if (session - target).days > max_wait_days:
        return None
    return session


def _build_curves_for_rows(
    rows: pd.DataFrame,
    prices_df: pd.DataFrame,
    horizon: int,
    *,
    available_through: pd.Timestamp | None = None,
) -> list:
    """Vectorized curve builder.

    Precomputes per-ticker (date_index_ns, price_values) once via the shared
    ``_price_arrays`` cache and uses searchsorted to slice windows instead of
    re-filtering pandas for every row.
    """
    price_cols = set(prices_df.columns)
    per_ticker: dict[str, tuple | None] = {}

    curves: list = []
    disclosures = rows["disclosure_date"].values
    tickers = rows["ticker"].values
    label_ends = (
        pd.to_datetime(rows["label_window_end"], errors="coerce").values
        if "label_window_end" in rows.columns
        else None
    )
    horizon_ns = pd.Timedelta(days=horizon).value
    available_ns = (
        pd.Timestamp(available_through).normalize().value
        if available_through is not None
        else None
    )
    market_sessions = (
        pd.DatetimeIndex(prices_df.index).normalize().unique().sort_values()
    )
    market_sessions_ns = np.asarray(
        [pd.Timestamp(session).value for session in market_sessions], dtype=np.int64
    )

    for i in range(len(rows)):
        tkr = tickers[i]
        if tkr not in price_cols:
            continue

        if tkr not in per_ticker:
            per_ticker[tkr] = _price_arrays(prices_df, tkr)

        cached = per_ticker[tkr]
        if cached is None:
            continue
        idx_ns, vals = cached
        if idx_ns is None:
            continue

        # Disclosure metadata has no publication timestamp. Determine the
        # expected exchange session first, then require this ticker's exact
        # quote. A later ticker quote cannot silently replace a missing label
        # endpoint or entry session.
        expected_entry = _expected_exchange_session(
            market_sessions_ns,
            pd.Timestamp(disclosures[i]),
            strictly_after=True,
        )
        if expected_entry is None:
            continue
        entry = _aligned_price_on_or_after_arrays(
            idx_ns, vals, expected_entry, max_wait_days=0
        )
        if entry is None:
            continue

        entry_ns = entry.date.value
        calendar_end = pd.Timestamp(entry_ns + horizon_ns)
        if label_ends is not None:
            if pd.isna(label_ends[i]):
                continue
            expected_end = pd.Timestamp(label_ends[i]).normalize()
            if expected_end < calendar_end:
                continue
        else:
            expected_end = _expected_exchange_session(
                market_sessions_ns, calendar_end, strictly_after=False
            )
            if expected_end is None:
                continue

        terminal = _aligned_price_on_or_after_arrays(
            idx_ns,
            vals,
            expected_end,
            max_wait_days=0,
        )
        if terminal is None:
            continue
        end_ns = terminal.date.value
        if available_ns is not None and end_ns > available_ns:
            continue

        lo = int(np.searchsorted(idx_ns, entry_ns, side="left"))
        hi = int(np.searchsorted(idx_ns, end_ns, side="right"))
        window = vals[lo:hi]
        valid = np.isfinite(window) & (window >= 0)
        window = window[valid]
        if len(window) < 3:
            continue
        curves.append(window / entry.price - 1.0)

    return curves
