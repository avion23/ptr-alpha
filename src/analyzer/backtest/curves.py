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


def _expected_exchange_session_on_or_after(
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


def _expected_exchange_session_on_or_before(
    sessions_ns: np.ndarray,
    target_date: pd.Timestamp,
    *,
    max_staleness_days: int = 7,
) -> pd.Timestamp | None:
    target = pd.Timestamp(target_date).normalize()
    pos = int(np.searchsorted(sessions_ns, target.value, side="right")) - 1
    if pos < 0:
        return None
    session = pd.Timestamp(int(sessions_ns[pos])).normalize()
    if (target - session).days > max_staleness_days:
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
    has_labeled_rows = "label_window_end" in rows.columns
    label_windows = (
        pd.to_datetime(rows["label_window_end"], errors="coerce").values
        if has_labeled_rows
        else None
    )
    label_entries = (
        pd.to_datetime(rows["label_entry_date"], errors="coerce").values
        if "label_entry_date" in rows.columns
        else None
    )
    label_exits = (
        pd.to_datetime(rows["label_exit_date"], errors="coerce").values
        if "label_exit_date" in rows.columns
        else None
    )
    available_ns = (
        pd.Timestamp(available_through).normalize().value
        if available_through is not None
        else None
    )
    price_dates = pd.DatetimeIndex(prices_df.index).normalize()
    market_sessions = price_dates.unique().sort_values()
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

        if has_labeled_rows:
            if label_entries is None or label_exits is None or label_windows is None:
                continue
            if (
                pd.isna(label_entries[i])
                or pd.isna(label_exits[i])
                or pd.isna(label_windows[i])
            ):
                continue
            expected_entry = pd.Timestamp(label_entries[i]).normalize()
            expected_end = pd.Timestamp(label_exits[i]).normalize()
            maturity = pd.Timestamp(label_windows[i]).normalize()
        else:
            # Date-only disclosures become available after that session. Use
            # the next exchange session as entry, then the last exchange
            # session on/before the calendar maturity, matching label/evaluate
            # semantics without substituting a different ticker quote.
            expected_entry = _expected_exchange_session_on_or_after(
                market_sessions_ns,
                pd.Timestamp(disclosures[i]),
                strictly_after=True,
            )
            if expected_entry is None:
                continue
            maturity = expected_entry + pd.Timedelta(days=horizon)
            expected_end = _expected_exchange_session_on_or_before(
                market_sessions_ns, maturity
            )
            if expected_end is None:
                continue

        if available_ns is not None and maturity.value > available_ns:
            continue
        if expected_end < expected_entry:
            continue

        entry = _aligned_price_on_or_after_arrays(
            idx_ns, vals, expected_entry, max_wait_days=0
        )
        terminal = _aligned_price_on_or_after_arrays(
            idx_ns, vals, expected_end, max_wait_days=0
        )
        if entry is None or terminal is None:
            continue

        end_ns = terminal.date.value
        if available_ns is not None and end_ns > available_ns:
            continue

        raw_window = pd.to_numeric(
            prices_df.loc[
                (price_dates >= entry.date) & (price_dates <= terminal.date), tkr
            ],
            errors="coerce",
        ).to_numpy(dtype=float)
        if (
            len(raw_window) < 3
            or not np.isfinite(raw_window).all()
            or (raw_window <= 0).any()
        ):
            continue
        curves.append(raw_window / entry.price - 1.0)

    return curves
