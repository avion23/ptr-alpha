"""Open-position tracker for skilled members.

Closed-window scoring judges each filing once and never revisits it. This
module answers the follow-up question instead: what do the best members
currently hold, what did it cost them, and which positions are on sale.

Congressional filings carry dollar bands, not share counts, so positions are
reconstructed from implied shares (amount_midpoint / entry close, where entry
is the first close on or after the transaction date). Sells relieve lots
FIFO. Option exercises file as Purchases and therefore already count as
accumulation; exchanges are ignored and reported.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from datetime import date
from functools import lru_cache

import numpy as np
import pandas as pd

from analyzer.canonical_scope import canonical_decision
from analyzer.member_names import canonical_member_key
from analyzer.parsing.cells import _extract_amount_midpoint

logger = logging.getLogger(__name__)

_BUY_TYPES = frozenset({"purchase"})
_SELL_TYPES = frozenset({"sale", "sale full", "sale partial", "partial sale"})
_OPTION_MARKERS = frozenset({"option", "call", "put", "stock option"})
_CONGRESS_SOURCES = frozenset({"house_pdf", "gemini_ocr", "senate_efd"})
# Dollar-band / close-price share estimates can leave small FIFO dust.
_POSITION_ESTIMATE_TOLERANCE = 0.10
_SHARE_EPSILON = 1e-9
_13F_SNAPSHOT_SUBTYPE = "13F_SHARE_POSITION_SNAPSHOT"
_13F_CORROBORATION_ENV = "INSIDER_ENABLE_13F_CORROBORATION"


def _13f_enabled(value):
    if value is None:
        return os.getenv(_13F_CORROBORATION_ENV, "").strip().lower() in {
            "1", "true", "yes", "on"
        }
    if not isinstance(value, bool):
        raise TypeError("enable_13f_corroboration must be a bool")
    return value


class PositionsError(Exception):
    """Raised when positions cannot be reconstructed safely."""


def gate_closed_events(
    events: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    enable_13f_corroboration: bool | None = None,
) -> pd.DataFrame:
    """Remove entry references from purchase events without an open position.

    All kinds gate uniformly on reconstructed holdings: congress ledgers see
    buys and sells, officer ledgers see fetched Form 4 buys and sales. An
    officer with no observed sales still shows open (fail open); only
    observed full exits close.
    """
    if events.empty:
        return events.copy()
    enable_13f_corroboration = _13f_enabled(enable_13f_corroboration)

    gated = events.copy()
    gated["blocked_reason"] = None
    open_positions = {}
    if {"ticker", "actor_id", "position_evidence"}.issubset(candidates.columns):
        open_rows = candidates.loc[candidates["position_evidence"].eq(True)]
        for _, position in open_rows.iterrows():
            key = (str(position["ticker"]).upper(), str(position["actor_id"]))
            open_positions[key] = (
                {
                    "event_date": position.get("event_date"),
                    "holding_state": position.get("holding_state"),
                    "shares_held": position.get("shares_held"),
                    "portfolio_pct": position.get("portfolio_pct"),
                    "portfolio_pct_basis": position.get("portfolio_pct_basis"),
                    "share_change": position.get("share_change"),
                }
                if enable_13f_corroboration and position.get("source") == "13f"
                else position.get("event_date")
            )

    # Row-major decisions without per-row Series construction: the mask and
    # the two masked assignments reproduce the iterrows loop exactly.
    def _cells(name):
        if name in gated.columns:
            return gated[name].tolist()
        return [None] * len(gated)

    position_rows = [
        open_positions.get((str(ticker).upper(), str(actor_id)))
        for ticker, actor_id in zip(_cells("ticker"), _cells("actor_id"))
    ]
    starts = [
        position.get("event_date") if isinstance(position, dict) else position
        for position in position_rows
    ]
    event_dates = _cells("event_date")
    event_sources = _cells("source")
    stale = []
    for position_start, event_date, position, source in zip(
        starts, event_dates, position_rows, event_sources
    ):
        if enable_13f_corroboration and source == "13f":
            held = (
                isinstance(position, dict)
                and position.get("holding_state") == "held"
                and _positive_shares(position.get("shares_held"))
            )
            stale.append(not held)
            continue
        stale.append(
            position_start is None
            or pd.isna(position_start)
            or event_date is None
            or pd.isna(event_date)
            or pd.Timestamp(event_date).date() < pd.Timestamp(position_start).date()
        )
    if any(stale):
        mask = pd.Series(stale, index=gated.index)
        gated.loc[mask, "entry_ref"] = None
        gated.loc[mask, "blocked_reason"] = "no open position at as_of"
        if enable_13f_corroboration:
            unavailable = mask & gated.get("source", pd.Series(index=gated.index)).eq("13f")
            gated.loc[unavailable, "blocked_reason"] = (
                "corroboration unavailable: no current 13F share snapshot"
            )
    if enable_13f_corroboration and "source" in gated:
        is_13f = gated["source"].eq("13f")
        for column in (
            "holding_state", "shares_held", "portfolio_pct",
            "portfolio_pct_basis", "share_change",
        ):
            if column not in gated:
                gated[column] = None
        for position_index, row_index in enumerate(gated.index):
            if not is_13f.loc[row_index]:
                continue
            position = position_rows[position_index]
            if isinstance(position, dict):
                gated.at[row_index, "holding_state"] = position["holding_state"]
                gated.at[row_index, "shares_held"] = position["shares_held"]
                gated.at[row_index, "portfolio_pct"] = position["portfolio_pct"]
                gated.at[row_index, "portfolio_pct_basis"] = position[
                    "portfolio_pct_basis"
                ]
                gated.at[row_index, "share_change"] = position["share_change"]
    return gated


def _positive_shares(value) -> bool:
    try:
        shares = float(value)
    except (OverflowError, TypeError, ValueError):
        return False
    return math.isfinite(shares) and shares > 0


def _is_option_row(description, instrument) -> bool:
    """Memoized option-evidence decision (see `_option_row`)."""
    return _option_row(description, instrument)


def entry_close(
    prices: pd.DataFrame, day: date, as_of: date | None = None
) -> float | None:
    """First close on or after the transaction date.

    Point-in-time: when as_of is given, closes after it are unknown, so a
    transaction whose entry bar does not exist yet yields None. The forward
    gap is capped at a week (weekends/holidays): a later bar is a different
    market, not the transaction price, so it must not manufacture a cost
    basis (and entry discipline) out of missing data.
    """
    frame = prices
    if as_of is not None:
        frame = prices.loc[prices.index <= pd.Timestamp(as_of)]
    closes = frame.loc[frame.index >= pd.Timestamp(day), "close"]
    closes = closes[closes.notna()]
    if closes.empty:
        return None
    bar_day = closes.index[0].date()
    if (bar_day - day).days > 7:
        return None
    price = float(closes.iloc[0])
    return price if math.isfinite(price) and price > 0 else None


_CURRENT_MISS = object()


class _EntryCloseLookup:
    """Memoized searchsorted twin of `entry_close` for one fixed as_of.

    Price-history frames are immutable for the lifetime of one positions
    build, so each (ticker, transaction-day) answer is a pure function of
    the frame. Sorted date indexes take a searchsorted path that returns
    exactly the value `entry_close` returns; anything unusual (unsorted,
    non-datetime, NaT-bearing index) falls back to `entry_close` itself.
    """

    __slots__ = ("_histories", "_as_of", "_prepared", "_cache",
                 "_current", "_shared_prepared", "_windows")

    def __init__(
        self, histories: dict[str, pd.DataFrame], as_of,
        prepared: dict[str, tuple] | None = None,
    ):
        self._histories = histories
        self._as_of = None if as_of is None else pd.Timestamp(as_of)
        self._prepared: dict[str, tuple | None] = {}
        self._shared_prepared: dict[str, tuple] = prepared or {}
        self._cache: dict[tuple[str, date], float | None] = {}
        self._current: dict[str, float | None] = {}
        self._windows: dict[date, tuple[pd.Timestamp, pd.Timestamp]] = {}

    def close(self, ticker: str, day: date) -> float | None:
        key = (ticker, day)
        try:
            return self._cache[key]
        except KeyError:
            pass
        result = self._compute(ticker, day)
        self._cache[key] = result
        return result

    def _prepared_frame(self, ticker: str):
        if ticker in self._prepared:
            return self._prepared[ticker]
        if ticker in self._shared_prepared:
            # Pre-cut by the loader under the same guarantees this method
            # enforces when preparing from a frame (datetime index, no NaT,
            # date-ordered, numeric closes, truncated at as_of).
            prepared = self._shared_prepared[ticker]
            self._prepared[ticker] = prepared
            return prepared
        history = self._histories.get(ticker)
        prepared = None
        if (
            history is not None
            and not history.empty
            and isinstance(history.index, pd.DatetimeIndex)
            and not history.index.hasnans
            and history.index.is_monotonic_increasing
            and self._as_of is not None
        ):
            truncated = history.loc[history.index <= self._as_of]
            values = truncated["close"].to_numpy()
            if values.dtype.kind not in "fiu":
                prepared = None
            else:
                prepared = (truncated.index, values)
        self._prepared[ticker] = prepared
        return prepared

    def _compute(self, ticker: str, day: date) -> float | None:
        prepared = self._prepared_frame(ticker)
        if prepared is None:
            return entry_close(self._histories.get(ticker), day, self._as_of)
        index, values = prepared
        day_ts, window_end = self._window(day)
        position = index.searchsorted(day_ts, side="left")
        if position >= len(index):
            return None
        # Monotonic dates: the 7-day gap cap is a pure date bound, so one
        # upper searchsorted replaces the per-bar Timestamp .date() walk.
        # Bars in [position, window_end) are exactly those with
        # (bar_day - day).days <= 7; the first non-NaN close among them is
        # the one entry_close's notna+iloc[0] prices.
        hi = index.searchsorted(window_end, side="left")
        for k in range(position, hi):
            value = values[k]
            if value == value:  # skip NaN closes exactly like closes.notna()
                price = float(value)
                return price if math.isfinite(price) and price > 0 else None
        return None

    def _window(self, day: date) -> tuple[pd.Timestamp, pd.Timestamp]:
        """Cached (day, day+8d) pair bounding the 7-day entry-gap cap."""
        cached = self._windows.get(day)
        if cached is None:
            stamp = pd.Timestamp(day)
            cached = (stamp, stamp + pd.Timedelta(days=8))
            self._windows[day] = cached
        return cached

    def current(self, ticker: str) -> float | None:
        """Memoized twin of `_current_close` for the fixed as_of."""
        cached = self._current.get(ticker, _CURRENT_MISS)
        if cached is not _CURRENT_MISS:
            return cached
        result = self._current_compute(ticker)
        self._current[ticker] = result
        return result

    def _current_compute(self, ticker: str) -> float | None:
        prepared = self._prepared_frame(ticker)
        if prepared is None:
            history = self._histories.get(ticker)
            if history is None or history.empty:
                return None
            return _current_close(history, self._as_of)
        index, values = prepared
        position = int(index.searchsorted(self._as_of, side="right")) - 1
        while position >= 0:
            value = values[position]
            if value == value:  # skip NaN closes exactly like closes.notna()
                price = float(value)
                return price if price > 0 else None
            position -= 1
        return None


def _current_close(prices: pd.DataFrame, as_of: date) -> float | None:
    """Last close on or before the as-of date."""
    closes = prices.loc[prices.index <= pd.Timestamp(as_of), "close"]
    closes = closes[closes.notna()]
    if closes.empty:
        return None
    price = float(closes.iloc[-1])
    return price if price > 0 else None


def _split_price_history(history: pd.DataFrame, as_of=None):
    """Per-ticker close frames, plus optional pre-prepared lookup arrays.

    The loaders fetch prices ``ORDER BY ticker, date``, so each ticker's
    rows arrive contiguous and date-ordered; numpy boundary detection then
    replaces the per-group ``groupby`` + ``set_index`` construction (which
    copies every group and deletes its date column separately). Any
    violated assumption (non-contiguous tickers, non-monotone or NaT
    dates) falls back to the original groupby construction verbatim.

    Returns ``(frames, prepared)`` where ``prepared`` maps ticker ->
    ``(truncated DatetimeIndex, close values)`` for ``_EntryCloseLookup``
    when ``as_of`` is given and closes are numeric, else ``None``.
    """
    frames: dict[str, pd.DataFrame] = {}
    if history.empty:
        return frames, None
    date_values = pd.to_datetime(history["date"]).to_numpy()
    close_values = history["close"].to_numpy()
    codes, uniques = pd.factorize(history["ticker"], sort=False)
    codes = np.asarray(codes)
    # Ticker groups must be contiguous (codes non-decreasing under
    # appearance-order factorization). Dates restart at every new ticker,
    # so a date decrease is valid only exactly at a group start; inside a
    # group the loader's date ordering must hold. Any NaT date falls back.
    group_start = codes.size == 0 or np.concatenate(
        ([True], codes[1:] != codes[:-1])
    )
    contiguous = bool(codes.size == 0) or (
        not np.any(codes[1:] < codes[:-1])
        and not np.any(
            (date_values[1:] < date_values[:-1]) & ~group_start[1:]
        )
        and not np.isnat(date_values).any()
    )
    if not contiguous:
        grouped = history.copy()
        grouped["date"] = pd.to_datetime(grouped["date"])
        for symbol, frame in grouped.groupby("ticker", sort=False):
            frames[str(symbol)] = frame.set_index("date")
        return frames, None

    boundaries = np.flatnonzero(
        np.concatenate(([True], codes[1:] != codes[:-1]))
    )
    ends = np.append(boundaries[1:], codes.size)
    index_name = history["date"].name or "date"
    # One shared DatetimeIndex: per-ticker frames and the prepared lookup
    # arrays all slice it (slicing wraps memory instead of re-validating),
    # and the two-arg ndarray frame constructor skips the per-column
    # sanitization the dict constructor pays.
    base_index = pd.DatetimeIndex(date_values, name=index_name)
    usable = as_of is not None and close_values.dtype.kind in "fiu"
    prepared: dict[str, tuple] | None = {} if usable else None
    as_of_ts = pd.Timestamp(as_of) if usable else None
    for position, symbol in enumerate(uniques):
        start, stop = int(boundaries[position]), int(ends[position])
        index = base_index[start:stop]
        frames[str(symbol)] = pd.DataFrame(
            close_values[start:stop], index=index, columns=["close"]
        )
        if prepared is not None:
            cut = int(index.searchsorted(as_of_ts, side="right"))
            prepared[str(symbol)] = (base_index[start:start + cut],
                                     close_values[start:start + cut])
    return frames, prepared


def _is_full_sale(kind: str, subtype) -> bool:
    for value in (kind, subtype):
        if value is None or pd.isna(value):
            continue
        normalized = re.sub(r"[^a-z]", "", str(value).lower())
        if normalized in {"salefull", "fullsale"}:
            return True
    return False


def _is_partial_sale(kind: str, subtype) -> bool:
    for value in (kind, subtype):
        if value is None or pd.isna(value):
            continue
        normalized = re.sub(r"[^a-z]", "", str(value).lower())
        if normalized in {"salepartial", "partialsale"}:
            return True
    return False


# Pure string classification over a tiny value space; the memo turns the
# per-row re.sub work into a dict lookup without changing any result.
_is_full_sale = lru_cache(maxsize=4096)(_is_full_sale)
_is_partial_sale = lru_cache(maxsize=4096)(_is_partial_sale)


def _owner_account(value) -> str | None:
    if value is None or pd.isna(value):
        return None
    account = str(value).strip().upper()
    if account in {"J", "JT"}:
        return "JT"
    return account or None


# Pure string normalizations over a tiny value space; memoization keeps the
# per-row FIFO loop off the repeated str/re work without changing results.
_owner_account = lru_cache(maxsize=4096)(_owner_account)


@lru_cache(maxsize=4096)
def _transaction_kind(value) -> str:
    return str(value or "").strip().lower()


@lru_cache(maxsize=8192)
def _option_row(description, instrument) -> bool:
    """Positive option evidence only; NULL/unknown instruments stay equity.

    House filings also mark options inline as ``[OP]`` while leaving
    instrument_type NULL on some rows; those must not count as shares.
    ``[ST]`` (stock) is authoritative and never an option. Cells are passed
    directly (None when the column is absent).
    """
    desc = ""
    if description is not None and not pd.isna(description):
        desc = str(description)
    if re.search(r"\[op\]", desc, re.IGNORECASE):
        return True
    if instrument is not None and not pd.isna(instrument):
        if str(instrument).strip().lower() in _OPTION_MARKERS:
            return True
    return False


def build_positions(
    trades: pd.DataFrame,
    price_history: dict[str, pd.DataFrame],
    *,
    as_of: date,
    entry_lookup: "_EntryCloseLookup | None" = None,
    presorted: bool = False,
) -> pd.DataFrame:
    """Reconstruct open lots per (member, ticker) with FIFO sale relief.

    `trades` needs member, ticker, transaction_type, transaction_date,
    amount_midpoint. `price_history` maps ticker -> frame indexed by date
    with a close column. Returns one row per open position with cost basis,
    current price, and discount_pct (negative = below member cost).
    """
    required = {
        "member",
        "ticker",
        "transaction_type",
        "transaction_date",
    }
    missing = required - set(trades.columns)
    if missing:
        raise PositionsError(f"trades missing columns: {sorted(missing)}")
    has_midpoint = "amount_midpoint" in trades.columns
    has_raw = "amount_raw" in trades.columns
    has_instrument = "instrument_type" in trades.columns
    has_disclosure = "disclosure_date" in trades.columns
    has_subtype = "raw_transaction_subtype" in trades.columns
    if not has_midpoint and not has_raw:
        raise PositionsError("trades need amount_midpoint or amount_raw for sizing")

    lots: dict[tuple[str, str], list] = {}
    last_activity: dict[tuple[str, str], date] = {}
    if entry_lookup is None:
        entry_lookup = _EntryCloseLookup(price_history, as_of)

    def _filing_ref(record_cell, row_cell, source) -> dict:
        """Canonical filing identity of one contributing lot row.

        Lots keep the source_row identity that opened them so a position can
        name exactly which filings contributed. Producers without ids record
        NULL explicitly; nothing is invented.
        """

        def _id(cell):
            if cell is None or pd.isna(cell):
                return None
            text = str(cell).strip()
            return text or None

        return {
            "source": source if isinstance(source, str) and source else None,
            "source_record_id": _id(record_cell) if has_record_id else None,
            "source_row_id": _id(row_cell) if has_row_id else None,
        }

    floor_sized: dict[tuple[str, str], int] = {}
    band_estimates: dict[tuple[str, str], dict[str | None, dict[str, float]]] = {}
    ineligible_for_band_tolerance: set[tuple[tuple[str, str], str | None]] = set()
    member_identities: dict[str, list[str]] = {}
    member_keys: dict[str, str] = {}
    member_labels: dict[str, str] = {}
    skipped = 0
    skipped_options = 0
    ignored_exchanges = 0

    _FILED_SHARES_RE = re.compile(
        r"([\d,.]+)\s+shares?\s+@\s*\$?([\d,.]+)", re.IGNORECASE
    )

    def _filed_lot(description) -> tuple[float, float] | None:
        """Exact (shares, price) from filed descriptions like '100 shares @ $100'.

        Form 4 rows state both; using them keeps buy-sell arithmetic exact
        instead of dividing band dollars by market closes (which leaves
        phantom shares when prices differ). None when the description cell
        is absent (no asset_description column) or unparseable.
        """
        if description is None or pd.isna(description):
            return None
        match = _FILED_SHARES_RE.search(str(description))
        if match is None:
            return None
        try:
            shares = float(match.group(1).replace(",", ""))
            price = float(match.group(2).replace(",", ""))
        except ValueError:
            return None
        if shares <= 0 or price <= 0 or not math.isfinite(shares + price):
            return None
        return shares, price

    def _size(midpoint, raw) -> tuple[float | None, bool]:
        """Dollar size with floor fallback for open-ended bands.

        Returns (size, sized_by_floor). A '$100,001 -' band parses to its
        $100,001 floor: a conservative lower bound, flagged as such.
        """
        if midpoint is not None and not pd.isna(midpoint) and float(midpoint) > 0:
            return float(midpoint), False
        if raw is None or pd.isna(raw):
            return None, False
        _, floor = _extract_amount_midpoint(str(raw))
        if floor is None or floor <= 0:
            return None, False
        return float(floor), True

    transaction_dates = pd.to_datetime(trades["transaction_date"], errors="coerce")
    disclosure_dates = (
        pd.to_datetime(trades["disclosure_date"], errors="coerce").fillna(transaction_dates)
        if has_disclosure
        else transaction_dates
    )
    known = (transaction_dates <= pd.Timestamp(as_of)) & (
        disclosure_dates <= pd.Timestamp(as_of)
    )
    skipped += int((~known).sum())
    # Filter before amendment resolution and alias grouping: future filings
    # must not remove or change the identity of a currently known lot.
    # SQL does not guarantee input order. FIFO (especially same-day buys and
    # full sales) needs a total, repeatable order, not a stable date-only sort.
    # Filing identity breaks ties where available; economic fields are the
    # fallback for older/slim inputs. This is not inferred intraday chronology.
    # A caller that already applied this exact stable key order (the batched
    # loader pre-sorts the shared canonical frame once) may pass
    # presorted=True: a stable sort of an identically-sorted frame is the
    # identity, so skipping it cannot reorder anything.
    tie_columns = dict.fromkeys(
        ("transaction_date", "disclosure_date", "source", "source_record_id",
         "source_row_id", "owner_code", *_LOADER_COLUMNS)
    )
    if presorted:
        ordered = trades.loc[known]
    else:
        ordered = trades.loc[known].sort_values(
            [column for column in tie_columns if column in trades],
            kind="mergesort", na_position="first",
        )
    if "amends_source_record_id" in ordered.columns:
        superseded = {
            str(value)
            for value in ordered["amends_source_record_id"].dropna()
            if str(value).strip()
        }
    else:
        superseded = set()
    if superseded and "source_record_id" in ordered.columns:
        mask = ordered["source_record_id"].astype("string").isin(superseded)
        dropped = int(mask.sum())
        if dropped:
            logger.info("Dropping %d rows superseded by amendments", dropped)
            ordered = ordered.loc[~mask].copy()
    has_record_id = "source_record_id" in trades.columns
    has_row_id = "source_row_id" in trades.columns
    has_description = "asset_description" in trades.columns
    has_owner = "owner_code" in trades.columns
    has_source = "source" in trades.columns

    # Column-major extraction: the FIFO decisions below are unchanged, but
    # they now index prebuilt cell lists instead of walking itertuples
    # namedtuples (per-row Series-free, per-cell extension-scalar unpack).
    # Same values, same order, same per-row decisions.
    def _column(name):
        return ordered[name].tolist()

    member_cells = _column("member")
    ticker_cells = _column("ticker")
    type_cells = _column("transaction_type")
    day_cells = _column("transaction_date")
    subtype_cells = _column("raw_transaction_subtype") if has_subtype else None
    midpoint_cells = _column("amount_midpoint") if has_midpoint else None
    raw_cells = _column("amount_raw") if has_raw else None
    owner_cells = _column("owner_code") if has_owner else None
    source_cells = _column("source") if has_source else None
    disclosure_cells = _column("disclosure_date") if has_disclosure else None
    instrument_cells = _column("instrument_type") if has_instrument else None
    description_cells = _column("asset_description") if has_description else None
    record_id_cells = _column("source_record_id") if has_record_id else None
    row_id_cells = _column("source_row_id") if has_row_id else None

    # Pure per-row derivations repeat across rows; cache them per call.
    _DAY_MISS = object()
    day_cache: dict = {}
    disclosure_day_cache: dict = {}

    for i in range(len(ordered)):
        member_value = member_cells[i]
        ticker_value = ticker_cells[i]
        if pd.isna(member_value) or pd.isna(ticker_value):
            skipped += 1
            continue
        member = str(member_value)
        canonical = canonical_member_key(member)
        member_key = member_keys.get(canonical)
        if member_key is None:
            # ponytail: scan distinct aliases; index by surname if this grows costly.
            member_key = next(
                (
                    identity
                    for identity, variants in member_identities.items()
                    if all(
                        _same_member_variant(canonical, variant)
                        for variant in variants
                    )
                ),
                canonical,
            )
            member_keys[canonical] = member_key
            member_identities.setdefault(member_key, []).append(canonical)
            member_labels.setdefault(member_key, member)
        ticker = str(ticker_value)
        key = (member_key, ticker)
        day_cell = day_cells[i]
        day = day_cache.get(day_cell, _DAY_MISS)
        if day is _DAY_MISS:
            day = pd.Timestamp(day_cell).date()
            day_cache[day_cell] = day
        kind = _transaction_kind(type_cells[i])
        subtype_cell = subtype_cells[i] if has_subtype else None
        full_sale = _is_full_sale(kind, subtype_cell)
        partial_sale = _is_partial_sale(kind, subtype_cell)
        account = _owner_account(owner_cells[i] if has_owner else None)
        estimate_key = (key, account)
        source = source_cells[i] if has_source else None
        congress_estimate = isinstance(source, str) and source in _CONGRESS_SOURCES
        if _is_option_row(
            description_cells[i] if has_description else None,
            instrument_cells[i] if has_instrument else None,
        ):
            skipped_options += 1
            continue
        disclosed = disclosure_cells[i] if has_disclosure else None
        disclosure_day = (
            disclosure_day_cache.get(disclosed, _DAY_MISS)
            if disclosed is not None and not pd.isna(disclosed)
            else day
        )
        if disclosure_day is _DAY_MISS:
            disclosure_day = pd.Timestamp(disclosed).date()
            disclosure_day_cache[disclosed] = disclosure_day
        last_activity[key] = day

        history = price_history.get(ticker)
        if history is None or history.empty:
            skipped += 1
            continue

        if kind in _BUY_TYPES:
            filed = _filed_lot(
                description_cells[i] if has_description else None
            )
            if filed is not None:
                shares, filed_price = filed
                size, by_floor = shares * filed_price, False
                if congress_estimate:
                    ineligible_for_band_tolerance.add(estimate_key)
            else:
                entry = entry_lookup.close(ticker, day)
                size, by_floor = _size(
                    midpoint_cells[i] if has_midpoint else None,
                    raw_cells[i] if has_raw else None,
                )
                if entry is None or size is None:
                    skipped += 1
                    continue
                shares = size / entry
                if congress_estimate:
                    if by_floor:
                        ineligible_for_band_tolerance.add(estimate_key)
                    else:
                        estimate = band_estimates.setdefault(key, {}).setdefault(
                            account,
                            {
                                "bought_value": 0.0,
                                "sold_value": 0.0,
                                "bought_shares": 0.0,
                            },
                        )
                        estimate["bought_value"] += size
                        estimate["bought_shares"] += shares
            lots.setdefault(key, []).append(
                [shares, size, disclosure_day, source, account, day,
                 _filing_ref(
                     record_id_cells[i] if has_record_id else None,
                     row_id_cells[i] if has_row_id else None,
                     source,
                 )]
            )
            if by_floor:
                floor_sized[key] = floor_sized.get(key, 0) + 1
        elif kind in _SELL_TYPES or full_sale:
            if full_sale:
                # A full sale closes the position regardless of the filed
                # dollar amount: estimated-amount FIFO can otherwise leave
                # phantom residual shares behind. When ownership is disclosed,
                # close only that account's lots; otherwise close the member's
                # full position in this ticker.
                if account is None:
                    lots[key] = []
                else:
                    lots[key] = [lot for lot in lots.get(key, []) if lot[4] != account]
                if account is None:
                    band_estimates.pop(key, None)
                    ineligible_for_band_tolerance.difference_update(
                        {
                            candidate
                            for candidate in ineligible_for_band_tolerance
                            if candidate[0] == key
                        }
                    )
                else:
                    estimates = band_estimates.get(key)
                    if estimates is not None:
                        estimates.pop(account, None)
                        if not estimates:
                            band_estimates.pop(key)
                    ineligible_for_band_tolerance.discard(estimate_key)
                continue
            sale_estimate_value = None
            sale_band_ineligible = False
            filed = _filed_lot(
                description_cells[i] if has_description else None
            )
            if filed is not None:
                to_relieve = filed[0]
                if congress_estimate:
                    sale_band_ineligible = True
            else:
                entry = entry_lookup.close(ticker, day)
                size, by_floor = _size(
                    midpoint_cells[i] if has_midpoint else None,
                    raw_cells[i] if has_raw else None,
                )
                if entry is None or size is None:
                    skipped += 1
                    continue
                to_relieve = size / entry
                if congress_estimate:
                    if by_floor:
                        sale_band_ineligible = True
                    else:
                        sale_estimate_value = size
            queue = lots.setdefault(key, [])
            sale_shares = to_relieve
            relieved_by_account: dict[str | None, float] = {}
            index = 0
            while to_relieve > 0 and index < len(queue):
                lot = queue[index]
                if account is not None and lot[4] != account:
                    index += 1
                    continue
                lot_shares, lot_cost = lot[0], lot[1]
                relieved_shares = min(lot_shares, to_relieve)
                relieved_by_account[lot[4]] = (
                    relieved_by_account.get(lot[4], 0.0) + relieved_shares
                )
                if lot_shares <= to_relieve + _SHARE_EPSILON:
                    to_relieve = max(0.0, to_relieve - lot_shares)
                    queue.pop(index)
                else:
                    fraction = to_relieve / lot_shares
                    remaining_shares = lot_shares - to_relieve
                    if abs(remaining_shares) <= _SHARE_EPSILON:
                        queue.pop(index)
                    else:
                        lot[0] = remaining_shares
                        lot[1] = lot_cost * (1 - fraction)
                    to_relieve = 0
            queue[:] = [lot for lot in queue if abs(lot[0]) > _SHARE_EPSILON]
            if congress_estimate and sale_shares > 0:
                for relieved_account, relieved_shares in relieved_by_account.items():
                    relieved_key = (key, relieved_account)
                    if sale_band_ineligible:
                        ineligible_for_band_tolerance.add(relieved_key)
                    else:
                        estimate = band_estimates.get(key, {}).get(relieved_account)
                        if estimate is not None and sale_estimate_value is not None:
                            estimate["sold_value"] += (
                                sale_estimate_value * relieved_shares / sale_shares
                            )
                    if not any(lot[4] == relieved_account for lot in queue):
                        estimates = band_estimates.get(key)
                        if estimates is not None:
                            estimates.pop(relieved_account, None)
                            if not estimates:
                                band_estimates.pop(key)
                        ineligible_for_band_tolerance.discard(relieved_key)
            if not partial_sale:
                for estimate_account, estimate in list(
                    band_estimates.get(key, {}).items()
                ):
                    estimate_key = (key, estimate_account)
                    if estimate_key in ineligible_for_band_tolerance:
                        continue
                    remaining_shares = sum(
                        lot[0] for lot in queue if lot[4] == estimate_account
                    )
                    if (
                        estimate["bought_value"] > 0
                        and estimate["sold_value"]
                        >= estimate["bought_value"]
                        * (1 - _POSITION_ESTIMATE_TOLERANCE)
                        and remaining_shares
                        <= estimate["bought_shares"]
                        * _POSITION_ESTIMATE_TOLERANCE
                    ):
                        queue[:] = [
                            lot for lot in queue if lot[4] != estimate_account
                        ]
                        band_estimates[key].pop(estimate_account)
                        ineligible_for_band_tolerance.discard(estimate_key)
                        if not band_estimates[key]:
                            band_estimates.pop(key)
            if not queue:
                band_estimates.pop(key, None)
                ineligible_for_band_tolerance.difference_update(
                    {
                        candidate
                        for candidate in ineligible_for_band_tolerance
                        if candidate[0] == key
                    }
                )
        else:
            ignored_exchanges += 1

    records = []
    for key, queue in lots.items():
        queue[:] = [lot for lot in queue if abs(lot[0]) > _SHARE_EPSILON]
        shares = sum(lot[0] for lot in queue)
        cost = sum(lot[1] for lot in queue)
        if shares <= _SHARE_EPSILON or cost <= 0:
            continue
        history = price_history[key[1]]
        current = entry_lookup.current(key[1])
        if current is None:
            skipped += 1
            continue
        basis = cost / shares
        records.append(
            {
                "member": member_labels[key[0]],
                "ticker": key[1],
                "shares": shares,
                "total_cost": cost,
                "cost_basis": basis,
                "current_price": current,
                "discount_pct": (current / basis - 1) * 100,
                "first_buy": min(lot[5] for lot in queue),
                "last_activity": last_activity[key],
                "disclosure_date": max(lot[2] for lot in queue),
                "source": queue[-1][3],
                "floor_sized_lots": floor_sized.get(key, 0),
                "filing_refs": _canonical_filing_refs(queue),
            }
        )

    logger.info(
        "Built %d open positions (%d skipped rows, %d option rows, %d exchanges ignored)",
        len(records),
        skipped,
        skipped_options,
        ignored_exchanges,
    )
    columns = [
        "member",
        "ticker",
        "shares",
        "total_cost",
        "cost_basis",
        "current_price",
        "discount_pct",
        "first_buy",
        "last_activity",
        "disclosure_date",
        "source",
        "floor_sized_lots",
        "filing_refs",
    ]
    return pd.DataFrame(records, columns=columns).sort_values(
        "discount_pct", kind="mergesort"
    )


def _canonical_filing_refs(queue) -> list[dict]:
    """Deduplicated, canonically sorted filing identities of surviving lots.

    Every lot names the filing that opened it (source + source_record_id +
    source_row_id, NULL when the producer has no ids), so replay can
    reconstruct which filings contributed to a position.
    """
    refs: list[dict] = []
    for lot in queue:
        ref = lot[6] if len(lot) > 6 else None
        if not isinstance(ref, dict):
            ref = {"source": None, "source_record_id": None, "source_row_id": None}
        if ref not in refs:
            refs.append(ref)
    return sorted(
        refs,
        key=lambda ref: (
            str(ref.get("source")),
            str(ref.get("source_record_id")),
            str(ref.get("source_row_id")),
        ),
    )


@canonical_decision("as_of")
def holdings_candidates(
    db,
    member: str,
    as_of: date,
    *,
    include_unpromoted: bool = False,
    enable_13f_corroboration: bool | None = None,
) -> pd.DataFrame:
    """Return open positions as uncorroborated, unexpired setup candidates.

    Lots count only when their disclosure date (or transaction date when
    missing) is no later than ``as_of``. Position valuation uses the last close
    on or before ``as_of``; future closes cannot enter a replay. Positions have
    no age expiry and remain candidates until sales close their lots.
    """
    enable_13f_corroboration = _13f_enabled(enable_13f_corroboration)
    columns = [
        "ticker",
        "actor_id",
        "kind",
        "source",
        "entry_ref",
        "event_date",
        "disclosure_date",
        "corroboration",
        "position_evidence",
        "as_of",
        "filing_refs",
    ]
    if enable_13f_corroboration:
        columns.extend(
            ["holding_state", "shares_held", "portfolio_pct", "share_change",
             "portfolio_pct_basis", "corroboration_reason"]
        )
    source_kinds = {
        "house_pdf": "congress",
        "gemini_ocr": "congress",
        "senate_efd": "congress",
        "form4": "officer",
        "13f": "manager",
    }

    try:
        from analyzer.actors import actor_id
    except ModuleNotFoundError as exc:
        if exc.name != "analyzer.actors":
            raise

        # analyzer.actors is not present in older installs; keep its stable format.
        def _actor_id(kind, key):
            return f"{kind}:{key}"

    else:
        _actor_id = actor_id

    trades = load_member_trades(db, member, include_unpromoted=include_unpromoted)
    if trades.empty:
        return pd.DataFrame(columns=columns)

    trades = trades.loc[trades["source"].isin(source_kinds)].copy()
    if trades.empty:
        return pd.DataFrame(columns=columns)

    histories = {
        str(ticker): load_price_history(db, str(ticker))
        for ticker in trades["ticker"].dropna().unique()
    }
    return _holdings_records(
        member, trades, histories, as_of, _actor_id, columns, source_kinds,
        enable_13f_corroboration=enable_13f_corroboration,
    )


@canonical_decision("as_of")
def holdings_candidates_batch(
    db,
    members: list[str],
    as_of: date,
    *,
    include_unpromoted: bool = False,
    enable_13f_corroboration: bool | None = None,
) -> dict[str, pd.DataFrame]:
    """Open-position candidates for many members with O(1) DB round-trips.

    Same per-member output as `holdings_candidates`, read under the same
    canonical as-of scope: rows promoted or accepted after the decision day
    are excluded. The distinct-name scan, the trades load, and the
    price-history load each happen once no matter how many members are
    requested; the `setups` path fans out over ~300 event members, where the
    per-member form costs ~10s each.
    """
    enable_13f_corroboration = _13f_enabled(enable_13f_corroboration)
    columns = [
        "ticker",
        "actor_id",
        "kind",
        "source",
        "entry_ref",
        "event_date",
        "disclosure_date",
        "corroboration",
        "position_evidence",
        "as_of",
        "filing_refs",
    ]
    if enable_13f_corroboration:
        columns.extend(
            ["holding_state", "shares_held", "portfolio_pct", "share_change",
             "portfolio_pct_basis", "corroboration_reason"]
        )
    source_kinds = {
        "house_pdf": "congress",
        "gemini_ocr": "congress",
        "senate_efd": "congress",
        "form4": "officer",
        "13f": "manager",
    }
    members = [m for m in dict.fromkeys(members) if isinstance(m, str) and m.strip()]
    if not members:
        return {}

    try:
        from analyzer.actors import actor_id
    except ModuleNotFoundError as exc:
        if exc.name != "analyzer.actors":
            raise

        def _actor_id(kind, key):
            return f"{kind}:{key}"

    else:
        _actor_id = actor_id

    all_names = _distinct_member_names(db)
    variants: dict[str, list[str]] = {}
    for member in members:
        wanted = canonical_member_key(member)
        variants[member] = _variants_for(member, wanted, all_names)
    universe = sorted({v for vs in variants.values() for v in vs})

    canonical = db.conn.execute(
        f"""
        SELECT {_loader_select(db, "canonical_transactions")}
        FROM canonical_transactions
        WHERE member = ANY(?)
        """,
        [universe],
    ).fetchdf()
    by_member: dict[str, pd.DataFrame] = {}
    kind_by_member: dict[str, pd.Series | None] = {}
    if not canonical.empty:
        # One stable sort by build_positions' full tie-key order for the whole
        # shared frame; every per-member partition inherits it, so the
        # per-member re-sorts become identities and are skipped.
        _tie_columns = dict.fromkeys(
            ("transaction_date", "disclosure_date", "source", "source_record_id",
             "source_row_id", "owner_code", *_LOADER_COLUMNS)
        )
        canonical = canonical.sort_values(
            [column for column in _tie_columns if column in canonical],
            kind="mergesort", na_position="first",
        )
        # One positional pass groups rows by raw filed name; each member's
        # partition is the ordered union of its variants' row positions.
        # Identical rows in identical order to the previous per-member
        # boolean masks, without M full-frame scans: boolean indexing walks
        # positions in frame order, and so does a sorted position union.
        # NaN/absent members never match a variant list, exactly like isin.
        name_groups = canonical.groupby("member", sort=False).indices
        kind_series = (
            canonical["source"].map(source_kinds)
            if "source" in canonical.columns
            else None
        )
        empty_kind = (
            pd.Series(np.nan, index=canonical.index[:0], dtype="object")
            if kind_series is not None
            else None
        )
        for member in members:
            found = [
                name_groups[variant]
                for variant in variants[member]
                if variant in name_groups
            ]
            if found:
                positions = np.sort(np.concatenate(found))
                by_member[member] = canonical.take(positions)
                if kind_series is not None:
                    kind_by_member[member] = kind_series.take(positions)
            else:
                by_member[member] = canonical.iloc[0:0]
                if kind_series is not None:
                    kind_by_member[member] = empty_kind
    else:
        by_member = {member: canonical for member in members}

    if include_unpromoted:
        raw = db.conn.execute(
            f"""
            SELECT {_loader_select(db, "transactions")}
            FROM (
                SELECT t.*,
                    ROW_NUMBER() OVER (
                        PARTITION BY member, ticker, transaction_date,
                            transaction_type, disclosure_date
                        ORDER BY amount_midpoint DESC NULLS LAST
                    ) AS rn
                FROM transactions t
                WHERE t.member = ANY(?)
                  AND NOT EXISTS (
                      SELECT 1 FROM canonical_transactions c
                      WHERE c.source IS NOT DISTINCT FROM t.source
                        AND c.source_record_id IS NOT DISTINCT FROM t.source_record_id
                        AND c.source_row_id IS NOT DISTINCT FROM t.source_row_id
                        AND c.ingestion_generation IS NOT DISTINCT FROM t.ingestion_generation
                  )
            )
            WHERE rn = 1
            """,
            [universe],
        ).fetchdf()
        if not raw.empty:
            for member in members:
                mine_raw = raw.loc[raw["member"].isin(variants[member])].copy()
                if mine_raw.empty:
                    continue
                combined = pd.concat([by_member[member], mine_raw], ignore_index=True)
                by_member[member] = combined.sort_values(
                    "transaction_date", kind="mergesort"
                )
                # The shared kind column no longer aligns with the combined
                # frame; let the record builder recompute it for this member.
                kind_by_member[member] = None

    tickers = sorted(
        {
            str(t)
            for frame in by_member.values()
            if not frame.empty and "ticker" in frame.columns
            for t in frame["ticker"].dropna().unique()
        }
    )
    histories: dict[str, pd.DataFrame] = {}
    prepared: dict[str, tuple] | None = None
    if tickers:
        history = db.conn.execute(
            "SELECT ticker, date, close FROM prices WHERE ticker = ANY(?) "
            "ORDER BY ticker, date",
            [tickers],
        ).fetchdf()
        if not history.empty:
            # ORDER BY ticker, date: contiguous, date-ordered groups let the
            # splitter build every frame with numpy slices (no per-group
            # groupby/set_index) and pre-cut the lookup's close arrays.
            histories, prepared = _split_price_history(history, as_of)
    for ticker in tickers:
        histories.setdefault(ticker, pd.DataFrame())

    entry_lookup = _EntryCloseLookup(histories, as_of, prepared=prepared)
    out: dict[str, pd.DataFrame] = {}
    for member in members:
        trades = by_member[member]
        if trades.empty:
            out[member] = pd.DataFrame(columns=columns)
            continue
        # The per-member source-kind column was computed once for the whole
        # shared canonical frame; the map here covers callers that bring
        # their own frame (and the concat-with-raw unpromoted path). Rows
        # with an unmapped source land in no actor-kind mask, exactly like
        # the previous source-isin prefilter.
        out[member] = _holdings_records(
            member, trades, histories, as_of, _actor_id, columns, source_kinds,
            entry_lookup=entry_lookup,
            presorted=not include_unpromoted,
            kind_cells=kind_by_member.get(member),
            enable_13f_corroboration=enable_13f_corroboration,
        )
    return out


def _holdings_records(
    member, trades: pd.DataFrame, histories: dict, as_of, _actor_id, columns, source_kinds,
    entry_lookup=None, presorted: bool = False, kind_cells: pd.Series | None = None,
    enable_13f_corroboration: bool = False,
) -> pd.DataFrame:
    """Shared per-member record builder over preloaded trades+histories."""
    records = []
    kinds_local = (
        kind_cells
        if kind_cells is not None
        else trades["source"].map(source_kinds)
    )
    for actor_kind in ("congress", "officer", "manager"):
        kind_trades = trades.loc[kinds_local.eq(actor_kind)]
        if kind_trades.empty:
            continue
        if actor_kind == "manager" and enable_13f_corroboration:
            records.extend(
                _13f_share_state_records(
                    member, kind_trades, as_of, _actor_id
                )
            )
            kind_trades = kind_trades.loc[kind_trades["source"].ne("13f")]
            if kind_trades.empty:
                continue
        positions = build_positions(
            kind_trades, histories, as_of=as_of, entry_lookup=entry_lookup,
            presorted=presorted,
        )
        for position in positions.itertuples(index=False):
            source = position.source
            # build_positions emits only net-long positions, so any row here
            # is open in available history. Only congress rows carry entry
            # references: officer rows come from buys-and-sales Form 4 flow
            # whose full history is not guaranteed, so their events initiate
            # with filed prices instead of a market-close basis here.
            congress_position = actor_kind == "congress"
            records.append(
                {
                    "ticker": position.ticker,
                    "actor_id": _actor_id(actor_kind, member),
                    "kind": actor_kind,
                    "source": source,
                    "entry_ref": (
                        float(position.cost_basis)
                        if congress_position and source != "13f"
                        else None
                    ),
                    "event_date": position.first_buy,
                    "disclosure_date": position.disclosure_date,
                    "corroboration": False,
                    "position_evidence": True,
                    "as_of": as_of,
                    "filing_refs": position.filing_refs,
                    **(
                        {
                            "holding_state": "held",
                            "shares_held": position.shares,
                            "portfolio_pct": None,
                            "portfolio_pct_basis": None,
                            "share_change": None,
                            "corroboration_reason": None,
                        }
                        if enable_13f_corroboration
                        else {}
                    ),
                }
            )

    result = pd.DataFrame(records, columns=columns)
    if not result.empty:
        result["entry_ref"] = pd.Series(
            [record["entry_ref"] for record in records], dtype=object
        )
    return result


def _parse_13f_snapshot(raw, subtype):
    if subtype != _13F_SNAPSHOT_SUBTYPE:
        return None, "corroboration unavailable: legacy 13F row lacks a share snapshot"
    try:
        snapshot = json.loads(str(raw))
    except (TypeError, ValueError):
        return None, "corroboration unavailable: malformed 13F share snapshot"
    if not isinstance(snapshot, dict) or snapshot.get("format") != "13f-share-snapshot-v1":
        return None, "corroboration unavailable: unsupported 13F share snapshot"
    shares = snapshot.get("shares_held")
    if isinstance(shares, bool) or not isinstance(shares, int) or shares < 0:
        return None, "corroboration unavailable: invalid shares-held value"
    change = snapshot.get("change_shares")
    if isinstance(change, bool) or not isinstance(change, int):
        return None, "corroboration unavailable: invalid 13F share change"
    portfolio_pct = snapshot.get("portfolio_pct")
    if portfolio_pct is not None:
        try:
            portfolio_pct = float(portfolio_pct)
        except (OverflowError, TypeError, ValueError):
            return None, "corroboration unavailable: invalid portfolio percentage"
        if not math.isfinite(portfolio_pct) or portfolio_pct < 0:
            return None, "corroboration unavailable: invalid portfolio percentage"
    state = "held" if shares > 0 else "absent"
    return {
        "holding_state": state,
        "shares_held": shares,
        "portfolio_pct": portfolio_pct,
        "portfolio_pct_basis": snapshot.get(
            "portfolio_pct_basis",
            "reported 13F value (all information-table entries)",
        ),
        "share_change": change,
    }, None


def _13f_share_state_records(member, trades, as_of, _actor_id):
    transaction_dates = pd.to_datetime(trades["transaction_date"], errors="coerce")
    disclosure_dates = (
        pd.to_datetime(trades["disclosure_date"], errors="coerce")
        if "disclosure_date" in trades
        else transaction_dates
    )
    known = (transaction_dates <= pd.Timestamp(as_of)) & (
        disclosure_dates <= pd.Timestamp(as_of)
    )
    ordered = trades.loc[known].copy()
    if ordered.empty:
        return []
    ordered["_snapshot_transaction_date"] = transaction_dates.loc[known]
    ordered["_snapshot_disclosure_date"] = disclosure_dates.loc[known]
    sort_columns = [
        column
        for column in (
            "ticker",
            "_snapshot_transaction_date",
            "_snapshot_disclosure_date",
            "source_record_id",
        )
        if column in ordered
    ]
    ordered = ordered.sort_values(sort_columns, kind="mergesort")
    records = []
    for ticker, ticker_rows in ordered.groupby("ticker", sort=True):
        latest_key = tuple(ticker_rows.iloc[-1][column] for column in sort_columns[1:])
        latest_mask = pd.Series(True, index=ticker_rows.index)
        for column, value in zip(sort_columns[1:], latest_key):
            latest_mask &= (
                ticker_rows[column].isna()
                if value is None or pd.isna(value)
                else ticker_rows[column].eq(value)
            )
        latest_rows = ticker_rows.loc[latest_mask].to_dict("records")
        snapshots = [
            _parse_13f_snapshot(
                row.get("amount_raw"), row.get("raw_transaction_subtype")
            )
            for row in latest_rows
        ]
        snapshot_error = next((reason for snapshot, reason in snapshots if snapshot is None), None)
        if snapshot_error:
            snapshot = {
                "holding_state": "unavailable",
                "shares_held": None,
                "portfolio_pct": None,
                "portfolio_pct_basis": None,
                "share_change": None,
            }
            reason = snapshot_error
        else:
            parsed = [snapshot for snapshot, _ in snapshots]
            shares_held = sum(item["shares_held"] for item in parsed)
            snapshot = {
                "holding_state": "held" if shares_held > 0 else "absent",
                "shares_held": shares_held,
                "portfolio_pct": (
                    sum(item["portfolio_pct"] for item in parsed)
                    if all(item["portfolio_pct"] is not None for item in parsed)
                    else None
                ),
                "portfolio_pct_basis": parsed[0]["portfolio_pct_basis"],
                "share_change": sum(item["share_change"] for item in parsed),
            }
            reason = None
        active = snapshot["holding_state"] == "held" and _positive_shares(
            snapshot["shares_held"]
        )
        row = latest_rows[-1]
        records.append(
            {
                "ticker": str(ticker),
                "actor_id": _actor_id("manager", member),
                "kind": "manager",
                "source": "13f",
                "entry_ref": None,
                "event_date": row.get("transaction_date"),
                "disclosure_date": row.get("disclosure_date"),
                "corroboration": active,
                "position_evidence": active,
                "as_of": as_of,
                "filing_refs": [
                    {
                        "source": "13f",
                        "source_record_id": ref.get("source_record_id"),
                        "source_row_id": ref.get("source_row_id"),
                    }
                    for ref in latest_rows
                ],
                **snapshot,
                "corroboration_reason": reason or (
                    None if active else "corroboration unavailable: 13F position absent at as_of"
                ),
            }
        )
    return records


def discount_alerts(positions: pd.DataFrame, *, threshold_pct: float = -10.0) -> pd.DataFrame:
    """Positions trading at or below the discount threshold (negative pct)."""
    if positions.empty:
        return positions.copy()
    return positions.loc[positions["discount_pct"] <= threshold_pct].copy()


_NICKNAMES = {
    "bob": "robert", "bill": "william", "jim": "james", "mike": "michael",
    "tom": "thomas", "dave": "david", "dan": "daniel", "steve": "steven",
    "joe": "joseph", "matt": "matthew", "nick": "nicholas", "alex": "alexander",
    "chris": "christopher", "pat": "patrick", "tim": "timothy",
    "jeff": "jeffrey", "greg": "gregory", "ron": "ronald", "ken": "kenneth",
    "larry": "lawrence", "rick": "richard", "chuck": "charles", "ro": "rohit",
    "hal": "harold", "dick": "richard", "ned": "edward", "ted": "theodore",
}


def _same_person(first: str, candidate: str) -> bool:
    """Last-name-gated nickname equivalence for filed name variants."""
    if first == candidate:
        return True
    for short, full in _NICKNAMES.items():
        if {first, candidate} == {short, full}:
            return True
    return False


def _same_member_variant(first: str, candidate: str) -> bool:
    """Match filed variants without joining conflicting middle initials."""
    first_tokens = first.split()
    candidate_tokens = candidate.split()
    if len(first_tokens) < 2 or len(candidate_tokens) < 2:
        return first == candidate
    if first_tokens[-1] != candidate_tokens[-1]:
        return False
    if not _same_person(first_tokens[0].lower(), candidate_tokens[0].lower()):
        return False
    first_middle = first_tokens[1][0] if len(first_tokens) > 2 else None
    candidate_middle = candidate_tokens[1][0] if len(candidate_tokens) > 2 else None
    return (
        first_middle is None
        or candidate_middle is None
        or first_middle == candidate_middle
    )


def same_member_identity(first: str, candidate: str) -> bool:
    """Whether two filed names denote the same human.

    Canonical keys retain middle names; the filed-variant rule covers
    nicknames and optional middle names (Chuck <-> Charles J.).
    """
    first_key = canonical_member_key(first)
    candidate_key = canonical_member_key(candidate)
    return first_key == candidate_key or _same_member_variant(
        first_key, candidate_key
    )


def same_actor_identity(first: str, candidate: str) -> bool:
    """Whether two actor_ids denote the same human initiator."""
    if first == candidate:
        return True
    first_kind, _, first_name = first.partition(":")
    candidate_kind, _, candidate_name = candidate.partition(":")
    return (
        first_kind == candidate_kind == "congress"
        and bool(first_name and candidate_name)
        and _same_actor_member(first_name, candidate_name)
    )


def _same_actor_member(first: str, candidate: str) -> bool:
    first_key = canonical_member_key(first)
    candidate_key = canonical_member_key(candidate)
    first_tokens = first_key.split()
    candidate_tokens = candidate_key.split()
    if first_key == candidate_key:
        return True
    if len(first_tokens) < 2 or len(candidate_tokens) < 2:
        return False
    if first_tokens[-1] != candidate_tokens[-1]:
        return False
    if not _same_person(first_tokens[0].lower(), candidate_tokens[0].lower()):
        return False
    first_middle = first_tokens[1:-1]
    candidate_middle = candidate_tokens[1:-1]
    return not first_middle or not candidate_middle or first_middle == candidate_middle


def coalesce_actor_variants(events: pd.DataFrame, from_positions: pd.DataFrame) -> pd.DataFrame:
    """Rewrite congress event actor_ids to their open-position identity.

    One human filing under two name variants must initiate once. Event
    actors matching an open position take that position's actor_id; the
    rest keep their own (first-seen wins within the events frame).
    """
    if events.empty or "actor_id" not in events.columns:
        return events
    if "_pre_coalesced_actor_id" not in events.columns:
        # Skill validation must retain the identity before shared scoring pools it.
        events["_pre_coalesced_actor_id"] = events["actor_id"].astype(str)
    representatives: list[str] = []
    if {"actor_id", "position_evidence"}.issubset(from_positions.columns):
        open_actors = from_positions.loc[
            from_positions["position_evidence"].eq(True), "actor_id"
        ].astype(str)
        representatives = list(dict.fromkeys(open_actors))
    # Row order only matters through first-seen actor_id uniqueness: once an
    # actor_id is mapped (or appended), later identical actor_ids scan the
    # same unchanged prefix of `representatives` and land on the same match.
    # So resolve the match once per distinct actor_id in first-seen order.
    mapping: dict[str, str] = {}
    for actor in events["actor_id"].astype(str):
        match = mapping.get(actor)
        if match is not None:
            continue
        match = next(
            (
                representative
                for representative in representatives
                if same_actor_identity(representative, actor)
            ),
            None,
        )
        if match is None:
            representatives.append(actor)
            match = actor
        mapping[actor] = match
    events["actor_id"] = [mapping[value] for value in events["actor_id"].astype(str)]
    return events


def _identity_key(name: str) -> str:
    try:
        return canonical_member_key(name)
    except (TypeError, ValueError):
        return str(name).strip().upper()


def dedupe_member_names(
    names: list[str], exclude: list[str] | tuple[str, ...] = ()
) -> list[str]:
    """Drop names denoting an already-listed (or excluded) human.

    Matches on canonical key first, then the filed-variant rule for
    nicknames and optional middle names. First spelling wins.
    """
    keys = [_identity_key(name) for name in exclude]
    kept: list[str] = []
    for name in names:
        key = _identity_key(name)
        if key in keys or any(
            _same_member_variant(key, seen) or _same_member_variant(seen, key)
            for seen in keys
        ):
            continue
        keys.append(key)
        kept.append(name)
    return kept


def _member_variants(db, member: str) -> list[str]:
    """All filed name variants for the member.

    Filers appear as 'Charles J. Fleischmann', 'Charles J. "Chuck"
    Fleischmann', and 'Charles J Fleischmann' across filings; users type
    'Chuck Fleischmann'. Match on canonical key first, then on
    last-name plus nickname-equivalent first name. An exact match on any
    one variant previously dropped the others' positions silently. Middle
    initials must agree when both names provide one.
    """
    if not isinstance(member, str) or not member.strip():
        return [member]
    wanted = canonical_member_key(member)
    try:
        names = db.conn.execute("SELECT DISTINCT member FROM transactions").fetchall()
    except Exception:
        names = []
    try:
        names += db.conn.execute(
            "SELECT DISTINCT member FROM canonical_transactions"
        ).fetchall()
    except Exception:
        pass
    all_names = [row[0] for row in names]
    return _variants_for(member, wanted, all_names)


@lru_cache(maxsize=65536)
def _matches_wanted(key: str, wanted: str) -> bool:
    """Memoized exact-key-or-filed-variant match (pure string decision)."""
    return key == wanted or _same_member_variant(key, wanted)


def _variants_for(member: str, wanted: str, all_names: list) -> list[str]:
    """Match variants against an already-loaded distinct-name list."""
    seen = []
    for name in all_names:
        if name in seen:
            continue
        key = canonical_member_key(name)
        if _matches_wanted(key, wanted):
            seen.append(name)
    return seen or [member]


def _distinct_member_names(db) -> list:
    """All filed member names, loaded once for batched variant matching."""
    try:
        names = db.conn.execute("SELECT DISTINCT member FROM transactions").fetchall()
    except Exception:
        names = []
    try:
        names += db.conn.execute(
            "SELECT DISTINCT member FROM canonical_transactions"
        ).fetchall()
    except Exception:
        pass
    return [row[0] for row in names]


_LOADER_COLUMNS = (
    "member",
    "ticker",
    "transaction_type",
    "raw_transaction_subtype",
    "transaction_date",
    "disclosure_date",
    "amount_midpoint",
    "amount_raw",
    "source",
    "owner_code",
    "instrument_type",
    "asset_description",
    "source_record_id",
    "source_row_id",
    "amends_source_record_id",
)


def _loader_select(db, table: str) -> str:
    """Column list for the trades loader, tolerating slim test doubles."""
    try:
        actual = {
            row[1] for row in db.conn.execute(f"PRAGMA table_info({table})").fetchall()
        }
    except Exception:
        return "*"
    selected = [column for column in _LOADER_COLUMNS if column in actual]
    return ", ".join(selected) if selected else "*"


def load_member_trades(db, member: str, *, include_unpromoted: bool = False) -> pd.DataFrame:
    """All canonical rows for one member across every year and source.

    With include_unpromoted, raw filings from not-yet-promoted House
    generations are added (deduped to one row per economic event, preferring
    closed bands). Those rows are UNVALIDATED: same filing may parse
    differently across generations, so amounts are approximate. Canonical
    rows always win; the flag only widens recency, never scoring.
    """
    canonical = db.conn.execute(
        f"""
        SELECT {_loader_select(db, "canonical_transactions")}
        FROM canonical_transactions
        WHERE member IN (SELECT UNNEST(?))
        """,
        [_member_variants(db, member)],
    ).fetchdf()
    if not include_unpromoted:
        return canonical.sort_values("transaction_date", kind="mergesort")

    raw = db.conn.execute(
        f"""
        SELECT {_loader_select(db, "transactions")}
        FROM (
            SELECT t.*,
                ROW_NUMBER() OVER (
                    PARTITION BY member, ticker, transaction_date,
                        transaction_type, disclosure_date
                    ORDER BY amount_midpoint DESC NULLS LAST
                ) AS rn
            FROM transactions t
            WHERE t.member IN (SELECT UNNEST(?))
              AND NOT EXISTS (
                  SELECT 1 FROM canonical_transactions c
                  WHERE c.source IS NOT DISTINCT FROM t.source
                    AND c.source_record_id IS NOT DISTINCT FROM t.source_record_id
                    AND c.source_row_id IS NOT DISTINCT FROM t.source_row_id
                    AND c.ingestion_generation IS NOT DISTINCT FROM t.ingestion_generation
              )
        )
        WHERE rn = 1
        """,
        [_member_variants(db, member)],
    ).fetchdf()
    combined = pd.concat([canonical, raw], ignore_index=True)
    logger.warning(
        "Including %d unvalidated unpromoted rows for %s alongside %d canonical",
        len(raw),
        member,
        len(canonical),
    )
    return combined.sort_values("transaction_date", kind="mergesort")


def load_price_history(db, ticker: str) -> pd.DataFrame:
    """Full close history for one ticker, indexed by date."""
    frame = db.conn.execute(
        "SELECT date, close FROM prices WHERE ticker = ? ORDER BY date",
        [ticker],
    ).fetchdf()
    if frame.empty:
        return frame
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.set_index("date").sort_index()
