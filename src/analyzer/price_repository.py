from __future__ import annotations

import logging
from datetime import date
from functools import lru_cache

import duckdb
import numpy as np
import pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    GoodFriday,
    Holiday,
    USLaborDay,
    USMartinLutherKingJr,
    USMemorialDay,
    USPresidentsDay,
    USThanksgivingDay,
    nearest_workday,
)
from pandas.tseries.offsets import CustomBusinessDay

from analyzer.ticker_resolver import TickerResolver


logger = logging.getLogger(__name__)


def _nyse_new_year_observance(day: pd.Timestamp) -> pd.Timestamp | None:
    """NYSE does not observe a Saturday New Year on the preceding Friday."""
    if day.weekday() == 5:
        return None
    if day.weekday() == 6:
        return day + pd.Timedelta(days=1)
    return day


class _NYSEHolidayCalendar(AbstractHolidayCalendar):
    """NYSE trading-day closures.

    ``USFederalHolidayCalendar`` is wrong for equity prices: it includes
    Columbus Day and Veterans Day (regular NYSE trading days) and omits
    Good Friday (an NYSE closure). Building the calendar from the actual
    NYSE closure rules keeps ``get_missing`` honest about missing prices.
    """

    rules = [
        Holiday(
            "New Year's Day",
            month=1,
            day=1,
            observance=_nyse_new_year_observance,
        ),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday(
            "Juneteenth National Independence Day",
            month=6,
            day=19,
            start_date="2022-01-01",
            observance=nearest_workday,
        ),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Christmas Day", month=12, day=25, observance=nearest_workday),
        # One-off full-session closures are not covered by recurring holiday
        # rules. Keep them explicit so a valid cache is not treated as missing
        # and entries do not target a day when the exchange was closed.
        Holiday(
            "National Day of Mourning for George H.W. Bush",
            month=12,
            day=5,
            start_date="2018-12-05",
            end_date="2018-12-05",
        ),
        Holiday(
            "National Day of Mourning for Jimmy Carter",
            month=1,
            day=9,
            start_date="2025-01-09",
            end_date="2025-01-09",
        ),
    ]


_NYSE_HOLIDAYS = _NYSEHolidayCalendar()
_NYSE_BUSINESS_DAY = CustomBusinessDay(calendar=_NYSE_HOLIDAYS)


def _normalize_session_date(value) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tz is not None:
        timestamp = timestamp.tz_localize(None)
    return timestamp.normalize()


def nyse_sessions(start, end) -> pd.DatetimeIndex:
    """Return expected full NYSE sessions for an inclusive date range."""
    start_ts = _normalize_session_date(start)
    end_ts = _normalize_session_date(end)
    if end_ts < start_ts:
        return pd.DatetimeIndex([])
    weekdays = pd.bdate_range(start_ts, end_ts)
    holidays = _NYSE_HOLIDAYS.holidays(start=start_ts, end=end_ts)
    return weekdays.difference(holidays)


@lru_cache(maxsize=8192)
def _next_nyse_session_ns(day_ns: int) -> int:
    day_ts = pd.Timestamp(day_ns)
    return pd.Timestamp(day_ts + _NYSE_BUSINESS_DAY).value


def next_nyse_session(day) -> pd.Timestamp:
    day_ts = _normalize_session_date(day)
    return pd.Timestamp(_next_nyse_session_ns(day_ts.value))


@lru_cache(maxsize=8192)
def _previous_nyse_session_ns(day_ns: int) -> int:
    day_ts = pd.Timestamp(day_ns)
    return pd.Timestamp(_NYSE_BUSINESS_DAY.rollback(day_ts)).value


def previous_nyse_session(day) -> pd.Timestamp:
    day_ts = _normalize_session_date(day)
    return pd.Timestamp(_previous_nyse_session_ns(day_ts.value))


class PriceRepository:
    def __init__(self, conn: duckdb.DuckDBPyConnection) -> None:
        self.conn = conn

    def get(self, tickers: list[str], start_date: date, end_date: date) -> pd.DataFrame:
        if not tickers:
            return pd.DataFrame()

        result = self.conn.execute(
            """
            SELECT date, ticker, close
            FROM prices
            WHERE ticker IN (SELECT UNNEST(?))
              AND date BETWEEN ? AND ?
              AND close > 0
              AND isfinite(close)
            ORDER BY date, ticker
        """,
            [tickers, start_date, end_date],
        ).fetchdf()

        if result.empty:
            return pd.DataFrame()

        pivot = result.pivot(index="date", columns="ticker", values="close")
        return pivot

    def upsert(self, df: pd.DataFrame) -> None:
        if df.empty:
            return

        normalized = df.copy()
        try:
            index = pd.DatetimeIndex(pd.to_datetime(normalized.index))
        except (TypeError, ValueError) as exc:
            raise ValueError("Price index must contain valid dates") from exc
        if index.tz is not None:
            index = index.tz_localize(None)
        index = index.normalize()
        if index.has_duplicates:
            raise ValueError("Price index contains duplicate calendar dates")
        normalized.index = index

        df_reset = normalized.reset_index().copy()
        index_col_name = df_reset.columns[0]
        prices_long = df_reset.melt(
            id_vars=[index_col_name], var_name="ticker", value_name="close"
        )
        prices_long = prices_long.rename(columns={index_col_name: "date"})
        prices_long["close"] = pd.to_numeric(prices_long["close"], errors="coerce")
        has_close = prices_long["close"].notna()
        valid_close = (
            has_close
            & np.isfinite(prices_long["close"])
            & (prices_long["close"] > 0)
        )
        rejected = int((has_close & ~valid_close).sum())
        if rejected:
            logger.warning("Rejected %d non-finite or non-positive price observations", rejected)
        prices_long = prices_long.loc[valid_close]
        if prices_long.empty:
            return

        self.conn.execute("""
            INSERT INTO prices (ticker, date, close)
            SELECT ticker, date, close
            FROM prices_long
            ON CONFLICT (ticker, date) DO UPDATE SET
                close = EXCLUDED.close
        """)

    def get_missing(
        self, tickers: list[str], start_date: date, end_date: date
    ) -> tuple[list[str], list[pd.Timestamp]]:
        if not tickers:
            return [], []

        required_dates = nyse_sessions(start_date, end_date)
        if required_dates.empty:
            return [], []

        existing = self.conn.execute(
            """
            SELECT DISTINCT ticker, date
            FROM prices
            WHERE ticker IN (SELECT UNNEST(?))
              AND date BETWEEN ? AND ?
              AND close > 0
              AND isfinite(close)
        """,
            [tickers, start_date, end_date],
        ).fetchdf()

        if existing.empty:
            return tickers, required_dates.to_list()

        existing_tickers = set(existing["ticker"].unique())
        missing_tickers = [t for t in tickers if t not in existing_tickers]

        existing["date"] = pd.to_datetime(existing["date"])
        required_dates_set = set(required_dates)
        per_ticker_dates = existing.groupby("ticker")["date"].apply(set)

        tickers_with_gaps = list(missing_tickers)
        gap_dates: set[pd.Timestamp] = set()
        for ticker in tickers:
            if ticker not in per_ticker_dates.index:
                gap_dates.update(required_dates_set)
                continue

            ticker_dates = per_ticker_dates[ticker]
            gaps = required_dates_set - ticker_dates
            if gaps:
                tickers_with_gaps.append(ticker)
                gap_dates.update(gaps)

        if tickers_with_gaps:
            missing_dates_list = sorted(gap_dates)
            logger.debug(
                "Per-ticker price gaps: %d tickers with missing dates (%d distinct dates)",
                len(tickers_with_gaps),
                len(missing_dates_list),
            )
            return tickers_with_gaps, missing_dates_list

        return [], []

    def get_entry_prices(
        self,
        tickers: list[str],
        start_date: date,
        end_date: date,
        resolver: TickerResolver | None = None,
    ) -> pd.DataFrame:
        if not tickers:
            return pd.DataFrame()

        if resolver is None:
            resolver = TickerResolver()

        # Entry identity is resolved at the public decision boundary. Rename
        # aliases therefore vary by disclosure date, not by the private
        # transaction date that was unavailable to the market at execution.
        pairs = self.conn.execute(
            """
            SELECT DISTINCT ticker, disclosure_date
            FROM canonical_transactions
            WHERE ticker IN (SELECT UNNEST(?))
              AND disclosure_date BETWEEN ? AND ?
            """,
            [tickers, start_date, end_date],
        ).fetchdf()

        if pairs.empty:
            return pd.DataFrame()

        alias_tickers = sorted(resolver.RENAME_MAP)
        map_raw: list[str] = []
        map_disclosure_date: list[object] = []
        map_resolved: list[str] = []
        expanded_tickers: list[str] = []
        seen: set[str] = set()
        for _, pair in pairs.iterrows():
            raw = pair["ticker"]
            disclosure_date = pair["disclosure_date"]
            resolution = resolver.resolve(raw, disclosure_date)
            if resolution.status in {"acquired", "date_required", "pre_listing"}:
                continue
            resolved = resolution.price_symbol
            map_raw.append(raw)
            map_disclosure_date.append(disclosure_date)
            map_resolved.append(resolved)
            for t in (raw, resolved):
                if t not in seen:
                    seen.add(t)
                    expanded_tickers.append(t)

        if not map_raw:
            return pd.DataFrame()

        result = self.conn.execute(
            """
            WITH ticker_map(raw, disclosure_date, resolved) AS (
                SELECT UNNEST(?), UNNEST(?), UNNEST(?)
            ),
            resolved_tickers AS (
                SELECT t.*, COALESCE(tm.resolved, t.ticker) AS resolved_ticker
                FROM canonical_transactions t
                JOIN ticker_map tm
                  ON t.ticker = tm.raw
                 AND t.disclosure_date IS NOT DISTINCT FROM tm.disclosure_date
            )
            SELECT r.member, r.ticker, r.resolved_ticker, r.transaction_date,
                   r.disclosure_date, r.transaction_type, r.owner_code,
                   r.amount_midpoint, r.instrument_type, r.strike_price,
                   r.expiry_date
            FROM resolved_tickers r
            WHERE r.disclosure_date BETWEEN ? AND ?
              AND (r.transaction_date IS NULL OR r.transaction_date <= r.disclosure_date)
            ORDER BY r.ticker, r.transaction_date, r.disclosure_date,
                     r.member, r.transaction_type, r.owner_code, r.id
        """,
            [
                map_raw,
                map_disclosure_date,
                map_resolved,
                start_date,
                end_date,
            ],
        ).fetchdf()

        if result.empty:
            return result.drop(
                columns=[c for c in ("resolved_ticker",) if c in result.columns]
            )

        return self._resolve_next_session_entries(
            result, expanded_tickers, alias_tickers
        )

    def _resolve_next_session_entries(
        self,
        result: pd.DataFrame,
        expanded_tickers: list[str],
        alias_tickers: list[str],
    ) -> pd.DataFrame:
        """Replace candidate rows with executable next-session entry prices.

        Outcomes enter on the next expected NYSE session after disclosure
        (see signals core). The entry price is the exact close on that
        session for the resolved ticker — never the last close on or before
        disclosure. Rows without an exact next-session quote have no
        executable entry and are dropped, matching _entry_prices_from_matrix.
        """
        if result.empty:
            return result.drop(
                columns=[
                    c
                    for c in ("resolved_ticker", "entry_price_date")
                    if c in result.columns
                ]
            )

        result = result.copy()
        result["disclosure_date"] = pd.to_datetime(result["disclosure_date"])
        result["entry_price_date"] = pd.to_datetime(
            [next_nyse_session(d) for d in result["disclosure_date"]]
        )

        min_entry = result["entry_price_date"].min()
        max_entry = result["entry_price_date"].max()
        price_rows = self.conn.execute(
            """
            SELECT ticker, date, close
            FROM prices
            WHERE ticker IN (SELECT UNNEST(?))
              AND date BETWEEN ? AND ?
              AND close > 0
              AND isfinite(close)
        """,
            [expanded_tickers, min_entry.date(), max_entry.date()],
        ).fetchdf()

        if price_rows.empty:
            empty = result.iloc[0:0].copy()
            empty["entry_price"] = pd.Series(dtype=float)
            return empty.drop(
                columns=[
                    c
                    for c in ("resolved_ticker", "entry_price_date")
                    if c in empty.columns
                ]
            )

        price_rows["date"] = pd.to_datetime(price_rows["date"]).dt.normalize()
        lookup = {
            (str(t), pd.Timestamp(d).normalize()): float(c)
            for t, d, c in zip(
                price_rows["ticker"], price_rows["date"], price_rows["close"]
            )
            if pd.notna(c)
        }

        alias_set = {str(t).strip().upper() for t in alias_tickers}
        entry_dates = pd.to_datetime(result["entry_price_date"]).dt.normalize()
        entry_prices: list[float | None] = []
        for (_, row), entry_date in zip(result.iterrows(), entry_dates):
            raw = row["ticker"]
            resolved = row["resolved_ticker"]
            if pd.isna(resolved):
                resolved = raw
            if str(raw).strip().upper() in alias_set:
                candidates = [resolved]
            elif str(resolved) == str(raw):
                candidates = [resolved]
            else:
                candidates = [resolved, raw]
            price: float | None = None
            entry_key = pd.Timestamp(entry_date).normalize()
            for cand in candidates:
                hit = lookup.get((str(cand), entry_key))
                if hit is not None:
                    price = hit
                    break
            entry_prices.append(price)

        result["entry_price"] = entry_prices
        result = result[result["entry_price"].notna()]
        if result.empty:
            return result.drop(
                columns=[
                    c
                    for c in ("resolved_ticker", "entry_price_date")
                    if c in result.columns
                ]
            )

        return result.drop(
            columns=[
                c
                for c in ("resolved_ticker", "entry_price_date")
                if c in result.columns
            ]
        )
