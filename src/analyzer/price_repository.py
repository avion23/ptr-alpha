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
        # and repeatedly downloaded.
        Holiday(
            "National Day of Mourning for Jimmy Carter",
            month=1,
            day=9,
            start_date="2025-01-09",
            end_date="2025-01-09",
        ),
    ]


_NYSE_HOLIDAYS = _NYSEHolidayCalendar()


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
    sessions = nyse_sessions(day_ts + pd.Timedelta(days=1), day_ts + pd.Timedelta(days=14))
    if sessions.empty:
        raise ValueError(f"No NYSE session found after {day_ts.date()}")
    return pd.Timestamp(sessions[0]).value


def next_nyse_session(day) -> pd.Timestamp:
    day_ts = _normalize_session_date(day)
    return pd.Timestamp(_next_nyse_session_ns(day_ts.value))


@lru_cache(maxsize=8192)
def _previous_nyse_session_ns(day_ns: int) -> int:
    day_ts = pd.Timestamp(day_ns)
    sessions = nyse_sessions(day_ts - pd.Timedelta(days=14), day_ts)
    if sessions.empty:
        raise ValueError(f"No NYSE session found on or before {day_ts.date()}")
    return pd.Timestamp(sessions[-1]).value


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

        start_cutoff = pd.Timestamp(start_date) + pd.Timedelta(days=7)
        ticker_starts = existing.groupby("ticker")["date"].min()
        insufficient: list[str] = []
        for t in tickers:
            if t in ticker_starts.index:
                val = ticker_starts.loc[t]
                if pd.notna(val) and pd.Timestamp(val) > start_cutoff:
                    insufficient.append(t)

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
            if gaps or ticker in insufficient:
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
        max_staleness_days: int = 30,
        resolver: TickerResolver | None = None,
    ) -> pd.DataFrame:
        if not tickers:
            return pd.DataFrame()

        if resolver is None:
            resolver = TickerResolver()

        # Rename aliases (FB -> META, SQ -> XYZ, BLL -> BALL) map to different
        # price symbols depending on when the trade happened, so resolution is
        # per (ticker, transaction_date) pair, never per raw ticker alone.
        pairs = self.conn.execute(
            """
            SELECT DISTINCT ticker, transaction_date
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
        map_tx_date: list[object] = []
        map_resolved: list[str] = []
        expanded_tickers: list[str] = []
        seen: set[str] = set()
        unresolved_alias_rows = 0
        for _, pair in pairs.iterrows():
            raw = pair["ticker"]
            tx_date = pair["transaction_date"]
            if tx_date is None or pd.isna(tx_date):
                tx_date = None
            normalized = str(raw).strip().upper()
            if normalized in alias_tickers and tx_date is None:
                # No-date alias calls fail explicit unverified: never price a
                # rename alias under the raw symbol without a transaction date.
                unresolved_alias_rows += 1
                continue
            resolved = resolver.resolve(raw, tx_date).price_symbol
            map_raw.append(raw)
            map_tx_date.append(tx_date)
            map_resolved.append(resolved)
            for t in (raw, resolved):
                if t not in seen:
                    seen.add(t)
                    expanded_tickers.append(t)

        if unresolved_alias_rows:
            logger.warning(
                "Excluded %d transactions for rename aliases without "
                "transaction_date: alias resolution is unverified without it",
                unresolved_alias_rows,
            )

        if not map_raw:
            # Every matching transaction was an excluded no-date alias; the
            # empty UNNEST arrays would also break DuckDB type inference.
            return pd.DataFrame()

        result = self.conn.execute(
            """
            WITH ticker_map(raw, tx_date, resolved) AS (
                SELECT UNNEST(?), UNNEST(?), UNNEST(?)
            ),
            resolved_tickers AS (
                SELECT t.*, COALESCE(tm.resolved, t.ticker) AS resolved_ticker
                FROM canonical_transactions t
                LEFT JOIN ticker_map tm
                  ON t.ticker = tm.raw
                 AND t.transaction_date IS NOT DISTINCT FROM tm.tx_date
            )
            SELECT r.member, r.ticker, r.transaction_date, r.disclosure_date,
                   r.transaction_type, r.owner_code, r.amount_midpoint,
                   r.instrument_type, r.strike_price, r.expiry_date,
                   CASE WHEN UPPER(r.ticker) IN (SELECT UNNEST(?))
                        THEN p_res.close
                        ELSE COALESCE(p_res.close, p_raw.close)
                   END AS entry_price,
                   CASE WHEN UPPER(r.ticker) IN (SELECT UNNEST(?))
                        THEN p_res.date
                        ELSE COALESCE(p_res.date, p_raw.date)
                   END AS entry_price_date
            FROM resolved_tickers r
            ASOF LEFT JOIN prices p_res
              ON r.resolved_ticker = p_res.ticker
              AND p_res.date <= r.disclosure_date
              AND p_res.close > 0
              AND isfinite(p_res.close)
            ASOF LEFT JOIN prices p_raw
              ON r.ticker = p_raw.ticker
              AND p_raw.date <= r.disclosure_date
              AND p_raw.close > 0
              AND isfinite(p_raw.close)
            WHERE r.ticker IN (SELECT UNNEST(?))
              AND r.disclosure_date BETWEEN ? AND ?
              AND NOT (UPPER(r.ticker) IN (SELECT UNNEST(?))
                       AND r.transaction_date IS NULL)
              AND (r.transaction_date IS NULL OR r.transaction_date <= r.disclosure_date)
              AND CASE WHEN UPPER(r.ticker) IN (SELECT UNNEST(?))
                       THEN p_res.close
                       ELSE COALESCE(p_res.close, p_raw.close)
                  END IS NOT NULL
            ORDER BY r.ticker, r.transaction_date, r.disclosure_date,
                     r.member, r.transaction_type, r.owner_code, r.id
        """,
            [
                map_raw,
                map_tx_date,
                map_resolved,
                alias_tickers,
                alias_tickers,
                expanded_tickers,
                start_date,
                end_date,
                alias_tickers,
                alias_tickers,
            ],
        ).fetchdf()

        if not result.empty:
            result["entry_price_date"] = pd.to_datetime(result["entry_price_date"])
            if max_staleness_days is not None:
                result["disclosure_date"] = pd.to_datetime(result["disclosure_date"])
                staleness = (
                    result["disclosure_date"] - result["entry_price_date"]
                ).dt.days
                result = result[staleness <= max_staleness_days]
            result = result.drop(columns=["entry_price_date"])

        return result
