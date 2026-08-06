from __future__ import annotations

import logging
from datetime import date

import duckdb
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


class _NYSEHolidayCalendar(AbstractHolidayCalendar):
    """NYSE trading-day closures.

    ``USFederalHolidayCalendar`` is wrong for equity prices: it includes
    Columbus Day and Veterans Day (regular NYSE trading days) and omits
    Good Friday (an NYSE closure). Building the calendar from the actual
    NYSE closure rules keeps ``get_missing`` honest about missing prices.
    """

    rules = [
        Holiday("New Year's Day", month=1, day=1, observance=nearest_workday),
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
    ]


_NYSE_HOLIDAYS = _NYSEHolidayCalendar()


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
            ORDER BY date
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

        df_reset = df.reset_index().copy()
        index_col_name = df_reset.columns[0]
        prices_long = df_reset.melt(
            id_vars=[index_col_name], var_name="ticker", value_name="close"
        )
        prices_long = prices_long.rename(columns={index_col_name: "date"})
        prices_long = prices_long.dropna(subset=["close"])

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

        business_dates = pd.bdate_range(start_date, end_date)
        holidays = _NYSE_HOLIDAYS.holidays(start=start_date, end=end_date)
        required_dates = business_dates.difference(holidays)
        if required_dates.empty:
            return [], []

        existing = self.conn.execute(
            """
            SELECT DISTINCT ticker, date
            FROM prices
            WHERE ticker IN (SELECT UNNEST(?))
              AND date BETWEEN ? AND ?
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
        resolutions = resolver.resolve_batch(tickers)
        expanded_tickers: list[str] = []
        seen: set[str] = set()
        ticker_map_entries: list[tuple[str, str]] = []
        map_seen: set[str] = set()
        for raw in tickers:
            if raw not in seen:
                seen.add(raw)
                expanded_tickers.append(raw)
            resolved = resolutions[raw].price_symbol
            if raw not in map_seen:
                map_seen.add(raw)
                ticker_map_entries.append((raw, resolved))
            if resolved not in seen:
                seen.add(resolved)
                expanded_tickers.append(resolved)

        raw_tickers = [raw for raw, _ in ticker_map_entries]
        resolved_tickers = [resolved for _, resolved in ticker_map_entries]

        result = self.conn.execute(
            """
            WITH ticker_map(raw, resolved) AS (
                SELECT UNNEST(?), UNNEST(?)
            ),
            resolved_tickers AS (
                SELECT t.*, COALESCE(tm.resolved, t.ticker) AS resolved_ticker
                FROM transactions t
                LEFT JOIN ticker_map tm ON t.ticker = tm.raw
            )
            SELECT r.member, r.ticker, r.disclosure_date, r.transaction_type,
                   r.owner_code, r.amount_midpoint, r.instrument_type, r.strike_price, r.expiry_date,
                   COALESCE(p_res.close, p_raw.close) AS entry_price,
                   COALESCE(p_res.date, p_raw.date) AS entry_price_date
            FROM resolved_tickers r
            ASOF LEFT JOIN prices p_res
              ON r.resolved_ticker = p_res.ticker
              AND p_res.date <= r.disclosure_date
            ASOF LEFT JOIN prices p_raw
              ON r.ticker = p_raw.ticker
              AND p_raw.date <= r.disclosure_date
            WHERE r.ticker IN (SELECT UNNEST(?))
              AND r.disclosure_date BETWEEN ? AND ?
              AND (r.transaction_date IS NULL OR r.transaction_date <= r.disclosure_date)
              AND COALESCE(p_res.close, p_raw.close) IS NOT NULL
        """,
            [raw_tickers, resolved_tickers, expanded_tickers, start_date, end_date],
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
