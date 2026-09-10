"""YFinance-backed price fetcher with cache merge."""

import logging
import re
import time
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from analyzer._price_index import _normalize_price_index
from analyzer.database import Database
from analyzer.exceptions import DataSourceError
from analyzer.settings import Settings
from analyzer.ticker_resolver import TickerResolver

logger = logging.getLogger(__name__)

_VALID_TICKER_RE = re.compile(r"^[A-Z]{1,5}([.-][A-Z]{1,2})?$")


# ── YFinancePriceSource: yfinance-backed price fetcher with cache merge ──


class YFinancePriceSource:
    def __init__(
        self, settings: Settings, read_only: bool = False, db: Database | None = None
    ):
        self.settings = settings
        self.data_dir = Path(settings.data.data_dir)
        self._owns_db = db is None
        self.db = (
            db
            if db is not None
            else Database(self.data_dir / "congress.duckdb", read_only=read_only)
        )

    def close(self) -> None:
        if self._owns_db:
            self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        return False

    def get_prices(self, tickers: list[str], start: date, end: date) -> pd.DataFrame:
        if len(tickers) == 0:
            raise DataSourceError("No tickers provided for price fetching")

        clean_tickers = _clean_tickers(tickers)
        all_tickers = sorted(
            list(
                set(
                    t
                    for t in _expand_rename_aliases(clean_tickers)
                    if _VALID_TICKER_RE.match(str(t))
                )
                | {"SPY"}
            )
        )

        raw_to_yf, yf_to_raw = _resolve_tickers(all_tickers)
        cached_prices = self.db.get_prices(all_tickers, start, end)
        if not cached_prices.empty:
            logger.info(
                f"Loaded cached prices: {len(cached_prices.columns)} tickers, "
                f"{len(cached_prices)} dates"
            )

        missing_tickers, missing_dates = self.db.get_missing_price_data(
            all_tickers,
            start,
            end,
        )

        if not missing_tickers and not missing_dates:
            logger.info(f"Using fully cached prices for {len(all_tickers)} tickers")
            return _validate_and_log_prices(cached_prices, all_tickers)

        return self._fetch_and_merge_prices(
            all_tickers,
            raw_to_yf,
            yf_to_raw,
            cached_prices,
            start,
            end,
            missing_tickers,
            missing_dates,
        )

    def _fetch_and_merge_prices(
        self,
        all_tickers,
        raw_to_yf,
        yf_to_raw,
        cached_prices,
        start,
        end,
        missing_tickers,
        missing_dates,
    ) -> pd.DataFrame:
        """Fetch missing data from yfinance and merge with the cache.

        A ticker that needs any data is re-fetched over the **full** analysis
        window. yfinance ``auto_adjust`` prices are only self-consistent within
        a single download; a narrow gap repair would mix a stale pre-action
        cached basis with the post-action basis after a split. Re-fetching the
        full window keeps every ticker on one adjustment basis. Because
        ``get_missing`` uses the NYSE trading-day calendar, complete tickers are
        never flagged, so this full-window path runs only for genuinely
        incomplete tickers.
        """
        fetch_tickers = missing_tickers if missing_tickers else all_tickers
        fetch_resolved = sorted({raw_to_yf.get(t, t) for t in fetch_tickers})

        logger.info(
            f"Fetching price data for {len(fetch_resolved)} tickers using yfinance"
        )

        data = self._download_yfinance(fetch_resolved, start, end)
        if data.empty:
            if not cached_prices.empty:
                logger.warning("yfinance failed, using cached data")
                return _validate_and_log_prices(cached_prices, all_tickers)
            raise DataSourceError(
                "No price data could be fetched from yfinance. Data source may be blocked or down."
            )

        new_prices = self._extract_close_prices(data, fetch_resolved)
        new_prices = self._normalize_price_index(new_prices)
        new_prices = new_prices.apply(pd.to_numeric, errors="coerce")
        invalid_mask = new_prices.notna() & (
            ~np.isfinite(new_prices) | new_prices.le(0)
        )
        invalid = invalid_mask.sum().sum()
        new_prices = new_prices.mask(invalid_mask)
        if invalid:
            logger.warning(
                "Rejected %d non-finite or non-positive fetched prices", invalid
            )
        new_prices = new_prices.dropna(axis=1, how="all")
        new_prices = _as_frame(new_prices)
        new_prices = self._rename_yf_columns(new_prices, raw_to_yf)

        if self.db.is_read_only:
            logger.info(
                f"Read-only mode: merging {len(new_prices.columns)} fetched tickers with cache"
            )
            # ``concat`` followed by duplicate-column removal discards the
            # fetched column entirely whenever a partial cached column exists.
            # Prefer freshly fetched observations while retaining cached dates
            # and tickers that were not fetched in this read-only session.
            prices = new_prices.combine_first(cached_prices)
        else:
            self.db.upsert_prices(new_prices)
            logger.info(f"Cached {len(new_prices.columns)} tickers to database")
            prices = self.db.get_prices(all_tickers, start, end)

        return _validate_and_log_prices(prices, all_tickers)

    def _download_yfinance(self, fetch_resolved: list[str], start, end) -> pd.DataFrame:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # yfinance treats ``end`` as exclusive while this public API
                # and the repository treat it as inclusive.
                download_end = pd.Timestamp(end) + timedelta(days=1)
                data = yf.download(
                    fetch_resolved,
                    start=start,
                    end=download_end,
                    progress=False,
                    threads=True,
                    auto_adjust=True,
                )
                if not isinstance(data, pd.DataFrame):
                    raise DataSourceError("yfinance returned no downloadable frame")
                return data
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = 2 ** (attempt + 1)
                    logger.warning(
                        f"yfinance request failed (attempt {attempt + 1}/{max_retries}: {e}), "
                        f"retrying in {delay}s"
                    )
                    time.sleep(delay)
                else:
                    logger.warning(
                        f"yfinance request failed after {max_retries} attempts ({e}), "
                        "falling back to cached data"
                    )
                    return pd.DataFrame()
        raise DataSourceError("yfinance download retries exhausted")

    @staticmethod
    def _extract_close_prices(
        data: pd.DataFrame, fetch_resolved: list[str]
    ) -> pd.DataFrame:
        """Normalize yfinance Close output for one or many symbols."""
        try:
            close = data["Close"]
        except KeyError as exc:
            raise DataSourceError(
                "yfinance response did not contain Close prices"
            ) from exc
        if isinstance(close, pd.Series):
            column = fetch_resolved[0] if len(fetch_resolved) == 1 else str(close.name)
            return close.to_frame(column)
        if not isinstance(close, pd.DataFrame):
            raise DataSourceError("Unsupported yfinance Close response shape")
        if len(fetch_resolved) == 1 and len(close.columns) == 1:
            return close.rename(columns={str(close.columns[0]): fetch_resolved[0]})
        return close.copy()

    @staticmethod
    def _normalize_price_index(prices: pd.DataFrame) -> pd.DataFrame:
        return _normalize_price_index(
            prices,
            invalid_error=DataSourceError,
            duplicate_error=DataSourceError,
            duplicate_message="yfinance returned duplicate calendar dates",
        )

    def _rename_yf_columns(
        self, new_prices: pd.DataFrame, raw_to_yf: dict
    ) -> pd.DataFrame:
        """Rename yf-symbol columns back to their raw tickers so downstream
        consumers see consistent identifiers across sources."""
        yf_to_raws: dict[str, list[str]] = {}
        for raw, sym in raw_to_yf.items():
            yf_to_raws.setdefault(sym, []).append(raw)
        # Only rename when a yf symbol maps to exactly one raw ticker (avoid collision).
        rename_map = {
            sym: raws[0]
            for sym, raws in yf_to_raws.items()
            if sym in new_prices.columns and len(raws) == 1
        }
        return new_prices.rename(columns=rename_map)


# ── Helpers ──


def _clean_tickers(tickers: list[str]) -> list[str]:
    """Filter empty values and known parser-token quarantines."""
    resolver = TickerResolver()
    clean: list[str] = []
    quarantined: list[str] = []
    for ticker in tickers:
        if not ticker or not str(ticker).strip() or str(ticker) == "nan":
            continue
        if resolver.resolve(str(ticker)).status == "quarantined":
            quarantined.append(str(ticker).strip().upper())
            continue
        clean.append(ticker)
    if quarantined:
        logger.debug(
            "Excluded %d quarantined ticker tokens from price fetch: %s",
            len(quarantined),
            ", ".join(sorted(set(quarantined))),
        )
    return clean


def _expand_rename_aliases(tickers: list[str]) -> set[str]:
    """Expand rename aliases into both temporal price symbols.

    A price acquisition window spans many disclosure dates, so a rename alias
    can require both its pre-rename and post-rename market symbols. Fetch both
    series and let per-row entry-price resolution choose the symbol tradable at
    public disclosure time.
    """
    resolver = TickerResolver()
    expanded: set[str] = set()
    for t in tickers:
        normalized = str(t).strip().upper()
        if normalized in resolver.RENAME_MAP:
            new_symbol, _ = resolver.RENAME_MAP[normalized]
            expanded.add(t)
            expanded.add(new_symbol)
        else:
            expanded.add(t)
    return expanded


def _resolve_tickers(all_tickers: list[str]) -> tuple[dict, dict]:
    """Build (raw -> yf, yf -> raw) mapping via TickerResolver."""
    resolver = TickerResolver()
    resolutions = resolver.resolve_batch(all_tickers)
    raw_to_yf = {r.raw_ticker: r.price_symbol for r in resolutions.values()}
    yf_to_raw: dict[str, str] = {}
    for raw, sym in raw_to_yf.items():
        if sym not in yf_to_raw:
            yf_to_raw[sym] = raw
    status_counts = Counter(r.status for r in resolutions.values())
    logger.debug("Ticker resolution summary: %s", dict(sorted(status_counts.items())))
    for resolution in resolutions.values():
        if resolution.status not in {"unverified", "class_share"}:
            logger.debug("Ticker resolution: %s", resolution.notes)
    return raw_to_yf, yf_to_raw


def _validate_and_log_prices(
    prices: pd.DataFrame, all_tickers: list[str]
) -> pd.DataFrame:
    """Quarantine invalid observations and report unresolved tickers."""
    prices = _as_frame(prices.apply(pd.to_numeric, errors="coerce"))
    invalid_mask = prices.notna() & (~np.isfinite(prices) | prices.le(0))
    if invalid_mask.any().any():
        logger.warning(
            "Quarantined %d invalid cached price observations",
            invalid_mask.sum().sum(),
        )
        prices = prices.mask(invalid_mask)
    prices = prices.dropna(axis=1, how="all")

    failed_tickers = sorted(set(all_tickers) - set(prices.columns))
    success_count = len([t for t in all_tickers if t in prices.columns])
    success_rate = success_count / len(all_tickers)

    if failed_tickers:
        logger.warning(
            f"Failed to fetch price data for {len(failed_tickers)} tickers: "
            f"{', '.join(failed_tickers[:10])}"
            f"{'...' if len(failed_tickers) > 10 else ''}"
        )

    logger.info(
        f"Successfully fetched prices for {success_count}/{len(all_tickers)} "
        f"tickers ({success_rate * 100:.1f}% success)"
    )
    return _select_columns(prices, all_tickers)


def _select_columns(prices: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Column-select in ticker order and drop all-empty columns."""
    available = [t for t in tickers if t in prices.columns]
    return _as_frame(prices[available]).dropna(axis=1, how="all")


def _as_frame(value: object) -> pd.DataFrame:
    """Narrow pandas ``apply`` results that stubs type as a union."""
    if not isinstance(value, pd.DataFrame):
        raise DataSourceError("unexpected price-table shape")
    return value
