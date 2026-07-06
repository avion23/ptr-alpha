"""YFinance-backed price fetcher with cache merge."""

import logging
import re
import time
from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf

from analyzer.database import Database
from analyzer.exceptions import DataSourceError
from analyzer.interfaces import PriceSource
from analyzer.settings import Settings
from analyzer.ticker_resolver import TickerResolver

logger = logging.getLogger(__name__)

_VALID_TICKER_RE = re.compile(r"^[A-Z]{1,5}([.-][A-Z]{1,2})?$")


# ── YFinancePriceSource: yfinance-backed price fetcher with cache merge ──

class YFinancePriceSource(PriceSource):
    def __init__(self, settings: Settings, read_only: bool = False, db: Database | None = None):
        self.settings = settings
        self.data_dir = Path(settings.data.data_dir)
        self._owns_db = db is None
        self.db = db if db is not None else Database(self.data_dir / "congress.duckdb", read_only=read_only)

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
            list(set(t for t in clean_tickers if _VALID_TICKER_RE.match(str(t))) | {"SPY"})
        )

        raw_to_yf, yf_to_raw = _resolve_tickers(all_tickers)
        cached_prices = self.db.get_prices(all_tickers, start, end)
        if not cached_prices.empty:
            logger.info(
                f"Loaded cached prices: {len(cached_prices.columns)} tickers, "
                f"{len(cached_prices)} dates"
            )

        missing_tickers, missing_dates = self.db.get_missing_price_data(
            all_tickers, start, end,
        )

        if not missing_tickers and not missing_dates:
            logger.info(f"Using fully cached prices for {len(all_tickers)} tickers")
            available_tickers = [t for t in all_tickers if t in cached_prices.columns]
            return cached_prices[available_tickers].dropna(axis=1, how="all")

        return self._fetch_and_merge_prices(
            all_tickers, raw_to_yf, yf_to_raw, cached_prices, start, end,
            missing_tickers,
        )

    def _fetch_and_merge_prices(
        self, all_tickers, raw_to_yf, yf_to_raw, cached_prices, start, end,
        missing_tickers,
    ) -> pd.DataFrame:
        """Fetch missing data from yfinance and merge with the cache."""
        fetch_tickers = missing_tickers if missing_tickers else all_tickers
        fetch_resolved = sorted(set(raw_to_yf.get(t, t) for t in fetch_tickers))

        logger.info(
            f"Fetching price data for {len(fetch_resolved)} tickers using yfinance"
        )

        data = self._download_yfinance(fetch_resolved, start, end)
        if data.empty:
            if not cached_prices.empty:
                logger.warning("yfinance failed, using cached data")
                available_tickers = [
                    t for t in all_tickers if t in cached_prices.columns
                ]
                return cached_prices[available_tickers].dropna(axis=1, how="all")
            raise DataSourceError(
                "No price data could be fetched from yfinance. Data source may be blocked or down."
            )

        new_prices = (
            data["Close"]
            if len(fetch_resolved) > 1
            else data["Close"].to_frame(fetch_resolved[0])
        )
        new_prices = new_prices.dropna(axis=1, how="all")
        new_prices = self._rename_yf_columns(new_prices, raw_to_yf)

        if self.db.is_read_only:
            logger.info(
                f"Read-only mode: merging {len(new_prices.columns)} fetched tickers with cache"
            )
            merged = pd.concat([cached_prices, new_prices], axis=1)
            merged = merged.loc[:, ~merged.columns.duplicated(keep="first")]
            prices = merged[~merged.index.duplicated(keep="last")]
        else:
            self.db.upsert_prices(new_prices)
            logger.info(f"Cached {len(new_prices.columns)} tickers to database")
            prices = self.db.get_prices(all_tickers, start, end)

        return _validate_and_log_prices(prices, all_tickers)

    def _download_yfinance(self, fetch_resolved: list[str], start, end) -> pd.DataFrame:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return yf.download(
                    fetch_resolved,
                    start=start,
                    end=end,
                    progress=False,
                    threads=True,
                    auto_adjust=True,
                )
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
        return pd.DataFrame()

    def _rename_yf_columns(self, new_prices: pd.DataFrame, raw_to_yf: dict) -> pd.DataFrame:
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
    """Filter out NaN/None/empty/garbage tickers from the input list."""
    return [t for t in tickers if t and str(t).strip() and str(t) != "nan"]


def _resolve_tickers(all_tickers: list[str]) -> tuple[dict, dict]:
    """Build (raw -> yf, yf -> raw) mapping via TickerResolver."""
    resolver = TickerResolver()
    resolutions = resolver.resolve_batch(all_tickers)
    raw_to_yf = {r.raw_ticker: r.price_symbol for r in resolutions.values()}
    yf_to_raw: dict[str, str] = {}
    for raw, sym in raw_to_yf.items():
        if sym not in yf_to_raw:
            yf_to_raw[sym] = raw
    for r in resolutions.values():
        if r.status != "valid":
            logger.info(f"Ticker resolution: {r.notes}")
    return raw_to_yf, yf_to_raw


def _validate_and_log_prices(prices: pd.DataFrame, all_tickers: list[str]) -> pd.DataFrame:
    """Fail loudly when too many tickers couldn't be fetched (>25%)."""
    failed_tickers = sorted(set(all_tickers) - set(prices.columns))
    success_count = len([t for t in all_tickers if t in prices.columns])
    success_rate = success_count / len(all_tickers)

    if success_rate < 0.75:
        raise DataSourceError(
            f"Price fetch failure rate too high: {(1 - success_rate) * 100:.1f}% failed "
            f"({len(failed_tickers)}/{len(all_tickers)}). Analysis would be unreliable."
        )
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
    available_tickers = [t for t in all_tickers if t in prices.columns]
    return prices[available_tickers].dropna(axis=1, how="all")
