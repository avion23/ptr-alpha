"""Sector data fetching for ticker analysis."""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def load_sector_data(tickers: list[str]) -> pd.DataFrame:
    """Load sector info for tickers using yfinance."""
    try:
        import yfinance as yf
        from concurrent.futures import ThreadPoolExecutor, as_completed
    except ImportError:
        logger.debug("yfinance not available, skipping sector data")
        return pd.DataFrame(columns=["ticker", "sector"])

    filtered = [t for t in tickers if t not in ("SPY", "SP500")]
    if not filtered:
        return pd.DataFrame(columns=["ticker", "sector"])

    def fetch_sector(ticker):
        try:
            return ticker, yf.Ticker(ticker).info.get("sector", "Unknown")
        except Exception as e:
            logger.debug("Failed to fetch sector for %s: %s", ticker, e)
            return ticker, "Unknown"

    records = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(fetch_sector, t): t for t in filtered}
        try:
            for future in as_completed(futures, timeout=30):
                try:
                    ticker, sector = future.result(timeout=5)
                except Exception as e:
                    logger.debug("Timeout fetching sector for %s: %s", futures[future], e)
                    records.append({"ticker": futures[future], "sector": "Unknown"})
                    continue
                records.append({"ticker": ticker, "sector": sector})
        except TimeoutError:
            for f, t in futures.items():
                if not f.done():
                    logger.debug("Outer timeout: sector lookup for %s incomplete", t)
                    records.append({"ticker": t, "sector": "Unknown"})

    records = [r for r in records if r["sector"] != "Unknown"]
    return pd.DataFrame(records)
