from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

logger = logging.getLogger(__name__)


@dataclass
class TickerResolution:
    raw_ticker: str
    price_symbol: str
    status: str  # valid, renamed, class_share, delisted, unresolved
    confidence: float
    notes: str


class TickerResolver:
    """Maps raw congressional disclosure tickers to yfinance-compatible price symbols."""

    RENAME_MAP: dict[str, tuple[str, str]] = {
        "FB": ("META", "2021-10-28"),
        "SQ": ("XYZ", "2024-02-01"),
        "ATVI": ("MSFT", "2023-10-13"),
        "CELG": ("BMY", "2019-11-20"),
        "BLL": ("AMCR", "2019-04-11"),
    }

    CLASS_SHARE_MAP: dict[str, str] = {
        "BRK.B": "BRK-B",
        "BRK.A": "BRK-A",
        "BF.B": "BF-B",
        "BF.A": "BF-A",
        "CWEN.A": "CWEN-A",
        "FOXA": "FOXA",
        "GOOG": "GOOG",
        "GOOGL": "GOOGL",
    }

    def resolve(self, raw_ticker: str, trade_date: date | None = None) -> TickerResolution:
        """Resolve a raw ticker to a yfinance symbol.

        Resolution order:
        1. Class-share dot-to-hyphen mapping
        2. Rename/acquisition mapping (with date-based validity check)
        3. Pass through as-is (already valid)
        """
        if not raw_ticker:
            return TickerResolution(
                raw_ticker=raw_ticker or "",
                price_symbol=raw_ticker or "",
                status="unresolved",
                confidence=0.0,
                notes="Empty ticker",
            )

        normalized = raw_ticker.strip().upper()

        # 1. Class-share mapping
        if normalized in self.CLASS_SHARE_MAP:
            mapped = self.CLASS_SHARE_MAP[normalized]
            return TickerResolution(
                raw_ticker=raw_ticker,
                price_symbol=mapped,
                status="class_share",
                confidence=1.0,
                notes=f"Class-share variant: {normalized} -> {mapped}",
            )

        # 2. Rename/acquisition mapping
        if normalized in self.RENAME_MAP:
            new_symbol, effective_date_str = self.RENAME_MAP[normalized]
            effective_date = date.fromisoformat(effective_date_str)

            if trade_date is None or trade_date >= effective_date:
                return TickerResolution(
                    raw_ticker=raw_ticker,
                    price_symbol=new_symbol,
                    status="renamed",
                    confidence=1.0,
                    notes=f"Renamed {normalized} -> {new_symbol} on {effective_date_str}",
                )
            else:
                return TickerResolution(
                    raw_ticker=raw_ticker,
                    price_symbol=normalized,
                    status="delisted",
                    confidence=0.3,
                    notes=(
                        f"Ticker {normalized} was renamed to {new_symbol} on "
                        f"{effective_date_str}, but trade date {trade_date} is before "
                        f"that. Using original symbol."
                    ),
                )

        # 3. Already valid — pass through
        return TickerResolution(
            raw_ticker=raw_ticker,
            price_symbol=normalized,
            status="valid",
            confidence=1.0,
            notes="No transformation needed",
        )

    def resolve_batch(
        self, tickers: list[str], trade_date: date | None = None
    ) -> dict[str, TickerResolution]:
        """Resolve multiple tickers."""
        return {t: self.resolve(t, trade_date) for t in tickers}

    def get_yfinance_tickers(
        self, tickers: list[str], trade_date: date | None = None
    ) -> list[str]:
        """Return deduplicated list of yfinance-compatible symbols for given raw tickers."""
        resolutions = self.resolve_batch(tickers, trade_date)
        seen: set[str] = set()
        result: list[str] = []
        for r in resolutions.values():
            if r.price_symbol not in seen:
                seen.add(r.price_symbol)
                result.append(r.price_symbol)
        return sorted(result)
