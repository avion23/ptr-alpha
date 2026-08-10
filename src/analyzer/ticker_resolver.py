from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime

logger = logging.getLogger(__name__)


def _normalize_trade_date(trade_date):
    """Coerce a date-like transaction date to ``datetime.date``.

    pandas Timestamps arrive from repository reads; NaT and other invalid
    values mean no date is known and must not silently compare as a date.
    """
    if trade_date is None:
        return None
    if isinstance(trade_date, date) and not isinstance(trade_date, datetime):
        return trade_date
    try:
        return trade_date.date()
    except (AttributeError, ValueError, TypeError):
        return None


@dataclass(frozen=True, slots=True)
class TickerResolution:
    raw_ticker: str
    price_symbol: str
    status: str  # verified mapping state; pass-through symbols are unverified
    confidence: float
    notes: str


class TickerResolver:
    """Maps raw congressional disclosure tickers to yfinance-compatible price symbols."""

    # Effective dates are ticker-change dates, not announcement/name-change dates.
    RENAME_MAP: dict[str, tuple[str, str]] = {
        "FB": ("META", "2022-06-09"),
        "SQ": ("XYZ", "2025-01-21"),
        "BLL": ("BALL", "2022-11-01"),
    }

    ACQUISITION_MAP: dict[str, tuple[str, str]] = {
        "ATVI": ("MSFT", "2023-10-13"),
        "CELG": ("BMY", "2019-11-20"),
    }

    CLASS_SHARE_MAP: dict[str, str] = {
        "BRK": "BRK-B",
        "BRKB": "BRK-B",
        "BRK.B": "BRK-B",
        "BRK.A": "BRK-A",
        "BF.B": "BF-B",
        "BF.A": "BF-A",
        "CWEN.A": "CWEN-A",
        "FOXA": "FOXA",
        "GOOG": "GOOG",
        "GOOGL": "GOOGL",
    }

    # Pseudo-tickers are allowed only when a real filing canary establishes the
    # intended public equity. Ambiguous tokens must never borrow another asset's
    # prices merely because the text prefixes look similar.
    PSEUDO_TICKER_MAP: dict[str, str] = {
        "ROBL": "RBLX",  # Verified Roblox Corporation parser artifact.
        "WARN": "WBD",  # Verified Warner Bros. Discovery parser artifact.
    }
    QUARANTINED_TICKERS: frozenset[str] = frozenset(
        {
            "SP",  # Owner code and company-name prefix; not an asset identity.
            "ALLI",  # Observed for both private Alliant Holdings and public ARLP.
            "MATT",  # Observed for a Matthews International mutual fund.
        }
    )
    SUSPICIOUS_TICKERS: frozenset[str] = frozenset(
        {
            "THE",
            "NEW",
            "MARY",
            "NORT",
            "CITI",
            "SOUT",
            "AMER",
            "BANK",
            "DEL",
            "MICH",
            "BERK",
            "WISC",
            "EAST",
            "FUND",
            "KING",
            "LAKE",
            "PORT",
            "TIPS",
        }
    )

    def resolve(
        self, raw_ticker: str, trade_date: date | None = None
    ) -> TickerResolution:
        """Resolve a raw ticker to a yfinance symbol.

        Resolution order:
        1. Class-share dot-to-hyphen mapping
        2. True rename mapping (with date-based validity check)
        3. Acquisition mapping (always evaluate under original symbol)
        4. Pseudo-ticker parser artifact mapping
        5. Pass through as-is (already valid)
        """
        if not raw_ticker:
            return TickerResolution(
                raw_ticker=raw_ticker or "",
                price_symbol=raw_ticker or "",
                status="unresolved",
                confidence=0.0,
                notes="Empty ticker",
            )

        trade_date = _normalize_trade_date(trade_date)
        normalized = raw_ticker.strip().upper()
        if (
            normalized in self.QUARANTINED_TICKERS
            or normalized in self.SUSPICIOUS_TICKERS
        ):
            return TickerResolution(
                raw_ticker=raw_ticker,
                price_symbol=normalized,
                status="quarantined",
                confidence=0.0,
                notes=f"Ambiguous parser token is not eligible for equity strategies: {normalized}",
            )

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

        # 2. True rename mapping
        if normalized in self.RENAME_MAP:
            new_symbol, effective_date_str = self.RENAME_MAP[normalized]
            effective_date = date.fromisoformat(effective_date_str)

            if trade_date is None:
                return TickerResolution(
                    raw_ticker=raw_ticker,
                    price_symbol=normalized,
                    status="unverified",
                    confidence=0.0,
                    notes=(
                        f"Ticker alias {normalized} -> {new_symbol} is temporally "
                        "unverified without transaction_date; pass the transaction date "
                        "before resolving a price symbol"
                    ),
                )
            if trade_date >= effective_date:
                return TickerResolution(
                    raw_ticker=raw_ticker,
                    price_symbol=new_symbol,
                    status="renamed",
                    confidence=1.0,
                    notes=f"Renamed {normalized} -> {new_symbol} on {effective_date_str}",
                )
            return TickerResolution(
                raw_ticker=raw_ticker,
                price_symbol=normalized,
                status="pre_rename",
                confidence=1.0,
                notes=(
                    f"Trade predates {normalized} -> {new_symbol} on "
                    f"{effective_date_str}; using the contemporaneous symbol"
                ),
            )

        # 3. Acquisition mapping. Never substitute the acquirer's equity.
        if normalized in self.ACQUISITION_MAP:
            acquirer, acquisition_date_str = self.ACQUISITION_MAP[normalized]
            acquisition_date = date.fromisoformat(acquisition_date_str)
            if trade_date is None:
                return TickerResolution(
                    raw_ticker=raw_ticker,
                    price_symbol=normalized,
                    status="date_required",
                    confidence=0.0,
                    notes=f"Acquired ticker {normalized} requires a trade date",
                )
            status = "pre_acquisition" if trade_date <= acquisition_date else "acquired"
            confidence = 1.0 if status == "pre_acquisition" else 0.0
            return TickerResolution(
                raw_ticker=raw_ticker,
                price_symbol=normalized,
                status=status,
                confidence=confidence,
                notes=(
                    f"{normalized} acquired by {acquirer} on {acquisition_date_str}; "
                    "the acquirer's prices are never substituted"
                ),
            )

        # 4. Pseudo-ticker mapping (pdftotext parser artifacts)
        if normalized in self.PSEUDO_TICKER_MAP:
            mapped = self.PSEUDO_TICKER_MAP[normalized]
            return TickerResolution(
                raw_ticker=raw_ticker,
                price_symbol=mapped,
                status="pseudo_ticker",
                confidence=0.8,
                notes=f"Pseudo-ticker {normalized} -> {mapped}",
            )

        # 5. Pass through, but do not claim that syntax proves a listed equity.
        return TickerResolution(
            raw_ticker=raw_ticker,
            price_symbol=normalized,
            status="unverified",
            confidence=0.0,
            notes="No authoritative symbol verification is available",
        )

    def is_strategy_eligible(
        self, raw_ticker: str, trade_date: date | None = None
    ) -> bool:
        """Return true only for a positively verified, contemporaneous mapping.

        Ticker syntax is not evidence of a listed equity. Callers handling aliases
        must pass the transaction date; source asset evidence is checked separately.
        """
        resolution = self.resolve(raw_ticker, trade_date)
        return resolution.confidence > 0 and resolution.status in {
            "class_share",
            "renamed",
            "pre_rename",
            "pre_acquisition",
            "pseudo_ticker",
        }

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
