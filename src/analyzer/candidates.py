"""Shared public-equity candidate selection for live analysis and replay."""

from __future__ import annotations

import logging
import re

import pandas as pd

from analyzer.member_names import canonical_member_key
from analyzer.models import TransactionType
from analyzer.ticker_resolver import TickerResolver

_UNSUPPORTED_ASSET_RE = re.compile(
    r"\b(?:mutual fund|index fund|exchange-traded fund|money market|treasury|"
    r"government securit|corporate bond|municipal bond|real estate|cryptocurrency|"
    r"private equity|limited partnership)\b",
    re.IGNORECASE,
)
_UNSUPPORTED_ASSET_CLASSES = {
    "government securities",
    "other",
    "corporate securities",
    "property/real estate",
    "stock option",
    "option",
    "call",
    "put",
}
_AUTHORITATIVE_TICKER_ORIGINS = {"official", "official_filing", "house", "senate"}
_AUTHORITATIVE_SOURCES = {"house_pdf", "gemini_ocr", "senate_efd"}
logger = logging.getLogger(__name__)


def filter_equity_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Keep normalized stock rows unless explicit metadata says non-equity."""
    if rows.empty:
        return rows
    if "instrument_type" not in rows.columns:
        return rows.iloc[0:0].copy()

    instruments = rows["instrument_type"].fillna("").astype(str).str.strip().str.lower()
    mask = instruments.eq("stock")

    if "raw_asset_class" in rows.columns:
        classes = rows["raw_asset_class"].fillna("").astype(str).str.strip().str.lower()
        mask &= ~classes.isin(_UNSUPPORTED_ASSET_CLASSES)

    descriptions = pd.Series("", index=rows.index, dtype="object")
    for column in ("asset_description", "raw_asset_description"):
        if column in rows.columns:
            descriptions = descriptions.str.cat(
                rows[column].fillna("").astype(str), sep=" "
            )
    mask &= ~descriptions.str.contains(_UNSUPPORTED_ASSET_RE, na=False)
    mask &= ~descriptions.str.contains(
        r"\[(?:OP|OT|GS|GB|MF|OL|CT|HN|OI|RS)\]", case=False, regex=True
    )

    # Economic duplicate candidates are preserved here. Buyer scoring collapses
    # to one latest disclosure per canonical member, so dropping every row in a
    # duplicate group would erase real buyers instead of merely deduplicating.
    eligible = mask
    logger.debug(
        "Equity universe: rows=%d stock=%d eligible=%d",
        len(rows),
        int(instruments.eq("stock").sum()),
        int(eligible.sum()),
    )
    return rows.loc[eligible].copy()


def eligible_candidate_rows(recent_trades: pd.DataFrame) -> pd.DataFrame:
    """Return purchase rows with a strategy-eligible ticker identity."""
    resolver = TickerResolver()
    recent_trades = filter_equity_rows(recent_trades)
    if "transaction_type" in recent_trades.columns:
        recent_trades = recent_trades[
            recent_trades["transaction_type"] == TransactionType.PURCHASE.value
        ].copy()
    logger.debug(
        "Candidate purchases: rows=%d tickers=%d members=%d",
        len(recent_trades),
        recent_trades["ticker"].nunique(dropna=True),
        recent_trades["member"].nunique(dropna=True),
    )
    if "ticker_origin" in recent_trades.columns:
        logger.debug(
            "Candidate ticker origins: %s",
            recent_trades["ticker_origin"].fillna("<null>").value_counts().to_dict(),
        )
    if "source" in recent_trades.columns:
        logger.debug(
            "Candidate sources: %s",
            recent_trades["source"].fillna("<null>").value_counts().to_dict(),
        )
    eligible_indices: list = []
    rejected_alias = 0
    rejected_resolved = 0
    rejected_unverified = 0
    for index, row in recent_trades.iterrows():
        ticker = str(row["ticker"]).strip().upper()
        if ticker in resolver.RENAME_MAP:
            rejected_alias += 1
            continue
        reference_date = None
        date_column = (
            "disclosure_date"
            if ticker in resolver.LISTING_START_MAP
            else "transaction_date"
        )
        if date_column in recent_trades.columns:
            parsed = pd.to_datetime(row.get(date_column), errors="coerce")
            if not pd.isna(parsed):
                reference_date = parsed.date()
        resolution = resolver.resolve(ticker, reference_date)
        if resolver.is_strategy_eligible(ticker, reference_date):
            eligible_indices.append(index)
            continue
        if resolution.status != "unverified":
            rejected_resolved += 1
            continue
        raw_origin = row.get("ticker_origin")
        raw_source = row.get("source")
        origin = "" if pd.isna(raw_origin) else str(raw_origin).strip().lower()
        source = "" if pd.isna(raw_source) else str(raw_source).strip().lower()
        if origin in _AUTHORITATIVE_TICKER_ORIGINS or source in _AUTHORITATIVE_SOURCES:
            eligible_indices.append(index)
        else:
            rejected_unverified += 1

    purchases = recent_trades.loc[eligible_indices].copy()
    logger.debug(
        "Candidate rows: eligible=%d tickers=%d rejected_alias=%d "
        "rejected_resolved=%d rejected_unverified=%d",
        len(purchases),
        purchases["ticker"].nunique(dropna=True),
        rejected_alias,
        rejected_resolved,
        rejected_unverified,
    )
    return purchases


def candidate_tickers(recent_trades: pd.DataFrame, min_buyers: int) -> list[str]:
    """Return strategy-eligible tickers with enough distinct canonical buyers."""
    purchases = eligible_candidate_rows(recent_trades)
    if purchases.empty:
        return []
    purchases["_member_canonical"] = purchases["member"].map(canonical_member_key)
    buyer_counts = purchases.groupby("ticker")["_member_canonical"].nunique()
    return buyer_counts[buyer_counts >= min_buyers].index.tolist()
