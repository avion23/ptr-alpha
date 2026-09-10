"""Production consensus scoring from public congressional disclosures."""

from __future__ import annotations

import re

import pandas as pd

from analyzer._memo import df_memoize
from analyzer.exceptions import AnalysisError
from analyzer.member_names import canonical_member_key
from analyzer.models import TransactionType
from analyzer.ticker_resolver import TickerResolver

CONSENSUS_SCORER_PROVENANCE = "identity_free_distinct_buyer_count_v2"
CONSENSUS_LOOKBACK_DAYS = 28
CONSENSUS_MIN_BUYERS = 3

_VALID_TICKER_RE = re.compile(r"^[A-Z]{1,5}(?:[.-][A-Z]{1,2})?$")
_TICKER_RESOLVER = TickerResolver()
_REJECTED_TICKER_STATUSES = frozenset({"unresolved", "quarantined", "acquired"})
_NON_EQUITY_TICKERS = frozenset(
    {
        "BOND",
        "BONDS",
        "CASH",
        "COUPON",
        "FUND",
        "NOTE",
        "NOTES",
        "STOCK",
        "TICKER",
    }
)
_NON_EQUITY_INSTRUMENTS = frozenset(
    {
        "bond",
        "bonds",
        "call",
        "cash",
        "fund",
        "mutual fund",
        "note",
        "notes",
        "option",
        "put",
        "stock option",
        "treasury",
    }
)
_REJECTED_TICKER_ORIGINS = frozenset({"invalid", "missing", "non_equity"})
_OFFICIAL_SOURCES = frozenset({"house_pdf", "gemini_ocr", "senate_efd"})
_UNSUPPORTED_ASSET_RE = re.compile(
    r"\b(?:mutual fund|index fund|exchange-traded fund|money market|treasury|"
    r"government securit|corporate bond|municipal bond|real estate|cryptocurrency|"
    r"private equity|limited partnership|stock\s*option|option\s*type)\b",
    re.IGNORECASE,
)


@df_memoize(copy=False)
def score_ticker_by_buyers(
    ticker: str,
    transactions_df: pd.DataFrame,
    min_buyers: int = CONSENSUS_MIN_BUYERS,
    *,
    as_of_date: pd.Timestamp,
) -> pd.DataFrame:
    """Return the identity-free distinct-buyer production score."""
    as_of = pd.Timestamp(as_of_date)
    if pd.isna(as_of):
        raise AnalysisError("consensus as_of_date must be a valid timestamp")

    normalized_ticker = _validate_ticker(ticker)
    ticker_trades = _get_consensus_ticker_purchases(normalized_ticker, transactions_df)
    if ticker_trades.empty:
        return _empty_ticker_result(normalized_ticker)

    disclosure_dates = pd.to_datetime(ticker_trades["disclosure_date"], errors="coerce")
    ticker_trades = ticker_trades[
        disclosure_dates.notna() & (disclosure_dates <= as_of)
    ].copy()
    if ticker_trades.empty:
        return _empty_ticker_result(normalized_ticker)

    ticker_trades["_member_canonical"] = ticker_trades["member"].map(
        _canonical_member_or_blank
    )
    ticker_trades = ticker_trades[ticker_trades["_member_canonical"].ne("")].copy()
    if ticker_trades.empty:
        return _empty_ticker_result(normalized_ticker)

    buyers = sorted(ticker_trades["_member_canonical"].unique())
    if len(buyers) < min_buyers:
        return _below_threshold_result(normalized_ticker, len(buyers), min_buyers)

    return _consensus_result(
        normalized_ticker,
        buyers,
        ticker_trades,
        as_of_date=as_of,
    )


def _validate_ticker(ticker: str) -> str:
    normalized = _normalize_ticker_text(ticker)
    if normalized is None or not _VALID_TICKER_RE.fullmatch(normalized):
        raise AnalysisError(f"Invalid equity ticker for scoring: {ticker!r}")
    if normalized in _NON_EQUITY_TICKERS:
        raise AnalysisError(f"Non-equity ticker cannot be scored: {ticker!r}")

    resolution = _TICKER_RESOLVER.resolve(normalized)
    # Ordinary source symbols (for example AAPL) are intentionally returned
    # as ``unverified`` by the resolver because it has no exchange-wide symbol
    # registry. Syntax-valid pass-through symbols remain usable; only its
    # explicit invalid/quarantine states are rejected here.
    if resolution.status in _REJECTED_TICKER_STATUSES:
        raise AnalysisError(
            f"Ticker {ticker!r} is not an eligible equity symbol: {resolution.notes}"
        )
    return normalized


def _normalize_ticker_text(value) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            return None
    normalized = str(value).strip().upper()
    if not normalized or normalized in {"NAN", "NAT", "NONE", "<NA>"}:
        return None
    return normalized


def _ticker_family_symbols(ticker: str) -> set[str]:
    """Return resolver-normalized symbols that identify the same equity."""
    resolution = _TICKER_RESOLVER.resolve(ticker)
    symbols = {resolution.price_symbol}

    for alias, (renamed, _) in _TICKER_RESOLVER.RENAME_MAP.items():
        if ticker in {alias, renamed} or resolution.price_symbol in {alias, renamed}:
            symbols.update({alias, renamed})

    for mapping in (
        _TICKER_RESOLVER.CLASS_SHARE_MAP,
        _TICKER_RESOLVER.PSEUDO_TICKER_MAP,
    ):
        for alias, mapped in mapping.items():
            if ticker in {alias, mapped} or resolution.price_symbol == mapped:
                symbols.update({alias, mapped})
    return symbols


def _get_consensus_ticker_purchases(
    ticker: str, transactions_df: pd.DataFrame
) -> pd.DataFrame:
    """Select purchases for a resolver-normalized equity identity.

    Consensus is live-facing and must not let source aliases, malformed rows,
    or non-equity provenance turn into buyer counts. Historical modes retain
    their existing exact-ticker lookup because their member effects are only
    descriptive.
    """
    purchases = _prepare_consensus_purchases(transactions_df)
    if purchases.empty:
        return purchases

    family_symbols = _ticker_family_symbols(ticker)
    return purchases.loc[purchases["_resolved_symbol"].isin(list(family_symbols))].drop(
        columns=["_resolved_symbol", "_member_canonical"]
    )


def _prepare_consensus_purchases(transactions_df: pd.DataFrame) -> pd.DataFrame:
    """Return valid purchase rows with resolver and member identities attached."""
    required = {"transaction_type", "transaction_date", "disclosure_date"}
    if transactions_df.empty or not required.issubset(transactions_df.columns):
        return transactions_df.iloc[0:0].copy()
    purchases = transactions_df[
        transactions_df["transaction_type"] == TransactionType.PURCHASE.value
    ].copy()
    if purchases.empty:
        return purchases

    transaction_dates = pd.to_datetime(purchases["transaction_date"], errors="coerce")
    disclosure_dates = pd.to_datetime(purchases["disclosure_date"], errors="coerce")
    purchases = purchases.loc[
        transaction_dates.notna()
        & disclosure_dates.notna()
        & (transaction_dates <= disclosure_dates)
    ].copy()
    if purchases.empty:
        return purchases

    resolved_symbols = []
    canonical_members = []
    for _, row in purchases.iterrows():
        if not _equity_transaction_row(row):
            resolved_symbols.append(None)
        else:
            ticker = _normalize_ticker_text(row.get("ticker"))
            date_field = (
                "disclosure_date"
                if ticker in _TICKER_RESOLVER.LISTING_START_MAP
                else "transaction_date"
            )
            reference_date = _transaction_date(row.get(date_field))
            resolved_symbols.append(
                _resolved_ticker_symbol(row.get("ticker"), reference_date)
            )
        canonical_members.append(_canonical_member_or_blank(row.get("member")))

    purchases["_resolved_symbol"] = resolved_symbols
    purchases["_member_canonical"] = canonical_members
    return purchases.loc[
        purchases["_resolved_symbol"].notna() & purchases["_member_canonical"].ne("")
    ].copy()


def _resolve_consensus_ticker(ticker: str, as_of_date: pd.Timestamp) -> str:
    """Return the tradable symbol for one equity identity at decision time."""
    normalized = _validate_ticker(ticker)
    decision_date = _transaction_date(as_of_date)
    if decision_date is None:
        raise AnalysisError("Consensus ticker resolution requires a valid as-of date")

    family_symbols = _ticker_family_symbols(normalized)
    # Rename aliases are the only families whose tradable symbol changes over
    # time. Resolve the old alias at decision time so pre-rename replays trade
    # the old symbol and later/delayed filings trade the renamed symbol.
    for alias in _TICKER_RESOLVER.RENAME_MAP:
        if alias in family_symbols:
            resolution = _TICKER_RESOLVER.resolve(alias, decision_date)
            if resolution.status in _REJECTED_TICKER_STATUSES or resolution.status in {
                "date_required",
                "pre_listing",
            }:
                raise AnalysisError(
                    f"Ticker {ticker!r} is not tradable at {decision_date}: "
                    f"{resolution.notes}"
                )
            return resolution.price_symbol

    resolution = _TICKER_RESOLVER.resolve(normalized, decision_date)
    if resolution.status in _REJECTED_TICKER_STATUSES or resolution.status in {
        "date_required",
        "pre_listing",
    }:
        raise AnalysisError(
            f"Ticker {ticker!r} is not tradable at {decision_date}: {resolution.notes}"
        )
    return resolution.price_symbol


def _get_consensus_candidate_tickers(
    transactions_df: pd.DataFrame,
    min_buyers: int,
    *,
    as_of_date: pd.Timestamp,
) -> list[str]:
    """Return one decision-time symbol per equity identity meeting the buyer gate."""
    purchases = _prepare_consensus_purchases(transactions_df)
    if purchases.empty:
        return []

    families: dict[frozenset[str], pd.DataFrame] = {}
    for resolved_symbol in purchases["_resolved_symbol"].dropna().astype(str).unique():
        family = frozenset(_ticker_family_symbols(resolved_symbol))
        if family not in families:
            families[family] = purchases[
                purchases["_resolved_symbol"].isin(family)
            ]

    candidates: set[str] = set()
    for family, family_rows in families.items():
        if family_rows["_member_canonical"].nunique() < min_buyers:
            continue
        try:
            seed = next(
                alias for alias in _TICKER_RESOLVER.RENAME_MAP if alias in family
            )
        except StopIteration:
            seed = sorted(family)[0]
        try:
            candidates.add(_resolve_consensus_ticker(seed, as_of_date))
        except AnalysisError:
            continue
    return sorted(candidates)


def _get_consensus_price_tickers(transactions_df: pd.DataFrame) -> list[str]:
    """Return the price symbols needed to evaluate eligible purchases.

    Class-share and pseudo-ticker spellings collapse to their resolved price
    symbol. Rename aliases retain both temporal symbols because a disclosure
    can arrive after the rename for a transaction executed before it.
    """
    purchases = _prepare_consensus_purchases(transactions_df)
    if purchases.empty:
        return []
    symbols = set(purchases["_resolved_symbol"].dropna().astype(str))
    for symbol in tuple(symbols):
        family = _ticker_family_symbols(symbol)
        for alias, (renamed, _) in _TICKER_RESOLVER.RENAME_MAP.items():
            if alias in family:
                symbols.update({alias, renamed})
    return sorted(symbols)


def _resolved_ticker_symbol(value, trade_date=None) -> str | None:
    normalized = _normalize_ticker_text(value)
    if normalized is None or not _VALID_TICKER_RE.fullmatch(normalized):
        return None
    if normalized in _NON_EQUITY_TICKERS:
        return None
    resolution = _TICKER_RESOLVER.resolve(normalized, trade_date)
    if resolution.status in _REJECTED_TICKER_STATUSES or resolution.status in {
        "date_required",
        "pre_listing",
    }:
        return None
    return resolution.price_symbol


def _transaction_date(value):
    if value is None:
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return pd.Timestamp(parsed).date()


def _equity_transaction_row(row: pd.Series) -> bool:
    origin = _normalize_ticker_text(row.get("ticker_origin"))
    if origin is not None and origin.lower() in _REJECTED_TICKER_ORIGINS:
        return False

    source = _normalize_ticker_text(row.get("source"))
    if source is not None and source.lower() not in _OFFICIAL_SOURCES:
        return False

    description = " ".join(
        str(value)
        for value in (
            row.get("raw_asset_class"),
            row.get("asset_description"),
            row.get("raw_asset_description"),
        )
        if value is not None and not pd.isna(value)
    )
    if _UNSUPPORTED_ASSET_RE.search(description):
        return False

    instrument = _normalize_ticker_text(row.get("instrument_type"))
    return instrument is None or instrument.lower() not in _NON_EQUITY_INSTRUMENTS


def _filter_equity_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Keep rows accepted by the consensus equity/provenance boundary."""
    if rows.empty:
        return rows
    mask = rows.apply(_equity_transaction_row, axis=1)
    return rows.loc[mask].copy()


def _canonical_member_or_blank(value) -> str:
    if not isinstance(value, str):
        return ""
    return canonical_member_key(value)


def _empty_ticker_result(ticker: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": [ticker],
            "num_buyers": [0],
            "signal_score": [0.0],
            "scorer_provenance": [CONSENSUS_SCORER_PROVENANCE],
        }
    )


def _below_threshold_result(
    ticker: str, min_trades: int, min_buyers: int
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": [ticker],
            "num_buyers": [min_trades],
            "signal_score": [0.0],
            "scorer_provenance": [CONSENSUS_SCORER_PROVENANCE],
            "note": [f"Below minimum buyer threshold ({min_buyers})"],
        }
    )


def _consensus_result(
    ticker: str,
    buyers: list[str],
    ticker_trades: pd.DataFrame,
    *,
    as_of_date: pd.Timestamp,
) -> pd.DataFrame:
    """Build the identity-free distinct-buyer production result."""
    with_disclosures = ticker_trades.copy()
    with_disclosures["_disclosure"] = pd.to_datetime(
        with_disclosures["disclosure_date"], errors="coerce"
    )
    disclosures = (
        with_disclosures.groupby("_member_canonical")["_disclosure"]
        .max()
        .reindex(buyers)
    )
    days_since = (as_of_date - disclosures).dt.days
    if days_since.isna().any() or (days_since < 0).any():
        raise AnalysisError("Consensus disclosures must be known on or before as_of_date")

    transaction_dates = pd.to_datetime(ticker_trades["transaction_date"], errors="coerce")
    row_disclosures = pd.to_datetime(ticker_trades["disclosure_date"], errors="coerce")
    disclosure_lag_days = (row_disclosures - transaction_dates).dt.days
    if disclosure_lag_days.isna().any() or (disclosure_lag_days < 0).any():
        raise AnalysisError("Consensus purchases require valid public chronology")

    score = float(len(buyers))
    return pd.DataFrame(
        {
            "ticker": [ticker],
            "num_buyers": [len(buyers)],
            "buyers": [", ".join(buyers[:3])],
            "signal_score": [score],
            "scorer_provenance": [CONSENSUS_SCORER_PROVENANCE],
            "max_trade_to_disclosure_days": [int(disclosure_lag_days.max())],
            "median_trade_to_disclosure_days": [float(disclosure_lag_days.median())],
            "oldest_transaction_date": [transaction_dates.min().date()],
            "latest_disclosure_date": [row_disclosures.max().date()],
        }
    )
