"""Score a ticker by its buyer composition.

`score_ticker_by_buyers` combines a ticker's disclosed buyers into a signal.
The safe default is an identity-free consensus score equal to the number of
distinct recent buyers. Historical member effects are descriptive,
noncausal opt-ins; Bayesian probability-times-alpha and solo posterior gates
are not tradable scores.
"""

from __future__ import annotations

import re

import pandas as pd

from analyzer import signals as _signals
from analyzer._memo import df_memoize
from analyzer.exceptions import AnalysisError
from analyzer.member_names import canonical_member_key
from analyzer.member_ranking.factors import _owner_score_factor, _size_score_factor
from analyzer.member_ranking.lookups import (
    _build_ranking_dicts,
    _get_ticker_purchases,
    _validate_scoring_mode,
)
from analyzer.member_ranking.ranking import rank_members
from analyzer.models import TransactionType
from analyzer.signals import TICKER_PERF_MIN_TRADES
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
    r"private equity|limited partnership)\b",
    re.IGNORECASE,
)


@df_memoize(copy=False)
def score_ticker_by_buyers(
    ticker: str,
    transactions_df: pd.DataFrame,
    signals_df: pd.DataFrame | None = None,
    horizon: int = 90,
    threshold: float = 5.0,
    member_rankings: pd.DataFrame | None = None,
    min_buyers: int = 2,
    ticker_perf_signals: pd.DataFrame | None = None,
    _bayes_prior_strength: float | None = None,
    _ranking_dicts: dict | None = None,
    scoring_mode: str = "consensus",
    as_of_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Score a ticker by its buyer composition. Memoized via @df_memoize.

    When ``_ranking_dicts`` is provided (pre-built by the caller), dict
    lookups replace DataFrame linear scans for buyer stats.
    """
    _validate_scoring_mode(scoring_mode)
    _validate_inputs(signals_df, transactions_df, scoring_mode)

    if scoring_mode == "consensus" and as_of_date is None:
        raise AnalysisError("consensus scoring requires an explicit as_of_date")
    if scoring_mode == "consensus" and pd.isna(pd.Timestamp(as_of_date)):
        raise AnalysisError("consensus as_of_date must be a valid timestamp")
    if scoring_mode != "consensus" and member_rankings is None:
        bayes_prior = (
            _bayes_prior_strength
            if _bayes_prior_strength is not None
            else _signals.BAYES_PRIOR_STRENGTH
        )
        member_rankings = rank_members(
            signals_df, horizon, threshold, _bayes_prior_strength=bayes_prior
        )

    if scoring_mode == "consensus":
        normalized_ticker = _validate_ticker(ticker)
        ticker_trades = _get_consensus_ticker_purchases(
            normalized_ticker, transactions_df
        )
        disclosure_dates = pd.to_datetime(
            ticker_trades["disclosure_date"], errors="coerce"
        )
        ticker_trades = ticker_trades[
            disclosure_dates.notna() & (disclosure_dates <= pd.Timestamp(as_of_date))
        ].copy()
    else:
        ticker_trades = _get_ticker_purchases(ticker, transactions_df).copy()
    if ticker_trades.empty:
        return _empty_ticker_result(ticker)

    canonicalizer = (
        _canonical_member_or_blank
        if scoring_mode == "consensus"
        else canonical_member_key
    )
    ticker_trades["_member_canonical"] = ticker_trades["member"].map(canonicalizer)
    if scoring_mode == "consensus":
        ticker_trades = ticker_trades[ticker_trades["_member_canonical"].ne("")].copy()
        if ticker_trades.empty:
            return _empty_ticker_result(ticker)

    min_trades = ticker_trades["_member_canonical"].nunique()
    if min_trades < min_buyers:
        return _below_threshold_result(ticker, min_trades, min_buyers)

    buyers = ticker_trades["_member_canonical"].unique()
    if scoring_mode == "consensus":
        alpha_dict = {}
        inputs = _consensus_inputs(
            buyers, ticker_trades, as_of_date=pd.Timestamp(as_of_date)
        )
    else:
        rd = (
            _ranking_dicts
            if _ranking_dicts is not None
            else _build_ranking_dicts(member_rankings, scoring_mode=scoring_mode)
        )
        dict_mode = rd.get("mode")
        if dict_mode != scoring_mode:
            raise AnalysisError(
                "_ranking_dicts must declare the same validated scoring_mode"
            )
        alpha_dict = rd["alpha"]
        trades_dict = rd["trades"]
        fallback = _member_only_inputs(
            ticker,
            buyers,
            alpha_dict,
            trades_dict,
            ticker_trades,
            signals_df,
            ticker_perf_signals,
        )
        if isinstance(fallback, pd.DataFrame):
            return fallback
        inputs = fallback
    inputs["scoring_mode"] = scoring_mode
    return _final_result(ticker, buyers, ticker_trades, inputs, alpha_dict)


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
    if transactions_df.empty:
        return transactions_df.copy()
    purchases = transactions_df[
        transactions_df["transaction_type"] == TransactionType.PURCHASE.value
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
        for value in (row.get("asset_description"), row.get("raw_asset_description"))
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


def _validate_inputs(
    signals_df: pd.DataFrame | None,
    transactions_df: pd.DataFrame,
    scoring_mode: str,
) -> None:
    if transactions_df.empty and scoring_mode != "consensus":
        raise AnalysisError("Empty transactions dataframe")
    if scoring_mode != "consensus" and (signals_df is None or signals_df.empty):
        raise AnalysisError("Historical scoring requires a non-empty signal dataframe")


def _empty_ticker_result(ticker: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": [ticker],
            "num_buyers": [0],
            "signal_score": [0.0],
            "signal_score_raw": [0.0],
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
            "signal_score_raw": [0.0],
            "note": [f"Below minimum buyer threshold ({min_buyers})"],
        }
    )


def _consensus_inputs(
    buyers, ticker_trades: pd.DataFrame, *, as_of_date: pd.Timestamp
) -> dict:
    """Build an identity-free distinct-buyer count.

    Candidate selection already applies the public disclosure window. Names,
    historical returns, trade counts, and member posteriors do not enter the
    score, so permuting member identities leaves it unchanged.
    """
    member_col = "_member_canonical"
    with_disclosures = ticker_trades.copy()
    with_disclosures["_disclosure"] = pd.to_datetime(
        with_disclosures["disclosure_date"], errors="coerce"
    )
    disclosures = (
        with_disclosures.groupby(member_col)["_disclosure"].max().reindex(buyers)
    )
    days_since = (as_of_date - disclosures).dt.days
    if days_since.isna().any() or (days_since < 0).any():
        raise AnalysisError(
            "Consensus disclosures must be known on or before as_of_date"
        )
    score = float(len(buyers))
    return {
        "base_signal_score": score,
        "rated_buyers_list": list(buyers),
        "best_rank": 1.0,
        "total_trades": len(buyers),
        "rated_buyers": len(buyers),
        "quality_adjusted_avg": score,
    }


def _member_only_inputs(
    ticker,
    buyers,
    alpha_dict,
    trades_dict,
    ticker_trades,
    signals_df,
    ticker_perf_signals,
):
    rated_buyers_list = [m for m in buyers if m in alpha_dict]
    if not rated_buyers_list:
        return _ticker_history_fallback(ticker, buyers, signals_df, ticker_perf_signals)

    best_rank = max(alpha_dict[m] for m in rated_buyers_list)
    total_trades = sum(trades_dict.get(m, 0) for m in rated_buyers_list)
    rated_buyers = len(rated_buyers_list)

    alpha_values = [alpha_dict[m] for m in rated_buyers_list]
    quality_adjusted_avg = sum(alpha_values) / len(alpha_values)

    return {
        "base_signal_score": quality_adjusted_avg,
        "rated_buyers_list": rated_buyers_list,
        "best_rank": best_rank,
        "total_trades": total_trades,
        "rated_buyers": rated_buyers,
        "quality_adjusted_avg": quality_adjusted_avg,
    }


def _ticker_history_fallback(ticker, buyers, signals_df, ticker_perf_signals):
    fallback_score = 0.0
    fallback_source = "none"
    perf_signals = (
        ticker_perf_signals if ticker_perf_signals is not None else signals_df
    )
    if not perf_signals.empty and "ticker" in perf_signals.columns:
        ticker_hist = perf_signals[
            (perf_signals["ticker"] == ticker)
            & (perf_signals["signal_type"] == TransactionType.PURCHASE.value)
            & (perf_signals["total_spy_alpha_pct"].notna())
        ]
        if "window_complete" in ticker_hist.columns:
            ticker_hist = ticker_hist[
                ticker_hist["window_complete"].fillna(False).astype(bool)
            ]
        if len(ticker_hist) >= TICKER_PERF_MIN_TRADES:
            fallback_score = float(ticker_hist["total_spy_alpha_pct"].mean())
            fallback_source = f"ticker_hist({len(ticker_hist)})"

    return pd.DataFrame(
        {
            "ticker": [ticker],
            "num_buyers": [len(buyers)],
            "buyers": [", ".join(buyers[:3])],
            "signal_score": [round(fallback_score, 2)],
            "signal_score_raw": [fallback_score],
            "fallback_source": [fallback_source],
        }
    )


def _final_result(
    ticker,
    buyers,
    ticker_trades,
    inputs,
    alpha_dict,
) -> pd.DataFrame:
    base_signal_score = inputs["base_signal_score"]
    rated_buyers_list = inputs["rated_buyers_list"]
    best_rank = inputs["best_rank"]
    total_trades = inputs["total_trades"]
    rated_buyers = inputs["rated_buyers"]
    quality_adjusted_avg = inputs["quality_adjusted_avg"]

    size_factor = _size_score_factor(ticker_trades)
    owner_factor = _owner_score_factor(ticker_trades)

    signal_score_raw = base_signal_score
    signal_score = round(signal_score_raw, 2)

    top_buyers = _top_buyers_for_label(buyers, rated_buyers_list, alpha_dict)
    buyer_label = _buyer_label(len(top_buyers), len(buyers))

    return pd.DataFrame(
        {
            "ticker": [ticker],
            "num_buyers": [len(buyers)],
            "rated_buyers": [rated_buyers],
            "buyer_label": [buyer_label],
            "buyers": [", ".join(top_buyers)],
            "avg_buyer_performance": [round(quality_adjusted_avg, 2)],
            "best_buyer_performance": [round(best_rank, 2)],
            "total_buyer_trades": [int(total_trades)],
            "convergence_factor": [1.0],
            "ticker_perf_factor": [round(1.0, 3)],
            "base_signal_score": [round(base_signal_score, 2)],
            "size_factor": [round(size_factor, 3)],
            "owner_factor": [round(owner_factor, 3)],
            "signal_score": [signal_score],
            "signal_score_raw": [signal_score_raw],
            "fallback_source": [inputs.get("scoring_mode", "member_ranked")],
            "scoring_mode": [inputs.get("scoring_mode", "custom")],
            "scorer_provenance": [
                CONSENSUS_SCORER_PROVENANCE
                if inputs.get("scoring_mode") == "consensus"
                else "descriptive_member_skill_v1"
            ],
        }
    )


def _top_buyers_for_label(buyers, rated_buyers_list, alpha_dict) -> list:
    if not rated_buyers_list:
        return list(buyers[:3])
    if not alpha_dict:
        return list(rated_buyers_list[:3])
    return sorted(rated_buyers_list, key=lambda m: alpha_dict.get(m, 0), reverse=True)[
        :3
    ]


def _buyer_label(num_top: int, num_buyers: int) -> str:
    return f"Top {num_top} of {num_buyers}" if num_buyers > 3 else f"{num_buyers}"
