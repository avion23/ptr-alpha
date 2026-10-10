"""Shared actor identity and source-adjusted skill weights."""

from __future__ import annotations

import math
from datetime import date

import pandas as pd

from analyzer.member_names import canonical_member_key
from analyzer.price_repository import next_nyse_session, previous_nyse_session
from analyzer.ticker_resolver import TickerResolver

SOURCE_PRIORS = {
    "house_pdf": 0.52,
    "gemini_ocr": 0.52,
    "senate_efd": 0.52,
    "form4": 0.60,
    "13f": None,
}

_KINDS = frozenset({"congress", "officer", "manager"})
_PRIOR_STRENGTH = 20
_SKILL_SOURCES = frozenset({"house_pdf", "gemini_ocr", "senate_efd", "form4"})
_SOURCE_KINDS = {
    "house_pdf": "congress",
    "gemini_ocr": "congress",
    "senate_efd": "congress",
    "form4": "officer",
}
SKILL_HORIZON_DAYS = 90
MIN_VALIDATED_SKILL_OBSERVATIONS = 3
UNKNOWN_SKILL_MULTIPLIER = 0.5
_SKILL_COLUMNS = ["actor_id", "n_matured", "mean_excess_pct"]


def actor_id(kind: str, key: str) -> str:
    """Return a source-independent actor ID from a name and actor kind."""
    if kind not in _KINDS:
        raise ValueError(f"Unsupported actor kind: {kind!r}")
    normalized = (
        canonical_member_key(key)
        if kind == "congress"
        else " ".join(key.upper().split())
    )
    return f"{kind}:{normalized}"


def compute_weight(kind: str, n_matured: int, hits: float, source: str) -> float:
    """Return a Beta-binomial skill estimate shrunk to the source prior.

    ``n_matured`` counts episodes whose outcomes are known as of the decision
    date; the caller must enforce that point-in-time boundary. ``hits`` may be
    fractional when episode hits have age weights. 13F never initiates, so its
    weight is always zero.
    """
    if kind not in _KINDS:
        raise ValueError(f"Unsupported actor kind: {kind!r}")
    if source not in SOURCE_PRIORS:
        raise ValueError(f"Unsupported actor source: {source!r}")
    if n_matured < 0 or not math.isfinite(hits) or not 0 <= hits <= n_matured:
        raise ValueError("n_matured and hits must satisfy 0 <= hits <= n_matured")

    prior = SOURCE_PRIORS[source]
    if prior is None:
        return 0.0
    if n_matured == 0:
        return prior
    return (hits + _PRIOR_STRENGTH * prior) / (n_matured + _PRIOR_STRENGTH)


def compute_skill_table(
    disclosures: pd.DataFrame, prices: pd.DataFrame, as_of: date | pd.Timestamp
) -> pd.DataFrame:
    """Summarize matured 90-day forward excess by actor at a strict cutoff.

    The entry is the first NYSE close after disclosure. The exit is the last
    NYSE close on or before disclosure + 90 calendar days. Both the disclosure
    and exit must be strictly before ``as_of``. Price rows on or after the
    cutoff are ignored, even when callers provide them.
    """
    if disclosures.empty:
        return pd.DataFrame(columns=_SKILL_COLUMNS)
    required = {"actor_id", "ticker", "source", "disclosure_date"}
    if not required.issubset(disclosures.columns):
        raise ValueError(f"disclosures are missing {required - set(disclosures.columns)}")
    if not {"ticker", "date", "close"}.issubset(prices.columns):
        raise ValueError("prices must have ticker, date, and close columns")

    cutoff = pd.Timestamp(as_of).normalize()
    price_dates = pd.to_datetime(prices["date"], errors="coerce")
    price_values = pd.to_numeric(prices["close"], errors="coerce")
    eligible_prices = prices.loc[
        price_dates.notna()
        & (price_dates < cutoff)
        & price_values.notna()
        & price_values.gt(0)
    ].copy()
    eligible_prices["_date"] = price_dates.loc[eligible_prices.index].dt.normalize()
    eligible_prices["_close"] = price_values.loc[eligible_prices.index]
    closes = {
        (str(ticker).strip().upper(), pd.Timestamp(day)): float(close)
        for ticker, day, close in eligible_prices[
            ["ticker", "_date", "_close"]
        ].itertuples(index=False, name=None)
    }

    disclosure_dates = pd.to_datetime(disclosures["disclosure_date"], errors="coerce")
    by_actor: dict[str, list[float]] = {}
    seen: set[tuple[str, str, pd.Timestamp]] = set()
    for (_, row), disclosed in zip(disclosures.iterrows(), disclosure_dates):
        actor = str(row["actor_id"]).strip()
        ticker = str(row["ticker"]).strip().upper()
        source = str(row["source"]).strip().lower()
        if source not in _SKILL_SOURCES or not actor or not ticker:
            continue
        if disclosed is None or pd.isna(disclosed) or disclosed >= cutoff:
            continue
        disclosed = pd.Timestamp(disclosed).normalize()
        key = (actor, ticker, disclosed)
        if key in seen:
            continue
        seen.add(key)

        entry = next_nyse_session(disclosed).normalize()
        exit_date = previous_nyse_session(
            disclosed + pd.Timedelta(days=SKILL_HORIZON_DAYS)
        ).normalize()
        if entry >= cutoff or exit_date >= cutoff or exit_date < entry:
            continue
        security_entry = closes.get((ticker, entry))
        security_exit = closes.get((ticker, exit_date))
        spy_entry = closes.get(("SPY", entry))
        spy_exit = closes.get(("SPY", exit_date))
        if None in (security_entry, security_exit, spy_entry, spy_exit):
            continue
        excess = (
            (security_exit / security_entry - 1.0)
            - (spy_exit / spy_entry - 1.0)
        ) * 100.0
        if math.isfinite(excess):
            by_actor.setdefault(actor, []).append(excess)

    return pd.DataFrame(
        [
            {
                "actor_id": actor,
                "n_matured": len(outcomes),
                "mean_excess_pct": sum(outcomes) / len(outcomes),
            }
            for actor, outcomes in sorted(by_actor.items())
        ],
        columns=_SKILL_COLUMNS,
    )


def load_skill_table(
    db,
    as_of: date | pd.Timestamp,
    actor_ids: set[str] | None = None,
) -> pd.DataFrame:
    """Load canonical, matured actor outcomes without crossing ``as_of``."""
    from analyzer.member_ranking.buyer_scoring import _prepare_consensus_purchases

    cutoff = pd.Timestamp(as_of).normalize()
    if cutoff.date() <= date.min:
        return pd.DataFrame(columns=_SKILL_COLUMNS)
    trades = db.get_transactions_by_date_range(
        date.min,
        (cutoff - pd.Timedelta(days=1)).date(),
    )
    if trades.empty:
        return pd.DataFrame(columns=_SKILL_COLUMNS)
    trades = trades.loc[trades["source"].isin(_SKILL_SOURCES)].copy()
    if trades.empty:
        return pd.DataFrame(columns=_SKILL_COLUMNS)
    if actor_ids is not None:
        trades = trades.loc[
            [
                isinstance(member, str)
                and actor_id(_SOURCE_KINDS[source], member) in actor_ids
                for source, member in zip(trades["source"], trades["member"])
            ]
        ].copy()
        if trades.empty:
            return pd.DataFrame(columns=_SKILL_COLUMNS)

    purchases = _prepare_consensus_purchases(trades)
    if purchases.empty:
        return pd.DataFrame(columns=_SKILL_COLUMNS)
    disclosure_dates = pd.to_datetime(purchases["disclosure_date"], errors="coerce")
    ticker_resolver = TickerResolver()
    rejected_ticker_statuses = {
        "acquired",
        "date_required",
        "pre_listing",
        "quarantined",
        "unresolved",
    }
    rows = []
    for index, purchase in purchases.iterrows():
        source = str(purchase["source"]).strip().lower()
        disclosed = disclosure_dates.loc[index]
        if source not in _SOURCE_KINDS or pd.isna(disclosed) or disclosed >= cutoff:
            continue
        disclosed = pd.Timestamp(disclosed).normalize()
        entry = next_nyse_session(disclosed).normalize()
        exit_date = previous_nyse_session(
            disclosed + pd.Timedelta(days=SKILL_HORIZON_DAYS)
        ).normalize()
        if entry >= cutoff or exit_date >= cutoff or exit_date < entry:
            continue
        member = purchase.get("member")
        raw_ticker = purchase.get("ticker")
        if not isinstance(member, str) or not isinstance(raw_ticker, str):
            continue
        resolution = ticker_resolver.resolve(raw_ticker, disclosed.date())
        if resolution.status in rejected_ticker_statuses:
            continue
        rows.append(
            {
                "actor_id": actor_id(_SOURCE_KINDS[source], member),
                "ticker": resolution.price_symbol,
                "source": source,
                "disclosure_date": disclosed,
            }
        )
    if not rows:
        return pd.DataFrame(columns=_SKILL_COLUMNS)

    disclosures = pd.DataFrame(rows).drop_duplicates(
        ["actor_id", "ticker", "disclosure_date"]
    )
    price_tickers = sorted(set(disclosures["ticker"]) | {"SPY"})
    min_date = min(
        next_nyse_session(day).normalize()
        for day in disclosures["disclosure_date"]
    ).date()
    prices = db.conn.execute(
        "SELECT ticker, date, close FROM prices "
        "WHERE ticker = ANY(?) AND date >= ? AND date < ?",
        [price_tickers, min_date, cutoff.date()],
    ).fetchdf()
    return compute_skill_table(disclosures, prices, cutoff)


def compute_actor_weights(
    candidates: pd.DataFrame, skill_table: pd.DataFrame
) -> dict[str, float]:
    """Apply validated skill multipliers to the unchanged source priors."""
    required_candidates = {"actor_id", "kind", "source"}
    if not required_candidates.issubset(candidates.columns):
        raise ValueError(
            f"candidates are missing {required_candidates - set(candidates.columns)}"
        )
    if not {"actor_id", "n_matured", "mean_excess_pct"}.issubset(
        skill_table.columns
    ):
        raise ValueError("skill_table must have actor_id, n_matured, mean_excess_pct")

    validated = {
        str(row.actor_id)
        for row in skill_table.itertuples(index=False)
        if int(row.n_matured) >= MIN_VALIDATED_SKILL_OBSERVATIONS
        and math.isfinite(float(row.mean_excess_pct))
        and float(row.mean_excess_pct) > 0
    }
    weights: dict[str, float] = {}
    for row in candidates.itertuples(index=False):
        key = str(row.actor_id)
        if key in weights:
            continue
        multiplier = 1.0 if key in validated else UNKNOWN_SKILL_MULTIPLIER
        weights[key] = compute_weight(str(row.kind), 0, 0.0, str(row.source)) * multiplier
    return weights


def weights_for_candidates(
    db, candidates: pd.DataFrame, as_of: date | pd.Timestamp
) -> dict[str, float]:
    """Shared point-in-time weight path for live setups and historical replay."""
    actors = set(candidates["actor_id"].dropna().astype(str))
    return compute_actor_weights(candidates, load_skill_table(db, as_of, actors))


def age_weight(days_old: int, half_life_days: int = 365) -> float:
    """Return exponential episode-age weight for the supplied half-life."""
    return 0.5 ** (days_old / half_life_days)
