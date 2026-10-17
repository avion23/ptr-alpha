"""Shared actor identity and source-adjusted skill weights.

Skill history uses exact actor IDs. Name variants split credit, so a variant
without its own matured history stays at 0.5x; this is conservative by design
and prevents false skill attribution.
"""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd

from analyzer.canonical_scope import canonical_decision
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
        & np.isfinite(price_values.to_numpy(dtype="float64"))
        & price_values.gt(0)
    ].copy()
    eligible_prices["_date"] = price_dates.loc[eligible_prices.index].dt.normalize()
    eligible_prices["_close"] = price_values.loc[eligible_prices.index]

    # Pass 1: resolve episodes and the exact price keys they will consult.
    disclosure_dates = pd.to_datetime(disclosures["disclosure_date"], errors="coerce")
    episodes: list[tuple[str, str, pd.Timestamp, pd.Timestamp]] = []
    seen: set[tuple[str, str, pd.Timestamp]] = set()
    identity_columns = disclosures[["actor_id", "ticker", "source"]]
    windows_by_disclosure: dict[pd.Timestamp, tuple[pd.Timestamp, pd.Timestamp]] = {}
    for identity_row, disclosed in zip(
        identity_columns.itertuples(index=False, name=None), disclosure_dates
    ):
        actor = str(identity_row[0]).strip()
        ticker = str(identity_row[1]).strip().upper()
        source = str(identity_row[2]).strip().lower()
        if source not in _SKILL_SOURCES or not actor or not ticker:
            continue
        if disclosed is None or pd.isna(disclosed) or disclosed >= cutoff:
            continue
        disclosed = pd.Timestamp(disclosed).normalize()
        key = (actor, ticker, disclosed)
        if key in seen:
            continue
        seen.add(key)

        windows = windows_by_disclosure.get(disclosed)
        if windows is None:
            entry = next_nyse_session(disclosed).normalize()
            exit_date = previous_nyse_session(
                disclosed + pd.Timedelta(days=SKILL_HORIZON_DAYS)
            ).normalize()
            windows = (entry, exit_date)
            windows_by_disclosure[disclosed] = windows
        entry, exit_date = windows
        if entry >= cutoff or exit_date >= cutoff or exit_date < entry:
            continue
        episodes.append((actor, ticker, entry, exit_date))

    needed: set[tuple[str, pd.Timestamp]] = set()
    for _, ticker, entry, exit_date in episodes:
        needed.add((ticker, entry))
        needed.add((ticker, exit_date))
        needed.add(("SPY", entry))
        needed.add(("SPY", exit_date))

    # Pass 2: price dict holds only consulted pairs. Rows are applied in the
    # same order as the previous full build, so duplicate ticker/date rows
    # keep the same last-write-wins value; unconsulted pairs are never read.
    ticker_key_of: dict[object, str] = {}
    key_cells = eligible_prices["ticker"].to_numpy()
    day_cells = pd.DatetimeIndex(eligible_prices["_date"])
    close_cells = eligible_prices["_close"].to_numpy()
    closes: dict[tuple[str, pd.Timestamp], float] = {}
    for raw_ticker, day, close in zip(key_cells, day_cells, close_cells):
        ticker_key = ticker_key_of.get(raw_ticker)
        if ticker_key is None:
            ticker_key = str(raw_ticker).strip().upper()
            ticker_key_of[raw_ticker] = ticker_key
        pair = (ticker_key, day)
        if pair in needed:
            closes[pair] = float(close)

    by_actor: dict[str, list[float]] = {}
    for actor, ticker, entry, exit_date in episodes:
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


def _load_actor_members(
    db, as_of: date | pd.Timestamp, actor_ids
) -> pd.DataFrame:
    """Return requested history rows and congress variants that coalesce to them."""
    cutoff = pd.Timestamp(as_of).normalize()
    if cutoff.date() <= date.min:
        return pd.DataFrame(columns=["source", "member", "actor_id"])

    requested = set(map(str, actor_ids))
    requested_by_last_name: dict[tuple[str, str], set[str]] = {}
    for value in requested:
        kind, separator, name = value.partition(":")
        if not separator or kind not in _KINDS:
            continue
        tokens = actor_id(kind, name).partition(":")[2].split()
        if tokens:
            requested_by_last_name.setdefault((kind, tokens[-1]), set()).add(value)

    requested_kinds = {kind for kind, _ in requested_by_last_name}
    sources = tuple(
        sorted(
            source
            for source, kind in _SOURCE_KINDS.items()
            if kind in requested_kinds
        )
    )
    if not sources:
        return pd.DataFrame(columns=["source", "member", "actor_id"])

    member_rows = db.conn.execute(
        "SELECT DISTINCT source, member FROM canonical_transactions "
        f"WHERE disclosure_date < ? AND source IN ({', '.join('?' for _ in sources)}) "
        "AND member IS NOT NULL ORDER BY source, member",
        [cutoff.date(), *sources],
    ).fetchall()
    members = pd.DataFrame(member_rows, columns=["source", "member"])
    members = members.loc[
        members["source"].isin(_SOURCE_KINDS)
        & members["member"].map(lambda member: isinstance(member, str))
    ].copy()
    if not members.empty:
        members["actor_id"] = [
            actor_id(_SOURCE_KINDS[source], member)
            for source, member in members[["source", "member"]].itertuples(
                index=False, name=None
            )
        ]
    else:
        members["actor_id"] = pd.Series(dtype="object")
    from analyzer.positions import same_actor_identity

    def matches_requested(history_id: str) -> bool:
        kind, _, name = history_id.partition(":")
        tokens = name.split()
        if not tokens:
            return False
        candidates = requested_by_last_name.get((kind, tokens[-1]), ())
        return any(
            same_actor_identity(candidate, history_id) for candidate in candidates
        )

    return members.loc[
        members["actor_id"].map(matches_requested),
        ["source", "member", "actor_id"],
    ]


@canonical_decision("as_of")
def load_skill_table(
    db,
    as_of: date | pd.Timestamp,
    actor_ids: set[str] | None = None,
) -> pd.DataFrame:
    """Load point-in-time outcomes for exact, single-identity matches only.

    Congress history includes every identity that position coalescing would
    merge. A requested ID is validated only when that history set is exactly
    the requested ID; compatible variants fail closed.
    """
    from analyzer.member_ranking.buyer_scoring import _prepare_consensus_purchases

    cutoff = pd.Timestamp(as_of).normalize()
    if cutoff.date() <= date.min:
        return pd.DataFrame(columns=_SKILL_COLUMNS)
    if actor_ids is None:
        trades = db.get_transactions_by_date_range(
            date.min,
            (cutoff - pd.Timedelta(days=1)).date(),
            sources=tuple(sorted(_SKILL_SOURCES)),
        )
    else:
        if not actor_ids:
            return pd.DataFrame(columns=_SKILL_COLUMNS)
        selected = _load_actor_members(db, cutoff, actor_ids).drop_duplicates()
        source_members = tuple(
            (str(row.source), str(row.member))
            for row in selected.itertuples(index=False)
        )
        if not source_members:
            return pd.DataFrame(columns=_SKILL_COLUMNS)
        trades = db.transactions.get_skill_history(cutoff.date(), source_members)
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
    has_member = "member" in purchases.columns
    has_ticker = "ticker" in purchases.columns
    member_cells = purchases["member"].tolist() if has_member else [None] * len(purchases)
    ticker_cells = purchases["ticker"].tolist() if has_ticker else [None] * len(purchases)
    source_cells = purchases["source"].tolist()
    resolutions: dict[tuple[str, object], object] = {}
    sessions: dict[pd.Timestamp, tuple[pd.Timestamp, pd.Timestamp]] = {}
    for source_raw, member, raw_ticker, disclosed in zip(
        source_cells, member_cells, ticker_cells, disclosure_dates
    ):
        source = str(source_raw).strip().lower()
        if source not in _SOURCE_KINDS or pd.isna(disclosed) or disclosed >= cutoff:
            continue
        disclosed = pd.Timestamp(disclosed).normalize()
        windows = sessions.get(disclosed)
        if windows is None:
            entry = next_nyse_session(disclosed).normalize()
            exit_date = previous_nyse_session(
                disclosed + pd.Timedelta(days=SKILL_HORIZON_DAYS)
            ).normalize()
            windows = (entry, exit_date)
            sessions[disclosed] = windows
        entry, exit_date = windows
        if entry >= cutoff or exit_date >= cutoff or exit_date < entry:
            continue
        if not isinstance(member, str) or not isinstance(raw_ticker, str):
            continue
        resolve_day = disclosed.date()
        resolution_key = (raw_ticker, resolve_day)
        resolution = resolutions.get(resolution_key)
        if resolution is None:
            resolution = ticker_resolver.resolve(raw_ticker, resolve_day)
            resolutions[resolution_key] = resolution
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

    disclosures = pd.DataFrame(rows)
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
    skill_table = compute_skill_table(disclosures, prices, cutoff)
    if actor_ids is not None:
        from analyzer.positions import same_actor_identity

        history_ids = set(selected["actor_id"])
        requested_ids = set(map(str, actor_ids))
        exact_single_identity_ids = {
            requested
            for requested in requested_ids
            if {
                history
                for history in history_ids
                if same_actor_identity(requested, history)
            }
            == {requested}
        }
        skill_table = skill_table.loc[
            skill_table["actor_id"].isin(exact_single_identity_ids)
        ]
    return skill_table


def compute_actor_weights(
    candidates: pd.DataFrame,
    skill_table: pd.DataFrame,
) -> dict[str, float]:
    """Apply validated skill multipliers to the unchanged source priors.

    Production reality (first-principles review 2026-10-16): this always calls
    compute_weight with n_matured=0, so output takes exactly five values —
    {prior, prior/2} per initiating source plus 0. The Beta-binomial shrinkage
    path in compute_weight never executes here; scores are actor counts with
    adjustments, not continuous skill estimates. Wire real n/hits through
    only with a revalidated backtest, never by silent default change.
    """
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
        and float(row.mean_excess_pct) >= 1e-9
    }
    identity_weights: dict[str, float] = {}
    for row in candidates.itertuples(index=False):
        key = str(row.actor_id)
        if key in identity_weights:
            continue
        multiplier = 1.0 if key in validated else UNKNOWN_SKILL_MULTIPLIER
        identity_weights[key] = (
            compute_weight(str(row.kind), 0, 0.0, str(row.source)) * multiplier
        )
    return identity_weights


def weights_for_candidates(
    db,
    candidates: pd.DataFrame,
    as_of: date | pd.Timestamp,
    *,
    skill_table_cache: dict[tuple[pd.Timestamp, tuple[str, ...]], pd.DataFrame]
    | None = None,
) -> dict[str, float]:
    """Shared point-in-time weight path for live setups and historical replay."""
    nonzero_prior = candidates["source"].map(
        lambda source: SOURCE_PRIORS.get(str(source)) is not None
    )
    actors = tuple(
        sorted(
            set(
                candidates.loc[nonzero_prior, "actor_id"]
                .dropna()
                .astype(str)
            )
        )
    )
    candidate_identities: dict[str, set[str]] = {}
    # Column-major extraction: same per-row merge into identity sets without
    # the Series construction cost of iterrows. A missing identity column
    # reads as the actor key itself, matching Series.get's default.
    has_pre = "_pre_coalesced_actor_id" in candidates.columns
    pre_cells = (
        candidates["_pre_coalesced_actor_id"].tolist() if has_pre else None
    )
    for index, key_raw in enumerate(candidates["actor_id"].tolist()):
        key = str(key_raw)
        original = pre_cells[index] if pre_cells is not None else key
        if pd.isna(original):
            original = key
        candidate_identities.setdefault(key, set()).add(str(original))
    skill_table = pd.DataFrame(columns=_SKILL_COLUMNS)
    if actors:
        cutoff = pd.Timestamp(as_of).normalize()
        key = (cutoff, actors)
        if skill_table_cache is not None and key in skill_table_cache:
            skill_table = skill_table_cache[key]
        else:
            skill_table = load_skill_table(
                db,
                cutoff,
                set(actors),
            )
            if skill_table_cache is not None:
                skill_table_cache[key] = skill_table
    # A coalesced event keeps its original ID. Skill cannot cross that rewrite.
    exact_candidates = {
        actor
        for actor, identities in candidate_identities.items()
        if identities == {actor}
    }
    skill_table = skill_table.loc[skill_table["actor_id"].isin(exact_candidates)]
    return compute_actor_weights(candidates, skill_table)
