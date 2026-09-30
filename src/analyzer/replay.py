"""Replay live candidate scoring at historical decision dates.

Point-in-time discipline is mandatory: candidate disclosures must not be after
``as_of``, and decision prices are read only from dates on or before ``as_of``.
Forward outcomes use the first close after ``as_of`` within the requested
calendar-day horizon. Actor weights use ``n_matured=0`` priors because replay
does not fit weights on each historical sample. The default member set is a v1
seed list; pass ``members`` to replay other members.
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from importlib import import_module

import pandas as pd

from analyzer.member_names import canonical_member_key

_V1_MEMBERS = (
    "Nancy Pelosi",
    "Mitch McConnell",
    "Michael McCaul",
    "Tommy Tuberville",
    "David McCormick",
)
_RESULT_COLUMNS = [
    "ticker",
    "as_of",
    "score",
    "n_actors",
    "fwd_return_pct",
    "spy_return_pct",
    "excess_pct",
    "blocked",
]


class ReplayError(Exception):
    """Raised when replay cannot enforce or evaluate its point-in-time inputs."""


def _same_member_identity(first: str, candidate: str, positions) -> bool:
    first_key = canonical_member_key(first)
    candidate_key = canonical_member_key(candidate)
    # Canonical keys retain middle names; reuse positions' filed-variant rule
    # for nickname and optional-middle-name matches.
    return first_key == candidate_key or positions._same_member_variant(
        first_key, candidate_key
    )


def _same_actor_identity(first: str, candidate: str, positions) -> bool:
    if first == candidate:
        return True
    first_kind, first_separator, first_name = first.partition(":")
    candidate_kind, candidate_separator, candidate_name = candidate.partition(":")
    return (
        first_kind == candidate_kind == "congress"
        and bool(first_separator and candidate_separator)
        and _same_member_identity(first_name, candidate_name, positions)
    )


def _load_dependencies():
    """Import live candidate and scoring functions only when replay is used."""
    modules = {}
    for name, piece in (
        ("analyzer.pipeline", "pipeline.eligible_events"),
        ("analyzer.positions", "positions.holdings_candidates"),
        ("analyzer.actors", "actors.compute_weight"),
        ("analyzer.setups", "setups.score"),
    ):
        try:
            module = import_module(name)
        except ImportError as exc:
            raise ReplayError(f"Missing replay piece: analyzer.{piece}") from exc
        attribute = piece.split(".", 1)[1]
        if not callable(getattr(module, attribute, None)):
            raise ReplayError(f"Missing replay piece: analyzer.{piece}")
        modules[name.rsplit(".", 1)[-1]] = module
    return (
        modules["pipeline"],
        modules["positions"],
        modules["actors"],
        modules["setups"],
    )


def _candidate_frame(value, piece: str) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame):
        raise ReplayError(f"{piece} must return a pandas DataFrame")
    return value.copy()


def _check_disclosures(candidates: pd.DataFrame, as_of: date, piece: str) -> None:
    if candidates.empty:
        return
    if "disclosure_date" not in candidates:
        raise ReplayError(f"{piece} candidates have no disclosure_date for as-of check")
    disclosures = pd.to_datetime(
        candidates["disclosure_date"], errors="coerce", utc=True
    )
    if disclosures.isna().any() or (disclosures.dt.date > as_of).any():
        raise ReplayError(
            f"{piece} returned a disclosure after {as_of} or an undated row"
        )


def _ticker_candidates(
    db,
    ticker: str,
    as_of: date,
    members: list[str],
    pipeline,
    positions,
) -> pd.DataFrame:
    events = _candidate_frame(
        pipeline.eligible_events(db, as_of), "pipeline.eligible_events"
    )
    _check_disclosures(events, as_of, "pipeline.eligible_events")
    if not events.empty:
        if "ticker" not in events:
            raise ReplayError(
                "pipeline.eligible_events candidates have no ticker column"
            )
        if "actor_id" not in events:
            raise ReplayError(
                "pipeline.eligible_events candidates have no actor_id column"
            )
        events = events.loc[
            events["ticker"].astype("string").str.upper() == ticker
        ].copy()

    event_members = set()
    if not events.empty:
        event_members = {
            str(actor_id).split(":", 1)[1]
            for actor_id in events["actor_id"].dropna().unique()
            if ":" in str(actor_id) and str(actor_id).split(":", 1)[1]
        }
    selected = []
    open_positions = set()
    open_position_actors = []
    processed_members = []
    for member in dict.fromkeys([*members, *sorted(event_members)]):
        if any(
            _same_member_identity(previous, member, positions)
            for previous in processed_members
        ):
            continue
        processed_members.append(member)
        frame = _candidate_frame(
            positions.holdings_candidates(db, member, as_of),
            "positions.holdings_candidates",
        )
        _check_disclosures(frame, as_of, "positions.holdings_candidates")
        if frame.empty:
            continue
        if "ticker" not in frame:
            raise ReplayError(
                "positions.holdings_candidates candidates have no ticker column"
            )
        matches = frame.loc[
            frame["ticker"].astype("string").str.upper() == ticker
        ]
        if not matches.empty:
            selected.append(matches)
            if "position_evidence" in matches and "actor_id" in matches:
                open_rows = matches.loc[matches["position_evidence"].eq(True)]
                for position_ticker, actor_id in zip(
                    open_rows["ticker"].astype("string").str.upper(),
                    open_rows["actor_id"].astype(str),
                ):
                    identity = (position_ticker, actor_id)
                    open_positions.add(identity)
                    if identity not in open_position_actors:
                        open_position_actors.append(identity)

    if not events.empty:
        events["blocked_reason"] = None
        representatives = list(open_position_actors)
        for index, row in events.iterrows():
            actor_id = str(row["actor_id"])
            identity = next(
                (
                    representative
                    for representative in representatives
                    if representative[0] == ticker
                    and _same_actor_identity(
                        representative[1], actor_id, positions
                    )
                ),
                None,
            )
            if identity is None:
                identity = (ticker, actor_id)
                representatives.append(identity)
            events.at[index, "actor_id"] = identity[1]
            if identity not in open_positions:
                events.at[index, "entry_ref"] = None
                events.at[index, "blocked_reason"] = "no open position at as_of"
        selected.append(events)

    if not selected:
        return pd.DataFrame(columns=["ticker", "member", "disclosure_date"])
    return pd.concat(selected, ignore_index=True).drop_duplicates(ignore_index=True)


def _latest_close(db, ticker: str, as_of: date) -> float | None:
    """Last known close on or before as_of; None when unknown."""
    connection = getattr(db, "conn", db)
    row = connection.execute(
        """
        SELECT close FROM prices
        WHERE ticker = ? AND date <= ? AND close IS NOT NULL
        ORDER BY date DESC LIMIT 1
        """,
        [ticker, as_of],
    ).fetchone()
    if row is None or row[0] is None:
        return None
    price = float(row[0])
    return price if math.isfinite(price) and price > 0 else None


def _score(
    db, ticker: str, as_of: date, candidates: pd.DataFrame, actors, setups
) -> tuple[object, bool, int]:
    if candidates.empty:
        return None, True, 0
    for column in ("actor_id", "kind", "source"):
        if column not in candidates:
            raise ReplayError(f"Replay candidates have no {column} column")
    weights = {}
    for row in candidates.itertuples():
        actor_id = str(row.actor_id)
        weights[actor_id] = actors.compute_weight(
            str(row.kind), 0, 0.0, str(row.source)
        )
    current = _latest_close(db, ticker, as_of)
    if current is None:
        return None, True, len(weights)
    try:
        ranked, blocked = setups.score(candidates, weights, {ticker: current})
    except (KeyError, TypeError, IndexError, ValueError) as exc:
        raise ReplayError("analyzer.setups.score violated its return contract") from exc
    hit = ranked.loc[
        ranked["ticker"].astype("string").str.upper() == ticker
    ]
    if hit.empty:
        return 0.0, True, len(weights)
    try:
        score_value = float(hit.iloc[0]["score"])
    except (TypeError, ValueError) as exc:
        raise ReplayError("analyzer.setups.score must return a numeric score") from exc
    if not math.isfinite(score_value):
        return None, True, len(weights)
    n_actors = len(weights)
    try:
        n_actors = int(hit.iloc[0].get("n_actors", n_actors))
    except (TypeError, ValueError):
        pass
    # Ranked membership is decided once, in setups.score via is_ranked;
    # replay only mirrors it.
    return score_value, False, n_actors


def _price_return(db, ticker: str, as_of: date, horizon_days: int) -> float | None:
    """Return from the last known close to the last in-horizon close.

    Incomplete horizons stay unknown: if the price history ends before
    as_of + horizon, there is no matured outcome to report, not a partial
    one. This keeps validation honest instead of grading 20-day moves as
    90-day returns.
    """
    connection = getattr(db, "conn", db)
    current_price = _latest_close(db, ticker, as_of)
    if current_price is None:
        return None
    outcome = connection.execute(
        """
        SELECT date, close FROM prices
        WHERE ticker = ? AND date > ? AND date <= ? AND close IS NOT NULL
        ORDER BY date DESC LIMIT 1
        """,
        [ticker, as_of, as_of + timedelta(days=horizon_days)],
    ).fetchall()
    if not outcome:
        return None
    outcome_date, outcome_price = outcome[0][0], float(outcome[0][1])
    if not math.isfinite(outcome_price) or outcome_price <= 0:
        return None
    horizon_end = as_of + timedelta(days=horizon_days)
    if pd.Timestamp(outcome_date).date() < horizon_end - timedelta(days=7):
        return None
    result = (outcome_price / current_price - 1) * 100
    return result if math.isfinite(result) else None


def _benchmark_return(db, as_of: date, horizon_days: int) -> float | None:
    connection = getattr(db, "conn", db)
    spy_exists = connection.execute(
        "SELECT 1 FROM prices WHERE ticker = 'SPY' AND date <= ? LIMIT 1",
        [as_of],
    ).fetchone()
    ticker = "SPY" if spy_exists is not None else "SPYM"
    return _price_return(db, ticker, as_of, horizon_days)


def replay(
    db,
    tickers: list[str],
    decision_dates: list[date],
    *,
    horizon_days: int = 90,
    members: list[str] | None = None,
) -> pd.DataFrame:
    """Evaluate live candidates and scoring at each historical as-of date.

    No disclosure after ``as_of`` or decision price after ``as_of`` is allowed.
    The default ``members`` are the v1 skilled-member seed list; ``n_matured=0``
    keeps actor weights at their priors for comparable historical decisions.
    """
    if horizon_days < 1:
        raise ValueError("horizon_days must be positive")
    pipeline, positions, actors, setups = _load_dependencies()
    selected_members = list(_V1_MEMBERS if members is None else members)
    rows = []
    for raw_ticker in tickers:
        ticker = str(raw_ticker).strip().upper()
        if not ticker:
            raise ValueError("tickers must not contain empty values")
        for raw_as_of in decision_dates:
            as_of = pd.Timestamp(raw_as_of).date()
            candidates = _ticker_candidates(
                db, ticker, as_of, selected_members, pipeline, positions
            )
            score_value, blocked, n_actors = _score(
                db, ticker, as_of, candidates, actors, setups
            )
            fwd_return = _price_return(db, ticker, as_of, horizon_days)
            spy_return = _benchmark_return(db, as_of, horizon_days)
            excess = (
                fwd_return - spy_return
                if fwd_return is not None and spy_return is not None
                else None
            )
            rows.append(
                {
                    "ticker": ticker,
                    "as_of": as_of,
                    "score": score_value,
                    "n_actors": n_actors,
                    "fwd_return_pct": fwd_return,
                    "spy_return_pct": spy_return,
                    "excess_pct": excess,
                    "blocked": blocked,
                }
            )
    return pd.DataFrame(rows, columns=_RESULT_COLUMNS)
