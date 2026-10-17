"""Actor-weighted setup scoring with entry-price discipline."""

from __future__ import annotations

import math
from collections.abc import Collection
from datetime import timedelta

import pandas as pd

from analyzer.manager_watchlist import normalize_manager_name

_RANKED_COLUMNS = [
    "ticker",
    "score",
    "actors",
    "n_actors",
    "current",
    "reasons",
    "min_disclosure_lag_days",
    "has_fresh_evidence",
]
_BLOCKED_COLUMNS = ["ticker", "actor_id", "reason"]

_RANKED_FLOOR = 1.0
_CORROBORATED_FLOOR = 0.25
_MIN_USABLE_CONTRIBUTION = 0.05


def is_ranked(score_value: float, n_actors: int, corroborated: bool = False) -> bool:
    """A setup ranks when strong alone, corroborated across actors, or
    backed by a watchlisted fund.

    score >= 1.0 is roughly two independent evidence pieces; two or more
    initiators rank at a lower floor, as does a lone initiator with
    watchlist-13F corroboration (officer + skilled fund). Single weak
    actors (June-ZTS style) stay visible in blocked, never in ranked.
    """
    if score_value >= _RANKED_FLOOR:
        return True
    if score_value <= _CORROBORATED_FLOOR:
        return False
    return n_actors >= 2 or corroborated


def score(
    candidates: pd.DataFrame,
    weights: dict[str, float],
    current_prices: dict[str, float | tuple[float, object]],
    *,
    entry_tol: float = 1.05,
    event_half_life_days: int = 90,
    watchlist: Collection[str] | None = None,
    audit: dict | None = None,
    ex_dividends: dict[str, list] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rank price-eligible initiating actors and report blocked actors.

    13F corroboration counts only for curated watchlist managers: pass
    ``manager_watchlist.watchlist_actor_ids()``. Without a watchlist no
    corroboration is recognized (fail closed). Dated current prices use
    ``(price, date)``; bare floats retain legacy behavior without freshness
    validation. ``ex_dividends`` maps ticker -> known ex-dates; a ticker with
    an ex-div in the 7 calendar days through the decision date is blocked
    (cum-div cost basis vs ex-div quote is a contaminated comparison).
    Missing rows mean unknown, never clean: without the map the gate is off.
    """
    if entry_tol <= 0 or event_half_life_days <= 0:
        raise ValueError("entry_tol and event_half_life_days must be positive")

    grouped: dict[str, dict[str, dict[str, list[dict]]]] = {}
    for row in candidates.to_dict("records"):
        ticker, actor_id = row["ticker"], row["actor_id"]
        actor_rows = grouped.setdefault(ticker, {}).setdefault(
            actor_id, {"initiators": [], "corroborators": []}
        )
        actor_rows["corroborators" if row["corroboration"] else "initiators"].append(
            row
        )

    ranked_rows = []
    blocked_rows = []
    for ticker in sorted(grouped):
        quote = current_prices.get(ticker)
        quote_date = None
        stale_quote = False
        if isinstance(quote, tuple):
            if len(quote) != 2:
                current, stale_quote = None, True
            else:
                current, raw_quote_date = quote
                try:
                    quote_date = _as_date(raw_quote_date)
                except (TypeError, ValueError):
                    stale_quote = True
        else:
            current = quote
        try:
            current = float(current) if current is not None else None
        except (OverflowError, TypeError, ValueError):
            current = None
        if current is not None and (not math.isfinite(current) or current <= 0):
            current = None

        actor_rows = grouped[ticker]
        decision_dates = [
            _as_date(row["as_of"])
            for rows in actor_rows.values()
            for row in rows["initiators"]
        ]
        if (
            quote_date is not None
            and decision_dates
            and max(decision_dates) - quote_date > timedelta(days=5)
        ):
            stale_quote = True
        initiator_ids = {
            actor for actor, rows in actor_rows.items() if rows["initiators"]
        }

        def _valid_corroborator(row) -> bool:
            if row.get("source") != "13f":
                return False
            if str(row.get("actor_id")) in initiator_ids:
                return False
            if watchlist is None:
                return False
            actor_name = str(row.get("actor_id")).partition(":")[2]
            if f"manager:{normalize_manager_name(actor_name)}" not in watchlist:
                return False
            try:
                disclosed = _as_date(row["disclosure_date"])
                decided = _as_date(row["as_of"])
            except (TypeError, ValueError):
                return False
            if row.get("blocked_reason") == "no open position at as_of":
                return False
            if disclosed > decided:
                return False
            # 13F filings are quarterly snapshots, not trades. A position
            # disclosed two quarters ago says nothing about today (the fund
            # may have exited since); ancient disclosures must not promote.
            if (decided - disclosed).days > 180:
                return False
            return True

        corroborated = any(
            _valid_corroborator(row)
            for rows in actor_rows.values()
            for row in rows["corroborators"]
        )
        if ex_dividends and decision_dates:
            window_end = max(decision_dates)
            window_start = window_end - timedelta(days=7)
            ex_dates = []
            for raw in ex_dividends.get(ticker, []) or []:
                try:
                    ex_dates.append(_as_date(raw))
                except (TypeError, ValueError):
                    continue
            hits = sorted({d for d in ex_dates if window_start < d <= window_end})
            if hits:
                reason = f"ex-div window (ex {hits[-1].isoformat()})"
                for actor_id in sorted(actor_rows):
                    blocked_rows.append(
                        {"ticker": ticker, "actor_id": actor_id, "reason": reason}
                    )
                if audit is not None:
                    audit.setdefault(ticker, {}).update(
                        ex_div_block=hits[-1].isoformat(),
                        ranking_gate=False,
                    )
                continue
        accepted = []
        for actor_id in sorted(actor_rows):
            evidence = actor_rows[actor_id]
            initiators = evidence["initiators"]
            if not initiators:
                blocked_rows.append(
                    {
                        "ticker": ticker,
                        "actor_id": actor_id,
                        "reason": "corroboration only",
                    }
                )
                continue
            if stale_quote:
                blocked_rows.append(
                    {"ticker": ticker, "actor_id": actor_id, "reason": "stale quote"}
                )
                continue

            usable = []
            seen_reference = False
            usable_lag = None
            for row in initiators:
                if _missing(row["entry_ref"]):
                    continue
                seen_reference = True
                try:
                    entry_ref = float(row["entry_ref"])
                except (OverflowError, TypeError, ValueError):
                    continue
                if not math.isfinite(entry_ref) or entry_ref <= 0:
                    continue
                if current is None or current > entry_tol * entry_ref:
                    continue

                as_of = _as_date(row["as_of"])
                disclosure_date = _as_date(row["disclosure_date"])
                if disclosure_date > as_of:
                    raise ValueError("disclosure_date must not be after as_of")
                days_since_disclosure = (as_of - disclosure_date).days
                # Position evidence persists at full weight; event-only
                # evidence decays continuously. No cliff: the flag, not an
                # age threshold, separates the two.
                position_evidence = bool(row.get("position_evidence", False))
                decay = (
                    1.0
                    if position_evidence
                    else 0.5 ** (days_since_disclosure / event_half_life_days)
                )
                change = current / entry_ref - 1.0
                estimated = " (est.)" if str(row.get("kind")) == "congress" else ""
                # position_evidence rows carry the blended FIFO cost basis
                # (positions.py), not a trade price — label it "basis".
                verb = "basis" if position_evidence else "buy"
                reason = (
                    f"{_actor_name(actor_id)} {verb} @{entry_ref:.2f}{estimated}, "
                    f"now {current:.2f} ({change:+.1%})"
                )
                usable.append((decay, reason, entry_ref))
                if usable_lag is None or days_since_disclosure < usable_lag:
                    usable_lag = days_since_disclosure

            if not usable:
                blocked_reason = next(
                    (
                        row.get("blocked_reason")
                        for row in initiators
                        if isinstance(row.get("blocked_reason"), str)
                        and row.get("blocked_reason").strip()
                    ),
                    None,
                )
                if blocked_reason:
                    reason = blocked_reason
                elif not seen_reference:
                    reason = "no entry reference"
                elif current is None:
                    reason = "no current price"
                else:
                    reason = "above entry tolerance"
                blocked_rows.append(
                    {
                        "ticker": ticker,
                        "actor_id": actor_id,
                        "reason": reason,
                    }
                )
                continue

            decay, reason, entry_ref = min(usable, key=lambda item: (-item[0], item[1]))
            contribution = float(weights.get(actor_id, 0.0)) * decay
            if not math.isfinite(contribution) or contribution < 0:
                contribution = 0.0
            if contribution < _MIN_USABLE_CONTRIBUTION:
                blocked_rows.append(
                    {
                        "ticker": ticker,
                        "actor_id": actor_id,
                        "reason": (
                            "no actor weight"
                            if contribution == 0
                            else "below usable evidence floor (0.05)"
                        ),
                    }
                )
                continue
            accepted.append((actor_id, contribution, reason, usable_lag))
            if audit is not None:
                audit.setdefault(ticker, {}).setdefault("members", []).append(
                    {"member": actor_id, "entry_ref": entry_ref,
                     "weight": float(weights.get(actor_id, 0.0)),
                     "contribution": contribution}
                )

        score_value = sum(item[1] for item in accepted) * (
            1.5 if corroborated else 1.0
        )
        if audit is not None:
            audit.setdefault(ticker, {}).update(
                corroborated_13f=corroborated,
                quote_date=str(quote_date) if quote_date else None,
                quote_value=current,
                stale_quote=stale_quote,
                ranking_gate=is_ranked(score_value, len(accepted), corroborated),
            )
        if accepted and not is_ranked(score_value, len(accepted), corroborated):
            blocked_rows.extend(
                {
                    "ticker": ticker,
                    "actor_id": actor_id,
                    "reason": "score below ranking floor (1.00)",
                }
                for actor_id, _, _, _ in accepted
            )
            continue
        if not accepted:
            continue
        lags = [lag for _, _, _, lag in accepted if lag is not None]
        min_lag = min(lags) if lags else None
        ranked_rows.append(
            {
                "ticker": ticker,
                "score": score_value,
                "actors": [item[0] for item in accepted],
                "n_actors": len(accepted),
                "current": current,
                "reasons": [item[2] for item in accepted],
                "min_disclosure_lag_days": min_lag,
                "has_fresh_evidence": min_lag is not None and min_lag <= 90,
            }
        )

    ranked = pd.DataFrame(ranked_rows, columns=_RANKED_COLUMNS)
    if not ranked.empty:
        ranked = ranked.sort_values(
            ["score", "ticker"], ascending=[False, True], kind="stable"
        ).reset_index(drop=True)
    return ranked, pd.DataFrame(blocked_rows, columns=_BLOCKED_COLUMNS)


def _missing(value) -> bool:
    return value is None or bool(pd.isna(value))


def _as_date(value):
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"invalid date: {value!r}")
    return parsed.date()


def _actor_name(actor_id: str) -> str:
    return actor_id.split(":", 1)[-1].replace("_", " ").upper()
