"""Actor-weighted setup scoring with entry-price discipline."""

from __future__ import annotations

import math

import pandas as pd

_RANKED_COLUMNS = ["ticker", "score", "actors", "n_actors", "current", "reasons"]
_BLOCKED_COLUMNS = ["ticker", "actor_id", "reason"]

_RANKED_FLOOR = 1.0
_CORROBORATED_FLOOR = 0.25


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
    current_prices: dict[str, float],
    *,
    entry_tol: float = 1.05,
    event_half_life_days: int = 90,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rank price-eligible initiating actors and report blocked actors."""
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
        current = current_prices.get(ticker)
        try:
            current = float(current) if current is not None else None
        except (TypeError, ValueError):
            current = None
        if current is not None and not math.isfinite(current):
            current = None

        actor_rows = grouped[ticker]
        initiator_ids = {
            actor for actor, rows in actor_rows.items() if rows["initiators"]
        }

        def _valid_corroborator(row) -> bool:
            if row.get("source") != "13f":
                return False
            if str(row.get("actor_id")) in initiator_ids:
                return False
            try:
                disclosed = _as_date(row["disclosure_date"])
                decided = _as_date(row["as_of"])
            except (TypeError, ValueError):
                return False
            return disclosed <= decided

        corroborated = any(
            _valid_corroborator(row)
            for rows in actor_rows.values()
            for row in rows["corroborators"]
        )
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

            usable = []
            seen_reference = False
            for row in initiators:
                if _missing(row["entry_ref"]):
                    continue
                seen_reference = True
                entry_ref = float(row["entry_ref"])
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
                reason = (
                    f"{_actor_name(actor_id)} buy @{entry_ref:.2f}{estimated}, "
                    f"now {current:.2f} ({change:+.1%})"
                )
                usable.append((decay, reason, entry_ref))

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

            decay, reason, _ = min(usable, key=lambda item: (-item[0], item[1]))
            contribution = float(weights.get(actor_id, 0.0)) * decay
            if not math.isfinite(contribution) or contribution < 0:
                contribution = 0.0
            accepted.append((actor_id, contribution, reason))

        blocked_rows.extend(
            {
                "ticker": ticker,
                "actor_id": actor_id,
                "reason": "no actor weight",
            }
            for actor_id, contribution, _ in accepted
            if contribution == 0
        )
        accepted = [item for item in accepted if item[1] != 0]
        score_value = sum(item[1] for item in accepted) * (
            1.5 if corroborated else 1.0
        )
        if accepted and not is_ranked(score_value, len(accepted), corroborated):
            blocked_rows.extend(
                {
                    "ticker": ticker,
                    "actor_id": actor_id,
                    "reason": "score below ranking floor (1.00)",
                }
                for actor_id, _, _ in accepted
            )
            continue
        if not accepted:
            continue
        ranked_rows.append(
            {
                "ticker": ticker,
                "score": score_value,
                "actors": [item[0] for item in accepted],
                "n_actors": len(accepted),
                "current": current,
                "reasons": [item[2] for item in accepted],
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
