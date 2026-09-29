"""Actor-weighted setup scoring with entry-price discipline."""

from __future__ import annotations

import math

import pandas as pd

_RANKED_COLUMNS = ["ticker", "score", "actors", "n_actors", "current", "reasons"]
_BLOCKED_COLUMNS = ["ticker", "actor_id", "reason"]


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
        corroborated = any(rows["corroborators"] for rows in actor_rows.values())
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

            references = [
                float(row["entry_ref"])
                for row in initiators
                if not _missing(row["entry_ref"])
            ]
            if not references:
                blocked_rows.append(
                    {
                        "ticker": ticker,
                        "actor_id": actor_id,
                        "reason": "no entry reference",
                    }
                )
                continue
            if current is None:
                blocked_rows.append(
                    {
                        "ticker": ticker,
                        "actor_id": actor_id,
                        "reason": "no current price",
                    }
                )
                continue

            initiating_sources = {str(row["source"]) for row in initiators}
            usable = []
            for row in initiators:
                if _missing(row["entry_ref"]):
                    continue
                entry_ref = float(row["entry_ref"])
                if current > entry_tol * entry_ref:
                    continue

                as_of = _as_date(row["as_of"])
                disclosure_date = _as_date(row["disclosure_date"])
                event_date = _as_date(row["event_date"])
                if disclosure_date > as_of:
                    raise ValueError("disclosure_date must not be after as_of")
                days_since_disclosure = (as_of - disclosure_date).days
                position_evidence = (
                    str(row["source"]) in initiating_sources
                    and (as_of - event_date).days > 60
                )
                decay = (
                    1.0
                    if position_evidence
                    else 0.5 ** (days_since_disclosure / event_half_life_days)
                )
                change = current / entry_ref - 1.0
                reason = (
                    f"{_actor_name(actor_id)} buy @{entry_ref:.2f}, "
                    f"now {current:.2f} ({change:+.1%})"
                )
                usable.append((decay, reason, entry_ref))

            if not usable:
                blocked_rows.append(
                    {
                        "ticker": ticker,
                        "actor_id": actor_id,
                        "reason": "above entry tolerance",
                    }
                )
                continue

            decay, reason, _ = min(usable, key=lambda item: (-item[0], item[1]))
            accepted.append(
                (actor_id, float(weights.get(actor_id, 0.0)) * decay, reason)
            )

        ranked_rows.append(
            {
                "ticker": ticker,
                "score": sum(item[1] for item in accepted)
                * (1.5 if corroborated else 1.0),
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
