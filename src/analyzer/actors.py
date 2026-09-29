"""Shared actor identity and source-adjusted skill weights."""

from __future__ import annotations

import math

from analyzer.member_names import canonical_member_key

SOURCE_PRIORS = {
    "house_pdf": 0.52,
    "gemini_ocr": 0.52,
    "senate_efd": 0.52,
    "form4": 0.60,
    "13f": None,
}

_KINDS = frozenset({"congress", "officer", "manager"})
_PRIOR_STRENGTH = 20


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


def age_weight(days_old: int, half_life_days: int = 365) -> float:
    """Return exponential episode-age weight for the supplied half-life."""
    return 0.5 ** (days_old / half_life_days)
