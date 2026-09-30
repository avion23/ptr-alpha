"""Curated institutional managers that file Form 13F reports."""

import re

_ENTITY_SUFFIX_RE = re.compile(
    r"\s+(INCORPORATED|CORPORATION|COMPANY|LIMITED|HOLDINGS|HOLDING|GROUP"
    r"|INC|CORP|LLC|LLP|LTD|PLC|LP|LLP|PA|NA|L P)\s*$"
)


def normalize_manager_name(name: str) -> str:
    """Uppercase, punctuation-free manager name with entity suffixes stripped."""
    normalized = re.sub(r"[^A-Z0-9]+", " ", str(name).upper()).strip()
    while True:
        stripped = _ENTITY_SUFFIX_RE.sub("", normalized).strip()
        if stripped == normalized or not stripped:
            return normalized
        normalized = stripped


def watchlist_actor_ids() -> set[str]:
    """Normalized ``manager:NAME`` ids for the curated watchlist."""
    return {
        f"manager:{normalize_manager_name(entry['name'])}"
        for entry in WATCHLIST_MANAGERS.values()
        if isinstance(entry, dict) and entry.get("name")
    }

WATCHLIST_MANAGERS = {
    # Small, concentrated public-equity manager tied to the VST accumulation signal.
    "thiel_macro": {"name": "Thiel Macro LLC", "cik": "1562087"},
    # David Tepper's concentrated Appaloosa portfolio is a useful smart-money signal.
    "appaloosa": {"name": "Appaloosa LP", "cik": "1656456"},
    # Berkshire is a well-known long-term equity holder and institutional benchmark.
    "berkshire_hathaway": {"name": "Berkshire Hathaway Inc", "cik": "1067983"},
    # Pershing Square is a prominent activist investor with concentrated positions.
    "pershing_square": {
        "name": "Pershing Square Capital Management, L.P.",
        "cik": "1336528",
    },
    # Citadel is a major hedge fund whose filings expose material position changes.
    "citadel_advisors": {"name": "Citadel Advisors LLC", "cik": "1423053"},
}
