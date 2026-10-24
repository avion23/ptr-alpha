"""Canonical member name normalization for congressional trading data.

Congresspeople appear under multiple name variants in disclosure filings —
e.g. 'MICHAEL T. MCCAUL', 'MICHAEL MCCAUL', and 'Michael T. McCaul' all
refer to the same person but split skill histories when used as raw keys.

`canonical_member_key` removes credentials but retains middle names and
initials. Adjacent initials are joined. This is a grouping key, not
proof of person identity.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from functools import lru_cache


# Tokens that are not part of a person's core first/last name and should be
# stripped before computing the canonical key.
_HONORIFICS = frozenset(
    {
        "DR",
        "MR",
        "MRS",
        "MS",
        "HON",
        "HONORABLE",
        "REP",
        "SEN",
        "SR",
        "JR",
        "II",
        "III",
        "IV",
    }
)

# Professional credential / degree / fellowship suffixes observed in filing
# member strings (data shows e.g. 'Neal Patrick Dunn MD, FACS'). Matched as
# whole tokens only, so surnames merely containing these letters ('MOODY')
# are never affected. 'DO' is deliberately excluded: it collides with a real
# surname and has no observed credential use in the data. 'PH' covers dotted
# 'Ph.D.' is normalized before punctuation splitting.
_CREDENTIALS = frozenset(
    {
        "MD",
        "FACS",
        "FACP",
        "FACC",
        "DDS",
        "DMD",
        "PHD",
        "PH",
        "EDD",
        "JD",
        "RN",
        "LPN",
        "CRNA",
        "DVM",
        "DPM",
        "PHARMD",
        "MBA",
        "MPH",
        "MSN",
        "MSW",
        "CPA",
        "CPC",
        "ESQ",
    }
)

# Explicit missing-value spellings seen at data boundaries. Keep this list
# narrow: ordinary names must not be rejected by a heuristic.
_MISSING_MEMBER_KEYS = frozenset(
    {
        "N A",
        "NA",
        "NONE",
        "NAN",
        "NULL",
        "UNKNOWN",
        "UNKNOWN MEMBER",
        "NOT AVAILABLE",
    }
)


@lru_cache(maxsize=65536)
def canonical_member_key(name: str) -> str:
    """Return a canonical lookup key for a member name.

    Algorithm:
    1. Uppercase and ASCII-fold (NFKD + ASCII ignore) so accented characters
       like 'é' → 'E' and 'ñ' → 'N'. Without this, 'Renée Zellweger' would
       collapse to 'REN ZELLWEGER' because the regex strips non-ASCII letters.
    2. Replace all non-alphanumeric characters (punctuation, dots, commas) with
       spaces so 'T.' and 'T' are both just 'T'.
    3. Drop honorifics and credentials, retaining initials.
    4. Keep all name tokens; join consecutive initials.

    Examples::

        canonical_member_key('MICHAEL T. MCCAUL')   # → 'MICHAEL T MCCAUL'
        canonical_member_key('Michael T. McCaul')   # → 'MICHAEL T MCCAUL'
        canonical_member_key('Michael McCaul')      # → 'MICHAEL MCCAUL'
        canonical_member_key('Diana Lynn Harshbarger') # → 'DIANA LYNN HARSHBARGER'
        canonical_member_key('Diana Harshbarger')   # → 'DIANA HARSHBARGER'
        canonical_member_key('Dr. John Smith Jr.')  # → 'JOHN SMITH'
        canonical_member_key('Renée Zellweger')     # → 'RENEE ZELLWEGER'
        canonical_member_key('José E. Serrano')     # → 'JOSE E SERRANO'
    """
    if not name:
        return ""

    # Step 1: uppercase + ASCII-fold (drop combining accents after NFKD).
    folded = (
        unicodedata.normalize("NFKD", name.upper())
        .encode("ascii", "ignore")
        .decode("ascii")
    )

    # Step 2: replace non-alphanumeric with spaces
    folded = re.sub(r"\bPH\.D\.", "PHD", folded)
    folded = re.sub(r"\b(?:[A-Z]\.){2,}", lambda m: m[0].replace(".", ""), folded)
    s = re.sub(r"[^A-Za-z0-9 ]", " ", folded)

    # Step 3: tokenize and drop honorifics, retaining middle initials.
    tokens = [t for t in s.split() if t not in _HONORIFICS]

    # Step 3b: drop credential/suffix tokens (MD, FACS, ...) after the first
    # token. Position 0 is never stripped so leading initials used as first
    # names (e.g. 'JD Vance') survive; matching is whole-token only. Never
    # collapse below two tokens so a real short surname is not eaten.
    if len(tokens) > 1:
        stripped = [tokens[0]] + [t for t in tokens[1:] if t not in _CREDENTIALS]
        if len(stripped) >= 2:
            tokens = stripped

    if not tokens:
        return ""
    if len(tokens) == 1:
        return tokens[0]

    # Preserve initials; join adjacent initials, including first names like JD.
    return re.sub(r"\b(?:[A-Z] ){1,}[A-Z]\b", lambda m: m[0].replace(" ", ""), " ".join(tokens))


def member_identity_key(name: str) -> str:
    """Return a canonical member identity, or blank for known missing markers.

    Unlike ``canonical_member_key``, this is for identity/counting decisions;
    the lower-level canonicalizer remains unchanged for persisted actor IDs.
    """
    key = canonical_member_key(name)
    return "" if key in _MISSING_MEMBER_KEYS else key


_KINDS = frozenset({"congress", "officer", "manager"})
_NICKNAMES = {
    "bob": "robert", "bill": "william", "jim": "james", "mike": "michael",
    "tom": "thomas", "dave": "david", "dan": "daniel", "steve": "steven",
    "joe": "joseph", "matt": "matthew", "nick": "nicholas", "alex": "alexander",
    "chris": "christopher", "pat": "patrick", "tim": "timothy",
    "jeff": "jeffrey", "greg": "gregory", "ron": "ronald", "ken": "kenneth",
    "larry": "lawrence", "rick": "richard", "chuck": "charles", "ro": "rohit",
    "hal": "harold", "dick": "richard", "ned": "edward", "ted": "theodore",
}


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


def _same_person(first: str, candidate: str) -> bool:
    """Last-name-gated nickname equivalence for filed name variants."""
    if first == candidate:
        return True
    for short, full in _NICKNAMES.items():
        if {first, candidate} == {short, full}:
            return True
    return False


def _same_member_variant(first: str, candidate: str) -> bool:
    """Match filed variants without joining conflicting full middle names."""
    first_tokens = first.split()
    candidate_tokens = candidate.split()
    if len(first_tokens) < 2 or len(candidate_tokens) < 2:
        return first == candidate
    if first_tokens[-1] != candidate_tokens[-1]:
        return False
    if not _same_person(first_tokens[0].lower(), candidate_tokens[0].lower()):
        return False
    first_names = (first_tokens[0].lower(), candidate_tokens[0].lower())
    first_middle = [
        token
        for token in first_tokens[1:-1]
        if not any(_same_person(name, token.lower()) for name in first_names)
    ]
    candidate_middle = [
        token
        for token in candidate_tokens[1:-1]
        if not any(_same_person(name, token.lower()) for name in first_names)
    ]
    if not first_middle or not candidate_middle:
        return True
    return len(first_middle) == len(candidate_middle) and all(
        first_token == candidate_token
        or (len(first_token) == 1 and first_token == candidate_token[0])
        or (len(candidate_token) == 1 and candidate_token == first_token[0])
        for first_token, candidate_token in zip(first_middle, candidate_middle)
    )


def same_member_identity(first: str, candidate: str) -> bool:
    """Whether two filed names denote the same human.

    Canonical keys retain middle names; the filed-variant rule covers
    nicknames and optional middle names (Chuck <-> Charles J.).
    """
    first_key = canonical_member_key(first)
    candidate_key = canonical_member_key(candidate)
    return first_key == candidate_key or _same_member_variant(
        first_key, candidate_key
    )


def member_identity_groups(names: Iterable[str]) -> dict[str, str]:
    """Map observed canonical names to conservative, order-independent groups.

    ``same_member_identity`` is a pairwise filed-variant predicate, not a
    transitive equivalence relation. A short variant that matches two
    conflicting full names is therefore marked ambiguous instead of bridging
    those identities. Its empty group key keeps it out of distinct-buyer counts.
    """
    canonical_names: set[str] = set()
    for name in names:
        if isinstance(name, str):
            key = member_identity_key(name)
            if key:
                canonical_names.add(key)
    keys = sorted(canonical_names)
    by_surname: dict[str, list[str]] = {}
    for key in keys:
        tokens = key.split()
        if len(tokens) > 1:
            by_surname.setdefault(tokens[-1], []).append(key)

    matches = {key: {key} for key in keys}
    for surname_keys in by_surname.values():
        for index, first in enumerate(surname_keys):
            for candidate in surname_keys[index + 1 :]:
                if same_member_identity(first, candidate):
                    matches[first].add(candidate)
                    matches[candidate].add(first)

    ambiguous: set[str] = set()
    for key, peers in matches.items():
        ordered_peers = sorted(peers)
        if any(
            not same_member_identity(first, candidate)
            for index, first in enumerate(ordered_peers)
            for candidate in ordered_peers[index + 1 :]
        ):
            ambiguous.add(key)
    remaining = set(keys) - ambiguous
    groups: dict[str, str] = {}
    while remaining:
        representative = min(remaining)
        component = {representative}
        pending = [representative]
        while pending:
            current = pending.pop()
            for candidate in matches[current] & remaining:
                if candidate not in component:
                    component.add(candidate)
                    pending.append(candidate)
        remaining.difference_update(component)
        for key in component:
            groups[key] = representative
    groups.update({key: "" for key in ambiguous})
    return groups


def same_actor_identity(first: str, candidate: str) -> bool:
    """Whether two actor_ids denote the same human initiator."""
    if first == candidate:
        return True
    first_kind, _, first_name = first.partition(":")
    candidate_kind, _, candidate_name = candidate.partition(":")
    return (
        first_kind == candidate_kind == "congress"
        and bool(first_name and candidate_name)
        and _same_actor_member(first_name, candidate_name)
    )


def _same_actor_member(first: str, candidate: str) -> bool:
    first_key = canonical_member_key(first)
    candidate_key = canonical_member_key(candidate)
    first_tokens = first_key.split()
    candidate_tokens = candidate_key.split()
    if first_key == candidate_key:
        return True
    if len(first_tokens) < 2 or len(candidate_tokens) < 2:
        return False
    if first_tokens[-1] != candidate_tokens[-1]:
        return False
    if not _same_person(first_tokens[0].lower(), candidate_tokens[0].lower()):
        return False
    first_middle = first_tokens[1:-1]
    candidate_middle = candidate_tokens[1:-1]
    return not first_middle or not candidate_middle or first_middle == candidate_middle


def _identity_key(name: str) -> str:
    try:
        return canonical_member_key(name)
    except (TypeError, ValueError):
        return str(name).strip().upper()


def dedupe_member_names(
    names: list[str], exclude: list[str] | tuple[str, ...] = ()
) -> list[str]:
    """Drop names denoting an already-listed (or excluded) human.

    Matches on canonical key first, then on filed variants. First spelling
    wins.
    """
    keys = [_identity_key(name) for name in exclude]
    kept: list[str] = []
    for name in names:
        key = _identity_key(name)
        if key in keys or any(
            _same_member_variant(key, seen) or _same_member_variant(seen, key)
            for seen in keys
        ):
            continue
        keys.append(key)
        kept.append(name)
    return kept


def chamber_scoped_member_key(name: str, chamber: str) -> str:
    """Scope a lossy canonical name key to its filing chamber.

    This is a grouping key, not a person identity. An official member ID and
    service-date dimension are still required to distinguish same-name people.
    """
    chamber_key = str(chamber).strip().lower()
    if chamber_key not in {"house", "senate"}:
        raise ValueError(f"Unsupported chamber: {chamber!r}")
    member_key = canonical_member_key(name)
    if not member_key:
        return ""
    return f"{chamber_key}:{member_key}"
