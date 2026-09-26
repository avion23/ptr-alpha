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
