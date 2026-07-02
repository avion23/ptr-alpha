"""Canonical member name normalization for congressional trading data.

Congresspeople appear under multiple name variants in disclosure filings —
e.g. 'MICHAEL T. MCCAUL', 'MICHAEL MCCAUL', and 'Michael T. McCaul' all
refer to the same person but split skill histories when used as raw keys.

`canonical_member_key` reduces any name variant to FIRSTNAME LASTNAME (upper,
no punctuation, no honorifics, no middle name/initial) so that all variants
of the same person map to one stable key.
"""

from __future__ import annotations

import re


# Tokens that are not part of a person's core first/last name and should be
# stripped before computing the canonical key.
_HONORIFICS = frozenset({
    "DR", "MR", "MRS", "MS", "HON", "REP", "SEN", "SR",
    "JR", "II", "III", "IV",
})


def canonical_member_key(name: str) -> str:
    """Return a canonical lookup key for a member name.

    Algorithm:
    1. Uppercase.
    2. Replace all non-alphanumeric characters (punctuation, dots, commas) with
       spaces so 'T.' and 'T' are both just 'T'.
    3. Split into tokens and drop honorifics (DR, JR, III, HON, …) and
       single-letter tokens (middle initials like 'T').
    4. If two or more tokens remain, return FIRST + LAST (drop all middle tokens).
       This ensures 'MICHAEL T. MCCAUL', 'MICHAEL MCCAUL', and 'Michael T. McCaul'
       all collapse to 'MICHAEL MCCAUL'.

    Examples::

        canonical_member_key('MICHAEL T. MCCAUL')   # → 'MICHAEL MCCAUL'
        canonical_member_key('Michael T. McCaul')   # → 'MICHAEL MCCAUL'
        canonical_member_key('Michael McCaul')      # → 'MICHAEL MCCAUL'
        canonical_member_key('Diana Lynn Harshbarger') # → 'DIANA HARSHBARGER'
        canonical_member_key('Diana Harshbarger')   # → 'DIANA HARSHBARGER'
        canonical_member_key('Dr. John Smith Jr.')  # → 'JOHN SMITH'
    """
    if not name:
        return ""

    # Step 1-2: uppercase, replace non-alpha with spaces
    s = re.sub(r"[^A-Za-z0-9 ]", " ", name.upper())

    # Step 3: tokenize, drop honorifics and single-letter tokens (middle initials)
    tokens = [t for t in s.split() if t not in _HONORIFICS and len(t) > 1]

    if not tokens:
        return ""
    if len(tokens) == 1:
        return tokens[0]

    # Step 4: keep only FIRST + LAST
    return f"{tokens[0]} {tokens[-1]}"
