"""Per-document agreement gate between two independent vision readings.

Engine A (Gemini) and engine B (any OpenAI-compatible vision endpoint) each
produce the same row shape for one document. The gate normalizes both row
multisets and decides per DOC:

* ``AGREE``   — identical normalized multisets; rows persist from the A side.
* ``DISAGREE``— nothing persists from either side; both readings plus a diff
  summary are serialized (JSON, bounded to the ``error_message`` column) for
  the manual-review parse-run inventory.

Normalization is deliberately shallow: transaction type is title-cased, dates
are converted to ISO, asset text is whitespace-collapsed and casefolded, and
amount letters are uppercased. Page attribution is intentionally NOT part of
the row key — a row's content, not its page coordinates, is what enters
canonical. Unparsable dates keep their raw casefolded text so two different
garbage readings never compare equal.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import date

AGREE = "AGREE"
DISAGREE = "DISAGREE"

_MISSING_DATE_TOKENS = {"", "n/a", "na", "none", "not applicable"}
_ISO_DATE = re.compile(r"(\d{4})-(\d{1,2})-(\d{1,2})")
_SLASH_DATE = re.compile(r"(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})")
_DIFF_PREVIEW_ROWS = 5


@dataclass(frozen=True)
class GateResult:
    """Outcome of the per-document agreement gate."""

    status: str
    diff_summary: str | None = None


def _text(value: object) -> str:
    return "" if value is None else str(value).strip()


def _field(tx: dict, *names: str) -> object:
    """First non-empty alias, mirroring the shared OCR row accessors."""
    for name in names:
        value = tx.get(name)
        if value is not None and value != "":
            return value
    return None


def normalize_asset(value: object) -> str:
    return " ".join(_text(value).split()).casefold()


def normalize_type(value: object) -> str:
    return " ".join(_text(value).split()).title()


def normalize_date_value(value: object) -> str | None:
    """ISO-normalize MM/DD[/-]YY(YY) or ISO dates; missing tokens -> None."""
    if value is None:
        return None
    text = _text(value)
    if text.casefold() in _MISSING_DATE_TOKENS:
        return None
    iso = _ISO_DATE.fullmatch(text)
    if iso is not None:
        year, month, day = (int(part) for part in iso.groups())
        try:
            date(year, month, day)
        except ValueError:
            return text.casefold()
        return f"{year:04d}-{month:02d}-{day:02d}"
    match = _SLASH_DATE.fullmatch(text)
    if match is not None:
        month, day, year = (int(part) for part in match.groups())
        if year < 100:
            year += 2000 if year < 50 else 1900
        try:
            date(year, month, day)
        except ValueError:
            return text.casefold()
        return f"{year:04d}-{month:02d}-{day:02d}"
    return text.casefold()


def reading_key(tx: object) -> tuple[str, str, str | None, str | None, str]:
    """Normalized multiset key for one row (page attribution excluded)."""
    row = tx if isinstance(tx, dict) else {}
    return (
        normalize_asset(_field(row, "asset", "asset_description")),
        normalize_type(_field(row, "type", "transaction_type")),
        normalize_date_value(_field(row, "date", "tx_date", "transaction_date")),
        normalize_date_value(
            _field(row, "notif_date", "notification_date", "disclosure_date")
        ),
        _text(_field(row, "amount_letter", "amount", "amount_raw")).upper(),
    )


def reading_tuple(tx: object) -> list:
    """Verbatim compact evidence for review storage (JSON-serializable)."""
    row = tx if isinstance(tx, dict) else {}
    page = row.get("page_number")
    return [
        _text(_field(row, "asset", "asset_description")),
        _text(_field(row, "type", "transaction_type")),
        _text(_field(row, "date", "tx_date", "transaction_date")),
        _text(_field(row, "notif_date", "notification_date", "disclosure_date")),
        _text(_field(row, "amount_letter", "amount", "amount_raw")).upper(),
        page if isinstance(page, int) else _text(page) or None,
    ]


def gate_rows(rows_a: list[dict], rows_b: list[dict]) -> GateResult:
    """Compare the normalized row multisets of both readings for one doc."""
    counter_a = Counter(reading_key(tx) for tx in (rows_a or []))
    counter_b = Counter(reading_key(tx) for tx in (rows_b or []))
    if counter_a == counter_b:
        return GateResult(AGREE)
    only_a = list((counter_a - counter_b).elements())
    only_b = list((counter_b - counter_a).elements())

    def render(keys: list[tuple]) -> str:
        return "; ".join(
            "|".join("" if part is None else str(part) for part in key)
            for key in keys[:_DIFF_PREVIEW_ROWS]
        )

    diff_summary = (
        f"only_in_a={len(only_a)}[{render(only_a)}]; "
        f"only_in_b={len(only_b)}[{render(only_b)}]"
    )
    return GateResult(DISAGREE, diff_summary)


def disagreement_notes(
    rows_a: list[dict],
    rows_b: list[dict] | None,
    *,
    reason: str,
    engines_attempted: str,
    diff_summary: str | None = None,
    member_a: str | None = None,
    member_b: str | None = None,
    raw_b: str | None = None,
    max_chars: int = 990,
) -> str:
    """Serialize both readings for the parse-run error/notes column.

    The column is bounded (``record_parse_run`` truncates at 1000 chars), so
    the payload shrinks deterministically — tail rows first, then B's raw
    text, then the diff — and always stays valid JSON.
    """
    reading_a = [reading_tuple(row) for row in (rows_a or [])]
    reading_b = [reading_tuple(row) for row in (rows_b or [])]
    notes: dict = {
        "decision": DISAGREE,
        "reason": reason,
        "engines_attempted": engines_attempted,
        "member_a": member_a,
        "member_b": member_b,
        "diff": diff_summary,
        "reading_a": reading_a,
        "reading_b": reading_b,
        "truncated": 0,
    }
    if raw_b is not None:
        notes["reading_b_raw"] = raw_b[:500]

    def dumps() -> str:
        return json.dumps(notes, ensure_ascii=True, sort_keys=True)

    while len(dumps()) > max_chars and (reading_a or reading_b):
        longer = reading_a if len(reading_a) >= len(reading_b) else reading_b
        if longer:
            longer.pop()
            notes["truncated"] = int(notes["truncated"]) + 1
        notes["reading_a"] = reading_a
        notes["reading_b"] = reading_b
    while len(dumps()) > max_chars and notes.get("reading_b_raw"):
        notes["reading_b_raw"] = notes["reading_b_raw"][: len(notes["reading_b_raw"]) // 2]
    while len(dumps()) > max_chars and notes.get("diff"):
        notes["diff"] = notes["diff"][: max(0, len(notes["diff"]) // 2)]
    if len(dumps()) > max_chars:
        notes = {
            "decision": DISAGREE,
            "reason": reason,
            "engines_attempted": engines_attempted,
            "truncated": "severe",
        }
    return dumps()
