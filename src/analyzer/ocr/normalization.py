"""Versioned, database-free OCR normalization; raw evidence is never mutated."""

from copy import deepcopy
from decimal import Decimal
import hashlib
import json
import math
import re

from analyzer.ocr.contracts import ExtractionLane, WorkEnvelope, ResultBundle
from analyzer.parsing.cells import extract_owner_code

NORMALIZATION_RECIPE = "house-ocr-normalization-v3"


def typed_instrument_fields(row):
    """SQL-compatible values; unknown is not a claim that an asset is stock."""
    instrument = row.get("instrument_type")
    if instrument is None or str(instrument).strip().casefold() in {"", "unknown"}:
        instrument = "unknown"
    strike = row.get("strike_price")
    if strike is None or str(strike).strip().casefold() in {"", "unknown"}:
        strike = None
    else:
        strike = float(strike)
        if not math.isfinite(strike):
            raise ValueError("nonfinite strike price")
    return {"instrument_type": instrument, "strike_price": strike}


CACHE_ROW_COLUMNS = (
    "source_row_id", "raw_asset_description", "raw_transaction_subtype",
    "transaction_date", "notification_date", "amount_raw", "owner_code", "raw_owner",
    "instrument_type", "strike_price", "expiry_date",
)


def canonical_row_projection(row):
    """The same typed projection for admitted rows and SQL cache comparisons."""
    from analyzer.ocr.common import _strict_date

    projected = {**row, **typed_instrument_fields(row)}
    for key in ("transaction_date", "notification_date", "expiry_date"):
        projected[key] = _strict_date(projected.get(key))
    return tuple(projected.get(key) for key in CACHE_ROW_COLUMNS)


def normalize_amount(value):
    from analyzer.ocr.common import AMOUNT_MIDPOINTS

    raw = str(value).strip() if value is not None else None
    result = dict(amount_raw=raw, amount_kind="unknown", amount_lower=None,
                  amount_upper=None, amount_exact=None, amount_midpoint=None)
    if not raw:
        return result
    if raw.upper() in AMOUNT_MIDPOINTS:
        result.update(amount_kind="band", amount_letter=raw.upper(),
                      amount_midpoint=AMOUNT_MIDPOINTS[raw.upper()])
        return result
    number = r"\$?((?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{1,2})?)"
    text = raw.replace("–", "-").replace("—", "-")
    exact = re.fullmatch(number, text)
    bounds = re.fullmatch(number + r"\s*(?:-|to)\s*" + number, text, re.I)
    opened = re.fullmatch(r"(?:over|greater than|>)\s*" + number, text, re.I)
    if exact:
        amount = str(Decimal(exact[1].replace(",", "")))
        result.update(amount_kind="exact", amount_exact=amount, amount_lower=amount,
                      amount_upper=amount, amount_midpoint=float(amount))
    elif bounds:
        low, high = [Decimal(v.replace(",", "")) for v in bounds.groups()]
        if low <= high:
            result.update(amount_kind="range", amount_lower=str(low), amount_upper=str(high),
                          amount_midpoint=float((low + high) / 2))
    elif opened:
        result.update(amount_kind="open", amount_lower=str(Decimal(opened[1].replace(",", ""))))
    return result


def normalize_rows(doc_id, member, transactions, filing_date, expected_member, artifact_sha256):
    """Shared admission/cache projection, preserving order and repeated occurrences."""
    from analyzer.ocr.common import apply_verified_house_corrections, validate_transactions, _strict_date

    raw = apply_verified_house_corrections(doc_id, artifact_sha256, deepcopy(transactions))
    filing = _strict_date(filing_date)
    if filing is None:
        return [], {"invalid_filing_date": len(raw) or 1}
    rows, findings = validate_transactions(doc_id, member, raw, filing_date, expected_member, as_of_date=filing)
    for occurrence, row in enumerate(rows, 1):
        row["raw_instrument_type"] = row.get("raw_instrument_type", row.get("instrument_type"))
        row["raw_strike_price"] = row.get("raw_strike_price", row.get("strike_price"))
        try:
            row.update(typed_instrument_fields(row))
        except (TypeError, ValueError, OverflowError):
            findings["invalid_strike_price"] = findings.get("invalid_strike_price", 0) + 1
        row["transaction_date"] = _strict_date(row.get("date") or row.get("transaction_date")).isoformat()
        row["date"] = row["transaction_date"]
        notification = _strict_date(row.get("notif_date") or row.get("notification_date"))
        row["notif_date"] = notification.isoformat() if notification else None
        row["notification_date"] = row["notif_date"]
        owner = row.get("raw_owner", row.get("owner", row.get("owner_code")))
        row["raw_owner"] = owner
        row["owner_code"] = extract_owner_code(owner)
        row["occurrence"] = occurrence
    return rows, findings


def complete_page_evidence(evidence, page_count):
    if not isinstance(evidence, dict) or type(evidence.get("page_count")) is not int:
        return False
    pages = evidence.get("page_outcomes")
    if evidence["page_count"] != page_count or page_count < 1 or not isinstance(pages, list) or len(pages) != page_count:
        return False
    return all(isinstance(page, dict) and type(page.get("page_number")) is int
               and page["page_number"] == number
               and page.get("outcome") in {"transactions", "verified_nontransaction", "unknown"}
               for number, page in enumerate(pages, 1))


def verified_empty_evidence(evidence):
    """Require trusted absence proof; model-reported empty pages remain unknown.

    Canonical evidence: page_count, page_outcomes[{page_number, outcome}], where
    outcome is transactions/verified_nontransaction/unknown. Only trusted source
    verification may set absence_verified=True, never a model's NO_TRANSACTIONS.
    """
    if not isinstance(evidence, dict):
        return False
    if evidence.get("absence_verified") is not True:
        return False
    count = evidence.get("page_count")
    if type(count) is not int or not complete_page_evidence(evidence, count):
        return False
    return all(page["outcome"] == "verified_nontransaction" for page in evidence["page_outcomes"])


def normalize_candidate(work: WorkEnvelope, result: ResultBundle) -> dict:
    if work != result.work:
        raise ValueError("result work identity does not match admission work")
    member = result.evidence.get("member", work.member)
    rows, findings = normalize_rows(work.doc_id, member, result.rows, work.filing_date,
                                    work.member, work.artifact_sha256)
    if not complete_page_evidence(result.evidence, work.page_count):
        findings["invalid_page_evidence"] = 1
    else:
        outcomes = result.evidence["page_outcomes"]
        for row in result.rows:
            page = row.get("page_number")
            if page is not None and (type(page) is not int or not 1 <= page <= work.page_count
                                     or outcomes[page - 1]["outcome"] == "verified_nontransaction"):
                findings["row_page_conflict"] = findings.get("row_page_conflict", 0) + 1
    schema = result.evidence.get("output_schema_version")
    legacy = type(schema) is not int or schema != 4 or any("raw_cells" not in row for row in result.rows)
    status = result.outcome.value
    if work.lane != ExtractionLane.LOCAL and legacy and status in {"candidate", "verified_empty"}:
        findings["obsolete_output_schema"] = len(result.rows) or 1
    if status not in {"candidate", "verified_empty", "needs_review", "blocked"}:
        status = "blocked"
    if findings or (result.rows and len(rows) != len(result.rows)):
        status = "needs_review"
    if not result.rows and status in {"candidate", "verified_empty"}:
        status = "verified_empty" if verified_empty_evidence(result.evidence) and result.evidence.get("page_count") == work.page_count else "needs_review"
    if result.rows and status == "verified_empty":
        status = "needs_review"
    digest = hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()
    return dict(normalization_recipe=NORMALIZATION_RECIPE, rows=rows,
                raw_rows=deepcopy(result.rows), findings=findings,
                page_outcomes=deepcopy(result.evidence.get("page_outcomes", [])),
                status=status, rowset_sha256=digest, legacy_limited=legacy)
