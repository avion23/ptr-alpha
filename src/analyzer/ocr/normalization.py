"""Versioned, database-free OCR normalization; raw evidence is never mutated."""

from copy import deepcopy
from decimal import Decimal
import hashlib
import json
import re

from analyzer.ocr.contracts import WorkEnvelope, ResultBundle
from analyzer.parsing.cells import extract_owner_code

NORMALIZATION_RECIPE = "house-ocr-normalization-v1"


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


def verified_empty_evidence(evidence):
    """Empty admission requires positive page outcomes, not a zero row count."""
    if not isinstance(evidence, dict):
        return False
    count = evidence.get("page_count")
    pages = evidence.get("page_outcomes")
    if type(count) is not int or count < 1 or not isinstance(pages, list) or len(pages) != count:
        return False
    return all(isinstance(p, dict) and p.get("page_number") == i
               and p.get("outcome") == "no_transactions" for i, p in enumerate(pages, 1))


def normalize_candidate(work: WorkEnvelope, result: ResultBundle) -> dict:
    if work != result.work:
        raise ValueError("result work identity does not match admission work")
    member = result.evidence.get("member", work.member)
    rows, findings = normalize_rows(work.doc_id, member, result.rows, work.filing_date,
                                    work.member, work.artifact_sha256)
    legacy = any("raw_cells" not in row for row in result.rows)
    status = result.outcome.value
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
