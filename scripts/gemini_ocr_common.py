#!/usr/bin/env python3
"""Shared Gemini OCR helpers for House PTR extraction scripts."""
from __future__ import annotations

import os
import subprocess
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from analyzer.member_names import canonical_member_key

MODEL = "gemini/gemini-3.1-flash-lite"
CACHE_DIR = "data/gemini_cache"

PROMPT = """This is a US House Periodic Transaction Report (PTR). The FIRST data row is an EXAMPLE labeled "Example: Mega Corp. Common Stock" - SKIP IT. Only extract REAL transactions below it.

Output format:
MEMBER: [full name of filer]
[asset name] | [Purchase/Sale/Exchange] | [MM/DD/YY] | [MM/DD/YY] | [amount range]

Amount ranges: A=$1K-15K, B=$15K-50K, C=$50K-100K, D=$100K-250K, E=$250K-500K, F=$500K-1M, G=$1M-5M, H=$5M-25M, I=$25M-50M, J=over $50M

One line per transaction. No markdown, no tables, no explanations."""


def cache_path(doc_id: str, cache_dir: str = CACHE_DIR) -> Path:
    return Path(cache_dir) / f"{doc_id}.txt"


def read_cached_response(doc_id: str, cache_dir: str = CACHE_DIR) -> str | None:
    path = cache_path(doc_id, cache_dir)
    if path.exists():
        return path.read_text()
    return None


def write_cached_response(doc_id: str, output: str, cache_dir: str = CACHE_DIR) -> None:
    path = cache_path(doc_id, cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(output)


def call_gemini(pdf_path: str, doc_id: str | None = None, refresh: bool = False,
                cache_dir: str = CACHE_DIR, timeout: int = 180) -> tuple[str | None, str]:
    """Call Gemini via llm -a, caching raw output by doc_id before parsing."""
    if doc_id and not refresh:
        cached = read_cached_response(str(doc_id), cache_dir)
        if cached is not None:
            return cached, ""
    try:
        result = subprocess.run(
            ["llm", "-m", MODEL, "-a", pdf_path, "-o", "temperature", "0", PROMPT],
            capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode != 0:
            return None, result.stderr.strip() or f"llm exited {result.returncode}"
        if doc_id:
            write_cached_response(str(doc_id), result.stdout, cache_dir)
        return result.stdout, ""
    except subprocess.TimeoutExpired:
        return None, "llm timed out"
    except Exception as exc:
        return None, str(exc)


def _parse_date(value) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%m/%d/%y", "%m/%d/%Y", "%m-%d-%y", "%m-%d-%Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def _tx_date(tx: dict):
    return tx.get("date") or tx.get("tx_date") or tx.get("transaction_date")


def _tx_type(tx: dict):
    return tx.get("type") or tx.get("transaction_type")


def _tx_amount(tx: dict):
    return tx.get("amount_letter") or tx.get("amount") or tx.get("amount_raw")


def _tx_asset(tx: dict):
    return tx.get("asset") or tx.get("asset_description") or ""


def validate_transactions(doc_id, member, transactions, filing_date, expected_member):
    """Validate OCR rows and return (valid_transactions, rejection_counts)."""
    rejections = defaultdict(int)
    raw_count = len(transactions)
    if raw_count > 300:
        return [], {"row_count_exceeds_cap": raw_count}

    filing = _parse_date(filing_date)
    start = filing - timedelta(days=400) if filing else None
    end = filing + timedelta(days=7) if filing else None

    effective_member = member or expected_member or "Unknown"
    if expected_member and canonical_member_key(member or "") != canonical_member_key(expected_member):
        effective_member = expected_member
        rejections["member_mismatch"] += 1

    seen = set()
    valid = []
    for tx in transactions:
        parsed_date = _parse_date(_tx_date(tx))
        if filing and start and end and (parsed_date is None or parsed_date < start or parsed_date > end):
            rejections["date_out_of_window"] += 1
            continue
        duplicate_key = (
            str(_tx_date(tx) or "").strip(),
            str(_tx_type(tx) or "").strip(),
            str(_tx_amount(tx) or "").strip(),
            str(_tx_asset(tx) or "").strip(),
        )
        if duplicate_key in seen:
            rejections["duplicate_collapsed"] += 1
            continue
        seen.add(duplicate_key)
        cleaned = dict(tx)
        cleaned["member"] = effective_member
        valid.append(cleaned)

    return valid, dict(rejections)
