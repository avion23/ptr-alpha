"""Resolve asset descriptions to unambiguous public-company tickers."""

from __future__ import annotations

import json
import os
import re
import time
from functools import lru_cache
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

_SEC_URL = "https://www.sec.gov/files/company_tickers.json"
_SEC_CACHE = Path("/tmp/ptr-alpha-company-tickers.json")
_CACHE_TTL_SECONDS = 7 * 24 * 60 * 60

# Company-name fragments for common disclosure spellings that do not match an
# active SEC registrant title exactly. Ambiguous names and investment products
# are intentionally absent.
_CURATED_ALIASES = {
    "ZOETIS": "ZTS",
    "BIO TECHNE": "TECH",
    "DENNYS": "DENN",
    "VISHAY PRECISION GROUP": "VPG",
    "BANK OF AMERICA": "BAC",
    "AXALTA COATING SYSTEMS": "AXTA",
    "PAR PETROLEUM": "PARR",
    "MICRON TECHNOLOGY": "MU",
    "DOORDASH": "DASH",
    "GE HEALTHCARE": "GEHC",
    "ALLSTATE": "ALL",
    "AMAZON": "AMZN",
    "URBAN OUTFITTERS": "URBN",
    "CTS CORP": "CTS",
    "ABBOTT LABORATORIES": "ABT",
    "BOOKING HOLDINGS": "BKNG",
    "AMERICAN TOWER": "AMT",
    "SHERWIN WILLIAMS": "SHW",
    "FEDERAL SIGNAL": "FSS",
    "LOWES": "LOW",
    "REGENERON PHARMACEUTICALS": "REGN",
    "BAXTER INTERNATIONAL": "BAX",
    "PFIZER": "PFE",
    "COMCAST": "CMCSA",
    "DANAHER": "DHR",
    "INTUITIVE SURGICAL": "ISRG",
    "GENERAL MOTORS": "GM",
    "SCHLUMBERGER": "SLB",
    "ELANCO ANIMAL HEALTH": "ELAN",
    "LINDE": "LIN",
    "TAKE TWO INTERACTIVE": "TTWO",
    "ANTERO MIDSTREAM": "AM",
    "QUALCOMM": "QCOM",
    "CVS HEALTH": "CVS",
    "FIDELITY NATIONAL INFORMATION": "FIS",
    "INTUIT": "INTU",
    "WALMART": "WMT",
    "NETFLIX": "NFLX",
    "TESLA": "TSLA",
    "SALESFORCE": "CRM",
    "UNION PACIFIC": "UNP",
    "PLAINS ALL AMERICAN PIPELINE": "PAA",
    "MERCK & CO": "MRK",
    "UNITEDHEALTH GROUP": "UNH",
    "JOHNSON & JOHNSON": "JNJ",
    "COGNIZANT TECHNOLOGY SOLUTIONS": "CTSH",
    "CATERPILLAR": "CAT",
    "VISA": "V",
    "JPMORGAN CHASE": "JPM",
}
_UNRESOLVABLE_TERMS = (
    "TREASURY",
    "T BILL",
    "T NOTE",
    "SWEEP",
    "MONEY MARKET",
    "MUTUAL FUND",
    "INDEX FUND",
    "INVESTMENT FUND",
    "ETF",
    "ETN",
    "FUND",
    "CASH",
    "BOND",
    "NOTE",
    "MTN",
    "FD",
)


_ENTITY_SUFFIXES = (
    "INC",
    "INCORPORATED",
    "CORP",
    "CORPORATION",
    "COMPANY",
    "COS",
    "LLC",
    "LTD",
    "LIMITED",
    "PLC",
    "HOLDINGS",
    "HOLDING",
    "GROUP",
    "LP",
    "LLP",
    "PA",
    "NA",
)


def _normalize_company_name(value: str) -> str:
    """Normalize punctuation and remove terminal security-class descriptors."""
    normalized = re.sub(r"[^A-Z0-9]+", " ", value.upper().replace("&", " AND ")).strip()
    while True:
        without_suffix = re.sub(
            r"\s+(?:CMN|COMMON STOCK|CLASS [ABC])$", "", normalized
        ).strip()
        without_suffix = re.sub(r"\s*\([A-Z]\)$", "", without_suffix).strip()
        tokens = without_suffix.split()
        if len(tokens) > 1 and tokens[-1] in _ENTITY_SUFFIXES:
            without_suffix = " ".join(tokens[:-1])
        if without_suffix == normalized:
            return normalized
        normalized = without_suffix


def _read_sec_cache() -> tuple[tuple[str, str], ...] | None:
    try:
        if time.time() - _SEC_CACHE.stat().st_mtime >= _CACHE_TTL_SECONDS:
            return None
        payload = json.loads(_SEC_CACHE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return _company_rows(payload)


def _company_rows(payload: object) -> tuple[tuple[str, str], ...]:
    if not isinstance(payload, dict):
        return ()
    return tuple(
        (entry["title"], entry["ticker"])
        for entry in payload.values()
        if isinstance(entry, dict)
        and isinstance(entry.get("title"), str)
        and isinstance(entry.get("ticker"), str)
    )


@lru_cache(maxsize=1)
def _load_sec_company_tickers() -> tuple[tuple[str, str], ...]:
    cached = _read_sec_cache()
    if cached is not None:
        return cached
    request = Request(
        _SEC_URL,
        headers={
            "User-Agent": os.environ.get("SEC_USER_AGENT")
            or "ptr-alpha research contact@example.com"
        },
    )
    try:
        with urlopen(request, timeout=20) as response:
            payload = json.load(response)
    except (OSError, URLError, TimeoutError, json.JSONDecodeError):
        return ()
    rows = _company_rows(payload)
    if rows:
        try:
            _SEC_CACHE.write_text(json.dumps(payload), encoding="utf-8")
        except OSError:
            pass
    return rows


@lru_cache(maxsize=1)
def _sec_company_name_index() -> dict[str, frozenset[str]]:
    index: dict[str, set[str]] = {}
    for title, ticker in _load_sec_company_tickers():
        normalized = _normalize_company_name(title)
        if normalized:
            index.setdefault(normalized, set()).add(ticker.strip().upper())
    return {name: frozenset(tickers) for name, tickers in index.items()}


def _has_unresolvable_instrument(normalized: str) -> bool:
    if normalized == "JT":
        return True
    return any(
        re.search(rf"\b{re.escape(term)}S?\b", normalized)
        for term in _UNRESOLVABLE_TERMS
    )


def _without_account_metadata(asset_text: str) -> str:
    return re.sub(
        r"\s*\[\s*(?:ACCOUNT|CUSTODIAN)\s*:[^\]]*\]"
        r"|\s+(?:ACCOUNT|CUSTODIAN)\s*[:=].*$",
        "",
        asset_text,
        flags=re.IGNORECASE,
    )


def resolve_asset_ticker(asset_text: str) -> str | None:
    """Resolve an asset description only when its company identity is unique."""
    if not isinstance(asset_text, str) or not asset_text.strip():
        return None
    normalized = _normalize_company_name(_without_account_metadata(asset_text))
    if not normalized or _has_unresolvable_instrument(normalized):
        return None

    padded = f" {normalized} "
    spans: list[tuple[int, int, set[str]]] = []
    for name, tickers in _sec_company_name_index().items():
        needle = f" {name} "
        start = padded.find(needle)
        while start != -1:
            spans.append((start, start + len(needle), set(tickers)))
            start = padded.find(needle, start + 1)
    for alias, ticker in _CURATED_ALIASES.items():
        needle = f" {_normalize_company_name(alias)} "
        start = padded.find(needle)
        while start != -1:
            spans.append((start, start + len(needle), {ticker}))
            start = padded.find(needle, start + 1)
    uncovered = [
        tickers
        for i, (s, e, tickers) in enumerate(spans)
        if not any(
            (os, oe) != (s, e) and os <= s and e <= oe
            for j, (os, oe, _) in enumerate(spans)
            if j != i
        )
    ]
    if not uncovered:
        return None
    matches = {t for tickers in uncovered for t in tickers}
    return next(iter(matches)) if len(matches) == 1 else None
