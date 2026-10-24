"""Resolve asset descriptions to unambiguous public-company tickers."""

from __future__ import annotations

import json
import os
import re
import secrets
import stat
import time
from functools import lru_cache
from urllib.error import URLError
from urllib.request import Request, urlopen

from analyzer.ocr.paths import DATA_DIR

_SEC_URL = "https://www.sec.gov/files/company_tickers.json"
_SEC_CACHE = DATA_DIR / "ticker_cache" / "company_tickers.json"
_CACHE_TTL_SECONDS = 7 * 24 * 60 * 60
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_O_NONBLOCK = getattr(os, "O_NONBLOCK", 0)

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


def _open_sec_cache_dir() -> int | None:
    if not _O_NOFOLLOW or not _O_NONBLOCK:
        return None
    try:
        _SEC_CACHE.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd = os.open(
            _SEC_CACHE.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | _O_NOFOLLOW,
        )
    except OSError:
        return None
    metadata = os.fstat(fd)
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or metadata.st_mode & 0o077
    ):
        os.close(fd)
        return None
    return fd


def _read_sec_cache() -> tuple[tuple[str, str], ...] | None:
    directory_fd = _open_sec_cache_dir()
    if directory_fd is None:
        return None
    fd = None
    try:
        fd = os.open(
            _SEC_CACHE.name,
            os.O_RDONLY | _O_NOFOLLOW | _O_NONBLOCK,
            dir_fd=directory_fd,
        )
        metadata = os.fstat(fd)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or metadata.st_mode & 0o077
            or time.time() - metadata.st_mtime >= _CACHE_TTL_SECONDS
        ):
            return None
        with os.fdopen(fd, encoding="utf-8") as source:
            fd = None
            payload = json.load(source)
    except (OSError, ValueError):
        return None
    finally:
        if fd is not None:
            os.close(fd)
        os.close(directory_fd)
    return _company_rows(payload)


def _write_sec_cache(payload: object) -> None:
    directory_fd = _open_sec_cache_dir()
    if directory_fd is None:
        return
    temp_name = f".{_SEC_CACHE.name}.{secrets.token_hex(8)}.tmp"
    fd = None
    try:
        fd = os.open(
            temp_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | _O_NOFOLLOW,
            0o600,
            dir_fd=directory_fd,
        )
        with os.fdopen(fd, "w", encoding="utf-8") as destination:
            fd = None
            json.dump(payload, destination)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(
            temp_name,
            _SEC_CACHE.name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        temp_name = ""
        os.fsync(directory_fd)
    except OSError:
        pass
    finally:
        if fd is not None:
            os.close(fd)
        if temp_name:
            try:
                os.unlink(temp_name, dir_fd=directory_fd)
            except OSError:
                pass
        os.close(directory_fd)


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
        _write_sec_cache(payload)
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
        r"\s*\[\s*(?:ACCOUNT|CUSTODIAN)\s*[:=][^\]]*\]"
        r"|\s*\(\s*(?:ACCOUNT|CUSTODIAN)\s*[:=][^)]*\)"
        r"|(?:^|\s+)(?:ACCOUNT|CUSTODIAN)\s*[:=].*$",
        "",
        asset_text,
        flags=re.IGNORECASE,
    )


def _resolve_abbreviated_company_name(normalized: str) -> str | None:
    words = normalized.split()
    if not words or any(len(word) < 4 for word in words):
        return None
    matches = {
        ticker
        for name, tickers in _sec_company_name_index().items()
        if len(company_words := name.split()) == len(words)
        and all(
            company_word.startswith(word)
            for word, company_word in zip(words, company_words, strict=True)
        )
        for ticker in tickers
    }
    return next(iter(matches)) if len(matches) == 1 else None


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
        return _resolve_abbreviated_company_name(normalized)
    matches = {t for tickers in uncovered for t in tickers}
    return next(iter(matches)) if len(matches) == 1 else None
