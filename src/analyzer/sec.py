"""Shared SEC request and filing-header contracts."""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime
from urllib.parse import urlsplit

import requests

SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data"
REQUEST_TIMEOUT = 30
REQUEST_INTERVAL = 1 / 8
MAX_RETRIES = 5
RETRY_BASE_SECONDS = 1.0
_ACCESSION_RE = re.compile(r"\d{10}-\d{2}-\d{6}\Z")

logger = logging.getLogger("analyzer.form4")


def accession_digits(accession):
    accession = str(accession)
    if not _ACCESSION_RE.fullmatch(accession):
        raise ValueError(f"Invalid SEC accession: {accession!r}")
    return accession.replace("-", "")


class SECClient:
    def __init__(self, user_agent, headers):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent, **headers})
        self._last_request_at = 0.0

    def close(self):
        self.session.close()

    def get(self, url, error_type):
        for attempt in range(MAX_RETRIES):
            delay = REQUEST_INTERVAL - (time.monotonic() - self._last_request_at)
            if delay > 0:
                time.sleep(delay)
            self._last_request_at = time.monotonic()
            try:
                response = self.session.get(url, timeout=REQUEST_TIMEOUT)
            except requests.RequestException as exc:
                if attempt + 1 == MAX_RETRIES:
                    raise error_type(f"SEC request failed for {url}: {exc}") from exc
                time.sleep(RETRY_BASE_SECONDS * (2**attempt))
                continue
            if response.status_code in (429, 503):
                if attempt + 1 == MAX_RETRIES:
                    raise error_type(
                        f"SEC rate-limited {url} with HTTP {response.status_code} "
                        f"after {MAX_RETRIES} attempts"
                    )
                retry_after = response.headers.get("Retry-After")
                try:
                    backoff = (
                        float(retry_after)
                        if retry_after
                        else RETRY_BASE_SECONDS * (2**attempt)
                    )
                except ValueError:
                    backoff = RETRY_BASE_SECONDS * (2**attempt)
                logger.warning(
                    "SEC HTTP %s for %s; retrying in %.1fs",
                    response.status_code,
                    url,
                    backoff,
                )
                time.sleep(max(backoff, 0))
                continue
            if response.status_code != 200:
                raise error_type(
                    f"SEC returned HTTP {response.status_code} for {url}"
                )
            final = urlsplit(response.url)
            expected = urlsplit(url)
            if final.scheme != "https" or final.hostname != expected.hostname:
                raise error_type(
                    f"SEC request redirected outside the official host: {url}"
                )
            return response
        raise error_type(f"SEC retry limit reached for {url}")

    def submissions(self, cik, error_type):
        url = SEC_SUBMISSIONS_URL.format(cik=str(int(cik)).zfill(10))
        try:
            data = self.get(url, error_type).json()
        except (ValueError, requests.RequestException) as exc:
            raise error_type(
                f"SEC submissions index for CIK {cik} is invalid"
            ) from exc
        if not isinstance(data, dict):
            raise error_type(f"SEC submissions index for CIK {cik} is not an object")
        return data


def acceptance_datetime(*, content: bytes = b"") -> datetime | None:
    """Authoritative submission-header clock, naive SEC Eastern local time.

    Do not convert submissions JSON's discrepant acceptanceDateTime field.
    Never substitute filingDate or insertion time when the header is absent.
    """
    match = re.search(
        rb"(?:<ACCEPTANCE-DATETIME>|ACCEPTANCE-DATETIME\t)\s*(\d{14})", content
    )
    if match:
        return datetime.strptime(match[1].decode("ascii"), "%Y%m%d%H%M%S")
    return None
