"""pdftotext backend: encrypted-PDF table extraction via system binary.

Uses `pdftotext -layout` to preserve column alignment, then parses the
structured text output into transaction rows. Two regex patterns cover
lines with and without an owner-code prefix; both fold multi-line asset
descriptions back into a single row.
"""

import logging
import re
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


_PDFTOTEXT_TIMEOUT = 15
_MAX_SINGLE_LETTER_LEN = 4

# Single-letter patterns removed — they caused false skips on transaction
# lines like "TripleBlind..." (starts with T) and "Treasury..." (starts with T).
_SKIP_PREFIXES = (
    'ID', 'Owner', 'Asset', 'Transaction', 'Date', 'Type',
    'Notification', 'Amount', 'Cap.', 'Gains', 'CERTIFY',
    'I CERTIFY', 'Digitally', 'Filing', 'Clerk', 'PERIODIC',
    'Name:', 'Status:', 'State/District:',
)
_SKIP_EXACT = {'ID', 'F', 'I', 'P', 'T', 'R', 'Cap.', 'Gains', 'CERTIFY'}

# Transaction type pattern (shared by both regex flavors)
_TX_TYPE = r'(?:S|P|E)(?:\s*\(partial\))?'
# Amount pattern (handles split amounts across lines)
_AMOUNT = r'(?:\$[\d,]+(?:\s*-\s*\$[\d,]+)?|[\-]+\$[\d,]+)'

_TX_WITH_OWNER = re.compile(
    r'^\s{2,}'
    r'([A-Z]{1,4})\s+'           # owner code
    r'(.+?)\s+'                  # asset name
    rf'({_TX_TYPE})\s+'          # type
    r'(\d{2}/\d{2}/\d{4})\s+'    # tx date
    r'(\d{2}/\d{2}/\d{4})\s+'    # notif date
    rf'({_AMOUNT})'              # amount
)

_TX_NO_OWNER = re.compile(
    r'^\s{0,30}'
    r'(.+?)\s+'                  # asset name
    rf'({_TX_TYPE})\s+'          # type
    r'(\d{2}/\d{2}/\d{4})\s+'    # tx date
    r'(\d{2}/\d{2}/\d{4})\s+'    # notif date
    rf'({_AMOUNT})'              # amount
)

_TX_CODE_INLINE = re.compile(r'\b(?:S|P|E)(?:\s*\(partial\))?\b')
_TICKER_PARENS = re.compile(r'\([A-Za-z][A-Za-z0-9.\-]{0,5}\)')
_OWNER_TX_HEAD = re.compile(r'^[A-Z]{1,4}\s+\S')
_OWNER_PREFIX = re.compile(r'^[A-Z]{1,4}\s+\S')


def extract_tables_with_pdftotext(pdf_path: Path) -> list[list[list[str]]]:
    """Extract transaction tables using pdftotext (handles encrypted PDFs).

    Uses pdftotext -layout to preserve column alignment, then parses the
    structured output to extract transaction rows.
    """
    text = _run_pdftotext(pdf_path)
    if text is None:
        return []

    lines = text.split('\n')
    transactions = _parse_pdftotext_lines(lines)
    if not transactions:
        return []

    table = [['Asset Name', 'Transaction Type', 'Transaction Date', 'Amount']] + transactions
    return [table]


def _run_pdftotext(pdf_path: Path) -> str | None:
    try:
        result = subprocess.run(
            ['pdftotext', '-layout', str(pdf_path), '-'],
            capture_output=True, text=True, timeout=_PDFTOTEXT_TIMEOUT
        )
        if result.returncode != 0 or not result.stdout.strip():
            return None
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        logger.debug(f"pdftotext failed for {pdf_path}: {e}")
        return None
    return result.stdout


def _parse_pdftotext_lines(lines: list[str]) -> list[list[str]]:
    transactions: list[list[str]] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if _is_skip_line(line):
            i += 1
            continue

        matched = _try_match_with_owner(line, i, lines)
        if matched is not None:
            asset, tx_type, tx_date, amount, j = matched
            transactions.append([asset, tx_type, tx_date, amount])
            i = j
            continue

        matched = _try_match_no_owner(line, i, lines)
        if matched is not None:
            asset, tx_type, tx_date, amount, j = matched
            transactions.append([asset, tx_type, tx_date, amount])
            i = j
            continue

        i += 1
    return transactions


def _is_skip_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) <= _MAX_SINGLE_LETTER_LEN:
        return True
    return any(stripped.startswith(s) for s in _SKIP_PREFIXES)


def _try_match_with_owner(line: str, i: int, lines: list[str]) -> tuple[str, str, str, str, int] | None:
    m = _TX_WITH_OWNER.match(line)
    if not m:
        return None
    owner, asset, tx_type, tx_date, notif_date, amount = m.groups()
    asset = asset.strip()
    asset, j = _collect_asset_continuation(asset, i + 1, lines)
    return asset, tx_type, tx_date, amount, j


def _try_match_no_owner(line: str, i: int, lines: list[str]) -> tuple[str, str, str, str, int] | None:
    m = _TX_NO_OWNER.match(line)
    if not m:
        return None
    asset, tx_type, tx_date, notif_date, amount = m.groups()
    asset = asset.strip()
    # Skip lines where "asset" is actually a header/metadata
    if asset in _SKIP_EXACT:
        return None
    asset, j = _collect_asset_continuation(asset, i + 1, lines)
    return asset, tx_type, tx_date, amount, j


def _collect_asset_continuation(asset: str, start: int, lines: list[str]) -> tuple[str, int]:
    """Fold continuation lines (e.g. 'Stock (NVDA) [ST]', '[GS]' markers) into the asset name.

    Stops on the next transaction header (owner + tx code) or numeric date line.
    Returns (folded_asset, next_index_after_consumed_lines).
    """
    j = start
    while j < len(lines):
        next_line = lines[j].strip()
        if not next_line:
            break
        if _is_new_tx_header(next_line):
            break
        if re.match(r'^\[', next_line) or re.match(r'^\d', next_line):
            asset += ' ' + next_line
            j += 1
        elif _TICKER_PARENS.search(next_line):
            asset += ' ' + next_line
            j += 1
        else:
            break
    return asset, j


def _is_new_tx_header(next_line: str) -> bool:
    return bool(_OWNER_TX_HEAD.match(next_line) and _TX_CODE_INLINE.search(next_line))
