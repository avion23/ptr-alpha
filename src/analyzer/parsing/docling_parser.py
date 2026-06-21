"""Docling OCR backend: scanned-PDF table extraction via subprocess.

Docling (IBM TableFormer + RapidOCR) is the benchmark winner for scanned
image-only PTR PDFs that pdfplumber cannot touch. Runs in an isolated
`uvx` subprocess so its ~500MB ML dependencies don't pollute the project
venv. Emits markdown that we convert back to 2D string grids via the
list-style (`_parse_docling_markdown`) or pipe-table (`_parse_markdown_tables`)
parsers below.
"""

import logging
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

_MAX_LOOKAHEAD_LINES = 8


def extract_tables_with_docling(pdf_path: Path, timeout: int = 300) -> list[list[list[str]]]:
    """OCR fallback using Docling (TableFormer + RapidOCR).

    Benchmark winner for scanned image PDFs: handles the 4 large 2024-2025
    scanned filings (8220660/0674/1212/0750, ~263 transactions) that no
    text-layer parser can touch. MIT license, IBM-backed. Runs as isolated
    subprocess via `uvx` so it does not pollute the project venv with its
    ~500MB ML model dependencies.

    Returns markdown-derived tables in the same 2D string grid format.
    """
    cmd = _build_docling_cmd(pdf_path)
    if cmd is None:
        return []

    with tempfile.TemporaryDirectory(prefix="docling_") as out_dir:
        full_cmd = cmd + [out_dir]
        text = _run_docling(full_cmd, pdf_path, timeout)
        if text is None:
            return []

        # For scanned PDFs, Docling renders transactions as markdown lists
        # (each tx spans 3-5 lines). The list parser handles this format;
        # the pipe-table parser is a fallback for cleaner digital PDFs.
        tables = _parse_docling_markdown(text)
        if not tables:
            tables = _parse_markdown_tables(text)
        return tables


def _build_docling_cmd(pdf_path: Path) -> list[str] | None:
    # Always prefer uvx (isolated env) — the system docling binary often has
    # OpenMP/runtime conflicts. Fall back to system docling only if no uvx.
    uvx = shutil.which("uvx")
    if uvx:
        return ["uvx", "--from", "docling", "docling", "convert",
                str(pdf_path), "--to", "md", "--output"]
    if shutil.which("docling"):
        return ["docling", "convert", str(pdf_path), "--to", "md", "--output"]
    logger.debug("Neither uvx nor docling on PATH — skipping Docling OCR")
    return None


def _run_docling(full_cmd: list[str], pdf_path: Path, timeout: int) -> str | None:
    try:
        result = subprocess.run(full_cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            logger.debug(f"docling failed for {pdf_path}: rc={result.returncode}, stderr tail={result.stderr[-300:]}")
            return None
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.debug(f"docling timed out/missing for {pdf_path}: {e}")
        return None

    # Docling emits <stem>.md and <stem>.json in the output dir
    md_files = list(Path(full_cmd[-1]).rglob("*.md"))
    if not md_files:
        return None
    return md_files[0].read_text(encoding="utf-8", errors="ignore")


def _parse_markdown_tables(text: str) -> list[list[list[str]]]:
    """Parse GitHub-flavored markdown tables into 2D string grids.

    Docling and Marker emit standard markdown pipe tables; this converts them
    back to the row/column grid that parse_pdf_table() expects.
    """
    tables: list[list[list[str]]] = []
    current: list[list[str]] = []

    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line.startswith("|"):
            if current:
                tables.append(current)
                current = []
            continue

        # Split on |, drop leading/trailing empty cells from outer pipes
        parts = [c.strip() for c in line.strip("|").split("|")]
        # Skip markdown separator rows: |---|---|---|
        if all(re.fullmatch(r":?-{2,}:?", p) for p in parts if p):
            continue
        # Replace None/empty with empty string, strip null bytes
        parts = [("" if p is None else p.replace("\x00", "")) for p in parts]
        current.append(parts)

    if current:
        tables.append(current)

    # Only keep tables that look like PTR data (>=2 rows, >=3 cols)
    return [t for t in tables if len(t) >= 2 and len(t[0]) >= 3]


def _parse_docling_markdown(text: str) -> list[list[list[str]]]:
    """Parse Docling's list-style markdown OCR output into transaction tables.

    Docling renders scanned PTR PDFs as markdown lists where each transaction
    spans 3-5 lines:
      Line 1: "- SP Asset Name (TICKER) [ST]"
      Line 2: "S" or "P" or "·S (partial)"      (transaction code)
      Line 3: "09/25/2025 10/01/2025 $1,001-$15,000"  (date + amount)

    This parser scans for ticker lines, then looks ahead for tx code + dates.
    Returns one synthetic table consumable by parse_pdf_table().
    """
    lines = text.split("\n")
    rows: list[list[str]] = []

    ticker_re = re.compile(r'(.+?)\s*\(([A-Za-z][A-Za-z0-9.\-]{0,5})\)\s*(\[[A-Z]+\])?')
    tx_code_re = re.compile(r'^[·\-\s]*([PSE])\s*(\(partial\))?\s*$', re.IGNORECASE)
    date_amount_re = re.compile(
        r'(\d{2}/\d{2}/\d{4})'                    # tx date
        r'(?:\s+\d{2}/\d{2}/\d{4})?'              # optional notification date
        r'\s*(\$[\d,]+\s*-\s*\$[\d,]+|\$[\d,]+)?'  # optional amount
    )

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        m = ticker_re.search(line)
        if not m or '[ST]' not in line and not re.search(r'\([A-Z]{1,6}\)', line):
            i += 1
            continue

        # Extract asset name and ticker
        asset_name = line.lstrip('- ').strip()

        tx_type, tx_date, amount, next_i = _scan_forward_for_tx(
            lines, i, ticker_re, tx_code_re, date_amount_re
        )
        if tx_type and tx_date:
            rows.append([asset_name, tx_type, tx_date, amount or ""])
        i = next_i

    if not rows:
        return []
    header = ['Asset Name', 'Transaction Type', 'Transaction Date', 'Amount']
    table = [header] + rows  # one table (list of rows)
    return [table]  # list of tables


def _scan_forward_for_tx(lines, i, ticker_re, tx_code_re, date_amount_re):
    """Look ahead up to _MAX_LOOKAHEAD_LINES from index i for tx code/date/amount.

    Returns (tx_type, tx_date, amount, next_index).
    """
    tx_type = None
    tx_date = None
    amount = None
    for j in range(i + 1, min(i + 1 + _MAX_LOOKAHEAD_LINES, len(lines))):
        lookahead = lines[j].strip()
        if not lookahead:
            continue
        # Stop if we hit the next ticker line
        if ticker_re.search(lookahead) and re.search(r'\([A-Z]{1,6}\)', lookahead):
            break

        # Check for tx code (standalone P/S/E)
        if tx_type is None:
            tm = tx_code_re.match(lookahead)
            if tm:
                code = tm.group(1).upper()
                partial = tm.group(2) is not None
                tx_type = f"{code} {'(partial)' if partial else ''}".strip()

        # Check for date + amount
        if tx_date is None:
            dm = date_amount_re.search(lookahead)
            if dm:
                tx_date = dm.group(1)
                if dm.group(2):
                    amount = dm.group(2)

        if tx_type and tx_date:
            break
    return tx_type, tx_date, amount, i + 1
