"""pdfplumber backend: text-layer table extraction.

Benchmark winner for text-based PDFs (0.075s avg). Handles encrypted PDFs
natively. Returns 0 tables on scanned image-only PDFs (no text layer) — caller
falls back to OCR for those. Returns tables in the same format as camelot:
list of 2D string grids.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def extract_tables_with_pdfplumber(pdf_path: Path) -> list[list[list[str]]]:
    """Extract transaction tables using pdfplumber (fast text-layer parser).

    Benchmark winner for text-based PDFs: 0.075s/PDF avg, 8/12 success on
    text-based PTRs vs camelot's slower lattice/stream engines. Handles
    encrypted PDFs natively. Returns 0 tables on scanned image-only PDFs
    (no text layer) — caller should fall back to OCR for those.

    Returns tables in the same format as camelot: list of 2D string grids.
    """
    try:
        import pdfplumber
    except ImportError as e:
        logger.debug(f"pdfplumber not available: {e}")
        return []

    tables_out: list[list[list[str]]] = []
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page in pdf.pages:
                # page.extract_tables() returns list[list[list[str|None]]]
                page_tables = page.extract_tables()
                for tbl in page_tables:
                    cleaned = _clean_table(tbl)
                    if cleaned is not None:
                        tables_out.append(cleaned)
    except Exception as e:
        logger.debug(f"pdfplumber failed for {pdf_path}: {e}")
        return []

    return tables_out


def _clean_table(tbl: list[list[str | None]]) -> list[list[str]] | None:
    if not tbl or len(tbl) < 2:
        return None
    cleaned = [
        [("" if cell is None else str(cell).replace("\x00", "").strip()) for cell in row]
        for row in tbl
    ]
    # Drop fully-empty rows
    cleaned = [row for row in cleaned if any(c for c in row)]
    if len(cleaned) < 2:
        return None
    return cleaned
