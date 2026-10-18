#!/usr/bin/env python3
"""Production OCR pipeline for 425 zero-row scanned PTR PDFs.

Uses `llm -a` with Gemini 3.1 Flash Lite to extract transactions.
Gemini auto-rotates PDFs and handles checkbox detection.

Engine modes: DUAL_READ=1 forces the dual-read gate (Gemini gated against
engine B); otherwise OCR_ENGINE selects — unset/'gemini' (single-engine
Gemini, the default; the ONLY mode that may persist A-only rows), 'visionb'
(GLM-primary single engine: reads land as STAGING records under
data/staging/glm-backlog — canonical ingestion of GLM rows happens ONLY via
``glm_stage promote`` after a passing audit), or 'dual' (alias of
DUAL_READ=1).
"""

import argparse
import datetime
import json
import os
import re
import time
from dataclasses import dataclass, replace

import duckdb
from pathlib import Path

from analyzer.database import Database
from analyzer.vision_engines import (
    A_ERROR,
    DualReadDoc,
    UNRESOLVED,
    VisionBConfig,
    VisionBUnavailable,
    dual_read_enabled,
    read_document_with_engine_b,
    run_dual_read_for_doc,
)
from scripts.gemini_ocr_common import (
    apply_verified_house_corrections,
    ArtifactMetadata,
    GEMINI_PARSER_VERSION,
    MODEL,
    GeminiOutputError as GeminiOutputError,
    call_gemini,
    inspect_cached_response,
    is_quota_error,
    parse_gemini_output,
    snapshot_pdf,
    validate_transactions,
)

DB_PATH = "data/congress.duckdb"
PROGRESS_PATH = "data/ocr_progress_gemini_manual.json"
COOLDOWN = 3  # seconds between requests (Lite model allows rapid fire)
# Engine-B-primary single-engine drain (OCR_ENGINE=visionb): GLM reads the
# backlog alone — engine A (Gemini) is never contacted, conserving its free
# quota for current-era docs. Readings land as STAGING records
# (data/staging/glm-backlog, via scripts.glm_stage); canonical ingestion of
# GLM rows happens ONLY through ``glm_stage promote`` after the pixel audit,
# and promote inserts through this module's gemini_ocr source family, insert
# path, and validation.
GLM_PARSER_VERSION = "v1-glm53-vision"
GLM_ENGINES_ATTEMPTED = "visionb"
OCR_ENGINE_ENV = "OCR_ENGINE"
REQUIRED_OCR_SCHEMA_COLUMNS = {
    "chamber",
    "source_record_id",
    "source_row_id",
    "official_filing_date",
    "available_date",
    "notification_date",
    "amends_source_record_id",
    "raw_transaction_subtype",
    "ticker_origin",
    "raw_ticker",
    "ticker_candidate",
    "raw_asset_class",
    "raw_owner",
    "raw_asset_description",
    "ingestion_generation",
    "artifact_sha256",
}

OCR_INSERT_COLUMNS = [
    "doc_id",
    "member",
    "ticker",
    "transaction_date",
    "disclosure_date",
    "transaction_type",
    "amount_raw",
    "amount_midpoint",
    "owner_code",
    "asset_description",
    "source",
    "chamber",
    "source_record_id",
    "source_row_id",
    "official_filing_date",
    "available_date",
    "notification_date",
    "amends_source_record_id",
    "raw_transaction_subtype",
    "ticker_origin",
    "raw_ticker",
    "ticker_candidate",
    "raw_asset_class",
    "raw_asset_description",
    "raw_owner",
    "ingestion_generation",
    "artifact_sha256",
]


def validate_ticker_provenance(row):
    origin = row["ticker_origin"]
    ticker = row["ticker"]
    raw_ticker = row["raw_ticker"]
    candidate = row["ticker_candidate"]
    if origin == "official":
        if not ticker or ticker != raw_ticker or candidate is not None:
            raise ValueError("official ticker provenance is inconsistent")
        return
    if origin == "asset_description":
        if not ticker or raw_ticker is not None or candidate is not None:
            raise ValueError("asset-description ticker provenance is inconsistent")
        return
    if origin == "unverified":
        if ticker is not None or not raw_ticker or candidate != raw_ticker:
            raise ValueError("unverified ticker provenance is inconsistent")
        return
    if origin == "not_reported":
        if ticker is not None or raw_ticker is not None or candidate is not None:
            raise ValueError("not-reported ticker provenance is inconsistent")
        return
    raise ValueError(f"invalid ticker_origin: {origin}")


def require_ocr_schema(conn):
    existing = {
        row[1] for row in conn.execute("PRAGMA table_info('transactions')").fetchall()
    }
    missing = sorted(REQUIRED_OCR_SCHEMA_COLUMNS - existing)
    if missing:
        raise RuntimeError(
            "OCR ingestion requires the provenance schema; missing columns: "
            + ", ".join(missing)
        )
    return existing


# Amount range midpoint estimates (for amount_midpoint column)
AMOUNT_MIDPOINTS = {
    "A": 8000,
    "B": 32500,
    "C": 75000,
    "D": 175000,
    "E": 375000,
    "F": 750000,
    "G": 3000000,
    "H": 15000000,
    "I": 37500000,
    "J": 50000000,
    "K": 1000000,
}


def get_ocr_work_items(
    *,
    db_path: str = DB_PATH,
    data_dir: str | Path | None = None,
    year: int | None = None,
    parser_version: str = GEMINI_PARSER_VERSION,
    require_schema: bool = True,
):
    """Select work from current artifacts, validated cache, schema, and DB rows."""
    conn = duckdb.connect(db_path, read_only=True)
    existing_columns = {
        row[1] for row in conn.execute("PRAGMA table_info('transactions')").fetchall()
    }
    schema_ready = REQUIRED_OCR_SCHEMA_COLUMNS <= existing_columns
    if require_schema and not schema_ready:
        require_ocr_schema(conn)
    base = Path(data_dir) if data_dir is not None else Path(db_path).parent
    generation_work = []
    if schema_ready and year is not None:
        conn.close()
        db = Database(db_path, read_only=True)
        try:
            generation = db.get_latest_house_generation(year)
            scoped_ptrs = 0
            if generation is not None:
                scoped_ptrs = int(
                    db.conn.execute(
                        """
                        SELECT COUNT(*) FROM house_generation_metadata
                        WHERE archive_year = ? AND generation_id = ?
                          AND filing_type = 'P'
                        """,
                        [year, generation],
                    ).fetchone()[0]
                )
            if generation is not None and scoped_ptrs > 0:
                unresolved = db.get_unresolved_house_doc_ids(year, generation)
                generation_work = [
                    (
                        doc_id,
                        year,
                        str(base / str(year) / "pdfs" / f"{doc_id}.pdf"),
                    )
                    for doc_id in unresolved
                ]
        finally:
            db.close()
        conn = duckdb.connect(db_path, read_only=True)

    if not schema_ready:
        legacy_rows = conn.execute(
            """
            WITH deterministic AS (
                SELECT *, ROW_NUMBER() OVER (
                    PARTITION BY doc_id ORDER BY parsed_at DESC
                ) AS rn
                FROM pdf_parse_runs
                WHERE NOT (
                    LOWER(COALESCE(parser_version, '')) LIKE '%gemini%'
                    OR LOWER(COALESCE(parser_version, '')) LIKE '%ling%'
                    OR LOWER(COALESCE(parser_version, '')) LIKE '%gpt%'
                    OR LOWER(COALESCE(parser_version, '')) LIKE '%openrouter%'
                    OR LOWER(COALESCE(engines_attempted, '')) LIKE 'gemini/%'
                    OR LOWER(COALESCE(engines_attempted, '')) LIKE 'openrouter/%'
                    OR LOWER(COALESCE(engines_attempted, '')) LIKE 'gpt-%'
                )
            )
            SELECT CAST(m.doc_id AS VARCHAR),
                   CAST(EXTRACT(YEAR FROM m.filing_date) AS INTEGER)
            FROM metadata m
            JOIN deterministic d ON d.doc_id = m.doc_id AND d.rn = 1
            WHERE m.filing_type = 'P'
              AND d.status IN ('zero_rows', 'error', 'rejected')
              AND (? IS NULL OR EXTRACT(YEAR FROM m.filing_date) = ?)
            ORDER BY EXTRACT(YEAR FROM m.filing_date), m.doc_id
            """,
            [year, year],
        ).fetchall()
        conn.close()
        return [
            (
                doc_id,
                doc_year,
                str(base / str(doc_year) / "pdfs" / f"{doc_id}.pdf"),
            )
            for doc_id, doc_year in legacy_rows
        ]
    rows = conn.execute(
        """
        WITH deterministic AS (
            -- Dual-read runs (parser_version v6-dual-validated, engines
            -- 'gemini+%') intentionally count as deterministic recency here:
            -- after an AGREE (success) or DISAGREE (invalid) decision the doc
            -- must leave the OCR work list instead of being re-read nightly.
            -- Engine-B-primary runs (parser_version v1-glm53-vision, engines
            -- 'visionb') count the same way: a GLM success/no_txs run is done
            -- and must leave the work list; only an 'error' run keeps the doc
            -- in work for the next nightly retry.
            SELECT *, ROW_NUMBER() OVER (
                PARTITION BY doc_id ORDER BY parsed_at DESC
            ) AS rn
            FROM pdf_parse_runs
            WHERE NOT (
                LOWER(COALESCE(parser_version, '')) LIKE '%gemini%'
                OR LOWER(COALESCE(parser_version, '')) LIKE '%ling%'
                OR LOWER(COALESCE(parser_version, '')) LIKE '%gpt%'
                OR LOWER(COALESCE(parser_version, '')) LIKE '%openrouter%'
                OR LOWER(COALESCE(engines_attempted, '')) LIKE 'gemini/%'
                OR LOWER(COALESCE(engines_attempted, '')) LIKE 'openrouter/%'
                OR LOWER(COALESCE(engines_attempted, '')) LIKE 'gpt-%'
            )
        ), current_ocr AS (
            SELECT *, ROW_NUMBER() OVER (
                PARTITION BY doc_id ORDER BY parsed_at DESC
            ) AS rn
            FROM pdf_parse_runs
            WHERE parser_version = ?
        ), tx AS (
            SELECT doc_id,
                   COUNT(*) FILTER (
                       WHERE source = 'gemini_ocr'
                             AND (chamber = 'House' OR chamber IS NULL)
                             AND ingestion_generation = ?
                   ) AS current_ocr_row_count,
                   COUNT(*) FILTER (
                       WHERE source = 'gemini_ocr'
                         AND (chamber = 'House' OR chamber IS NULL)
                         AND (ingestion_generation IS NULL OR ingestion_generation <> ?)
                   ) AS stale_ocr_row_count
            FROM transactions
            GROUP BY doc_id
        )
        SELECT CAST(m.doc_id AS VARCHAR),
               CAST(EXTRACT(YEAR FROM m.filing_date) AS INTEGER),
               o.status,
               o.raw_row_count,
               o.transaction_count,
               COALESCE(tx.current_ocr_row_count, 0),
               COALESCE(tx.stale_ocr_row_count, 0)
        FROM metadata m
        JOIN deterministic d ON d.doc_id = m.doc_id AND d.rn = 1
        LEFT JOIN current_ocr o ON o.doc_id = m.doc_id AND o.rn = 1
        LEFT JOIN tx ON tx.doc_id = m.doc_id
        WHERE m.filing_type = 'P'
          AND d.status IN ('zero_rows', 'error', 'rejected')
          AND (? IS NULL OR EXTRACT(YEAR FROM m.filing_date) = ?)
        ORDER BY EXTRACT(YEAR FROM m.filing_date), m.doc_id
        """,
        [parser_version, parser_version, parser_version, year, year],
    ).fetchall()
    cache_dir = str(base / "gemini_cache")
    unresolved = []
    for (
        doc_id,
        doc_year,
        status,
        recorded_raw_count,
        recorded_count,
        current_ocr_row_count,
        stale_ocr_row_count,
    ) in rows:
        pdf_path = str(base / str(doc_year) / "pdfs" / f"{doc_id}.pdf")
        try:
            cached = inspect_cached_response(
                doc_id, pdf_path, cache_dir, parser_version
            )
        except (OSError, ValueError):
            cached = None
        if cached is None or not schema_ready:
            unresolved.append((doc_id, doc_year, pdf_path))
            continue
        parsed_count = len(cached.parsed.transactions)
        try:
            ingestion_generation = _resolve_ingestion_generation(
                conn, doc_id, doc_year, cached.pdf_sha256
            )
        except RuntimeError:
            unresolved.append((doc_id, doc_year, pdf_path))
            continue
        bound_run = conn.execute(
            """
            SELECT status, raw_row_count, transaction_count
            FROM pdf_parse_runs
            WHERE doc_id = ? AND parser_version = ?
              AND artifact_sha256 = ? AND ingestion_generation = ?
            ORDER BY parsed_at DESC LIMIT 1
            """,
            [
                doc_id,
                parser_version,
                cached.pdf_sha256,
                ingestion_generation,
            ],
        ).fetchone()
        if bound_run is None:
            status = None
            recorded_raw_count = None
            recorded_count = None
        else:
            status, recorded_raw_count, recorded_count = bound_run
        matching_rows = conn.execute(
            """
            SELECT source_row_id, raw_asset_description, raw_transaction_subtype,
                   transaction_date, notification_date, amount_raw
            FROM transactions
            WHERE doc_id = ? AND source = 'gemini_ocr'
              AND (chamber = 'House' OR chamber IS NULL)
              AND ingestion_generation = ? AND artifact_sha256 = ?
            ORDER BY source_row_id
            """,
            [doc_id, ingestion_generation, cached.pdf_sha256],
        ).fetchall()
        current_ocr_row_count = len(matching_rows)
        stale_ocr_row_count = conn.execute(
            """
            SELECT COUNT(*) FROM transactions
            WHERE doc_id = ? AND source = 'gemini_ocr'
              AND (chamber = 'House' OR chamber IS NULL)
              AND NOT (
                  ingestion_generation IS NOT DISTINCT FROM ?
                  AND artifact_sha256 IS NOT DISTINCT FROM ?
              )
            """,
            [doc_id, ingestion_generation, cached.pdf_sha256],
        ).fetchone()[0]
        expected_rows = []
        for source_row_number, transaction in enumerate(
            cached.parsed.transactions, start=1
        ):
            page_number = transaction.get("page_number")
            source_row_id = (
                f"{doc_id}:page:{page_number}:row:{source_row_number}"
                if page_number is not None
                else f"{doc_id}:row:{source_row_number}"
            )
            expected_rows.append(
                (
                    source_row_id,
                    transaction["asset"][:500],
                    transaction["type"],
                    datetime.date.fromisoformat(normalize_date(transaction["date"])),
                    datetime.date.fromisoformat(
                        normalize_date(transaction["notif_date"])
                    ) if transaction["notif_date"] is not None else None,
                    transaction["amount_letter"],
                )
            )
        success_matches = (
            status == "success"
            and recorded_raw_count == cached.parsed.raw_row_count
            and recorded_count == parsed_count
            and current_ocr_row_count == parsed_count
            and stale_ocr_row_count == 0
            and matching_rows == sorted(expected_rows)
        )
        no_txs_matches = (
            status == "no_txs"
            and cached.parsed.no_transactions
            and current_ocr_row_count == 0
            and stale_ocr_row_count == 0
        )
        if not (success_matches or no_txs_matches):
            unresolved.append((doc_id, doc_year, pdf_path))
    conn.close()
    return sorted(set(unresolved) | set(generation_work))


def get_zero_row_pdfs():
    """Backward-compatible all-year work selection from current DB state."""
    return get_ocr_work_items()


def load_progress(path: str | Path = PROGRESS_PATH):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"completed": [], "errors": [], "no_txs": []}


def save_progress(progress, path: str | Path = PROGRESS_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(progress, f, indent=2)
    os.replace(tmp_path, path)


def mark_progress(progress, doc_id, status):
    """Record one confirmed terminal/current result without stale overlaps."""
    key = "completed" if status == "success" else status
    if key not in {"completed", "errors", "no_txs"}:
        raise ValueError(f"unsupported progress status: {status}")
    doc_id = str(doc_id)
    for values in progress.values():
        while doc_id in values:
            values.remove(doc_id)
    progress[key].append(doc_id)


def parse_output(output, *, expected_page_count=None):
    """Parse a schema-validated Gemini response into the legacy tuple API."""
    parsed = parse_gemini_output(output, expected_page_count=expected_page_count)
    return parsed.member, parsed.transactions


def normalize_date(date_str):
    """Convert strict MM/DD[/YY] or MM-DD[-YY] dates to YYYY-MM-DD."""
    if not date_str:
        return None
    m = re.fullmatch(r"(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})", date_str.strip())
    if not m:
        return None
    month, day, year = m.groups()
    month_i, day_i = int(month), int(day)
    if month_i < 1 or month_i > 12 or day_i < 1 or day_i > 31:
        return None
    if len(year) == 2:
        year = "20" + year if int(year) < 50 else "19" + year
    year_i = int(year)
    try:
        datetime.date(year_i, month_i, day_i)
    except ValueError:
        return None
    return f"{year}-{month_i:02d}-{day_i:02d}"


def extract_ticker(asset, *, explicit_only=False):
    """Extract stock ticker from common House asset formats."""
    if not asset:
        return None
    text = asset.upper()
    patterns = [
        r"\bTICKER\s*[:=]\s*([A-Z]{1,5}(?:\.[AB])?)\b",
    ]
    if not explicit_only:
        patterns.extend([
            r"\(([A-Z]{1,5}(?:\.[AB])?)\)",
            r"\$([A-Z]{1,5}(?:\.[AB])?)\b",
        ])
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            ticker = match.group(1)
            if ticker not in {"LLC", "LP", "LLP", "INC", "ETF", "ST", "MF", "OT"}:
                return ticker
    return None


# Hardcoded mapping of common company names/keywords → ticker symbols
# Used when OCR asset descriptions lack explicit ticker symbols.
COMPANY_TICKER_MAP = {
    # Mega-cap tech
    "apple": "AAPL",
    "microsoft": "MSFT",
    "amazon": "AMZN",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "meta": "META",
    "facebook": "META",
    "tesla": "TSLA",
    "nvidia": "NVDA",
    "netflix": "NFLX",
    "adobe": "ADBE",
    "salesforce": "CRM",
    "oracle": "ORCL",
    "intel": "INTC",
    "amd": "AMD",
    "broadcom": "AVGO",
    "cisco": "CSCO",
    "qualcomm": "QCOM",
    "ibm": "IBM",
    "tx": "TXN",
    "intuit": "INTU",
    "paypal": "PYPL",
    "shopify": "SHOP",
    "uber": "UBER",
    "lyft": "LYFT",
    "snap": "SNAP",
    "pinterest": "PINS",
    "robinhood": "HOOD",
    "coinbase": "COIN",
    "block": "SQ",
    "square": "SQ",
    "zoom": "ZM",
    "crowdstrike": "CRWD",
    "palo alto": "PANW",
    "cloudflare": "NET",
    "datadog": "DDOG",
    "mongodb": "MDB",
    "snowflake": "SNOW",
    "databricks": "DBX",
    "twilio": "TWLO",
    "spotify": "SPOT",
    "roku": "ROKU",
    "palantir": "PLTR",
    "unity": "U",
    "roblox": "RBLX",
    "doximity": "DOCS",
    # Finance / Banking
    "jpmorgan": "JPM",
    "jp morgan": "JPM",
    "bank of america": "BAC",
    "goldman sachs": "GS",
    "goldman": "GS",
    "morgan stanley": "MS",
    "wells fargo": "WFC",
    "citigroup": "C",
    "citi": "C",
    "us bancorp": "USB",
    "truist": "TFC",
    "pnc": "PNC",
    "charles schwab": "SCHW",
    "schwab": "SCHW",
    "barclays": "BCS",
    "hsbc": "HSBC",
    "ubs": "UBS",
    "credit suisse": "CS",
    "american express": "AXP",
    "visa": "V",
    "mastercard": "MA",
    "blackrock": "BLK",
    "blackstone": "BX",
    "vanguard": "VT",
    "fidelity": "FNF",
    "state street": "STT",
    "northern trust": "NTRS",
    "t rowe price": "TROW",
    "berkshire": "BRK",
    "berkshire hathaway": "BRK",
    "capital one": "COF",
    "discover": "DFS",
    "synchrony": "SYF",
    "stock exchange": "ICE",
    "cme group": "CME",
    "nasdaq": "NDAQ",
    "intercontinental exchange": "ICE",
    "spglobal": "SPGI",
    "s&p global": "SPGI",
    "moody": "MCO",
    "moody's": "MCO",
    # Healthcare / Pharma
    "johnson & johnson": "JNJ",
    "johnson and johnson": "JNJ",
    "pfizer": "PFE",
    "unitedhealth": "UNH",
    "united health": "UNH",
    "abbvie": "ABBV",
    "merck": "MRK",
    "abbott": "ABT",
    "amgen": "AMGN",
    "gilead": "GILD",
    "bristol-myers": "BMY",
    "bristol myers": "BMY",
    "eli lilly": "LLY",
    "lilly": "LLY",
    "regeneron": "REGN",
    "vertex": "VRTX",
    "moderna": "MRNA",
    "biogen": "BIIB",
    "vertex pharmaceuticals": "VRTX",
    "cigna": "CI",
    "humana": "HUM",
    "anthem": "ELV",
    "elevance": "ELV",
    "centene": "CNC",
    "aetna": "AET",
    "mcdermott": "MCD",
    "medtronic": "MDT",
    "baxter": "BAX",
    "becton dickinson": "BDX",
    "stryker": "SYK",
    "zimmer biomet": "ZBH",
    "intuitive surgical": "ISRG",
    "intuitive": "ISRG",
    "hologic": "HOLX",
    "idexx": "IDXX",
    "charles river": "CRL",
    # Consumer / Retail
    "walmart": "WMT",
    "costco": "COST",
    "target": "TGT",
    "home depot": "HD",
    "lowes": "LOW",
    "ikea": "INGKA",
    "dollar general": "DG",
    "dollar tree": "DLTR",
    "best buy": "BBY",
    "nordstrom": "JWN",
    "macys": "M",
    "macy's": "M",
    "kohl's": "KSS",
    "tjx": "TJX",
    "ross stores": "ROST",
    "lululemon": "LULU",
    "nike": "NKE",
    "adidas": "ADDYY",
    "puma": "PUMAY",
    "under armour": "UAA",
    "gap": "GPS",
    "old navy": "GPS",
    "zara": "ITX",
    "h&m": "HNNMY",
    "starbucks": "SBUX",
    "mcdonald's": "MCD",
    "mcdonalds": "MCD",
    "chipotle": "CMG",
    "yum": "YUM",
    "yum brands": "YUM",
    "coca-cola": "KO",
    "coca cola": "KO",
    "pepsi": "PEP",
    "pepsico": "PEP",
    "procter": "PG",
    "procter & gamble": "PG",
    "unilever": "UL",
    "colgate": "CL",
    "kellogg's": "K",
    "kellogg": "K",
    "general mills": "GIS",
    "campbell": "CPB",
    "conagra": "CAG",
    "kraft heinz": "KHC",
    "mondelez": "MDLZ",
    "mars": "MWWC",
    "nestle": "NSRGY",
    # Energy / Oil & Gas
    "exxon": "XOM",
    "exxon mobil": "XOM",
    "chevron": "CVX",
    "conocophillips": "COP",
    "shell": "SHEL",
    "bp": "BP",
    "occidental": "OXY",
    "occidental petroleum": "OXY",
    "devon": "DVN",
    "devon energy": "DVN",
    "marathon": "MPC",
    "marathon petroleum": "MPC",
    "valero": "VLO",
    "phillips 66": "PSX",
    "sunoco": "SUN",
    "nextera": "NEE",
    "nextera energy": "NEE",
    "duke energy": "DUK",
    "southern company": "SO",
    "dominion": "D",
    "dominion energy": "D",
    "american electric": "AEP",
    "first solar": "FSLR",
    "enphase": "ENPH",
    "solar edge": "SEDG",
    # Industrials / Aerospace
    "boeing": "BA",
    "lockheed": "LMT",
    "lockheed martin": "LMT",
    "raytheon": "RTX",
    "rtx": "RTX",
    "northrop grumman": "NOC",
    "general dynamics": "GD",
    "l3harris": "LHX",
    "transdigm": "TDG",
    "honeywell": "HON",
    "3m": "MMM",
    "caterpillar": "CAT",
    "deere": "DE",
    "john deere": "DE",
    "general electric": "GE",
    "siemens": "SIEGY",
    "schneider electric": "SU.PA",
    "emerson": "EMR",
    "parker hannifin": "PH",
    "rockwell": "ROK",
    "illinois tool": "ITW",
    "union pacific": "UNP",
    "burlington northern": "BRK",
    "csx": "CSX",
    "norfolk southern": "NSC",
    "fedex": "FDX",
    "ups": "UPS",
    "xpo": "XPO",
    "j.b. hunt": "JBHT",
    "waste management": "WM",
    "republic services": "RSG",
    "cintas": "CTAS",
    "aramark": "ARMK",
    # Telecom / Media
    "at&t": "T",
    "at&t inc": "T",
    "verizon": "VZ",
    "t-mobile": "TMUS",
    "comcast": "CMCSA",
    "charter": "CHTR",
    "charter communications": "CHTR",
    "dish": "DISH",
    "dish network": "DISH",
    "disney": "DIS",
    "warner bros": "WBD",
    "warner discovery": "WBD",
    "paramount": "PARA",
    "fox": "FOX",
    "fox corporation": "FOX",
    "nbcuniversal": "CMCSA",
    "viacom": "VIA",
    "live nation": "LYV",
    "spotify technology": "SPOT",
    # Real Estate / REITs
    "prologis": "PLD",
    "american tower": "AMT",
    "equinix": "EQIX",
    "realty income": "O",
    "public storage": "PSA",
    "welltower": "WELL",
    "avalonbay": "AVB",
    "easterly government": "DEA",
    "digital realty": "DLR",
    "crown castle": "CCI",
    "simon property": "SPG",
    "tanger": "SKT",
    # Food / Beverage
    "starbucks corporation": "SBUX",
    "monster beverage": "MNST",
    "constellation": "STZ",
    "brown forman": "BF.B",
    "domino's": "DPZ",
    "dominos": "DPZ",
    "dard": "DRI",
    "darden": "DRI",
    "wingstop": "WING",
    "sweetgreen": "SG",
    "shake shack": "SHAK",
    "caesars": "CZR",
    "caesars entertainment": "CZR",
    "las vegas sands": "LVS",
    "mgm": "MGM",
    "wynn": "WYNN",
    "melco": "MLCO",
    # EV / Auto
    "rivian": "RIVN",
    "lucid": "LCID",
    "lucid motors": "LCID",
    "nio": "NIO",
    "xpeng": "XPEV",
    "li auto": "LI",
    "toyota": "TM",
    "honda": "HMC",
    "hyundai": "HYMTF",
    "ford": "F",
    "general motors": "GM",
    "gm": "GM",
    "stellantis": "STLA",
    "ferrari": "RACE",
    "porsche": "POAHY",
    "lamborghini": "VOW3.DE",
    # Semiconductor
    "tsmc": "TSM",
    "taiwan semiconductor": "TSM",
    "samsung": "SSNLF",
    "asml": "ASML",
    "arm holdings": "ARM",
    "arm": "ARM",
    "micron": "MU",
    "micron technology": "MU",
    "on semiconductor": "ON",
    "onsemi": "ON",
    "marvell": "MRVL",
    "marvell technology": "MRVL",
    "analog devices": "ADI",
    "maxim": "MXIM",
    "microchip": "MCHP",
    "nxpi": "NXPI",
    "nvidia corporation": "NVDA",
    "amd inc": "AMD",
    # Software / SaaS
    "microsoft corporation": "MSFT",
    "salesforce inc": "CRM",
    "servicenow": "NOW",
    "workday": "WDAY",
    "adobe inc": "ADBE",
    "vmware": "VMW",
    "broadcom inc": "AVGO",
    "synopsys": "SNPS",
    "cadence": "CDNS",
    "ansys": "ANSS",
    "splunk": "SPLK",
    "zscaler": "ZS",
    "okta": "OKTA",
    "cloudflare inc": "NET",
    "dynatrace": "DT",
    "new relic": "NEWR",
    "jfrog": "FROG",
    "confluent": "CFLT",
    "elastic": "ESTC",
    "hashicorp": "HCP",
    "gitlab": "GTLB",
    "atlassian": "TEAM",
    "hubspot": "HUBS",
    "zendesk": "ZEN",
    "freshworks": "FRSH",
    "monday.com": "MNDY",
    "c3.ai": "AI",
    "c3 ai": "AI",
    "openai": "PRIV",
    "anthropic": "PRIV",
    "databricks inc": "DBX",
    # Travel / Hospitality
    "marriott": "MAR",
    "marriott international": "MAR",
    "hilton": "HLT",
    "hilton worldwide": "HLT",
    "airbnb": "ABNB",
    "booking": "BKNG",
    "booking holdings": "BKNG",
    "expedia": "EXPE",
    "tripadvisor": "TRIP",
    "american airlines": "AAL",
    "delta": "DAL",
    "delta air lines": "DAL",
    "united airlines": "UAL",
    "southwest": "LUV",
    "southwest airlines": "LUV",
    "jetblue": "JBLU",
    "spirit airlines": "SAVE",
    "carnival": "CCL",
    "royal caribbean": "RCL",
    "norwegian": "NCLH",
    # Misc / Other
    "berkshire hathaway inc": "BRK",
    "berkshire hathaway class a": "BRK.A",
    "berkshire hathaway class b": "BRK.B",
    "leidos": "LDOS",
    "booz allen": "BAH",
    "accenture": "ACN",
    "deloitte": "PRIVATE",
    "pwc": "PRIVATE",
    "ey": "PRIVATE",
    "kpmg": "PRIVATE",
    "mckinsey": "PRIVATE",
    "spacex": "PRIV",
    "stripe": "PRIV",
    "airtable": "PRIV",
    "canva": "PRIV",
    "notion": "PRIV",
    "figma": "PRIVATE",
    "discord": "PRIV",
    "tiktok": "PRIV",
    "bytedance": "PRIV",
    "temu": "PDD",
    "pinduoduo": "PDD",
    "alibaba": "BABA",
    "baba": "BABA",
    "jd.com": "JD",
    "baidu": "BIDU",
    "tencent": "TCEHY",
    "xiaomi": "XIACF",
    "huawei": "PRIVATE",
    "boeing company": "BA",
    "the boeing company": "BA",
    "lockheed martin corp": "LMT",
    "raytheon technologies": "RTX",
    "lennar": "LEN",
    "pultegroup": "PHM",
    "d.r. horton": "DHI",
    "dr horton": "DHI",
    "meritage": "MTH",
    "toll brothers": "TOL",
    "nvr": "NVR",
    "pulte": "PHM",
    "dream finders": "DFH",
    "green brick": "GRBK",
    "schlumberger": "SLB",
    "slb": "SLB",
    "halliburton": "HAL",
    "baker hughes": "BKR",
    "weatherford": "WFRD",
    "crown holdings": "CCK",
    "sealed air": "SEE",
    "ball corporation": "BLL",
    "silgan": "SLGN",
    "verisk": "VRSK",
    "willis towers watson": "WTW",
    "marsh": "MMC",
    "marsh & mclennan": "MMC",
    "aon": "AON",
    "gallagher": "AJG",
    "arthur j": "AJG",
    "progressive": "PGR",
    "allstate": "ALL",
    "chubb": "CB",
    "hartford": "HIG",
    "travelers": "TRV",
    "metlife": "MET",
    "prudential": "PRU",
    "lincoln national": "LNC",
    "unum": "UNM",
    "aflac": "AFL",
    "cigna group": "CI",
    "hca healthcare": "HCA",
    "tenet": "TCP",
    "tenet healthcare": "TCP",
    "universal health": "UHS",
    "community health": "CYH",
    "davita": "DVA",
    "encompass health": "EHC",
    "waste connections": "WCN",
    "clean harcbors": "CLH",
    "stericycle": "SRCL",
    "republic services inc": "RSG",
    "deckers": "DECK",
    "columbia sportswear": "COLM",
    "ralph lauren": "RL",
    "capri": "CPRI",
    "tapestry": "TPR",
    "coach": "TPR",
    "kate spade": "CPRI",
    "michael kors": "CPRI",
    "hugo boss": "BOSS.DE",
    "burberry": "BRBY.L",
    "lvmh": "MC.PA",
    "hermes": "RMS.PA",
    "kering": "KER.PA",
    "christian dior": "CDI.PA",
    "prada": "1913.HK",
    "toyota motor": "TM",
    "honda motor": "HMC",
    "volkswagen": "VOW3.DE",
    "bmw": "BAMXF",
    "mercedes": "MBG.DE",
    "stellantis nv": "STLA",
    "qualcomm inc": "QCOM",
    "broadcom limited": "AVGO",
    "advanced micro devices": "AMD",
    "micron technology inc": "MU",
    "applovin": "APP",
    "applovin corp": "APP",
    "sea limited": "SE",
    "sea": "SE",
    "grab holdings": "GRAB",
    "grab": "GRAB",
    "sofi": "SOFI",
    "sofi technologies": "SOFI",
    "affirm": "AFRM",
    "affirm holdings": "AFRM",
    "upstart": "UPST",
    "upstart holdings": "UPST",
    "lendingclub": "LC",
    "lending club": "LC",
    "chime": "PRIV",
    "nubank": "NU",
    "nubank holdings": "NU",
    "nu holdings": "NU",
    "rocket companies": "RKT",
    "rocket mortgage": "RKT",
    "uwm": "UWMC",
    "united wholesale mortgage": "UWMC",
    "zillow": "Z",
    "zillow group": "Z",
    "redfin": "RDFN",
    "redfin corp": "RDFN",
    "opendoor": "OPEN",
    "opendoor technologies": "OPEN",
    "we work": "WE",
    "wework": "WE",
    "peloton": "PTON",
    "peloton interactive": "PTON",
    "whatnot": "PRIV",
    "poshmark": "POSH",
    "depop": "ETSY",
    "etsy": "ETSY",
    "etsy inc": "ETSY",
    "ebay": "EBAY",
    "ebay inc": "EBAY",
    "craigslist": "PRIVATE",
    "facebook marketplace": "META",
    "amazon marketplace": "AMZN",
    "amazon web services": "AMZN",
    "aws": "AMZN",
    "microsoft azure": "MSFT",
    "azure": "MSFT",
    "google cloud": "GOOGL",
    "gcp": "GOOGL",
    "openai chatgpt": "PRIV",
    "chatgpt": "PRIV",
    "midjourney": "PRIV",
    "stability ai": "PRIV",
    "cathie wood": "ARKK",
    "ark invest": "ARKK",
    "ark innovation": "ARKK",
    "ark genomics": "ARKG",
    "vanguard s&p 500": "VOO",
    "vanguard total": "VT",
    "ishares": "IVV",
    "spdr s&p": "SPY",
    "qqq": "QQQ",
    "invesco qqq": "QQQ",
    "ark": "ARKK",
    "bitcoin": "BTC-USD",
    "ethereum": "ETH-USD",
    "solana": "SOL-USD",
    "crypto": "BTC-USD",
    "treasury": "TLT",
    "treasury bond": "TLT",
    "i bonds": "GOVT",
    "tips": "TIP",
    "real estate investment trust": "VNQ",
    "reit": "VNQ",
    "vanguard real estate": "VNQ",
    "jpmorgan chase": "JPM",
    "jpmorgan chase & co": "JPM",
    "jpmorgan chase and co": "JPM",
    "bank of new york": "BK",
    "bnymellon": "BK",
    "state street corp": "STT",
    "northern trust corp": "NTRS",
    "visa inc": "V",
    "mastercard incorporated": "MA",
    "mastercard inc": "MA",
    "booz allen hamilton": "BAH",
    "booz allen hamilton holding": "BAH",
    "leidos holdings": "LDOS",
    "leidos inc": "LDOS",
    "general dynamics corp": "GD",
    "l3harris technologies": "LHX",
    "transdigm group": "TDG",
    "transdigm holdings": "TDG",
    "northrop grumman corp": "NOC",
}


def resolve_ticker(asset):
    """Resolve ticker from asset description text using keyword matching.

    Tries substring matching against COMPANY_TICKER_MAP keys. Returns
    the longest matching key's ticker (most specific match), or None.
    """
    if not asset:
        return None
    text = asset.lower()
    best_ticker = None
    best_len = 0
    for name, ticker in COMPANY_TICKER_MAP.items():
        if len(name) > best_len and name in text:
            best_ticker = ticker
            best_len = len(name)
    return best_ticker


def get_filing_date(conn, doc_id):
    row = conn.execute(
        "SELECT filing_date FROM metadata WHERE doc_id = ?",
        [str(doc_id)],
    ).fetchone()
    return row[0] if row else None


def get_metadata_member(conn, doc_id):
    row = conn.execute(
        "SELECT first_name, last_name FROM metadata WHERE doc_id = ?",
        [str(doc_id)],
    ).fetchone()
    if not row:
        return None
    return " ".join(part for part in row if part).strip() or None


def _resolve_ingestion_generation(
    conn, doc_id: str, year: int, artifact_sha256: str
) -> str:
    row = conn.execute(
        """
        SELECT artifact.generation_id
        FROM house_pdf_artifacts AS artifact
        JOIN house_archive_generations AS generation
          ON generation.archive_year = artifact.archive_year
         AND generation.generation_id = artifact.generation_id
        WHERE artifact.doc_id = ?
          AND artifact.archive_year = ?
          AND artifact.artifact_sha256 = ?
          AND artifact.generation_id = (
              SELECT generation_id FROM house_archive_generations
              WHERE archive_year = artifact.archive_year
              ORDER BY promoted_at DESC, generation_id DESC LIMIT 1
          )
        ORDER BY artifact.acquired_at DESC, artifact.generation_id DESC
        LIMIT 1
        """,
        [str(doc_id), year, artifact_sha256],
    ).fetchone()
    if row is None:
        raise RuntimeError(
            "OCR artifact is not bound to an acquired House ingestion generation"
        )
    return str(row[0])


def resolve_ingestion_generation(
    db_path: str, doc_id: str, year: int, artifact_sha256: str
) -> str:
    conn = duckdb.connect(db_path, read_only=True)
    try:
        return _resolve_ingestion_generation(conn, doc_id, year, artifact_sha256)
    finally:
        conn.close()


def record_parse_run(
    conn,
    doc_id,
    year,
    status,
    raw_count,
    tx_count,
    error_message="",
    parser_version=GEMINI_PARSER_VERSION,
    artifact_sha256: str | None = None,
    ingestion_generation: str | None = None,
    engines_attempted: str = MODEL,
):
    identity = [
        str(doc_id),
        parser_version,
        artifact_sha256,
        ingestion_generation,
    ]
    _archive_rows(
        conn, "pdf_parse_runs",
        "doc_id = ? AND parser_version = ? "
        "AND artifact_sha256 IS NOT DISTINCT FROM ? "
        "AND ingestion_generation IS NOT DISTINCT FROM ?", identity,
    )
    conn.execute(
        """
        DELETE FROM pdf_parse_runs
        WHERE doc_id = ? AND parser_version = ?
          AND artifact_sha256 IS NOT DISTINCT FROM ?
          AND ingestion_generation IS NOT DISTINCT FROM ?
        """,
        identity,
    )
    conn.execute(
        """
        INSERT INTO pdf_parse_runs (
            doc_id, year, parser_version, status, engines_attempted,
            raw_row_count, transaction_count, error_message,
            artifact_sha256, ingestion_generation, parsed_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            str(doc_id),
            year,
            parser_version,
            status,
            engines_attempted,
            raw_count,
            tx_count,
            error_message[:1000],
            artifact_sha256,
            ingestion_generation,
            datetime.datetime.now(),
        ],
    )
    # New schemas bind acceptance to the exact persisted source rowset rather
    # than relying on transaction-start versus wall-clock timestamp ordering.
    columns = {r[1] for r in conn.execute("PRAGMA table_info('pdf_parse_runs')").fetchall()}
    if "rowset_created_at" in columns:
        conn.execute(
            """UPDATE pdf_parse_runs SET rowset_created_at = (
                SELECT MAX(created_at) FROM transactions WHERE doc_id = ?
                  AND artifact_sha256 IS NOT DISTINCT FROM ?
                  AND ingestion_generation IS NOT DISTINCT FROM ? AND source = 'gemini_ocr')
                WHERE doc_id = ? AND parser_version = ?
                  AND artifact_sha256 IS NOT DISTINCT FROM ?
                  AND ingestion_generation IS NOT DISTINCT FROM ?""",
            [str(doc_id), artifact_sha256, ingestion_generation, *identity],
        )


def _archive_rows(conn, table, predicate, parameters):
    """Archive exactly the rows a transactional mutation is about to retire."""
    history = {"transactions": "transaction_versions", "pdf_parse_runs": "parse_run_versions"}[table]
    conn.execute(
        f"CREATE TABLE IF NOT EXISTS {history} AS "
        f"SELECT *, CURRENT_TIMESTAMP AS retired_at FROM {table} WHERE FALSE"
    )
    conn.execute(
        f"INSERT INTO {history} SELECT *, CURRENT_TIMESTAMP FROM {table} WHERE {predicate}",
        parameters,
    )


def _record_failed_ocr_attempt(
    conn,
    doc_id,
    year,
    raw_count,
    error_message,
    *,
    parser_version: str = GEMINI_PARSER_VERSION,
    artifact_sha256: str | None = None,
    ingestion_generation: str | None = None,
    engine_model: str = MODEL,
) -> None:
    """Record a failed OCR attempt and retire rows for that exact artifact."""
    conn.execute("BEGIN TRANSACTION")
    try:
        if artifact_sha256 and ingestion_generation:
            _archive_rows(conn, "transactions",
                          "source = 'gemini_ocr' AND doc_id = ? "
                          "AND (chamber = 'House' OR chamber IS NULL) "
                          "AND ingestion_generation = ? AND artifact_sha256 = ?",
                          [str(doc_id), ingestion_generation, artifact_sha256])
            conn.execute(
                """
                DELETE FROM transactions
                WHERE source = 'gemini_ocr' AND doc_id = ?
                  AND (chamber = 'House' OR chamber IS NULL)
                  AND ingestion_generation = ? AND artifact_sha256 = ?
                """,
                [str(doc_id), ingestion_generation, artifact_sha256],
            )
        record_parse_run(
            conn,
            doc_id,
            year,
            "error",
            raw_count,
            0,
            error_message,
            parser_version=parser_version,
            artifact_sha256=artifact_sha256,
            ingestion_generation=ingestion_generation,
            engines_attempted=engine_model,
        )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise


@dataclass(frozen=True)
class InsertOutcome:
    """The explicit committed outcome of one insert_transactions call.

    ``committed`` is true only when a transaction actually COMMITed (rows or
    a no_txs parse run). ``complete`` is true only when everything the caller
    handed in was committed with no rejections — a partial insert (row-level
    rejections kept out) or a member/row rejection is never "complete", so
    callers cannot mistake partial output for a finished document.
    """

    status: str  # success | partial | no_txs | rejected | error
    inserted: int = 0
    expected: int = 0
    committed: bool = False
    rejections: dict | None = None
    detail: str | None = None

    @property
    def complete(self) -> bool:
        return (
            self.committed
            and self.status in ("success", "no_txs")
            and self.inserted == self.expected
        )

    def as_dict(self) -> dict:
        return {
            "status": self.status,
            "inserted": self.inserted,
            "expected": self.expected,
            "committed": self.committed,
            "rejections": self.rejections,
            "detail": self.detail,
        }


def insert_outcome_from_report(
    report: dict | None, *, inserted: int, expected: int
) -> InsertOutcome:
    """The InsertOutcome for an insert_transactions call.

    Reads the ``report`` dict the call was given; if the callee ignored it
    (monkeypatched fakes, older callers), fall back to a conservative
    outcome derived from the committed count alone: a positive count is a
    committed but UNVERIFIED full insert only when it matches ``expected``,
    and zero is an error — never silently "complete".
    """
    outcome = (report or {}).get("outcome")
    if isinstance(outcome, InsertOutcome):
        return outcome
    if (report or {}).get("outcome") is not None:
        return InsertOutcome(
            status="success" if inserted == expected and inserted else "error",
            inserted=inserted,
            expected=expected,
            committed=bool(inserted),
        )
    if expected == 0 and inserted == 0:
        # Empty input committed nothing by definition; only the callee knows
        # whether a no_txs run was recorded, so treat it as unverified.
        return InsertOutcome(status="error", inserted=0, expected=0)
    if inserted and inserted == expected:
        return InsertOutcome(
            status="success", inserted=inserted, expected=expected, committed=True
        )
    return InsertOutcome(
        status="error", inserted=inserted, expected=expected, committed=bool(inserted)
    )


def insert_transactions(
    doc_id,
    year,
    member,
    transactions,
    *,
    db_path: str,
    parser_version: str = GEMINI_PARSER_VERSION,
    raw_count: int | None = None,
    artifact_sha256: str | None = None,
    ingestion_generation: str | None = None,
    engine_model: str = MODEL,
    checkpoint: bool = True,
    report: dict | None = None,
):
    """Insert transactions into DB. Returns count inserted.

    ``report`` (optional) receives ``report["outcome"] = InsertOutcome`` —
    the explicit committed outcome (status, committed count, rejections) so
    callers can tell a full committed insert from a partial/rejected one
    without re-deriving it. The int return value stays the committed count
    for backward compatibility.
    """

    def _report(outcome: InsertOutcome) -> int:
        if report is not None:
            report["outcome"] = outcome
        return outcome.inserted

    conn = duckdb.connect(db_path)
    require_ocr_schema(conn)
    if not artifact_sha256 or not ingestion_generation:
        conn.close()
        raise RuntimeError(
            "OCR insertion requires artifact_sha256 and ingestion_generation"
        )
    input_expected = raw_count if raw_count is not None else len(transactions)
    try:
        bound_generation = _resolve_ingestion_generation(conn, doc_id, year, artifact_sha256)
        if bound_generation != ingestion_generation:
            raise RuntimeError("OCR insertion generation does not match the current artifact")
        metadata = conn.execute(
            "SELECT filing_date, first_name, last_name FROM house_generation_metadata "
            "WHERE archive_year = ? AND generation_id = ? AND doc_id = ? AND filing_type = 'P'",
            [year, ingestion_generation, str(doc_id)],
        ).fetchone()
        if metadata is None:
            raise RuntimeError("OCR insertion requires generation-bound filing metadata")
        filing_date = metadata[0]
        expected_member = " ".join(part for part in metadata[1:] if part).strip()
    except Exception:
        conn.close()
        raise
    _, identity_rejections = validate_transactions(doc_id, member, [], filing_date, expected_member)
    if identity_rejections:
        try:
            _record_failed_ocr_attempt(
                conn, doc_id, year, raw_count or len(transactions),
                json.dumps(identity_rejections, sort_keys=True),
                parser_version=parser_version, artifact_sha256=artifact_sha256,
                ingestion_generation=ingestion_generation, engine_model=engine_model,
            )
        finally:
            conn.close()
        # Member-level rejection: a no_txs-looking zero is NOT a no_txs.
        return _report(
            InsertOutcome(
                status="rejected",
                inserted=0,
                expected=input_expected,
                rejections=dict(identity_rejections),
            )
        )
    if not transactions:
        status = "error" if raw_count else "no_txs"
        message = "semantic_zero_after_raw_rows" if raw_count else ""
        if raw_count:
            try:
                _record_failed_ocr_attempt(
                    conn, doc_id, year, raw_count, message,
                    parser_version=parser_version,
                    artifact_sha256=artifact_sha256,
                    ingestion_generation=ingestion_generation,
                    engine_model=engine_model,
                )
            finally:
                conn.close()
            return _report(
                InsertOutcome(
                    status="rejected",
                    inserted=0,
                    expected=input_expected,
                    detail=message,
                )
            )
        conn.execute("BEGIN TRANSACTION")
        try:
            if status == "no_txs":
                _archive_rows(conn, "transactions",
                              "source = 'gemini_ocr' AND doc_id = ? "
                              "AND (chamber = 'House' OR chamber IS NULL)", [str(doc_id)])
                conn.execute(
                    "DELETE FROM transactions WHERE source = 'gemini_ocr' "
                    "AND doc_id = ? AND (chamber = 'House' OR chamber IS NULL)",
                    [str(doc_id)],
                )
            record_parse_run(
                conn,
                doc_id,
                year,
                status,
                raw_count or 0,
                0,
                message,
                parser_version=parser_version,
                artifact_sha256=artifact_sha256,
                ingestion_generation=ingestion_generation,
                engines_attempted=engine_model,
            )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()
        return _report(
            InsertOutcome(
                status="no_txs", inserted=0, expected=0, committed=True
            )
        )

    input_count = raw_count if raw_count is not None else len(transactions)
    transactions = apply_verified_house_corrections(
        doc_id, artifact_sha256, transactions
    )
    validated, rejections = validate_transactions(
        doc_id, member, transactions, filing_date, expected_member
    )
    # Identity failures are whole-doc fatal (wrong human). Row-level
    # rejections (impossible dates on individual rows, usually filer typos)
    # keep the valid rows: one bad row must not hold the filing hostage.
    identity_fatal = {
        key: count
        for key, count in rejections.items()
        if key in ("member_mismatch", "invalid_member")
    }
    row_rejections = {
        key: count for key, count in rejections.items() if key not in identity_fatal
    }
    if identity_fatal or not validated:
        _record_failed_ocr_attempt(
            conn,
            doc_id,
            year,
            input_count,
            json.dumps(rejections, sort_keys=True),
            parser_version=parser_version,
            artifact_sha256=artifact_sha256,
            ingestion_generation=ingestion_generation,
            engine_model=engine_model,
        )
        conn.close()
        return _report(
            InsertOutcome(
                status="rejected",
                inserted=0,
                expected=input_count,
                rejections=dict(rejections),
            )
        )
    transactions = validated
    partial_note = (
        f"; row-level rejections kept out: {json.dumps(row_rejections, sort_keys=True)}"
        if row_rejections
        else ""
    )

    errors = []
    rows: list[dict] = []
    for source_row_number, tx in enumerate(transactions, start=1):
        try:
            tx_date = normalize_date(tx["date"])
            notification_date = normalize_date(tx.get("notif_date"))
            disclosure_date = filing_date or notification_date or tx_date
            disclosed_ticker = extract_ticker(tx["asset"], explicit_only=True)
            ticker_candidate = None
            if disclosed_ticker:
                ticker = disclosed_ticker
                raw_ticker = disclosed_ticker
                ticker_origin = "official"
            elif ticker := extract_ticker(tx["asset"]):
                raw_ticker = None
                ticker_origin = "asset_description"
            else:
                ticker_candidate = resolve_ticker(tx["asset"])
                ticker = None
                raw_ticker = ticker_candidate
                ticker_origin = "unverified" if ticker_candidate else "not_reported"
            if not tx_date:
                errors.append(f"bad transaction date: {tx['date']}")
                continue

            values = {
                "doc_id": str(doc_id),
                "member": tx.get("member") or member,
                "ticker": ticker,
                "transaction_date": tx_date,
                "disclosure_date": disclosure_date,
                "transaction_type": tx["type"],
                "amount_raw": tx["amount_letter"],
                "amount_midpoint": tx["amount_midpoint"],
                "owner_code": None,
                "asset_description": tx["asset"][:500],
                "source": "gemini_ocr",
            }
            page_number = tx.get("page_number")
            row_identity = (
                f"{doc_id}:page:{page_number}:row:{source_row_number}"
                if page_number is not None
                else f"{doc_id}:row:{source_row_number}"
            )
            authoritative_provenance = {
                "chamber": "House",
                "source_record_id": str(doc_id),
                "source_row_id": row_identity,
                "official_filing_date": filing_date,
                "available_date": filing_date,
                "notification_date": notification_date,
                "raw_transaction_subtype": tx["type"],
                "ticker_origin": ticker_origin,
                "raw_ticker": raw_ticker,
                "ticker_candidate": ticker_candidate,
                "raw_asset_class": "Not separately reported",
                "raw_asset_description": tx["asset"][:500],
                "ingestion_generation": ingestion_generation,
                "artifact_sha256": artifact_sha256,
            }
            values.update(authoritative_provenance)
            row = {column: values.get(column) for column in OCR_INSERT_COLUMNS}
            validate_ticker_provenance(row)
            rows.append(row)
        except Exception as exc:
            errors.append(f"{tx.get('asset', '?')}: {exc}")
    if errors:
        message = "; ".join(errors)
        _record_failed_ocr_attempt(
            conn,
            doc_id,
            year,
            input_count,
            message,
            parser_version=parser_version,
            artifact_sha256=artifact_sha256,
            ingestion_generation=ingestion_generation,
            engine_model=engine_model,
        )
        conn.close()
        return _report(
            InsertOutcome(
                status="error",
                inserted=0,
                expected=input_count,
                detail=message,
            )
        )
    if not rows:
        # If transactions were provided but all failed validation, record as
        # "error" so get_zero_row_pdfs() will retry them; only use "no_txs"
        # when the caller passed an empty list (nothing to retry).
        input_count = raw_count if raw_count is not None else len(transactions)
        if input_count:
            _record_failed_ocr_attempt(
                conn,
                doc_id,
                year,
                input_count,
                "; ".join(errors),
                parser_version=parser_version,
                artifact_sha256=artifact_sha256,
                ingestion_generation=ingestion_generation,
                engine_model=engine_model,
            )
        else:
            record_parse_run(
                conn,
                doc_id,
                year,
                "no_txs",
                0,
                0,
                "",
                parser_version=parser_version,
                artifact_sha256=artifact_sha256,
                ingestion_generation=ingestion_generation,
                engines_attempted=engine_model,
            )
        if checkpoint:
            conn.execute("CHECKPOINT")
        conn.close()
        return _report(
            InsertOutcome(
                status="no_txs" if not input_count else "error",
                inserted=0,
                expected=input_count,
                committed=not input_count,
            )
        )

    count = 0
    committed = False
    staging_table = "staging_gemini_ocr_transactions"
    try:
        conn.execute(f"DROP TABLE IF EXISTS {staging_table}")
        conn.execute(
            f"CREATE TEMP TABLE {staging_table} AS "
            "SELECT * FROM transactions WHERE FALSE"
        )
        for row in rows:
            columns = list(row)
            placeholders = ", ".join("?" for _ in columns)
            conn.execute(
                f"INSERT INTO {staging_table} ({', '.join(columns)}) "
                f"VALUES ({placeholders})",
                [row[column] for column in columns],
            )
        columns = OCR_INSERT_COLUMNS
        conn.execute("BEGIN TRANSACTION")
        invalid_house_rows = int(
            conn.execute(
                """
                SELECT COUNT(*)
                FROM transactions
                WHERE source = 'house_pdf'
                  AND doc_id = ?
                  AND ingestion_generation = ?
                  AND artifact_sha256 = ?
                  AND (
                      transaction_date IS NULL
                      OR disclosure_date IS NULL
                      OR transaction_date > disclosure_date
                      OR (
                          notification_date IS NOT NULL
                          AND notification_date < transaction_date
                      )
                      OR chamber IS NULL OR TRIM(chamber) = ''
                      OR source_record_id IS NULL OR TRIM(source_record_id) = ''
                      OR source_row_id IS NULL OR TRIM(source_row_id) = ''
                      OR official_filing_date IS NULL
                  )
                """,
                [str(doc_id), ingestion_generation, artifact_sha256],
            ).fetchone()[0]
        )
        if invalid_house_rows:
            _archive_rows(conn, "transactions",
                          "source = 'house_pdf' AND doc_id = ? "
                          "AND ingestion_generation = ? AND artifact_sha256 = ?",
                          [str(doc_id), ingestion_generation, artifact_sha256])
            # Only non-OCR terminal runs are invalidated below.
            _archive_rows(conn, "pdf_parse_runs",
                          "doc_id = ? AND ingestion_generation = ? AND artifact_sha256 = ? "
                          "AND NOT (LOWER(COALESCE(parser_version, '')) LIKE '%gemini%' "
                          "OR LOWER(COALESCE(parser_version, '')) LIKE '%ling%' "
                          "OR LOWER(COALESCE(parser_version, '')) LIKE '%gpt%' "
                          "OR LOWER(COALESCE(parser_version, '')) LIKE '%openrouter%' "
                          "OR LOWER(COALESCE(engines_attempted, '')) LIKE 'gemini/%' "
                          "OR LOWER(COALESCE(engines_attempted, '')) LIKE 'openrouter/%' "
                          "OR LOWER(COALESCE(engines_attempted, '')) LIKE 'gpt-%' "
                          "OR LOWER(COALESCE(engines_attempted, '')) LIKE 'gemini+%' "
                          "OR LOWER(COALESCE(engines_attempted, '')) LIKE 'visionb%') "
                          "AND status IN ('success', 'no_txs')",
                          [str(doc_id), ingestion_generation, artifact_sha256])
            conn.execute(
                """
                DELETE FROM transactions
                WHERE source = 'house_pdf'
                  AND doc_id = ?
                  AND ingestion_generation = ?
                  AND artifact_sha256 = ?
                """,
                [str(doc_id), ingestion_generation, artifact_sha256],
            )
            conn.execute(
                """
                UPDATE pdf_parse_runs
                SET status = 'invalidated', transaction_count = 0, parsed_at = ?,
                    error_message = 'superseded by validated gemini_ocr fallback after semantic validation failure'
                WHERE doc_id = ?
                  AND ingestion_generation = ?
                  AND artifact_sha256 = ?
                  AND NOT (
                      LOWER(COALESCE(parser_version, '')) LIKE '%gemini%'
                      OR LOWER(COALESCE(parser_version, '')) LIKE '%ling%'
                      OR LOWER(COALESCE(parser_version, '')) LIKE '%gpt%'
                      OR LOWER(COALESCE(parser_version, '')) LIKE '%openrouter%'
                      OR LOWER(COALESCE(engines_attempted, '')) LIKE 'gemini/%'
                      OR LOWER(COALESCE(engines_attempted, '')) LIKE 'openrouter/%'
                      OR LOWER(COALESCE(engines_attempted, '')) LIKE 'gpt-%'
                      OR LOWER(COALESCE(engines_attempted, '')) LIKE 'gemini+%'
                      OR LOWER(COALESCE(engines_attempted, '')) LIKE 'visionb%'
                  )
                  AND status IN ('success', 'no_txs')
                """,
                [datetime.datetime.now(), str(doc_id), ingestion_generation, artifact_sha256],
            )
        identity_columns = [
            "source",
            "chamber",
            "source_record_id",
            "source_row_id",
            "ingestion_generation",
        ]
        mutable_columns = [
            column for column in columns if column not in identity_columns
        ]
        identity_join = " AND ".join(
            f"target.{column} = staged.{column}" for column in identity_columns
        )
        updates = ", ".join(f"{column} = staged.{column}" for column in mutable_columns)
        _archive_rows(conn, "transactions",
                      f"EXISTS (SELECT 1 FROM {staging_table} AS staged WHERE "
                      + identity_join.replace("target.", "transactions.") + ")", [])
        conn.execute(
            f"UPDATE transactions AS target SET {updates}, created_at = ? "
            f"FROM {staging_table} AS staged WHERE {identity_join}", [datetime.datetime.now()]
        )
        conn.execute(
            f"INSERT INTO transactions ({', '.join(columns)}) "
            f"SELECT {', '.join(f'staged.{column}' for column in columns)} "
            f"FROM {staging_table} AS staged WHERE NOT EXISTS ("
            f"SELECT 1 FROM transactions AS target WHERE {identity_join})"
        )
        stale_identity_join = " AND ".join(
            f"staged.{column} = target.{column}" for column in identity_columns
        )
        _archive_rows(conn, "transactions",
                      "source = 'gemini_ocr' AND doc_id = ? "
                      "AND (chamber = 'House' OR chamber IS NULL) "
                      f"AND NOT EXISTS (SELECT 1 FROM {staging_table} AS staged WHERE "
                      + stale_identity_join.replace("target.", "transactions.") + ")", [str(doc_id)])
        conn.execute(
            "DELETE FROM transactions AS target "
            "WHERE target.source = 'gemini_ocr' AND target.doc_id = ? "
            "AND (target.chamber = 'House' OR target.chamber IS NULL) "
            f"AND NOT EXISTS (SELECT 1 FROM {staging_table} AS staged "
            f"WHERE {stale_identity_join})",
            [str(doc_id)],
        )
        count = len(rows)
        record_parse_run(
            conn,
            doc_id,
            year,
            "success",
            input_count,
            count,
            f"partial insert{partial_note}" if partial_note else "",
            parser_version=parser_version,
            artifact_sha256=artifact_sha256,
            ingestion_generation=ingestion_generation,
            engines_attempted=engine_model,
        )
        conn.execute("COMMIT")
        committed = True
        if checkpoint:
            conn.execute("CHECKPOINT")
        return _report(
            InsertOutcome(
                status="partial" if row_rejections else "success",
                inserted=count,
                expected=input_count,
                committed=True,
                rejections=dict(row_rejections) if row_rejections else None,
                detail=partial_note.lstrip("; ") or None,
            )
        )
    except Exception as exc:
        if not committed:
            try:
                conn.execute("ROLLBACK")
            except Exception:
                pass
            _record_failed_ocr_attempt(
                conn,
                doc_id,
                year,
                input_count,
                str(exc),
                parser_version=parser_version,
                artifact_sha256=artifact_sha256,
                ingestion_generation=ingestion_generation,
                engine_model=engine_model,
            )
        raise
    finally:
        conn.execute(f"DROP TABLE IF EXISTS {staging_table}")
        conn.close()


def insert_transactions_outcome(doc_id, year, member, transactions, **kwargs):
    """insert_transactions, returning the explicit committed outcome.

    Same insert path and arguments; the int return value is replaced by the
    ``InsertOutcome`` so callers never have to guess whether a positive
    count was the full row set or a partial insert.
    """
    report: dict = {}
    inserted = insert_transactions(
        doc_id, year, member, transactions, report=report, **kwargs
    )
    return insert_outcome_from_report(
        report, inserted=inserted, expected=kwargs.get("raw_count") if kwargs.get("raw_count") is not None else len(transactions)
    )


def validate_for_insert(conn, doc_id, member, transactions):
    filing_date = get_filing_date(conn, doc_id)
    expected_member = get_metadata_member(conn, doc_id)
    return validate_transactions(
        doc_id, member, transactions, filing_date, expected_member
    )


def _engine_b_configured() -> bool:
    """True when an engine-B endpoint is configured for dual read.

    Kept for callers that only need a cheap env check; main()'s dual mode
    now preflights the configuration and alerts/stops on a defect instead of
    silently degrading to the single-engine drain.
    """
    try:
        return VisionBConfig.from_env() is not None
    except VisionBUnavailable:
        return False


def _ocr_engine_mode(env=None) -> str:
    """Resolve the nightly drain's engine mode: 'dual', 'visionb', 'gemini'.

    Precedence: DUAL_READ=1 forces dual regardless of OCR_ENGINE; otherwise
    OCR_ENGINE selects — 'visionb' (GLM-primary single engine), 'dual'
    (alias of DUAL_READ=1), or unset/'gemini' (the historical Gemini-only
    drain, the default). Anything else is a typo and must fail loudly rather
    than silently drain the backlog with the wrong engine's quota.
    """
    source = os.environ if env is None else env
    if dual_read_enabled(source):
        return "dual"
    engine = str(source.get(OCR_ENGINE_ENV) or "").strip().lower()
    if engine in ("", "gemini"):
        return "gemini"
    if engine in ("visionb", "dual"):
        return engine
    raise ValueError(
        f"unsupported {OCR_ENGINE_ENV}: {engine!r} "
        "(expected 'gemini', 'visionb', or 'dual')"
    )


def call_vision_b(
    pdf_path: str,
    doc_id: str | None = None,
    *,
    config: VisionBConfig | None = None,
    page_images=None,
) -> tuple[str | None, str, ArtifactMetadata | None]:
    """Engine-B-primary read with the call_gemini contract.

    Hashes and page-counts the same immutable snapshot bytes the artifact
    binding is checked against, asks engine B once with the shared extraction
    prompt, and validates the completion with the shared parser. No response
    cache. Programmatic helper only: the OCR_ENGINE=visionb batch drain now
    routes through ``scripts.glm_stage``'s staging functions (canonical
    ingestion of GLM rows happens ONLY via ``glm_stage promote``), so the
    resume point for that drain is the staging record, not a parse run.
    Transport, configuration, and schema failures return
    (None, error, metadata) — the batch-level quota stop is the caller's
    is_quota_error check.
    """
    metadata = None
    try:
        with snapshot_pdf(pdf_path) as snapshot:
            metadata = ArtifactMetadata(
                snapshot.sha256,
                snapshot.page_count,
                snapshot.byte_count,
                str(Path(pdf_path)),
            )
            doc = DualReadDoc(
                str(doc_id or Path(pdf_path).stem), 0, str(snapshot.path)
            )
            try:
                output = read_document_with_engine_b(
                    doc,
                    config=config,
                    page_images=page_images,
                    page_count=snapshot.page_count,
                )
            except VisionBUnavailable as exc:
                return None, str(exc), metadata
            try:
                parsed = parse_gemini_output(
                    output, expected_page_count=snapshot.page_count
                )
            except GeminiOutputError as exc:
                return None, f"invalid_response: {exc}", metadata
            return output, "", replace(metadata, parsed=parsed)
    except Exception as exc:
        return None, str(exc), metadata


def main():
    parser = argparse.ArgumentParser(
        description="OCR zero-row House PTR PDFs with Gemini"
    )
    parser.add_argument(
        "--refresh", action="store_true", help="Ignore cached Gemini responses"
    )
    args = parser.parse_args()

    mode = _ocr_engine_mode()
    vision_b_config = None
    parser_version, engine_model = GEMINI_PARSER_VERSION, MODEL
    if mode == "visionb":
        # GLM-primary mode must fail closed on an unusable engine-B setup:
        # silently draining with engine A would spend the scarce Gemini
        # quota this mode exists to conserve.
        try:
            vision_b_config = VisionBConfig.from_env()
        except VisionBUnavailable as exc:
            print(f"  visionb unusable: {exc}; stopping batch", flush=True)
            return
        if vision_b_config is None:
            print(
                "  visionb requires VISION_B_BASE_URL and VISION_B_MODEL; "
                "stopping batch",
                flush=True,
            )
            return
    if mode == "dual":
        # A dual run without a usable engine B would be unresolved for every
        # doc in the batch: alert (config defect) and stop before spending
        # engine A on readings nothing may persist.
        try:
            dual_config = VisionBConfig.from_env()
        except VisionBUnavailable as exc:
            print(
                f"ALERT: dual read requested but engine B is unusable: {exc}; "
                "stopping batch",
                flush=True,
            )
            return
        if dual_config is None:
            print(
                "ALERT: dual read requested but VISION_B_BASE_URL/"
                "VISION_B_MODEL are not configured; stopping batch",
                flush=True,
            )
            return

    pdfs = get_zero_row_pdfs()
    progress = load_progress()
    remaining = pdfs
    print(f"Current unresolved OCR work: {len(remaining)}")
    if mode == "visionb":
        _run_visionb_staging(remaining, vision_b_config)
        print("\n=== DONE ===")
        print(
            "Total inserted: 0 (visionb mode stages only; canonical "
            "ingestion of GLM rows happens via glm_stage promote)"
        )
        return

    total_inserted = 0
    for i, (doc_id, year, path) in enumerate(remaining):
        idx = i + 1
        print(f"\n[{idx}/{len(remaining)}] {doc_id} ({year})...", flush=True)

        time.sleep(COOLDOWN)

        if mode == "dual":
            # Opt-in dual vision read (DUAL_READ=1 / OCR_ENGINE=dual, engine
            # B configured): engine A (Gemini) is gated against engine B per
            # doc. AGREE and DISAGREE are persisted or quarantined by the
            # orchestrator (per-doc commit inside insert_transactions/
            # record_disagreement); engine-B absence is UNRESOLVED — nothing
            # persists A-only, quota stops the batch, other outages land an
            # error run that retries the doc next night.
            outcome = run_dual_read_for_doc(
                DualReadDoc(
                    str(doc_id),
                    year,
                    path,
                    data_dir=os.path.dirname(DB_PATH) or ".",
                    db_path=DB_PATH,
                ),
                refresh=args.refresh,
            )
            total_inserted += outcome.inserted
            if outcome.handled:
                mark_progress(progress, doc_id, outcome.progress_key)
                save_progress(progress)
                print(
                    f"  DUAL {outcome.decision.status}: inserted={outcome.inserted}",
                    flush=True,
                )
                continue
            if outcome.decision.status == UNRESOLVED:
                # Engine-B quota stop (unhandled == nothing persisted).
                print("  Engine B quota exhausted; stopping batch", flush=True)
                break
            output, error, artifact_metadata = (
                outcome.output,
                outcome.error,
                outcome.metadata,
            )
        else:
            output, error, artifact_metadata = call_gemini(
                path, doc_id=doc_id, refresh=args.refresh
            )
        # A_ERROR passthrough carries error=None only when engine A
        # succeeded; the single-engine path always has a string here.
        if is_quota_error(error or ""):
            print("  Quota exhausted; stopping batch", flush=True)
            break
        ingestion_generation = None
        if artifact_metadata is not None:
            try:
                ingestion_generation = resolve_ingestion_generation(
                    DB_PATH, doc_id, year, artifact_metadata.sha256
                )
            except RuntimeError as exc:
                output, error = None, str(exc)
        if output is None or error:
            conn = duckdb.connect(DB_PATH)
            try:
                _record_failed_ocr_attempt(
                    conn,
                    doc_id,
                    year,
                    0,
                    error,
                    parser_version=parser_version,
                    engine_model=engine_model,
                    artifact_sha256=(
                        artifact_metadata.sha256 if artifact_metadata else None
                    ),
                    ingestion_generation=ingestion_generation,
                )
            finally:
                conn.close()
            mark_progress(progress, doc_id, "errors")
            save_progress(progress)
            print(f"  ERROR: {error}", flush=True)
            continue

        assert artifact_metadata is not None, "successful Gemini response requires artifact metadata"
        parsed = artifact_metadata.parsed
        assert parsed is not None, "successful Gemini response requires validated output"
        member, transactions = parsed.member, parsed.transactions
        report: dict = {}
        insert_outcome = None
        if not transactions:
            insert_outcome = insert_transactions_outcome(
                doc_id,
                year,
                member,
                [],
                db_path=DB_PATH,
                raw_count=0,
                parser_version=parser_version,
                engine_model=engine_model,
                artifact_sha256=artifact_metadata.sha256,
                ingestion_generation=ingestion_generation,
            )
            # A member rejection is not a no_txs: only a committed no_txs
            # run may close the doc as "no transactions found".
            if insert_outcome.complete:
                mark_progress(progress, doc_id, "no_txs")
                save_progress(progress)
                print("  No transactions found", flush=True)
            else:
                mark_progress(progress, doc_id, "errors")
                save_progress(progress)
                print(
                    "  ERROR: no-transaction run not committed "
                    f"({insert_outcome.status})",
                    flush=True,
                )
            continue

        raw_count = len(transactions)
        insert_outcome = insert_transactions_outcome(
            doc_id,
            year,
            member,
            transactions,
            db_path=DB_PATH,
            raw_count=raw_count,
            parser_version=parser_version,
            engine_model=engine_model,
            artifact_sha256=artifact_metadata.sha256,
            ingestion_generation=ingestion_generation,
        )
        total_inserted += insert_outcome.inserted
        # Partial or rejected output is never "complete": mark errors so the
        # operator sees it, mirroring the dual path.
        mark_progress(
            progress,
            doc_id,
            "success" if insert_outcome.complete else "errors",
        )
        save_progress(progress)
        print(f"  Member: {member}", flush=True)
        print(
            f"  Inserted: {insert_outcome.inserted}/{raw_count} transactions "
            f"(total: {total_inserted})",
            flush=True,
        )
        if not insert_outcome.complete:
            print(
                f"  ERROR: insert incomplete ({insert_outcome.status}); "
                "partial output is not marked complete",
                flush=True,
            )
        for tx in transactions[:3]:
            print(
                f"    {tx['asset']} | {tx['type']} | {tx['date']} | {tx['amount_letter']}",
                flush=True,
            )

    print("\n=== DONE ===")
    print(f"Total inserted: {total_inserted}")


def _run_visionb_staging(remaining, vision_b_config) -> None:
    """GLM-primary drain: write STAGING records, never canonical rows.

    Engine A is never contacted and the canonical DB is never opened here:
    every reading lands as a JSON staging record under
    ``data/staging/glm-backlog`` (via ``scripts.glm_stage``'s staging
    functions), and canonical ingestion happens ONLY through
    ``glm_stage promote`` after the pixel audit. Already-staged docs are
    skipped; quota exhaustion stops the batch cleanly; auth/config defects
    alert and stop (every remaining doc would fail the same way).
    """
    from scripts import glm_stage

    data_dir = os.path.dirname(DB_PATH) or "."
    for i, (doc_id, year, path) in enumerate(remaining):
        idx = i + 1
        print(f"\n[{idx}/{len(remaining)}] {doc_id} ({year})...", flush=True)
        if glm_stage.load_staging_record(data_dir, year, doc_id) is not None:
            print(
                "  already staged; skipping (canonical ingestion happens "
                "only via glm_stage promote)",
                flush=True,
            )
            continue
        time.sleep(COOLDOWN)
        record = glm_stage.build_staging_record(doc_id, year, path, vision_b_config)
        error = record.get("error") or ""
        if record["parse_status"] == "error" and (
            record.get("error_kind") == "quota" or is_quota_error(error)
        ):
            print("  Quota exhausted; stopping batch", flush=True)
            return
        if record["parse_status"] == "error" and record.get("error_kind") in (
            "auth",
            "config",
        ):
            print(
                f"ALERT: visionb {record.get('error_kind')} defect: {error}; "
                "stopping batch",
                flush=True,
            )
            return
        glm_stage.write_staging_record(data_dir, record)
        if record["parse_status"] == "error":
            print(
                f"  ERROR: {error} (staged as error record; canonical "
                "ingestion happens only via glm_stage promote)",
                flush=True,
            )
            continue
        print(
            f"  STAGED {record['parse_status']} rows={len(record['rows'])} "
            "(canonical ingestion happens only via glm_stage promote)",
            flush=True,
        )


def run_gemini_ocr_for_year(year: int, data_dir: str = "data", refresh: bool = False):
    """Process unresolved deterministic OCR candidates for one year."""
    db_path = os.path.join(data_dir, "congress.duckdb")
    progress_path = Path(data_dir) / "ocr_progress_gemini_manual.json"
    remaining = get_ocr_work_items(db_path=db_path, data_dir=data_dir, year=year)
    print(f"Current unresolved OCR work for {year}: {len(remaining)}")
    if dual_read_enabled():
        # Same contract as main()'s dual mode: a dual run without a usable
        # engine B is unresolved for every doc — alert the config defect and
        # stop rather than draining A-only.
        try:
            dual_config = VisionBConfig.from_env()
        except VisionBUnavailable as exc:
            print(
                f"ALERT: dual read requested but engine B is unusable: {exc}; "
                "stopping batch",
                flush=True,
            )
            return 0
        if dual_config is None:
            print(
                "ALERT: dual read requested but VISION_B_BASE_URL/"
                "VISION_B_MODEL are not configured; stopping batch",
                flush=True,
            )
            return 0
    progress = load_progress(progress_path)

    total_inserted = 0
    for i, (doc_id, yr, path) in enumerate(remaining):
        print(f"\n[{i + 1}/{len(remaining)}] {doc_id} ({yr})...", flush=True)
        time.sleep(COOLDOWN)
        if dual_read_enabled():
            # Opt-in dual vision read (DUAL_READ=1): engine A (Gemini) is
            # gated against engine B per doc. AGREE/DISAGREE are persisted or
            # quarantined by the orchestrator; engine-B absence is UNRESOLVED
            # — quota stops the batch, other outages land an error run and
            # are retried next time. Nothing persists A-only.
            outcome = run_dual_read_for_doc(
                DualReadDoc(str(doc_id), yr, path, data_dir=data_dir, db_path=db_path),
                refresh=refresh,
            )
            total_inserted += outcome.inserted
            if outcome.handled:
                mark_progress(progress, doc_id, outcome.progress_key)
                save_progress(progress, progress_path)
                print(
                    f"  DUAL {outcome.decision.status}: inserted={outcome.inserted}",
                    flush=True,
                )
                continue
            if outcome.decision.status == UNRESOLVED:
                # Engine-B quota stop (unhandled == nothing persisted).
                print("  Engine B quota exhausted; stopping batch", flush=True)
                break
            output, error, artifact_metadata = (
                outcome.output,
                outcome.error,
                outcome.metadata,
            )
        else:
            output, error, artifact_metadata = call_gemini(
                path,
                doc_id=doc_id,
                refresh=refresh,
                cache_dir=os.path.join(data_dir, "gemini_cache"),
                parser_version=GEMINI_PARSER_VERSION,
            )
        # A_ERROR passthrough carries error=None (engine A succeeded).
        if is_quota_error(error or ""):
            print("  Quota exhausted; stopping batch", flush=True)
            break
        ingestion_generation = None
        if artifact_metadata is not None:
            try:
                ingestion_generation = resolve_ingestion_generation(
                    db_path, doc_id, yr, artifact_metadata.sha256
                )
            except RuntimeError as exc:
                output, error = None, str(exc)
        if output is None or error:
            conn = duckdb.connect(db_path)
            try:
                _record_failed_ocr_attempt(
                    conn,
                    doc_id,
                    yr,
                    0,
                    error,
                    artifact_sha256=(
                        artifact_metadata.sha256 if artifact_metadata else None
                    ),
                    ingestion_generation=ingestion_generation,
                )
            finally:
                conn.close()
            mark_progress(progress, doc_id, "errors")
            save_progress(progress, progress_path)
            print(f"  ERROR: {error}", flush=True)
            continue

        assert artifact_metadata is not None, "successful Gemini response requires artifact metadata"
        parsed = artifact_metadata.parsed
        assert parsed is not None, "successful Gemini response requires validated output"
        member, transactions = parsed.member, parsed.transactions
        if not transactions:
            no_tx_outcome = insert_transactions_outcome(
                doc_id,
                yr,
                member,
                [],
                db_path=db_path,
                raw_count=0,
                parser_version=GEMINI_PARSER_VERSION,
                artifact_sha256=artifact_metadata.sha256,
                ingestion_generation=ingestion_generation,
            )
            if no_tx_outcome.complete:
                mark_progress(progress, doc_id, "no_txs")
                save_progress(progress, progress_path)
                print("  No transactions found", flush=True)
            else:
                mark_progress(progress, doc_id, "errors")
                save_progress(progress, progress_path)
                print(
                    "  ERROR: no-transaction run not committed "
                    f"({no_tx_outcome.status})",
                    flush=True,
                )
            continue

        raw_count = len(transactions)
        insert_outcome = insert_transactions_outcome(
            doc_id,
            yr,
            member,
            transactions,
            db_path=db_path,
            parser_version=GEMINI_PARSER_VERSION,
            raw_count=raw_count,
            artifact_sha256=artifact_metadata.sha256,
            ingestion_generation=ingestion_generation,
        )
        if not insert_outcome.complete:
            # Rejected or partial: never call the output complete.
            mark_progress(progress, doc_id, "errors")
            save_progress(progress, progress_path)
            print(
                f"  ERROR: insert incomplete ({insert_outcome.status}); "
                f"inserted={insert_outcome.inserted}/{raw_count}",
                flush=True,
            )
            continue
        total_inserted += insert_outcome.inserted
        mark_progress(progress, doc_id, "success")
        save_progress(progress, progress_path)
        print(f"  Inserted: {insert_outcome.inserted}/{raw_count}", flush=True)

    return total_inserted


if __name__ == "__main__":
    main()
