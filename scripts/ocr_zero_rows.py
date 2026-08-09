#!/usr/bin/env python3
"""Production OCR pipeline for 425 zero-row scanned PTR PDFs.

Uses `llm -a` with Gemini 3.1 Flash Lite to extract transactions.
Gemini auto-rotates PDFs and handles checkbox detection.
"""

import argparse
import datetime
import json
import os
import re
import time

import duckdb
from pathlib import Path

from scripts.gemini_ocr_common import (
    GEMINI_PARSER_VERSION,
    MODEL,
    GeminiOutputError as GeminiOutputError,
    call_gemini,
    inspect_cached_response,
    parse_gemini_output,
    validate_transactions,
)

DB_PATH = "data/congress.duckdb"
PROGRESS_PATH = "data/ocr_progress_gemini_manual.json"
COOLDOWN = 3  # seconds between requests (Lite model allows rapid fire)
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
    if not schema_ready:
        legacy_rows = conn.execute(
            """
            WITH deterministic AS (
                SELECT *, ROW_NUMBER() OVER (
                    PARTITION BY doc_id ORDER BY parsed_at DESC
                ) AS rn
                FROM pdf_parse_runs
                WHERE parser_version NOT LIKE '%gemini%'
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
            SELECT *, ROW_NUMBER() OVER (
                PARTITION BY doc_id ORDER BY parsed_at DESC
            ) AS rn
            FROM pdf_parse_runs
            WHERE parser_version NOT LIKE '%gemini%'
        ), current_ocr AS (
            SELECT *, ROW_NUMBER() OVER (
                PARTITION BY doc_id ORDER BY parsed_at DESC
            ) AS rn
            FROM pdf_parse_runs
            WHERE parser_version = ?
        ), tx AS (
            SELECT doc_id,
                   COUNT(*) FILTER (
                       WHERE source = 'gemini_ocr' AND ingestion_generation = ?
                   ) AS current_ocr_row_count,
                   COUNT(*) FILTER (
                       WHERE source = 'gemini_ocr'
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
        matching_rows = conn.execute(
            """
            SELECT source_row_id, raw_asset_description, raw_transaction_subtype,
                   transaction_date, notification_date, amount_raw
            FROM transactions
            WHERE doc_id = ? AND source = 'gemini_ocr'
              AND ingestion_generation = ? AND artifact_sha256 = ?
            ORDER BY source_row_id
            """,
            [doc_id, parser_version, cached.pdf_sha256],
        ).fetchall()
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
                    ),
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
    return unresolved


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


def parse_output(output):
    """Parse a schema-validated Gemini response into the legacy tuple API."""
    parsed = parse_gemini_output(output)
    return parsed.member, parsed.transactions


def normalize_date(date_str):
    """Convert MM/DD/YY or MM/DD/YYYY to YYYY-MM-DD for DuckDB."""
    if not date_str:
        return None
    m = re.match(r"(\d{1,2})/(\d{1,2})/(\d{2,4})", date_str.strip())
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


def extract_ticker(asset):
    """Extract stock ticker from common House asset formats."""
    if not asset:
        return None
    text = asset.upper()
    patterns = [
        r"\(([A-Z]{1,5}(?:\.[AB])?)\)",
        r"\bTICKER\s*[:=]\s*([A-Z]{1,5}(?:\.[AB])?)\b",
        r"\$([A-Z]{1,5}(?:\.[AB])?)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
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


def record_parse_run(
    conn,
    doc_id,
    year,
    status,
    raw_count,
    tx_count,
    error_message="",
    parser_version=GEMINI_PARSER_VERSION,
):
    conn.execute(
        """
        INSERT INTO pdf_parse_runs (
            doc_id, year, parser_version, status, engines_attempted,
            raw_row_count, transaction_count, error_message
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """,
        [
            str(doc_id),
            year,
            parser_version,
            status,
            MODEL,
            raw_count,
            tx_count,
            error_message[:1000],
        ],
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
):
    """Insert transactions into DB. Returns count inserted."""
    conn = duckdb.connect(db_path)
    require_ocr_schema(conn)
    if not artifact_sha256 and transactions:
        conn.close()
        raise RuntimeError("OCR insertion requires artifact_sha256")
    filing_date = get_filing_date(conn, doc_id)
    expected_member = get_metadata_member(conn, doc_id)
    if not transactions:
        status = "error" if raw_count else "no_txs"
        message = "semantic_zero_after_raw_rows" if raw_count else ""
        conn.execute("BEGIN TRANSACTION")
        try:
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
            )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()
        return 0

    input_count = raw_count if raw_count is not None else len(transactions)
    validated, rejections = validate_transactions(
        doc_id, member, transactions, filing_date, expected_member
    )
    fatal_rejections = {
        key: value for key, value in rejections.items() if key != "member_mismatch"
    }
    if fatal_rejections:
        record_parse_run(
            conn,
            doc_id,
            year,
            "error",
            input_count,
            0,
            json.dumps(fatal_rejections, sort_keys=True),
            parser_version=parser_version,
        )
        conn.close()
        return 0
    transactions = validated

    errors = []
    rows: list[dict] = []
    for source_row_number, tx in enumerate(transactions, start=1):
        try:
            tx_date = normalize_date(tx["date"])
            notification_date = normalize_date(tx["notif_date"])
            disclosure_date = filing_date or notification_date or tx_date
            disclosed_ticker = extract_ticker(tx["asset"])
            ticker_candidate = None
            if disclosed_ticker:
                ticker = disclosed_ticker
                raw_ticker = disclosed_ticker
                ticker_origin = "official"
            else:
                ticker_candidate = resolve_ticker(tx["asset"])
                ticker = None
                raw_ticker = ticker_candidate
                ticker_origin = "unverified" if ticker_candidate else "not_reported"
            if not tx_date or not notification_date:
                errors.append(f"bad date: {tx['date']} / {tx['notif_date']}")
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
                "ingestion_generation": parser_version,
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
        record_parse_run(
            conn,
            doc_id,
            year,
            "error",
            input_count,
            0,
            message,
            parser_version=parser_version,
        )
        conn.close()
        return 0
    if not rows:
        # If transactions were provided but all failed validation, record as
        # "error" so get_zero_row_pdfs() will retry them; only use "no_txs"
        # when the caller passed an empty list (nothing to retry).
        input_count = raw_count if raw_count is not None else len(transactions)
        status = "no_txs" if input_count == 0 else "error"
        record_parse_run(
            conn,
            doc_id,
            year,
            status,
            input_count,
            0,
            "; ".join(errors),
            parser_version=parser_version,
        )
        conn.execute("CHECKPOINT")
        conn.close()
        return 0

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
        conn.execute(
            f"UPDATE transactions AS target SET {updates} "
            f"FROM {staging_table} AS staged WHERE {identity_join}"
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
            "",
            parser_version=parser_version,
        )
        conn.execute("COMMIT")
        committed = True
        conn.execute("CHECKPOINT")
        return count
    except Exception as exc:
        if not committed:
            try:
                conn.execute("ROLLBACK")
            except Exception:
                pass
            record_parse_run(
                conn,
                doc_id,
                year,
                "error",
                input_count,
                0,
                str(exc),
                parser_version=parser_version,
            )
        raise
    finally:
        conn.execute(f"DROP TABLE IF EXISTS {staging_table}")
        conn.close()


def validate_for_insert(conn, doc_id, member, transactions):
    filing_date = get_filing_date(conn, doc_id)
    expected_member = get_metadata_member(conn, doc_id)
    return validate_transactions(
        doc_id, member, transactions, filing_date, expected_member
    )


def main():
    parser = argparse.ArgumentParser(
        description="OCR zero-row House PTR PDFs with Gemini"
    )
    parser.add_argument(
        "--refresh", action="store_true", help="Ignore cached Gemini responses"
    )
    args = parser.parse_args()

    pdfs = get_zero_row_pdfs()
    progress = load_progress()
    remaining = pdfs
    print(f"Current unresolved OCR work: {len(remaining)}")

    total_inserted = 0
    for i, (doc_id, year, path) in enumerate(remaining):
        idx = i + 1
        print(f"\n[{idx}/{len(remaining)}] {doc_id} ({year})...", flush=True)

        time.sleep(COOLDOWN)

        output, error, artifact_metadata = call_gemini(
            path, doc_id=doc_id, refresh=args.refresh
        )
        if output is None or error:
            conn = duckdb.connect(DB_PATH)
            try:
                record_parse_run(conn, doc_id, year, "error", 0, 0, error)
            finally:
                conn.close()
            mark_progress(progress, doc_id, "errors")
            save_progress(progress)
            print(f"  ERROR: {error}", flush=True)
            continue

        member, transactions = parse_output(output)
        if not transactions:
            insert_transactions(doc_id, year, member, [], db_path=DB_PATH, raw_count=0)
            mark_progress(progress, doc_id, "no_txs")
            save_progress(progress)
            print("  No transactions found", flush=True)
            continue

        raw_count = len(transactions)
        conn = duckdb.connect(DB_PATH, read_only=True)
        transactions, rejections = validate_for_insert(
            conn, doc_id, member, transactions
        )
        conn.close()
        print(f"  Validation rejections: {rejections}", flush=True)
        fatal_rejections = {
            key: value for key, value in rejections.items() if key != "member_mismatch"
        }
        if fatal_rejections:
            status = (
                "rejected" if "row_count_exceeds_cap" in fatal_rejections else "error"
            )
            message = json.dumps(fatal_rejections, sort_keys=True)
            conn = duckdb.connect(DB_PATH)
            try:
                record_parse_run(conn, doc_id, year, status, raw_count, 0, message)
            finally:
                conn.close()
            mark_progress(progress, doc_id, "errors")
            save_progress(progress)
            print(f"  {status.upper()}: {message}", flush=True)
            continue
        member = transactions[0].get("member", member) if transactions else member
        inserted = insert_transactions(
            doc_id,
            year,
            member,
            transactions,
            db_path=DB_PATH,
            raw_count=raw_count,
            artifact_sha256=artifact_metadata.sha256,
        )
        total_inserted += inserted
        mark_progress(progress, doc_id, "success" if inserted else "errors")
        save_progress(progress)
        print(f"  Member: {member}", flush=True)
        print(
            f"  Inserted: {inserted}/{len(transactions)} transactions (total: {total_inserted})",
            flush=True,
        )
        for tx in transactions[:3]:
            print(
                f"    {tx['asset']} | {tx['type']} | {tx['date']} | {tx['amount_letter']}",
                flush=True,
            )

    print("\n=== DONE ===")
    print(f"Total inserted: {total_inserted}")


def run_gemini_ocr_for_year(year: int, data_dir: str = "data", refresh: bool = False):
    """Process unresolved deterministic OCR candidates for one year."""
    db_path = os.path.join(data_dir, "congress.duckdb")
    progress_path = Path(data_dir) / "ocr_progress_gemini_manual.json"
    remaining = get_ocr_work_items(db_path=db_path, data_dir=data_dir, year=year)
    print(f"Current unresolved OCR work for {year}: {len(remaining)}")
    progress = load_progress(progress_path)

    total_inserted = 0
    for i, (doc_id, yr, path) in enumerate(remaining):
        print(f"\n[{i + 1}/{len(remaining)}] {doc_id} ({yr})...", flush=True)
        time.sleep(COOLDOWN)
        output, error, artifact_metadata = call_gemini(
            path,
            doc_id=doc_id,
            refresh=refresh,
            cache_dir=os.path.join(data_dir, "gemini_cache"),
            parser_version=GEMINI_PARSER_VERSION,
        )
        if output is None or error:
            conn = duckdb.connect(db_path)
            try:
                record_parse_run(conn, doc_id, yr, "error", 0, 0, error)
            finally:
                conn.close()
            mark_progress(progress, doc_id, "errors")
            save_progress(progress, progress_path)
            print(f"  ERROR: {error}", flush=True)
            continue

        member, transactions = parse_output(output)
        if not transactions:
            insert_transactions(
                doc_id,
                yr,
                member,
                [],
                db_path=db_path,
                raw_count=0,
                parser_version=GEMINI_PARSER_VERSION,
            )
            mark_progress(progress, doc_id, "no_txs")
            save_progress(progress, progress_path)
            print("  No transactions found", flush=True)
            continue

        raw_count = len(transactions)
        conn = duckdb.connect(db_path, read_only=True)
        try:
            transactions, rejections = validate_for_insert(
                conn, doc_id, member, transactions
            )
        finally:
            conn.close()
        print(f"  Validation rejections: {rejections}", flush=True)
        fatal_rejections = {
            key: value for key, value in rejections.items() if key != "member_mismatch"
        }
        if fatal_rejections:
            status = (
                "rejected" if "row_count_exceeds_cap" in fatal_rejections else "error"
            )
            message = json.dumps(fatal_rejections, sort_keys=True)
            conn = duckdb.connect(db_path)
            try:
                record_parse_run(
                    conn,
                    doc_id,
                    yr,
                    status,
                    raw_count,
                    0,
                    message,
                    parser_version=GEMINI_PARSER_VERSION,
                )
            finally:
                conn.close()
            mark_progress(progress, doc_id, "errors")
            save_progress(progress, progress_path)
            print(f"  {status.upper()}: {message}", flush=True)
            continue

        member = transactions[0]["member"]
        inserted = insert_transactions(
            doc_id,
            yr,
            member,
            transactions,
            db_path=db_path,
            parser_version=GEMINI_PARSER_VERSION,
            raw_count=raw_count,
            artifact_sha256=artifact_metadata.sha256,
        )
        if inserted <= 0:
            mark_progress(progress, doc_id, "errors")
            save_progress(progress, progress_path)
            print("  ERROR: validated rows were not inserted", flush=True)
            continue
        total_inserted += inserted
        mark_progress(progress, doc_id, "success")
        save_progress(progress, progress_path)
        print(f"  Inserted: {inserted}/{len(transactions)}", flush=True)

    return total_inserted


if __name__ == "__main__":
    main()
