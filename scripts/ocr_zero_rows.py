#!/usr/bin/env python3
"""Production OCR pipeline for 425 zero-row scanned PTR PDFs.

Uses `llm -a` with Gemini 3.1 Flash Lite to extract transactions.
Gemini auto-rotates PDFs and handles checkbox detection.
"""
import argparse, json, os, re, time, duckdb
from pathlib import Path

from scripts.gemini_ocr_common import MODEL, call_gemini, validate_transactions

DB_PATH = "data/congress.duckdb"
PROGRESS_PATH = "data/ocr_progress_gemini_manual.json"
COOLDOWN = 3  # seconds between requests (Lite model allows rapid fire)
# Amount range midpoint estimates (for amount_midpoint column)
AMOUNT_MIDPOINTS = {
    "A": 8000, "B": 32500, "C": 75000, "D": 175000, "E": 375000,
    "F": 750000, "G": 3000000, "H": 15000000, "I": 37500000, "J": 50000000
}

def get_zero_row_pdfs():
    """Get all zero-row PDFs from DB that haven't been OCR'd yet."""
    conn = duckdb.connect(DB_PATH, read_only=True)
    rows = conn.execute("""
        WITH latest AS (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY doc_id ORDER BY parsed_at DESC) as rn
            FROM pdf_parse_runs
        )
        SELECT l.doc_id, l.year
        FROM latest l
        WHERE l.rn = 1 AND l.status IN ('zero_rows', 'error')
    """).fetchall()
    conn.close()
    return [(d, y, f"data/{y}/pdfs/{d}.pdf") for d, y in rows
            if os.path.exists(f"data/{y}/pdfs/{d}.pdf")]

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

def parse_output(output):
    """Parse Gemini output into structured data."""
    if not output:
        return None, []
    
    member = None
    transactions = []
    
    for line in output.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        
        # Extract MEMBER
        m = re.match(r"MEMBER:\s*(.+)", line, re.IGNORECASE)
        if m:
            member = m.group(1).strip()
            continue
        
        # Skip markdown table separators and headers
        if "|" not in line or "---" in line or "ASSET" in line.upper():
            continue
        
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 4:
            continue
        
        asset = parts[0]
        tx_type = parts[1] if len(parts) > 1 else ""
        tx_date = parts[2] if len(parts) > 2 else ""
        notif_date = parts[3] if len(parts) > 3 else ""
        amount = parts[4] if len(parts) > 4 else ""
        
        # Validate
        if not re.search(r"\d{2}/\d{2}/\d{2}", tx_date):
            continue
        if tx_type not in ("Purchase", "Sale", "Exchange", "Partial Sale", "P", "S", "E"):
            # Try fuzzy match
            tx_lower = tx_type.lower()
            if "purchase" in tx_lower or tx_lower == "p":
                tx_type = "Purchase"
            elif "sale" in tx_lower or tx_lower == "s":
                tx_type = "Sale"
            elif "exchange" in tx_lower or tx_lower == "e":
                tx_type = "Exchange"
            else:
                continue
        
        # Normalize Partial Sale to Sale for consistency with TransactionType enum
        if tx_type == "Partial Sale":
            tx_type = "Sale"
        
        # Map amount letter
        amt_letter = ""
        amount_clean = amount.strip().upper()
        if amount_clean and amount_clean[0] in AMOUNT_MIDPOINTS:
            amt_letter = amount_clean[0]
        amt_mid = AMOUNT_MIDPOINTS.get(amt_letter)
        
        transactions.append({
            "asset": asset,
            "type": tx_type,
            "date": tx_date,
            "notif_date": notif_date,
            "amount_letter": amt_letter,
            "amount_midpoint": amt_mid,
        })
    
    return member, transactions

def normalize_date(date_str):
    """Convert MM/DD/YY or MM/DD/YYYY to YYYY-MM-DD for DuckDB."""
    if not date_str:
        return None
    m = re.match(r'(\d{1,2})/(\d{1,2})/(\d{2,4})', date_str.strip())
    if not m:
        return None
    month, day, year = m.groups()
    if len(year) == 2:
        year = "20" + year if int(year) < 50 else "19" + year
    return f"{year}-{int(month):02d}-{int(day):02d}"

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
    "apple": "AAPL", "microsoft": "MSFT", "amazon": "AMZN", "google": "GOOGL",
    "alphabet": "GOOGL", "meta": "META", "facebook": "META", "tesla": "TSLA",
    "nvidia": "NVDA", "netflix": "NFLX", "adobe": "ADBE", "salesforce": "CRM",
    "oracle": "ORCL", "intel": "INTC", "amd": "AMD", "broadcom": "AVGO",
    "cisco": "CSCO", "qualcomm": "QCOM", "ibm": "IBM", "tx": "TXN",
    "intuit": "INTU", "paypal": "PYPL", "shopify": "SHOP", "uber": "UBER",
    "lyft": "LYFT", "snap": "SNAP", "pinterest": "PINS", "robinhood": "HOOD",
    "coinbase": "COIN", "block": "SQ", "square": "SQ", "zoom": "ZM",
    "crowdstrike": "CRWD", "palo alto": "PANW", "cloudflare": "NET",
    "datadog": "DDOG", "mongodb": "MDB", "snowflake": "SNOW", "databricks": "DBX",
    "twilio": "TWLO", "spotify": "SPOT", "roku": "ROKU",
    "palantir": "PLTR", "unity": "U", "roblox": "RBLX", "doximity": "DOCS",
    # Finance / Banking
    "jpmorgan": "JPM", "jp morgan": "JPM", "bank of america": "BAC",
    "goldman sachs": "GS", "goldman": "GS", "morgan stanley": "MS",
    "wells fargo": "WFC", "citigroup": "C", "citi": "C", "us bancorp": "USB",
    "truist": "TFC", "pnc": "PNC", "charles schwab": "SCHW", "schwab": "SCHW",
    "barclays": "BCS", "hsbc": "HSBC", "ubs": "UBS", "credit suisse": "CS",
    "american express": "AXP", "visa": "V", "mastercard": "MA",
    "blackrock": "BLK", "blackstone": "BX", "vanguard": "VT", "fidelity": "FNF",
    "state street": "STT", "northern trust": "NTRS", "t rowe price": "TROW",
    "berkshire": "BRK", "berkshire hathaway": "BRK",
    "capital one": "COF", "discover": "DFS", "synchrony": "SYF",
    "stock exchange": "ICE", "cme group": "CME", "nasdaq": "NDAQ",
    "intercontinental exchange": "ICE", "spglobal": "SPGI",
    "s&p global": "SPGI", "moody": "MCO", "moody's": "MCO",
    # Healthcare / Pharma
    "johnson & johnson": "JNJ", "johnson and johnson": "JNJ", "pfizer": "PFE",
    "unitedhealth": "UNH", "united health": "UNH", "abbvie": "ABBV",
    "merck": "MRK", "abbott": "ABT", "amgen": "AMGN", "gilead": "GILD",
    "bristol-myers": "BMY", "bristol myers": "BMY", "eli lilly": "LLY",
    "lilly": "LLY", "regeneron": "REGN", "vertex": "VRTX", "moderna": "MRNA",
    "biogen": "BIIB", "vertex pharmaceuticals": "VRTX", "cigna": "CI",
    "humana": "HUM", "anthem": "ELV", "elevance": "ELV", "centene": "CNC",
    "aetna": "AET", "mcdermott": "MCD", "medtronic": "MDT",
    "baxter": "BAX", "becton dickinson": "BDX", "stryker": "SYK",
    "zimmer biomet": "ZBH", "intuitive surgical": "ISRG", "intuitive": "ISRG",
    "hologic": "HOLX", "idexx": "IDXX", "charles river": "CRL",
    # Consumer / Retail
    "walmart": "WMT", "costco": "COST", "target": "TGT", "home depot": "HD",
    "lowes": "LOW", "ikea": "INGKA", "dollar general": "DG",
    "dollar tree": "DLTR", "best buy": "BBY", "nordstrom": "JWN",
    "macys": "M", "macy's": "M", "kohl's": "KSS", "tjx": "TJX",
    "ross stores": "ROST", "lululemon": "LULU", "nike": "NKE",
    "adidas": "ADDYY", "puma": "PUMAY", "under armour": "UAA",
    "gap": "GPS", "old navy": "GPS", "zara": "ITX", "h&m": "HNNMY",
    "starbucks": "SBUX", "mcdonald's": "MCD", "mcdonalds": "MCD",
    "chipotle": "CMG", "yum": "YUM", "yum brands": "YUM",
    "coca-cola": "KO", "coca cola": "KO", "pepsi": "PEP", "pepsico": "PEP",
    "procter": "PG", "procter & gamble": "PG", "unilever": "UL",
    "colgate": "CL", "kellogg's": "K", "kellogg": "K", "general mills": "GIS",
    "campbell": "CPB", "conagra": "CAG", "kraft heinz": "KHC",
    "mondelez": "MDLZ", "mars": "MWWC", "nestle": "NSRGY",
    # Energy / Oil & Gas
    "exxon": "XOM", "exxon mobil": "XOM", "chevron": "CVX",
    "conocophillips": "COP", "shell": "SHEL", "bp": "BP",
    "occidental": "OXY", "occidental petroleum": "OXY",
    "devon": "DVN", "devon energy": "DVN", "marathon": "MPC",
    "marathon petroleum": "MPC", "valero": "VLO", "phillips 66": "PSX",
    "sunoco": "SUN", "nextera": "NEE", "nextera energy": "NEE",
    "duke energy": "DUK", "southern company": "SO", "dominion": "D",
    "dominion energy": "D", "american electric": "AEP",
    "first solar": "FSLR", "enphase": "ENPH", "solar edge": "SEDG",
    # Industrials / Aerospace
    "boeing": "BA", "lockheed": "LMT", "lockheed martin": "LMT",
    "raytheon": "RTX", "rtx": "RTX", "northrop grumman": "NOC",
    "general dynamics": "GD", "l3harris": "LHX", "transdigm": "TDG",
    "honeywell": "HON", "3m": "MMM", "caterpillar": "CAT",
    "deere": "DE", "john deere": "DE", "general electric": "GE",
    "siemens": "SIEGY", "schneider electric": "SU.PA", "emerson": "EMR",
    "parker hannifin": "PH", "rockwell": "ROK", "illinois tool": "ITW",
    "union pacific": "UNP", "burlington northern": "BRK",
    "csx": "CSX", "norfolk southern": "NSC", "fedex": "FDX",
    "ups": "UPS", "xpo": "XPO", "j.b. hunt": "JBHT",
    "waste management": "WM", "republic services": "RSG",
    "cintas": "CTAS", "aramark": "ARMK",
    # Telecom / Media
    "at&t": "T", "at&t inc": "T", "verizon": "VZ", "t-mobile": "TMUS",
    "comcast": "CMCSA", "charter": "CHTR", "charter communications": "CHTR",
    "dish": "DISH", "dish network": "DISH", "disney": "DIS",
    "warner bros": "WBD", "warner discovery": "WBD", "paramount": "PARA",
    "fox": "FOX", "fox corporation": "FOX", "nbcuniversal": "CMCSA",
    "viacom": "VIA", "live nation": "LYV", "spotify technology": "SPOT",
    # Real Estate / REITs
    "prologis": "PLD", "american tower": "AMT", "equinix": "EQIX",
    "realty income": "O", "public storage": "PSA", "welltower": "WELL",
    "avalonbay": "AVB", "easterly government": "DEA",
    "digital realty": "DLR", "crown castle": "CCI",
    "simon property": "SPG", "tanger": "SKT",
    # Food / Beverage
    "starbucks corporation": "SBUX", "monster beverage": "MNST",
    "constellation": "STZ", "brown forman": "BF.B",
    "domino's": "DPZ", "dominos": "DPZ", "dard": "DRI",
    "darden": "DRI", "wingstop": "WING", "sweetgreen": "SG",
    "shake shack": "SHAK", "caesars": "CZR", "caesars entertainment": "CZR",
    "las vegas sands": "LVS", "mgm": "MGM", "wynn": "WYNN",
    "melco": "MLCO",
    # EV / Auto
    "rivian": "RIVN", "lucid": "LCID", "lucid motors": "LCID",
    "nio": "NIO", "xpeng": "XPEV", "li auto": "LI",
    "toyota": "TM", "honda": "HMC", "hyundai": "HYMTF",
    "ford": "F", "general motors": "GM", "gm": "GM",
    "stellantis": "STLA", "ferrari": "RACE", "porsche": "POAHY",
    "lamborghini": "VOW3.DE",
    # Semiconductor
    "tsmc": "TSM", "taiwan semiconductor": "TSM", "samsung": "SSNLF",
    "asml": "ASML", "arm holdings": "ARM", "arm": "ARM",
    "micron": "MU", "micron technology": "MU", "on semiconductor": "ON",
    "onsemi": "ON", "marvell": "MRVL", "marvell technology": "MRVL",
    "analog devices": "ADI", "maxim": "MXIM", "microchip": "MCHP",
    "nxpi": "NXPI", "nvidia corporation": "NVDA", "amd inc": "AMD",
    # Software / SaaS
    "microsoft corporation": "MSFT", "salesforce inc": "CRM",
    "servicenow": "NOW", "workday": "WDAY", "adobe inc": "ADBE",
    "vmware": "VMW", "broadcom inc": "AVGO", "synopsys": "SNPS",
    "cadence": "CDNS", "ansys": "ANSS", "splunk": "SPLK",
    "zscaler": "ZS", "okta": "OKTA", "cloudflare inc": "NET",
    "dynatrace": "DT", "new relic": "NEWR", "jfrog": "FROG",
    "confluent": "CFLT", "elastic": "ESTC", "hashicorp": "HCP",
    "gitlab": "GTLB", "atlassian": "TEAM", "hubspot": "HUBS",
    "zendesk": "ZEN", "freshworks": "FRSH",
    "monday.com": "MNDY", "c3.ai": "AI", "c3 ai": "AI",
    "openai": "PRIV", "anthropic": "PRIV", "databricks inc": "DBX",
    # Travel / Hospitality
    "marriott": "MAR", "marriott international": "MAR",
    "hilton": "HLT", "hilton worldwide": "HLT",
    "airbnb": "ABNB", "booking": "BKNG", "booking holdings": "BKNG",
    "expedia": "EXPE", "tripadvisor": "TRIP", "american airlines": "AAL",
    "delta": "DAL", "delta air lines": "DAL", "united airlines": "UAL",
    "southwest": "LUV", "southwest airlines": "LUV",
    "jetblue": "JBLU", "spirit airlines": "SAVE",
    "carnival": "CCL", "royal caribbean": "RCL", "norwegian": "NCLH",
    # Misc / Other
    "berkshire hathaway inc": "BRK", "berkshire hathaway class a": "BRK.A",
    "berkshire hathaway class b": "BRK.B",
    "leidos": "LDOS", "booz allen": "BAH", "accenture": "ACN",
    "deloitte": "PRIVATE", "pwc": "PRIVATE", "ey": "PRIVATE",
    "kpmg": "PRIVATE", "mckinsey": "PRIVATE",
    "spacex": "PRIV", "stripe": "PRIV", "airtable": "PRIV",
    "canva": "PRIV", "notion": "PRIV", "figma": "PRIVATE",
    "discord": "PRIV", "tiktok": "PRIV", "bytedance": "PRIV",
    "temu": "PDD", "pinduoduo": "PDD", "alibaba": "BABA",
    "baba": "BABA", "jd.com": "JD", "baidu": "BIDU",
    "tencent": "TCEHY", "xiaomi": "XIACF", "huawei": "PRIVATE",
    "boeing company": "BA", "the boeing company": "BA",
    "lockheed martin corp": "LMT", "raytheon technologies": "RTX",
    "lennar": "LEN", "pultegroup": "PHM", "d.r. horton": "DHI",
    "dr horton": "DHI", "meritage": "MTH", "toll brothers": "TOL",
    "nvr": "NVR", "pulte": "PHM", "dream finders": "DFH",
    "green brick": "GRBK",
    "schlumberger": "SLB", "slb": "SLB", "halliburton": "HAL",
    "baker hughes": "BKR", "weatherford": "WFRD",
    "crown holdings": "CCK", "sealed air": "SEE",
    "ball corporation": "BLL", "silgan": "SLGN",
    "verisk": "VRSK", "willis towers watson": "WTW",
    "marsh": "MMC", "marsh & mclennan": "MMC",
    "aon": "AON", "gallagher": "AJG", "arthur j": "AJG",
    "progressive": "PGR", "allstate": "ALL", "chubb": "CB",
    "hartford": "HIG", "travelers": "TRV", "metlife": "MET",
    "prudential": "PRU", "lincoln national": "LNC",
    "unum": "UNM", "aflac": "AFL", "cigna group": "CI",
    "hca healthcare": "HCA", "tenet": "TCP", "tenet healthcare": "TCP",
    "universal health": "UHS", "community health": "CYH",
    "davita": "DVA", "encompass health": "EHC",
    "waste connections": "WCN", "clean harcbors": "CLH",
    "stericycle": "SRCL", "republic services inc": "RSG",
    "deckers": "DECK", "columbia sportswear": "COLM",
    "ralph lauren": "RL", "capri": "CPRI", "tapestry": "TPR",
    "coach": "TPR", "kate spade": "CPRI", "michael kors": "CPRI",
    "hugo boss": "BOSS.DE", "burberry": "BRBY.L",
    "lvmh": "MC.PA", "hermes": "RMS.PA", "kering": "KER.PA",
    "christian dior": "CDI.PA", "prada": "1913.HK",
    "toyota motor": "TM", "honda motor": "HMC",
    "volkswagen": "VOW3.DE", "bmw": "BAMXF", "mercedes": "MBG.DE",
    "stellantis nv": "STLA",
    "qualcomm inc": "QCOM", "broadcom limited": "AVGO",
    "advanced micro devices": "AMD", "micron technology inc": "MU",
    "applovin": "APP", "applovin corp": "APP",
    "sea limited": "SE", "sea": "SE",
    "grab holdings": "GRAB", "grab": "GRAB",
    "sofi": "SOFI", "sofi technologies": "SOFI",
    "affirm": "AFRM", "affirm holdings": "AFRM",
    "upstart": "UPST", "upstart holdings": "UPST",
    "lendingclub": "LC", "lending club": "LC",
    "chime": "PRIV", "nubank": "NU",
    "nubank holdings": "NU", "nu holdings": "NU",
    "rocket companies": "RKT", "rocket mortgage": "RKT",
    "uwm": "UWMC", "united wholesale mortgage": "UWMC",
    "zillow": "Z", "zillow group": "Z",
    "redfin": "RDFN", "redfin corp": "RDFN",
    "opendoor": "OPEN", "opendoor technologies": "OPEN",
    "we work": "WE", "wework": "WE",
    "peloton": "PTON", "peloton interactive": "PTON",
    "whatnot": "PRIV", "poshmark": "POSH",
    "depop": "ETSY", "etsy": "ETSY", "etsy inc": "ETSY",
    "ebay": "EBAY", "ebay inc": "EBAY",
    "craigslist": "PRIVATE", "facebook marketplace": "META",
    "amazon marketplace": "AMZN", "amazon web services": "AMZN",
    "aws": "AMZN", "microsoft azure": "MSFT", "azure": "MSFT",
    "google cloud": "GOOGL", "gcp": "GOOGL",
    "openai chatgpt": "PRIV", "chatgpt": "PRIV",
    "midjourney": "PRIV", "stability ai": "PRIV",
    "cathie wood": "ARKK", "ark invest": "ARKK",
    "ark innovation": "ARKK", "ark genomics": "ARKG",
    "vanguard s&p 500": "VOO", "vanguard total": "VT",
    "ishares": "IVV", "spdr s&p": "SPY", "qqq": "QQQ",
    "invesco qqq": "QQQ", "ark": "ARKK",
    "bitcoin": "BTC-USD", "ethereum": "ETH-USD",
    "solana": "SOL-USD", "crypto": "BTC-USD",
    "treasury": "TLT", "treasury bond": "TLT",
    "i bonds": "GOVT", "tips": "TIP",
    "real estate investment trust": "VNQ",
    "reit": "VNQ", "vanguard real estate": "VNQ",
    "jpmorgan chase": "JPM", "jpmorgan chase & co": "JPM",
    "jpmorgan chase and co": "JPM",
    "bank of new york": "BK", "bnymellon": "BK",
    "state street corp": "STT", "northern trust corp": "NTRS",
    "visa inc": "V", "mastercard incorporated": "MA",
    "mastercard inc": "MA",
    "booz allen hamilton": "BAH", "booz allen hamilton holding": "BAH",
    "leidos holdings": "LDOS", "leidos inc": "LDOS",
    "general dynamics corp": "GD", "l3harris technologies": "LHX",
    "transdigm group": "TDG", "transdigm holdings": "TDG",
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

def record_parse_run(conn, doc_id, year, status, raw_count, tx_count, error_message="", parser_version="v4-gemini-manual"):
    conn.execute("""
        INSERT INTO pdf_parse_runs (
            doc_id, year, parser_version, status, engines_attempted,
            raw_row_count, transaction_count, error_message
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, [
        str(doc_id), year, parser_version, status, MODEL,
        raw_count, tx_count, error_message[:1000],
    ])

def insert_transactions(doc_id, year, member, transactions, *, db_path: str,
                        parser_version: str = "v4-gemini-manual", raw_count: int | None = None):
    """Insert transactions into DB. Returns count inserted."""
    conn = duckdb.connect(db_path)
    filing_date = get_filing_date(conn, doc_id)
    if not transactions:
        record_parse_run(conn, doc_id, year, "no_txs", raw_count or 0, 0, parser_version=parser_version)
        conn.close()
        return 0

    errors = []
    rows = []
    for tx in transactions:
        try:
            # Convert dates to YYYY-MM-DD format
            tx_date = normalize_date(tx["date"])
            notif_date = filing_date or normalize_date(tx["notif_date"]) or tx_date
            ticker = extract_ticker(tx["asset"]) or resolve_ticker(tx["asset"])
            
            if not tx_date:
                errors.append(f"bad date: {tx['date']}")
                continue
            
            # Normalize type
            tx_type = tx["type"]
            if tx_type in ("P",):
                tx_type = "Purchase"
            elif tx_type in ("S",):
                tx_type = "Sale"
            elif tx_type in ("E",):
                tx_type = "Exchange"

            rows.append([
                str(doc_id), tx.get("member") or member or "Unknown", ticker,
                tx_date, notif_date,
                tx_type, tx["amount_letter"] or "Unknown", tx["amount_midpoint"],
                None, tx["asset"][:500] if tx.get("asset") else None, "gemini_ocr"
            ])
        except Exception as e:
            errors.append(f"{tx.get('asset', '?')}: {e}")
    if errors:
        print(f"  INSERT ERRORS ({len(errors)}): {errors[:3]}", flush=True)
    if not rows:
        record_parse_run(conn, doc_id, year, "no_txs", raw_count if raw_count is not None else len(transactions), 0, "; ".join(errors), parser_version=parser_version)
        conn.execute("CHECKPOINT")
        conn.close()
        return 0

    count = 0
    committed = False
    try:
        conn.execute("BEGIN TRANSACTION")
        conn.execute("DELETE FROM transactions WHERE doc_id = ?", [str(doc_id)])
        for row in rows:
            conn.execute("""
                INSERT INTO transactions
                (doc_id, member, ticker, transaction_date, disclosure_date,
                 transaction_type, amount_raw, amount_midpoint, owner_code, created_at,
                 asset_description, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?, ?)
            """, row)
            count += 1
        record_parse_run(conn, doc_id, year, "success", raw_count if raw_count is not None else len(transactions), count, "; ".join(errors), parser_version=parser_version)
        conn.execute("COMMIT")
        committed = True
        conn.execute("CHECKPOINT")
        return count
    except Exception as e:
        # A CHECKPOINT failure after COMMIT must not trigger ROLLBACK (no
        # active transaction — would mask the original error); the data is
        # already committed, so only re-raise.
        if not committed:
            conn.execute("ROLLBACK")
            record_parse_run(conn, doc_id, year, "error", raw_count if raw_count is not None else len(transactions), 0, str(e), parser_version=parser_version)
        raise
    finally:
        conn.close()

def validate_for_insert(conn, doc_id, member, transactions):
    filing_date = get_filing_date(conn, doc_id)
    expected_member = get_metadata_member(conn, doc_id)
    return validate_transactions(doc_id, member, transactions, filing_date, expected_member)

def main():
    parser = argparse.ArgumentParser(description="OCR zero-row House PTR PDFs with Gemini")
    parser.add_argument("--refresh", action="store_true", help="Ignore cached Gemini responses")
    args = parser.parse_args()

    pdfs = get_zero_row_pdfs()
    progress = load_progress()
    completed = set(progress["completed"] + progress["no_txs"])
    
    remaining = [p for p in pdfs if p[0] not in completed]
    print(f"Total zero-row PDFs: {len(pdfs)}")
    print(f"Already processed: {len(completed)}")
    print(f"Remaining: {len(remaining)}")
    
    total_inserted = 0
    for i, (doc_id, year, path) in enumerate(remaining):
        idx = i + 1
        print(f"\n[{idx}/{len(remaining)}] {doc_id} ({year})...", flush=True)
        
        time.sleep(COOLDOWN)
        
        output, error = call_gemini(path, doc_id=doc_id, refresh=args.refresh)
        if output is None or error:
            progress["errors"].append(doc_id)
            save_progress(progress)
            conn = duckdb.connect(DB_PATH)
            record_parse_run(conn, doc_id, year, "error", 0, 0, error)
            conn.close()
            print(f"  ERROR: {error}", flush=True)
            continue
        
        member, transactions = parse_output(output)
        if not transactions:
            insert_transactions(doc_id, year, member, [], db_path=DB_PATH)
            progress["no_txs"].append(doc_id)
            save_progress(progress)
            print(f"  No transactions found", flush=True)
            continue
        
        raw_count = len(transactions)
        conn = duckdb.connect(DB_PATH, read_only=True)
        transactions, rejections = validate_for_insert(conn, doc_id, member, transactions)
        conn.close()
        print(f"  Validation rejections: {rejections}", flush=True)
        if rejections.get("row_count_exceeds_cap"):
            conn = duckdb.connect(DB_PATH)
            record_parse_run(conn, doc_id, year, "rejected", raw_count, 0, "row_count_exceeds_cap")
            conn.close()
            progress["errors"].append(doc_id)
            save_progress(progress)
            print(f"  REJECTED: row_count_exceeds_cap ({raw_count})", flush=True)
            continue
        member = transactions[0].get("member", member) if transactions else member
        inserted = insert_transactions(doc_id, year, member, transactions, db_path=DB_PATH)
        total_inserted += inserted
        progress["completed" if inserted else "no_txs"].append(doc_id)
        save_progress(progress)
        print(f"  Member: {member}", flush=True)
        print(f"  Inserted: {inserted}/{len(transactions)} transactions (total: {total_inserted})", flush=True)
        for tx in transactions[:3]:
            print(f"    {tx['asset']} | {tx['type']} | {tx['date']} | {tx['amount_letter']}", flush=True)
    
    print(f"\n=== DONE ===")
    print(f"Total inserted: {total_inserted}")


def run_gemini_ocr_for_year(year: int, data_dir: str = "data", refresh: bool = False):
    """Process all zero-row PDFs for a specific year."""
    db_path = os.path.join(data_dir, "congress.duckdb")
    progress_path = Path(data_dir) / "ocr_progress_gemini_manual.json"
    conn = duckdb.connect(db_path, read_only=True)
    rows = conn.execute("""
        WITH latest AS (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY doc_id ORDER BY parsed_at DESC) as rn
            FROM pdf_parse_runs
        )
        SELECT l.doc_id, l.year
        FROM latest l
        WHERE l.rn = 1 AND l.status IN ('zero_rows', 'error') AND l.year = ?
    """, [year]).fetchall()
    conn.close()
    
    pdfs = [(d, y, os.path.join(data_dir, str(y), "pdfs", f"{d}.pdf")) for d, y in rows
            if os.path.exists(os.path.join(data_dir, str(y), "pdfs", f"{d}.pdf"))]
    print(f"Zero-row PDFs for {year}: {len(pdfs)}")
    
    progress = load_progress(progress_path)
    completed = set(progress["completed"] + progress["no_txs"])
    remaining = [p for p in pdfs if p[0] not in completed]
    print(f"Remaining: {len(remaining)}")
    
    total_inserted = 0
    for i, (doc_id, yr, path) in enumerate(remaining):
        print(f"\n[{i+1}/{len(remaining)}] {doc_id} ({yr})...", flush=True)
        time.sleep(COOLDOWN)
        output, error = call_gemini(path, doc_id=doc_id, refresh=refresh, cache_dir=os.path.join(data_dir, "gemini_cache"))
        if output is None or error:
            progress["errors"].append(doc_id)
            save_progress(progress, progress_path)
            conn = duckdb.connect(db_path)
            record_parse_run(conn, doc_id, yr, "error", 0, 0, error)
            conn.close()
            print(f"  ERROR: {error}", flush=True)
            continue
        member, transactions = parse_output(output)
        if not transactions:
            insert_transactions(doc_id, yr, member, [], db_path=db_path)
            progress["no_txs"].append(doc_id)
            save_progress(progress, progress_path)
            print(f"  No transactions found", flush=True)
            continue
        raw_count = len(transactions)
        conn = duckdb.connect(db_path, read_only=True)
        transactions, rejections = validate_for_insert(conn, doc_id, member, transactions)
        conn.close()
        print(f"  Validation rejections: {rejections}", flush=True)
        if rejections.get("row_count_exceeds_cap"):
            conn = duckdb.connect(db_path)
            record_parse_run(conn, doc_id, yr, "rejected", raw_count, 0, "row_count_exceeds_cap")
            conn.close()
            progress["errors"].append(doc_id)
            save_progress(progress, progress_path)
            print(f"  REJECTED: row_count_exceeds_cap ({raw_count})", flush=True)
            continue
        member = transactions[0].get("member", member) if transactions else member
        inserted = insert_transactions(doc_id, yr, member, transactions, db_path=db_path)
        total_inserted += inserted
        progress["completed" if inserted else "no_txs"].append(doc_id)
        save_progress(progress, progress_path)
        print(f"  Inserted: {inserted}/{len(transactions)}", flush=True)

    return total_inserted


if __name__ == "__main__":
    main()
