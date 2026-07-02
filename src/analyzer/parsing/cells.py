"""Cell-level extraction helpers for PTR table cells.

Each helper extracts a single piece of information from a raw cell string:
ticker, transaction type, date, owner code, amount, instrument type, option
details. Used by row-level processing in `rows.py` after column mapping.
"""

import re

from analyzer.models import TransactionType

_TICKER_BLACKLIST = {
    # Transaction type letters accidentally captured
    'P', 'S', 'E',
    # Common non-ticker words
    'CASH', 'FUND', 'BOND', 'NOTE', 'BILLS', 'TIPS',
    'THE', 'NEW', 'DEL', 'OLD',
    # Single letters with high false-positive rate in garbled PDFs
    'A', 'I', 'O', 'X', 'Y',
    # Confirmed garbage fragments present in the DB (partial company name words
    # that OCR/pdftotext incorrectly captures as ticker symbols)
    'UNIT', 'TECH', 'NORT', 'MARY', 'CITI', 'AMER', 'BERK', 'BANK', 'MICH', 'WISC',
    'KING', 'SOUT', 'EAST', 'WEST', 'PORT', 'LAKE',
}


def clean_text(text: str | None) -> str:
    if text is None:
        return ""
    return re.sub(r'\s+', ' ', str(text)).strip()


def _extract_ticker(asset_cell: str | None) -> str | None:
    if not asset_cell:
        return None
    ticker_match = re.search(r'\(([A-Za-z][A-Za-z0-9.\-]{0,5})\)', asset_cell)
    if ticker_match:
        candidate = ticker_match.group(1).upper()
        if candidate not in _TICKER_BLACKLIST:
            return candidate
    # Also match $TICKER and TICKER: formats
    dollar_match = re.search(r'\$([A-Za-z][A-Za-z0-9.\-]{0,5})', asset_cell)
    if dollar_match:
        candidate = dollar_match.group(1).upper()
        if candidate not in _TICKER_BLACKLIST:
            return candidate
    colon_match = re.search(r'\b([A-Za-z][A-Za-z0-9.\-]{0,5}):', asset_cell)
    if colon_match:
        candidate = colon_match.group(1).upper()
        if candidate not in _TICKER_BLACKLIST:
            return candidate
    return _resolve_company_name_ticker(asset_cell)


# Company-name-to-ticker mapping used as fallback when regex patterns fail.
# Sorted longest-first so more specific names match before shorter ones.
_COMPANY_NAME_TICKER_MAP: dict[str, str] = {
    "berkshire hathaway class b": "BRK.B", "berkshire hathaway class a": "BRK.A",
    "berkshire hathaway inc": "BRK", "berkshire hathaway": "BRK",
    "bank of america": "BAC", "bank of new york": "BK",
    "booz allen hamilton holding": "BAH", "booz allen hamilton": "BAH",
    "coca-cola company": "KO", "coca-cola": "KO", "coca cola": "KO",
    "charter communications": "CHTR", "columbia sportswear": "COLM",
    "costco wholesale": "COST", "costco": "COST",
    "delta air lines": "DAL",
    "dominos": "DPZ",
    "d.r. horton": "DHI", "dr horton": "DHI",
    "general electric": "GE",
    "general mills": "GIS",
    "goldman sachs": "GS", "google cloud": "GOOGL",
    "home depot": "HD", "honda motor": "HMC",
    "johnson & johnson": "JNJ", "johnson and johnson": "JNJ",
    "jpmorgan chase & co": "JPM", "jpmorgan chase and co": "JPM",
    "jpmorgan chase": "JPM", "jp morgan": "JPM",
    "kraft heinz": "KHC", "l3harris technologies": "LHX",
    "lennar": "LEN", "leidos holdings": "LDOS", "leidos": "LDOS",
    "lockheed martin": "LMT", "louis vuitton": "MC.PA",
    "marsh & mclennan": "MMC", "marriott international": "MAR",
    "mastercard incorporated": "MA", "mastercard inc": "MA",
    "mcdonald's": "MCD", "mcdonalds": "MCD",
    "merck & co": "MRK", "microsoft corporation": "MSFT",
    "micron technology": "MU", "morgan stanley": "MS",
    "monster beverage": "MNST", "northrop grumman": "NOC",
    "norfolk southern": "NSC", "northern trust corp": "NTRS",
    "northern trust": "NTRS",
    "occidental petroleum": "OXY", "palo alto networks": "PANW",
    "pepsico": "PEP", "pultegroup": "PHM", "pulte": "PHM",
    "procter & gamble": "PG", "procter": "PG",
    "qualcomm inc": "QCOM", "ralph lauren": "RL",
    "republic services": "RSG", "raytheon technologies": "RTX",
    "ross stores": "ROST", "royal caribbean": "RCL",
    "schlumberger": "SLB", "seal air": "SEE",
    "simon property": "SPG", "southwest airlines": "LUV",
    "state street corp": "STT", "state street": "STT",
    "taiwan semiconductor": "TSM",
    "the boeing company": "BA", "toll brothers": "TOL",
    "toyota motor": "TM", "transdigm group": "TDG",
    "transdigm holdings": "TDG",
    "united health": "UNH",
    "unitedhealth": "UNH", "universal health": "UHS",
    "ups": "UPS", "visa inc": "V",
    "waste connections": "WCN",
    "willis towers watson": "WTW",
    # Mega-cap tech
    "apple": "AAPL", "microsoft": "MSFT", "amazon": "AMZN",
    "alphabet": "GOOGL", "meta": "META", "facebook": "META",
    "tesla": "TSLA", "nvidia": "NVDA", "netflix": "NFLX",
    "adobe": "ADBE", "salesforce": "CRM", "oracle": "ORCL",
    "intel": "INTC", "amd": "AMD", "broadcom": "AVGO",
    "cisco": "CSCO", "qualcomm": "QCOM", "ibm": "IBM",
    "intuit": "INTU", "paypal": "PYPL", "shopify": "SHOP",
    "uber": "UBER", "lyft": "LYFT", "snap": "SNAP",
    "pinterest": "PINS", "robinhood": "HOOD",
    "coinbase": "COIN", "block": "SQ", "square": "SQ",
    "zoom": "ZM", "crowdstrike": "CRWD",
    "cloudflare": "NET", "datadog": "DDOG",
    "mongodb": "MDB", "snowflake": "SNOW",
    "twilio": "TWLO", "spotify": "SPOT", "roku": "ROKU",
    "palantir": "PLTR", "roblox": "RBLX",
    "applovin": "APP", "sofi": "SOFI",
    # Finance / Banking
    "jpmorgan": "JPM", "goldman": "GS",
    "wells fargo": "WFC", "citigroup": "C", "citi": "C",
    "us bancorp": "USB", "truist": "TFC", "charles schwab": "SCHW",
    "schwab": "SCHW", "american express": "AXP",
    "visa": "V", "mastercard": "MA",
    "blackrock": "BLK", "blackstone": "BX",
    "berkshire": "BRK",
    "capital one": "COF", "discover": "DFS",
    "nasdaq": "NDAQ", "nasdaq inc": "NDAQ",
    "intercontinental exchange": "ICE",
    "s&p global": "SPGI", "spglobal": "SPGI",
    "moody's": "MCO", "moody": "MCO",
    # Healthcare / Pharma
    "pfizer": "PFE", "abbvie": "ABBV",
    "merck": "MRK", "abbott": "ABT", "amgen": "AMGN",
    "gilead": "GILD", "bristol-myers": "BMY",
    "bristol myers": "BMY", "eli lilly": "LLY", "lilly": "LLY",
    "regeneron": "REGN", "vertex": "VRTX",
    "moderna": "MRNA", "biogen": "BIIB",
    "cigna": "CI", "humana": "HUM", "anthem": "ELV",
    "elevance": "ELV", "centene": "CNC",
    "medtronic": "MDT", "stryker": "SYK",
    "intuitive surgical": "ISRG", "intuitive": "ISRG",
    "hca healthcare": "HCA", "tenet healthcare": "THC",
    "davita": "DVA", "encompass health": "EHC",
    # Consumer / Retail
    "walmart": "WMT", "target": "TGT",
    "lowes": "LOW", "dollar general": "DG",
    "dollar tree": "DLTR", "best buy": "BBY",
    "lululemon": "LULU", "nike": "NKE",
    "under armour": "UAA", "gap": "GPS",
    "starbucks": "SBUX", "chipotle": "CMG",
    "yum brands": "YUM", "yum": "YUM",
    "pepsi": "PEP",
    "colgate": "CL", "kellogg": "K", "kellogg's": "K",
    "campbell": "CPB", "conagra": "CAG",
    "mondelez": "MDLZ", "nestle": "NSRGY",
    # Energy / Oil & Gas
    "exxon": "XOM", "exxon mobil": "XOM", "chevron": "CVX",
    "conocophillips": "COP", "shell": "SHEL", "bp": "BP",
    "occidental": "OXY", "devon energy": "DVN", "devon": "DVN",
    "marathon petroleum": "MPC", "marathon": "MPC",
    "valero": "VLO", "phillips 66": "PSX",
    "nextera energy": "NEE", "nextera": "NEE",
    "duke energy": "DUK", "southern company": "SO",
    "dominion energy": "D", "dominion": "D",
    "american electric": "AEP",
    "first solar": "FSLR", "enphase": "ENPH",
    # Industrials / Aerospace
    "boeing": "BA", "lockheed": "LMT",
    "raytheon": "RTX", "rtx": "RTX",
    "general dynamics": "GD", "l3harris": "LHX",
    "transdigm": "TDG", "honeywell": "HON",
    "3m": "MMM", "caterpillar": "CAT",
    "deere": "DE", "john deere": "DE",
    "siemens": "SIEGY", "emerson": "EMR",
    "union pacific": "UNP", "csx": "CSX",
    "fedex": "FDX", "xpo": "XPO",
    "waste management": "WM",
    "cintas": "CTAS", "aramark": "ARMK",
    # Telecom / Media
    "at&t": "T", "verizon": "VZ", "t-mobile": "TMUS",
    "comcast": "CMCSA", "charter": "CHTR",
    "disney": "DIS", "warner bros": "WBD",
    "warner discovery": "WBD", "paramount": "PARA",
    "fox": "FOX", "live nation": "LYV",
    # Real Estate / REITs
    "prologis": "PLD", "american tower": "AMT",
    "equinix": "EQIX", "realty income": "O",
    "public storage": "PSA", "welltower": "WELL",
    "digital realty": "DLR", "crown castle": "CCI",
    # Food / Beverage
    "constellation": "STZ", "domino's": "DPZ",
    "darden": "DRI", "wingstop": "WING",
    "shake shack": "SHAK", "caesars": "CZR",
    "las vegas sands": "LVS", "mgm": "MGM",
    "wynn": "WYNN",
    # EV / Auto
    "rivian": "RIVN", "lucid": "LCID",
    "nio": "NIO", "xpeng": "XPEV", "li auto": "LI",
    "toyota": "TM", "honda": "HMC", "hyundai": "HYMTF",
    "ford": "F", "general motors": "GM", "gm": "GM",
    "stellantis": "STLA", "ferrari": "RACE",
    # Semiconductor
    "tsmc": "TSM", "asml": "ASML", "arm": "ARM",
    "micron": "MU", "onsemi": "ON",
    "marvell": "MRVL", "analog devices": "ADI",
    "microchip": "MCHP",
    # Software / SaaS
    "servicenow": "NOW", "workday": "WDAY",
    "synopsys": "SNPS", "cadence": "CDNS",
    "zscaler": "ZS", "okta": "OKTA",
    "dynatrace": "DT", "confluent": "CFLT",
    "elastic": "ESTC", "gitlab": "GTLB",
    "atlassian": "TEAM", "hubspot": "HUBS",
    "c3.ai": "AI", "c3 ai": "AI",
    # Travel / Hospitality
    "marriott": "MAR", "hilton": "HLT",
    "airbnb": "ABNB", "booking": "BKNG",
    "booking holdings": "BKNG",
    "american airlines": "AAL", "delta": "DAL",
    "united airlines": "UAL", "southwest": "LUV",
    "jetblue": "JBLU", "carnival": "CCL",
    "norwegian": "NCLH",
    # Misc
    "accenture": "ACN", "booz allen": "BAH",
    "alibaba": "BABA", "baidu": "BIDU",
    "tencent": "TCEHY",
    "progressive": "PGR", "allstate": "ALL",
    "chubb": "CB", "hartford": "HIG",
    "travelers": "TRV", "metlife": "MET",
    "prudential": "PRU", "aflac": "AFL",
    "affirm": "AFRM", "upstart": "UPST",
    "sofi technologies": "SOFI",
    "rocket companies": "RKT", "rocket mortgage": "RKT",
    "zillow": "Z", "redfin": "RDFN",
    "peloton": "PTON", "etsy": "ETSY",
    "ebay": "EBAY",
    "deckers": "DECK", "tapestry": "TPR",
    "coach": "TPR",
}


def _resolve_company_name_ticker(asset_cell: str) -> str | None:
    """Match asset description against company names. Longest match wins."""
    text = asset_cell.lower()
    best_ticker = None
    best_len = 0
    for name, ticker in _COMPANY_NAME_TICKER_MAP.items():
        # NOTE: We do NOT filter by _TICKER_BLACKLIST here. The blacklist
        # guards regex-based extraction (where single letters are noisy),
        # but a company-name match is a strong signal — blocking valid
        # tickers like "O" (Realty Income) would be a false negative.
        if len(name) > best_len and name in text:
            best_ticker = ticker
            best_len = len(name)
    return best_ticker


def _extract_transaction_type(tx_type_cell: str | None) -> str | None:
    if not tx_type_cell:
        return None
    raw = tx_type_cell.strip()
    s = raw.lower()
    # Handle "(partial)" suffix: "P (partial)", "S (partial)", "Purchase (partial)", etc.
    s_stripped = re.sub(r'\s*\(partial\)\s*$', '', s).strip()
    if s_stripped in ('p', 'purchase', 'buy'):
        return TransactionType.PURCHASE.value
    if s_stripped in ('s', 'sale', 'sold'):
        return TransactionType.SALE.value
    if s_stripped in ('e', 'exchange'):
        return TransactionType.EXCHANGE.value
    if 'purchase' in s or re.search(r'\bbuy\b', s):
        return TransactionType.PURCHASE.value
    if re.search(r'\bsale\b', s) or re.search(r'\bsell\b', s) or re.search(r'\bsold\b', s):
        return TransactionType.SALE.value
    if 'exchange' in s:
        return TransactionType.EXCHANGE.value
    if s_stripped.startswith('p') and len(s_stripped) <= 2:
        return TransactionType.PURCHASE.value
    if s_stripped.startswith('s') and len(s_stripped) <= 2:
        return TransactionType.SALE.value
    return None


def _extract_date(date_cell: str | None) -> str | None:
    if not date_cell:
        return None
    # Support MM/DD/YYYY, YYYY-MM-DD, and MM/DD/YY formats
    date_match = re.search(r'(\d{1,2}/\d{1,2}/\d{4}|\d{4}-\d{2}-\d{2})', date_cell)
    if date_match:
        return date_match.group(1)
    # MM/DD/YY format — normalize 2-digit year
    date_short = re.search(r'(\d{1,2}/\d{1,2}/(\d{2}))', date_cell)
    if date_short:
        full = date_short.group(1)
        yy = int(date_short.group(2))
        year_prefix = "19" if yy >= 50 else "20"
        date_part = full.rsplit('/', 1)[0]
        return f"{date_part}/{year_prefix}{date_short.group(2)}"
    return None


def _extract_owner_code(owner_cell: str | None) -> str | None:
    owner = clean_text(owner_cell).upper()
    if not owner:
        return None
    if owner.startswith("DEPENDENT"):
        return "DC"
    if owner.startswith("SPOUSE"):
        return "SP"
    if owner.startswith("JOINT"):
        return "J"
    if owner.startswith("SELF"):
        return "S"
    if owner in ("DC", "SP", "J", "S"):
        return owner
    return owner[:8]


def _extract_instrument_type(asset_cell: str | None) -> str:
    """Detect whether an asset description is a stock, call option, or put option.

    Handles common PTR formats:
      - "NVIDIA Corp Common Stock Call Option (NVDA)"
      - "NVDA Call $120 Exp 12/20/2024"
      - "Call Option" / "Put Option" as separate field in asset text
      - "Stock Option (NVDA)" — a standalone phrase without explicit call/put
    """
    if not asset_cell:
        return 'stock'
    text = asset_cell.lower()

    # Put detection — use compound patterns first; avoid matching bare "put" (too many false positives)
    if re.search(r'\bput\s+option\b', text):
        return 'put'
    if re.search(r'\bput\s+opt\b', text):
        return 'put'
    if re.search(r'\bstock\s+put\b', text):
        return 'put'
    if re.search(r'\bput\b.*\b(?:strike|exp|expir)\b', text):
        return 'put'

    # Call detection — use compound patterns first; avoid matching bare "call" (too many false positives)
    if re.search(r'\bcall\s+option\b', text):
        return 'call'
    if re.search(r'\bcall\s+opt\b', text):
        return 'call'
    if re.search(r'\bstock\s+call\b', text):
        return 'call'
    if re.search(r'\bcall\b.*\b(?:strike|exp|expir)\b', text):
        return 'call'

    # "Stock Option" standalone — a common PTR label; default to 'call' as most
    # stock options in congressional disclosures are call options
    if re.search(r'\bstock\s+option\b', text):
        return 'call'

    # Generic "option" without call/put qualifier — try to infer from context
    if re.search(r'\boption\b', text):
        if re.search(r'\b(?:strike|exp|expir)\b', text):
            return 'call'
    return 'stock'


def _extract_option_details(asset_cell: str | None) -> dict:
    """Extract strike price and expiry date from an option asset description.

    Returns dict with optional 'strike_price' (float) and 'expiry_date' (str MM/DD/YYYY).
    Handles formats:
      - "Strike $150" / "Strike: 150.00"
      - "$120" preceding "Exp" in "NVDA Call $120 Exp 12/20/2024"
      - "Exp MM/DD/YYYY" / "Expire MM/DD/YYYY" / "Expiring MM/DD/YYYY"
      - "Exp 12/20/2024" (bare exp abbreviation)
      - "Strike Price $150.00"
      - "NVDA Call Option $120 Exp 12/20/2024"
    """
    details: dict = {}
    if not asset_cell:
        return details

    # Strike price: "Strike Price $150", "Strike $150", "Strike: 150.00"
    strike_match = re.search(r'strike\s*(?:price)?[:\s]*\$?(\d+(?:\.\d+)?)', asset_cell, re.IGNORECASE)
    if strike_match:
        details['strike_price'] = float(strike_match.group(1))
    else:
        # Fallback: dollar amount before Exp/expiry, e.g. "$120 Exp 12/20/2024"
        strike_fallback = re.search(r'\$(\d+(?:\.\d+)?)\s+(?:exp|strike)', asset_cell, re.IGNORECASE)
        if strike_fallback:
            details['strike_price'] = float(strike_fallback.group(1))

    # Expiry date: "Exp MM/DD/YYYY" / "Expire: MM/DD/YYYY" / "Expiring MM/DD/YYYY"
    exp_match = re.search(r'(?:exp(?:ir(?:e|ation|ing)?)?[:\s]+(\d{1,2}/\d{1,2}/\d{4}))', asset_cell, re.IGNORECASE)
    if exp_match:
        details['expiry_date'] = exp_match.group(1)
    return details


def _extract_amount_midpoint(amount_cell: str | None) -> tuple[str | None, float | None]:
    amount = clean_text(amount_cell)
    if not amount:
        return None, None
    values = [float(value.replace(",", "")) for value in re.findall(r'\$([0-9][0-9,]*)', amount)]
    if not values:
        return amount, None
    return amount, sum(values[:2]) / min(len(values), 2)
