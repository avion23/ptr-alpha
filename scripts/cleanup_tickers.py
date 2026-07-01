"""Clean company-name-as-ticker entries. Idempotent.

Distinguishes "matched to None" (explicit NULL) vs "no match" (leave alone).
"""
from __future__ import annotations
import duckdb
from analyzer.parsing.cells import _TICKER_BLACKLIST

# Mappings: company name prefix -> proper ticker
NAME_TO_TICKER = {
    'MICROSOFT': 'MSFT', 'ALPHABET': 'GOOGL', 'ADOBE': 'ADBE', 'AMAZON.COM': 'AMZN',
    'AMAZON': 'AMZN', 'UNITEDHEALTH': 'UNH', 'META': 'META', 'NETFLIX': 'NFLX',
    'GENERAL MOTORS': 'GM', 'HUMANA': 'HUM', 'APPLE': 'AAPL', 'ACCENTURE': 'ACN',
    'MICRON': 'MU', 'INTUIT': 'INTU', 'WALT DISNEY': 'DIS', 'DISNEY': 'DIS',
    'AON': 'AON', 'EPAM': 'EPAM', 'WELLS FARGO': 'WFC', 'NETAPP': 'NTAP',
    'NET APP': 'NTAP', 'MERCK': 'MRK', 'COMCAST': 'CMCSA', 'SERVICENOW': 'NOW',
    'BERKSHIRE HATHAWAY': 'BRK-B', 'BERKSHIRE': 'BRK-B', 'CVS': 'CVS',
    'FORTINET': 'FTNT', 'CHARLES SCHWAB': 'SCHW', 'SCHWAB': 'SCHW',
    'CATERPILLAR': 'CAT', 'AUTODESK': 'ADSK', 'VISA': 'V', 'SYSCO': 'SYY',
    'FISERV': 'FI', 'GLOBAL PAYMENTS': 'GPN', 'PEPSICO': 'PEP',
    'ELANCO': 'ELAN', 'SALESFORCE': 'CRM', 'INTEL': 'INTC', 'AXALTA': 'AXTA',
    'DANAHER': 'DHR', 'NIKE': 'NKE', 'SHERWIN-WILLIAMS': 'SHW', 'SHERWIN': 'SHW',
    'CISCO': 'CSCO', 'JOHNSON & JOHNSON': 'JNJ', 'JOHNSON': 'JNJ',
    'PROCTER': 'PG', 'CHEVRON': 'CVX', 'EXXON': 'XOM', 'WALMART': 'WMT',
    'COCA-COLA': 'KO', 'COCA': 'KO', 'BANK OF AMERICA': 'BAC',
    'GOLDMAN': 'GS', 'MORGAN STANLEY': 'MS', 'JPMORGAN': 'JPM',
    'PAYPAL': 'PYPL', 'ORACLE': 'ORCL', 'IBM': 'IBM',
    'PFIZER': 'PFE', 'MODERNA': 'MRNA', 'AMGEN': 'AMGN', 'ELI LILLY': 'LLY',
    'ABBVIE': 'ABBV', 'BRISTOL-MYERS': 'BMY', 'BRISTOL': 'BMY',
    'THERMO FISHER': 'TMO', 'GILEAD': 'GILD', 'VERTEX': 'VRTX',
    'LOCKHEED': 'LMT', 'RAYTHEON': 'RTX', 'BOEING': 'BA', 'NORTHROP': 'NOC',
    'HONEYWELL': 'HON', 'UNION PACIFIC': 'UNP', 'COSTCO': 'COST',
    'HOME DEPOT': 'HD', 'MCDONALD': 'MCD', 'STARBUCKS': 'SBUX',
    'NVIDIA': 'NVDA', 'ADVANCED MICRO': 'AMD', 'TEXAS INSTRUMENTS': 'TXN',
    'QUALCOMM': 'QCOM', 'BROADCOM': 'AVGO', 'LAM RESEARCH': 'LRCX',
    'APPLIED MATERIALS': 'AMAT', 'MARVELL': 'MRVL', 'PHILIP MORRIS': 'PM',
    'ALTRIA': 'MO', 'WASTE MANAGEMENT': 'WM', 'FEDEX': 'FDX',
    'FEDERAL EXPRESS': 'FDX', 'UNITED PARCEL': 'UPS', 'CHIPOTLE': 'CMG',
    'BLACKROCK': 'BLK', 'AMERICAN EXPRESS': 'AXP',
    'TRAVELERS': 'TRV', 'TARGET CORPORATION': 'TGT', 'TARGET': 'TGT',
    'NASDAQ': 'NDAQ', 'PENN ENTERTAINMENT': 'PENN', 'PINTEREST': 'PINS',
    'GENERAL MILLS': 'GIS', 'PARKER-HANNIFIN': 'PH', 'PARKER': 'PH',
    'ALLSTATE': 'ALL', 'MOHAWK': 'MWK', 'DOLLAR TREE': 'DLTR',
    'BAXTER': 'BAX', 'TESLA': 'TSLA', 'KKR': 'KKR', 'NCR VOYIX': 'VYX',
    'NCR': 'VYX', 'TENABLE': 'TENB', 'EXTRA SPACE': 'EXR', 'METLIFE': 'MET',
    'VERIZON': 'VZ', 'LULULEMON': 'LULU', 'ABBOTT': 'ABT',
    'URBAN OUTFITTERS': 'URBN', 'LINDE': 'LIN', 'CENTENE': 'CNC',
    'FACTSET': 'FDS', 'SS&C': 'SSNC',
}

# Cash/fund pseudo-tickers that should be NULL-ed (not real stocks)
CASH_PATTERNS = [
    'JPM 100% US TREAS', 'U.S. TREASURY', 'US TREASURY', 'TREASURY BILL',
    'JPMORGAN CHASE BANK', 'GOVERNMENT MONEY', 'MONEY MARKET', 'FEDERAL MONEY',
    'LLM FAMILY', 'MAYS', 'TREA', 'VANGUARD', 'SPDR', 'ISHARES',
    'MUTUAL FUND', 'ETP ', 'ETF ', 'GOVERNMENT SECUR', 'CASH ',
]


def is_cash_or_fund(ticker: str) -> bool:
    upper = ticker.upper()
    # Common pseudo-tickers for cash/treasurys
    if upper in ('US', 'TREA', 'NEW', '--', 'SP', 'CASH'):
        return True
    return any(p in upper for p in CASH_PATTERNS)


def main():
    con = duckdb.connect('data/congress.duckdb')
    
    # Get distinct tickers that don't look like proper ticker symbols
    rows = con.execute("""
        SELECT DISTINCT ticker FROM transactions
        WHERE ticker IS NOT NULL
          AND ticker NOT SIMILAR TO '[A-Z]{1,5}(\\.[AB])?'
          AND length(ticker) > 5
    """).fetchall()
    
    total_updated = 0
    total_deleted = 0
    total_nulled = 0
    distinct_fixed = 0
    
    for (orig,) in rows:
        upper = orig.upper()
        
        # 1. Try name-to-ticker mapping
        matched_ticker = None
        for name, ticker in sorted(NAME_TO_TICKER.items(), key=lambda x: -len(x[0])):
            if upper.startswith(name):
                matched_ticker = ticker
                break
        
        if matched_ticker:
            # Check for duplicates (ticker-version already exists for same key)
            duplicates = con.execute("""
                WITH name_rows AS (
                    SELECT doc_id, member, transaction_date, transaction_type
                    FROM transactions WHERE ticker=?
                )
                SELECT COUNT(*) FROM transactions t
                INNER JOIN name_rows n USING (doc_id, member, transaction_date, transaction_type)
                WHERE t.ticker=?
            """, [orig, matched_ticker]).fetchone()[0]
            
            if duplicates > 0:
                con.execute("""
                    DELETE FROM transactions WHERE ticker=?
                      AND (doc_id, member, transaction_date, transaction_type) IN (
                        SELECT doc_id, member, transaction_date, transaction_type
                        FROM transactions WHERE ticker=?
                      )
                """, [orig, matched_ticker])
                total_deleted += duplicates
            
            remaining = con.execute("SELECT COUNT(*) FROM transactions WHERE ticker=?", [orig]).fetchone()[0]
            if remaining > 0:
                con.execute("UPDATE transactions SET ticker=? WHERE ticker=?", [matched_ticker, orig])
                total_updated += remaining
                distinct_fixed += 1
            continue
        
        # 2. Cash/fund → NULL
        if is_cash_or_fund(orig):
            cnt = con.execute("SELECT COUNT(*) FROM transactions WHERE ticker=?", [orig]).fetchone()[0]
            con.execute("UPDATE transactions SET ticker=NULL WHERE ticker=?", [orig])
            total_nulled += cnt
            continue
        
        # 3. No match — LEAVE ALONE (this is the bug-fix)
    
    # Also handle single-token junk
    for tok in ('US', 'TREA', 'NEW', '--', 'SP'):
        cnt = con.execute("SELECT COUNT(*) FROM transactions WHERE ticker=?", [tok]).fetchone()[0]
        if cnt:
            con.execute("UPDATE transactions SET ticker=NULL WHERE ticker=?", [tok])
            total_nulled += cnt

    # Fix 6: null out blacklisted garbage fragments regardless of length
    # (the query above only targets length > 5; short confirmed-garbage tickers
    # like UNIT, TECH, BERK, BANK etc. must also be cleared).
    for blacklisted in sorted(_TICKER_BLACKLIST):
        cnt = con.execute("SELECT COUNT(*) FROM transactions WHERE ticker=?", [blacklisted]).fetchone()[0]
        if cnt:
            con.execute("UPDATE transactions SET ticker=NULL WHERE ticker=?", [blacklisted])
            total_nulled += cnt
    
    con.execute("CHECKPOINT")
    print(f'Fixed {distinct_fixed} distinct name→ticker mappings ({total_updated} rows updated)')
    print(f'Deleted {total_deleted} duplicates after remap')
    print(f'NULL-ed {total_nulled} cash/fund rows')
    print()
    print('Final state:')
    print('  Total tx:', con.execute("SELECT COUNT(*) FROM transactions").fetchone()[0])
    print('  With ticker:', con.execute("SELECT COUNT(*) FROM transactions WHERE ticker IS NOT NULL").fetchone()[0])
    print('  Distinct tickers:', con.execute("SELECT COUNT(DISTINCT ticker) FROM transactions WHERE ticker IS NOT NULL").fetchone()[0])
    con.close()


if __name__ == "__main__":
    main()
