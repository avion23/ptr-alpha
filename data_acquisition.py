import requests
import pandas as pd
import investpy
import yfinance as yf
import zipfile
from io import BytesIO
from datetime import timedelta
import re
import pdfplumber
import os
import pathlib
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_quiver_data():
    cache_path = pathlib.Path("data/quiver_transactions.csv")
    if cache_path.exists():
        logger.info(f"Loading cached Quiver data from {cache_path}")
        return pd.read_csv(cache_path, parse_dates=['transaction_date', 'disclosure_date'])

    url = "https://api.quiverquant.com/beta/bulk/congresstrading/2022,2023,2024"
    try:
        logger.info("Fetching Quiver Quantitative data")
        response = requests.get(url, timeout=30)
        if response.status_code != 200:
            logger.warning(f"Quiver API returned status {response.status_code}")
            return pd.DataFrame()

        data = response.json()
        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df['ReportDate'] = pd.to_datetime(df['ReportDate'])
        df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])

        result = df.rename(columns={
            'Representative': 'member',
            'TransactionDate': 'transaction_date',
            'ReportDate': 'disclosure_date',
            'Ticker': 'ticker',
            'Transaction': 'transaction_type'
        })[['member', 'transaction_date', 'disclosure_date', 'ticker', 'transaction_type']]

        os.makedirs("data", exist_ok=True)
        result.to_csv(cache_path, index=False)
        logger.info(f"Cached {len(result)} Quiver transactions to {cache_path}")
        return result
    except Exception as e:
        logger.error(f"Failed to fetch Quiver data: {e}")
        return pd.DataFrame()

TICKER_CACHE = {}
DATA_DIR = pathlib.Path("data")

def get_ticker_from_name(asset_name):
    if asset_name in TICKER_CACHE:
        return TICKER_CACHE[asset_name]

    match = re.search(r'\((.*?)\)', asset_name)
    if match:
        ticker = match.group(1).strip()
        if 1 <= len(ticker) <= 5 and ticker.isalpha() and ticker.isupper():
            TICKER_CACHE[asset_name] = ticker
            return ticker

    TICKER_CACHE[asset_name] = 'UNKNOWN'
    return 'UNKNOWN'


def clean_pdf_text(text):
    if text is None:
        return ""
    return re.sub(r'\s+', ' ', str(text)).strip()

def parse_ptr_pdf(pdf_path):
    transactions = []
    try:
        pass
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                for table in page.extract_tables():
                    if not table or len(table) < 2:
                        continue

                    header = [clean_pdf_text(h).lower() for h in table[0]]
                    if not ('asset' in str(header) and ('transaction' in str(header) or 'type' in str(header))):
                        continue

                    col_map = {}
                    for i, h in enumerate(header):
                        if 'asset' in h:
                            col_map['asset'] = i
                        if 'transaction' in h or 'type' in h:
                            col_map['transaction_type'] = i
                        if 'date' in h and 'notification' not in h:
                            col_map['date'] = i

                    if 'asset' not in col_map:
                        continue

                    for row in table[1:]:
                        if all(c is None or str(c).strip() == '' for c in row):
                            continue

                        asset_name = clean_pdf_text(row[col_map['asset']])
                        if not asset_name:
                            continue

                        raw_tx_type = clean_pdf_text(row[col_map.get('transaction_type', 0)])
                        tx_type = 'Other'
                        if raw_tx_type.lower().startswith('p'):
                            tx_type = 'Purchase'
                        elif raw_tx_type.lower().startswith('s'):
                            tx_type = 'Sale'

                        if tx_type in ['Purchase', 'Sale']:
                            transactions.append({
                                'ticker': get_ticker_from_name(asset_name),
                                'transaction_type': tx_type,
                                'transaction_date': clean_pdf_text(row[col_map.get('date', 0)]),
                            })

        valid_transactions = [tx for tx in transactions if tx['ticker'] != 'UNKNOWN']
        if valid_transactions:
            pass
        return valid_transactions
    except Exception as e:
        logger.warning(f"Failed to parse PDF {pdf_path}: {e}")
        return []

def fetch_house_metadata(year):
    year_dir = DATA_DIR / str(year)
    metadata_path = year_dir / "metadata.csv"

    if metadata_path.exists():
        logger.info(f"Loading cached metadata for {year}")
        return pd.read_csv(metadata_path)

    metadata_url = f"https://disclosures-clerk.house.gov/public_disc/financial-pdfs/{year}FD.ZIP"
    try:
        logger.info(f"Downloading metadata for {year} from House disclosures")
        response = requests.get(metadata_url, timeout=30)
        if response.status_code != 200:
            logger.warning(f"Failed to fetch metadata for {year}, status: {response.status_code}")
            return pd.DataFrame()

        with zipfile.ZipFile(BytesIO(response.content)) as z:
            text_files = [f for f in z.namelist() if f.endswith('.txt')]
            if not text_files:
                return pd.DataFrame()

            with z.open(text_files[0]) as f:
                content = f.read().decode('utf-8', errors='ignore')

        lines = content.strip().split('\n')
        if len(lines) < 2:
            return pd.DataFrame()

        header = [col.strip() for col in lines[0].split('\t')]
        data = []
        for line in lines[1:]:
            if line.strip():
                row = [col.strip() for col in line.split('\t')]
                if len(row) >= len(header):
                    data.append(row[:len(header)])

        df = pd.DataFrame(data, columns=header)
        df['FilingDate'] = pd.to_datetime(df['FilingDate'], errors='coerce')

        os.makedirs(year_dir, exist_ok=True)
        df.to_csv(metadata_path, index=False)
        logger.info(f"Cached metadata for {year}: {len(df)} records")
        return df
    except Exception as e:
        logger.error(f"Failed to fetch metadata for {year}: {e}")
        return pd.DataFrame()

def fetch_and_cache_pdfs(year):
    metadata = fetch_house_metadata(year)
    if metadata.empty:
        logger.error(f"No metadata found for {year}")
        return

    ptrs = metadata[metadata['FilingType'] == 'P']
    pdf_dir = DATA_DIR / str(year) / "pdfs"
    os.makedirs(pdf_dir, exist_ok=True)

    logger.info(f"Processing {len(ptrs)} PTR filings for {year}")

    session = requests.Session()
    downloaded = 0
    skipped = 0

    for _, row in ptrs.iterrows():
        doc_id = row['DocID']
        pdf_path = pdf_dir / f"{doc_id}.pdf"

        if pdf_path.exists():
            skipped += 1
            continue

        url = f"https://disclosures-clerk.house.gov/public_disc/ptr-pdfs/{year}/{doc_id}.pdf"
        try:
            response = session.get(url, timeout=30)
            if response.status_code == 200:
                with open(pdf_path, 'wb') as f:
                    f.write(response.content)
                downloaded += 1
                if downloaded % 50 == 0:
                    logger.info(f"Downloaded {downloaded} PDFs...")
            else:
                logger.warning(f"Failed to download {doc_id}: status {response.status_code}")
        except Exception as e:
            logger.error(f"Error downloading {doc_id}: {e}")

    logger.info(f"PDF download complete: {downloaded} downloaded, {skipped} skipped")

def parse_cached_pdfs(year):
    metadata = fetch_house_metadata(year)
    ptrs = metadata[metadata['FilingType'] == 'P']
    pdf_dir = DATA_DIR / str(year) / "pdfs"
    output_csv = DATA_DIR / str(year) / "transactions.csv"

    if not pdf_dir.exists():
        logger.error(f"PDF directory not found: {pdf_dir}")
        return

    logger.info(f"Parsing {len(ptrs)} PDFs for {year}")

    all_transactions = []
    processed = 0

    for _, row in ptrs.iterrows():
        pdf_path = pdf_dir / f"{row['DocID']}.pdf"
        if not pdf_path.exists():
            continue

        processed += 1
        if processed % 100 == 0:
            logger.info(f"Processed {processed} PDFs...")

        pdf_transactions = parse_ptr_pdf(pdf_path)
        for tx in pdf_transactions:
            all_transactions.append({
                'member': f"{row['First']} {row['Last']}",
                'transaction_date': tx['transaction_date'],
                'disclosure_date': row['FilingDate'],
                'ticker': tx['ticker'],
                'transaction_type': tx['transaction_type'],
            })

    if not all_transactions:
        logger.warning("No transactions found")
        return

    df = pd.DataFrame(all_transactions)
    df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
    df['disclosure_date'] = pd.to_datetime(df['disclosure_date'], errors='coerce')
    df.dropna(subset=['transaction_date'], inplace=True)

    os.makedirs(output_csv.parent, exist_ok=True)
    df.to_csv(output_csv, index=False)
    logger.info(f"Saved {len(df)} transactions to {output_csv}")

def load_cached_data(year):
    transactions_csv = DATA_DIR / str(year) / "transactions.csv"
    if not transactions_csv.exists():
        logger.error(f"No cached data found for {year}")
        return pd.DataFrame()

    df = pd.read_csv(transactions_csv, parse_dates=['transaction_date', 'disclosure_date'])
    logger.info(f"Loaded {len(df)} cached transactions for {year}")
    return df

def fetch_prices(tickers, start, end):
    all_tickers = sorted(list(set(tickers) | {"SPY"}))
    logger.info(f"Fetching price data for {len(all_tickers)} tickers (yfinance with investpy fallback)")

    try:
        data = yf.download(all_tickers, start=start, end=end, progress=False, auto_adjust=True)
        if not data.empty:
            if len(all_tickers) == 1:
                return pd.DataFrame({all_tickers[0]: data['Close']}) if 'Close' in data else pd.DataFrame()
            if 'Close' in data.columns:
                return data['Close'].dropna(axis=1, how='all')
    except Exception as e:
        logger.warning(f"yfinance failed: {e}, falling back to investpy")

    start_str, end_str = start.strftime('%d/%m/%Y'), end.strftime('%d/%m/%Y')
    ETF_MAP = {'SPY': 'SPDR S&P 500'}
    price_series = []

    for ticker in all_tickers:
        try:
            fetcher = investpy.get_etf_historical_data if ticker in ETF_MAP else investpy.get_stock_historical_data
            params = {
                'country': 'united states',
                'from_date': start_str,
                'to_date': end_str,
                ('etf' if ticker in ETF_MAP else 'stock'): ETF_MAP.get(ticker, ticker)
            }
            df = fetcher(**params)
            if not df.empty:
                price_series.append(df['Close'].rename(ticker))
        except Exception:
            continue

    return pd.concat(price_series, axis=1).dropna(axis=1, how='all') if price_series else pd.DataFrame()

def load_data(source, year=2024):
    if source == 'quiver':
        return fetch_quiver_data()
    return load_cached_data(year)