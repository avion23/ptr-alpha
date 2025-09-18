import requests
import pandas as pd
import yfinance as yf
import investpy
import zipfile
import pdfplumber
import os
import pathlib
import logging
from io import BytesIO
from multiprocessing import Pool, cpu_count
from exceptions import DataSourceError, ParsingError
from parsing import parse_pdf_table, normalize_quiver_data, normalize_house_metadata, consolidate_transactions

logger = logging.getLogger(__name__)

class Config:
    def __init__(self, data_dir="data", cache_enabled=True, parallel_workers=None):
        self.data_dir = pathlib.Path(data_dir)
        self.cache_enabled = cache_enabled
        self.parallel_workers = parallel_workers or max(1, cpu_count() - 1)
        self.quiver_url = "https://api.quiverquant.com/beta/bulk/congresstrading/2022,2023,2024"
        self.house_metadata_url_template = "https://disclosures-clerk.house.gov/public_disc/financial-pdfs/{year}FD.ZIP"
        self.house_pdf_url_template = "https://disclosures-clerk.house.gov/public_disc/ptr-pdfs/{year}/{doc_id}.pdf"

def fetch_quiver_data(config):
    cache_path = config.data_dir / "quiver_transactions.csv"

    if config.cache_enabled and cache_path.exists():
        logger.info(f"Loading cached Quiver data from {cache_path}")
        try:
            return pd.read_csv(cache_path, parse_dates=['transaction_date', 'disclosure_date'])
        except Exception as e:
            logger.warning(f"Failed to load cached data: {e}")

    try:
        logger.info("Fetching Quiver Quantitative data")
        response = requests.get(config.quiver_url, timeout=30)
        if response.status_code != 200:
            raise DataSourceError(f"Quiver API returned status {response.status_code}")

        data = response.json()
        result = normalize_quiver_data(data)

        if config.cache_enabled:
            os.makedirs(config.data_dir, exist_ok=True)
            result.to_csv(cache_path, index=False)
            logger.info(f"Cached {len(result)} Quiver transactions to {cache_path}")

        return result
    except requests.RequestException as e:
        raise DataSourceError(f"Failed to fetch Quiver data: {e}")

def fetch_house_metadata(year, config):
    year_dir = config.data_dir / str(year)
    metadata_path = year_dir / "metadata.csv"

    if config.cache_enabled and metadata_path.exists():
        logger.info(f"Loading cached metadata for {year}")
        try:
            return pd.read_csv(metadata_path)
        except Exception as e:
            logger.warning(f"Failed to load cached metadata: {e}")

    metadata_url = config.house_metadata_url_template.format(year=year)
    try:
        logger.info(f"Downloading metadata for {year} from House disclosures")
        response = requests.get(metadata_url, timeout=30)
        if response.status_code != 200:
            raise DataSourceError(f"Failed to fetch metadata for {year}, status: {response.status_code}")

        with zipfile.ZipFile(BytesIO(response.content)) as z:
            text_files = [f for f in z.namelist() if f.endswith('.txt')]
            if not text_files:
                raise ParsingError(f"No text files found in metadata ZIP for {year}")

            with z.open(text_files[0]) as f:
                content = f.read().decode('utf-8', errors='ignore')

        df = normalize_house_metadata(content)

        if config.cache_enabled:
            os.makedirs(year_dir, exist_ok=True)
            df.to_csv(metadata_path, index=False)
            logger.info(f"Cached metadata for {year}: {len(df)} records")

        return df
    except requests.RequestException as e:
        raise DataSourceError(f"Failed to fetch metadata for {year}: {e}")

def _download_pdf_worker(args):
    doc_id, year, pdf_path, url = args
    if pdf_path.exists():
        return f"skipped:{doc_id}"

    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            with open(pdf_path, 'wb') as f:
                f.write(response.content)
            return f"downloaded:{doc_id}"
        else:
            return f"failed:{doc_id}:{response.status_code}"
    except Exception as e:
        return f"error:{doc_id}:{e}"

def fetch_and_cache_pdfs(year, config):
    metadata = fetch_house_metadata(year, config)
    ptrs = metadata[metadata['FilingType'] == 'P']
    pdf_dir = config.data_dir / str(year) / "pdfs"
    os.makedirs(pdf_dir, exist_ok=True)

    logger.info(f"Processing {len(ptrs)} PTR filings for {year}")

    args_list = []
    for _, row in ptrs.iterrows():
        doc_id = row['DocID']
        pdf_path = pdf_dir / f"{doc_id}.pdf"
        url = config.house_pdf_url_template.format(year=year, doc_id=doc_id)
        args_list.append((doc_id, year, pdf_path, url))

    with Pool(config.parallel_workers) as pool:
        results = pool.map(_download_pdf_worker, args_list)

    downloaded = sum(1 for r in results if r.startswith('downloaded:'))
    skipped = sum(1 for r in results if r.startswith('skipped:'))
    failed = sum(1 for r in results if r.startswith('failed:') or r.startswith('error:'))

    logger.info(f"PDF download complete: {downloaded} downloaded, {skipped} skipped, {failed} failed")

def _parse_pdf_worker(pdf_path):
    try:
        transactions = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                for table in page.extract_tables():
                    transactions.extend(parse_pdf_table(table))
        return pdf_path, transactions
    except Exception as e:
        logger.warning(f"Failed to parse PDF {pdf_path}: {e}")
        return pdf_path, []

def parse_cached_pdfs(year, config):
    metadata = fetch_house_metadata(year, config)
    ptrs = metadata[metadata['FilingType'] == 'P']
    pdf_dir = config.data_dir / str(year) / "pdfs"
    output_csv = config.data_dir / str(year) / "transactions.csv"

    if not pdf_dir.exists():
        raise DataSourceError(f"PDF directory not found: {pdf_dir}")

    logger.info(f"Parsing {len(ptrs)} PDFs for {year}")

    pdf_paths = []
    member_lookup = {}
    for _, row in ptrs.iterrows():
        pdf_path = pdf_dir / f"{row['DocID']}.pdf"
        if pdf_path.exists():
            pdf_paths.append(pdf_path)
            member_lookup[row['DocID']] = {
                'First': row['First'],
                'Last': row['Last'],
                'FilingDate': row['FilingDate']
            }

    if not pdf_paths:
        raise DataSourceError(f"No PDF files found in {pdf_dir}")

    with Pool(config.parallel_workers) as pool:
        results = pool.map(_parse_pdf_worker, pdf_paths)

    pdf_transactions = {pdf_path: transactions for pdf_path, transactions in results}
    df = consolidate_transactions(pdf_transactions, member_lookup)

    if df.empty:
        raise ParsingError("No transactions found after parsing all PDFs")

    os.makedirs(output_csv.parent, exist_ok=True)
    df.to_csv(output_csv, index=False)
    logger.info(f"Saved {len(df)} transactions to {output_csv}")

def load_cached_data(year, config):
    transactions_csv = config.data_dir / str(year) / "transactions.csv"
    if not transactions_csv.exists():
        raise DataSourceError(f"No cached data found for {year}")

    try:
        df = pd.read_csv(transactions_csv, parse_dates=['transaction_date', 'disclosure_date'])
        logger.info(f"Loaded {len(df)} cached transactions for {year}")
        return df
    except Exception as e:
        raise DataSourceError(f"Failed to load cached data for {year}: {e}")

def fetch_prices(tickers, start, end, config):
    if len(tickers) == 0:
        raise DataSourceError("No tickers provided for price fetching")

    all_tickers = sorted(list(set(tickers) | {"SPY"}))
    logger.info(f"Fetching price data for {len(all_tickers)} tickers")

    try:
        data = yf.download(all_tickers, start=start, end=end, progress=False, auto_adjust=True)
        if not data.empty:
            if len(all_tickers) == 1:
                if 'Close' in data:
                    return pd.DataFrame({all_tickers[0]: data['Close']})
                else:
                    raise DataSourceError(f"No Close prices found for {all_tickers[0]}")
            if 'Close' in data.columns:
                prices = data['Close'].dropna(axis=1, how='all')
                if prices.empty:
                    raise DataSourceError("All price data is NaN after cleaning")
                return prices
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

    if not price_series:
        raise DataSourceError("No price data could be fetched from any source")

    return pd.concat(price_series, axis=1).dropna(axis=1, how='all')

def load_data(source, year, config):
    if source == 'quiver':
        return fetch_quiver_data(config)
    return load_cached_data(year, config)

def save_results(table, output_format, member_filter, show_signals, config):
    if output_format == 'csv':
        if member_filter:
            filename = f"{member_filter.replace(' ', '_').lower()}_signals.csv"
        elif show_signals:
            filename = "top_signals.csv"
        else:
            filename = "member_rankings.csv"

        filepath = config.data_dir / filename
        os.makedirs(config.data_dir, exist_ok=True)
        table.to_csv(filepath, index=False)
        logger.info(f"Results saved to {filepath}")
        return True
    else:
        print(table.to_string(index=False))
        return True