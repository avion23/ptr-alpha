import requests
import pandas as pd
import yfinance as yf
import zipfile
import pdfplumber
import os
import pathlib
import logging
import time
from io import BytesIO
from multiprocessing import Pool, cpu_count
from analyzer.exceptions import DataSourceError, ParsingError
from analyzer.parsing import parse_pdf_table, normalize_house_metadata, consolidate_transactions

logger = logging.getLogger(__name__)

class Config:
    def __init__(self, data_dir="data", cache_enabled=True, parallel_workers=None):
        self.data_dir = pathlib.Path(data_dir)
        self.cache_enabled = cache_enabled
        self.parallel_workers = parallel_workers or max(1, cpu_count() - 1)
        self.house_metadata_url_template = "https://disclosures-clerk.house.gov/public_disc/financial-pdfs/{year}FD.ZIP"
        self.house_pdf_url_template = "https://disclosures-clerk.house.gov/public_disc/ptr-pdfs/{year}/{doc_id}.pdf"


def fetch_house_metadata(year, config):
    year_dir = config.data_dir / str(year)
    metadata_path = year_dir / "metadata.parquet"

    if config.cache_enabled and metadata_path.exists():
        logger.info(f"Loading cached metadata for {year}")
        try:
            return pd.read_parquet(metadata_path)
        except (OSError, pd.errors.ParserError) as e:
            logger.warning(f"Failed to load cached metadata: {e}")
        except Exception as e:
            logger.exception(f"Unexpected error loading cached metadata: {e}")
            raise

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
            df.to_parquet(metadata_path, index=False)
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
    except (requests.RequestException, OSError) as e:
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
    except (OSError, ParsingError) as e:
        logger.warning(f"Failed to parse PDF {pdf_path}: {e}")
        return pdf_path, []

def parse_cached_pdfs(year, config):
    metadata = fetch_house_metadata(year, config)
    ptrs = metadata[metadata['FilingType'] == 'P']
    pdf_dir = config.data_dir / str(year) / "pdfs"
    output_file = config.data_dir / str(year) / "transactions.parquet"
    legacy_csv = config.data_dir / str(year) / "transactions.csv"

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

    os.makedirs(output_file.parent, exist_ok=True)
    df.to_parquet(output_file, index=False)

    if legacy_csv.exists():
        legacy_csv.unlink()
        logger.info(f"Deleted legacy CSV file")

    logger.info(f"Saved {len(df)} transactions to {output_file}")

def load_cached_data(year, config):
    transactions_parquet = config.data_dir / str(year) / "transactions.parquet"
    transactions_csv = config.data_dir / str(year) / "transactions.csv"

    if transactions_parquet.exists():
        try:
            df = pd.read_parquet(transactions_parquet)
            logger.info(f"Loaded {len(df)} cached transactions for {year}")
            return df
        except (OSError, pd.errors.ParserError) as e:
            raise DataSourceError(f"Corrupted Parquet cache for {year}: {e}. Run 'insider-trading parse --year {year}' to rebuild.")
        except Exception as e:
            logger.exception(f"Unexpected error loading Parquet cache: {e}")
            raise

    if transactions_csv.exists():
        logger.warning(f"Found legacy CSV data for {year}. Converting to Parquet format...")
        try:
            df = pd.read_csv(transactions_csv, parse_dates=['transaction_date', 'disclosure_date'])
            df.to_parquet(transactions_parquet, index=False)
            transactions_csv.unlink()
            logger.info(f"Converted {len(df)} transactions to Parquet and deleted legacy CSV")
            return df
        except (OSError, pd.errors.ParserError) as e:
            raise DataSourceError(f"Failed to convert legacy CSV data for {year}: {e}")
        except Exception as e:
            logger.exception(f"Unexpected error converting legacy CSV: {e}")
            raise

    raise DataSourceError(f"No cached data found for {year}. Run 'insider-trading parse --year {year}' first.")

def fetch_prices(tickers, start, end, config):
    if len(tickers) == 0:
        raise DataSourceError("No tickers provided for price fetching")

    all_tickers = sorted(list(set(tickers) | {"SPY"}))
    logger.info(f"Fetching price data for {len(all_tickers)} tickers using yfinance")

    data = yf.download(all_tickers, start=start, end=end, progress=False, threads=True, auto_adjust=True)

    if data.empty:
        raise DataSourceError("No price data could be fetched from yfinance. Data source may be blocked or down.")

    prices = data['Close'] if len(all_tickers) > 1 else data['Close'].to_frame(all_tickers[0])
    prices = prices.dropna(axis=1, how='all')

    failed_tickers = sorted(set(all_tickers) - set(prices.columns))
    success_count = len(prices.columns)
    success_rate = success_count / len(all_tickers)

    if success_rate < 0.9:
        raise DataSourceError(f"Price fetch failure rate too high: {(1-success_rate)*100:.1f}% failed ({len(failed_tickers)}/{len(all_tickers)}). Analysis would be unreliable.")

    if failed_tickers:
        logger.warning(f"Failed to fetch price data for {len(failed_tickers)} tickers: {', '.join(failed_tickers[:10])}{'...' if len(failed_tickers) > 10 else ''}")

    logger.info(f"Successfully fetched prices for {success_count}/{len(all_tickers)} tickers ({success_rate*100:.1f}% success)")
    return prices

def load_data(source, year, config):
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