import pandas as pd
import re
from exceptions import ParsingError

def clean_text(text):
    if text is None:
        return ""
    return re.sub(r'\s+', ' ', str(text)).strip()

def extract_ticker_from_name(asset_name):
    if not asset_name:
        return None

    match = re.search(r'\((.*?)\)', asset_name)
    if match:
        ticker = match.group(1).strip()
        if 1 <= len(ticker) <= 5 and ticker.isalpha() and ticker.isupper():
            return ticker
    return None

def parse_pdf_table(table):
    if not table or len(table) < 2:
        return []

    header = [clean_text(h).lower() for h in table[0]]
    header_str = ' '.join(header)
    if not ('asset' in header_str and ('transaction' in header_str or 'type' in header_str)):
        return []

    col_map = {}
    for i, h in enumerate(header):
        if 'asset' in h and 'transaction' not in h:
            col_map['asset'] = i
        elif ('transaction' in h and 'type' in h) or (h == 'type' and 'transaction' not in h):
            col_map['transaction_type'] = i
        elif 'date' in h and 'notification' not in h and 'transaction' in h:
            col_map['date'] = i

    if 'asset' not in col_map:
        return []

    transactions = []
    for row in table[1:]:
        if all(c is None or str(c).strip() == '' for c in row):
            continue

        asset_name = clean_text(row[col_map['asset']])
        if not asset_name:
            continue

        ticker = extract_ticker_from_name(asset_name)
        if not ticker:
            continue

        raw_tx_type = clean_text(row[col_map.get('transaction_type', 0)])
        tx_type = 'Other'
        if raw_tx_type.lower().startswith('p'):
            tx_type = 'Purchase'
        elif raw_tx_type.lower().startswith('s'):
            tx_type = 'Sale'

        if tx_type in ['Purchase', 'Sale']:
            transactions.append({
                'ticker': ticker,
                'transaction_type': tx_type,
                'transaction_date': clean_text(row[col_map.get('date', 0)]),
            })

    return transactions

def normalize_quiver_data(raw_data):
    if not raw_data:
        raise ParsingError("Empty Quiver API response")

    df = pd.DataFrame(raw_data)
    if df.empty:
        raise ParsingError("Empty Quiver dataframe")

    required_cols = {'Representative', 'TransactionDate', 'ReportDate', 'Ticker', 'Transaction'}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ParsingError(f"Missing columns in Quiver data: {missing_cols}")

    df['ReportDate'] = pd.to_datetime(df['ReportDate'], errors='coerce')
    df['TransactionDate'] = pd.to_datetime(df['TransactionDate'], errors='coerce')

    result = df.rename(columns={
        'Representative': 'member',
        'TransactionDate': 'transaction_date',
        'ReportDate': 'disclosure_date',
        'Ticker': 'ticker',
        'Transaction': 'transaction_type'
    })[['member', 'transaction_date', 'disclosure_date', 'ticker', 'transaction_type']]

    result = result.dropna(subset=['transaction_date', 'disclosure_date'])
    if result.empty:
        raise ParsingError("No valid transactions after date parsing")

    return result

def normalize_house_metadata(content):
    if not content:
        raise ParsingError("Empty metadata content")

    lines = content.strip().split('\n')
    if len(lines) < 2:
        raise ParsingError("Insufficient metadata lines")

    header = [col.strip() for col in lines[0].split('\t')]
    data = []
    for line in lines[1:]:
        if line.strip():
            row = [col.strip() for col in line.split('\t')]
            if len(row) >= len(header):
                data.append(row[:len(header)])

    if not data:
        raise ParsingError("No data rows in metadata")

    df = pd.DataFrame(data, columns=header)

    if 'FilingDate' not in df.columns:
        raise ParsingError("Missing FilingDate column in metadata")

    df['FilingDate'] = pd.to_datetime(df['FilingDate'], errors='coerce')
    df = df.dropna(subset=['FilingDate'])

    if df.empty:
        raise ParsingError("No valid filing dates in metadata")

    return df

def consolidate_transactions(pdf_transactions, member_metadata):
    if not pdf_transactions:
        return pd.DataFrame()

    all_transactions = []
    for pdf_path, transactions in pdf_transactions.items():
        if not transactions:
            continue

        doc_id = pdf_path.stem
        member_info = member_metadata.get(doc_id)
        if not member_info:
            continue

        for tx in transactions:
            all_transactions.append({
                'member': f"{member_info['First']} {member_info['Last']}",
                'transaction_date': tx['transaction_date'],
                'disclosure_date': member_info['FilingDate'],
                'ticker': tx['ticker'],
                'transaction_type': tx['transaction_type'],
            })

    if not all_transactions:
        return pd.DataFrame()

    df = pd.DataFrame(all_transactions)
    df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
    df['disclosure_date'] = pd.to_datetime(df['disclosure_date'], errors='coerce')

    return df.dropna(subset=['transaction_date'])