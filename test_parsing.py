import unittest
import pandas as pd
from parsing import (
    clean_text, extract_ticker_from_name, parse_pdf_table,
    normalize_quiver_data, normalize_house_metadata, consolidate_transactions
)
from exceptions import ParsingError

class TestParsing(unittest.TestCase):

    def test_clean_text_basic(self):
        self.assertEqual(clean_text("  hello   world  "), "hello world")
        self.assertEqual(clean_text(None), "")
        self.assertEqual(clean_text(""), "")
        self.assertEqual(clean_text("normal text"), "normal text")

    def test_extract_ticker_from_name(self):
        self.assertEqual(extract_ticker_from_name("Apple Inc. (AAPL)"), "AAPL")
        self.assertEqual(extract_ticker_from_name("Google (GOOGL)"), "GOOGL")
        self.assertEqual(extract_ticker_from_name("Microsoft Corp (MSFT)"), "MSFT")

        self.assertIsNone(extract_ticker_from_name("No ticker here"))
        self.assertIsNone(extract_ticker_from_name("Too long ticker (TOOLONG)"))
        self.assertIsNone(extract_ticker_from_name("Numbers (123A)"))
        self.assertIsNone(extract_ticker_from_name("Lowercase (aapl)"))
        self.assertIsNone(extract_ticker_from_name(""))
        self.assertIsNone(extract_ticker_from_name(None))

    def test_parse_pdf_table_valid(self):
        table = [
            ['Asset Name', 'Transaction Type', 'Transaction Date'],
            ['Apple Inc. (AAPL)', 'Purchase', '2024-01-01'],
            ['Google LLC (GOOGL)', 'Sale', '2024-01-02'],
            ['Microsoft (MSFT)', 'Purchase', '2024-01-03']
        ]

        transactions = parse_pdf_table(table)

        self.assertEqual(len(transactions), 3)
        self.assertEqual(transactions[0]['ticker'], 'AAPL')
        self.assertEqual(transactions[0]['transaction_type'], 'Purchase')
        self.assertEqual(transactions[1]['ticker'], 'GOOGL')
        self.assertEqual(transactions[1]['transaction_type'], 'Sale')

    def test_parse_pdf_table_no_asset_column(self):
        table = [
            ['Name', 'Type', 'Date'],
            ['Apple Inc.', 'Purchase', '2024-01-01']
        ]

        transactions = parse_pdf_table(table)
        self.assertEqual(len(transactions), 0)

    def test_parse_pdf_table_empty_or_invalid(self):
        self.assertEqual(parse_pdf_table([]), [])
        self.assertEqual(parse_pdf_table(None), [])
        self.assertEqual(parse_pdf_table([['header']]), [])

    def test_parse_pdf_table_no_valid_tickers(self):
        table = [
            ['Asset Name', 'Transaction Type', 'Date'],
            ['Some Company', 'Purchase', '2024-01-01'],
            ['Another Firm', 'Sale', '2024-01-02']
        ]

        transactions = parse_pdf_table(table)
        self.assertEqual(len(transactions), 0)

    def test_normalize_quiver_data_valid(self):
        raw_data = [
            {
                'Representative': 'John Doe',
                'TransactionDate': '2024-01-01',
                'ReportDate': '2024-01-05',
                'Ticker': 'AAPL',
                'Transaction': 'Purchase'
            },
            {
                'Representative': 'Jane Smith',
                'TransactionDate': '2024-01-02',
                'ReportDate': '2024-01-06',
                'Ticker': 'GOOGL',
                'Transaction': 'Sale'
            }
        ]

        df = normalize_quiver_data(raw_data)

        self.assertEqual(len(df), 2)
        self.assertEqual(list(df.columns), ['member', 'transaction_date', 'disclosure_date', 'ticker', 'transaction_type'])
        self.assertEqual(df.iloc[0]['member'], 'John Doe')
        self.assertEqual(df.iloc[0]['ticker'], 'AAPL')

    def test_normalize_quiver_data_empty(self):
        with self.assertRaises(ParsingError):
            normalize_quiver_data([])

        with self.assertRaises(ParsingError):
            normalize_quiver_data(None)

    def test_normalize_quiver_data_missing_columns(self):
        raw_data = [{'Representative': 'John Doe'}]

        with self.assertRaises(ParsingError):
            normalize_quiver_data(raw_data)

    def test_normalize_house_metadata_valid(self):
        content = "DocID\tFirst\tLast\tFilingDate\tFilingType\n"
        content += "12345\tJohn\tDoe\t2024-01-01\tP\n"
        content += "67890\tJane\tSmith\t2024-01-02\tA\n"

        df = normalize_house_metadata(content)

        self.assertEqual(len(df), 2)
        self.assertTrue('FilingDate' in df.columns)
        self.assertEqual(df.iloc[0]['First'], 'John')
        self.assertEqual(df.iloc[0]['Last'], 'Doe')

    def test_normalize_house_metadata_empty(self):
        with self.assertRaises(ParsingError):
            normalize_house_metadata("")

        with self.assertRaises(ParsingError):
            normalize_house_metadata(None)

    def test_normalize_house_metadata_no_filing_date(self):
        content = "DocID\tFirst\tLast\n12345\tJohn\tDoe\n"

        with self.assertRaises(ParsingError):
            normalize_house_metadata(content)

    def test_consolidate_transactions_valid(self):
        from pathlib import Path

        pdf_transactions = {
            Path('12345.pdf'): [
                {'ticker': 'AAPL', 'transaction_type': 'Purchase', 'transaction_date': '2024-01-01'}
            ],
            Path('67890.pdf'): [
                {'ticker': 'GOOGL', 'transaction_type': 'Sale', 'transaction_date': '2024-01-02'}
            ]
        }

        member_metadata = {
            '12345': {'First': 'John', 'Last': 'Doe', 'FilingDate': pd.to_datetime('2024-01-05')},
            '67890': {'First': 'Jane', 'Last': 'Smith', 'FilingDate': pd.to_datetime('2024-01-06')}
        }

        df = consolidate_transactions(pdf_transactions, member_metadata)

        self.assertEqual(len(df), 2)
        self.assertEqual(df.iloc[0]['member'], 'John Doe')
        self.assertEqual(df.iloc[0]['ticker'], 'AAPL')
        self.assertEqual(df.iloc[1]['member'], 'Jane Smith')
        self.assertEqual(df.iloc[1]['ticker'], 'GOOGL')

    def test_consolidate_transactions_empty(self):
        df = consolidate_transactions({}, {})
        self.assertTrue(df.empty)

        df = consolidate_transactions(None, {})
        self.assertTrue(df.empty)

    def test_consolidate_transactions_no_member_info(self):
        from pathlib import Path

        pdf_transactions = {
            Path('12345.pdf'): [
                {'ticker': 'AAPL', 'transaction_type': 'Purchase', 'transaction_date': '2024-01-01'}
            ]
        }

        member_metadata = {}

        df = consolidate_transactions(pdf_transactions, member_metadata)
        self.assertTrue(df.empty)

if __name__ == '__main__':
    unittest.main()