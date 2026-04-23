import unittest
import pandas as pd
from analyzer.parsing import (
    clean_text, _extract_ticker, _extract_date, _extract_transaction_type,
    parse_pdf_table, normalize_house_metadata, consolidate_transactions
)
from analyzer.exceptions import ParsingError

class TestParsing(unittest.TestCase):

    def test_clean_text_basic(self):
        self.assertEqual(clean_text("  hello   world  "), "hello world")
        self.assertEqual(clean_text(None), "")
        self.assertEqual(clean_text(""), "")
        self.assertEqual(clean_text("normal text"), "normal text")

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

    def test_extract_ticker_valid(self):
        self.assertEqual(_extract_ticker("Apple Inc. (AAPL)"), "AAPL")
        self.assertEqual(_extract_ticker("Google LLC (GOOGL)"), "GOOGL")
        self.assertEqual(_extract_ticker("Microsoft Corp (MSFT)"), "MSFT")
        self.assertEqual(_extract_ticker("BRK-B (BRK-B)"), "BRK-B")

    def test_extract_ticker_no_parens(self):
        self.assertIsNone(_extract_ticker("Apple Inc"))
        self.assertIsNone(_extract_ticker("Some Company Name"))
        self.assertIsNone(_extract_ticker("lowercase (nope)"))

    def test_extract_ticker_empty(self):
        self.assertIsNone(_extract_ticker(""))
        self.assertIsNone(_extract_ticker(None))

    def test_extract_ticker_multiple_parens(self):
        self.assertEqual(_extract_ticker("Apple Inc. (AAPL) extra (STUFF)"), "AAPL")
        self.assertEqual(_extract_ticker("Fund (ABC) Def (XYZ)"), "ABC")

    def test_extract_date_mm_dd_yyyy(self):
        self.assertEqual(_extract_date("01/15/2024"), "01/15/2024")
        self.assertEqual(_extract_date("Transaction on 12/31/2023 confirmed"), "12/31/2023")

    def test_extract_date_yyyy_mm_dd(self):
        self.assertEqual(_extract_date("2024-01-15"), "2024-01-15")
        self.assertEqual(_extract_date("Date: 2023-12-31 end"), "2023-12-31")

    def test_extract_date_no_date(self):
        self.assertIsNone(_extract_date("no date here"))
        self.assertIsNone(_extract_date("random text without dates"))

    def test_extract_date_empty(self):
        self.assertIsNone(_extract_date(""))
        self.assertIsNone(_extract_date(None))

    def test_extract_transaction_type_purchase_variants(self):
        self.assertEqual(_extract_transaction_type("Purchase"), "Purchase")
        self.assertEqual(_extract_transaction_type("PURCHASE"), "Purchase")
        self.assertEqual(_extract_transaction_type("P"), "Purchase")
        self.assertEqual(_extract_transaction_type("p"), "Purchase")
        self.assertEqual(_extract_transaction_type("  purchase  "), "Purchase")
        self.assertEqual(_extract_transaction_type("Partial Purchase"), "Purchase")

    def test_extract_transaction_type_sale_variants(self):
        self.assertEqual(_extract_transaction_type("Sale"), "Sale")
        self.assertEqual(_extract_transaction_type("SALE"), "Sale")
        self.assertEqual(_extract_transaction_type("S"), "Sale")
        self.assertEqual(_extract_transaction_type("s"), "Sale")
        self.assertEqual(_extract_transaction_type("  sale  "), "Sale")
        self.assertEqual(_extract_transaction_type("Sell"), "Sale")
        self.assertEqual(_extract_transaction_type("Partial Sale"), "Sale")

    def test_extract_transaction_type_unknown(self):
        self.assertIsNone(_extract_transaction_type("Exchange"))
        self.assertIsNone(_extract_transaction_type("X"))
        self.assertIsNone(_extract_transaction_type("Hold"))
        self.assertIsNone(_extract_transaction_type(""))
        self.assertIsNone(_extract_transaction_type(None))

    def test_parse_pdf_table_mm_dd_yyyy_dates(self):
        table = [
            ['Asset', 'Type', 'Date'],
            ['Apple Inc. (AAPL)', 'Purchase', '01/15/2024'],
            ['Tesla Inc. (TSLA)', 'Sale', '02/20/2024']
        ]
        transactions = parse_pdf_table(table)
        self.assertEqual(len(transactions), 2)
        self.assertEqual(transactions[0]['transaction_date'], '01/15/2024')
        self.assertEqual(transactions[1]['transaction_date'], '02/20/2024')
        self.assertEqual(transactions[1]['ticker'], 'TSLA')
        self.assertEqual(transactions[1]['transaction_type'], 'Sale')

    def test_parse_pdf_table_single_row(self):
        table = [
            ['Asset', 'Type', 'Date'],
            ['Apple Inc. (AAPL)', 'Purchase', '01/15/2024']
        ]
        transactions = parse_pdf_table(table)
        self.assertEqual(len(transactions), 1)
        self.assertEqual(transactions[0]['ticker'], 'AAPL')
        self.assertEqual(transactions[0]['transaction_type'], 'Purchase')
        self.assertEqual(transactions[0]['transaction_date'], '01/15/2024')

    def test_parse_pdf_table_mixed_valid_invalid(self):
        table = [
            ['Asset Name', 'Transaction Type', 'Transaction Date'],
            ['Apple Inc. (AAPL)', 'Purchase', '2024-01-01'],
            ['No Ticker Company', 'Purchase', '2024-01-02'],
            ['Google LLC (GOOGL)', 'Sale', '2024-01-03'],
            ['Another No Ticker', 'Sale', '2024-01-04']
        ]
        transactions = parse_pdf_table(table)
        self.assertEqual(len(transactions), 2)
        self.assertEqual(transactions[0]['ticker'], 'AAPL')
        self.assertEqual(transactions[1]['ticker'], 'GOOGL')

    def test_normalize_house_metadata_mm_dd_yyyy(self):
        content = "DocID\tFirst\tLast\tFilingDate\tFilingType\n"
        content += "001\tJohn\tDoe\t01/15/2024\tP\n"
        content += "002\tJane\tSmith\t01/20/2024\tP"
        df = normalize_house_metadata(content)
        self.assertEqual(len(df), 2)
        self.assertEqual(df.iloc[0]['First'], 'John')
        self.assertEqual(df.iloc[1]['Last'], 'Smith')

    def test_normalize_house_metadata_insufficient_lines(self):
        with self.assertRaises(ParsingError):
            normalize_house_metadata("DocID\tFirst\tLast")
        with self.assertRaises(ParsingError):
            normalize_house_metadata("single line")

    def test_normalize_house_metadata_no_data_rows(self):
        content = "DocID\tFirst\tLast\tFilingDate\n"
        with self.assertRaises(ParsingError):
            normalize_house_metadata(content)

    def test_normalize_house_metadata_invalid_dates(self):
        content = "DocID\tFirst\tLast\tFilingDate\n"
        content += "001\tJohn\tDoe\tnotadate\n"
        with self.assertRaises(ParsingError):
            normalize_house_metadata(content)

if __name__ == '__main__':
    unittest.main()