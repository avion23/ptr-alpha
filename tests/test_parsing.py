import unittest
import pandas as pd
from analyzer.parsing import (
    clean_text, _extract_ticker, _extract_date, _extract_transaction_type,
    parse_pdf_table, normalize_house_metadata, consolidate_transactions,
    _extract_amount_midpoint, _extract_owner_code, _parse_ocr_text_to_rows,
    _find_header_row, _extract_instrument_type, _extract_option_details,
    _column_indexes, _find_amount_in_row,
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

    def test_parse_pdf_table_house_owner_and_amount_columns(self):
        table = [
            ['Asset Name', 'Owner', 'Transaction Type', 'Transaction Date', 'Notification Date', 'Amount'],
            ['TransDigm Group Incorporated (TDG)', 'Dependent Child', 'P', '04/16/2026', '05/06/2026', '$15,001 - $50,000'],
            ['Packaging Corporation of America (PKG)', 'DC', 'Purchase', '04/24/2026', '05/06/2026', '$1,001 - $15,000'],
        ]

        transactions = parse_pdf_table(table)

        self.assertEqual(len(transactions), 2)
        self.assertEqual(transactions[0]['ticker'], 'TDG')
        self.assertEqual(transactions[0]['owner_code'], 'DC')
        self.assertEqual(transactions[0]['amount_raw'], '$15,001 - $50,000')
        self.assertAlmostEqual(transactions[0]['amount_midpoint'], 32500.5)
        self.assertEqual(transactions[1]['owner_code'], 'DC')

    def test_extract_owner_code_dependent_child(self):
        self.assertEqual(_extract_owner_code('Dependent Child'), 'DC')
        self.assertEqual(_extract_owner_code('DC'), 'DC')
        self.assertIsNone(_extract_owner_code(''))

    def test_extract_owner_code_normalizes_house_codes(self):
        self.assertEqual(_extract_owner_code('Spouse'), 'SP')
        self.assertEqual(_extract_owner_code('SP'), 'SP')
        self.assertEqual(_extract_owner_code('Joint'), 'J')
        self.assertEqual(_extract_owner_code('J'), 'J')
        self.assertEqual(_extract_owner_code('Self'), 'S')
        self.assertEqual(_extract_owner_code('S'), 'S')

    def test_extract_owner_code_independent_not_dependent(self):
        self.assertNotEqual(_extract_owner_code('Independent'), 'DC')

    def test_extract_amount_midpoint_range(self):
        amount_raw, amount_midpoint = _extract_amount_midpoint('$1,001 - $15,000')

        self.assertEqual(amount_raw, '$1,001 - $15,000')
        self.assertAlmostEqual(amount_midpoint, 8000.5)

    def test_extract_amount_midpoint_no_dollar_sign(self):
        amount_raw, amount_midpoint = _extract_amount_midpoint('1,001 - 15,000')
        self.assertEqual(amount_raw, '1,001 - 15,000')
        self.assertIsNone(amount_midpoint)

    def test_extract_amount_midpoint_single_value(self):
        amount_raw, amount_midpoint = _extract_amount_midpoint('$50,000')
        self.assertEqual(amount_raw, '$50,000')
        self.assertAlmostEqual(amount_midpoint, 50000.0)

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

    def test_consolidate_transactions_preserves_owner_and_amount(self):
        from pathlib import Path

        pdf_transactions = {
            Path('12345.pdf'): [{
                'ticker': 'PKG',
                'transaction_type': 'Purchase',
                'transaction_date': '2026-04-24',
                'owner_code': 'DC',
                'amount_raw': '$1,001 - $15,000',
                'amount_midpoint': 8000.5,
            }]
        }
        member_metadata = {
            '12345': {'First': 'April', 'Last': 'Delaney', 'FilingDate': pd.to_datetime('2026-05-06')}
        }

        df = consolidate_transactions(pdf_transactions, member_metadata)

        self.assertEqual(df.iloc[0]['owner_code'], 'DC')
        self.assertEqual(df.iloc[0]['amount_raw'], '$1,001 - $15,000')
        self.assertAlmostEqual(df.iloc[0]['amount_midpoint'], 8000.5)

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
        # Lowercase tickers are now accepted (case-insensitive) and uppercased
        self.assertEqual(_extract_ticker("lowercase (nope)"), "NOPE")

    def test_extract_ticker_empty(self):
        self.assertIsNone(_extract_ticker(""))
        self.assertIsNone(_extract_ticker(None))

    def test_extract_ticker_multiple_parens(self):
        self.assertEqual(_extract_ticker("Apple Inc. (AAPL) extra (STUFF)"), "AAPL")
        self.assertEqual(_extract_ticker("Fund (ABC) Def (XYZ)"), "ABC")

    def test_extract_ticker_single_letter(self):
        self.assertEqual(_extract_ticker("Ford Motor (F)"), "F")
        self.assertEqual(_extract_ticker("Citigroup (C)"), "C")
        self.assertEqual(_extract_ticker("Visa Inc. (V)"), "V")
        self.assertEqual(_extract_ticker("Kellanova (K)"), "K")

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

    def test_parse_pdf_table_single_letter_ticker(self):
        table = [
            ['Asset', 'Type', 'Date'],
            ['Ford Motor Co (F)', 'Purchase', '01/15/2024'],
            ['Visa Inc (V)', 'Sale', '01/16/2024']
        ]
        transactions = parse_pdf_table(table)
        self.assertEqual(len(transactions), 2)
        self.assertEqual(transactions[0]['ticker'], 'F')
        self.assertEqual(transactions[1]['ticker'], 'V')

    def test_parse_ocr_text_single_letter_ticker(self):
        text = "Ford Motor Co (F) P 01/15/2024\nVisa Inc (V) S 01/16/2024\n"
        rows = _parse_ocr_text_to_rows(text)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0][0], "Ford Motor Co (F)")
        self.assertEqual(rows[1][0], "Visa Inc (V)")

    def test_parse_pdf_table_owner_code_normalization(self):
        table = [
            ['Asset Name', 'Owner', 'Transaction Type', 'Transaction Date'],
            ['Apple Inc (AAPL)', 'Spouse', 'Purchase', '2024-01-01'],
            ['Microsoft Corp (MSFT)', 'Joint', 'Sale', '2024-01-02'],
            ['Ford Motor Co (F)', 'Self', 'Purchase', '2024-01-03'],
        ]
        transactions = parse_pdf_table(table)
        self.assertEqual(len(transactions), 3)
        self.assertEqual(transactions[0]['owner_code'], 'SP')
        self.assertEqual(transactions[1]['owner_code'], 'J')
        self.assertEqual(transactions[2]['owner_code'], 'S')
        self.assertEqual(transactions[2]['ticker'], 'F')

    def test_extract_transaction_type_partial_suffix(self):
        self.assertEqual(_extract_transaction_type("P (partial)"), "Purchase")
        self.assertEqual(_extract_transaction_type("S (partial)"), "Sale")
        self.assertEqual(_extract_transaction_type("Purchase (partial)"), "Purchase")
        self.assertEqual(_extract_transaction_type("Sale (partial)"), "Sale")
        self.assertEqual(_extract_transaction_type("S (PARTIAL)"), "Sale")
        self.assertEqual(_extract_transaction_type("P (PARTIAL)"), "Purchase")

    def test_extract_transaction_type_bare_s_and_p(self):
        self.assertEqual(_extract_transaction_type("S"), "Sale")
        self.assertEqual(_extract_transaction_type("P"), "Purchase")
        self.assertEqual(_extract_transaction_type("s"), "Sale")
        self.assertEqual(_extract_transaction_type("p"), "Purchase")

    def test_extract_transaction_type_buy_sold(self):
        self.assertEqual(_extract_transaction_type("Buy"), "Purchase")
        self.assertEqual(_extract_transaction_type("Sold"), "Sale")
        self.assertEqual(_extract_transaction_type("buy"), "Purchase")
        self.assertEqual(_extract_transaction_type("sold"), "Sale")

    def test_extract_transaction_type_sale_purchase_full_words(self):
        self.assertEqual(_extract_transaction_type("Sale"), "Sale")
        self.assertEqual(_extract_transaction_type("Purchase"), "Purchase")

    def test_find_header_row_in_row_0(self):
        table = [
            ['Asset', 'Type', 'Date'],
            ['Apple Inc. (AAPL)', 'Purchase', '2024-01-01'],
        ]
        self.assertEqual(_find_header_row(table), 0)

    def test_find_header_row_in_row_1(self):
        table = [
            ['Some title row'],
            ['Asset Name', 'Transaction Type', 'Transaction Date'],
            ['Apple Inc. (AAPL)', 'Purchase', '2024-01-01'],
        ]
        self.assertEqual(_find_header_row(table), 1)

    def test_find_header_row_in_row_2(self):
        table = [
            ['Title row'],
            ['Subtitle row'],
            ['Asset Name', 'Owner', 'Transaction Type', 'Transaction Date'],
            ['Apple Inc. (AAPL)', 'Self', 'Purchase', '2024-01-01'],
        ]
        self.assertEqual(_find_header_row(table), 2)

    def test_find_header_row_returns_none_when_no_match(self):
        table = [
            ['Random', 'Stuff', 'Here'],
            ['More', 'Random', 'Data'],
        ]
        self.assertIsNone(_find_header_row(table))

    def test_parse_pdf_table_header_not_in_row_0(self):
        table = [
            ['Quarterly Transaction Report 2024'],
            ['Asset Name', 'Transaction Type', 'Transaction Date'],
            ['Apple Inc. (AAPL)', 'Purchase', '2024-01-01'],
            ['Google LLC (GOOGL)', 'Sale', '2024-01-02'],
        ]
        transactions = parse_pdf_table(table)
        self.assertEqual(len(transactions), 2)
        self.assertEqual(transactions[0]['ticker'], 'AAPL')
        self.assertEqual(transactions[0]['transaction_type'], 'Purchase')
        self.assertEqual(transactions[1]['ticker'], 'GOOGL')
        self.assertEqual(transactions[1]['transaction_type'], 'Sale')

    def test_parse_pdf_table_header_with_owner_in_row_1(self):
        table = [
            ['Filing Header'],
            ['Asset Name', 'Owner', 'Transaction Type', 'Transaction Date', 'Amount'],
            ['TransDigm Group Inc (TDG)', 'Dependent Child', 'P', '04/16/2026', '$15,001 - $50,000'],
        ]
        transactions = parse_pdf_table(table)
        self.assertEqual(len(transactions), 1)
        self.assertEqual(transactions[0]['ticker'], 'TDG')
        self.assertEqual(transactions[0]['owner_code'], 'DC')

    def test_parse_ocr_text_partial_suffix(self):
        text = "Apple Inc (AAPL) P (partial) 01/15/2024\nGoogle LLC (GOOGL) S (partial) 01/16/2024\n"
        rows = _parse_ocr_text_to_rows(text)
        # Partial suffixes on their own lines won't have a ticker on the same line
        # but the P/S prefix pattern should match
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0][1], "Purchase")
        self.assertEqual(rows[1][1], "Sale")

    # --- Column index robustness ---

    def test_column_indexes_amount_value_header(self):
        indexes = _column_indexes(['Asset', 'Type', 'Date', 'Value'])
        self.assertEqual(indexes['amount'], 3)

    def test_column_indexes_amount_proceeds_header(self):
        indexes = _column_indexes(['Description', 'TxType', 'TxDate', 'Proceeds'])
        self.assertEqual(indexes['amount'], 3)

    def test_column_indexes_amount_transaction_value_header(self):
        indexes = _column_indexes(['Asset Name', 'Transaction Type', 'Transaction Date', 'TransactionValue'])
        self.assertEqual(indexes['amount'], 3)

    def test_column_indexes_no_amount_col(self):
        indexes = _column_indexes(['Asset', 'Type', 'Date'])
        self.assertIsNone(indexes.get('amount'))

    def test_column_indexes_fallback_to_defaults_when_core_missing(self):
        indexes = _column_indexes(['Random', 'Stuff', 'Here'])
        self.assertEqual(indexes['asset'], 0)
        self.assertEqual(indexes['type'], 1)
        self.assertEqual(indexes['date'], 2)

    # --- Amount fallback ---

    def test_find_amount_in_row_range(self):
        row = ['Apple Inc (AAPL)', 'Purchase', '01/15/2024', '', '$1,001 - $15,000']
        self.assertEqual(_find_amount_in_row(row), '$1,001 - $15,000')

    def test_find_amount_in_row_single(self):
        row = ['Apple Inc (AAPL)', 'Purchase', '01/15/2024', '$50,000']
        self.assertEqual(_find_amount_in_row(row), '$50,000')

    def test_find_amount_in_row_none(self):
        row = ['Apple Inc (AAPL)', 'Purchase', '01/15/2024']
        self.assertIsNone(_find_amount_in_row(row))

    def test_find_amount_in_row_embedded(self):
        row = ['Apple Inc (AAPL)', 'Purchase', '01/15/2024', 'some text $10,001 - $50,000 here']
        self.assertEqual(_find_amount_in_row(row), '$10,001 - $50,000')

    def test_parse_pdf_table_amount_fallback_no_amount_col(self):
        """When no amount column exists in headers, fallback should find $ pattern in cells."""
        table = [
            ['Asset Name', 'Transaction Type', 'Transaction Date', 'Extra Col'],
            ['Apple Inc. (AAPL)', 'Purchase', '01/15/2024', '$1,001 - $15,000'],
        ]
        transactions = parse_pdf_table(table)
        self.assertEqual(len(transactions), 1)
        self.assertEqual(transactions[0]['amount_raw'], '$1,001 - $15,000')
        self.assertAlmostEqual(transactions[0]['amount_midpoint'], 8000.5)

    # --- Options detection ---

    def test_instrument_type_call_option(self):
        self.assertEqual(_extract_instrument_type('NVIDIA Corp Common Stock Call Option (NVDA)'), 'call')

    def test_instrument_type_put_option(self):
        self.assertEqual(_extract_instrument_type('NVIDIA Corp Common Stock Put Option (NVDA)'), 'put')

    def test_instrument_type_bare_call(self):
        self.assertEqual(_extract_instrument_type('NVDA Call $120 Exp 12/20/2024'), 'call')

    def test_instrument_type_bare_put(self):
        self.assertEqual(_extract_instrument_type('NVDA Put $120 Exp 12/20/2024'), 'put')

    def test_instrument_type_stock(self):
        self.assertEqual(_extract_instrument_type('Apple Inc. Common Stock (AAPL)'), 'stock')

    def test_instrument_type_option_without_call_put(self):
        """When 'option' appears with strike/expiry but no call/put, defaults to 'call'."""
        self.assertEqual(_extract_instrument_type('Some Fund Option Strike $50 Exp 06/30/2025'), 'call')

    def test_instrument_type_empty(self):
        self.assertEqual(_extract_instrument_type(''), 'stock')
        self.assertEqual(_extract_instrument_type(None), 'stock')

    # --- Option details extraction ---

    def test_option_details_strike_and_expiry(self):
        details = _extract_option_details('NVDA Call Strike $120 Exp 12/20/2024')
        self.assertAlmostEqual(details['strike_price'], 120.0)
        self.assertEqual(details['expiry_date'], '12/20/2024')

    def test_option_details_inline_strike(self):
        """$120 before 'Exp' should be captured as strike."""
        details = _extract_option_details('NVDA Call $150.50 Exp 06/30/2025')
        self.assertAlmostEqual(details['strike_price'], 150.50)
        self.assertEqual(details['expiry_date'], '06/30/2025')

    def test_option_details_no_details(self):
        details = _extract_option_details('Apple Inc (AAPL)')
        self.assertNotIn('strike_price', details)
        self.assertNotIn('expiry_date', details)

    def test_option_details_expiry_only(self):
        details = _extract_option_details('Some Option Exp 01/15/2025')
        self.assertNotIn('strike_price', details)
        self.assertEqual(details['expiry_date'], '01/15/2025')

    def test_option_details_strike_only(self):
        details = _extract_option_details('Some Option Strike $75')
        self.assertAlmostEqual(details['strike_price'], 75.0)
        self.assertNotIn('expiry_date', details)

    # --- Full parse with options ---

    def test_parse_pdf_table_with_options(self):
        table = [
            ['Asset Name', 'Transaction Type', 'Transaction Date'],
            ['NVIDIA Corp Call Option (NVDA)', 'Purchase', '2024-01-15'],
            ['Apple Inc. Common Stock (AAPL)', 'Sale', '2024-01-16'],
        ]
        transactions = parse_pdf_table(table)
        self.assertEqual(len(transactions), 2)
        self.assertEqual(transactions[0]['instrument_type'], 'call')
        self.assertEqual(transactions[1]['instrument_type'], 'stock')

    def test_parse_pdf_table_with_put_option(self):
        table = [
            ['Asset Name', 'Transaction Type', 'Transaction Date'],
            ['Tesla Inc Put Option (TSLA)', 'Sale', '2024-03-01'],
        ]
        transactions = parse_pdf_table(table)
        self.assertEqual(len(transactions), 1)
        self.assertEqual(transactions[0]['instrument_type'], 'put')
        self.assertEqual(transactions[0]['ticker'], 'TSLA')

if __name__ == '__main__':
    unittest.main()
