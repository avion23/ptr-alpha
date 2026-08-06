import unittest
import pandas as pd
from analyzer.parsing import (
    clean_text,
    parse_pdf_table,
    normalize_house_metadata,
    consolidate_transactions,
)
from analyzer.exceptions import ParsingError


class TestParsing(unittest.TestCase):
    def test_parse_headerless_single_transaction(self):
        table = [["Apple Inc. (AAPL)", "P", "01/02/2024"]]

        transactions = parse_pdf_table(table)

        self.assertEqual(len(transactions), 1)
        self.assertEqual(transactions[0]["ticker"], "AAPL")

    def test_clean_text_basic(self):
        self.assertEqual(clean_text("  hello   world  "), "hello world")
        self.assertEqual(clean_text(None), "")
        self.assertEqual(clean_text(""), "")
        self.assertEqual(clean_text("normal text"), "normal text")

    def test_parse_pdf_table_valid(self):
        table = [
            ["Asset Name", "Transaction Type", "Transaction Date"],
            ["Apple Inc. (AAPL)", "Purchase", "2024-01-01"],
            ["Google LLC (GOOGL)", "Sale", "2024-01-02"],
            ["Microsoft (MSFT)", "Purchase", "2024-01-03"],
        ]

        transactions = parse_pdf_table(table)

        self.assertEqual(len(transactions), 3)
        self.assertEqual(transactions[0]["ticker"], "AAPL")
        self.assertEqual(transactions[0]["transaction_type"], "Purchase")
        self.assertEqual(transactions[1]["ticker"], "GOOGL")
        self.assertEqual(transactions[1]["transaction_type"], "Sale")

    def test_parse_pdf_table_without_header_keeps_first_transaction(self):
        table = [
            ["Apple Inc. (AAPL)", "Purchase", "01/15/2024"],
            ["Google LLC (GOOGL)", "Sale", "01/16/2024"],
        ]

        transactions = parse_pdf_table(table)

        self.assertEqual([tx["ticker"] for tx in transactions], ["AAPL", "GOOGL"])

    def test_parse_pdf_table_merges_split_row_that_already_has_ticker(self):
        table = [
            ["Asset", "Type", "Date", "Amount"],
            ["Apple Inc. (AAPL)", "", "", ""],
            ["", "Purchase", "01/15/2024", "$1,001 - $15,000"],
        ]

        transactions = parse_pdf_table(table)

        self.assertEqual(len(transactions), 1)
        self.assertEqual(transactions[0]["ticker"], "AAPL")
        self.assertEqual(transactions[0]["transaction_type"], "Purchase")
        self.assertEqual(transactions[0]["transaction_date"], "01/15/2024")
        self.assertEqual(transactions[0]["amount_raw"], "$1,001 - $15,000")

    def test_parse_pdf_table_merges_transaction_split_across_three_rows(self):
        table = [
            ["Asset", "Type", "Date", "Amount"],
            ["Berkshire Hathaway", "", "", ""],
            ["Class B (BRK.B)", "", "", ""],
            ["", "Purchase", "01/15/2024", "$1,001 - $15,000"],
            ["Apple Inc. (AAPL)", "Sale", "01/16/2024", "$15,001 - $50,000"],
        ]

        transactions = parse_pdf_table(table)

        self.assertEqual([tx["ticker"] for tx in transactions], ["BRK.B", "AAPL"])
        self.assertEqual(
            transactions[0]["asset_description"], "Berkshire Hathaway Class B (BRK.B)"
        )
        self.assertEqual(transactions[0]["amount_raw"], "$1,001 - $15,000")

    def test_parse_pdf_table_house_owner_and_amount_columns(self):
        table = [
            [
                "Asset Name",
                "Owner",
                "Transaction Type",
                "Transaction Date",
                "Notification Date",
                "Amount",
            ],
            [
                "TransDigm Group Incorporated (TDG)",
                "Dependent Child",
                "P",
                "04/16/2026",
                "05/06/2026",
                "$15,001 - $50,000",
            ],
            [
                "Packaging Corporation of America (PKG)",
                "DC",
                "Purchase",
                "04/24/2026",
                "05/06/2026",
                "$1,001 - $15,000",
            ],
        ]

        transactions = parse_pdf_table(table)

        self.assertEqual(len(transactions), 2)
        self.assertEqual(transactions[0]["ticker"], "TDG")
        self.assertEqual(transactions[0]["owner_code"], "DC")
        self.assertEqual(transactions[0]["amount_raw"], "$15,001 - $50,000")
        self.assertAlmostEqual(transactions[0]["amount_midpoint"], 32500.5)
        self.assertEqual(transactions[1]["owner_code"], "DC")

    def test_parse_pdf_table_no_asset_column(self):
        table = [["Name", "Type", "Date"], ["Apple Inc.", "Purchase", "2024-01-01"]]

        transactions = parse_pdf_table(table)
        # "Apple Inc." resolves via company name matching even without explicit asset column
        self.assertEqual(len(transactions), 1)
        self.assertEqual(transactions[0]["ticker"], "AAPL")

    def test_parse_pdf_table_empty_or_invalid(self):
        self.assertEqual(parse_pdf_table([]), [])
        self.assertEqual(parse_pdf_table(None), [])
        self.assertEqual(parse_pdf_table([["header"]]), [])

    def test_parse_pdf_table_no_valid_tickers(self):
        table = [
            ["Asset Name", "Transaction Type", "Date"],
            ["Some Company", "Purchase", "2024-01-01"],
            ["Another Firm", "Sale", "2024-01-02"],
        ]

        transactions = parse_pdf_table(table)
        # No ticker patterns, but rows are kept with ticker=None for later backfill
        self.assertEqual(len(transactions), 2)
        self.assertTrue(all(t["ticker"] is None for t in transactions))

    def test_normalize_house_metadata_valid(self):
        content = "DocID\tFirst\tLast\tFilingDate\tFilingType\n"
        content += "12345\tJohn\tDoe\t2024-01-01\tP\n"
        content += "67890\tJane\tSmith\t2024-01-02\tA\n"

        df = normalize_house_metadata(content)

        self.assertEqual(len(df), 2)
        self.assertTrue("FilingDate" in df.columns)
        self.assertEqual(df.iloc[0]["First"], "John")
        self.assertEqual(df.iloc[0]["Last"], "Doe")

    def test_normalize_house_metadata_empty(self):
        with self.assertRaises(ParsingError):
            normalize_house_metadata("")

        with self.assertRaises(ParsingError):
            normalize_house_metadata(None)

    def test_normalize_house_metadata_no_filing_date(self):
        content = "DocID\tFirst\tLast\n12345\tJohn\tDoe\n"

        with self.assertRaises(ParsingError):
            normalize_house_metadata(content)

    def test_normalize_house_metadata_strips_utf8_bom(self):
        content = "\ufeffDocID\tFirst\tLast\tFilingDate\n123\tJane\tDoe\t2024-01-01\n"

        df = normalize_house_metadata(content)

        self.assertEqual(df.iloc[0]["DocID"], "123")

    def test_normalize_house_metadata_requires_identity_columns(self):
        content = "DocID\tFirst\tFilingDate\n123\tJane\t2024-01-01\n"

        with self.assertRaisesRegex(ParsingError, "Last"):
            normalize_house_metadata(content)

    def test_consolidate_transactions_valid(self):
        from pathlib import Path

        pdf_transactions = {
            Path("12345.pdf"): [
                {
                    "ticker": "AAPL",
                    "transaction_type": "Purchase",
                    "transaction_date": "2024-01-01",
                }
            ],
            Path("67890.pdf"): [
                {
                    "ticker": "GOOGL",
                    "transaction_type": "Sale",
                    "transaction_date": "2024-01-02",
                }
            ],
        }

        member_metadata = {
            "12345": {
                "First": "John",
                "Last": "Doe",
                "FilingDate": pd.to_datetime("2024-01-05"),
            },
            "67890": {
                "First": "Jane",
                "Last": "Smith",
                "FilingDate": pd.to_datetime("2024-01-06"),
            },
        }

        df = consolidate_transactions(pdf_transactions, member_metadata)

        self.assertEqual(len(df), 2)
        self.assertEqual(df.iloc[0]["member"], "John Doe")
        self.assertEqual(df.iloc[0]["ticker"], "AAPL")
        self.assertEqual(df.iloc[1]["member"], "Jane Smith")
        self.assertEqual(df.iloc[1]["ticker"], "GOOGL")

    def test_consolidate_transactions_preserves_owner_and_amount(self):
        from pathlib import Path

        pdf_transactions = {
            Path("12345.pdf"): [
                {
                    "ticker": "PKG",
                    "transaction_type": "Purchase",
                    "transaction_date": "2026-04-24",
                    "owner_code": "DC",
                    "amount_raw": "$1,001 - $15,000",
                    "amount_midpoint": 8000.5,
                }
            ]
        }
        member_metadata = {
            "12345": {
                "First": "April",
                "Last": "Delaney",
                "FilingDate": pd.to_datetime("2026-05-06"),
            }
        }

        df = consolidate_transactions(pdf_transactions, member_metadata)

        self.assertEqual(df.iloc[0]["owner_code"], "DC")
        self.assertEqual(df.iloc[0]["amount_raw"], "$1,001 - $15,000")
        self.assertAlmostEqual(df.iloc[0]["amount_midpoint"], 8000.5)

    def test_consolidate_transactions_empty(self):
        df = consolidate_transactions({}, {})
        self.assertTrue(df.empty)

        df = consolidate_transactions(None, {})
        self.assertTrue(df.empty)

    def test_consolidate_transactions_no_member_info(self):
        from pathlib import Path

        pdf_transactions = {
            Path("12345.pdf"): [
                {
                    "ticker": "AAPL",
                    "transaction_type": "Purchase",
                    "transaction_date": "2024-01-01",
                }
            ]
        }

        member_metadata = {}

        df = consolidate_transactions(pdf_transactions, member_metadata)
        self.assertTrue(df.empty)

    def test_parse_pdf_table_mm_dd_yyyy_dates(self):
        table = [
            ["Asset", "Type", "Date"],
            ["Apple Inc. (AAPL)", "Purchase", "01/15/2024"],
            ["Tesla Inc. (TSLA)", "Sale", "02/20/2024"],
        ]
        transactions = parse_pdf_table(table)
        self.assertEqual(len(transactions), 2)
        self.assertEqual(transactions[0]["transaction_date"], "01/15/2024")
        self.assertEqual(transactions[1]["transaction_date"], "02/20/2024")
        self.assertEqual(transactions[1]["ticker"], "TSLA")
        self.assertEqual(transactions[1]["transaction_type"], "Sale")

    def test_parse_pdf_table_single_row(self):
        table = [
            ["Asset", "Type", "Date"],
            ["Apple Inc. (AAPL)", "Purchase", "01/15/2024"],
        ]
        transactions = parse_pdf_table(table)
        self.assertEqual(len(transactions), 1)
        self.assertEqual(transactions[0]["ticker"], "AAPL")
        self.assertEqual(transactions[0]["transaction_type"], "Purchase")
        self.assertEqual(transactions[0]["transaction_date"], "01/15/2024")

    def test_parse_pdf_table_mixed_valid_invalid(self):
        table = [
            ["Asset Name", "Transaction Type", "Transaction Date"],
            ["Apple Inc. (AAPL)", "Purchase", "2024-01-01"],
            ["No Ticker Company", "Purchase", "2024-01-02"],
            ["Google LLC (GOOGL)", "Sale", "2024-01-03"],
            ["Another No Ticker", "Sale", "2024-01-04"],
        ]
        transactions = parse_pdf_table(table)
        # All valid rows are now kept (even without tickers) for later backfill.
        # Rows with their own transaction fields are not treated as continuations.
        self.assertEqual(len(transactions), 4)
        tickers = [t["ticker"] for t in transactions]
        self.assertIn("AAPL", tickers)
        self.assertIn("GOOGL", tickers)
        none_count = sum(1 for t in transactions if t["ticker"] is None)
        self.assertEqual(none_count, 2)
        self.assertEqual(transactions[0]["ticker"], "AAPL")
        self.assertEqual(transactions[2]["ticker"], "GOOGL")

    def test_normalize_house_metadata_mm_dd_yyyy(self):
        content = "DocID\tFirst\tLast\tFilingDate\tFilingType\n"
        content += "001\tJohn\tDoe\t01/15/2024\tP\n"
        content += "002\tJane\tSmith\t01/20/2024\tP"
        df = normalize_house_metadata(content)
        self.assertEqual(len(df), 2)
        self.assertEqual(df.iloc[0]["First"], "John")
        self.assertEqual(df.iloc[1]["Last"], "Smith")

    def test_normalize_house_metadata_insufficient_lines(self):
        with self.assertRaises(ParsingError):
            normalize_house_metadata("DocID\tFirst\tLast")
        with self.assertRaises(ParsingError):
            normalize_house_metadata("single line")

    def test_normalize_house_metadata_no_data_rows(self):
        content = "DocID\tFirst\tLast\tFilingDate\n"
        with self.assertRaises(ParsingError):
            normalize_house_metadata(content)

    def test_normalize_house_metadata_rejects_blank_and_duplicate_doc_ids(self):
        header = "DocID\tFirst\tLast\tFilingDate\tFilingType\n"
        with self.assertRaisesRegex(ParsingError, "Blank DocID"):
            normalize_house_metadata(header + "\tA\tOne\t01/01/2024\tP")
        with self.assertRaisesRegex(ParsingError, "Duplicate DocID"):
            normalize_house_metadata(
                header + "1\tA\tOne\t01/01/2024\tP\n1\tB\tTwo\t01/02/2024\tP"
            )
        duplicate = normalize_house_metadata(
            header + "1\tA\tOne\t01/01/2024\tP\n1\tA\tOne\t01/01/2024\tP"
        )
        self.assertEqual(len(duplicate), 1)

    def test_normalize_house_metadata_drops_extra_column_rows(self):
        content = (
            "DocID\tFirst\tLast\tFilingDate\tFilingType\n"
            "bad\tA\tOne\t01/01/2024\tP\textra\n"
            "good\tB\tTwo\t01/02/2024\tP"
        )
        assert normalize_house_metadata(content)["DocID"].tolist() == ["good"]

    def test_normalize_house_metadata_invalid_dates(self):
        content = "DocID\tFirst\tLast\tFilingDate\n"
        content += "001\tJohn\tDoe\tnotadate\n"
        with self.assertRaises(ParsingError):
            normalize_house_metadata(content)

    def test_normalize_house_metadata_rejects_extra_columns(self):
        content = (
            "DocID\tFirst\tLast\tFilingDate\n001\tJohn\tDoe\t2024-01-01\tshifted\n"
        )

        with self.assertRaisesRegex(ParsingError, "No data rows"):
            normalize_house_metadata(content)

    def test_normalize_house_metadata_rejects_blank_doc_id(self):
        content = "DocID\tFirst\tLast\tFilingDate\n\tJohn\tDoe\t2024-01-01\n"

        with self.assertRaisesRegex(ParsingError, "Blank DocID"):
            normalize_house_metadata(content)

    def test_normalize_house_metadata_rejects_duplicate_doc_ids(self):
        content = (
            "DocID\tFirst\tLast\tFilingDate\n"
            "001\tJohn\tDoe\t2024-01-01\n"
            "001\tJane\tRoe\t2024-01-02\n"
        )

        with self.assertRaisesRegex(ParsingError, "Duplicate DocID"):
            normalize_house_metadata(content)

    def test_parse_pdf_table_single_letter_ticker(self):
        table = [
            ["Asset", "Type", "Date"],
            ["Ford Motor Co (F)", "Purchase", "01/15/2024"],
            ["Dominion Energy (D)", "Sale", "01/16/2024"],
        ]
        transactions = parse_pdf_table(table)
        self.assertEqual(len(transactions), 2)
        self.assertEqual(transactions[0]["ticker"], "F")
        self.assertEqual(transactions[1]["ticker"], "D")

    def test_parse_pdf_table_owner_code_normalization(self):
        table = [
            ["Asset Name", "Owner", "Transaction Type", "Transaction Date"],
            ["Apple Inc (AAPL)", "Spouse", "Purchase", "2024-01-01"],
            ["Microsoft Corp (MSFT)", "Joint", "Sale", "2024-01-02"],
            ["Ford Motor Co (F)", "Self", "Purchase", "2024-01-03"],
        ]
        transactions = parse_pdf_table(table)
        self.assertEqual(len(transactions), 3)
        self.assertEqual(transactions[0]["owner_code"], "SP")
        self.assertEqual(transactions[1]["owner_code"], "J")
        self.assertEqual(transactions[2]["owner_code"], "S")
        self.assertEqual(transactions[2]["ticker"], "F")

    def test_parse_pdf_table_header_not_in_row_0(self):
        table = [
            ["Quarterly Transaction Report 2024"],
            ["Asset Name", "Transaction Type", "Transaction Date"],
            ["Apple Inc. (AAPL)", "Purchase", "2024-01-01"],
            ["Google LLC (GOOGL)", "Sale", "2024-01-02"],
        ]
        transactions = parse_pdf_table(table)
        self.assertEqual(len(transactions), 2)
        self.assertEqual(transactions[0]["ticker"], "AAPL")
        self.assertEqual(transactions[0]["transaction_type"], "Purchase")
        self.assertEqual(transactions[1]["ticker"], "GOOGL")
        self.assertEqual(transactions[1]["transaction_type"], "Sale")

    def test_parse_pdf_table_header_with_owner_in_row_1(self):
        table = [
            ["Filing Header"],
            ["Asset Name", "Owner", "Transaction Type", "Transaction Date", "Amount"],
            [
                "TransDigm Group Inc (TDG)",
                "Dependent Child",
                "P",
                "04/16/2026",
                "$15,001 - $50,000",
            ],
        ]
        transactions = parse_pdf_table(table)
        self.assertEqual(len(transactions), 1)
        self.assertEqual(transactions[0]["ticker"], "TDG")
        self.assertEqual(transactions[0]["owner_code"], "DC")

    def test_parse_pdf_table_amount_fallback_no_amount_col(self):
        """When no amount column exists in headers, fallback should find $ pattern in cells."""
        table = [
            ["Asset Name", "Transaction Type", "Transaction Date", "Extra Col"],
            ["Apple Inc. (AAPL)", "Purchase", "01/15/2024", "$1,001 - $15,000"],
        ]
        transactions = parse_pdf_table(table)
        self.assertEqual(len(transactions), 1)
        self.assertEqual(transactions[0]["amount_raw"], "$1,001 - $15,000")
        self.assertAlmostEqual(transactions[0]["amount_midpoint"], 8000.5)

    # --- Full parse with options ---

    def test_parse_pdf_table_with_options(self):
        table = [
            ["Asset Name", "Transaction Type", "Transaction Date"],
            ["NVIDIA Corp Call Option (NVDA)", "Purchase", "2024-01-15"],
            ["Apple Inc. Common Stock (AAPL)", "Sale", "2024-01-16"],
        ]
        transactions = parse_pdf_table(table)
        self.assertEqual(len(transactions), 2)
        self.assertEqual(transactions[0]["instrument_type"], "call")
        self.assertEqual(transactions[1]["instrument_type"], "stock")

    def test_parse_pdf_table_with_put_option(self):
        table = [
            ["Asset Name", "Transaction Type", "Transaction Date"],
            ["Tesla Inc Put Option (TSLA)", "Sale", "2024-03-01"],
        ]
        transactions = parse_pdf_table(table)
        self.assertEqual(len(transactions), 1)
        self.assertEqual(transactions[0]["instrument_type"], "put")
        self.assertEqual(transactions[0]["ticker"], "TSLA")

    def test_parse_pdf_table_exchange_type(self):
        table = [
            ["Asset Name", "Transaction Type", "Transaction Date"],
            ["Apple Inc. (AAPL)", "Exchange", "2024-01-15"],
        ]
        transactions = parse_pdf_table(table)
        self.assertEqual(len(transactions), 1)
        self.assertEqual(transactions[0]["transaction_type"], "Exchange")
        self.assertEqual(transactions[0]["ticker"], "AAPL")

    def test_parse_pdf_table_single_digit_dates(self):
        table = [
            ["Asset Name", "Transaction Type", "Transaction Date"],
            ["Apple Inc. (AAPL)", "Purchase", "1/5/2024"],
            ["Google LLC (GOOGL)", "Sale", "3/7/2024"],
        ]
        transactions = parse_pdf_table(table)
        self.assertEqual(len(transactions), 2)
        self.assertEqual(transactions[0]["transaction_date"], "1/5/2024")
        self.assertEqual(transactions[1]["transaction_date"], "3/7/2024")


if __name__ == "__main__":
    unittest.main()
