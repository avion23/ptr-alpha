import unittest
import hashlib
import os
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import patch
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

    def test_pdfplumber_flattened_rows_do_not_pollute_following_asset(self):
        from analyzer.parsing.pdfplumber_parser import (
            _expand_flattened_transaction_rows,
        )

        table = [
            [
                "ID",
                "Owner",
                "Asset",
                "Transaction Type",
                "Date",
                "Notification Date",
                "Amount",
            ],
            [
                "Abbott Laboratories Common Stock P 06/16/2026 07/02/2026 "
                "$1,001 - $15,000\n(ABT) [ST]\nF S: New\nS O: Trust Account",
                "",
                "",
                "",
                "",
                "",
                "",
            ],
            ["", "", "F S: New\nS O: Trust Account", "", "", "", ""],
            [
                "",
                "",
                "Accenture plc Class A Ordinary Shares (ACN) [ST]",
                "P",
                "06/16/2026",
                "07/02/2026",
                "$1,001 - $15,000",
            ],
        ]

        transactions = parse_pdf_table(_expand_flattened_transaction_rows(table))

        self.assertEqual([tx["ticker"] for tx in transactions], ["ABT", "ACN"])
        self.assertEqual(
            transactions[1]["asset_description"],
            "Accenture plc Class A Ordinary Shares (ACN) [ST]",
        )

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
                    "asset_description": "Packaging Corporation of America (PKG)",
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
        self.assertEqual(
            df.iloc[0]["asset_description"],
            "Packaging Corporation of America (PKG)",
        )

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


class TestLocalOcrCanaries(unittest.TestCase):
    def test_tickerless_legacy_two_digit_year(self):
        from analyzer.parsing.ocr_parser import _parse_ocr_text_to_rows

        rows = _parse_ocr_text_to_rows(
            "Acme Private Fund [ST] P 01/15/24 $1,001 - $15,000"
        )
        self.assertEqual(
            rows,
            [["Acme Private Fund", "Purchase", "01/15/24", "$1,001 - $15,000"]],
        )

    def test_partial_first_orientation_is_reconciled_with_rotated_rows(self):
        from analyzer.parsing.ocr_parser import extract_tables_with_ocr

        class Image:
            def __init__(self, name):
                self.name = name

            def rotate(self, angle, expand):
                self.assert_rotate = (angle, expand)
                return Image("rotated")

            def close(self):
                pass

        pdf2image = ModuleType("pdf2image")
        pdf2image.convert_from_path = lambda path, dpi: [Image("original")]
        pytesseract = ModuleType("pytesseract")
        pytesseract.image_to_osd = lambda image: "Rotate: 90\n"
        pytesseract.image_to_string = lambda image: (
            "Apple (AAPL) P 01/15/24 $1,001 - $15,000"
            if image.name == "original"
            else "Microsoft P 01/16/24 $15,001 - $50,000"
        )
        with patch.dict(
            sys.modules, {"pdf2image": pdf2image, "pytesseract": pytesseract}
        ):
            tables = extract_tables_with_ocr(Path("canary.pdf"))

        self.assertEqual(len(tables[0]), 3)
        self.assertEqual(tables[0][1][0], "Apple (AAPL)")
        self.assertEqual(tables[0][2][0], "Microsoft")

    def test_orientation_failure_preserves_first_pass_rows_as_incomplete(self):
        from analyzer.parsing.ocr_parser import (
            OcrIncompleteError,
            extract_tables_with_ocr,
        )

        class Image:
            def close(self):
                pass

        pdf2image = ModuleType("pdf2image")
        pdf2image.convert_from_path = lambda path, dpi: [Image()]
        pytesseract = ModuleType("pytesseract")
        pytesseract.image_to_string = lambda image: (
            "Apple (AAPL) P 01/15/24 $1,001 - $15,000"
        )
        pytesseract.image_to_osd = lambda image: (_ for _ in ()).throw(
            RuntimeError("osd unavailable")
        )
        with patch.dict(
            sys.modules, {"pdf2image": pdf2image, "pytesseract": pytesseract}
        ):
            with self.assertRaises(OcrIncompleteError) as raised:
                extract_tables_with_ocr(Path("orientation.pdf"))
        self.assertEqual(raised.exception.partial_tables[0][1][0], "Apple (AAPL)")

    def test_zero_and_partial_page_ocr_are_incomplete(self):
        from analyzer.parsing.ocr_parser import (
            OcrIncompleteError,
            extract_tables_with_ocr,
        )

        class Image:
            def __init__(self, page):
                self.page = page

            def close(self):
                pass

        pdf2image = ModuleType("pdf2image")
        pdf2image.convert_from_path = lambda path, dpi: [Image(1), Image(2)]
        pytesseract = ModuleType("pytesseract")
        pytesseract.image_to_string = lambda image: (
            "Apple (AAPL) P 01/15/24 $1,001 - $15,000" if image.page == 1 else ""
        )
        pytesseract.image_to_osd = lambda image: "Rotate: 0\n"
        with patch.dict(
            sys.modules, {"pdf2image": pdf2image, "pytesseract": pytesseract}
        ):
            with self.assertRaisesRegex(OcrIncompleteError, "page 2") as raised:
                extract_tables_with_ocr(Path("partial.pdf"))
        self.assertEqual(len(raised.exception.partial_tables[0]), 2)

    def test_backend_failure_is_not_true_zero(self):
        from analyzer.parsing.ocr_parser import OcrBackendError, extract_tables_with_ocr

        pdf2image = ModuleType("pdf2image")
        pdf2image.convert_from_path = lambda path, dpi: (_ for _ in ()).throw(
            ValueError("corrupt")
        )
        pytesseract = ModuleType("pytesseract")
        with patch.dict(
            sys.modules, {"pdf2image": pdf2image, "pytesseract": pytesseract}
        ):
            with self.assertRaisesRegex(OcrBackendError, "failed to rasterize"):
                extract_tables_with_ocr(Path("corrupt.pdf"))

    def test_cascade_compares_all_text_engines_and_prefers_complete_trusted_tie(self):
        from analyzer import parser_cascade

        def transactions(count):
            return [
                {
                    "transaction_date": f"2024-01-{index + 1:02d}",
                    "transaction_type": "Purchase",
                    "amount_midpoint": 8000,
                    "asset_description": f"Asset {index}",
                }
                for index in range(count)
            ]

        with (
            patch.object(
                parser_cascade, "_try_pdfplumber", return_value=transactions(2)
            ),
            patch.object(parser_cascade, "_try_camelot_lattice", return_value=[]),
            patch.object(
                parser_cascade, "_try_camelot_stream", return_value=transactions(4)
            ),
            patch.object(
                parser_cascade, "_try_pdftotext", return_value=transactions(4)
            ),
        ):
            _, result, engines = parser_cascade._parse_pdf_worker(Path("canary.pdf"))

        self.assertEqual(len(result), 4)
        self.assertIn("won:pdftotext", engines)
        self.assertTrue(any(item.startswith("row_disagreement:") for item in engines))

    def test_cascade_reconciles_complementary_rows_only_after_complete_ocr(self):
        from analyzer import parser_cascade

        def tx(asset):
            return {
                "transaction_date": "2024-01-01",
                "transaction_type": "Purchase",
                "amount_midpoint": 8000,
                "asset_description": asset,
            }

        with (
            patch.object(parser_cascade, "_try_pdfplumber", return_value=[tx("A")]),
            patch.object(parser_cascade, "_try_camelot_lattice", return_value=[]),
            patch.object(parser_cascade, "_try_camelot_stream", return_value=[tx("B")]),
            patch.object(parser_cascade, "_try_pdftotext", return_value=[]),
            patch.object(parser_cascade, "_try_docling", return_value=[]),
            patch.object(
                parser_cascade, "_try_tesseract", return_value=[tx("A"), tx("B")]
            ),
        ):
            _, rows, engines = parser_cascade._parse_pdf_worker(Path("complement.pdf"))
        self.assertEqual({row["asset_description"] for row in rows}, {"A", "B"})
        self.assertIn("won:reconciled_complete_ocr", engines)

    def test_disjoint_or_single_text_engine_is_unresolved_without_complete_ocr(self):
        from analyzer import parser_cascade

        def tx(asset):
            return {
                "transaction_date": "2024-01-01",
                "transaction_type": "Purchase",
                "amount_midpoint": 8000,
                "asset_description": asset,
            }

        scenarios = [
            ([tx("A")], [tx("B")]),
            ([], [tx("Only")]),
        ]
        for pdfplumber_rows, pdftotext_rows in scenarios:
            with (
                patch.object(
                    parser_cascade, "_try_pdfplumber", return_value=pdfplumber_rows
                ),
                patch.object(parser_cascade, "_try_camelot_lattice", return_value=[]),
                patch.object(parser_cascade, "_try_camelot_stream", return_value=[]),
                patch.object(
                    parser_cascade, "_try_pdftotext", return_value=pdftotext_rows
                ),
                patch.object(parser_cascade, "_try_docling", return_value=[]),
                patch.object(
                    parser_cascade,
                    "_try_tesseract",
                    side_effect=parser_cascade.ParserBackendError(
                        "ocr", RuntimeError("incomplete")
                    ),
                ),
            ):
                with self.assertRaises(parser_cascade.ParserCascadeError):
                    parser_cascade._parse_pdf_worker(Path("uncertain.pdf"))

    def test_seventeen_plus_three_complements_require_complete_ocr(self):
        from analyzer import parser_cascade

        def tx(index):
            return {
                "ticker": f"T{index}",
                "transaction_date": "2024-01-01",
                "transaction_type": "Purchase",
                "amount_midpoint": 8000,
                "asset_description": f"Asset {index}",
            }

        shared = [tx(index) for index in range(17)]
        first = shared + [tx(index) for index in range(17, 20)]
        second = shared + [tx(index) for index in range(20, 23)]
        corroborated = [tx(index) for index in range(23)]
        with (
            patch.object(parser_cascade, "_try_pdfplumber", return_value=first),
            patch.object(parser_cascade, "_try_camelot_lattice", return_value=[]),
            patch.object(parser_cascade, "_try_camelot_stream", return_value=[]),
            patch.object(parser_cascade, "_try_pdftotext", return_value=second),
            patch.object(parser_cascade, "_try_docling", return_value=[]),
            patch.object(parser_cascade, "_try_tesseract", return_value=corroborated),
        ):
            _, rows, engines = parser_cascade._parse_pdf_worker(Path("17-plus-3.pdf"))
        self.assertEqual(len(rows), 23)
        self.assertIn("won:reconciled_complete_ocr", engines)

    def test_reconciliation_preserves_maximum_source_lot_multiplicity(self):
        from analyzer import parser_cascade

        row = {
            "transaction_date": "2024-01-01",
            "transaction_type": "Purchase",
            "amount_midpoint": 8000,
            "asset_description": "A",
        }
        with (
            patch.object(parser_cascade, "_try_pdfplumber", return_value=[row] * 5),
            patch.object(parser_cascade, "_try_camelot_lattice", return_value=[]),
            patch.object(parser_cascade, "_try_camelot_stream", return_value=[]),
            patch.object(parser_cascade, "_try_pdftotext", return_value=[row]),
            patch.object(parser_cascade, "_try_docling", return_value=[]),
            patch.object(parser_cascade, "_try_tesseract", return_value=[row]),
        ):
            _, rows, _ = parser_cascade._parse_pdf_worker(Path("duplicates.pdf"))
        self.assertEqual(rows, [row] * 5)

    def test_known_real_pdf_hash_and_row_count_canaries(self):
        from analyzer.parser_cascade import _parse_pdf_worker

        data_dir_value = os.environ.get("PTR_OCR_CANARY_DATA")
        if not data_dir_value:
            self.skipTest("set PTR_OCR_CANARY_DATA to run local corpus canaries")
        data_dir = Path(data_dir_value)
        expected = {
            "20030977": (
                "76053146c191866009c30ba05b192e472aac616195137db9f5ea0e87274da39a",
                224,
            ),
            "20033737": (
                "0b717e5a003cba305e42bcafc6e37042e45fdba7b8c4b9c6ca3528237eeef6b9",
                16,
            ),
            "20033921": (
                "b486c612866c86738cc2810f34aaa1613c20e537daac4d5a467ee02da889f96d",
                15,
            ),
        }
        for doc_id, (expected_hash, expected_count) in expected.items():
            matches = list(data_dir.glob(f"*/pdfs/{doc_id}.pdf"))
            self.assertEqual(len(matches), 1, doc_id)
            pdf = matches[0]
            self.assertEqual(
                hashlib.sha256(pdf.read_bytes()).hexdigest(), expected_hash
            )
            with patch.dict(os.environ, {"PTR_SKIP_DOCLING": "1"}):
                _, rows, engines = _parse_pdf_worker(pdf)
            self.assertEqual(len(rows), expected_count, (doc_id, engines))
            self.assertIn("won:pdftotext", engines)

        scan_hashes = {
            "8221322": "26f1ce2fb7823d2e84ea4fbde24514c5c6371b43a828720d50f21b1c8c7ad314",
            "9115808": "05b2fa3becd71c9bb141690130708079407e52a6e169cdacf42a467e09e0bda5",
            "9115813": "737955c7c26c497eda37f4378e1af51409b6231204a82d7ae2c3f25c10e0ae84",
            "9116141": "716cdcc10bd57c400f10d8bb4133eb667931a9699fb1835ed3b7deca010a36a1",
        }
        for doc_id, expected_hash in scan_hashes.items():
            matches = list(data_dir.glob(f"*/pdfs/{doc_id}.pdf"))
            self.assertEqual(len(matches), 1, doc_id)
            self.assertEqual(
                hashlib.sha256(matches[0].read_bytes()).hexdigest(), expected_hash
            )

        from analyzer.parsing.ocr_parser import OcrBackendError, extract_tables_with_ocr

        scan_truth = {
            "9115808": (1, "spdr", "03/31/26"),
            "9115813": (9, "richmond", "04/15/26"),
            "9116141": (134, "whittier", "05/11/26"),
        }
        for doc_id, (
            expected_count,
            asset_fragment,
            expected_date,
        ) in scan_truth.items():
            pdf = next(data_dir.glob(f"*/pdfs/{doc_id}.pdf"))
            try:
                tables = extract_tables_with_ocr(pdf)
            except OcrBackendError:
                continue
            rows = tables[0][1:]
            self.assertEqual(len(rows), expected_count, doc_id)
            self.assertTrue(
                any(
                    asset_fragment in row[0].casefold() and row[2] == expected_date
                    for row in rows
                ),
                doc_id,
            )

        from scripts.ocr_zero_rows import get_ocr_work_items

        database_path = data_dir / "congress.duckdb"
        unresolved = get_ocr_work_items(
            db_path=str(database_path),
            data_dir=data_dir,
            year=2026,
            require_schema=False,
        )
        self.assertIn("8221322", {doc_id for doc_id, _, _ in unresolved})
