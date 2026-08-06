"""Tests for scripts.ocr_zero_rows module."""

import unittest


class TestOcrZeroRows(unittest.TestCase):
    # --- normalize_date ---

    def test_normalize_date_us_format(self):
        from scripts import ocr_zero_rows

        self.assertEqual(ocr_zero_rows.normalize_date("01/15/24"), "2024-01-15")
        self.assertEqual(ocr_zero_rows.normalize_date("12/31/99"), "1999-12-31")

    def test_normalize_date_full_year(self):
        from scripts import ocr_zero_rows

        self.assertEqual(ocr_zero_rows.normalize_date("06/15/2024"), "2024-06-15")

    def test_normalize_date_invalid_returns_none(self):
        from scripts import ocr_zero_rows

        self.assertIsNone(ocr_zero_rows.normalize_date(""))
        self.assertIsNone(ocr_zero_rows.normalize_date("not a date"))
        self.assertIsNone(ocr_zero_rows.normalize_date("2024-01-15"))  # wrong format

    def test_normalize_date_invalid_month_returns_none(self):
        from scripts import ocr_zero_rows

        self.assertIsNone(ocr_zero_rows.normalize_date("13/15/24"))
        self.assertIsNone(ocr_zero_rows.normalize_date("00/15/24"))

    def test_normalize_date_invalid_day_returns_none(self):
        from scripts import ocr_zero_rows

        self.assertIsNone(ocr_zero_rows.normalize_date("01/32/24"))
        self.assertIsNone(ocr_zero_rows.normalize_date("01/00/24"))
        self.assertIsNone(ocr_zero_rows.normalize_date("02/31/24"))

    # --- parse_output ---

    def test_parse_output_basic(self):
        from scripts import ocr_zero_rows

        output = (
            "MEMBER: Jane Doe\n"
            "Apple Inc. (AAPL) | Purchase | 01/15/24 | 01/20/24 | A\n"
            "Microsoft Corp. (MSFT) | Sale | 02/10/24 | 02/15/24 | B\n"
        )
        member, txs = ocr_zero_rows.parse_output(output)
        self.assertEqual(member, "Jane Doe")
        self.assertEqual(len(txs), 2)
        self.assertEqual(txs[0]["asset"], "Apple Inc. (AAPL)")
        self.assertEqual(txs[0]["type"], "Purchase")
        self.assertEqual(txs[1]["type"], "Sale")

    def test_parse_output_partial_sale_normalized_to_sale(self):
        from scripts import ocr_zero_rows

        output = "MEMBER: Jane Doe\nApple Inc. (AAPL) | Partial Sale | 01/15/24 | 01/20/24 | A\n"
        _, txs = ocr_zero_rows.parse_output(output)
        self.assertEqual(len(txs), 1)
        self.assertEqual(txs[0]["type"], "Sale")

    def test_parse_output_skips_invalid_date(self):
        from scripts import ocr_zero_rows

        output = (
            "MEMBER: Jane Doe\n"
            "Apple Inc. (AAPL) | Purchase | 01/15/24 | 01/20/24 | A\n"
            "Bad Corp. (BAD) | Purchase | not-a-date | 01/20/24 | A\n"
        )
        _, txs = ocr_zero_rows.parse_output(output)
        self.assertEqual(len(txs), 1)

    def test_parse_output_fuzzy_match_transaction_types(self):
        from scripts import ocr_zero_rows

        output = (
            "MEMBER: Jane Doe\n"
            "Apple Inc. (AAPL) | purchase | 01/15/24 | 01/20/24 | A\n"
            "Microsoft Corp. (MSFT) | p | 02/10/24 | 02/15/24 | B\n"
            "Google (GOOGL) | sale | 03/01/24 | 03/05/24 | C\n"
        )
        _, txs = ocr_zero_rows.parse_output(output)
        self.assertEqual(len(txs), 3)
        self.assertEqual(txs[0]["type"], "Purchase")
        self.assertEqual(txs[1]["type"], "Purchase")
        self.assertEqual(txs[2]["type"], "Sale")

    def test_parse_output_skips_table_headers(self):
        from scripts import ocr_zero_rows

        output = (
            "MEMBER: Jane Doe\n"
            "ASSET | TYPE | DATE | DISC | AMOUNT\n"
            "--- | --- | --- | --- | ---\n"
            "Apple Inc. (AAPL) | Purchase | 01/15/24 | 01/20/24 | A\n"
        )
        _, txs = ocr_zero_rows.parse_output(output)
        self.assertEqual(len(txs), 1)

    def test_parse_output_amount_midpoint_mapped(self):
        from scripts import ocr_zero_rows

        output = (
            "MEMBER: Jane Doe\nApple Inc. (AAPL) | Purchase | 01/15/24 | 01/20/24 | D\n"
        )
        _, txs = ocr_zero_rows.parse_output(output)
        self.assertEqual(len(txs), 1)
        self.assertEqual(txs[0]["amount_letter"], "D")
        self.assertEqual(txs[0]["amount_midpoint"], 175000)

    # --- extract_ticker ---

    def test_extract_ticker_parenthesized(self):
        from scripts import ocr_zero_rows

        self.assertEqual(ocr_zero_rows.extract_ticker("Apple Inc. (AAPL)"), "AAPL")
        self.assertEqual(ocr_zero_rows.extract_ticker("BRK.B"), None)  # no parens
        self.assertEqual(ocr_zero_rows.extract_ticker("Class B (BRK.B)"), "BRK.B")

    def test_extract_ticker_no_match(self):
        from scripts import ocr_zero_rows

        self.assertIsNone(ocr_zero_rows.extract_ticker("Cash"))
        self.assertIsNone(ocr_zero_rows.extract_ticker(""))
        self.assertIsNone(ocr_zero_rows.extract_ticker(None))

    # --- resolve_ticker ---

    def test_resolve_ticker_exact_match(self):
        from scripts import ocr_zero_rows

        self.assertEqual(ocr_zero_rows.resolve_ticker("Apple"), "AAPL")
        self.assertEqual(ocr_zero_rows.resolve_ticker("Microsoft Corporation"), "MSFT")

    def test_resolve_ticker_longest_match_preferred(self):
        from scripts import ocr_zero_rows

        # "berkshire hathaway" (length 19) should match before "berkshire" (length 10)
        self.assertEqual(ocr_zero_rows.resolve_ticker("Berkshire Hathaway Inc"), "BRK")

    def test_resolve_ticker_no_match(self):
        from scripts import ocr_zero_rows

        self.assertIsNone(ocr_zero_rows.resolve_ticker("Nonexistent Corp"))
        self.assertIsNone(ocr_zero_rows.resolve_ticker(""))
        self.assertIsNone(ocr_zero_rows.resolve_ticker(None))


if __name__ == "__main__":
    unittest.main()
