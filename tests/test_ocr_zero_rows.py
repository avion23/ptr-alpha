"""Smoke tests for scripts.ocr_zero_rows module."""
import unittest


class TestOcrZeroRows(unittest.TestCase):

    def test_module_imports(self):
        from scripts import ocr_zero_rows
        assert callable(ocr_zero_rows.get_zero_row_pdfs)
        assert callable(ocr_zero_rows.load_progress)
        assert callable(ocr_zero_rows.save_progress)
        assert callable(ocr_zero_rows.call_gemini)
        assert callable(ocr_zero_rows.parse_output)
        assert callable(ocr_zero_rows.normalize_date)
        assert callable(ocr_zero_rows.insert_transactions)
        assert callable(ocr_zero_rows.main)
        assert callable(ocr_zero_rows.run_gemini_ocr_for_year)

    def test_amount_midpoints_defined(self):
        from scripts import ocr_zero_rows
        self.assertEqual(len(ocr_zero_rows.AMOUNT_MIDPOINTS), 10)
        self.assertIn("A", ocr_zero_rows.AMOUNT_MIDPOINTS)
        self.assertIn("J", ocr_zero_rows.AMOUNT_MIDPOINTS)

    def test_normalize_date_us_format(self):
        from scripts import ocr_zero_rows
        # MM/DD/YY -> YYYY-MM-DD (2-digit year conversion)
        self.assertEqual(ocr_zero_rows.normalize_date("01/15/24"), "2024-01-15")
        self.assertEqual(ocr_zero_rows.normalize_date("12/31/99"), "1999-12-31")

    def test_normalize_date_invalid_returns_none(self):
        from scripts import ocr_zero_rows
        self.assertIsNone(ocr_zero_rows.normalize_date(""))
        self.assertIsNone(ocr_zero_rows.normalize_date("not a date"))


if __name__ == "__main__":
    unittest.main()