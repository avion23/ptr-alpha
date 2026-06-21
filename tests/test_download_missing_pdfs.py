"""Smoke tests for scripts.download_missing_pdfs module."""
import unittest
from unittest.mock import MagicMock

import pandas as pd



class TestDownloadMissingPdfs(unittest.TestCase):

    def test_module_imports(self):
        from scripts import download_missing_pdfs
        assert callable(download_missing_pdfs.fetch_metadata_for_year)
        assert callable(download_missing_pdfs.download_pdf)
        assert callable(download_missing_pdfs.download_missing_pdfs)
        assert callable(download_missing_pdfs.main)

    def test_download_pdf_signature(self):
        # Just verify the function is callable with expected signature
        from scripts import download_missing_pdfs
        import inspect
        sig = inspect.signature(download_missing_pdfs.download_pdf)
        params = list(sig.parameters.keys())
        self.assertIn("session", params)
        self.assertIn("doc_id", params)
        self.assertIn("pdf_path", params)
        self.assertIn("url", params)


class TestFetchMetadataForYear(unittest.TestCase):

    def test_uses_cached_metadata_when_present(self):
        from scripts import download_missing_pdfs

        mock_db = MagicMock()
        mock_db.metadata_exists.return_value = True
        cached_df = pd.DataFrame({"DocID": ["d1"], "FilingType": ["P"]})
        mock_db.get_metadata.return_value = cached_df

        result = download_missing_pdfs.fetch_metadata_for_year(
            MagicMock(), mock_db, 2024, MagicMock(),
        )

        self.assertEqual(len(result), 1)
        mock_db.metadata_exists.assert_called_once_with(2024)
        mock_db.get_metadata.assert_called_once_with(2024)


if __name__ == "__main__":
    unittest.main()