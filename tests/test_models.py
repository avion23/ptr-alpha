"""Smoke tests for analyzer.models module."""
import unittest

from analyzer.models import (
    DownloadResult,
    DownloadStatus,
    FilingType,
    TransactionType,
)


class TestTransactionType(unittest.TestCase):

    def test_values(self):
        self.assertEqual(TransactionType.PURCHASE.value, "Purchase")
        self.assertEqual(TransactionType.SALE.value, "Sale")

    def test_str_enum_is_str(self):
        # StrEnum members are also strings.
        self.assertEqual(TransactionType.PURCHASE, "Purchase")


class TestFilingType(unittest.TestCase):

    def test_values(self):
        self.assertEqual(FilingType.PTR.value, "P")
        self.assertEqual(FilingType.AMENDMENT.value, "A")


class TestDownloadStatus(unittest.TestCase):

    def test_values(self):
        self.assertEqual(DownloadStatus.SUCCESS.value, "success")
        self.assertEqual(DownloadStatus.SKIPPED.value, "skipped")
        self.assertEqual(DownloadStatus.FAILED.value, "failed")
        self.assertEqual(DownloadStatus.ERROR.value, "error")


class TestDownloadResult(unittest.TestCase):

    def test_defaults(self):
        r = DownloadResult(doc_id="doc-1", status=DownloadStatus.SUCCESS)
        self.assertEqual(r.doc_id, "doc-1")
        self.assertEqual(r.status, DownloadStatus.SUCCESS)
        self.assertEqual(r.error_message, "")
        self.assertEqual(r.status_code, 0)

    def test_overrides(self):
        r = DownloadResult(
            doc_id="doc-2",
            status=DownloadStatus.FAILED,
            error_message="network error",
            status_code=503,
        )
        self.assertEqual(r.status, DownloadStatus.FAILED)
        self.assertEqual(r.error_message, "network error")
        self.assertEqual(r.status_code, 503)


if __name__ == "__main__":
    unittest.main()