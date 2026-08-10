"""Tests for the local Tesseract scan sweep (scripts/scan_ocr.py)."""

import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from scripts import scan_ocr as so


class TestDateNormalization(unittest.TestCase):
    def test_canary_dates_normalize_to_mm_dd_yy(self):
        self.assertEqual(so.normalize_date("3-31-26"), "03/31/26")
        self.assertEqual(so.normalize_date("4/15/2026"), "04/15/26")
        self.assertEqual(so.normalize_date("05/11/2026"), "05/11/26")
        self.assertEqual(so.normalize_date("02/05/24"), "02/05/24")

    def test_invalid_dates_rejected(self):
        self.assertIsNone(so.normalize_date("8/5/14"))  # year out of 24-26
        self.assertIsNone(so.normalize_date("13/40/2026"))
        self.assertIsNone(so.normalize_date("nonsense"))
        self.assertIsNone(so.normalize_date("2026"))

    def test_iso_date(self):
        self.assertEqual(so.to_iso_date("3-31-26"), "2026-03-31")
        self.assertIsNone(so.to_iso_date("nope"))


class TestRowParsing(unittest.TestCase):
    def test_strict_dates_found(self):
        line = "SP WHITTIER CALIF UN HIGH SCH DIST GO x 05/11/2026 6/5/2026 X"
        self.assertEqual(so._strict_dates(line), ["05/11/2026", "6/5/2026"])

    def test_residue_dates_found(self):
        self.assertTrue(so._has_residue_date("Illinois St Toll Hwy AU ... Jfae2026"))
        self.assertTrue(so._has_residue_date("RY BEY Municipal Bond asi2026_|a/2a/2026"))
        self.assertFalse(so._has_residue_date("2026 HAY -8 PHI: 24"))
        self.assertFalse(so._has_residue_date("Page 1 of 4"))

    def test_example_rows_skipped(self):
        lines = [
            "JT Example: Mega Corp. Common Stock Xx 0205/24 03/07/24 X",
            "SP WHITTIER CALIF UN HIGH SCH DIST GO x 05/11/2026 6/5/2026 X",
        ]
        rows = [
            so._parse_row_line(l, page_number=1, row_index=i + 1)
            for i, l in enumerate(lines)
            if not so._EXAMPLE_RE.search(l)
        ]
        self.assertEqual(len(rows), 1)

    def test_classify_type(self):
        self.assertEqual(so._classify_type("SP WHITTIER ..."), "Sale")
        self.assertEqual(so._classify_type("sP CARLISLE ..."), "Sale")
        self.assertEqual(so._classify_type("BP FOO"), "Purchase")
        self.assertEqual(so._classify_type("Buy St Str 3-31-26"), "Purchase")
        self.assertIsNone(so._classify_type("WHITTIER CALIF 05/11/2026"))

    def test_extract_asset_strips_type_words_and_marks(self):
        self.assertEqual(
            so._extract_asset("Buy St Str IL IIL LJ] 3-31-26", "3-31-26"),
            "St Str IL IIL LJ",
        )
        self.assertEqual(
            so._extract_asset("SP WHITTIER CALIF UN HIGH SCH DIST GO x 05/11/2026", "05/11/2026"),
            "WHITTIER CALIF UN HIGH SCH DIST GO",
        )

    def test_row_date_unresolved_flag(self):
        row = so._parse_row_line(
            "Illinois St Toll Hwy AU ... Jfae2026", page_number=1, row_index=1
        )
        self.assertTrue(row.date_unresolved)
        self.assertIsNone(row.transaction_date())
        row2 = so._parse_row_line(
            "SP WHITTIER ... 05/11/2026 6/5/2026", page_number=1, row_index=1
        )
        self.assertFalse(row2.date_unresolved)
        self.assertEqual(row2.transaction_date(), "05/11/26")


class TestCanary(unittest.TestCase):
    def _result(self, doc_id, rows, resolved=True):
        return {
            "doc_id": doc_id,
            "rows": rows,
            "resolved": resolved,
        }

    def test_canary_pass(self):
        rows = [
            so.OcrRow(
                page_number=1,
                row_index=1,
                asset_description="SPDR ETF",
                transaction_type="Purchase",
                transaction_date_raw="3-31-26",
                notification_date_raw=None,
            )
        ]
        check = so.canary_result("9115808", self._result("9115808", rows))
        self.assertTrue(check["passed"])
        self.assertEqual(check["actual"]["row_count"], 1)

    def test_canary_count_mismatch_fails_closed(self):
        rows = [
            so.OcrRow(page_number=1, row_index=i, asset_description="X", transaction_type=None, transaction_date_raw=None, notification_date_raw=None)
            for i in range(1, 3)
        ]
        check = so.canary_result("9115808", self._result("9115808", rows))
        self.assertFalse(check["passed"])

    def test_canary_asset_miss_fails(self):
        rows = [
            so.OcrRow(
                page_number=1,
                row_index=1,
                asset_description="VANGUARD",
                transaction_type="Sale",
                transaction_date_raw="3-31-26",
                notification_date_raw=None,
            )
        ]
        check = so.canary_result("9115808", self._result("9115808", rows))
        self.assertFalse(check["passed"])

    def test_must_be_unresolved(self):
        result = self._result("8221322", [], resolved=True)
        check = so.canary_result("8221322", result)
        self.assertFalse(check["passed"])
        self.assertEqual(check["expected"], "unresolved")

    def test_apply_canaries_demotes(self):
        rows = [
            so.OcrRow(page_number=1, row_index=1, asset_description="X", transaction_type=None, transaction_date_raw=None, notification_date_raw=None)
        ]
        result = self._result("9115808", rows, resolved=True)
        result["reasons"] = []
        results = so.apply_canaries({"9115808": result})
        self.assertFalse(results["9115808"]["resolved"])
        self.assertIn("canary failed", results["9115808"]["reasons"][0])


class TestMergeVariants(unittest.TestCase):
    def test_primary_rows_win_when_usable(self):
        primary = so.PageResult(
            page_number=1,
            rows=[
                so.OcrRow(
                    page_number=1, row_index=1, asset_description="A", transaction_type=None, transaction_date_raw="05/11/2026", notification_date_raw=None
                )
            ],
            text="periodic transaction report",
        )
        secondary = so.PageResult(
            page_number=1,
            rows=[
                so.OcrRow(
                    page_number=1, row_index=1, asset_description="GARBAGE", transaction_type=None, transaction_date_raw="05/11/2026", notification_date_raw=None
                )
            ],
            text="garbage",
        )
        merged = so._merge_page_results(primary, secondary)
        self.assertEqual(len(merged.rows), 1)
        self.assertEqual(merged.rows[0].asset_description, "A")
        self.assertEqual(merged.text, "periodic transaction report")

    def test_secondary_fills_when_primary_unusable(self):
        primary = so.PageResult(
            page_number=1,
            rows=[so.OcrRow(page_number=1, row_index=1, asset_description="X", transaction_type=None, transaction_date_raw="3-3-", notification_date_raw=None, date_unresolved=True)],
            text="",
        )
        secondary = so.PageResult(
            page_number=1,
            rows=[
                so.OcrRow(
                    page_number=1, row_index=1, asset_description="SPDR ETF", transaction_type=None, transaction_date_raw="3-31-26", notification_date_raw=None
                )
            ],
            text="periodic transaction report",
        )
        merged = so._merge_page_results(primary, secondary)
        # Primary rows are kept; strict-dated secondary rows are appended.
        self.assertEqual(len(merged.rows), 2)
        self.assertTrue(
            any(r.transaction_date() == "03/31/26" for r in merged.rows)
        )


class TestEmptyPageClassification(unittest.TestCase):
    def _classify(self, lines, text=None):
        uncovered, no_tx, covers, notes = [], [], [], []
        so._classify_empty_page(
            1,
            text or " ".join(lines).casefold(),
            lines,
            uncovered=uncovered,
            no_tx_pages=no_tx,
            cover_pages=covers,
            notes=notes,
            note="no transaction rows",
        )
        return uncovered, no_tx, covers, notes

    def test_nothing_to_report_page(self):
        uncovered, _no_tx, _covers, _notes = self._classify(
            ["Nothing to report for January 2026"]
        )
        self.assertEqual(_no_tx, [1])
        self.assertEqual(uncovered, [])

    def test_cover_page(self):
        uncovered, _no_tx, _covers, _notes = self._classify(
            [
                "UNITED STATES HOUSE OF REPRESENTATIVES",
                "Periodic Transaction Report",
                "NAME: Rohit Khanna",
                "OFFICE TELEPHONE: 202-225-2631",
            ]
        )
        self.assertEqual(_covers, [1])
        self.assertEqual(uncovered, [])

    def test_row_like_content_is_not_a_cover(self):
        uncovered, _no_tx, _covers, _notes = self._classify(
            [
                "UNITED STATES HOUSE OF REPRESENTATIVES",
                "Periodic Transaction Report",
                "NAME: X",
                "OFFICE TELEPHONE: 202-225-0000",
                "SP] NVIDIA Cov o)jo/25- \\oifio/2e",
                "SB Merck tlo, Due",
            ]
        )
        self.assertEqual(uncovered, [1])
        self.assertEqual(_covers, [])

    def test_unrecognized_page_fails_closed(self):
        uncovered, _no_tx, _covers, _notes = self._classify(["some garbage text"])
        self.assertEqual(uncovered, [1])


class TestStaging(unittest.TestCase):
    def test_row_to_dict_contract(self):
        row = so.OcrRow(
            page_number=2,
            row_index=3,
            asset_description="WHITTIER CALIF UN HIGH SCH DIST GO",
            transaction_type="Sale",
            transaction_date_raw="05/11/2026",
            notification_date_raw="6/5/2026",
        )
        d = so.row_to_dict(row, "9116141", "abc123")
        self.assertEqual(d["source_row_id"], "9116141:page:2:row:3")
        self.assertEqual(d["artifact_sha256"], "abc123")
        self.assertEqual(d["ingestion_generation"], so.GENERATION)
        self.assertEqual(d["transaction_date"], "2026-05-11")
        self.assertEqual(d["notification_date"], "2026-06-05")
        self.assertEqual(d["chamber"], "house")
        for field in so.ROW_FIELDS:
            self.assertIn(field, d)

    def test_write_staging_manifest_and_rows(self):
        rows = [
            so.OcrRow(
                page_number=1,
                row_index=1,
                asset_description="SPDR ETF",
                transaction_type="Purchase",
                transaction_date_raw="3-31-26",
                notification_date_raw=None,
            )
        ]
        results = {
            "9115808": {
                "doc_id": "9115808",
                "artifact_sha256": "aa",
                "page_count": 1,
                "rows": rows,
                "resolved": True,
                "status": "resolved",
                "reasons": [],
                "uncovered_pages": [],
                "canary": {"doc_id": "9115808", "expected": "x", "passed": True},
            },
            "9116197": {
                "doc_id": "9116197",
                "artifact_sha256": "bb",
                "page_count": 2,
                "rows": [],
                "resolved": False,
                "status": "unresolved",
                "reasons": ["page 1: no OCR text"],
                "uncovered_pages": [1, 2],
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            out = so.write_staging(tmp, results, data_dir="data", year=2026)
            manifest = json.loads((out / "manifest.json").read_text())
            self.assertEqual(manifest["generation"], so.GENERATION)
            self.assertIn("9115808", manifest["resolved"])
            self.assertIn("9116197", manifest["unresolved"])
            self.assertEqual(manifest["total_rows"], 1)
            rows_lines = (out / "rows" / "9115808.jsonl").read_text().strip().splitlines()
            self.assertEqual(len(rows_lines), 1)
            row = json.loads(rows_lines[0])
            self.assertEqual(row["source_row_id"], "9115808:page:1:row:1")
            self.assertTrue((out / "docs" / "9115808.json").exists())
            self.assertTrue((out / "docs" / "9116197.json").exists())


class TestDiscovery(unittest.TestCase):
    def test_is_scanned(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "a.pdf"
            p.write_bytes(b"%PDF-1.4\nfake")
            # No text layer extractable -> treated as scanned.
            self.assertTrue(so.is_scanned(p))


@pytest.mark.skipif(
    not os.environ.get("PTR_OCR_CANARY_DATA"), reason="set PTR_OCR_CANARY_DATA to run real scan canaries"
)
class TestRealScanCanaries(unittest.TestCase):
    """Ground-truth canaries over the pinned 2026 House scans."""

    DATA_DIR = os.environ.get("PTR_OCR_CANARY_DATA", "")

    def test_pinned_scan_hashes(self):
        expected = {
            "8221322": "26f1ce2fb7823d2e84ea4fbde24514c5c6371b43a828720d50f21b1c8c7ad314",
            "9115808": "05b2fa3becd71c9bb141690130708079407e52a6e169cdacf42a467e09e0bda5",
            "9115813": "737955c7c26c497eda37f4378e1af51409b6231204a82d7ae2c3f25c10e0ae84",
            "9116141": "716cdcc10bd57c400f10d8bb4133eb667931a9699fb1835ed3b7deca010a36a1",
        }
        for doc_id, digest in expected.items():
            matches = list(Path(self.DATA_DIR).glob(f"*/pdfs/{doc_id}.pdf"))
            self.assertEqual(len(matches), 1, doc_id)
            self.assertEqual(hashlib.sha256(matches[0].read_bytes()).hexdigest(), digest)

    def test_canary_docs_sweep_meets_ground_truth(self):
        results = so.run_sweep(self.DATA_DIR, 2026, workers=4, doc_ids=list(so.CANARY_TRUTH))
        for doc_id, (count, fragment, date) in so.CANARY_TRUTH.items():
            check = results[doc_id]["canary"]
            self.assertTrue(check["passed"], f"{doc_id}: {check['detail']}")
            self.assertEqual(len(results[doc_id]["rows"]), count, doc_id)
            self.assertTrue(results[doc_id]["resolved"], doc_id)

    def test_8221322_fails_closed(self):
        results = so.run_sweep(self.DATA_DIR, 2026, workers=4, doc_ids=["8221322"])
        self.assertFalse(results["8221322"]["resolved"])
        check = results["8221322"]["canary"]
        self.assertEqual(check["expected"], "unresolved")


if __name__ == "__main__":
    unittest.main()
