"""Tests for the local OCR second pass (scripts/ocr_local_sweep.py).

Covers: date normalization, docling remarks-residue stripping, amount-letter
mapping, Gemini-contract row building, fail-closed validation, empty-page
classification (incl. the 8221322 cover-page checkbox case), pinned canary
enforcement, and the staging manifest contract.

Real canaries (PTR_OCR_CANARY_DATA) run the full pipeline over the pinned
2026 House scans and are skipped unless the env var is set, mirroring
tests/test_scan_ocr.py.
"""

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

from scripts import ocr_local_sweep as ocr  # noqa: E402


class TestNormalizeIsoDate(unittest.TestCase):
    def test_slash_and_dash_separators(self):
        self.assertEqual(ocr._normalize_iso_date("3-31-26"), "2026-03-31")
        self.assertEqual(ocr._normalize_iso_date("3/31/26"), "2026-03-31")
        self.assertEqual(ocr._normalize_iso_date("05/11/2026"), "2026-05-11")
        self.assertEqual(ocr._normalize_iso_date("02/05/24"), "2024-02-05")
        self.assertEqual(ocr._normalize_iso_date("2026-05-11"), "2026-05-11")
        self.assertEqual(ocr._normalize_iso_date("10/29/2014"), "2014-10-29")

    def test_invalid_rejected(self):
        self.assertIsNone(ocr._normalize_iso_date("13/40/2026"))
        self.assertIsNone(ocr._normalize_iso_date("nonsense"))
        self.assertIsNone(ocr._normalize_iso_date("2026"))
        self.assertIsNone(ocr._normalize_iso_date(""))
        self.assertIsNone(ocr._normalize_iso_date(None))


class TestFilingResidue(unittest.TestCase):
    def test_strips_residue_from_asset(self):
        self.assertEqual(
            ocr._strip_filing_residue(
                "apple Inc. (aaPl) F IlINg S TaTuS : New D ESCRIPTIoN : "
                "Dependent child a"
            ),
            "apple Inc. (aaPl)",
        )
        self.assertEqual(
            ocr._strip_filing_residue(
                "Stonegate Bank (SgBK) F IlINg S TaTuS : New"
            ),
            "Stonegate Bank (SgBK)",
        )

    def test_clean_asset_untouched(self):
        self.assertEqual(
            ocr._strip_filing_residue("Community Bank of Broward Common Stock"),
            "Community Bank of Broward Common Stock",
        )

    def test_pure_residue_empties(self):
        self.assertEqual(ocr._strip_filing_residue("F IlINg S TaTuS : New"), "")


class TestAmountLetter(unittest.TestCase):
    def test_range_to_letter(self):
        self.assertEqual(ocr._amount_letter("$1,001 - $15,000", 8000.5), ("A", 8000))
        self.assertEqual(ocr._amount_letter("$15,000 - $50,000", 32500), ("B", 32500))
        self.assertEqual(ocr._amount_letter("$100,001 - $250,000", 175000), ("D", 175000))

    def test_letter_passthrough(self):
        self.assertEqual(ocr._amount_letter("A", None), ("A", 8000))

    def test_missing_amount(self):
        self.assertEqual(ocr._amount_letter(None, None), (None, None))


class TestBuildRow(unittest.TestCase):
    def test_contract_fields(self):
        tx = {
            "asset_description": "Apple Inc. (AAPL)",
            "transaction_type": "Purchase",
            "transaction_date": "10/29/2014",
            "amount_raw": "$1,001 - $15,000",
            "amount_midpoint": 8000.5,
            "owner_code": "DC",
            "page_number": 1,
        }
        row = ocr.build_row(
            "20000883", 2015, tx,
            member="Debbie Wasserman Schultz",
            artifact_sha256="abc123",
            row_index=7,
        )
        self.assertEqual(row["source_row_id"], "20000883:page:1:row:7")
        self.assertEqual(row["transaction_date"], "2014-10-29")
        self.assertEqual(row["amount_raw"], "A")
        self.assertEqual(row["amount_midpoint"], 8000)
        self.assertEqual(row["ticker_origin"], "official")
        self.assertEqual(row["raw_ticker"], "AAPL")
        self.assertIsNone(row["ticker_candidate"])
        self.assertEqual(row["member"], "Debbie Wasserman Schultz")
        self.assertEqual(row["ingestion_generation"], ocr.GENERATION)
        self.assertEqual(row["artifact_sha256"], "abc123")
        self.assertEqual(row["year"], 2015)
        self.assertEqual(row["chamber"], "house")

    def test_unreported_ticker_provenance(self):
        tx = {
            "asset_description": "Whittier Calif Un High Sch Dist Go",
            "transaction_type": None,
            "transaction_date": "05/11/2026",
            "page_number": 2,
        }
        row = ocr.build_row("9116141", 2026, tx, member="M", artifact_sha256="x", row_index=1)
        self.assertEqual(row["ticker_origin"], "not_reported")
        self.assertIsNone(row["raw_ticker"])
        self.assertIsNone(row["ticker_candidate"])
        self.assertIsNone(row["transaction_type"])
        self.assertIsNone(row["notification_date"])

    def test_cascade_row_without_page(self):
        tx = {"asset_description": "X Corp", "transaction_date": "01/02/2024"}
        row = ocr.build_row("123", 2024, tx, member="M", artifact_sha256="x", row_index=3)
        self.assertEqual(row["source_row_id"], "123:row:3")
        self.assertIsNone(row["page_number"])


class TestLocalValidate(unittest.TestCase):
    def test_empty_asset_dropped(self):
        rows = [
            {"asset_description": "   ", "transaction_date": "01/02/2024"},
            {"asset_description": "SPDR ETF", "transaction_date": "3-31-26"},
        ]
        valid, rejections = ocr._local_validate(rows)
        self.assertEqual(len(valid), 1)
        self.assertEqual(rejections, {"invalid_asset": 1})

    def test_non_strict_rows_kept(self):
        rows = [
            {"asset_description": "Garbled Jfae2026", "transaction_date": "azorz026"},
        ]
        valid, rejections = ocr._local_validate(rows)
        self.assertEqual(len(valid), 1)
        self.assertEqual(rejections, {})


class TestEmptyPageClassification(unittest.TestCase):
    def _classify(self, lines, text=None):
        uncovered, no_tx, covers, notes = [], [], [], []
        ocr.classify_empty_page(
            1,
            text or " ".join(lines).casefold(),
            lines,
            uncovered=uncovered,
            no_tx_pages=no_tx,
            cover_pages=covers,
            notes=notes,
        )
        return uncovered, no_tx, covers, notes

    def test_nothing_to_report_page(self):
        _u, no_tx, _c, _n = self._classify(["Nothing to report for January 2026"])
        self.assertEqual(no_tx, [1])

    def test_cover_page_with_checkbox_line(self):
        # 8221322 page 1: 'x Member of the U.S. House of Representatives'
        # must NOT count as row-like transaction content.
        lines = [
            "UNITED STATES HOUSE OF REPRESENTATIVES",
            "Periodic Transaction Report",
            "NAME: Rohit Khanna",
            "OFFICE TELEPHONE: 202-225-2631",
            "x Member of the U.S. House of Representatives",
        ]
        u, _n, covers, _notes = self._classify(lines)
        self.assertEqual(covers, [1])
        self.assertEqual(u, [])

    def test_row_like_content_is_not_a_cover(self):
        lines = [
            "UNITED STATES HOUSE OF REPRESENTATIVES",
            "Periodic Transaction Report",
            "NAME: X",
            "OFFICE TELEPHONE: 202-225-0000",
            "SP] NVIDIA Cov o)jo/25- \\oifio/2e",
        ]
        u, _n, _c, _notes = self._classify(lines)
        self.assertEqual(u, [1])

    def test_unrecognized_page_fails_closed(self):
        u, _n, _c, _notes = self._classify(["some garbage text"])
        self.assertEqual(u, [1])


class TestCanary(unittest.TestCase):
    def _result(self, doc_id, status="resolved", rows=None, pages=1, uncovered=None):
        return {
            "doc_id": doc_id,
            "status": status,
            "rows": rows or [],
            "row_count": len(rows or []),
            "page_count": pages,
            "uncovered_pages": uncovered or [],
            "reasons": [],
        }

    def test_canary_pass(self):
        rows = [{"page_number": 1}] * 9
        r = ocr.check_canary(self._result("9115813", rows=rows, pages=2))
        self.assertTrue(r["canary"]["passed"])
        self.assertEqual(r["status"], "resolved")

    def test_canary_row_count_mismatch_demotes(self):
        rows = [{"page_number": 1}] * 4
        r = ocr.check_canary(self._result("9115813", rows=rows, pages=2))
        self.assertFalse(r["canary"]["passed"])
        self.assertEqual(r["status"], "unresolved")
        self.assertTrue(any("canary failed" in reason for reason in r["reasons"]))

    def test_8221322_page2_min(self):
        rows = [{"page_number": 2}] * 19 + [{"page_number": 1}] * 5
        r = ocr.check_canary(
            self._result("8221322", rows=rows, pages=56)
        )
        self.assertTrue(r["canary"]["passed"])
        self.assertEqual(r["canary"]["actual"]["page2_rows"], 19)

    def test_8221322_page2_min_fails_closed(self):
        rows = [{"page_number": 2}] * 10 + [{"page_number": 1}] * 5
        r = ocr.check_canary(self._result("8221322", rows=rows, pages=56))
        self.assertFalse(r["canary"]["passed"])
        self.assertEqual(r["status"], "unresolved")

    def test_unpinned_doc_untouched(self):
        r = ocr.check_canary(self._result("20000883", status="unresolved"))
        self.assertIsNone(r.get("canary"))
        self.assertEqual(r["status"], "unresolved")


class TestStagingManifest(unittest.TestCase):
    def test_write_manifest_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            for doc_id, status, rows, year in [
                ("9115808", "resolved", [{"a": 1}], 2026),
                ("9106064", "unresolved", [], 2015),
                ("9110845", "no_txs", [], 2017),
            ]:
                ocr.stage_document(
                    out,
                    {
                        "doc_id": doc_id,
                        "year": year,
                        "status": status,
                        "artifact_sha256": f"sha-{doc_id}",
                        "page_count": 2,
                        "covered_pages": [1, 2],
                        "uncovered_pages": [],
                        "row_count": len(rows),
                        "rows": rows,
                        "reasons": ["some reason"] if status != "resolved" else [],
                        "engines": ["docling"],
                        "canary": None,
                        "elapsed_s": 1.0,
                    },
                )
            ocr.write_manifest(out, kind="sweep", data_dir="/tmp/data")
            manifest = json.loads((out / "manifest.json").read_text())
            self.assertEqual(manifest["generation"], ocr.GENERATION)
            self.assertEqual(manifest["doc_count"], 3)
            self.assertEqual(manifest["resolved_count"], 1)
            self.assertEqual(manifest["unresolved_count"], 1)
            self.assertEqual(manifest["no_txs_count"], 1)
            self.assertEqual(manifest["total_rows"], 1)
            self.assertEqual(manifest["by_year"]["2026"]["resolved_count"], 1)
            self.assertEqual(manifest["by_year"]["2015"]["unresolved_count"], 1)
            self.assertEqual(
                manifest["staged_files_sha256"]["rows/9115808.jsonl"],
                ocr.sha256_file(out / "rows" / "9115808.jsonl"),
            )
            # rows file hash must be reproducible
            self.assertTrue((out / "docs" / "9115808.json").exists())


class TestDoclingMarkdownParsing(unittest.TestCase):
    def test_pipe_table_parsing_with_residue(self):
        md = (
            "| ID | Owner | Asset | Type | Date | Notification | Amount |\n"
            "|---|---|---|---|---|---|---|\n"
            "| | DC | apple Inc. (aaPl) F IlINg S TaTuS : New | P | "
            "10/29/2014 | 05/19/2015 | $1,001 - $15,000 |\n"
            "| | | Community Bank of Broward | E | 02/9/2015 | | "
            "$1,001 - $15,000 |\n"
        )
        tables = ocr._parse_markdown_tables(md)
        self.assertEqual(len(tables), 1)
        rows = ocr.parse_pdf_table(tables[0])
        self.assertEqual(len(rows), 2)
        stripped = ocr._strip_filing_residue(str(rows[0]["asset_description"]))
        self.assertEqual(stripped, "apple Inc. (aaPl)")
        self.assertEqual(rows[1]["asset_description"], "Community Bank of Broward")


@pytest.mark.skipif(
    not os.environ.get("PTR_OCR_CANARY_DATA"),
    reason="set PTR_OCR_CANARY_DATA (staged gen dir) to run real canaries",
)
class TestRealCanaries(unittest.TestCase):
    """Full local pipeline over the pinned 2026 House scans."""

    DATA_DIR = os.environ.get("PTR_OCR_CANARY_DATA", "")
    DB = os.environ.get("PTR_OCR_CANARY_DB", "")
    MANIFEST = os.environ.get("PTR_OCR_CANARY_MANIFEST", "")

    def test_pinned_docs_resolve_with_canary_truth(self):
        import time  # noqa: PLC0415

        metadata = ocr.load_metadata(self.DB)
        for doc_id in ["9115808", "9115813", "9116141", "8221322"]:
            pdf = Path(self.DATA_DIR) / "2026" / "pdfs" / f"{doc_id}.pdf"
            self.assertTrue(pdf.exists(), doc_id)
            result = ocr.process_document(
                doc_id, 2026, pdf, metadata.get(doc_id, {})
            )
            result = ocr.check_canary(result)
            self.assertTrue(
                result["canary"]["passed"],
                f"{doc_id}: {result['canary']['detail']}",
            )
            self.assertEqual(result["status"], "resolved", doc_id)
            self.assertFalse(result["uncovered_pages"], doc_id)


if __name__ == "__main__":
    unittest.main()
