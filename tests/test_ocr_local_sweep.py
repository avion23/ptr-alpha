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


class TestEnvTunables(unittest.TestCase):
    def test_env_overrides_parsed(self):
        """Constants are env-driven; defaults match the canary-validated set."""
        self.assertEqual(ocr.RENDER_DPI, 300)
        self.assertEqual(ocr.PREPROCESS_SCALE, 2.0)
        self.assertEqual(ocr.PREPROCESS_MEDIAN, 3)
        self.assertTrue(ocr.DOCLING_ENABLED)
        self.assertTrue(ocr.CASCADE_ENABLED)
        self.assertEqual(ocr.TESSERACT_PSM, 3)
        self.assertEqual(ocr.DEFAULT_WORKERS, 3)




class TestNoTxsPolicy(unittest.TestCase):
    """Terminal no_txs requires nothing-to-report evidence on EVERY page;
    cover-only documents stay unresolved (fail-closed)."""

    def test_all_nothing_to_report_is_no_txs(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            ocr.stage_document(
                out,
                {
                    "doc_id": "x1", "year": 2015, "status": "no_txs",
                    "artifact_sha256": "s", "page_count": 2,
                    "covered_pages": [1, 2], "uncovered_pages": [],
                    "row_count": 0, "rows": [],
                    "reasons": ["page 1: reports no transactions",
                                "page 2: reports no transactions",
                                "filing reports no transactions (nothing-to-report pages: 2/2)"],
                    "engines": ["docling", "tesseract"], "canary": None,
                    "elapsed_s": 1.0,
                },
            )
            ocr.write_manifest(out, kind="sweep", data_dir="/tmp/d")
            mf = json.loads((out / "manifest.json").read_text())
            self.assertEqual(mf["no_txs_count"], 1)
            self.assertIn("x1", mf["no_txs"])

    def test_cover_only_doc_is_not_no_txs(self):
        # A doc whose only evidence is cover-classified pages must not be
        # terminal no_txs; simulate the envelope a cover-only doc would have
        # produced under the fixed rule: status unresolved with the reason.
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            ocr.stage_document(
                out,
                {
                    "doc_id": "x2", "year": 2015, "status": "unresolved",
                    "artifact_sha256": "s", "page_count": 1,
                    "covered_pages": [], "uncovered_pages": [1],
                    "row_count": 0, "rows": [],
                    "reasons": ["page 1: cover page (no transaction rows)",
                                "zero transactions unconfirmed: all pages cover-classified (filer block only, no readable rows)"],
                    "engines": ["docling", "tesseract"], "canary": None,
                    "elapsed_s": 1.0,
                },
            )
            ocr.write_manifest(out, kind="sweep", data_dir="/tmp/d")
            mf = json.loads((out / "manifest.json").read_text())
            self.assertEqual(mf["no_txs_count"], 0)
            self.assertEqual(mf["unresolved_count"], 1)
            self.assertIn("x2", mf["unresolved"])




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

    def test_certification_page_accepted(self):
        # Real 20016481 page 62 OCR (docling probe, casefolded): the
        # trailing certification/signature page has no transaction table
        # and no filer_block markers, so it used to fall to uncovered.
        lines = [
            "## certification and signature g f e d c b",
            "i certify that the statements i have made on the attached",
            "periodic transaction report are true, complete, and correct",
            "to the best of my knowledge and belief.",
            "digitally signed: hon. donna shalala , 04/27/2020",
        ]
        u, _n, covers, _notes = self._classify(lines)
        self.assertEqual(covers, [1])
        self.assertEqual(u, [])

    def test_cert_page_with_row_like_content_stays_uncovered(self):
        # A page carrying BOTH certification text and unparsed transaction
        # rows (the 20007778/20017648 trailing-page shape: cert block plus
        # a table that failed to map) must fail closed, never be accepted
        # as a transaction-free certification page.
        lines = [
            "## certification and signature g f e d c b",
            "i certify that the statements i have made on the attached",
            "periodic transaction report are true, complete, and correct",
            "SP] Verizon Communications Inc. (VZ) FILING STATUS: New",
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


class TestDoclingDataframeMapping(unittest.TestCase):
    """Old-form PTR column layout -> tx dicts via Docling structured tables.

    Guards the mapping that fixes the 2015-2020 scans the markdown text
    path cannot read (20002501 et al.): ticker from the Asset parenthetical,
    transaction type from the P/S/E letter (merged account residue
    tolerated), transaction/notification dates, and the amount range.
    """

    @staticmethod
    def _df(data):
        import pandas as pd  # noqa: PLC0415

        return pd.DataFrame(
            data,
            columns=[
                "iD", "owner", "asset", "transaction type",
                "Date", "notification Date", "amount",
            ],
        )

    def test_old_form_rows_mapped(self):
        df = self._df(
            [
                ["", "", "Cerner Corporation (CERN) F IlINg S TaTuS : New",
                 "P aCCoUNTS", "02/3/2015", "02/5/2015", "$1,001 - $15,000"],
                ["", "", "Fossil group, Inc. (FoSL) F IlINg S TaTuS : New",
                 "aCCoUNTS", "01/27/2015", "02/5/2015", "$1,001 - $15,000"],
            ]
        )
        rows = ocr._dataframe_rows(df, 1)
        self.assertEqual(len(rows), 2)
        first = rows[0]
        self.assertEqual(first["asset_description"], "Cerner Corporation (CERN)")
        self.assertEqual(first["transaction_type"], "Purchase")
        self.assertEqual(first["transaction_date"], "02/3/2015")
        self.assertEqual(first["notification_date"], "02/5/2015")
        self.assertEqual(first["amount_midpoint"], 8000.5)
        self.assertEqual(first["page_number"], 1)
        # OCR dropped the type letter on the second row -> NULL type, kept.
        self.assertIsNone(rows[1]["transaction_type"])
        self.assertEqual(rows[1]["asset_description"], "Fossil group, Inc. (FoSL)")

    def test_checkbox_layout_rejected(self):
        # 2026-style grid: distinct per-column type/amount headers, no
        # parenthetical tickers or dollar cells -> must map to nothing so
        # the tesseract fallback (which produces the pinned canaries)
        # keeps running.
        import pandas as pd  # noqa: PLC0415

        df = pd.DataFrame(
            [
                ["", "FULL ASSET NAME", "Type of transaction.Purchase",
                 "Type of transaction.Sale", "Date of Transaction",
                 "Date Notified of Transaction", "Amount of Transaction.B"],
                ["×", "North TX WY Auth Rev BE/R Municipal Bond",
                 "×", "", "4/9/2026", "4/29/2026", ""],
                ["×", "Kentucky ST PPTY & BLDG SVC REV BE/R Municipal",
                 "", "×", "4/9/2026", "4/29/2026", "×"],
            ]
        )
        self.assertEqual(ocr._dataframe_rows(df, 2), [])

    def test_degenerate_table_rejected(self):
        import pandas as pd  # noqa: PLC0415

        self.assertEqual(ocr._dataframe_rows(pd.DataFrame([["h"]], columns=["x"]), 1), [])
        self.assertEqual(ocr._dataframe_rows(pd.DataFrame(), 1), [])

    def test_letter_type_without_parenthetical_ticker_maps(self):
        # 20002425-style row: no parenthetical ticker but a P/S/E type
        # letter and dollar range -> old-form evidence, must map (the
        # Gemini ground truth for 20002425 is Sale/2014-12-19/G).
        df = self._df(
            [
                ["", "sP", "CaMPR Partners limited", "s", "12/19/2014",
                 "12/22/2014", "$1,000,001 - $5,000,000"],
            ]
        )
        rows = ocr._dataframe_rows(df, 1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["transaction_type"], "Sale")
        self.assertEqual(rows[0]["transaction_date"], "12/19/2014")
        self.assertEqual(rows[0]["amount_midpoint"], 3000000.5)
        self.assertEqual(rows[0]["notification_date"], "12/22/2014")

    def test_merged_type_cell_weak_evidence_maps(self):
        # 20006695-style: Docling merged the P/S/E letter mid-cell ("C (ua)
        # P") and the strict date sits in the second type/date column; the
        # row passes the relaxed bar (letter AND date/dollar) even though
        # the pre-fix gate demanded ticker-or-letter AND dollar AND strict
        # date across the table (0 rows before the fix).
        import pandas as pd  # noqa: PLC0415

        df = pd.DataFrame(
            [
                ["", "under armour, Inc. Class", "C (ua) P", "02/7/2017",
                 "02/7/2017", "$1,001 - $15,000"],
                ["", "", "F IlINg s TaTus : New", "", "", ""],
            ],
            columns=[
                "iD", "owner asset", "transaction type Date",
                "transaction type Date", "notification Date", "amount",
            ],
        )
        rows = ocr._dataframe_rows(df, 1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["asset_description"], "under armour, Inc. Class")
        self.assertEqual(rows[0]["transaction_type"], "Purchase")
        self.assertEqual(rows[0]["transaction_date"], "02/7/2017")
        self.assertEqual(rows[0]["amount_midpoint"], 8000.5)

    def test_asset_merged_into_type_cell_maps(self):
        # 20007778 p3-style: the Owner Asset cell is empty and the asset
        # text + P/S/E letter were merged into the Transaction Type cell;
        # the relaxed bar maps the row and recovers the asset from the type
        # cell (asset fallback; empty-asset rows stayed 0 before the fix).
        import pandas as pd  # noqa: PLC0415

        df = pd.DataFrame(
            [
                ["", "", "Verizon Communications Inc. (VZ) P F IlINg s TATus : New",
                 "06/12/2017", "07/12/2017", "$1,001 - $15,000"],
                ["", "", "Wal-Mart stores, Inc. (WMT) P F IlINg s TATus : New",
                 "06/12/2017", "07/12/2017", "$1,001 - $15,000"],
            ],
            columns=[
                "ID", "Owner Asset", "Transaction Type", "Date",
                "Notification Date", "Amount",
            ],
        )
        rows = ocr._dataframe_rows(df, 3)
        self.assertEqual(len(rows), 2)
        self.assertEqual(
            rows[0]["asset_description"], "Verizon Communications Inc. (VZ)"
        )
        self.assertEqual(rows[0]["transaction_type"], "Purchase")
        self.assertEqual(rows[0]["transaction_date"], "06/12/2017")
        self.assertEqual(rows[0]["amount_midpoint"], 8000.5)
        self.assertEqual(
            rows[1]["asset_description"], "Wal-Mart stores, Inc. (WMT)"
        )

    def test_merged_cell_without_date_or_dollar_still_dropped(self):
        # Residue-only type cell ("F IlINg s TaTus : New") carries a
        # standalone 's' letter but neither date nor dollar -> the relaxed
        # bar (letter AND date-or-dollar) still drops it.
        import pandas as pd  # noqa: PLC0415

        df = pd.DataFrame(
            [
                ["", "", "F IlINg s TaTus : New", "", "", ""],
            ],
            columns=[
                "ID", "Owner Asset", "Transaction Type", "Date",
                "Notification Date", "Amount",
            ],
        )
        self.assertEqual(ocr._dataframe_rows(df, 1), [])

    def test_letter_and_dollar_without_strict_date_dropped(self):
        # 2026 checkbox-grid header row (real 9116141 p2): lettered
        # PURCHASE/SALE/EXCHANGE headers and dollar-range cells but only a
        # "(MM/DD/YY)" date placeholder -> the relaxed bar still requires a
        # strict date, so the header row never maps and the page stays on
        # the tesseract fallback (9116141 canary = 134 tesseract rows).
        import pandas as pd  # noqa: PLC0415

        df = pd.DataFrame(
            [
                ["", "", "PURCHASE", "SALE", "EXCHANGE", "(MM/DD/YY)",
                 "(MM/DD/YY)", "$1,000-$15,000", "$15,001-$50,000"],
            ],
            columns=[
                "iD", "owner", "asset", "transaction type",
                "transaction type", "Date", "notification Date", "amount",
                "amount",
            ],
        )
        self.assertEqual(ocr._dataframe_rows(df, 2), [])

    def test_residue_only_row_dropped(self):
        df = self._df(
            [
                ["", "", "F IlINg S TaTuS : New D ESCRIPTIoN : note text",
                 "S", "12/19/2014", "12/22/2014", "$1,000,001 - $5,000,000"],
            ]
        )
        self.assertEqual(ocr._dataframe_rows(df, 1), [])


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


@pytest.mark.skipif(
    not os.environ.get("PTR_OCR_CANARY_DATA"),
    reason="set PTR_OCR_CANARY_DATA (staged gen dir) to run real canaries",
)
class TestCertificationPageGate(unittest.TestCase):
    """Pinned scenario: 20016481's trailing "CERTIFICATION AND SIGNATURE"
    page (62 of 62) carries no transaction rows, but classify_empty_page has
    no gate for it (its filer_block markers need "office telephone"/"member
    of the u.s. house", which cert pages lack), so the sweep kept the whole
    doc unresolved despite 582 mapped rows across pages 1-61.  The
    certification-page gate must accept the real page (uncovered stays
    empty) and the full pipeline must stage rows and resolve the doc."""

    DATA_DIR = os.environ.get("PTR_OCR_CANARY_DATA", "")
    DB = os.environ.get("PTR_OCR_CANARY_DB", "")

    def test_real_cert_page_accepts_and_resolves(self):
        pdf = Path(self.DATA_DIR) / "2020" / "pdfs" / "20016481.pdf"
        self.assertTrue(pdf.exists(), f"missing {pdf}")
        pages, err = ocr.docling_pages(pdf)
        self.assertIsNone(err)
        cert = pages[-1]
        self.assertEqual(cert["page"], 62)
        self.assertFalse(cert["rows"], "cert page must not map rows")
        uncovered, no_tx, covers, notes = [], [], [], []
        ocr.classify_empty_page(
            cert["page"],
            cert["text"],
            ocr._plain_lines_from_text(cert["text"]),
            uncovered=uncovered,
            no_tx_pages=no_tx,
            cover_pages=covers,
            notes=notes,
        )
        self.assertEqual(uncovered, [])
        self.assertEqual(covers, [62])
        # Full pipeline: the doc must stage rows and resolve (previously
        # "page 62: no transaction rows" kept it unresolved).
        metadata = ocr.load_metadata(self.DB)
        result = ocr.process_document(
            "20016481", 2020, pdf, metadata.get("20016481", {})
        )
        self.assertEqual(result["status"], "resolved", result["reasons"])
        self.assertFalse(result["uncovered_pages"])
        self.assertGreaterEqual(result["row_count"], 500)


@pytest.mark.skipif(
    not os.environ.get("PTR_OCR_CANARY_DATA"),
    reason="set PTR_OCR_CANARY_DATA (staged gen dir) to run real canaries",
)
class TestOldFormDoclingDataframe(unittest.TestCase):
    """Pinned scenario: old-form column-layout page read via Docling's
    structured dataframe.

    Real 20002501.pdf (2015 House PTR, scans/old form) has 7 transactions
    on page 1 that the markdown text path cannot read (0 rows pre-fix:
    the merged account text in the Transaction Type cell defeats
    _extract_transaction_type).  Docling's structured table maps them with
    ticker from the Asset parenthetical, type, dates and amount range; the
    page must yield >=5 rows carrying ticker + transaction date + amount.
    """

    DATA_DIR = os.environ.get("PTR_OCR_CANARY_DATA", "")

    def test_old_form_page_yields_five_plus_rows(self):
        pdf = Path(self.DATA_DIR) / "2015" / "pdfs" / "20002501.pdf"
        self.assertTrue(pdf.exists(), f"missing {pdf}")
        pages, err = ocr.docling_pages(pdf)
        self.assertIsNone(err)
        rows = [tx for page in pages for tx in page["rows"]]
        self.assertGreaterEqual(
            len(rows), 5,
            f"20002501 old-form rows from Docling dataframe: {len(rows)}",
        )
        complete = [
            tx for tx in rows
            if ocr.extract_ticker(tx["asset_description"])
            and ocr._normalize_iso_date(tx.get("transaction_date"))
            and tx.get("amount_midpoint")
        ]
        self.assertGreaterEqual(
            len(complete), 5,
            f"rows with ticker+date+amount: {len(complete)} of {len(rows)}",
        )
        # every mapped row must carry the parenthetical ticker the task's
        # mapping is built on (CERN/FOSL/JEC/PCP/SLH/TCBI)
        self.assertTrue(
            all(ocr.extract_ticker(tx["asset_description"]) for tx in rows)
        )



@pytest.mark.skipif(
    not os.environ.get("PTR_OCR_CANARY_DATA"),
    reason="set PTR_OCR_CANARY_DATA (staged gen dir) to run real canaries",
)
class TestConsumerGapDoclingDataframe(unittest.TestCase):
    """Pinned scenario: consumer-gap docs whose schema-matching tables the
    pre-fix three-part evidence gate rejected (ticker-or-letter AND dollar
    AND strict date across the table) but whose rows carry weaker per-row
    evidence.

    20006695 p1: type cell merged to "C (ua) P", date "02/7/2017" -- P/S/E
    letter + date + dollar -> its single Under Armour transaction must map.
    20007778 p3: asset text merged into the type cell
    ("Verizon Communications Inc. (VZ) P ...") with an empty Owner Asset
    cell -- letter + date + dollar -> both page-3 purchases must map.
    """

    DATA_DIR = os.environ.get("PTR_OCR_CANARY_DATA", "")

    def test_20006695_single_row_maps(self):
        pdf = Path(self.DATA_DIR) / "2017" / "pdfs" / "20006695.pdf"
        self.assertTrue(pdf.exists(), f"missing {pdf}")
        pages, err = ocr.docling_pages(pdf)
        self.assertIsNone(err)
        rows = [tx for page in pages for tx in page["rows"]]
        self.assertEqual(
            len(rows), 1,
            f"20006695 rows from Docling dataframe: {len(rows)}",
        )
        self.assertEqual(rows[0]["asset_description"], "under armour, Inc. Class")
        self.assertEqual(rows[0]["transaction_type"], "Purchase")
        self.assertEqual(rows[0]["transaction_date"], "02/7/2017")
        self.assertEqual(rows[0]["amount_midpoint"], 8000.5)

    def test_20007778_page3_rows_map(self):
        pdf = Path(self.DATA_DIR) / "2017" / "pdfs" / "20007778.pdf"
        self.assertTrue(pdf.exists(), f"missing {pdf}")
        pages, err = ocr.docling_pages(pdf)
        self.assertIsNone(err)
        by_page = {page["page"]: page["rows"] for page in pages}
        self.assertEqual(
            len(by_page[3]), 2,
            f"20007778 page 3 rows from Docling dataframe: {len(by_page[3])}",
        )
        self.assertEqual(
            by_page[3][0]["asset_description"], "Verizon Communications Inc. (VZ)"
        )
        self.assertEqual(by_page[3][1]["asset_description"], "Wal-Mart stores, Inc. (WMT)")
        self.assertTrue(
            all(
                tx["transaction_type"] == "Purchase"
                and tx["transaction_date"] == "06/12/2017"
                for tx in by_page[3]
            )
        )


class TestRunPoolStreaming(unittest.TestCase):
    """_run_pool must stream per-doc results (imap_unordered), never hold
    them until batch end like pool.map (which stalled staging on slow
    stragglers)."""

    def test_imap_unordered_streams_results(self):
        class FakePool:
            def __init__(self, workers):
                self.workers = workers

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def imap_unordered(self, fn, items):
                for item in items:
                    yield fn(item)

            def map(self, fn, items):  # pragma: no cover -- must not be used
                raise AssertionError("pool.map would stage only at batch end")

        original_pool = ocr.Pool
        ocr.Pool = FakePool
        try:
            items = [
                ("d1", 2015, "p1.pdf", {}),
                ("d2", 2015, "p2.pdf", {}),
                ("d3", 2015, "p3.pdf", {}),
            ]
            got = list(ocr._run_pool(items, 2))
            self.assertEqual([r["doc_id"] for r in got], ["d1", "d2", "d3"])
        finally:
            ocr.Pool = original_pool


class TestRunSweepIncrementalStaging(unittest.TestCase):
    """run_sweep stages every doc the moment its result streams in, not at
    batch end (durable incremental progress)."""

    def test_per_doc_staging_happens_before_next_result(self):
        from types import SimpleNamespace  # noqa: PLC0415

        staged_calls = []

        def fake_process_one(item):
            doc_id, year, pdf, meta = item
            return {
                "doc_id": doc_id,
                "year": year,
                "status": "resolved" if doc_id != "d2" else "unresolved",
                "row_count": 1,
                "reasons": [],
                "rows": [{"asset_description": "X (TICK)", "transaction_date": "01/02/2015"}],
                "canary": None,
            }

        class FakeStreamPool:
            def __init__(self, workers):
                self.workers = workers

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def imap_unordered(self, fn, items):
                for item in items:
                    yield fn(item)
                    # The caller must have staged this doc before asking for
                    # the next result -- i.e. staging is incremental, not
                    # deferred to batch end.
                    doc_id = item[0]
                    if doc_id not in [c[0] for c in staged_calls]:
                        raise AssertionError(
                            f"doc {doc_id} not staged before next result was pulled"
                        )

            def map(self, fn, items):  # pragma: no cover
                raise AssertionError("pool.map must not be used")

        original_pool = ocr.Pool
        original_process_one = ocr._process_one
        original_load_input = ocr.load_input_list
        original_load_meta = ocr.load_metadata
        original_stage = ocr.stage_document
        original_manifest = ocr.write_manifest
        ocr.Pool = FakeStreamPool
        ocr._process_one = fake_process_one
        ocr.load_input_list = lambda manifest, data_dir: [
            ("d1", 2015, "p1.pdf"), ("d2", 2015, "p2.pdf"), ("d3", 2015, "p3.pdf"),
        ]
        ocr.load_metadata = lambda db: {}
        ocr.stage_document = lambda out, result: staged_calls.append(
            (result["doc_id"], out)
        )
        ocr.write_manifest = lambda *a, **kw: None
        try:
            import tempfile  # noqa: PLC0415

            with tempfile.TemporaryDirectory() as tmp:
                args = SimpleNamespace(
                    out=tmp, merge_only=False, data_dir="/tmp/data",
                    db="/tmp/congress.duckdb", manifest="/tmp/manifest.json",
                    years=None, workers=2, skip_staged=False, max_docs=None,
                )
                rc = ocr.run_sweep(args)
                self.assertEqual(rc, 0)
                self.assertEqual(
                    [c[0] for c in staged_calls], ["d1", "d2", "d3"]
                )
        finally:
            ocr.Pool = original_pool
            ocr._process_one = original_process_one
            ocr.load_input_list = original_load_input
            ocr.load_metadata = original_load_meta
            ocr.stage_document = original_stage
            ocr.write_manifest = original_manifest


if __name__ == "__main__":
    unittest.main()
