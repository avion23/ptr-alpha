"""Regression tests for review finding 3 (2026-10-18): the deterministic
cascade accepted demonstrated corruption for data/2017/pdfs/20006438.pdf.

Pre-fix the text-only worker returned 16 rows for a document that visibly
contains 14: the visible ETP and PSXP transactions were omitted, four lots
were duplicated (pdfplumber's table grid and its text layout each contributed
a copy with a different asset/owner reading), and the literal "FILING STATUS:"
label text produced ticker "STATUS". Downstream consolidation kept all 16 rows
because the cascade treated pdftotext's 8-row subset as proven coverage of
pdfplumber's 16-row reading.

Reference evidence (not asserted directly -- engines may legitimately disagree
on marginal rows): GLM's independently pixel-audited 14-row reading is stored
in data/staging/glm-backlog/audit-review/2017/20006438/record.json; it lists
exactly one ETP row, exactly one PSXP row, no "STATUS" ticker, and no
content-duplicated lot (the two SOUTH CAROLINA lots differ by amount band).
"""

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from analyzer import parser_cascade
from analyzer.parsing.cells import _extract_ticker
from analyzer.parsing.pdftotext_parser import _parse_pdftotext_lines
from analyzer.parsing.rows import parse_pdf_table

HEADER = [
    "Asset Name",
    "Owner",
    "Transaction Type",
    "Transaction Date",
    "Notification Date",
    "Amount",
]


def _data_root():
    return next(
        (
            parent / "data"
            for parent in Path(__file__).resolve().parents
            if (parent / "data").is_dir()
        ),
        None,
    )


def tx(
    asset,
    *,
    ticker=None,
    date="12/21/2016",
    tx_type="Exchange",
    amount="$50,001 - $100,000",
    source_row_id=None,
):
    transaction = {
        "ticker": ticker,
        "transaction_type": tx_type,
        "transaction_date": date,
        "owner_code": None,
        "amount_raw": amount,
        "amount_midpoint": 8000,
        "instrument_type": "stock",
        "strike_price": None,
        "expiry_date": None,
        "asset_description": asset,
        "source_row_id": source_row_id,
    }
    return transaction


class TestTickerMetadataContamination(unittest.TestCase):
    """cells.py must never turn a filing-metadata label into a ticker."""

    def test_filing_status_label_does_not_become_ticker(self):
        asset = (
            "PENNFIELD MICH SCHS DIST Go 4% 05/01/22 MN - MUNI BoND "
            "FILING STATUS: New SUBHoLDING oF: Appendix A (100% interest)"
        )
        self.assertIsNone(_extract_ticker(asset))

    def test_ticker_before_the_label_is_still_resolved(self):
        asset = (
            "EQT Midstream Partners, LP Common Units representing Limited "
            "Partner Interests (EQM) FILING STATUS: New SUBHoLDING oF: Trust"
        )
        self.assertEqual(_extract_ticker(asset), "EQM")

    def test_short_label_variants_do_not_become_tickers(self):
        for asset in (
            "Some MUNI BoND F S: New",
            "SoUTH CARoLINA PUB SVCE AUTH REV S O: Trust Account",
            "Foo Bar [ST] SOURCE OF: Pension",
        ):
            with self.subTest(asset=asset):
                self.assertNotEqual(_extract_ticker(asset), "STATUS")
                self.assertNotEqual(_extract_ticker(asset), "S")
                self.assertNotEqual(_extract_ticker(asset), "OF")

    def test_colon_and_dollar_tickers_still_work_without_labels(self):
        self.assertEqual(_extract_ticker("Pay to ACME: settled"), "ACME")
        self.assertEqual(_extract_ticker("Holdings $BRK.B"), "BRK.B")


class TestPdftotextLayoutRepair(unittest.TestCase):
    """Single-digit dates and FILING STATUS lines reached the asset folder."""

    LINES = (
        (
            "                    Energy Transfer Partners, L.P. ENERGY P"
            "                       12/1/2016   12/2/2016        $1,001 - $15,000"
        ),
        "                    TRANSFER PARNTERS (ETP)",
        "                    FILING STATUS: New",
        "                    SUBHoLDING oF: Revocable Living Trust",
    )

    def test_single_digit_dates_are_transaction_rows(self):
        rows = _parse_pdftotext_lines(list(self.LINES))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][3], "12/1/2016")
        transactions = parse_pdf_table([HEADER] + rows)
        self.assertEqual(len(transactions), 1)
        self.assertEqual(transactions[0]["ticker"], "ETP")
        self.assertEqual(transactions[0]["transaction_date"], "12/1/2016")

    def test_filing_status_line_is_not_folded_into_the_asset(self):
        rows = _parse_pdftotext_lines(list(self.LINES))
        self.assertEqual(len(rows), 1)
        self.assertNotIn("FILING STATUS", rows[0][0])
        self.assertNotIn("SUBHoLDING", rows[0][0])
        self.assertIn("TRANSFER PARNTERS (ETP)", rows[0][0])

    def test_two_digit_dates_still_parse(self):
        rows = _parse_pdftotext_lines(
            [
                "  Apple Inc. (AAPL) P 12/28/2016 12/29/2016 $250,001 - $500,000",
                "  FILING STATUS: New",
            ]
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][3], "12/28/2016")


class TestStrictSubsetEscalation(unittest.TestCase):
    """The cascade must not read subset relationships as proven coverage."""

    def _run_text_rung(self, pdfplumber_rows, pdftotext_rows):
        with (
            patch.object(
                parser_cascade, "_has_document_text_coverage", return_value=True
            ),
            patch.object(
                parser_cascade, "_try_pdfplumber", return_value=pdfplumber_rows
            ),
            patch.object(parser_cascade, "_try_camelot_lattice", return_value=[]),
            patch.object(parser_cascade, "_try_camelot_stream", return_value=[]),
            patch.object(
                parser_cascade, "_try_pdftotext", return_value=pdftotext_rows
            ),
        ):
            return parser_cascade._run_text_engines(Path("20006438.pdf"))

    def test_superset_with_unobserved_rows_is_not_trusted(self):
        # The demonstrated pre-fix shape: pdftotext reads a subset of
        # pdfplumber, and pdfplumber's extra rows (here: truncated table cells
        # with a STATUS-contaminated twin) are corroborated by nobody.
        shared = tx("MoRGAN STANLEY FINANCE LLC (MS)", source_row_id="pdftotext:l9")
        phantom_a = tx(
            "JPMoRGAN CHASE & Co. STRUCTURED NoTE",
            ticker="STATUS",
            date="12/28/2016",
            amount="$250,001 - $500,000",
            source_row_id="pdfplumber:p1:t0:r4",
        )
        phantom_b = tx(
            "JPMoRGAN CHASE & Co. STRUCTURED NoTE FILING STATUS: New",
            ticker="STATUS",
            date="12/28/2016",
            amount="$250,001 - $500,000",
            source_row_id="pdfplumber:p1:l7",
        )
        result, _, _, engines, _ = self._run_text_rung(
            [shared, phantom_a, phantom_b], [shared]
        )
        self.assertIsNone(result)
        self.assertIn("text_row_disagreement_unresolved", engines)

    def test_mixed_coordinate_system_duplicates_are_not_trusted(self):
        # The merge-artifact signature from 20006438: the same lot reported by
        # pdfplumber's table grid AND its text layout with a diverging
        # owner/asset split. Coordinate distinctness cannot prove two visual
        # rows across systems, so the reading must escalate.
        layout_reading = tx(
            "C ST PUB SVC AUTH REV - MUNI BoND",
            source_row_id="pdfplumber:p2:l47",
        )
        table_reading = tx(
            "S C ST PUB SVC AUTH REV - MUNI BoND",
            source_row_id="pdfplumber:p2:t0:r9",
        )
        result, _, _, engines, _ = self._run_text_rung(
            [layout_reading, table_reading], [layout_reading]
        )
        self.assertIsNone(result)
        self.assertIn("text_row_disagreement_unresolved", engines)

    def test_same_system_identical_lots_remain_trusted(self):
        # Legitimate identical lots: one parser reports the same content from
        # two rows of its own coordinate system. The superset reading that saw
        # both lots wins and every lot is preserved.
        lot_one = tx("Identical Trust Lot", source_row_id="pdftotext:l17")
        lot_two = tx("Identical Trust Lot", source_row_id="pdftotext:l17")
        lot_three = tx("Other Trust Lot", source_row_id="pdftotext:l39")
        result, _, _, engines, _ = self._run_text_rung(
            [lot_one, lot_two, lot_three], [lot_one, lot_three]
        )
        self.assertIsNotNone(result)
        self.assertIn("won:pdfplumber", engines)
        self.assertEqual(len(result[1]), 3)

    def test_equal_text_engines_stay_trusted(self):
        shared = tx("Apple Inc. (AAPL)", source_row_id="pdftotext:l5")
        result, _, _, engines, _ = self._run_text_rung([shared], [shared])
        self.assertIsNotNone(result)
        self.assertIn("won:pdftotext", engines)
        self.assertIn("trusted:pdfplumber_subset_pdftotext", engines)


class TestRealDocument20006438(unittest.TestCase):
    """The real disclosure PDF that demonstrated the corruption."""

    def test_deterministic_cascade_never_returns_the_corrupted_reading(self):
        data_root = _data_root()
        pdf = data_root / "2017" / "pdfs" / "20006438.pdf" if data_root else None
        if pdf is None or not pdf.is_file():
            self.skipTest("real disclosure canary artifact 20006438.pdf is absent")

        # Engine-level repair: pdftotext alone must now read the visible
        # document faithfully (audited reference: 14 rows).
        from analyzer.parsing import extract_tables_with_pdftotext

        rows = [
            transaction
            for table in extract_tables_with_pdftotext(pdf)
            for transaction in parse_pdf_table(table)
        ]
        tickers = {transaction["ticker"] for transaction in rows}
        self.assertIn("ETP", tickers, "visible ETP transaction was omitted")
        self.assertIn("PSXP", tickers, "visible PSXP transaction was omitted")
        self.assertNotIn(
            "STATUS", tickers, "FILING STATUS metadata leaked into a ticker"
        )
        identities = [
            parser_cascade._transaction_identity(transaction)
            for transaction in rows
        ]
        self.assertEqual(
            len(identities),
            len(set(identities)),
            "phantom duplicated lot: two rows share one transaction identity",
        )

        # Cascade level: the text-only rung must never hand the corrupted
        # 16-row reading downstream. Correct rows or escalation are both
        # acceptable; the demonstrated defects are not.
        previous = os.environ.get("PTR_SKIP_DOCLING")
        os.environ["PTR_SKIP_DOCLING"] = "1"
        try:
            try:
                _, cascade_rows, _ = parser_cascade._parse_text_only_worker(pdf)
            except parser_cascade.ParserCascadeError:
                return  # escalated to the OCR rung: corrupted reading is gone
            cascade_tickers = {
                transaction["ticker"] for transaction in cascade_rows
            }
            self.assertNotIn("STATUS", cascade_tickers)
            self.assertIn("ETP", cascade_tickers)
            self.assertIn("PSXP", cascade_tickers)
            cascade_identities = [
                parser_cascade._transaction_identity(transaction)
                for transaction in cascade_rows
            ]
            self.assertEqual(len(cascade_identities), len(set(cascade_identities)))
        finally:
            if previous is None:
                os.environ.pop("PTR_SKIP_DOCLING", None)
            else:
                os.environ["PTR_SKIP_DOCLING"] = previous


if __name__ == "__main__":
    unittest.main()
