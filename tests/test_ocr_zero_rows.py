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
            "MEMBER: Jane Doe\nPAGES: 1\nPAGE: 1\n"
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

        output = "MEMBER: Jane Doe\nPAGES: 1\nPAGE: 1\nApple Inc. (AAPL) | Partial Sale | 01/15/24 | 01/20/24 | A\n"
        _, txs = ocr_zero_rows.parse_output(output)
        self.assertEqual(len(txs), 1)
        self.assertEqual(txs[0]["type"], "Sale")

    def test_parse_output_skips_invalid_date(self):
        from scripts import ocr_zero_rows

        output = (
            "MEMBER: Jane Doe\nPAGES: 1\nPAGE: 1\n"
            "Apple Inc. (AAPL) | Purchase | 01/15/24 | 01/20/24 | A\n"
            "Bad Corp. (BAD) | Purchase | not-a-date | 01/20/24 | A\n"
        )
        with self.assertRaises(ocr_zero_rows.GeminiOutputError):
            ocr_zero_rows.parse_output(output)

    def test_parse_output_fuzzy_match_transaction_types(self):
        from scripts import ocr_zero_rows

        output = (
            "MEMBER: Jane Doe\nPAGES: 1\nPAGE: 1\n"
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
            "MEMBER: Jane Doe\nPAGES: 1\nPAGE: 1\n"
            "ASSET | TYPE | DATE | DISC | AMOUNT\n"
            "--- | --- | --- | --- | ---\n"
            "Apple Inc. (AAPL) | Purchase | 01/15/24 | 01/20/24 | A\n"
        )
        with self.assertRaises(ocr_zero_rows.GeminiOutputError):
            ocr_zero_rows.parse_output(output)

    def test_parse_output_amount_midpoint_mapped(self):
        from scripts import ocr_zero_rows

        output = "MEMBER: Jane Doe\nPAGES: 1\nPAGE: 1\nApple Inc. (AAPL) | Purchase | 01/15/24 | 01/20/24 | D\n"
        _, txs = ocr_zero_rows.parse_output(output)
        self.assertEqual(len(txs), 1)
        self.assertEqual(txs[0]["amount_letter"], "D")
        self.assertEqual(txs[0]["amount_midpoint"], 175000)

    def test_parse_output_skips_prompt_example(self):
        from scripts import ocr_zero_rows

        output = (
            "MEMBER: Jane Doe\nPAGES: 1\nPAGE: 1\n"
            "Example: Mega Corp. Common Stock | Purchase | 01/01/24 | 01/02/24 | A\n"
            "Apple (AAPL) | Purchase | 01/15/24 | 01/20/24 | B\n"
        )
        _, txs = ocr_zero_rows.parse_output(output)
        self.assertEqual([tx["asset"] for tx in txs], ["Apple (AAPL)"])

    def test_parse_output_explicit_no_transactions(self):
        from scripts import ocr_zero_rows

        member, txs = ocr_zero_rows.parse_output(
            "MEMBER: Jane Doe\nPAGES: 1\nPAGE: 1\nNO_TRANSACTIONS"
        )
        self.assertEqual(member, "Jane Doe")
        self.assertEqual(txs, [])

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


# --------------------------------------------------------------------------
# finding 8: the authoritative unresolved queue as an exportable manifest
# --------------------------------------------------------------------------


def test_export_work_manifest_binds_artifacts_and_generations(tmp_path, monkeypatch):
    import json

    import duckdb

    from scripts import ocr_zero_rows

    pdf = tmp_path / "2024" / "pdfs" / "m1.pdf"
    pdf.parent.mkdir(parents=True)
    pdf.write_bytes(b"%PDF-1.4 manifest-test")
    missing = tmp_path / "2023" / "pdfs" / "gone.pdf"
    monkeypatch.setattr(
        ocr_zero_rows,
        "get_ocr_work_items",
        lambda **kwargs: [("m1", 2024, str(pdf)), ("gone", 2023, str(missing))],
    )
    monkeypatch.setattr(
        ocr_zero_rows,
        "_resolve_ingestion_generation",
        lambda conn, doc_id, year, sha: "gen-7",
    )
    db_path = tmp_path / "c.duckdb"
    duckdb.connect(str(db_path)).close()

    out = tmp_path / "staging" / "glm-backlog" / "work-manifest.json"
    envelope = ocr_zero_rows.export_work_manifest(
        out, db_path=str(db_path), data_dir=tmp_path
    )
    assert envelope["schema"] == ocr_zero_rows.WORK_MANIFEST_SCHEMA
    assert envelope["item_count"] == 2
    by_id = {item["doc_id"]: item for item in envelope["items"]}
    # The local artifact is bound at export time (present doc)...
    assert by_id["m1"]["artifact_sha256"] == ocr_zero_rows.pdf_sha256(pdf)
    assert by_id["m1"]["ingestion_generation"] == "gen-7"
    # ...and absence stays honest (missing PDF -> no invented binding).
    assert by_id["gone"]["artifact_sha256"] is None
    assert by_id["gone"]["ingestion_generation"] is None
    assert json.loads(out.read_text()) == envelope


def test_cli_export_work_manifest_writes_and_exits(tmp_path, monkeypatch, capsys):
    import json
    import sys

    import duckdb

    from scripts import ocr_zero_rows

    db_path = tmp_path / "c.duckdb"
    duckdb.connect(str(db_path)).close()
    monkeypatch.setattr(ocr_zero_rows, "DB_PATH", str(db_path))
    monkeypatch.setattr(ocr_zero_rows, "get_ocr_work_items", lambda **kwargs: [])
    manifest = tmp_path / "custom" / "work-manifest.json"
    monkeypatch.setattr(
        sys, "argv", ["ocr_zero_rows.py", "--export-work-manifest", str(manifest)]
    )

    ocr_zero_rows.main()  # exports and exits before any engine-mode setup

    assert "0 unresolved docs" in capsys.readouterr().out
    envelope = json.loads(manifest.read_text())
    assert envelope["schema"] == ocr_zero_rows.WORK_MANIFEST_SCHEMA
    assert envelope["items"] == []


def test_export_work_manifest_default_path_is_the_staging_root(tmp_path, monkeypatch):
    from scripts import ocr_zero_rows

    db_path = tmp_path / "c.duckdb"
    import duckdb

    duckdb.connect(str(db_path)).close()
    monkeypatch.setattr(ocr_zero_rows, "DB_PATH", str(db_path))
    monkeypatch.setattr(ocr_zero_rows, "get_ocr_work_items", lambda **kwargs: [])
    monkeypatch.chdir(tmp_path)

    envelope = ocr_zero_rows.export_work_manifest()
    assert envelope["items"] == []
    assert (
        tmp_path / "data" / "staging" / "glm-backlog" / "work-manifest.json"
    ).exists()


if __name__ == "__main__":
    unittest.main()
