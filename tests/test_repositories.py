"""Direct unit tests for the 4 repository modules.

These were previously only exercised indirectly through the Database facade.
Each repository takes a raw duckdb connection; here we instantiate them with
``self.db.conn`` (the Database has already run schema migrations).
"""

import unittest
from datetime import date, datetime

import pandas as pd

from analyzer.metadata_repository import MetadataRepository
from analyzer.parse_run_repository import ParseRunRepository
from analyzer.price_repository import PriceRepository
from analyzer.transaction_repository import TransactionRepository

from .conftest import DatabaseTestCase


# ---------------------------------------------------------------------------
# TransactionRepository
# ---------------------------------------------------------------------------


class TestTransactionRepository(DatabaseTestCase):

    def setUp(self):
        super().setUp()
        self.repo = TransactionRepository(self.db.conn)

    def _make_df(self, **overrides):
        row = {
            "doc_id": "doc1",
            "member": "John Doe",
            "ticker": "AAPL",
            "transaction_date": date(2024, 3, 10),
            "disclosure_date": date(2024, 3, 15),
            "transaction_type": "Purchase",
            "owner_code": "DC",
            "amount_raw": "$1,001 - $15,000",
            "amount_midpoint": 8000.5,
            "instrument_type": None,
            "strike_price": None,
            "expiry_date": None,
            "asset_description": None,
        }
        row.update(overrides)
        return pd.DataFrame([row])

    def test_get_by_year_returns_correct_rows(self):
        df = pd.DataFrame([
            {
                "doc_id": "doc-a",
                "member": "John Doe",
                "ticker": "AAPL",
                "transaction_date": date(2024, 3, 10),
                "disclosure_date": date(2024, 3, 15),
                "transaction_type": "Purchase",
                "owner_code": "DC",
                "amount_raw": "$1,001 - $15,000",
                "amount_midpoint": 8000.0,
            },
            {
                "doc_id": "doc-b",
                "member": "Jane Smith",
                "ticker": "MSFT",
                "transaction_date": date(2024, 5, 5),
                "disclosure_date": date(2024, 5, 10),
                "transaction_type": "Sale",
                "amount_raw": "$15,001 - $50,000",
                "amount_midpoint": 32500.0,
            },
            {
                "doc_id": "doc-c",
                "member": "Other Year",
                "ticker": "GOOG",
                "transaction_date": date(2023, 1, 5),
                "disclosure_date": date(2023, 1, 10),
                "transaction_type": "Purchase",
            },
        ])
        self.repo.upsert(df, source="house_pdf")

        result = self.repo.get_by_year(2024)
        self.assertEqual(len(result), 2)
        self.assertEqual(set(result["ticker"]), {"AAPL", "MSFT"})
        # Ordered by disclosure_date DESC: MSFT (May) before AAPL (Mar)
        self.assertEqual(result.iloc[0]["ticker"], "MSFT")
        self.assertEqual(result.iloc[-1]["ticker"], "AAPL")

    def test_get_by_year_excludes_future_transaction_dates(self):
        # Row with transaction_date AFTER disclosure_date — likely OCR swap.
        bad_df = self._make_df(
            doc_id="doc-bad",
            transaction_date=date(2024, 3, 20),
            disclosure_date=date(2024, 3, 15),
        )
        good_df = self._make_df(
            doc_id="doc-good",
            ticker="MSFT",
            transaction_date=date(2024, 3, 10),
            disclosure_date=date(2024, 3, 15),
        )
        self.repo.upsert(bad_df, source="house_pdf")
        self.repo.upsert(good_df, source="house_pdf")

        # The bad row IS persisted in the table...
        count_in_table = self.db.conn.execute(
            "SELECT COUNT(*) FROM transactions WHERE doc_id = 'doc-bad'"
        ).fetchone()[0]
        self.assertEqual(count_in_table, 1)

        # ...but excluded from get_by_year.
        result = self.repo.get_by_year(2024)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["ticker"], "MSFT")

    def test_get_by_date_range(self):
        df = pd.DataFrame([
            {
                "doc_id": "d1",
                "member": "A",
                "ticker": "AAPL",
                "transaction_date": date(2024, 1, 5),
                "disclosure_date": date(2024, 1, 10),
                "transaction_type": "Purchase",
            },
            {
                "doc_id": "d2",
                "member": "B",
                "ticker": "MSFT",
                "transaction_date": date(2024, 2, 5),
                "disclosure_date": date(2024, 2, 10),
                "transaction_type": "Sale",
            },
            {
                "doc_id": "d3",
                "member": "C",
                "ticker": "GOOG",
                "transaction_date": date(2024, 3, 5),
                "disclosure_date": date(2024, 3, 10),
                "transaction_type": "Purchase",
            },
        ])
        self.repo.upsert(df, source="house_pdf")

        result = self.repo.get_by_date_range(date(2024, 1, 15), date(2024, 2, 15))
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["ticker"], "MSFT")

        # Inclusive endpoints.
        result2 = self.repo.get_by_date_range(date(2024, 1, 10), date(2024, 2, 10))
        self.assertEqual(len(result2), 2)
        self.assertEqual(set(result2["ticker"]), {"AAPL", "MSFT"})

    def test_upsert_deduplication(self):
        # Two identical rows (same dedup key) in one upsert call collapse to one.
        df = pd.DataFrame([
            {
                "doc_id": "doc-dupe",
                "member": "John Doe",
                "ticker": "AAPL",
                "transaction_date": date(2024, 3, 10),
                "disclosure_date": date(2024, 3, 15),
                "transaction_type": "Purchase",
                "amount_raw": "$1,001 - $15,000",
                "owner_code": "DC",
                "asset_description": "Apple",
            },
            {
                "doc_id": "doc-dupe",
                "member": "John Doe",
                "ticker": "AAPL",
                "transaction_date": date(2024, 3, 10),
                "disclosure_date": date(2024, 3, 15),
                "transaction_type": "Purchase",
                "amount_raw": "$1,001 - $15,000",
                "owner_code": "DC",
                "asset_description": "Apple",
            },
        ])
        self.repo.upsert(df, source="house_pdf")
        count = self.db.conn.execute(
            "SELECT COUNT(*) FROM transactions WHERE doc_id = 'doc-dupe'"
        ).fetchone()[0]
        self.assertEqual(count, 1)

    def test_upsert_idempotent(self):
        df = self._make_df()
        self.repo.upsert(df, source="house_pdf")
        self.repo.upsert(df, source="house_pdf")

        count = self.db.conn.execute(
            "SELECT COUNT(*) FROM transactions WHERE doc_id = 'doc1'"
        ).fetchone()[0]
        self.assertEqual(count, 1)

    def test_upsert_null_ticker_inserts(self):
        df = pd.DataFrame([
            {
                "doc_id": "doc-null",
                "member": "John Doe",
                "ticker": None,
                "transaction_date": date(2024, 3, 10),
                "disclosure_date": date(2024, 3, 15),
                "transaction_type": "Purchase",
                "amount_raw": "$1,001 - $15,000",
                "asset_description": "Municipal bond",
            },
        ])
        self.repo.upsert(df, source="house_pdf")

        rows = self.db.conn.execute(
            "SELECT ticker, asset_description FROM transactions WHERE doc_id = 'doc-null'"
        ).fetchall()
        self.assertEqual(len(rows), 1)
        self.assertIsNone(rows[0][0])
        self.assertEqual(rows[0][1], "Municipal bond")

    def test_upsert_dedup_keeps_first_row(self):
        # Two rows share the full dedup key but differ in non-key columns
        # (disclosure_date, amount_midpoint). keep="first" semantics must
        # retain the FIRST row and drop the rest within one upsert call.
        df = pd.DataFrame([
            {
                "doc_id": "doc-k", "member": "John Doe", "ticker": "AAPL",
                "transaction_date": date(2024, 3, 10),
                "disclosure_date": date(2024, 3, 15),
                "transaction_type": "Purchase",
                "amount_raw": "$1,001 - $15,000", "owner_code": "DC",
                "amount_midpoint": 8000.0,
            },
            {
                "doc_id": "doc-k", "member": "John Doe", "ticker": "AAPL",
                "transaction_date": date(2024, 3, 10),
                "disclosure_date": date(2024, 4, 1),
                "transaction_type": "Purchase",
                "amount_raw": "$1,001 - $15,000", "owner_code": "DC",
                "amount_midpoint": 9999.0,
            },
        ])
        self.repo.upsert(df, source="house_pdf")
        rows = self.db.conn.execute(
            "SELECT disclosure_date, amount_midpoint FROM transactions "
            "WHERE doc_id = 'doc-k'"
        ).fetchall()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][0], date(2024, 3, 15))
        self.assertEqual(rows[0][1], 8000.0)

    def test_upsert_null_ticker_idempotent(self):
        # The separate NULL-ticker INSERT path has no ON CONFLICT (NULLs do
        # not collide on the unique index). Idempotency therefore relies on
        # the LEFT-JOIN filter -- re-inserting the same NULL-ticker row must
        # NOT produce a duplicate.
        df = pd.DataFrame([
            {
                "doc_id": "doc-n", "member": "John Doe", "ticker": None,
                "transaction_date": date(2024, 3, 10),
                "disclosure_date": date(2024, 3, 15),
                "transaction_type": "Purchase",
                "amount_raw": "$1,001 - $15,000",
                "asset_description": "Municipal bond",
            }
        ])
        self.repo.upsert(df, source="house_pdf")
        self.repo.upsert(df, source="house_pdf")
        count = self.db.conn.execute(
            "SELECT COUNT(*) FROM transactions WHERE doc_id = 'doc-n'"
        ).fetchone()[0]
        self.assertEqual(count, 1)

    def test_upsert_mixed_null_and_nonnull_ticker(self):
        # A single upsert containing one NULL-ticker row and one non-NULL
        # row must insert BOTH (exercising both INSERT statements).
        df = pd.DataFrame([
            {
                "doc_id": "doc-m", "member": "A", "ticker": "AAPL",
                "transaction_date": date(2024, 3, 10),
                "disclosure_date": date(2024, 3, 15),
                "transaction_type": "Purchase",
            },
            {
                "doc_id": "doc-m", "member": "A", "ticker": None,
                "transaction_date": date(2024, 3, 10),
                "disclosure_date": date(2024, 3, 15),
                "transaction_type": "Purchase",
                "asset_description": "Muni bond",
            },
        ])
        self.repo.upsert(df, source="house_pdf")
        rows = self.db.conn.execute(
            "SELECT ticker FROM transactions WHERE doc_id = 'doc-m' "
            "ORDER BY ticker NULLS LAST"
        ).fetchall()
        self.assertEqual([r[0] for r in rows], ["AAPL", None])

    def test_upsert_rolls_back_when_null_ticker_insert_is_malformed(self):
        """A failure in the second INSERT must not persist the first INSERT."""
        df = pd.DataFrame([
            {
                "doc_id": "doc-good", "member": "A", "ticker": "AAPL",
                "transaction_date": "2024-03-10", "disclosure_date": "2024-03-15",
                "transaction_type": "Purchase",
            },
            {
                "doc_id": "doc-bad", "member": "B", "ticker": None,
                "transaction_date": "not-a-date", "disclosure_date": "2024-03-15",
                "transaction_type": "Purchase",
            },
        ])

        with self.assertRaises(Exception):
            self.repo.upsert(df, source="house_pdf")

        count = self.db.conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
        self.assertEqual(count, 0)

    def test_replace_for_docs_rolls_back_deletes_when_replacement_is_malformed(self):
        self.repo.upsert(self._make_df(doc_id="doc-original"), source="house_pdf")
        malformed = self._make_df(
            doc_id="doc-original",
            transaction_date="not-a-date",
        )

        with self.assertRaises(Exception):
            self.db.replace_transactions_for_docs(malformed, source="house_pdf")

        stored = self.repo.get_for_doc("doc-original")
        self.assertEqual(len(stored), 1)
        self.assertEqual(stored.iloc[0]["ticker"], "AAPL")

    def test_replace_for_docs_rolls_back_transactions_when_parse_run_fails(self):
        self.repo.upsert(self._make_df(doc_id="doc-original"), source="house_pdf")
        replacement = self._make_df(doc_id="doc-original", ticker="MSFT")
        malformed_run = {
            "doc_id": "doc-original",
            "year": "not-a-year",
            "parser_version": "v3",
            "status": "success",
            "engines_attempted": "pdfplumber",
            "raw_row_count": 1,
            "transaction_count": 1,
        }

        with self.assertRaises(Exception):
            self.db.replace_transactions_for_docs(
                replacement, source="house_pdf", parse_runs=[malformed_run],
            )

        stored = self.repo.get_for_doc("doc-original")
        self.assertEqual(stored["ticker"].tolist(), ["AAPL"])
        audit_count = self.db.conn.execute(
            "SELECT COUNT(*) FROM pdf_parse_runs WHERE doc_id = 'doc-original'"
        ).fetchone()[0]
        self.assertEqual(audit_count, 0)

    def test_get_by_year_includes_null_transaction_date(self):
        # The exclusion clause is "txn_date IS NULL OR txn_date <=
        # disclosure_date" -- a NULL transaction_date must be INCLUDED.
        self.repo.upsert(
            self._make_df(doc_id="doc-null-td", transaction_date=None),
            source="house_pdf",
        )
        result = self.repo.get_by_year(2024)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["ticker"], "AAPL")

    def test_get_by_date_range_excludes_future_transaction_dates(self):
        # The OCR-swap (transaction_date > disclosure_date) exclusion also
        # applies to get_by_date_range, not just get_by_year.
        bad_df = self._make_df(
            doc_id="doc-bad",
            transaction_date=date(2024, 3, 20),
            disclosure_date=date(2024, 3, 15),
        )
        good_df = self._make_df(
            doc_id="doc-good",
            ticker="MSFT",
            transaction_date=date(2024, 3, 10),
            disclosure_date=date(2024, 3, 15),
        )
        self.repo.upsert(bad_df, source="house_pdf")
        self.repo.upsert(good_df, source="house_pdf")

        # Bad row is persisted...
        persisted = self.db.conn.execute(
            "SELECT COUNT(*) FROM transactions WHERE doc_id = 'doc-bad'"
        ).fetchone()[0]
        self.assertEqual(persisted, 1)

        # ...but excluded from the range query.
        result = self.repo.get_by_date_range(date(2024, 3, 1), date(2024, 3, 31))
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["ticker"], "MSFT")

    def test_delete_for_doc(self):
        df = pd.DataFrame([
            {
                "doc_id": "doc-1",
                "member": "A",
                "ticker": "AAPL",
                "transaction_date": date(2024, 3, 10),
                "disclosure_date": date(2024, 3, 15),
                "transaction_type": "Purchase",
            },
            {
                "doc_id": "doc-2",
                "member": "B",
                "ticker": "MSFT",
                "transaction_date": date(2024, 3, 10),
                "disclosure_date": date(2024, 3, 15),
                "transaction_type": "Sale",
            },
        ])
        self.repo.upsert(df, source="house_pdf")
        self.repo.delete_for_doc("doc-1")

        remaining = self.db.conn.execute(
            "SELECT doc_id FROM transactions ORDER BY doc_id"
        ).fetchall()
        self.assertEqual([r[0] for r in remaining], ["doc-2"])

    def test_count_for_docs(self):
        df = pd.DataFrame([
            {
                "doc_id": "c1",
                "member": "A",
                "ticker": "AAPL",
                "transaction_date": date(2024, 3, 10),
                "disclosure_date": date(2024, 3, 15),
                "transaction_type": "Purchase",
            },
            {
                "doc_id": "c1",
                "member": "A",
                "ticker": "MSFT",
                "transaction_date": date(2024, 3, 11),
                "disclosure_date": date(2024, 3, 15),
                "transaction_type": "Sale",
            },
            {
                "doc_id": "c2",
                "member": "B",
                "ticker": "GOOG",
                "transaction_date": date(2024, 3, 10),
                "disclosure_date": date(2024, 3, 15),
                "transaction_type": "Purchase",
            },
        ])
        self.repo.upsert(df, source="house_pdf")

        counts = self.repo.count_for_docs(["c1", "c2", "missing"])
        self.assertEqual(counts, {"c1": 2, "c2": 1})

    def test_count_for_docs_empty_list(self):
        self.assertEqual(self.repo.count_for_docs([]), {})

    def test_exists_true_and_false(self):
        self.assertFalse(self.repo.exists(2024))
        self.repo.upsert(self._make_df(), source="house_pdf")
        self.assertTrue(self.repo.exists(2024))
        self.assertFalse(self.repo.exists(2023))


# ---------------------------------------------------------------------------
# PriceRepository
# ---------------------------------------------------------------------------


class TestPriceRepository(DatabaseTestCase):

    def setUp(self):
        super().setUp()
        self.repo = PriceRepository(self.db.conn)

    def test_upsert_and_get_roundtrip(self):
        dates = pd.date_range("2024-01-01", "2024-01-04", freq="B")
        price_data = pd.DataFrame(
            {"AAPL": [180.0, 181.0, 182.0, 183.0], "MSFT": [370.0, 371.0, 372.0, 373.0]},
            index=dates,
        )
        self.repo.upsert(price_data)

        result = self.repo.get(["AAPL", "MSFT"], date(2024, 1, 1), date(2024, 1, 5))
        self.assertEqual(len(result), 4)
        self.assertIn("AAPL", result.columns)
        self.assertIn("MSFT", result.columns)
        self.assertAlmostEqual(result.loc[pd.Timestamp("2024-01-02"), "AAPL"], 181.0)
        self.assertAlmostEqual(result.loc[pd.Timestamp("2024-01-03"), "MSFT"], 372.0)

    def test_get_empty_tickers_returns_empty(self):
        result = self.repo.get([], date(2024, 1, 1), date(2024, 1, 5))
        self.assertTrue(result.empty)

    def test_upsert_empty_df_noop(self):
        self.repo.upsert(pd.DataFrame())
        # No error; nothing in the table.
        count = self.db.conn.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
        self.assertEqual(count, 0)

    def test_get_missing_all_missing(self):
        # Brand-new ticker → full business-day date range returned.
        missing_tickers, missing_dates = self.repo.get_missing(
            ["AAPL"], date(2024, 1, 1), date(2024, 1, 5)
        )
        self.assertEqual(missing_tickers, ["AAPL"])
        expected = list(pd.date_range("2024-01-01", "2024-01-05", freq="B"))
        self.assertEqual(missing_dates, expected)

    def test_get_missing_partial_gaps(self):
        # AAPL has Jan 1-2 only; gaps are Jan 3-5 business days.
        dates = pd.bdate_range("2024-01-01", "2024-01-02")
        prices = pd.DataFrame({"AAPL": [100.0, 101.0]}, index=dates)
        self.repo.upsert(prices)

        missing_tickers, missing_dates = self.repo.get_missing(
            ["AAPL"], date(2024, 1, 1), date(2024, 1, 5)
        )
        # Exact ticker list (not just membership) and exact ordered date list.
        self.assertEqual(missing_tickers, ["AAPL"])
        expected = list(pd.bdate_range("2024-01-03", "2024-01-05"))
        self.assertEqual(missing_dates, expected)
        # Returned dates must be sorted ascending (gap_dates is sorted()).
        self.assertEqual(missing_dates, sorted(missing_dates))

    def test_get_missing_complete(self):
        dates = pd.bdate_range("2024-01-01", "2024-01-05")
        prices = pd.DataFrame({"AAPL": range(len(dates))}, index=dates)
        self.repo.upsert(prices)

        missing_tickers, missing_dates = self.repo.get_missing(
            ["AAPL"], date(2024, 1, 1), date(2024, 1, 5)
        )
        self.assertEqual(missing_tickers, [])
        self.assertEqual(missing_dates, [])

    def test_upsert_updates_existing(self):
        dates = pd.bdate_range("2024-01-01", periods=2)
        self.repo.upsert(pd.DataFrame({"AAPL": [100.0, 101.0]}, index=dates))
        # New close for Jan 1; Jan 2 unchanged.
        self.repo.upsert(pd.DataFrame({"AAPL": [200.0, 101.0]}, index=dates))

        result = self.repo.get(["AAPL"], date(2024, 1, 1), date(2024, 1, 3))
        self.assertAlmostEqual(result.loc[pd.Timestamp("2024-01-01"), "AAPL"], 200.0)
        self.assertAlmostEqual(result.loc[pd.Timestamp("2024-01-02"), "AAPL"], 101.0)

    def test_upsert_drops_nan_close(self):
        # melt + dropna(subset=["close"]) must drop NaN cells, not persist
        # them and not break the surrounding rows.
        dates = pd.bdate_range("2024-01-01", "2024-01-03")
        prices = pd.DataFrame(
            {"AAPL": [100.0, float("nan"), 102.0]}, index=dates
        )
        self.repo.upsert(prices)

        rows = self.db.conn.execute(
            "SELECT date, close FROM prices WHERE ticker = 'AAPL' ORDER BY date"
        ).fetchall()
        self.assertEqual(len(rows), 2)
        self.assertEqual([r[0] for r in rows], [date(2024, 1, 1), date(2024, 1, 3)])
        self.assertEqual([r[1] for r in rows], [100.0, 102.0])

    def test_get_missing_insufficient_history(self):
        # Ticker EXISTS but its earliest price is more than 7 calendar days
        # after start_date (start_cutoff) -> the "insufficient" branch
        # triggers a FULL refetch (all business days), not just the early gap.
        late_dates = pd.bdate_range("2024-01-20", "2024-01-31")
        prices = pd.DataFrame({"AAPL": range(len(late_dates))}, index=late_dates)
        self.repo.upsert(prices)

        missing_tickers, missing_dates = self.repo.get_missing(
            ["AAPL"], date(2024, 1, 1), date(2024, 1, 31)
        )
        self.assertEqual(missing_tickers, ["AAPL"])
        # Full business-day range Jan 1-31, not just the Jan 1-19 hole.
        self.assertEqual(
            missing_dates, list(pd.bdate_range("2024-01-01", "2024-01-31"))
        )

    # -- get_entry_prices (previously completely untested) ---------------

    def _seed_transactions(self, rows):
        tx_df = pd.DataFrame(rows)
        TransactionRepository(self.db.conn).upsert(tx_df, source="house_pdf")

    def test_entry_prices_basic_asof(self):
        # ASOF join picks the latest price on or before disclosure_date.
        self._seed_transactions([{
            "doc_id": "d1", "member": "John Doe", "ticker": "AAPL",
            "transaction_date": date(2024, 3, 10),
            "disclosure_date": date(2024, 3, 15),
            "transaction_type": "Purchase", "owner_code": "DC",
            "amount_midpoint": 8000.0,
        }])
        # Disclosure Mar 15 (Fri). Prices Mar 13=200, Mar 14=210.
        # No price on Mar 15 -> ASOF picks Mar 14 (210).
        self.repo.upsert(pd.DataFrame(
            {"AAPL": [200.0, 210.0]},
            index=pd.bdate_range("2024-03-13", "2024-03-14"),
        ))

        result = self.repo.get_entry_prices(
            ["AAPL"], date(2024, 3, 1), date(2024, 3, 31), max_staleness_days=30
        )
        self.assertEqual(len(result), 1)
        row = result.iloc[0]
        self.assertEqual(row["ticker"], "AAPL")
        self.assertAlmostEqual(row["entry_price"], 210.0)
        self.assertEqual(row["member"], "John Doe")
        self.assertEqual(row["transaction_type"], "Purchase")

    def test_entry_prices_picks_same_day_price(self):
        # A price exactly ON disclosure_date wins (<= is inclusive).
        self._seed_transactions([{
            "doc_id": "d1", "member": "A", "ticker": "AAPL",
            "transaction_date": date(2024, 3, 10),
            "disclosure_date": date(2024, 3, 15),
            "transaction_type": "Purchase",
        }])
        self.repo.upsert(pd.DataFrame(
            {"AAPL": [200.0, 220.0]},
            index=pd.bdate_range("2024-03-14", "2024-03-15"),
        ))
        result = self.repo.get_entry_prices(
            ["AAPL"], date(2024, 3, 1), date(2024, 3, 31), max_staleness_days=30
        )
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result.iloc[0]["entry_price"], 220.0)

    def test_entry_prices_excludes_ticker_with_no_prices(self):
        # Two transactions; only AAPL has price history. The COALESCE(...) IS
        # NOT NULL clause must drop the MSFT row entirely.
        self._seed_transactions([
            {"doc_id": "d1", "member": "A", "ticker": "AAPL",
             "transaction_date": date(2024, 3, 10),
             "disclosure_date": date(2024, 3, 15),
             "transaction_type": "Purchase"},
            {"doc_id": "d2", "member": "B", "ticker": "MSFT",
             "transaction_date": date(2024, 2, 5),
             "disclosure_date": date(2024, 2, 15),
             "transaction_type": "Purchase"},
        ])
        self.repo.upsert(pd.DataFrame(
            {"AAPL": [200.0, 210.0]},
            index=pd.bdate_range("2024-03-13", "2024-03-14"),
        ))

        result = self.repo.get_entry_prices(
            ["AAPL", "MSFT"], date(2024, 1, 1), date(2024, 12, 31),
            max_staleness_days=30,
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["ticker"], "AAPL")

    def test_entry_prices_staleness_filter_drops_stale(self):
        # Disclosure Mar 15; latest prior price Mar 14 -> 1 day stale.
        # max_staleness_days=1 keeps it; =0 drops it.
        self._seed_transactions([{
            "doc_id": "d1", "member": "A", "ticker": "AAPL",
            "transaction_date": date(2024, 3, 10),
            "disclosure_date": date(2024, 3, 15),
            "transaction_type": "Purchase",
        }])
        self.repo.upsert(pd.DataFrame(
            {"AAPL": [200.0, 210.0]},
            index=pd.bdate_range("2024-03-13", "2024-03-14"),
        ))

        fresh = self.repo.get_entry_prices(
            ["AAPL"], date(2024, 3, 1), date(2024, 3, 31), max_staleness_days=1
        )
        self.assertEqual(len(fresh), 1)

        stale = self.repo.get_entry_prices(
            ["AAPL"], date(2024, 3, 1), date(2024, 3, 31), max_staleness_days=0
        )
        self.assertEqual(len(stale), 0)

    def test_entry_prices_empty_tickers_returns_empty(self):
        result = self.repo.get_entry_prices([], date(2024, 1, 1), date(2024, 12, 31))
        self.assertTrue(result.empty)

    def test_entry_prices_ticker_resolution(self):
        # Raw disclosure ticker "BRK.B" resolves to price symbol "BRK-B".
        # Price lives under the resolved symbol and must be found via the
        # p_res ASOF leg; the raw ticker is preserved in the output.
        self._seed_transactions([{
            "doc_id": "d1", "member": "A", "ticker": "BRK.B",
            "transaction_date": date(2024, 3, 10),
            "disclosure_date": date(2024, 3, 15),
            "transaction_type": "Purchase",
        }])
        self.repo.upsert(pd.DataFrame(
            {"BRK-B": [300.0, 310.0]},
            index=pd.bdate_range("2024-03-13", "2024-03-14"),
        ))
        result = self.repo.get_entry_prices(
            ["BRK.B"], date(2024, 3, 1), date(2024, 3, 31), max_staleness_days=30
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["ticker"], "BRK.B")
        self.assertAlmostEqual(result.iloc[0]["entry_price"], 310.0)

    def test_entry_prices_duplicate_ticker_no_duplicate_rows(self):
        # Passing the same raw ticker twice must not produce duplicate rows
        # (regression: ticker_map_entries was not deduped, so the LEFT JOIN
        # matched multiple map rows and multiplied the output).
        self._seed_transactions([{
            "doc_id": "d1", "member": "A", "ticker": "AAPL",
            "transaction_date": date(2024, 3, 10),
            "disclosure_date": date(2024, 3, 15),
            "transaction_type": "Purchase",
        }])
        self.repo.upsert(pd.DataFrame(
            {"AAPL": [180.0]},
            index=pd.bdate_range("2024-03-14", "2024-03-14"),
        ))
        result = self.repo.get_entry_prices(
            ["AAPL", "AAPL"], date(2024, 3, 1), date(2024, 3, 31), max_staleness_days=30
        )
        self.assertEqual(len(result), 1)


# ---------------------------------------------------------------------------
# MetadataRepository
# ---------------------------------------------------------------------------


class TestMetadataRepository(DatabaseTestCase):

    def setUp(self):
        super().setUp()
        self.repo = MetadataRepository(self.db.conn)

    def _make_df(self, **overrides):
        row = {
            "doc_id": "doc1",
            "first_name": "John",
            "last_name": "Doe",
            "filing_date": datetime(2024, 3, 15, 12, 0, 0),
            "filing_type": "F1",
            "fetched_at": datetime(2024, 3, 16, 8, 0, 0),
        }
        row.update(overrides)
        return pd.DataFrame([row])

    def test_upsert_and_get_by_year(self):
        df = pd.DataFrame([
            {
                "doc_id": "doc1",
                "first_name": "John",
                "last_name": "Doe",
                "filing_date": datetime(2024, 3, 15, 12, 0, 0),
                "filing_type": "F1",
                "fetched_at": datetime(2024, 3, 16, 8, 0, 0),
            },
            {
                "doc_id": "doc2",
                "first_name": "Jane",
                "last_name": "Smith",
                "filing_date": datetime(2023, 6, 20, 14, 0, 0),
                "filing_type": "F2",
                "fetched_at": datetime(2023, 6, 21, 9, 0, 0),
            },
        ])
        self.repo.upsert(df)

        result_2024 = self.repo.get_by_year(2024)
        self.assertEqual(len(result_2024), 1)
        self.assertEqual(result_2024.iloc[0]["DocID"], "doc1")
        self.assertEqual(result_2024.iloc[0]["First"], "John")
        self.assertEqual(result_2024.iloc[0]["Last"], "Doe")
        self.assertEqual(result_2024.iloc[0]["FilingType"], "F1")
        self.assertEqual(
            pd.Timestamp(result_2024.iloc[0]["FilingDate"]),
            pd.Timestamp(datetime(2024, 3, 15, 12, 0, 0)),
        )

        result_2023 = self.repo.get_by_year(2023)
        self.assertEqual(len(result_2023), 1)
        self.assertEqual(result_2023.iloc[0]["DocID"], "doc2")

    def test_upsert_conflict_updates(self):
        df1 = self._make_df(first_name="John", filing_type="F1")
        df2 = self._make_df(first_name="Jonathan", filing_type="F1-AMENDED",
                            fetched_at=datetime(2024, 3, 17, 10, 0, 0))
        self.repo.upsert(df1)
        self.repo.upsert(df2)

        result = self.repo.get_by_year(2024)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["First"], "Jonathan")
        self.assertEqual(result.iloc[0]["FilingType"], "F1-AMENDED")

    def test_exists_true_and_false(self):
        self.assertFalse(self.repo.exists(2024))
        self.repo.upsert(self._make_df())
        self.assertTrue(self.repo.exists(2024))
        self.assertFalse(self.repo.exists(2023))

    def test_clear_removes_year(self):
        df = pd.DataFrame([
            {
                "doc_id": "doc-2024",
                "first_name": "John",
                "last_name": "Doe",
                "filing_date": datetime(2024, 3, 15, 12, 0, 0),
                "filing_type": "F1",
                "fetched_at": datetime(2024, 3, 16, 8, 0, 0),
            },
            {
                "doc_id": "doc-2023",
                "first_name": "Jane",
                "last_name": "Smith",
                "filing_date": datetime(2023, 6, 20, 14, 0, 0),
                "filing_type": "F2",
                "fetched_at": datetime(2023, 6, 21, 9, 0, 0),
            },
        ])
        self.repo.upsert(df)
        self.assertTrue(self.repo.exists(2024))
        self.assertTrue(self.repo.exists(2023))

        self.repo.clear(2024)
        self.assertFalse(self.repo.exists(2024))
        self.assertTrue(self.repo.exists(2023))


# ---------------------------------------------------------------------------
# ParseRunRepository
# ---------------------------------------------------------------------------


class TestParseRunRepository(DatabaseTestCase):

    def setUp(self):
        super().setUp()
        self.repo = ParseRunRepository(self.db.conn)

    def test_upsert_inserts_row(self):
        self.repo.upsert(
            doc_id="doc1",
            year=2024,
            parser_version="v2",
            status="success",
            engines_attempted="lattice,stream",
            raw_row_count=10,
            transaction_count=5,
        )
        rows = self.db.conn.execute(
            "SELECT doc_id, year, parser_version, status, engines_attempted, "
            "raw_row_count, transaction_count, error_message "
            "FROM pdf_parse_runs WHERE doc_id = 'doc1'"
        ).fetchall()
        self.assertEqual(len(rows), 1)
        r = rows[0]
        self.assertEqual(r[0], "doc1")
        self.assertEqual(r[1], 2024)
        self.assertEqual(r[2], "v2")
        self.assertEqual(r[3], "success")
        self.assertEqual(r[4], "lattice,stream")
        self.assertEqual(r[5], 10)
        self.assertEqual(r[6], 5)
        self.assertIsNone(r[7])

    def test_upsert_is_idempotent(self):
        kwargs = dict(
            doc_id="doc-x",
            year=2024,
            parser_version="v2",
            status="success",
            engines_attempted="lattice",
            raw_row_count=3,
            transaction_count=2,
        )
        self.repo.upsert(**kwargs)
        self.repo.upsert(**kwargs)

        count = self.db.conn.execute(
            "SELECT COUNT(*) FROM pdf_parse_runs WHERE doc_id = 'doc-x'"
        ).fetchone()[0]
        self.assertEqual(count, 1)

    def test_upsert_persists_error_message(self):
        # The error_message column is omitted from the other tests (defaults
        # to None); verify it is actually written when provided.
        self.repo.upsert(
            doc_id="doc-err",
            year=2024,
            parser_version="v2",
            status="error",
            engines_attempted="lattice,ocr",
            raw_row_count=0,
            transaction_count=0,
            error_message="Traceback: boom at line 42",
        )
        row = self.db.conn.execute(
            "SELECT status, error_message FROM pdf_parse_runs "
            "WHERE doc_id = 'doc-err'"
        ).fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row[0], "error")
        self.assertEqual(row[1], "Traceback: boom at line 42")

    def test_upsert_replaces_with_new_status(self):
        # Second call should fully replace (delete + insert) -- e.g. status change.
        self.repo.upsert(
            doc_id="doc-r",
            year=2024,
            parser_version="v2",
            status="zero_rows",
            engines_attempted="lattice",
            raw_row_count=0,
            transaction_count=0,
        )
        self.repo.upsert(
            doc_id="doc-r",
            year=2024,
            parser_version="v2",
            status="success",
            engines_attempted="lattice,ocr",
            raw_row_count=4,
            transaction_count=3,
        )
        rows = self.db.conn.execute(
            "SELECT status, engines_attempted, transaction_count "
            "FROM pdf_parse_runs WHERE doc_id = 'doc-r'"
        ).fetchall()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][0], "success")
        self.assertEqual(rows[0][1], "lattice,ocr")
        self.assertEqual(rows[0][2], 3)

    def test_upsert_rolls_back_delete_when_replacement_is_malformed(self):
        self.repo.upsert(
            doc_id="doc-r",
            year=2024,
            parser_version="v2",
            status="success",
            engines_attempted="lattice",
            raw_row_count=4,
            transaction_count=3,
        )

        with self.assertRaises(Exception):
            self.repo.upsert(
                doc_id="doc-r",
                year="not-a-year",
                parser_version="v3",
                status="error",
                engines_attempted="ocr",
                raw_row_count=0,
                transaction_count=0,
            )

        row = self.db.conn.execute(
            "SELECT year, parser_version, status FROM pdf_parse_runs WHERE doc_id = 'doc-r'"
        ).fetchone()
        self.assertEqual(row, (2024, "v2", "success"))


if __name__ == "__main__":
    unittest.main()
