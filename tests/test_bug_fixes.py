"""Regression tests for the 8 verified data-layer bug fixes.

Each test class corresponds to one numbered finding.
All tests use temp DuckDB files — never the production data/congress.duckdb.
"""

import unittest
from datetime import date

import pandas as pd

from .conftest import DatabaseTestCase


# ---------------------------------------------------------------------------
# Fix 1 – Dedup key collapses distinct trades
# ---------------------------------------------------------------------------
class TestDedupKeyFix(DatabaseTestCase):
    """Two same-day lots with different amount_raw must both survive.
    Re-inserting the identical row must still dedupe to one.
    SP vs JT trades (different owner_code) must each survive.
    """

    def _base_tx(self, **overrides):
        row = {
            "doc_id": "D1",
            "member": "John Doe",
            "ticker": "AAPL",
            "transaction_date": date(2024, 3, 10),
            "disclosure_date": date(2024, 3, 15),
            "transaction_type": "Purchase",
            "owner_code": None,
            "amount_raw": None,
        }
        row.update(overrides)
        return row

    def test_two_lots_different_amount_raw_both_survive(self):
        df = pd.DataFrame(
            [
                self._base_tx(amount_raw="$1,001 - $15,000"),
                self._base_tx(amount_raw="$15,001 - $50,000"),
            ]
        )
        self.db.upsert_transactions(df, source="house_pdf")
        result = self.db.get_transactions(2024)
        self.assertEqual(len(result), 2, "Two distinct lots must not be collapsed")

    def test_reinserting_same_row_dedupes_to_one(self):
        row = self._base_tx(amount_raw="$1,001 - $15,000")
        df = pd.DataFrame([row])
        self.db.upsert_transactions(df, source="house_pdf")
        self.db.upsert_transactions(df, source="house_pdf")  # re-insert same row
        result = self.db.get_transactions(2024)
        self.assertEqual(
            len(result), 1, "Re-inserting same row must not create duplicate"
        )

    def test_sp_vs_jt_owner_code_both_survive(self):
        """SP-owned and JT-owned trades on the same day must be kept separate."""
        df = pd.DataFrame(
            [
                self._base_tx(owner_code="SP", amount_raw="$1,001 - $15,000"),
                self._base_tx(owner_code="J", amount_raw="$1,001 - $15,000"),
            ]
        )
        self.db.upsert_transactions(df, source="house_pdf")
        result = self.db.get_transactions(2024)
        self.assertEqual(len(result), 2, "SP and JT trades must not collapse")

    def test_null_amount_raw_normalised_to_empty_string(self):
        """NULL amount_raw must be stored as '' so re-parses can still dedupe."""
        df = pd.DataFrame([self._base_tx(amount_raw=None)])
        self.db.upsert_transactions(df, source="house_pdf")
        raw = self.db.conn.execute(
            "SELECT amount_raw FROM transactions WHERE ticker='AAPL'"
        ).fetchone()[0]
        self.assertEqual(raw, "", "NULL amount_raw must be normalised to ''")


# ---------------------------------------------------------------------------
# Fix 3 – Entry price join misses when price is cached under raw ticker
# ---------------------------------------------------------------------------
class TestEntryPriceRawTickerFallback(DatabaseTestCase):
    """BRK.B transaction + BRK.B prices → non-empty entry price.

    TickerResolver maps BRK.B → BRK-B, but YFinance stores prices under 'BRK.B'.
    The ASOF join must fall back to the raw ticker when the resolved one has no row.
    """

    def test_brk_b_raw_ticker_price_lookup(self):
        # Insert prices under the RAW symbol 'BRK.B'
        dates = pd.bdate_range("2024-01-01", "2024-01-05")
        prices = pd.DataFrame(
            {"BRK.B": [350.0 + i for i in range(len(dates))]}, index=dates
        )
        self.db.upsert_prices(prices)

        # Insert transaction with raw ticker 'BRK.B'
        tx = pd.DataFrame(
            [
                {
                    "doc_id": "brk-test",
                    "member": "Pelosi",
                    "ticker": "BRK.B",
                    "transaction_date": date(2024, 1, 3),
                    "disclosure_date": date(2024, 1, 3),
                    "transaction_type": "Purchase",
                }
            ]
        )
        self.db.upsert_transactions(tx, source="house_pdf")

        result = self.db.get_entry_prices(["BRK.B"], date(2024, 1, 1), date(2024, 1, 5))
        self.assertFalse(
            result.empty, "Entry price must be found via raw ticker fallback"
        )
        self.assertIsNotNone(result.iloc[0]["entry_price"])
        self.assertGreater(result.iloc[0]["entry_price"], 0)

    def test_standard_ticker_unaffected(self):
        """Non-remapped tickers (AAPL) still return correct price."""
        dates = pd.bdate_range("2024-01-01", "2024-01-05")
        prices = pd.DataFrame(
            {"AAPL": [180.0 + i for i in range(len(dates))]}, index=dates
        )
        self.db.upsert_prices(prices)

        tx = pd.DataFrame(
            [
                {
                    "doc_id": "aapl-test",
                    "member": "Smith",
                    "ticker": "AAPL",
                    "transaction_date": date(2024, 1, 2),
                    "disclosure_date": date(2024, 1, 2),
                    "transaction_type": "Purchase",
                }
            ]
        )
        self.db.upsert_transactions(tx, source="house_pdf")

        result = self.db.get_entry_prices(["AAPL"], date(2024, 1, 1), date(2024, 1, 5))
        self.assertFalse(result.empty)
        self.assertAlmostEqual(result.iloc[0]["entry_price"], 181.0)


# ---------------------------------------------------------------------------
# Fix 4 – Price cache completeness (per-ticker gaps)
# ---------------------------------------------------------------------------
class TestMissingPriceDataPerTicker(DatabaseTestCase):
    def test_ticker_with_gap_appears_in_missing_tickers(self):
        """Ticker missing some dates must appear in missing_tickers, not just missing_dates."""
        # AAPL has Jan 1-2, MSFT has all dates Jan 1-5
        aapl_dates = pd.bdate_range("2024-01-01", "2024-01-02")
        msft_dates = pd.bdate_range("2024-01-01", "2024-01-05")
        self.db.upsert_prices(
            pd.DataFrame({"AAPL": range(len(aapl_dates))}, index=aapl_dates)
        )
        self.db.upsert_prices(
            pd.DataFrame({"MSFT": range(len(msft_dates))}, index=msft_dates)
        )

        missing_tickers, missing_dates = self.db.get_missing_price_data(
            ["AAPL", "MSFT"], date(2024, 1, 1), date(2024, 1, 5)
        )
        self.assertIn(
            "AAPL",
            missing_tickers,
            "AAPL has a per-ticker gap and must appear in missing_tickers",
        )
        self.assertNotIn(
            "MSFT",
            missing_tickers,
            "MSFT has complete coverage and must not appear in missing_tickers",
        )
        self.assertTrue(len(missing_dates) > 0)

    def test_both_tickers_complete_returns_empty(self):
        """No gaps → both lists empty."""
        dates = pd.bdate_range("2024-01-01", "2024-01-05")
        self.db.upsert_prices(
            pd.DataFrame(
                {"AAPL": range(len(dates)), "MSFT": range(len(dates))}, index=dates
            )
        )
        missing_tickers, missing_dates = self.db.get_missing_price_data(
            ["AAPL", "MSFT"], date(2024, 1, 1), date(2024, 1, 5)
        )
        self.assertEqual(missing_tickers, [])
        self.assertEqual(len(missing_dates), 0)

    def test_complete_ticker_not_in_missing_when_sibling_has_gap(self):
        """MSFT complete → must NOT be in missing_tickers even when AAPL has gaps."""
        aapl_dates = pd.bdate_range("2024-01-01", "2024-01-01")  # only 1 day
        msft_dates = pd.bdate_range("2024-01-01", "2024-01-05")  # all days
        self.db.upsert_prices(pd.DataFrame({"AAPL": [100.0]}, index=aapl_dates))
        self.db.upsert_prices(
            pd.DataFrame({"MSFT": range(len(msft_dates))}, index=msft_dates)
        )

        missing_tickers, _ = self.db.get_missing_price_data(
            ["AAPL", "MSFT"], date(2024, 1, 1), date(2024, 1, 5)
        )
        self.assertNotIn("MSFT", missing_tickers)


# ---------------------------------------------------------------------------
# Fix 5 – Continuation rows lose amount/owner
# ---------------------------------------------------------------------------
class TestContinuationRowAmountOwner(unittest.TestCase):
    def test_amount_from_next_row_in_continuation(self):
        """Amount lives in next_row when asset name spans two rows."""
        from analyzer.parsing.rows import parse_pdf_table

        table = [
            ["Asset Name", "Owner", "Transaction Type", "Transaction Date", "Amount"],
            # Row 1: asset name is split — no ticker here
            ["Apple Inc. Common", None, None, None, None],
            # Row 2: completion of asset + transaction data including amount
            ["Stock (AAPL)", "Self", "P", "03/10/2024", "$1,001 - $15,000"],
        ]
        txs = parse_pdf_table(table)
        self.assertEqual(len(txs), 1)
        tx = txs[0]
        self.assertEqual(tx["ticker"], "AAPL")
        # Fix 5: amount must now come from next_row
        self.assertEqual(
            tx["amount_raw"],
            "$1,001 - $15,000",
            "amount_raw must be populated from the continuation row",
        )
        self.assertIsNotNone(
            tx["amount_midpoint"],
            "amount_midpoint must be parsed from continuation row amount",
        )

    def test_non_continuation_row_amount_unaffected(self):
        """Normal (non-continuation) rows still extract amount from the same row."""
        from analyzer.parsing.rows import parse_pdf_table

        table = [
            ["Asset Name", "Transaction Type", "Transaction Date", "Amount"],
            ["Apple Inc. (AAPL)", "P", "03/10/2024", "$50,001 - $100,000"],
        ]
        txs = parse_pdf_table(table)
        self.assertEqual(len(txs), 1)
        self.assertAlmostEqual(txs[0]["amount_midpoint"], 75000.5)


# ---------------------------------------------------------------------------
# Fix 7 – Member name splits → canonical key
# ---------------------------------------------------------------------------
class TestCanonicalMemberKey(unittest.TestCase):
    def test_mccaul_variants_map_to_same_key(self):
        from analyzer.member_names import canonical_member_key

        k1 = canonical_member_key("MICHAEL T. MCCAUL")
        k2 = canonical_member_key("MICHAEL MCCAUL")
        k3 = canonical_member_key("Michael T. McCaul")
        self.assertEqual(k1, k2)
        self.assertEqual(k1, k3)

    def test_harshbarger_variants_map_to_same_key(self):
        from analyzer.member_names import canonical_member_key

        k1 = canonical_member_key("Diana Harshbarger")
        k2 = canonical_member_key("Diana Lynn Harshbarger")
        self.assertEqual(k1, k2)

    def test_two_different_members_do_not_collide(self):
        from analyzer.member_names import canonical_member_key

        k1 = canonical_member_key("Michael McCaul")
        k2 = canonical_member_key("Nancy Pelosi")
        self.assertNotEqual(k1, k2, "Different members must not share a canonical key")

    def test_honorifics_stripped(self):
        from analyzer.member_names import canonical_member_key

        self.assertEqual(canonical_member_key("Dr. John Smith"), "JOHN SMITH")
        self.assertEqual(canonical_member_key("John Smith Jr."), "JOHN SMITH")
        self.assertEqual(canonical_member_key("John Smith III"), "JOHN SMITH")
        self.assertEqual(canonical_member_key("Hon. John Smith"), "JOHN SMITH")

    def test_empty_name_returns_empty(self):
        from analyzer.member_names import canonical_member_key

        self.assertEqual(canonical_member_key(""), "")
        self.assertEqual(canonical_member_key(None), "")

    def test_single_name_token_returned_as_is(self):
        from analyzer.member_names import canonical_member_key

        self.assertEqual(canonical_member_key("PELOSI"), "PELOSI")

    def test_canonical_key_is_uppercase(self):
        from analyzer.member_names import canonical_member_key

        key = canonical_member_key("john doe")
        self.assertEqual(key, key.upper())


# ---------------------------------------------------------------------------
# Fix 8 – Negative lag (transaction_date > disclosure_date)
# ---------------------------------------------------------------------------
class TestNegativeLagFilter(DatabaseTestCase):
    def _insert_negative_lag_tx(self):
        """Insert one valid + one impossible (tx_date > disclosure_date) row."""
        df = pd.DataFrame(
            [
                {
                    "doc_id": "good",
                    "member": "Alice",
                    "ticker": "AAPL",
                    "transaction_date": date(2024, 1, 10),
                    "disclosure_date": date(2024, 1, 15),
                    "transaction_type": "Purchase",
                },
                {
                    "doc_id": "bad",
                    "member": "Bob",
                    "ticker": "MSFT",
                    # OCR date swap: tx_date is AFTER disclosure_date
                    "transaction_date": date(2024, 3, 15),
                    "disclosure_date": date(2024, 1, 15),
                    "transaction_type": "Purchase",
                },
            ]
        )
        self.db.upsert_transactions(df, source="house_pdf")

    def test_get_transactions_excludes_negative_lag(self):
        self._insert_negative_lag_tx()
        result = self.db.get_transactions(2024)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["ticker"], "AAPL")

    def test_get_transactions_by_date_range_excludes_negative_lag(self):
        self._insert_negative_lag_tx()
        result = self.db.get_transactions_by_date_range(
            date(2024, 1, 1), date(2024, 12, 31)
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["ticker"], "AAPL")

    def test_row_with_null_transaction_date_not_excluded(self):
        """Rows with NULL transaction_date must not be filtered out."""
        df = pd.DataFrame(
            [
                {
                    "doc_id": "null-date",
                    "member": "Carol",
                    "ticker": "GOOG",
                    "transaction_date": None,
                    "disclosure_date": date(2024, 6, 1),
                    "transaction_type": "Purchase",
                }
            ]
        )
        self.db.upsert_transactions(df, source="house_pdf")
        result = self.db.get_transactions(2024)
        self.assertEqual(len(result), 1)

    def test_valid_row_where_dates_equal_not_excluded(self):
        """transaction_date == disclosure_date is valid and must be kept."""
        df = pd.DataFrame(
            [
                {
                    "doc_id": "same-date",
                    "member": "Dave",
                    "ticker": "TSLA",
                    "transaction_date": date(2024, 5, 1),
                    "disclosure_date": date(2024, 5, 1),
                    "transaction_type": "Sale",
                }
            ]
        )
        self.db.upsert_transactions(df, source="house_pdf")
        result = self.db.get_transactions(2024)
        self.assertEqual(len(result), 1)


if __name__ == "__main__":
    unittest.main()
