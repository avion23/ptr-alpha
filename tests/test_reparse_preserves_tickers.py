"""Regression tests: re-parsing must MERGE, not clobber, resolved fields.

Re-parsing a document deletes and re-inserts its transactions (delete then
upsert). If the fresh parse resolves a ticker/amount to NULL that a previous
parse had correctly resolved, the good data must be carried forward. These
tests exercise ``preserve_existing_fields`` and the delete+reinsert path it
protects, mirroring the style of ``tests/test_repositories.py``.
"""

import unittest
from datetime import date

import pandas as pd

from analyzer.database import Database
from analyzer.download import preserve_existing_fields

from .conftest import DatabaseTestCase


def _row(**overrides) -> dict:
    row = {
        "doc_id": "doc1",
        "member": "John Doe",
        "ticker": "SMPL",
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
    return row


def _merge_path(db: Database, df: pd.DataFrame) -> pd.DataFrame:
    """Apply the protected re-parse path: preserve, delete, reinsert."""
    df = preserve_existing_fields(df, db)
    for doc_id in df["doc_id"].unique():
        db.delete_transactions_for_doc(doc_id)
    db.upsert_transactions(df, source="house_pdf")
    return df


class TestReparsePreservesTickers(DatabaseTestCase):
    def test_existing_ticker_carried_forward_when_new_is_null(self):
        orig = pd.DataFrame([_row(ticker="SMPL")])
        self.db.upsert_transactions(orig, source="house_pdf")

        fresh = pd.DataFrame([_row(ticker=None)])
        merged = preserve_existing_fields(fresh, self.db)

        self.assertEqual(merged.at[0, "ticker"], "SMPL")

        _merge_path(self.db, fresh)
        stored = self.db.get_transactions_for_doc("doc1")
        self.assertEqual(len(stored), 1)
        self.assertEqual(stored.iloc[0]["ticker"], "SMPL")

    def test_existing_amount_raw_carried_forward_when_new_is_empty(self):
        orig = pd.DataFrame([_row(amount_raw="$1,001 - $15,000")])
        self.db.upsert_transactions(orig, source="house_pdf")

        fresh = pd.DataFrame([_row(ticker="SMPL", amount_raw=None)])
        merged = preserve_existing_fields(fresh, self.db)

        self.assertEqual(merged.at[0, "amount_raw"], "$1,001 - $15,000")

        _merge_path(self.db, fresh)
        stored = self.db.get_transactions_for_doc("doc1")
        self.assertEqual(stored.iloc[0]["amount_raw"], "$1,001 - $15,000")

    def test_new_valid_ticker_is_kept_not_downgraded(self):
        orig = pd.DataFrame([_row(ticker="SMPL")])
        self.db.upsert_transactions(orig, source="house_pdf")

        fresh = pd.DataFrame([_row(ticker="AAPL")])
        merged = preserve_existing_fields(fresh, self.db)

        self.assertEqual(merged.at[0, "ticker"], "AAPL")

        _merge_path(self.db, fresh)
        stored = self.db.get_transactions_for_doc("doc1")
        self.assertEqual(stored.iloc[0]["ticker"], "AAPL")

    def test_transactions_not_dropped_count_preserved(self):
        orig = pd.DataFrame(
            [
                _row(ticker="SMPL", transaction_type="Purchase"),
                _row(ticker="CBRL", transaction_type="Sale"),
                _row(
                    ticker="DNUT",
                    transaction_type="Purchase",
                    transaction_date=date(2024, 3, 11),
                ),
            ]
        )
        self.db.upsert_transactions(orig, source="house_pdf")

        # Fresh parse loses all tickers (the bug scenario) but keeps all rows.
        # Each row has a distinct identity so carry-forward is unambiguous.
        fresh = pd.DataFrame(
            [
                _row(ticker=None, transaction_type="Purchase"),
                _row(ticker=None, transaction_type="Sale"),
                _row(
                    ticker=None,
                    transaction_type="Purchase",
                    transaction_date=date(2024, 3, 11),
                ),
            ]
        )
        merged = preserve_existing_fields(fresh, self.db)
        self.assertEqual(len(merged), 3)

        _merge_path(self.db, fresh)
        stored = self.db.get_transactions_for_doc("doc1")
        self.assertEqual(len(stored), 3)
        self.assertSetEqual(set(stored["ticker"]), {"SMPL", "CBRL", "DNUT"})

    def test_doc_with_no_existing_rows_is_noop(self):
        fresh = pd.DataFrame([_row(ticker=None, amount_raw=None)])
        merged = preserve_existing_fields(fresh, self.db)
        self.assertTrue(
            merged.at[0, "ticker"] is None or pd.isna(merged.at[0, "ticker"])
        )

    def test_multiple_existing_rows_any_non_null_ticker_carried(self):
        orig = pd.DataFrame(
            [
                # Two rows share the same identity but only one resolved a ticker.
                _row(ticker=None, owner_code="DC"),
                _row(ticker="HOG", owner_code="DC", disclosure_date=date(2024, 3, 16)),
            ]
        )
        self.db.upsert_transactions(orig, source="house_pdf")

        fresh = pd.DataFrame([_row(ticker=None, owner_code="DC")])
        merged = preserve_existing_fields(fresh, self.db)
        self.assertEqual(merged.at[0, "ticker"], "HOG")

    def test_ambiguous_identity_does_not_misassign_ticker(self):
        # Same (member, transaction_date, transaction_type) but two different
        # tickers resolved on prior parses. A fresh parse with tickerless rows
        # must NOT mislabel one purchase as the other.
        orig = pd.DataFrame(
            [
                _row(ticker="AAPL", owner_code="DC"),
                _row(ticker="MSFT", owner_code="DC", disclosure_date=date(2024, 3, 16)),
            ]
        )
        self.db.upsert_transactions(orig, source="house_pdf")

        fresh = pd.DataFrame(
            [
                _row(ticker=None, owner_code="DC"),
                _row(ticker=None, owner_code="DC", disclosure_date=date(2024, 3, 16)),
            ]
        )
        merged = preserve_existing_fields(fresh, self.db)

        # Disagreement -> neither row gets a ticker (no silent misassignment).
        for idx in merged.index:
            self.assertTrue(
                merged.at[idx, "ticker"] is None or pd.isna(merged.at[idx, "ticker"]),
                msg=f"row {idx} got ticker {merged.at[idx, 'ticker']!r}, expected blank",
            )


if __name__ == "__main__":
    unittest.main()
