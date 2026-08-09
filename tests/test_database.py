import unittest
from datetime import date, datetime

import duckdb
import pandas as pd

from analyzer.database import Database
from analyzer.transaction_repository import SOURCE_TRANSACTION_COLUMNS
from scripts.purge_phantom_rows import count_phantom_rows, purge_phantom_rows
from .conftest import DatabaseTestCase


class TestMetadata(DatabaseTestCase):
    def test_replace_metadata_is_atomic_and_removes_stale_rows(self):
        old = pd.DataFrame(
            [
                {
                    "doc_id": "old",
                    "first_name": "Old",
                    "last_name": "Row",
                    "filing_date": datetime(2024, 1, 1),
                    "filing_type": "P",
                    "fetched_at": datetime(2024, 1, 2),
                },
                {
                    "doc_id": "other-year",
                    "first_name": "Other",
                    "last_name": "Row",
                    "filing_date": datetime(2023, 1, 1),
                    "filing_type": "P",
                    "fetched_at": datetime(2023, 1, 2),
                },
            ]
        )
        fresh = pd.DataFrame(
            [
                {
                    "doc_id": "fresh",
                    "first_name": "Fresh",
                    "last_name": "Row",
                    "filing_date": datetime(2024, 2, 1),
                    "filing_type": "P",
                    "fetched_at": datetime(2024, 2, 2),
                },
            ]
        )
        self.db.upsert_metadata(old)

        self.db.replace_metadata(2024, fresh)

        self.assertEqual(self.db.get_metadata(2024)["DocID"].tolist(), ["fresh"])
        self.assertEqual(self.db.get_metadata(2023)["DocID"].tolist(), ["other-year"])

    def test_upsert_and_get_metadata_round_trip(self):
        df = pd.DataFrame(
            [
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
                    "filing_date": datetime(2024, 6, 20, 14, 0, 0),
                    "filing_type": "F2",
                    "fetched_at": datetime(2024, 6, 21, 9, 0, 0),
                },
            ]
        )
        self.db.upsert_metadata(df)
        result = self.db.get_metadata(2024)
        self.assertEqual(len(result), 2)
        doc_ids = set(result["DocID"])
        self.assertEqual(doc_ids, {"doc1", "doc2"})
        row_doc1 = result[result["DocID"] == "doc1"].iloc[0]
        self.assertEqual(row_doc1["First"], "John")
        row_doc2 = result[result["DocID"] == "doc2"].iloc[0]
        self.assertEqual(row_doc2["Last"], "Smith")

    def test_upsert_metadata_conflict_updates(self):
        df1 = pd.DataFrame(
            [
                {
                    "doc_id": "doc1",
                    "first_name": "John",
                    "last_name": "Doe",
                    "filing_date": datetime(2024, 3, 15, 12, 0, 0),
                    "filing_type": "F1",
                    "fetched_at": datetime(2024, 3, 16, 8, 0, 0),
                },
            ]
        )
        df2 = pd.DataFrame(
            [
                {
                    "doc_id": "doc1",
                    "first_name": "Jonathan",
                    "last_name": "Doe",
                    "filing_date": datetime(2024, 3, 15, 12, 0, 0),
                    "filing_type": "F1-AMENDED",
                    "fetched_at": datetime(2024, 3, 17, 10, 0, 0),
                },
            ]
        )
        self.db.upsert_metadata(df1)
        self.db.upsert_metadata(df2)
        result = self.db.get_metadata(2024)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["First"], "Jonathan")
        self.assertEqual(result.iloc[0]["FilingType"], "F1-AMENDED")

    def test_metadata_exists_returns_true_when_present(self):
        df = pd.DataFrame(
            [
                {
                    "doc_id": "doc1",
                    "first_name": "John",
                    "last_name": "Doe",
                    "filing_date": datetime(2024, 3, 15, 12, 0, 0),
                    "filing_type": "F1",
                    "fetched_at": datetime(2024, 3, 16, 8, 0, 0),
                },
            ]
        )
        self.db.upsert_metadata(df)
        self.assertTrue(self.db.metadata_exists(2024))

    def test_metadata_exists_returns_false_when_absent(self):
        self.assertFalse(self.db.metadata_exists(2024))

    def test_clear_metadata_removes_records(self):
        df = pd.DataFrame(
            [
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
            ]
        )
        self.db.upsert_metadata(df)
        self.assertTrue(self.db.metadata_exists(2024))
        self.assertTrue(self.db.metadata_exists(2023))
        self.db.clear_metadata(2024)
        self.assertFalse(self.db.metadata_exists(2024))
        self.assertTrue(self.db.metadata_exists(2023))


class TestTransactions(DatabaseTestCase):
    def test_upsert_and_get_transactions_round_trip(self):
        df = pd.DataFrame(
            [
                {
                    "doc_id": "doc1",
                    "member": "John Doe",
                    "ticker": "AAPL",
                    "transaction_date": date(2024, 3, 10),
                    "disclosure_date": date(2024, 3, 15),
                    "transaction_type": "Buy",
                    "owner_code": "DC",
                    "amount_raw": "$1,001 - $15,000",
                    "amount_midpoint": 8000.5,
                },
                {
                    "doc_id": "doc2",
                    "member": "Jane Smith",
                    "ticker": "MSFT",
                    "transaction_date": date(2024, 5, 5),
                    "disclosure_date": date(2024, 5, 10),
                    "transaction_type": "Sell",
                    "owner_code": None,
                    "amount_raw": "$15,001 - $50,000",
                    "amount_midpoint": 32500.5,
                },
            ]
        )
        self.db.upsert_transactions(df, source="house_pdf")
        result = self.db.get_transactions(2024)
        self.assertEqual(len(result), 2)
        cols = {
            "member",
            "ticker",
            "transaction_date",
            "disclosure_date",
            "transaction_type",
            "owner_code",
            "amount_raw",
            "amount_midpoint",
        }
        self.assertTrue(cols.issubset(set(result.columns)))
        aapl = result[result["ticker"] == "AAPL"].iloc[0]
        self.assertEqual(aapl["owner_code"], "DC")
        self.assertAlmostEqual(aapl["amount_midpoint"], 8000.5)

    def test_transactions_exist_returns_true_when_present(self):
        df = pd.DataFrame(
            [
                {
                    "doc_id": "doc1",
                    "member": "John Doe",
                    "ticker": "AAPL",
                    "transaction_date": date(2024, 3, 10),
                    "disclosure_date": date(2024, 3, 15),
                    "transaction_type": "Buy",
                },
            ]
        )
        self.db.upsert_transactions(df, source="house_pdf")
        self.assertTrue(self.db.transactions_exist(2024))

    def test_transactions_exist_returns_false_when_absent(self):
        self.assertFalse(self.db.transactions_exist(2024))

    def test_upsert_preserves_buy_and_sell_same_ticker_same_date(self):
        df = pd.DataFrame(
            [
                {
                    "doc_id": "doc1",
                    "member": "John Doe",
                    "ticker": "AAPL",
                    "transaction_date": date(2024, 3, 10),
                    "disclosure_date": date(2024, 3, 15),
                    "transaction_type": "Purchase",
                },
                {
                    "doc_id": "doc1",
                    "member": "John Doe",
                    "ticker": "AAPL",
                    "transaction_date": date(2024, 3, 10),
                    "disclosure_date": date(2024, 3, 15),
                    "transaction_type": "Sale",
                },
            ]
        )
        self.db.upsert_transactions(df, source="house_pdf")
        result = self.db.get_transactions(2024)
        self.assertEqual(len(result), 2)
        types = set(result["transaction_type"].values)
        self.assertEqual(types, {"Purchase", "Sale"})

    def test_upsert_dedupes_repeated_null_ticker_rows(self):
        df = pd.DataFrame(
            [
                {
                    "doc_id": "doc-null",
                    "member": "John Doe",
                    "ticker": None,
                    "transaction_date": date(2024, 3, 10),
                    "disclosure_date": date(2024, 3, 15),
                    "transaction_type": "Purchase",
                    "amount_raw": "$1,001 - $15,000",
                    "owner_code": None,
                    "asset_description": "Corporate bond",
                }
            ]
        )
        self.db.upsert_transactions(df, source="house_pdf")
        self.db.upsert_transactions(df, source="house_pdf")

        count = self.db.conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
        ticker = self.db.conn.execute("SELECT ticker FROM transactions").fetchone()[0]
        self.assertEqual(count, 1)
        self.assertIsNone(ticker)

    def test_upsert_keeps_distinct_null_ticker_asset_descriptions(self):
        df = pd.DataFrame(
            [
                {
                    "doc_id": "doc-null-assets",
                    "member": "John Doe",
                    "ticker": None,
                    "transaction_date": date(2024, 3, 10),
                    "disclosure_date": date(2024, 3, 15),
                    "transaction_type": "Purchase",
                    "amount_raw": "$1,001 - $15,000",
                    "asset_description": "Municipal bond A",
                },
                {
                    "doc_id": "doc-null-assets",
                    "member": "John Doe",
                    "ticker": None,
                    "transaction_date": date(2024, 3, 10),
                    "disclosure_date": date(2024, 3, 15),
                    "transaction_type": "Purchase",
                    "amount_raw": "$1,001 - $15,000",
                    "asset_description": "Municipal bond B",
                },
            ]
        )
        self.db.upsert_transactions(df, source="house_pdf")

        descriptions = self.db.conn.execute(
            "SELECT asset_description FROM transactions ORDER BY asset_description"
        ).fetchall()
        self.assertEqual(
            [row[0] for row in descriptions], ["Municipal bond A", "Municipal bond B"]
        )

    def test_upsert_writes_source_and_keeps_distinct_asset_descriptions(self):
        df1 = pd.DataFrame(
            [
                {
                    "doc_id": "doc-source",
                    "member": "Jane Doe",
                    "ticker": "AAPL",
                    "transaction_date": date(2024, 4, 1),
                    "disclosure_date": date(2024, 4, 5),
                    "transaction_type": "Purchase",
                    "amount_raw": "$1,001 - $15,000",
                    "asset_description": "Apple Inc",
                }
            ]
        )
        df2 = df1.copy()
        df2["asset_description"] = "Apple Inc updated"

        self.db.upsert_transactions(df1, source="house_pdf")
        self.db.upsert_transactions(df2, source="capitol_trades")

        rows = self.db.conn.execute(
            "SELECT source, asset_description FROM transactions WHERE doc_id = 'doc-source' ORDER BY asset_description"
        ).fetchall()
        self.assertEqual(
            rows, [("house_pdf", "Apple Inc"), ("capitol_trades", "Apple Inc updated")]
        )

    def test_upsert_identical_asset_description_is_idempotent(self):
        df = pd.DataFrame(
            [
                {
                    "doc_id": "doc-idempotent-asset",
                    "member": "Jane Doe",
                    "ticker": "AAPL",
                    "transaction_date": date(2024, 4, 1),
                    "disclosure_date": date(2024, 4, 5),
                    "transaction_type": "Purchase",
                    "amount_raw": "$1,001 - $15,000",
                    "asset_description": "Apple Inc",
                }
            ]
        )

        self.db.upsert_transactions(df, source="house_pdf")
        self.db.upsert_transactions(df, source="house_pdf")

        count = self.db.conn.execute(
            "SELECT COUNT(*) FROM transactions WHERE doc_id = 'doc-idempotent-asset'"
        ).fetchone()[0]
        self.assertEqual(count, 1)

    def test_count_transactions_for_docs_returns_counts_by_doc_id(self):
        df = pd.DataFrame(
            [
                {
                    "doc_id": "doc-count-1",
                    "member": "Jane Doe",
                    "ticker": "AAPL",
                    "transaction_date": date(2024, 4, 1),
                    "disclosure_date": date(2024, 4, 5),
                    "transaction_type": "Purchase",
                    "amount_raw": "$1,001 - $15,000",
                },
                {
                    "doc_id": "doc-count-1",
                    "member": "Jane Doe",
                    "ticker": "MSFT",
                    "transaction_date": date(2024, 4, 2),
                    "disclosure_date": date(2024, 4, 5),
                    "transaction_type": "Sale",
                    "amount_raw": "$1,001 - $15,000",
                },
                {
                    "doc_id": "doc-count-2",
                    "member": "John Doe",
                    "ticker": "GOOG",
                    "transaction_date": date(2024, 4, 3),
                    "disclosure_date": date(2024, 4, 5),
                    "transaction_type": "Purchase",
                    "amount_raw": "$1,001 - $15,000",
                },
            ]
        )
        self.db.upsert_transactions(df, source="house_pdf")

        counts = self.db.count_transactions_for_docs(
            [
                "doc-count-1",
                "doc-count-2",
                "missing-doc",
            ]
        )

        self.assertEqual(counts, {"doc-count-1": 2, "doc-count-2": 1})

    def test_count_transactions_for_docs_returns_empty_for_empty_input(self):
        self.assertEqual(self.db.count_transactions_for_docs([]), {})


class TestPrices(DatabaseTestCase):
    def test_upsert_and_get_prices_round_trip(self):
        dates = pd.date_range("2024-01-01", "2024-01-04", freq="B")
        price_data = pd.DataFrame(
            {
                "AAPL": [180.0, 181.0, 182.0, 183.0],
                "MSFT": [370.0, 371.0, 372.0, 373.0],
            },
            index=dates,
        )
        self.db.upsert_prices(price_data)
        result = self.db.get_prices(
            ["AAPL", "MSFT"], date(2024, 1, 1), date(2024, 1, 5)
        )
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("AAPL", result.columns)
        self.assertIn("MSFT", result.columns)
        self.assertEqual(len(result), 4)
        self.assertAlmostEqual(result.loc[pd.Timestamp("2024-01-02"), "AAPL"], 181.0)

    def test_get_prices_empty_for_no_data(self):
        result = self.db.get_prices(["AAPL"], date(2024, 1, 1), date(2024, 1, 5))
        self.assertTrue(result.empty)

    def test_get_prices_empty_ticker_list(self):
        result = self.db.get_prices([], date(2024, 1, 1), date(2024, 1, 5))
        self.assertTrue(result.empty)

    def test_upsert_prices_empty_df_noop(self):
        df = pd.DataFrame()
        self.db.upsert_prices(df)
        result = self.db.get_prices(["AAPL"], date(2024, 1, 1), date(2024, 1, 5))
        self.assertTrue(result.empty)

    def test_upsert_prices_conflict_updates(self):
        dates1 = pd.date_range("2024-01-01", periods=2, freq="B")
        prices1 = pd.DataFrame({"AAPL": [100.0, 101.0]}, index=dates1)
        self.db.upsert_prices(prices1)

        dates2 = pd.date_range("2024-01-01", periods=2, freq="B")
        prices2 = pd.DataFrame({"AAPL": [200.0, 101.0]}, index=dates2)
        self.db.upsert_prices(prices2)

        result = self.db.get_prices(["AAPL"], date(2024, 1, 1), date(2024, 1, 3))
        self.assertAlmostEqual(result.loc[pd.Timestamp("2024-01-01"), "AAPL"], 200.0)
        self.assertAlmostEqual(result.loc[pd.Timestamp("2024-01-02"), "AAPL"], 101.0)


class TestGetMissingPriceData(DatabaseTestCase):
    def test_missing_tickers_returned_when_no_data(self):
        missing_tickers, missing_dates = self.db.get_missing_price_data(
            ["AAPL", "MSFT"], date(2024, 1, 1), date(2024, 1, 5)
        )
        self.assertEqual(set(missing_tickers), {"AAPL", "MSFT"})
        self.assertTrue(len(missing_dates) > 0)

    def test_no_missing_when_all_data_present(self):
        dates = pd.bdate_range("2024-01-01", "2024-01-05")
        prices = pd.DataFrame({"AAPL": range(len(dates))}, index=dates)
        self.db.upsert_prices(prices)
        missing_tickers, missing_dates = self.db.get_missing_price_data(
            ["AAPL"], date(2024, 1, 1), date(2024, 1, 5)
        )
        self.assertEqual(missing_tickers, [])
        self.assertEqual(len(missing_dates), 0)

    def test_partial_missing_dates(self):
        # AAPL has only Jan 1-2 prices but the range is Jan 1-5.
        # Fix 4: per-ticker gap detection — AAPL must appear in missing_tickers
        # (old behavior incorrectly returned [] because the global date union was used).
        dates = pd.bdate_range("2024-01-01", "2024-01-02")
        prices = pd.DataFrame({"AAPL": [100.0, 101.0]}, index=dates)
        self.db.upsert_prices(prices)
        missing_tickers, missing_dates = self.db.get_missing_price_data(
            ["AAPL"], date(2024, 1, 1), date(2024, 1, 5)
        )
        self.assertIn("AAPL", missing_tickers)
        self.assertTrue(len(missing_dates) > 0)

    def test_mixed_missing_tickers_and_dates(self):
        dates = pd.bdate_range("2024-01-01", "2024-01-05")
        prices = pd.DataFrame({"AAPL": range(len(dates))}, index=dates)
        self.db.upsert_prices(prices)
        missing_tickers, missing_dates = self.db.get_missing_price_data(
            ["AAPL", "MSFT"], date(2024, 1, 1), date(2024, 1, 5)
        )
        self.assertIn("MSFT", missing_tickers)
        self.assertNotIn("AAPL", missing_tickers)


class TestGetEntryPrices(DatabaseTestCase):
    def test_asof_join_returns_price_on_disclosure_date(self):
        dates = pd.bdate_range("2024-01-01", "2024-01-10")
        aapl_prices = [150.0 + i for i in range(len(dates))]
        prices = pd.DataFrame({"AAPL": aapl_prices}, index=dates)
        self.db.upsert_prices(prices)

        tx = pd.DataFrame(
            [
                {
                    "doc_id": "doc-tx1",
                    "member": "John Doe",
                    "ticker": "AAPL",
                    "transaction_date": date(2024, 1, 5),
                    "disclosure_date": date(2024, 1, 5),
                    "transaction_type": "Buy",
                    "owner_code": "DC",
                    "amount_midpoint": 8000.5,
                },
            ]
        )
        self.db.upsert_transactions(tx, source="house_pdf")

        result = self.db.get_entry_prices(["AAPL"], date(2024, 1, 1), date(2024, 1, 10))
        self.assertEqual(len(result), 1)
        disclosure_jan5 = pd.Timestamp("2024-01-05")
        idx_5 = list(dates).index(disclosure_jan5)
        expected_price = aapl_prices[idx_5]
        self.assertAlmostEqual(result.iloc[0]["entry_price"], expected_price)
        self.assertEqual(result.iloc[0]["owner_code"], "DC")
        self.assertAlmostEqual(result.iloc[0]["amount_midpoint"], 8000.5)

    def test_asof_join_returns_prior_price_when_no_exact_match(self):
        dates = pd.bdate_range("2024-01-01", "2024-01-04")
        aapl_prices = [150.0, 151.0, 152.0, 153.0]
        prices = pd.DataFrame({"AAPL": aapl_prices}, index=dates)
        self.db.upsert_prices(prices)

        tx = pd.DataFrame(
            [
                {
                    "doc_id": "doc-tx2",
                    "member": "Jane Smith",
                    "ticker": "AAPL",
                    "transaction_date": date(2024, 1, 6),
                    "disclosure_date": date(2024, 1, 7),
                    "transaction_type": "Sell",
                },
            ]
        )
        self.db.upsert_transactions(tx, source="house_pdf")

        result = self.db.get_entry_prices(["AAPL"], date(2024, 1, 1), date(2024, 1, 10))
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result.iloc[0]["entry_price"], 153.0)

    def test_get_entry_prices_empty_tickers(self):
        result = self.db.get_entry_prices([], date(2024, 1, 1), date(2024, 1, 10))
        self.assertTrue(result.empty)

    def test_get_entry_prices_no_transactions(self):
        dates = pd.bdate_range("2024-01-01", "2024-01-10")
        prices = pd.DataFrame({"AAPL": range(len(dates))}, index=dates)
        self.db.upsert_prices(prices)
        result = self.db.get_entry_prices(["AAPL"], date(2024, 1, 1), date(2024, 1, 10))
        self.assertTrue(result.empty)

    def test_get_entry_prices_filters_by_date_range(self):
        dates = pd.bdate_range("2024-01-01", "2024-01-10")
        prices = pd.DataFrame({"AAPL": range(len(dates))}, index=dates)
        self.db.upsert_prices(prices)

        tx = pd.DataFrame(
            [
                {
                    "doc_id": "doc-tx3",
                    "member": "John Doe",
                    "ticker": "AAPL",
                    "transaction_date": date(2024, 1, 5),
                    "disclosure_date": date(2024, 1, 5),
                    "transaction_type": "Buy",
                },
                {
                    "doc_id": "doc-tx4",
                    "member": "John Doe",
                    "ticker": "AAPL",
                    "transaction_date": date(2024, 1, 15),
                    "disclosure_date": date(2024, 1, 15),
                    "transaction_type": "Buy",
                },
            ]
        )
        self.db.upsert_transactions(tx, source="house_pdf")

        result = self.db.get_entry_prices(["AAPL"], date(2024, 1, 1), date(2024, 1, 10))
        self.assertEqual(len(result), 1)

    def test_get_entry_prices_filters_stale_prices(self):
        # Transaction disclosure is 2024-06-01, last price before that is 2024-01-05
        # That's ~147 days stale — should be filtered with max_staleness_days=30
        early_prices = pd.bdate_range("2024-01-01", "2024-01-10")
        early_price_data = pd.DataFrame(
            {"AAPL": [100.0 + i for i in range(len(early_prices))]}, index=early_prices
        )
        self.db.upsert_prices(early_price_data)

        tx = pd.DataFrame(
            [
                {
                    "doc_id": "doc-stale",
                    "member": "Alice",
                    "ticker": "AAPL",
                    "transaction_date": date(2024, 5, 25),
                    "disclosure_date": date(2024, 6, 1),
                    "transaction_type": "Purchase",
                }
            ]
        )
        self.db.upsert_transactions(tx, source="house_pdf")

        result = self.db.get_entry_prices(
            ["AAPL"], date(2024, 6, 1), date(2024, 6, 1), max_staleness_days=30
        )
        self.assertTrue(result.empty)

    def test_get_entry_prices_keeps_fresh_prices(self):
        dates = pd.bdate_range("2024-01-01", "2024-01-31")
        price_data = pd.DataFrame(
            {"AAPL": [100.0 + i for i in range(len(dates))]}, index=dates
        )
        self.db.upsert_prices(price_data)

        tx = pd.DataFrame(
            [
                {
                    "doc_id": "doc-fresh",
                    "member": "Alice",
                    "ticker": "AAPL",
                    "transaction_date": date(2024, 1, 25),
                    "disclosure_date": date(2024, 1, 28),
                    "transaction_type": "Purchase",
                }
            ]
        )
        self.db.upsert_transactions(tx, source="house_pdf")

        result = self.db.get_entry_prices(
            ["AAPL"], date(2024, 1, 28), date(2024, 1, 28), max_staleness_days=30
        )
        self.assertEqual(len(result), 1)
        self.assertIsNotNone(result.iloc[0]["entry_price"])


class TestDeleteTransactionsForDoc(DatabaseTestCase):
    def test_delete_removes_only_that_docs_rows(self):
        df = pd.DataFrame(
            [
                {
                    "doc_id": "doc1",
                    "member": "John Doe",
                    "ticker": "AAPL",
                    "transaction_date": date(2024, 3, 10),
                    "disclosure_date": date(2024, 3, 15),
                    "transaction_type": "Purchase",
                },
                {
                    "doc_id": "doc2",
                    "member": "Jane Smith",
                    "ticker": "GOOG",
                    "transaction_date": date(2024, 4, 1),
                    "disclosure_date": date(2024, 4, 5),
                    "transaction_type": "Sale",
                },
            ]
        )
        self.db.upsert_transactions(df, source="house_pdf")
        self.assertEqual(len(self.db.get_transactions(2024)), 2)

        self.db.delete_transactions_for_doc("doc1")
        result = self.db.get_transactions(2024)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["ticker"], "GOOG")

    def test_delete_nonexistent_doc_is_noop(self):
        df = pd.DataFrame(
            [
                {
                    "doc_id": "doc1",
                    "member": "John Doe",
                    "ticker": "AAPL",
                    "transaction_date": date(2024, 3, 10),
                    "disclosure_date": date(2024, 3, 15),
                    "transaction_type": "Purchase",
                },
            ]
        )
        self.db.upsert_transactions(df, source="house_pdf")
        self.db.delete_transactions_for_doc("nonexistent")
        self.assertEqual(len(self.db.get_transactions(2024)), 1)


class TestParseRunsTable(DatabaseTestCase):
    def test_upsert_parse_run_zero_rows_status(self):
        self.db.upsert_parse_run(
            doc_id="doc2",
            year=2024,
            parser_version="v2",
            status="zero_rows",
            engines_attempted="lattice,stream,ocr",
            raw_row_count=0,
            transaction_count=0,
        )
        result = self.db.conn.execute(
            "SELECT status FROM pdf_parse_runs WHERE doc_id = 'doc2'"
        ).fetchone()
        self.assertEqual(result[0], "zero_rows")

    def test_upsert_parse_run_error_status(self):
        self.db.upsert_parse_run(
            doc_id="doc3",
            year=2024,
            parser_version="v2",
            status="error",
            engines_attempted="lattice,stream,ocr",
            raw_row_count=0,
            transaction_count=0,
            error_message="PDFTextExtractionNotAllowed",
        )
        result = self.db.conn.execute(
            "SELECT error_message FROM pdf_parse_runs WHERE doc_id = 'doc3'"
        ).fetchone()
        self.assertEqual(result[0], "PDFTextExtractionNotAllowed")


class TestPhantomPurge(DatabaseTestCase):
    def _insert_transaction(self, doc_id, ticker, asset_description):
        self.db.conn.execute(
            """
            INSERT INTO transactions (
                doc_id, member, ticker, transaction_date, disclosure_date,
                transaction_type, owner_code, amount_raw, asset_description, source
            ) VALUES (?, 'John Doe', ?, DATE '2024-03-10', DATE '2024-03-15',
                      'Purchase', '', '$1,001 - $15,000', ?, 'house_pdf')
        """,
            [doc_id, ticker, asset_description],
        )

    def test_purge_script_dry_run_counts_and_execute_deletes_duplicates(self):
        self._insert_transaction("doc-null-dupe", None, "Bond A")
        self._insert_transaction("doc-null-dupe", None, "Bond A")
        self._insert_transaction("doc-null-dupe", None, "Bond B")
        self._insert_transaction("doc-ticker", "AAPL", "Apple")

        counts = count_phantom_rows(self.db.conn)
        self.assertEqual(counts, {True: 1})

        stats = purge_phantom_rows(self.db.conn)
        self.assertEqual(stats, {"before": 4, "deleted": 1, "after": 3})
        self.assertEqual(count_phantom_rows(self.db.conn), {})
        remaining = self.db.conn.execute(
            "SELECT COUNT(*) FROM transactions"
        ).fetchone()[0]
        self.assertEqual(remaining, 3)


class TestSourceReports(DatabaseTestCase):
    COLUMNS = [
        "ingestion_generation",
        "chamber",
        "source_record_id",
        "report_path",
        "member",
        "official_filing_date",
        "outcome",
        "artifact_sha256",
        "landing_sha256",
        "paper_artifact_url",
        "paper_artifact_sha256",
        "error_message",
        "raw_row_count",
        "accepted_row_count",
        "rejected_row_count",
    ]

    @classmethod
    def reports(cls, *rows):
        return pd.DataFrame(rows, columns=cls.COLUMNS)

    @staticmethod
    def transaction(**overrides):
        row = {
            "doc_id": "11111111-1111-4111-8111-111111111111",
            "chamber": "senate",
            "source_record_id": "11111111-1111-4111-8111-111111111111",
            "source_row_id": "table:000001",
            "source_report_path": (
                "/search/view/ptr/11111111-1111-4111-8111-111111111111/"
            ),
            "member": "Jane Doe",
            "member_key": "JANE DOE",
            "chamber_member_key": "senate:JANE DOE",
            "ticker": "AAPL",
            "raw_ticker": "AAPL",
            "ticker_candidate": None,
            "transaction_date": date(2026, 7, 1),
            "disclosure_date": date(2026, 7, 10),
            "official_filing_date": date(2026, 7, 10),
            "available_date": date(2026, 7, 10),
            "notification_date": None,
            "transaction_type": "Purchase",
            "raw_transaction_subtype": "Purchase",
            "owner_code": "Self",
            "raw_owner": "Self",
            "amount_raw": "$1,001 - $15,000",
            "amount_midpoint": 8000.5,
            "instrument_type": "stock",
            "raw_asset_class": "Stock",
            "strike_price": None,
            "expiry_date": None,
            "asset_description": "Apple Inc.",
            "raw_asset_description": "Apple Inc.",
            "ticker_origin": "official",
            "amends_source_record_id": None,
            "ingestion_generation": "gen-1",
            "artifact_sha256": "a" * 64,
        }
        row.update(overrides)
        return row

    def test_round_trip_and_reconciliation_retain_zero_transaction_outcomes(self):
        parsed_sha = "a" * 64
        reports = self.reports(
            (
                "refresh-2026-08-09",
                "house",
                "1001",
                "2026/1001.pdf",
                "Jane Doe",
                date(2026, 8, 1),
                "parsed",
                parsed_sha,
                parsed_sha,
                None,
                None,
                None,
                1,
                1,
                0,
            ),
            (
                "refresh-2026-08-09",
                "house",
                "1002",
                "2026/1002.pdf",
                "John Doe",
                date(2026, 8, 2),
                "paper_only",
                "b" * 64,
                "b" * 64,
                "https://efdsearch.senate.gov/media/1002.pdf",
                "b" * 64,
                None,
                0,
                0,
                0,
            ),
            (
                "refresh-2026-08-09",
                "house",
                "1003",
                "2026/1003.pdf",
                "Alex Doe",
                date(2026, 8, 3),
                "parsed",
                "c" * 64,
                "c" * 64,
                None,
                None,
                None,
                1,
                1,
                0,
            ),
        )

        self.db.replace_source_reports(
            "refresh-2026-08-09", "house_pdf", "house", reports
        )

        stored = self.db.get_source_reports("refresh-2026-08-09", "house_pdf", "house")
        self.assertEqual(
            stored.columns.tolist(),
            ["ingestion_generation", "source", *self.COLUMNS[1:]],
        )
        self.assertEqual(set(stored["source"]), {"house_pdf"})
        self.assertEqual(stored["source_record_id"].tolist(), ["1001", "1002", "1003"])
        self.assertEqual(stored.iloc[0]["artifact_sha256"], parsed_sha)
        self.assertEqual(stored.iloc[0]["outcome"], "parsed")
        self.assertEqual(stored.iloc[1]["outcome"], "paper_only")
        self.assertEqual(stored.iloc[2]["outcome"], "parsed")
        self.assertEqual(stored.iloc[0]["landing_sha256"], parsed_sha)
        self.assertEqual(stored.iloc[1]["paper_artifact_sha256"], "b" * 64)
        self.assertEqual(
            self.db.get_source_report_reconciliation(
                "refresh-2026-08-09", "house_pdf", "house"
            ),
            {
                "found": 3,
                "parsed": 2,
                "paper_only": 1,
                "unavailable": 0,
                "failed": 0,
            },
        )

    def test_replacement_is_scoped_to_generation_and_chamber(self):
        def one(generation, chamber, source_record_id):
            return self.reports(
                (
                    generation,
                    chamber,
                    source_record_id,
                    f"{source_record_id}.pdf",
                    "Member",
                    date(2026, 8, 1),
                    "parsed",
                    source_record_id[0] * 64,
                    source_record_id[0] * 64,
                    None,
                    None,
                    None,
                    1,
                    1,
                    0,
                )
            )

        self.db.replace_source_reports(
            "gen-1", "house_pdf", "house", one("gen-1", "house", "a1")
        )
        self.db.replace_source_reports(
            "gen-1", "house_pdf", "senate", one("gen-1", "senate", "b1")
        )
        self.db.replace_source_reports(
            "gen-2", "house_pdf", "house", one("gen-2", "house", "c1")
        )

        self.db.replace_source_reports(
            "gen-1", "house_pdf", "house", one("gen-1", "house", "d1")
        )

        self.assertEqual(
            self.db.get_source_reports("gen-1", "house_pdf", "house")[
                "source_record_id"
            ].tolist(),
            ["d1"],
        )
        self.assertEqual(
            self.db.get_source_reports("gen-1", "house_pdf", "senate")[
                "source_record_id"
            ].tolist(),
            ["b1"],
        )
        self.assertEqual(
            self.db.get_source_reports("gen-2", "house_pdf", "house")[
                "source_record_id"
            ].tolist(),
            ["c1"],
        )

    def test_duplicate_source_record_ids_are_rejected_without_replacement(self):
        original = self.reports(
            (
                "gen-1",
                "house",
                "original",
                "original.pdf",
                "Member",
                date(2026, 8, 1),
                "parsed",
                "a" * 64,
                "a" * 64,
                None,
                None,
                None,
                1,
                1,
                0,
            )
        )
        self.db.replace_source_reports("gen-1", "house_pdf", "house", original)
        duplicate = pd.concat(
            [
                original.assign(source_record_id="duplicate"),
                original.assign(source_record_id="duplicate"),
            ],
            ignore_index=True,
        )

        with self.assertRaisesRegex(ValueError, "duplicate source_record_id"):
            self.db.replace_source_reports("gen-1", "house_pdf", "house", duplicate)

        stored = self.db.get_source_reports("gen-1", "house_pdf", "house")
        self.assertEqual(stored["source_record_id"].tolist(), ["original"])

    def test_failed_or_unclassified_inventory_cannot_commit(self):
        failed = self.reports(
            (
                "gen-1",
                "house",
                "failed-report",
                None,
                "Member",
                date(2026, 8, 1),
                "failed",
                None,
                None,
                None,
                None,
                "parser failed",
                0,
                0,
                0,
            )
        )
        with self.assertRaisesRegex(ValueError, "requires unavailable=0 and failed=0"):
            self.db.replace_source_reports("gen-1", "house_pdf", "house", failed)

        unavailable = failed.assign(
            source_record_id="unavailable-report",
            outcome="unavailable",
            error_message="artifact unavailable",
        )
        with self.assertRaisesRegex(ValueError, "requires unavailable=0 and failed=0"):
            self.db.replace_source_reports("gen-1", "house_pdf", "house", unavailable)

        bad_row_counts = failed.assign(
            source_record_id="bad-counts",
            outcome="parsed",
            artifact_sha256="a" * 64,
            error_message=None,
            raw_row_count=2,
            accepted_row_count=1,
            rejected_row_count=0,
        )
        with self.assertRaisesRegex(ValueError, "parsed reports require"):
            self.db.replace_source_reports(
                "gen-1", "house_pdf", "house", bad_row_counts
            )

        unknown = failed.assign(
            source_record_id="unknown-report",
            outcome="not_classified",
            error_message=None,
        )
        with self.assertRaisesRegex(ValueError, "reconciliation failed"):
            self.db.replace_source_reports("gen-1", "house_pdf", "house", unknown)

        self.assertEqual(
            self.db.get_source_report_reconciliation("gen-1", "house_pdf", "house")[
                "found"
            ],
            0,
        )

    def test_insert_failure_rolls_back_delete(self):
        original = self.reports(
            (
                "gen-1",
                "house",
                "original",
                "original.pdf",
                "Member",
                date(2026, 8, 1),
                "parsed",
                "a" * 64,
                "a" * 64,
                None,
                None,
                None,
                1,
                1,
                0,
            )
        )
        self.db.replace_source_reports("gen-1", "house_pdf", "house", original)
        invalid = original.assign(
            source_record_id="replacement",
            official_filing_date="not-a-date",
        )

        with self.assertRaisesRegex(ValueError, "official_filing_date"):
            self.db.replace_source_reports("gen-1", "house_pdf", "house", invalid)

        stored = self.db.get_source_reports("gen-1", "house_pdf", "house")
        self.assertEqual(stored["source_record_id"].tolist(), ["original"])

    def test_persist_source_refresh_preserves_distinct_source_rows(self):
        transactions = pd.DataFrame(
            [
                self.transaction(source_row_id="table:000001"),
                self.transaction(source_row_id="table:000002"),
            ],
            columns=SOURCE_TRANSACTION_COLUMNS,
        )
        reports = pd.DataFrame(
            [
                {
                    "ingestion_generation": "gen-1",
                    "chamber": "senate",
                    "source_record_id": "11111111-1111-4111-8111-111111111111",
                    "report_path": "/search/view/ptr/11111111-1111-4111-8111-111111111111/",
                    "member": "Jane Doe",
                    "official_filing_date": date(2026, 7, 10),
                    "outcome": "parsed",
                    "artifact_sha256": "a" * 64,
                    "landing_sha256": "a" * 64,
                    "paper_artifact_url": None,
                    "error_message": None,
                    "raw_row_count": 2,
                    "accepted_row_count": 2,
                    "rejected_row_count": 0,
                },
                {
                    "ingestion_generation": "gen-1",
                    "chamber": "senate",
                    "source_record_id": "22222222-2222-4222-8222-222222222222",
                    "report_path": "/search/view/ptr/22222222-2222-4222-8222-222222222222/",
                    "member": "John Doe",
                    "official_filing_date": date(2026, 7, 11),
                    "outcome": "paper_only",
                    "artifact_sha256": "b" * 64,
                    "landing_sha256": "b" * 64,
                    "paper_artifact_url": "https://efdsearch.senate.gov/media/paper.pdf",
                    "paper_artifact_sha256": "b" * 64,
                    "error_message": None,
                    "raw_row_count": 0,
                    "accepted_row_count": 0,
                    "rejected_row_count": 0,
                },
            ],
            columns=self.COLUMNS,
        )

        inserted = self.db.persist_source_refresh(
            transactions=transactions,
            reports=reports,
            source="senate_efd",
            chamber="senate",
            ingestion_generation="gen-1",
        )

        self.assertEqual(inserted, 2)
        stored_rows = self.db.conn.execute(
            """
            SELECT source, chamber, source_record_id, source_row_id,
                   ingestion_generation, artifact_sha256, raw_owner,
                   available_date, official_filing_date, raw_ticker
            FROM transactions
            WHERE source = 'senate_efd'
            ORDER BY source_row_id
            """
        ).fetchall()
        self.assertEqual(
            [row[3] for row in stored_rows], ["table:000001", "table:000002"]
        )
        self.assertTrue(all(row[0] == "senate_efd" for row in stored_rows))
        self.assertTrue(all(row[1] == "senate" for row in stored_rows))
        self.assertTrue(all(row[4] == "gen-1" for row in stored_rows))
        self.assertTrue(all(row[5] == "a" * 64 for row in stored_rows))
        self.assertTrue(all(row[6] == "Self" for row in stored_rows))
        self.assertTrue(all(row[9] == "AAPL" for row in stored_rows))
        stored_reports = self.db.get_source_reports("gen-1", "senate_efd", "senate")
        self.assertEqual(
            stored_reports["source_record_id"].tolist(),
            [
                "11111111-1111-4111-8111-111111111111",
                "22222222-2222-4222-8222-222222222222",
            ],
        )
        self.assertEqual(stored_reports.iloc[1]["outcome"], "paper_only")
        self.assertEqual(stored_reports.iloc[1]["accepted_row_count"], 0)

        read_rows = self.db.get_transactions_by_date_range(
            date(2026, 7, 1), date(2026, 7, 31)
        )
        self.assertEqual(len(read_rows), 2)
        self.assertEqual(set(read_rows["source"]), {"senate_efd"})
        self.assertEqual(set(read_rows["chamber"]), {"senate"})
        self.assertEqual(
            set(read_rows["available_date"]), {pd.Timestamp(date(2026, 7, 10))}
        )

        other_transactions = pd.DataFrame(
            [
                self.transaction(
                    source_row_id="table:000001",
                    ingestion_generation="gen-1",
                )
            ],
            columns=SOURCE_TRANSACTION_COLUMNS,
        )
        other_reports = reports[
            reports["source_record_id"].eq("11111111-1111-4111-8111-111111111111")
        ].copy()
        other_reports["ingestion_generation"] = "gen-1"
        other_reports["raw_row_count"] = 1
        other_reports["accepted_row_count"] = 1
        self.db.persist_source_refresh(
            transactions=other_transactions,
            reports=other_reports,
            source="other_official_source",
            chamber="senate",
            ingestion_generation="gen-1",
        )

        replacement_transactions = pd.DataFrame(
            [
                self.transaction(
                    source_row_id="table:000003", ingestion_generation="gen-2"
                )
            ],
            columns=SOURCE_TRANSACTION_COLUMNS,
        )
        replacement_reports = reports[
            reports["source_record_id"].eq("11111111-1111-4111-8111-111111111111")
        ].copy()
        replacement_reports["ingestion_generation"] = "gen-2"
        replacement_reports["raw_row_count"] = 1
        replacement_reports["accepted_row_count"] = 1
        self.assertEqual(
            self.db.persist_source_refresh(
                transactions=replacement_transactions,
                reports=replacement_reports,
                source="senate_efd",
                chamber="senate",
                ingestion_generation="gen-2",
            ),
            1,
        )
        replaced_rows = self.db.conn.execute(
            """
            SELECT source_row_id FROM transactions
            WHERE source = 'senate_efd' AND chamber = 'senate'
            """
        ).fetchall()
        self.assertEqual(replaced_rows, [("table:000003",)])
        self.assertTrue(
            self.db.get_source_reports("gen-1", "senate_efd", "senate").empty
        )
        self.assertEqual(
            self.db.get_source_reports("gen-2", "senate_efd", "senate")[
                "source_record_id"
            ].tolist(),
            ["11111111-1111-4111-8111-111111111111"],
        )
        current_rows = self.db.get_transactions_by_date_range(
            date(2026, 7, 1), date(2026, 7, 31)
        )
        self.assertEqual(len(current_rows[current_rows["source"].eq("senate_efd")]), 1)
        self.assertEqual(
            self.db.conn.execute(
                """
                SELECT COUNT(*) FROM transactions
                WHERE source = 'other_official_source' AND chamber = 'senate'
                """
            ).fetchone()[0],
            1,
        )
        self.assertEqual(
            self.db.get_source_reports("gen-1", "other_official_source", "senate")[
                "source_record_id"
            ].tolist(),
            ["11111111-1111-4111-8111-111111111111"],
        )

    def test_persist_source_refresh_retains_all_zero_transaction_reports(self):
        reports = pd.DataFrame(
            [
                {
                    "ingestion_generation": "empty-gen",
                    "chamber": "senate",
                    "source_record_id": "11111111-1111-4111-8111-111111111111",
                    "report_path": "/search/view/ptr/11111111-1111-4111-8111-111111111111/",
                    "member": "Jane Doe",
                    "official_filing_date": date(2026, 7, 10),
                    "outcome": "paper_only",
                    "artifact_sha256": "c" * 64,
                    "landing_sha256": "c" * 64,
                    "paper_artifact_url": "https://efdsearch.senate.gov/media/zero.pdf",
                    "paper_artifact_sha256": "d" * 64,
                    "error_message": None,
                    "raw_row_count": 0,
                    "accepted_row_count": 0,
                    "rejected_row_count": 0,
                }
            ],
            columns=self.COLUMNS,
        )
        transactions = pd.DataFrame(columns=SOURCE_TRANSACTION_COLUMNS)

        inserted = self.db.persist_source_refresh(
            transactions=transactions,
            reports=reports,
            source="senate_efd",
            chamber="senate",
            ingestion_generation="empty-gen",
        )

        self.assertEqual(inserted, 0)
        self.assertEqual(
            self.db.get_source_reports("empty-gen", "senate_efd", "senate")[
                "outcome"
            ].tolist(),
            ["paper_only"],
        )
        self.assertEqual(
            self.db.get_source_report_reconciliation(
                "empty-gen", "senate_efd", "senate"
            ),
            {
                "found": 1,
                "parsed": 0,
                "paper_only": 1,
                "unavailable": 0,
                "failed": 0,
            },
        )

    def test_persist_source_refresh_rolls_back_both_frames(self):
        original_transactions = pd.DataFrame(
            [self.transaction()], columns=SOURCE_TRANSACTION_COLUMNS
        )
        original_reports = pd.DataFrame(
            [
                {
                    "ingestion_generation": "gen-1",
                    "chamber": "senate",
                    "source_record_id": "11111111-1111-4111-8111-111111111111",
                    "report_path": "/search/view/ptr/11111111-1111-4111-8111-111111111111/",
                    "member": "Jane Doe",
                    "official_filing_date": date(2026, 7, 10),
                    "outcome": "parsed",
                    "artifact_sha256": "a" * 64,
                    "landing_sha256": "a" * 64,
                    "paper_artifact_url": None,
                    "error_message": None,
                    "raw_row_count": 1,
                    "accepted_row_count": 1,
                    "rejected_row_count": 0,
                }
            ],
            columns=self.COLUMNS,
        )
        self.db.persist_source_refresh(
            transactions=original_transactions,
            reports=original_reports,
            source="senate_efd",
            chamber="senate",
            ingestion_generation="gen-1",
        )
        invalid_transactions = original_transactions.assign(
            source_row_id="replacement", amount_midpoint="not-a-number"
        )
        replacement_reports = original_reports.copy()

        with self.assertRaises(duckdb.ConversionException):
            self.db.persist_source_refresh(
                transactions=invalid_transactions,
                reports=replacement_reports,
                source="senate_efd",
                chamber="senate",
                ingestion_generation="gen-1",
            )

        stored_transaction = self.db.conn.execute(
            """
            SELECT source_row_id FROM transactions
            WHERE source = 'senate_efd' AND chamber = 'senate'
              AND ingestion_generation = 'gen-1'
            """
        ).fetchone()
        self.assertEqual(stored_transaction[0], "table:000001")
        stored_report = self.db.get_source_reports(
            "gen-1", "senate_efd", "senate"
        ).iloc[0]
        self.assertEqual(
            stored_report["report_path"],
            "/search/view/ptr/11111111-1111-4111-8111-111111111111/",
        )

    def test_persist_source_refresh_rejects_mapping_and_count_contradictions(self):
        transactions = pd.DataFrame(
            [self.transaction()], columns=SOURCE_TRANSACTION_COLUMNS
        )
        reports = pd.DataFrame(
            [
                {
                    "ingestion_generation": "gen-1",
                    "chamber": "senate",
                    "source_record_id": "11111111-1111-4111-8111-111111111111",
                    "report_path": "/search/view/ptr/11111111-1111-4111-8111-111111111111/",
                    "member": "Jane Doe",
                    "official_filing_date": date(2026, 7, 10),
                    "outcome": "parsed",
                    "artifact_sha256": "a" * 64,
                    "landing_sha256": "a" * 64,
                    "paper_artifact_url": None,
                    "error_message": None,
                    "raw_row_count": 1,
                    "accepted_row_count": 1,
                    "rejected_row_count": 0,
                }
            ],
            columns=self.COLUMNS,
        )

        with self.assertRaisesRegex(ValueError, "duplicate source row identities"):
            self.db.persist_source_refresh(
                transactions=pd.concat([transactions, transactions], ignore_index=True),
                reports=reports,
                source="senate_efd",
                chamber="senate",
                ingestion_generation="gen-1",
            )
        with self.assertRaisesRegex(ValueError, "artifact hash does not match"):
            self.db.persist_source_refresh(
                transactions=transactions.assign(artifact_sha256="b" * 64),
                reports=reports,
                source="senate_efd",
                chamber="senate",
                ingestion_generation="gen-1",
            )
        with self.assertRaisesRegex(ValueError, "accepted transaction count"):
            self.db.persist_source_refresh(
                transactions=pd.concat(
                    [
                        transactions,
                        transactions.assign(source_row_id="table:000002"),
                    ],
                    ignore_index=True,
                ),
                reports=reports,
                source="senate_efd",
                chamber="senate",
                ingestion_generation="gen-1",
            )

    def test_persist_rejects_null_identity_and_bad_report_bindings(self):
        reports = pd.DataFrame(
            [
                {
                    "ingestion_generation": "gen-1",
                    "chamber": "senate",
                    "source_record_id": "11111111-1111-4111-8111-111111111111",
                    "report_path": (
                        "/search/view/ptr/11111111-1111-4111-8111-111111111111/"
                    ),
                    "member": "Jane Doe",
                    "official_filing_date": date(2026, 7, 10),
                    "outcome": "parsed",
                    "artifact_sha256": "a" * 64,
                    "landing_sha256": "a" * 64,
                    "paper_artifact_url": None,
                    "paper_artifact_sha256": None,
                    "error_message": None,
                    "raw_row_count": 1,
                    "accepted_row_count": 1,
                    "rejected_row_count": 0,
                }
            ],
            columns=self.COLUMNS,
        )
        for transaction, message in [
            (self.transaction(source_row_id=None), "provenance values are incomplete"),
            (self.transaction(source_row_id=""), "source_row_id must be"),
            (self.transaction(member="Other Member"), "member does not match"),
            (self.transaction(member_key="WRONG"), "member keys do not match"),
            (self.transaction(source_report_path="/wrong/"), "path does not match"),
            (self.transaction(available_date=date(2026, 7, 11)), "dates do not match"),
            (self.transaction(artifact_sha256="b" * 64), "landing hash"),
        ]:
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(ValueError, message),
            ):
                self.db.persist_source_refresh(
                    transactions=pd.DataFrame(
                        [transaction], columns=SOURCE_TRANSACTION_COLUMNS
                    ),
                    reports=reports,
                    source="senate_efd",
                    chamber="senate",
                    ingestion_generation="gen-1",
                )

    def test_ticker_origin_matrix_rejects_every_inconsistent_shape(self):
        bad_rows = [
            {"ticker_origin": "official", "ticker": "AAPL", "ticker_candidate": "AAPL"},
            {"ticker_origin": "official", "ticker": "AAPL1", "ticker_candidate": None},
            {
                "ticker_origin": "asset_description",
                "ticker": "BOND",
                "ticker_candidate": None,
            },
            {
                "ticker_origin": "unverified",
                "ticker": "AAPL",
                "ticker_candidate": "AAPL",
            },
            {"ticker_origin": "unverified", "ticker": None, "ticker_candidate": "NOTE"},
            {"ticker_origin": "non_equity", "ticker": "AAPL", "ticker_candidate": None},
            {"ticker_origin": "missing", "ticker": None, "ticker_candidate": "AAPL"},
            {"ticker_origin": "invalid", "ticker": None, "ticker_candidate": "AAPL"},
            {"ticker_origin": "unknown", "ticker": None, "ticker_candidate": None},
        ]
        for overrides in bad_rows:
            row = self.transaction(**overrides)
            frame = pd.DataFrame([row], columns=SOURCE_TRANSACTION_COLUMNS)
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.db._validate_ticker_origin_matrix(frame)

    def test_strict_report_outcomes_reject_partial_and_bad_paper(self):
        parsed = self.reports(
            (
                "gen-1",
                "senate",
                "record",
                "/record",
                "Member",
                date(2026, 8, 1),
                "parsed",
                "a" * 64,
                "a" * 64,
                None,
                None,
                None,
                1,
                1,
                0,
            )
        )
        with self.assertRaisesRegex(ValueError, "parsed reports require"):
            self.db.replace_source_reports(
                "gen-1",
                "senate_efd",
                "senate",
                parsed.assign(raw_row_count=2, rejected_row_count=1),
            )
        paper = parsed.assign(
            outcome="paper_only",
            raw_row_count=0,
            accepted_row_count=0,
            artifact_sha256="b" * 64,
            landing_sha256="b" * 64,
            paper_artifact_sha256="c" * 64,
            paper_artifact_url="https://evil.test/media/paper.pdf",
        )
        with self.assertRaisesRegex(ValueError, "official Senate paper URL"):
            self.db.replace_source_reports("gen-1", "senate_efd", "senate", paper)
        with self.assertRaisesRegex(ValueError, "must not set paper artifact fields"):
            self.db.replace_source_reports(
                "gen-1",
                "senate_efd",
                "senate",
                parsed.assign(paper_artifact_sha256="c" * 64),
            )

    def test_initialization_drops_legacy_v3_unique_index(self):
        self.db.conn.execute(
            """
            CREATE UNIQUE INDEX idx_tx_unique_v3 ON transactions (
                doc_id, ticker, transaction_date, member, transaction_type,
                amount_raw, owner_code, asset_description
            )
            """
        )
        self.db.conn.execute(
            """
            INSERT INTO transactions (
                doc_id, ticker, transaction_date, member, transaction_type,
                amount_raw, owner_code, asset_description
            ) VALUES
                ('legacy', 'AAPL', '2026-01-01', 'Member', 'Purchase', NULL, NULL, 'Asset'),
                ('legacy', 'AAPL', '2026-01-01', 'Member', 'Purchase', NULL, NULL, 'Asset')
            """
        )
        self.db.close()
        self.db = Database(self.db_path)
        index_count = self.db.conn.execute(
            """
            SELECT COUNT(*) FROM duckdb_indexes()
            WHERE index_name = 'idx_tx_unique_v3'
            """
        ).fetchone()[0]
        self.assertEqual(index_count, 0)

    def test_unverified_ticker_candidate_is_not_canonical(self):
        reports = pd.DataFrame(
            [
                {
                    "ingestion_generation": "gen-1",
                    "chamber": "senate",
                    "source_record_id": "11111111-1111-4111-8111-111111111111",
                    "report_path": "/search/view/ptr/11111111-1111-4111-8111-111111111111/",
                    "member": "Jane Doe",
                    "official_filing_date": date(2026, 7, 10),
                    "outcome": "parsed",
                    "artifact_sha256": "a" * 64,
                    "landing_sha256": "a" * 64,
                    "paper_artifact_sha256": None,
                    "paper_artifact_url": None,
                    "error_message": None,
                    "raw_row_count": 1,
                    "accepted_row_count": 1,
                    "rejected_row_count": 0,
                }
            ],
            columns=self.COLUMNS,
        )
        transactions = pd.DataFrame(
            [
                self.transaction(
                    ticker=None,
                    raw_ticker="BRK/B",
                    ticker_candidate="BRK.B",
                    ticker_origin="unverified",
                )
            ],
            columns=SOURCE_TRANSACTION_COLUMNS,
        )

        self.assertEqual(
            self.db.persist_source_refresh(
                transactions=transactions,
                reports=reports,
                source="senate_efd",
                chamber="senate",
                ingestion_generation="gen-1",
            ),
            1,
        )
        stored = self.db.conn.execute(
            "SELECT ticker, raw_ticker, ticker_candidate FROM transactions"
        ).fetchone()
        self.assertEqual(stored, (None, "BRK/B", "BRK.B"))

        with self.assertRaisesRegex(ValueError, "unverified ticker origin"):
            self.db.persist_source_refresh(
                transactions=transactions.assign(ticker="BRK.B"),
                reports=reports,
                source="senate_efd",
                chamber="senate",
                ingestion_generation="gen-1",
            )


if __name__ == "__main__":
    unittest.main()
