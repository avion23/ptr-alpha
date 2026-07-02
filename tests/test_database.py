import unittest
from datetime import date, datetime
from pathlib import Path

import duckdb
import pandas as pd

from analyzer.database import Database
from scripts.purge_phantom_rows import count_phantom_rows, purge_phantom_rows
from .conftest import DatabaseTestCase


class TestDatabaseSchema(DatabaseTestCase):

    def test_tables_created_on_init(self):
        tables = self.db.conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
        ).fetchall()
        table_names = {t[0] for t in tables}
        self.assertIn("metadata", table_names)
        self.assertIn("transactions", table_names)
        self.assertIn("prices", table_names)

    def test_transactions_unique_index_exists(self):
        indexes = self.db.conn.execute(
            "SELECT index_name FROM duckdb_indexes()"
        ).fetchall()
        index_names = {i[0] for i in indexes}
        self.assertIn("idx_tx_unique", index_names)


class TestMetadata(DatabaseTestCase):

    def test_upsert_and_get_metadata_round_trip(self):
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
                "filing_date": datetime(2024, 6, 20, 14, 0, 0),
                "filing_type": "F2",
                "fetched_at": datetime(2024, 6, 21, 9, 0, 0),
            },
        ])
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
        df1 = pd.DataFrame([
            {
                "doc_id": "doc1",
                "first_name": "John",
                "last_name": "Doe",
                "filing_date": datetime(2024, 3, 15, 12, 0, 0),
                "filing_type": "F1",
                "fetched_at": datetime(2024, 3, 16, 8, 0, 0),
            },
        ])
        df2 = pd.DataFrame([
            {
                "doc_id": "doc1",
                "first_name": "Jonathan",
                "last_name": "Doe",
                "filing_date": datetime(2024, 3, 15, 12, 0, 0),
                "filing_type": "F1-AMENDED",
                "fetched_at": datetime(2024, 3, 17, 10, 0, 0),
            },
        ])
        self.db.upsert_metadata(df1)
        self.db.upsert_metadata(df2)
        result = self.db.get_metadata(2024)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["First"], "Jonathan")
        self.assertEqual(result.iloc[0]["FilingType"], "F1-AMENDED")

    def test_metadata_exists_returns_true_when_present(self):
        df = pd.DataFrame([
            {
                "doc_id": "doc1",
                "first_name": "John",
                "last_name": "Doe",
                "filing_date": datetime(2024, 3, 15, 12, 0, 0),
                "filing_type": "F1",
                "fetched_at": datetime(2024, 3, 16, 8, 0, 0),
            },
        ])
        self.db.upsert_metadata(df)
        self.assertTrue(self.db.metadata_exists(2024))

    def test_metadata_exists_returns_false_when_absent(self):
        self.assertFalse(self.db.metadata_exists(2024))

    def test_clear_metadata_removes_records(self):
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
        self.db.upsert_metadata(df)
        self.assertTrue(self.db.metadata_exists(2024))
        self.assertTrue(self.db.metadata_exists(2023))
        self.db.clear_metadata(2024)
        self.assertFalse(self.db.metadata_exists(2024))
        self.assertTrue(self.db.metadata_exists(2023))


class TestTransactions(DatabaseTestCase):

    def test_upsert_and_get_transactions_round_trip(self):
        df = pd.DataFrame([
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
        ])
        self.db.upsert_transactions(df, source="house_pdf")
        result = self.db.get_transactions(2024)
        self.assertEqual(len(result), 2)
        cols = {
            "member", "ticker", "transaction_date", "disclosure_date", "transaction_type",
            "owner_code", "amount_raw", "amount_midpoint",
        }
        self.assertTrue(cols.issubset(set(result.columns)))
        aapl = result[result["ticker"] == "AAPL"].iloc[0]
        self.assertEqual(aapl["owner_code"], "DC")
        self.assertAlmostEqual(aapl["amount_midpoint"], 8000.5)

    def test_transactions_exist_returns_true_when_present(self):
        df = pd.DataFrame([
            {
                "doc_id": "doc1",
                "member": "John Doe",
                "ticker": "AAPL",
                "transaction_date": date(2024, 3, 10),
                "disclosure_date": date(2024, 3, 15),
                "transaction_type": "Buy",
            },
        ])
        self.db.upsert_transactions(df, source="house_pdf")
        self.assertTrue(self.db.transactions_exist(2024))

    def test_transactions_exist_returns_false_when_absent(self):
        self.assertFalse(self.db.transactions_exist(2024))

    def test_upsert_preserves_buy_and_sell_same_ticker_same_date(self):
        df = pd.DataFrame([
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
        ])
        self.db.upsert_transactions(df, source="house_pdf")
        result = self.db.get_transactions(2024)
        self.assertEqual(len(result), 2)
        types = set(result["transaction_type"].values)
        self.assertEqual(types, {"Purchase", "Sale"})

    def test_upsert_dedupes_repeated_null_ticker_rows(self):
        df = pd.DataFrame([{
            "doc_id": "doc-null",
            "member": "John Doe",
            "ticker": None,
            "transaction_date": date(2024, 3, 10),
            "disclosure_date": date(2024, 3, 15),
            "transaction_type": "Purchase",
            "amount_raw": "$1,001 - $15,000",
            "owner_code": None,
            "asset_description": "Corporate bond",
        }])
        self.db.upsert_transactions(df, source="house_pdf")
        self.db.upsert_transactions(df, source="house_pdf")

        count = self.db.conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
        ticker = self.db.conn.execute("SELECT ticker FROM transactions").fetchone()[0]
        self.assertEqual(count, 1)
        self.assertIsNone(ticker)

    def test_upsert_keeps_distinct_null_ticker_asset_descriptions(self):
        df = pd.DataFrame([
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
        ])
        self.db.upsert_transactions(df, source="house_pdf")

        descriptions = self.db.conn.execute(
            "SELECT asset_description FROM transactions ORDER BY asset_description"
        ).fetchall()
        self.assertEqual([row[0] for row in descriptions], ["Municipal bond A", "Municipal bond B"])

    def test_upsert_writes_source_and_conflict_preserves_existing_source(self):
        df1 = pd.DataFrame([{
            "doc_id": "doc-source",
            "member": "Jane Doe",
            "ticker": "AAPL",
            "transaction_date": date(2024, 4, 1),
            "disclosure_date": date(2024, 4, 5),
            "transaction_type": "Purchase",
            "amount_raw": "$1,001 - $15,000",
            "asset_description": "Apple Inc",
        }])
        df2 = df1.copy()
        df2["asset_description"] = "Apple Inc updated"

        self.db.upsert_transactions(df1, source="house_pdf")
        self.db.upsert_transactions(df2, source="capitol_trades")

        row = self.db.conn.execute(
            "SELECT source, asset_description FROM transactions WHERE doc_id = 'doc-source'"
        ).fetchone()
        self.assertEqual(row[0], "house_pdf")
        self.assertEqual(row[1], "Apple Inc updated")

    def test_count_transactions_for_docs_returns_counts_by_doc_id(self):
        df = pd.DataFrame([
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
        ])
        self.db.upsert_transactions(df, source="house_pdf")

        counts = self.db.count_transactions_for_docs([
            "doc-count-1",
            "doc-count-2",
            "missing-doc",
        ])

        self.assertEqual(counts, {"doc-count-1": 2, "doc-count-2": 1})

    def test_count_transactions_for_docs_returns_empty_for_empty_input(self):
        self.assertEqual(self.db.count_transactions_for_docs([]), {})


class TestPrices(DatabaseTestCase):

    def test_upsert_and_get_prices_round_trip(self):
        dates = pd.date_range("2024-01-01", "2024-01-04", freq="B")
        price_data = pd.DataFrame({
            "AAPL": [180.0, 181.0, 182.0, 183.0],
            "MSFT": [370.0, 371.0, 372.0, 373.0],
        }, index=dates)
        self.db.upsert_prices(price_data)
        result = self.db.get_prices(["AAPL", "MSFT"], date(2024, 1, 1), date(2024, 1, 5))
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

        tx = pd.DataFrame([
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
        ])
        self.db.upsert_transactions(tx, source="house_pdf")

        result = self.db.get_entry_prices(
            ["AAPL"], date(2024, 1, 1), date(2024, 1, 10)
        )
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

        tx = pd.DataFrame([
            {
                "doc_id": "doc-tx2",
                "member": "Jane Smith",
                "ticker": "AAPL",
                "transaction_date": date(2024, 1, 6),
                "disclosure_date": date(2024, 1, 7),
                "transaction_type": "Sell",
            },
        ])
        self.db.upsert_transactions(tx, source="house_pdf")

        result = self.db.get_entry_prices(
            ["AAPL"], date(2024, 1, 1), date(2024, 1, 10)
        )
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result.iloc[0]["entry_price"], 153.0)

    def test_get_entry_prices_empty_tickers(self):
        result = self.db.get_entry_prices([], date(2024, 1, 1), date(2024, 1, 10))
        self.assertTrue(result.empty)

    def test_get_entry_prices_no_transactions(self):
        dates = pd.bdate_range("2024-01-01", "2024-01-10")
        prices = pd.DataFrame({"AAPL": range(len(dates))}, index=dates)
        self.db.upsert_prices(prices)
        result = self.db.get_entry_prices(
            ["AAPL"], date(2024, 1, 1), date(2024, 1, 10)
        )
        self.assertTrue(result.empty)

    def test_get_entry_prices_filters_by_date_range(self):
        dates = pd.bdate_range("2024-01-01", "2024-01-10")
        prices = pd.DataFrame({"AAPL": range(len(dates))}, index=dates)
        self.db.upsert_prices(prices)

        tx = pd.DataFrame([
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
        ])
        self.db.upsert_transactions(tx, source="house_pdf")

        result = self.db.get_entry_prices(
            ["AAPL"], date(2024, 1, 1), date(2024, 1, 10)
        )
        self.assertEqual(len(result), 1)

    def test_get_entry_prices_filters_stale_prices(self):
        # Transaction disclosure is 2024-06-01, last price before that is 2024-01-05
        # That's ~147 days stale — should be filtered with max_staleness_days=30
        early_prices = pd.bdate_range("2024-01-01", "2024-01-10")
        early_price_data = pd.DataFrame({"AAPL": [100.0 + i for i in range(len(early_prices))]}, index=early_prices)
        self.db.upsert_prices(early_price_data)

        tx = pd.DataFrame([{
            "doc_id": "doc-stale",
            "member": "Alice",
            "ticker": "AAPL",
            "transaction_date": date(2024, 5, 25),
            "disclosure_date": date(2024, 6, 1),
            "transaction_type": "Purchase",
        }])
        self.db.upsert_transactions(tx, source="house_pdf")

        result = self.db.get_entry_prices(
            ["AAPL"], date(2024, 6, 1), date(2024, 6, 1), max_staleness_days=30
        )
        self.assertTrue(result.empty)

    def test_get_entry_prices_keeps_fresh_prices(self):
        dates = pd.bdate_range("2024-01-01", "2024-01-31")
        price_data = pd.DataFrame({"AAPL": [100.0 + i for i in range(len(dates))]}, index=dates)
        self.db.upsert_prices(price_data)

        tx = pd.DataFrame([{
            "doc_id": "doc-fresh",
            "member": "Alice",
            "ticker": "AAPL",
            "transaction_date": date(2024, 1, 25),
            "disclosure_date": date(2024, 1, 28),
            "transaction_type": "Purchase",
        }])
        self.db.upsert_transactions(tx, source="house_pdf")

        result = self.db.get_entry_prices(
            ["AAPL"], date(2024, 1, 28), date(2024, 1, 28), max_staleness_days=30
        )
        self.assertEqual(len(result), 1)
        self.assertIsNotNone(result.iloc[0]["entry_price"])


class TestContextManager(DatabaseTestCase):
    def test_context_manager_closes_connection(self):
        db_path = Path(self.tmp_dir) / "test_ctx.duckdb"
        with Database(db_path) as db:
            tables = db.conn.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
            ).fetchall()
            self.assertTrue(len(tables) > 0)
        with self.assertRaises(duckdb.ConnectionException):
            db.conn.execute("SELECT 1")

    def test_read_only_mode(self):
        db_path = Path(self.tmp_dir) / "test_ro.duckdb"
        with Database(db_path) as rw_db:
            rw_db.close()
        ro_db = Database(db_path, read_only=True)
        self.assertTrue(ro_db.is_read_only)
        ro_db.close()


class TestDeleteTransactionsForDoc(DatabaseTestCase):

    def test_delete_removes_only_that_docs_rows(self):
        df = pd.DataFrame([
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
        ])
        self.db.upsert_transactions(df, source="house_pdf")
        self.assertEqual(len(self.db.get_transactions(2024)), 2)

        self.db.delete_transactions_for_doc("doc1")
        result = self.db.get_transactions(2024)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["ticker"], "GOOG")

    def test_delete_nonexistent_doc_is_noop(self):
        df = pd.DataFrame([
            {
                "doc_id": "doc1",
                "member": "John Doe",
                "ticker": "AAPL",
                "transaction_date": date(2024, 3, 10),
                "disclosure_date": date(2024, 3, 15),
                "transaction_type": "Purchase",
            },
        ])
        self.db.upsert_transactions(df, source="house_pdf")
        self.db.delete_transactions_for_doc("nonexistent")
        self.assertEqual(len(self.db.get_transactions(2024)), 1)


class TestParseRunsTable(DatabaseTestCase):

    def test_pdf_parse_runs_table_exists(self):
        tables = self.db.conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
        ).fetchall()
        table_names = {t[0] for t in tables}
        self.assertIn("pdf_parse_runs", table_names)

    def test_upsert_parse_run_inserts_row(self):
        self.db.upsert_parse_run(
            doc_id="doc1",
            year=2024,
            parser_version="v2",
            status="success",
            engines_attempted="lattice,stream,ocr",
            raw_row_count=10,
            transaction_count=5,
        )
        result = self.db.conn.execute("SELECT * FROM pdf_parse_runs WHERE doc_id = 'doc1'").fetchall()
        self.assertEqual(len(result), 1)
        row = result[0]
        self.assertEqual(row[0], "doc1")   # doc_id
        self.assertEqual(row[1], 2024)     # year
        self.assertEqual(row[2], "v2")     # parser_version
        self.assertEqual(row[3], "success")  # status
        self.assertEqual(row[4], "lattice,stream,ocr")  # engines_attempted
        self.assertEqual(row[6], 5)        # transaction_count

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
        result = self.db.conn.execute("SELECT status FROM pdf_parse_runs WHERE doc_id = 'doc2'").fetchone()
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

    def test_reparse_replaces_old_rows(self):
        """Re-parsing a doc_id should delete old rows then insert new ones."""
        df1 = pd.DataFrame([
            {
                "doc_id": "doc1",
                "member": "John Doe",
                "ticker": "AAPL",
                "transaction_date": date(2024, 3, 10),
                "disclosure_date": date(2024, 3, 15),
                "transaction_type": "Purchase",
            },
        ])
        self.db.upsert_transactions(df1, source="house_pdf")
        self.assertEqual(len(self.db.get_transactions(2024)), 1)

        # Simulate re-parse: delete old, insert new with different data
        self.db.delete_transactions_for_doc("doc1")
        df2 = pd.DataFrame([
            {
                "doc_id": "doc1",
                "member": "John Doe",
                "ticker": "MSFT",
                "transaction_date": date(2024, 5, 1),
                "disclosure_date": date(2024, 5, 5),
                "transaction_type": "Sale",
            },
        ])
        self.db.upsert_transactions(df2, source="house_pdf")
        result = self.db.get_transactions(2024)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["ticker"], "MSFT")
        self.assertEqual(result.iloc[0]["transaction_type"], "Sale")


class TestPhantomPurge(DatabaseTestCase):

    def _insert_transaction(self, doc_id, ticker, asset_description):
        self.db.conn.execute("""
            INSERT INTO transactions (
                doc_id, member, ticker, transaction_date, disclosure_date,
                transaction_type, owner_code, amount_raw, asset_description, source
            ) VALUES (?, 'John Doe', ?, DATE '2024-03-10', DATE '2024-03-15',
                      'Purchase', '', '$1,001 - $15,000', ?, 'house_pdf')
        """, [doc_id, ticker, asset_description])

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
        remaining = self.db.conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
        self.assertEqual(remaining, 3)


if __name__ == "__main__":
    unittest.main()
