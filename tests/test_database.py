import unittest
from datetime import date, datetime
from pathlib import Path

import duckdb
import pandas as pd

from analyzer.database import Database
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
        self.assertIn("pdf_downloads", table_names)

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
        self.db.upsert_transactions(df)
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
        self.db.upsert_transactions(df)
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
        self.db.upsert_transactions(df)
        result = self.db.get_transactions(2024)
        self.assertEqual(len(result), 2)
        types = set(result["transaction_type"].values)
        self.assertEqual(types, {"Purchase", "Sale"})


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
        dates = pd.bdate_range("2024-01-01", "2024-01-02")
        prices = pd.DataFrame({"AAPL": [100.0, 101.0]}, index=dates)
        self.db.upsert_prices(prices)
        missing_tickers, missing_dates = self.db.get_missing_price_data(
            ["AAPL"], date(2024, 1, 1), date(2024, 1, 5)
        )
        self.assertEqual(missing_tickers, [])
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
        self.db.upsert_transactions(tx)

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
        self.db.upsert_transactions(tx)

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
        self.db.upsert_transactions(tx)

        result = self.db.get_entry_prices(
            ["AAPL"], date(2024, 1, 1), date(2024, 1, 10)
        )
        self.assertEqual(len(result), 1)


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


if __name__ == "__main__":
    unittest.main()
