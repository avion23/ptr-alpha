import unittest
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch
from analyzer.settings import Settings, DataSettings
from analyzer.pipeline import run_analysis_pipeline, AnalysisParams
from analyzer.download import HouseTransactionSource
from analyzer.price_source import YFinancePriceSource
from analyzer.database import Database
from analyzer.models import AnalysisMode
from analyzer.exceptions import DataResult, DataSourceError


class TestIntegration(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.temp_dir = temporary.name
        self.settings = Settings(
            data=DataSettings(data_dir=self.temp_dir, cache_enabled=False)
        )
        local_prices = patch.object(
            YFinancePriceSource,
            "_download_yfinance",
            side_effect=AssertionError("integration prices must remain local"),
        )
        local_prices.start()
        self.addCleanup(local_prices.stop)

    def _insert_mock_data(self):
        np.random.seed(42)
        db = Database(Path(self.temp_dir) / "congress.duckdb")
        transactions = pd.DataFrame(
            {
                "doc_id": [f"DOC-{i:04d}" for i in range(30)],
                "member": ["Alice Smith", "Bob Jones", "Charlie Brown"] * 10,
                "ticker": ["AAPL", "GOOGL", "MSFT"] * 10,
                "disclosure_date": pd.date_range("2024-01-01", periods=30, freq="D"),
                "transaction_date": pd.date_range("2023-12-25", periods=30, freq="D"),
                "transaction_type": np.random.choice(
                    ["Purchase", "Sale"], 30, p=[0.7, 0.3]
                ),
            }
        )
        transactions["chamber"] = "house"
        transactions["official_filing_date"] = transactions["disclosure_date"]
        transactions["ingestion_generation"] = "integration"
        transactions["artifact_sha256"] = "fixture-sha"
        transactions["source_record_id"] = transactions["doc_id"]
        transactions["source_row_id"] = "fixture:1"
        db.conn.execute(
            "INSERT INTO house_archive_generations "
            "(archive_year, generation_id, parse_status, promoted_at) "
            "VALUES (2024, 'integration', 'incomplete', CURRENT_TIMESTAMP)"
        )
        for doc_id in transactions["doc_id"]:
            db.conn.execute(
                "INSERT INTO house_generation_metadata "
                "(archive_year, generation_id, doc_id, filing_type) "
                "VALUES (2024, 'integration', ?, 'P')",
                [doc_id],
            )
            db.conn.execute(
                "INSERT INTO house_pdf_artifacts "
                "(archive_year, generation_id, doc_id, artifact_sha256) "
                "VALUES (2024, 'integration', ?, 'fixture-sha')",
                [doc_id],
            )
        db.upsert_transactions(transactions, source="house_pdf")
        for doc_id in transactions["doc_id"]:
            db.upsert_parse_run(
                doc_id=doc_id,
                year=2024,
                parser_version="pdftotext",
                status="success",
                engines_attempted="pdftotext",
                raw_row_count=1,
                transaction_count=1,
                artifact_sha256="fixture-sha",
                ingestion_generation="integration",
            )
        self.assertEqual(len(db.get_transactions(2024)), len(transactions))

        np.random.seed(99)
        tickers = sorted(list(set(transactions["ticker"].unique()) | {"SPY"}))
        dates = pd.date_range("2023-11-01", "2024-06-30", freq="D")
        price_data = {}
        for ticker in tickers:
            base_price = 100.0 + hash(ticker) % 50
            price_data[ticker] = base_price + np.cumsum(
                np.random.normal(0.1, 2, len(dates))
            )
        db.upsert_prices(pd.DataFrame(price_data, index=dates))

        db.close()

    def test_end_to_end_house_analysis(self):
        self._insert_mock_data()
        transaction_source = HouseTransactionSource(self.settings)
        price_source = YFinancePriceSource(self.settings)
        self.addCleanup(transaction_source.close)
        self.addCleanup(price_source.close)

        params = AnalysisParams(
            year=2024,
            horizons=(90,),
            top_n=10,
        )

        result = run_analysis_pipeline(params, transaction_source, price_source)

        self.assertTrue(result.success, result.error)
        self.assertIsNotNone(result.data)
        self.assertFalse(result.data["table"].empty)
        self.assertEqual(
            set(result.data["table"]["member"]),
            {"Alice Smith", "Bob Jones", "Charlie Brown"},
        )

    def test_member_specific_analysis(self):
        self._insert_mock_data()
        transaction_source = HouseTransactionSource(self.settings)
        price_source = YFinancePriceSource(self.settings)
        self.addCleanup(transaction_source.close)
        self.addCleanup(price_source.close)

        params = AnalysisParams(
            year=2024,
            horizons=(30, 90),
            member_filter="Alice Smith",
            mode=AnalysisMode.MEMBER_SIGNALS,
            top_n=5,
        )

        result = run_analysis_pipeline(params, transaction_source, price_source)

        self.assertTrue(result.success, result.error)
        self.assertIsNotNone(result.data)
        self.assertFalse(result.data["table"].empty)
        self.assertEqual(result.data["member_filter"], "Alice Smith")
        self.assertEqual(set(result.data["table"]["ticker"]), {"AAPL"})

    def test_empty_analysis_returns_failed_data_result(self):
        transaction_source = HouseTransactionSource(self.settings)
        price_source = YFinancePriceSource(self.settings)
        self.addCleanup(transaction_source.close)
        self.addCleanup(price_source.close)

        result = run_analysis_pipeline(
            AnalysisParams(year=2024, horizons=(90,)), transaction_source, price_source
        )

        self.assertIsInstance(result, DataResult)
        self.assertFalse(result.success)
        self.assertIsNone(result.data)
        self.assertIsInstance(result.error, DataSourceError)


if __name__ == "__main__":
    unittest.main()
