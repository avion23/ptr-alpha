import unittest
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path
from analyzer.settings import Settings, DataSettings
from analyzer.pipeline import run_analysis_pipeline, AnalysisParams
from analyzer.datasources import HouseTransactionSource, YFinancePriceSource
from analyzer.database import Database


class TestIntegration(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.settings = Settings(data=DataSettings(data_dir=self.temp_dir, cache_enabled=False))

    def _insert_mock_data(self):
        np.random.seed(42)
        db = Database(Path(self.temp_dir) / "congress.duckdb")
        transactions = pd.DataFrame({
            'doc_id': [f'DOC-{i:04d}' for i in range(30)],
            'member': ['Alice Smith', 'Bob Jones', 'Charlie Brown'] * 10,
            'ticker': np.random.choice(['AAPL', 'GOOGL', 'MSFT', 'TSLA'], 30),
            'disclosure_date': pd.date_range('2024-01-01', periods=30, freq='D'),
            'transaction_date': pd.date_range('2023-12-25', periods=30, freq='D'),
            'transaction_type': np.random.choice(['Purchase', 'Sale'], 30, p=[0.7, 0.3])
        })
        db.upsert_transactions(transactions, source="house_pdf")

        np.random.seed(99)
        tickers = sorted(list(set(transactions['ticker'].unique()) | {'SPY'}))
        dates = pd.date_range('2023-11-01', '2024-06-30', freq='D')
        price_data = {}
        for ticker in tickers:
            base_price = 100.0 + hash(ticker) % 50
            price_data[ticker] = base_price + np.cumsum(np.random.normal(0.1, 2, len(dates)))
        db.upsert_prices(pd.DataFrame(price_data, index=dates))

        db.close()

    def test_end_to_end_house_analysis(self):
        self._insert_mock_data()
        transaction_source = HouseTransactionSource(self.settings)
        price_source = YFinancePriceSource(self.settings)

        params = AnalysisParams(
            source='house',
            year=2024,
            horizons=[90],
            threshold=5.0,
            top_n=10,
        )

        result = run_analysis_pipeline(
            params, transaction_source, price_source
        )

        self.assertTrue(result, "Analysis pipeline should succeed")

    def test_member_specific_analysis(self):
        self._insert_mock_data()
        transaction_source = HouseTransactionSource(self.settings)
        price_source = YFinancePriceSource(self.settings)

        params = AnalysisParams(
            source='house',
            year=2024,
            horizons=[30, 90],
            threshold=5.0,
            member_filter='Alice Smith',
            top_n=5,
        )

        result = run_analysis_pipeline(
            params, transaction_source, price_source
        )

        self.assertTrue(result, "Member analysis pipeline should succeed")


if __name__ == '__main__':
    unittest.main()
