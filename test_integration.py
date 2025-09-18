import unittest
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path
from sources import Config
from pipeline import run_analysis_pipeline

class TestIntegration(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = Config(data_dir=self.temp_dir, cache_enabled=False)

    def create_mock_data(self):
        np.random.seed(42)

        # Create mock transactions
        transactions = pd.DataFrame({
            'member': ['Alice Smith', 'Bob Jones', 'Charlie Brown'] * 10,
            'ticker': np.random.choice(['AAPL', 'GOOGL', 'MSFT', 'TSLA'], 30),
            'disclosure_date': pd.date_range('2024-01-01', periods=30, freq='D'),
            'transaction_date': pd.date_range('2023-12-25', periods=30, freq='D'),
            'transaction_type': np.random.choice(['Purchase', 'Sale'], 30, p=[0.7, 0.3])
        })

        # Save mock transactions
        transactions_file = Path(self.temp_dir) / "2024" / "transactions.csv"
        transactions_file.parent.mkdir(exist_ok=True)
        transactions.to_csv(transactions_file, index=False)

        return transactions

    def test_end_to_end_house_analysis(self):
        transactions = self.create_mock_data()

        # Mock the sources functions
        import sources
        original_load_cached_data = sources.load_cached_data
        original_fetch_prices = sources.fetch_prices

        def mock_load_cached_data(year, config):
            return transactions

        def mock_fetch_prices(tickers, start_date, end_date, config):
            # Create mock price data for the tickers
            dates = pd.date_range(start_date, end_date, freq='D')
            price_data = {}
            for ticker in tickers:
                price_data[ticker] = [100.0 + np.random.normal(0, 5) for _ in range(len(dates))]
            return pd.DataFrame(price_data, index=dates)

        sources.load_cached_data = mock_load_cached_data
        sources.fetch_prices = mock_fetch_prices

        try:
            result = run_analysis_pipeline(
                source='house',
                year=2024,
                horizons=[90],
                threshold=5.0,
                member_filter=None,
                top_n=10,
                show_signals=False,
                output_format='console',
                config=self.config
            )

            self.assertTrue(result, "Analysis pipeline should succeed")

        finally:
            sources.load_cached_data = original_load_cached_data
            sources.fetch_prices = original_fetch_prices

    def test_member_specific_analysis(self):
        transactions = self.create_mock_data()

        import sources
        original_load_cached_data = sources.load_cached_data
        original_fetch_prices = sources.fetch_prices

        def mock_load_cached_data(year, config):
            return transactions

        def mock_fetch_prices(tickers, start_date, end_date, config):
            dates = pd.date_range(start_date, end_date, freq='D')
            price_data = {}
            for ticker in tickers:
                price_data[ticker] = [100.0 + np.random.normal(0, 5) for _ in range(len(dates))]
            return pd.DataFrame(price_data, index=dates)

        sources.load_cached_data = mock_load_cached_data
        sources.fetch_prices = mock_fetch_prices

        try:
            result = run_analysis_pipeline(
                source='house',
                year=2024,
                horizons=[30, 90],
                threshold=5.0,
                member_filter='Alice Smith',
                top_n=5,
                show_signals=False,
                output_format='console',
                config=self.config
            )

            self.assertTrue(result, "Member analysis pipeline should succeed")

        finally:
            sources.load_cached_data = original_load_cached_data
            sources.fetch_prices = original_fetch_prices

if __name__ == '__main__':
    unittest.main()