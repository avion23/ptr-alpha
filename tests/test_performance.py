import unittest
import pandas as pd
import numpy as np
import time
from analyzer.analysis import calculate_signal_potential

def create_large_test_data(n_transactions=10000, n_tickers=500):
    np.random.seed(42)

    members = [f"Member_{i}" for i in range(n_transactions // 50)]
    tickers = [f"TICK{i:03d}" for i in range(n_tickers)]

    transactions = pd.DataFrame({
        'member': np.random.choice(members, n_transactions),
        'ticker': np.random.choice(tickers, n_transactions),
        'disclosure_date': pd.date_range('2024-01-01', periods=n_transactions, freq='6h'),
        'transaction_type': np.random.choice(['Purchase', 'Sale'], n_transactions, p=[0.6, 0.4])
    })

    date_range = pd.date_range('2023-12-01', '2024-12-31', freq='D')
    prices_data = {}

    for ticker in tickers[:min(100, n_tickers)]:
        base_price = np.random.uniform(10, 500)
        price_changes = np.cumsum(np.random.randn(len(date_range)) * 0.02)
        prices_data[ticker] = base_price * np.exp(price_changes)

    prices = pd.DataFrame(prices_data, index=date_range)

    return transactions, prices

class TestPerformance(unittest.TestCase):

    def test_vectorized_performance_small(self):
        transactions, prices = create_large_test_data(1000, 50)

        start_time = time.time()
        signals = calculate_signal_potential(transactions, prices, [30, 90])
        elapsed_time = time.time() - start_time

        self.assertFalse(signals.empty)
        self.assertLess(elapsed_time, 5.0, f"Small dataset took {elapsed_time:.2f}s, should be under 5s")

        self.assertGreater(len(signals), 100, "Should generate reasonable number of signals")

        print(f"Small dataset ({len(transactions)} transactions): {elapsed_time:.2f}s")
        print(f"Generated {len(signals)} signals")

    def test_vectorized_performance_medium(self):
        transactions, prices = create_large_test_data(5000, 100)

        start_time = time.time()
        signals = calculate_signal_potential(transactions, prices, [90])
        elapsed_time = time.time() - start_time

        self.assertFalse(signals.empty)
        self.assertLess(elapsed_time, 15.0, f"Medium dataset took {elapsed_time:.2f}s, should be under 15s")

        print(f"Medium dataset ({len(transactions)} transactions): {elapsed_time:.2f}s")
        print(f"Generated {len(signals)} signals")

    def test_signal_calculation_correctness(self):
        np.random.seed(123)

        transactions = pd.DataFrame({
            'member': ['Alice', 'Bob'],
            'ticker': ['AAPL', 'GOOGL'],
            'disclosure_date': pd.to_datetime(['2024-01-15', '2024-02-01']),
            'transaction_type': ['Purchase', 'Sale']
        })

        dates = pd.date_range('2024-01-01', '2024-04-01', freq='D')
        prices = pd.DataFrame({
            'AAPL': [100] + [100 + i * 0.5 for i in range(1, len(dates))],
            'GOOGL': [2000] + [2000 - i * 2 for i in range(1, len(dates))]
        }, index=dates)

        signals = calculate_signal_potential(transactions, prices, [30])

        self.assertFalse(signals.empty, "Should generate signals for the test data")
        self.assertGreater(len(signals), 0)

        alice_signal = signals[signals['member'] == 'Alice'].iloc[0]
        bob_signal = signals[signals['member'] == 'Bob'].iloc[0]

        self.assertEqual(alice_signal['signal_type'], 'Purchase')
        self.assertEqual(bob_signal['signal_type'], 'Sale')

        self.assertTrue(np.isfinite(alice_signal['peak_potential_pct']))
        self.assertTrue(np.isfinite(bob_signal['peak_potential_pct']))

if __name__ == '__main__':
    unittest.main()