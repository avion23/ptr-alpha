import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from analyzer.analysis import (
    calculate_signal_potential, rank_members, get_horizon_performance,
    get_top_signals, get_member_signals, get_analysis_table
)
from analyzer.exceptions import AnalysisError

class TestAnalysis(unittest.TestCase):

    def setUp(self):
        self.sample_transactions = pd.DataFrame({
            'member': ['Alice', 'Bob', 'Alice', 'Charlie'],
            'ticker': ['AAPL', 'GOOGL', 'MSFT', 'AAPL'],
            'disclosure_date': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04']),
            'transaction_type': ['Purchase', 'Sale', 'Purchase', 'Purchase']
        })

        dates = pd.date_range('2023-12-15', '2024-02-15', freq='D')
        np.random.seed(42)
        self.sample_prices = pd.DataFrame({
            'AAPL': 100 + np.cumsum(np.random.randn(len(dates)) * 0.5),
            'GOOGL': 2000 + np.cumsum(np.random.randn(len(dates)) * 2),
            'MSFT': 300 + np.cumsum(np.random.randn(len(dates)) * 1)
        }, index=dates)

    def test_calculate_signal_potential_basic(self):
        signals = calculate_signal_potential(self.sample_transactions, self.sample_prices, [30, 90])

        self.assertFalse(signals.empty)
        self.assertEqual(len(signals), 8)  # 4 transactions * 2 horizons

        expected_cols = ['member', 'ticker', 'disclosure_date', 'signal_type', 'horizon_days', 'entry_price', 'peak_potential_pct']
        self.assertEqual(list(signals.columns), expected_cols)

        self.assertTrue(all(h in [30, 90] for h in signals['horizon_days'].unique()))
        self.assertTrue(all(st in ['Purchase', 'Sale'] for st in signals['signal_type'].unique()))

        self.assertFalse(signals['peak_potential_pct'].isna().any())
        self.assertTrue((signals['entry_price'] > 0).all())

    def test_calculate_signal_potential_empty_input(self):
        with self.assertRaises(AnalysisError):
            calculate_signal_potential(pd.DataFrame(), self.sample_prices)

        with self.assertRaises(AnalysisError):
            calculate_signal_potential(self.sample_transactions, pd.DataFrame())

    def test_calculate_signal_potential_missing_columns(self):
        bad_transactions = self.sample_transactions.drop(columns=['ticker'])
        with self.assertRaises(AnalysisError):
            calculate_signal_potential(bad_transactions, self.sample_prices)

    def test_calculate_signal_potential_purchase_vs_sale(self):
        signals = calculate_signal_potential(self.sample_transactions, self.sample_prices, [30])

        purchases = signals[signals['signal_type'] == 'Purchase']
        sales = signals[signals['signal_type'] == 'Sale']

        self.assertEqual(len(purchases), 3)
        self.assertEqual(len(sales), 1)

        for _, row in purchases.iterrows():
            self.assertTrue(row['peak_potential_pct'] >= -100)

        for _, row in sales.iterrows():
            self.assertTrue(row['peak_potential_pct'] >= -100)

    def test_rank_members_basic(self):
        signals = calculate_signal_potential(self.sample_transactions, self.sample_prices, [90])
        rankings = rank_members(signals, horizon=90, threshold=5.0)

        self.assertFalse(rankings.empty)
        self.assertTrue('member' in rankings.columns)
        self.assertTrue('avg_peak_return_pct' in rankings.columns)

        if len(rankings) > 1:
            returns = rankings['avg_peak_return_pct'].dropna()
            if len(returns) > 1:
                values = returns.values
                self.assertTrue((values[:-1] >= values[1:]).all() or (values[:-1] <= values[1:]).all())

    def test_rank_members_empty_input(self):
        with self.assertRaises(AnalysisError):
            rank_members(pd.DataFrame())

    def test_get_top_signals_basic(self):
        signals = calculate_signal_potential(self.sample_transactions, self.sample_prices, [90])
        top_signals = get_top_signals(signals, horizon=90, top_n=2)

        self.assertFalse(top_signals.empty)
        self.assertLessEqual(len(top_signals), 2)

        expected_cols = ['member', 'ticker', 'disclosure_date', 'peak_potential_pct']
        self.assertEqual(list(top_signals.columns), expected_cols)

        if len(top_signals) > 1:
            values = top_signals['peak_potential_pct'].values
            self.assertTrue((values[:-1] >= values[1:]).all())

    def test_get_top_signals_empty_input(self):
        with self.assertRaises(AnalysisError):
            get_top_signals(pd.DataFrame())

    def test_get_member_signals_basic(self):
        signals = calculate_signal_potential(self.sample_transactions, self.sample_prices, [90])
        member_signals = get_member_signals(signals, 'Alice', horizon=90, top_n=5)

        self.assertFalse(member_signals.empty)
        self.assertTrue(all(signals[signals['member'] == 'Alice']['signal_type'] == 'Purchase'))

        expected_cols = ['ticker', 'disclosure_date', 'peak_potential_pct']
        self.assertEqual(list(member_signals.columns), expected_cols)

    def test_get_member_signals_nonexistent_member(self):
        signals = calculate_signal_potential(self.sample_transactions, self.sample_prices, [90])
        with self.assertRaises(AnalysisError):
            get_member_signals(signals, 'NonExistent', horizon=90, top_n=5)

    def test_get_analysis_table_member_filter(self):
        signals = calculate_signal_potential(self.sample_transactions, self.sample_prices, [90])
        table = get_analysis_table(signals, 'Alice', False, 90, 5, 5.0)

        self.assertFalse(table.empty)
        expected_cols = ['ticker', 'disclosure_date', 'peak_potential_pct']
        self.assertEqual(list(table.columns), expected_cols)

    def test_get_analysis_table_show_signals(self):
        signals = calculate_signal_potential(self.sample_transactions, self.sample_prices, [90])
        table = get_analysis_table(signals, None, True, 90, 5, 5.0)

        self.assertFalse(table.empty)
        expected_cols = ['member', 'ticker', 'disclosure_date', 'peak_potential_pct']
        self.assertEqual(list(table.columns), expected_cols)

    def test_get_analysis_table_rank_members(self):
        signals = calculate_signal_potential(self.sample_transactions, self.sample_prices, [90])
        table = get_analysis_table(signals, None, False, 90, 5, 5.0)

        self.assertFalse(table.empty)
        self.assertTrue('member' in table.columns)

    def test_horizon_performance(self):
        signals = calculate_signal_potential(self.sample_transactions, self.sample_prices, [30, 90])
        perf = get_horizon_performance(signals, threshold=5.0)

        self.assertFalse(perf.empty)
        self.assertTrue(30 in perf.index)
        self.assertTrue(90 in perf.index)

        expected_cols = ['avg_peak_pct', 'hit_rate_pct']
        self.assertEqual(list(perf.columns), expected_cols)

if __name__ == '__main__':
    unittest.main()