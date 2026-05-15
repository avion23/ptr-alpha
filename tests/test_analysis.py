import unittest
import pandas as pd
import numpy as np
from analyzer.analysis import (
    calculate_signal_potential, rank_members,
    get_top_signals, get_member_signals, get_analysis_table,
    score_ticker_by_buyers
)
from analyzer.exceptions import AnalysisError


def _make_entry_prices(transactions_df, prices_df):
    prices_long = prices_df.stack().reset_index(name="price")
    prices_long.columns = ["price_date", "ticker", "price"]
    prices_long = prices_long.sort_values("price_date")
    trans_sorted = transactions_df.sort_values("disclosure_date")
    merged = pd.merge_asof(
        trans_sorted, prices_long,
        left_on="disclosure_date", right_on="price_date", by="ticker"
    ).dropna(subset=["price"])
    optional_columns = [column for column in ["owner_code", "amount_midpoint"] if column in merged.columns]
    return merged[["member", "ticker", "disclosure_date", "transaction_type", "price", *optional_columns]].rename(
        columns={"price": "entry_price"}
    ).reset_index(drop=True)


class TestAnalysis(unittest.TestCase):

    def setUp(self):
        self.sample_transactions = pd.DataFrame({
            'member': ['Alice', 'Bob', 'Alice', 'Charlie'],
            'ticker': ['AAPL', 'GOOGL', 'MSFT', 'AAPL'],
            'disclosure_date': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04']),
            'transaction_type': ['Purchase', 'Sale', 'Purchase', 'Purchase'],
            'owner_code': [None, None, 'DC', None],
            'amount_midpoint': [8000.5, 8000.5, 32500.5, 100000.0],
        })

        dates = pd.date_range('2023-12-15', '2024-02-15', freq='D')
        np.random.seed(42)
        self.sample_prices = pd.DataFrame({
            'AAPL': 100 + np.cumsum(np.random.randn(len(dates)) * 0.5),
            'GOOGL': 2000 + np.cumsum(np.random.randn(len(dates)) * 2),
            'MSFT': 300 + np.cumsum(np.random.randn(len(dates)) * 1)
        }, index=dates)

        self.entry_prices = _make_entry_prices(self.sample_transactions, self.sample_prices)

    def test_calculate_signal_potential_basic(self):
        signals = calculate_signal_potential(self.entry_prices, self.sample_prices, [30, 90])

        self.assertFalse(signals.empty)
        self.assertEqual(len(signals), 8)

        required_cols = ['member', 'ticker', 'disclosure_date', 'signal_type', 'horizon_days', 'entry_price', 'peak_potential_pct']
        for col in required_cols:
            self.assertIn(col, signals.columns)

        self.assertTrue(all(h in [30, 90] for h in signals['horizon_days'].unique()))
        self.assertTrue(all(st in ['Purchase', 'Sale'] for st in signals['signal_type'].unique()))
        self.assertIn('owner_code', signals.columns)
        self.assertIn('amount_midpoint', signals.columns)

        self.assertFalse(signals['peak_potential_pct'].isna().any())
        self.assertTrue((signals['entry_price'] > 0).all())

    def test_score_ticker_by_buyers_applies_small_metadata_adjustments(self):
        transactions = pd.DataFrame({
            'member': ['Alice', 'Charlie'],
            'ticker': ['AAPL', 'AAPL'],
            'transaction_date': pd.to_datetime(['2024-01-01', '2024-01-02']),
            'disclosure_date': pd.to_datetime(['2024-01-03', '2024-01-04']),
            'transaction_type': ['Purchase', 'Purchase'],
            'owner_code': [None, 'DC'],
            'amount_midpoint': [100000.0, 100000.0],
        })
        signals = pd.DataFrame({
            'member': ['Alice', 'Charlie'],
            'ticker': ['AAPL', 'AAPL'],
            'signal_type': ['Purchase', 'Purchase'],
            'horizon_days': [90, 90],
            'decayed_return_pct': [10.0, 10.0],
            'peak_potential_pct': [12.0, 12.0],
            'spy_alpha_pct': [10.0, 10.0],
        })

        score = score_ticker_by_buyers('AAPL', transactions, signals)

        self.assertEqual(score.iloc[0]['base_signal_score'], 20.0)
        self.assertGreater(score.iloc[0]['size_factor'], 1.0)
        self.assertLess(score.iloc[0]['owner_factor'], 1.0)
        self.assertNotEqual(score.iloc[0]['signal_score'], score.iloc[0]['base_signal_score'])

    def test_calculate_signal_potential_empty_input(self):
        with self.assertRaises(AnalysisError):
            calculate_signal_potential(pd.DataFrame(), self.sample_prices)

        with self.assertRaises(AnalysisError):
            calculate_signal_potential(self.entry_prices, pd.DataFrame())

    def test_calculate_signal_potential_missing_columns(self):
        bad_data = self.entry_prices.drop(columns=['ticker'])
        with self.assertRaises(AnalysisError):
            calculate_signal_potential(bad_data, self.sample_prices)

    def test_calculate_signal_potential_purchase_vs_sale(self):
        signals = calculate_signal_potential(self.entry_prices, self.sample_prices, [30])

        purchases = signals[signals['signal_type'] == 'Purchase']
        sales = signals[signals['signal_type'] == 'Sale']

        self.assertEqual(len(purchases), 3)
        self.assertEqual(len(sales), 1)

        for _, row in purchases.iterrows():
            self.assertTrue(row['peak_potential_pct'] >= -100)

        for _, row in sales.iterrows():
            self.assertTrue(row['peak_potential_pct'] >= -100)

    def test_rank_members_basic(self):
        signals = calculate_signal_potential(self.entry_prices, self.sample_prices, [90])
        rankings = rank_members(signals, horizon=90, threshold=5.0)

        self.assertFalse(rankings.empty)
        self.assertTrue('member' in rankings.columns)
        self.assertTrue('avg_peak_return_pct' in rankings.columns)

        returns = rankings['avg_peak_return_pct'].dropna()
        self.assertTrue(len(returns) > 0)

    def test_rank_members_empty_input(self):
        with self.assertRaises(AnalysisError):
            rank_members(pd.DataFrame())

    def test_get_top_signals_basic(self):
        signals = calculate_signal_potential(self.entry_prices, self.sample_prices, [90])
        top_signals = get_top_signals(signals, horizon=90, top_n=2)

        self.assertFalse(top_signals.empty)
        self.assertLessEqual(len(top_signals), 2)

        for col in ['member', 'ticker', 'disclosure_date', 'peak_potential_pct']:
            self.assertIn(col, top_signals.columns)

        if len(top_signals) > 1:
            values = top_signals['spy_alpha_pct'].values
            self.assertTrue((values[:-1] >= values[1:]).all())

    def test_get_top_signals_empty_input(self):
        with self.assertRaises(AnalysisError):
            get_top_signals(pd.DataFrame())

    def test_get_member_signals_basic(self):
        signals = calculate_signal_potential(self.entry_prices, self.sample_prices, [90])
        member_signals = get_member_signals(signals, 'Alice', horizon=90, top_n=5)

        self.assertFalse(member_signals.empty)
        if 'signal_type' in member_signals.columns:
            self.assertTrue(all(s in ['Purchase'] for s in member_signals['signal_type'].unique()))

        for col in ['ticker', 'disclosure_date', 'peak_potential_pct']:
            self.assertIn(col, member_signals.columns)

    def test_get_member_signals_nonexistent_member(self):
        signals = calculate_signal_potential(self.entry_prices, self.sample_prices, [90])
        with self.assertRaises(AnalysisError):
            get_member_signals(signals, 'NonExistent', horizon=90, top_n=5)

    def test_get_analysis_table_member_filter(self):
        signals = calculate_signal_potential(self.entry_prices, self.sample_prices, [90])
        table = get_analysis_table(signals, 'Alice', False, 90, 5, 5.0)

        self.assertFalse(table.empty)
        self.assertIn('ticker', table.columns)

    def test_get_analysis_table_show_signals(self):
        signals = calculate_signal_potential(self.entry_prices, self.sample_prices, [90])
        table = get_analysis_table(signals, None, True, 90, 5, 5.0)

        self.assertFalse(table.empty)
        for col in ['member', 'ticker', 'disclosure_date', 'peak_potential_pct']:
            self.assertIn(col, table.columns)

    def test_get_analysis_table_rank_members(self):
        signals = calculate_signal_potential(self.entry_prices, self.sample_prices, [90])
        table = get_analysis_table(signals, None, False, 90, 5, 5.0)

        self.assertFalse(table.empty)
        self.assertTrue('member' in table.columns)

if __name__ == '__main__':
    unittest.main()
