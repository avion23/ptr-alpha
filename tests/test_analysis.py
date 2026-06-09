import unittest
import pandas as pd
import numpy as np
from analyzer.analysis import (
    calculate_signal_potential, rank_members, rank_sales,
    get_top_signals, get_member_signals, get_analysis_table,
    score_ticker_by_buyers, bayesian_win_probability,
    bayes_factor_against_market, _collapse_to_episodes,
)
from analyzer.exceptions import AnalysisError

from .conftest import make_entry_prices


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

        self.entry_prices = make_entry_prices(self.sample_transactions, self.sample_prices)

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

        buyer_diminishing = 1.0 + np.log1p(1) * 0.30
        expected_base = round(buyer_diminishing * 10.0, 2)
        self.assertEqual(score.iloc[0]['base_signal_score'], expected_base)
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
        self.assertTrue('avg_decay_return_pct' in rankings.columns)

        returns = rankings['avg_decay_return_pct'].dropna()
        self.assertTrue(len(returns) > 0)

    def test_rank_members_empty_input(self):
        with self.assertRaises(AnalysisError):
            rank_members(pd.DataFrame())

    def test_rank_members_filters_by_horizon(self):
        signals = pd.DataFrame({
            'member': ['Alice', 'Alice'],
            'ticker': ['AAPL', 'AAPL'],
            'signal_type': ['Purchase', 'Purchase'],
            'horizon_days': [30, 90],
            'decayed_return_pct': [-50.0, 50.0],
            'peak_potential_pct': [-40.0, 60.0],
            'spy_alpha_pct': [-45.0, 45.0],
        })

        r30 = rank_members(signals, horizon=30)
        r90 = rank_members(signals, horizon=90)

        self.assertEqual(r30.iloc[0]['avg_spy_alpha_pct'], -45.0)
        self.assertEqual(r90.iloc[0]['avg_spy_alpha_pct'], 45.0)
        self.assertEqual(r30.iloc[0]['purchase_trades'], 1)
        self.assertEqual(r90.iloc[0]['purchase_trades'], 1)

    def test_rank_sales_filters_by_horizon(self):
        signals = pd.DataFrame({
            'member': ['Alice', 'Alice'],
            'ticker': ['AAPL', 'AAPL'],
            'signal_type': ['Sale', 'Sale'],
            'horizon_days': [30, 90],
            'decayed_return_pct': [-20.0, 20.0],
            'peak_potential_pct': [30.0, -10.0],
            'spy_alpha_pct': [-15.0, 15.0],
        })

        r30 = rank_sales(signals, horizon=30)
        r90 = rank_sales(signals, horizon=90)

        self.assertEqual(r30.iloc[0]['avg_loss_avoided_pct'], 20.0)
        self.assertEqual(r90.iloc[0]['avg_loss_avoided_pct'], -20.0)
        self.assertEqual(r30.iloc[0]['sale_trades'], 1)
        self.assertEqual(r90.iloc[0]['sale_trades'], 1)

    def test_bayesian_win_probability_formula(self):
        posterior = bayesian_win_probability(0, 3, 0.55)
        expected = (0.55 * 20) / (20 + 3)

        self.assertAlmostEqual(posterior, expected)

    def test_bayes_factor_against_market_formula(self):
        bayes_factor = bayes_factor_against_market(5, 5, 0.55)

        self.assertGreater(bayes_factor, 0)
        self.assertLess(bayes_factor, 2)

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

    def test_score_ticker_by_buyers_uses_rated_buyers_not_all_buyers(self):
        transactions = pd.DataFrame({
            'member': ['Alice', 'Charlie', 'Unranked'],
            'ticker': ['AAPL', 'AAPL', 'AAPL'],
            'transaction_date': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03']),
            'disclosure_date': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03']),
            'transaction_type': ['Purchase', 'Purchase', 'Purchase'],
        })
        member_rankings = pd.DataFrame({
            'member': ['Alice', 'Charlie'],
            'avg_spy_alpha_pct': [10.0, 20.0],
            'purchase_trades': [3, 2],
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

        score = score_ticker_by_buyers('AAPL', transactions, signals, member_rankings=member_rankings)

        buyer_diminishing = 1.0 + np.log1p(1) * 0.30
        alice_weight = np.sqrt(3) * np.exp(-0.03)
        charlie_weight = np.sqrt(2)
        quality_weighted_sum = 10.0 * alice_weight + 20.0 * charlie_weight
        quality_adjusted_avg = quality_weighted_sum / (alice_weight + charlie_weight)
        expected_base = round(buyer_diminishing * quality_adjusted_avg, 2)
        inflated_buyers = 1.0 + np.log1p(2) * 0.30
        inflated_base = round(inflated_buyers * quality_adjusted_avg, 2)
        self.assertEqual(score.iloc[0]['num_buyers'], 2)
        self.assertEqual(score.iloc[0]['total_buyers'], 3)
        self.assertEqual(score.iloc[0]['base_signal_score'], expected_base)
        self.assertNotEqual(score.iloc[0]['base_signal_score'], inflated_base)

    def test_score_ticker_by_buyers_uses_sqrt_confidence_not_trade_count_dominance(self):
        transactions = pd.DataFrame({
            'member': ['Focused', 'NoiseBot'],
            'ticker': ['AAPL', 'AAPL'],
            'transaction_date': pd.to_datetime(['2024-01-01', '2024-01-02']),
            'disclosure_date': pd.to_datetime(['2024-01-01', '2024-01-02']),
            'transaction_type': ['Purchase', 'Purchase'],
        })
        member_rankings = pd.DataFrame({
            'member': ['Focused', 'NoiseBot'],
            'avg_spy_alpha_pct': [18.0, 3.0],
            'purchase_trades': [5, 500],
            'bayes_win_prob': [0.75, 0.55],
        })
        signals = pd.DataFrame({
            'member': ['Focused', 'NoiseBot'],
            'ticker': ['AAPL', 'AAPL'],
            'signal_type': ['Purchase', 'Purchase'],
            'horizon_days': [90, 90],
            'decayed_return_pct': [18.0, 3.0],
            'peak_potential_pct': [20.0, 5.0],
            'spy_alpha_pct': [18.0, 3.0],
        })

        score = score_ticker_by_buyers('AAPL', transactions, signals, member_rankings=member_rankings)

        self.assertGreater(score.iloc[0]['avg_buyer_performance'], 3.0)
        self.assertLess(score.iloc[0]['avg_buyer_performance'], 8.0)

    def test_rank_members_skips_members_with_all_nan_returns(self):
        signals = pd.DataFrame({
            'member': ['Alice', 'Bob'],
            'ticker': ['AAPL', 'GOOGL'],
            'signal_type': ['Purchase', 'Purchase'],
            'horizon_days': [90, 90],
            'decayed_return_pct': [10.0, float('nan')],
            'peak_potential_pct': [12.0, float('nan')],
            'spy_alpha_pct': [10.0, float('nan')],
        })

        rankings = rank_members(signals, horizon=90)

        self.assertEqual(len(rankings), 1)
        self.assertEqual(rankings.iloc[0]['member'], 'Alice')
        self.assertFalse(np.isnan(rankings.iloc[0]['avg_decay_return_pct']))

    def test_rank_sales_skips_members_with_all_nan_returns(self):
        signals = pd.DataFrame({
            'member': ['Alice', 'Bob'],
            'ticker': ['AAPL', 'GOOGL'],
            'signal_type': ['Sale', 'Sale'],
            'horizon_days': [90, 90],
            'decayed_return_pct': [5.0, float('nan')],
            'peak_potential_pct': [8.0, float('nan')],
            'spy_alpha_pct': [5.0, float('nan')],
        })

        rankings = rank_sales(signals, horizon=90)

        self.assertEqual(len(rankings), 1)
        self.assertEqual(rankings.iloc[0]['member'], 'Alice')
        self.assertFalse(np.isnan(rankings.iloc[0]['avg_loss_avoided_pct']))

    def test_rank_sales_rewards_post_sale_declines(self):
        signals = pd.DataFrame({
            'member': ['Good Seller', 'Bad Seller'],
            'ticker': ['AAPL', 'GOOGL'],
            'signal_type': ['Sale', 'Sale'],
            'horizon_days': [90, 90],
            'decayed_return_pct': [-10.0, 10.0],
            'peak_potential_pct': [10.0, 0.0],
            'spy_alpha_pct': [-5.0, 5.0],
        })

        rankings = rank_sales(signals, horizon=90)

        self.assertEqual(rankings.iloc[0]['member'], 'Good Seller')
        self.assertEqual(rankings.iloc[0]['avg_loss_avoided_pct'], 10.0)
        self.assertEqual(rankings.iloc[0]['avg_spy_alpha_pct'], 5.0)

    def test_missing_price_windows_do_not_count_as_zero_return_trades(self):
        entry_prices = pd.DataFrame({
            'member': ['Alice', 'Alice'],
            'ticker': ['AAPL', 'MSFT'],
            'disclosure_date': pd.to_datetime(['2024-01-01', '2024-06-01']),
            'transaction_type': ['Purchase', 'Purchase'],
            'entry_price': [100.0, 200.0],
        })
        prices = pd.DataFrame({
            'AAPL': [100.0, 110.0],
            'MSFT': [np.nan, np.nan],
            'SPY': [100.0, 100.0],
        }, index=pd.to_datetime(['2024-01-01', '2024-01-02']))

        signals = calculate_signal_potential(entry_prices, prices, [30])
        rankings = rank_members(signals, horizon=30)

        self.assertTrue(np.isnan(signals.loc[signals['ticker'] == 'MSFT', 'decayed_return_pct'].iloc[0]))
        self.assertEqual(rankings.iloc[0]['purchase_trades'], 1)

    def test_sale_peak_potential_no_nan_with_valid_data(self):
        transactions = pd.DataFrame({
            'member': ['Alice'],
            'ticker': ['AAPL'],
            'disclosure_date': pd.to_datetime(['2024-01-15']),
            'transaction_type': ['Sale'],
            'owner_code': [None],
            'amount_midpoint': [50000.0],
        })

        dates = pd.date_range('2023-12-15', '2024-02-15', freq='D')
        np.random.seed(99)
        prices = pd.DataFrame({
            'AAPL': 100 + np.cumsum(np.random.randn(len(dates)) * 0.5),
        }, index=dates)

        entry_prices = make_entry_prices(transactions, prices)
        signals = calculate_signal_potential(entry_prices, prices, [90])

        sales = signals[signals['signal_type'] == 'Sale']
        self.assertEqual(len(sales), 1)
        self.assertFalse(np.isnan(sales.iloc[0]['peak_potential_pct']))


class TestEpisodeCollapse(unittest.TestCase):

    def test_collapses_same_ticker_close_dates_into_single_episode(self):
        signals = pd.DataFrame({
            "member": ["Alice"] * 3,
            "ticker": ["AAPL"] * 3,
            "signal_type": ["Purchase"] * 3,
            "horizon_days": [90] * 3,
            "disclosure_date": pd.to_datetime(["2024-01-01", "2024-01-10", "2024-01-15"]),
            "decayed_return_pct": [10.0, 12.0, 8.0],
            "spy_alpha_pct": [5.0, 7.0, 3.0],
            "total_return_pct": [15.0, 18.0, 12.0],
            "total_spy_alpha_pct": [10.0, 13.0, 7.0],
            "peak_potential_pct": [20.0, 25.0, 15.0],
            "entry_price": [100.0, 100.0, 100.0],
            "owner_code": ["S"] * 3,
            "amount_midpoint": [1000.0, 4000.0, 1000.0],
        })
        collapsed = _collapse_to_episodes(signals)
        self.assertEqual(len(collapsed), 1)
        self.assertEqual(collapsed.iloc[0]["episode_count"], 3)
        self.assertAlmostEqual(collapsed.iloc[0]["decayed_return_pct"], 11.0)

    def test_keeps_different_tickers_as_separate_episodes(self):
        signals = pd.DataFrame({
            "member": ["Alice", "Alice"],
            "ticker": ["AAPL", "GOOGL"],
            "signal_type": ["Purchase", "Purchase"],
            "horizon_days": [90] * 2,
            "disclosure_date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "decayed_return_pct": [10.0, 20.0],
            "spy_alpha_pct": [5.0, 15.0],
            "total_return_pct": [15.0, 25.0],
            "total_spy_alpha_pct": [10.0, 20.0],
            "peak_potential_pct": [20.0, 30.0],
            "entry_price": [100.0, 200.0],
            "owner_code": [None, None],
            "amount_midpoint": [None, None],
        })
        collapsed = _collapse_to_episodes(signals)
        self.assertEqual(len(collapsed), 2)

    def test_splits_same_ticker_far_apart_dates(self):
        signals = pd.DataFrame({
            "member": ["Alice"] * 2,
            "ticker": ["AAPL"] * 2,
            "signal_type": ["Purchase"] * 2,
            "horizon_days": [90] * 2,
            "disclosure_date": pd.to_datetime(["2024-01-01", "2024-02-15"]),
            "decayed_return_pct": [10.0, 20.0],
            "spy_alpha_pct": [5.0, 15.0],
            "total_return_pct": [15.0, 25.0],
            "total_spy_alpha_pct": [10.0, 20.0],
            "peak_potential_pct": [20.0, 30.0],
            "entry_price": [100.0, 100.0],
            "owner_code": [None, None],
            "amount_midpoint": [None, None],
        })
        collapsed = _collapse_to_episodes(signals)
        self.assertEqual(len(collapsed), 2)

    def test_rank_members_uses_fewer_observations_for_clustered_trades(self):
        signals = pd.DataFrame({
            "member": ["Alice"] * 3 + ["Alice"],
            "ticker": ["AAPL"] * 3 + ["MSFT"],
            "signal_type": ["Purchase"] * 4,
            "horizon_days": [90] * 4,
            "disclosure_date": pd.to_datetime(["2024-01-01", "2024-01-05", "2024-01-10", "2024-01-02"]),
            "decayed_return_pct": [10.0, 12.0, 8.0, 5.0],
            "peak_potential_pct": [15.0] * 4,
            "spy_alpha_pct": [5.0, 7.0, 3.0, 2.0],
            "entry_price": [100.0] * 4,
        })
        rankings = rank_members(signals, horizon=90, threshold=5.0)
        self.assertEqual(rankings.iloc[0]["purchase_trades"], 2)

if __name__ == '__main__':
    unittest.main()
