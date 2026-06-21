"""Smoke tests for member_profitability.analysis module."""
import unittest

import numpy as np
import pandas as pd


def _make_wf_df():
    """Two-window synthetic walk-forward observations DataFrame."""
    rows: list[dict] = []
    # Must include every metric in METRICS_TO_TEST (member_profitability.config)
    metrics = [
        "shrunk_alpha", "bayes_win_prob", "conviction_score",
        "sharpe_ratio", "prob_up_given_buy", "avg_spy_alpha_pct",
    ]
    rng = np.random.default_rng(42)
    for wi in range(2):
        for i in range(30):
            row: dict = {"window": wi}
            for m in metrics:
                row[m] = float(rng.random())
            row["purchase_trades"] = int(rng.integers(1, 20))
            row["test_alpha"] = float(rng.normal(0, 1))
            rows.append(row)
    return pd.DataFrame(rows)


class TestAnalysisImports(unittest.TestCase):

    def test_module_imports(self):
        import member_profitability.analysis
        self.assertTrue(callable(member_profitability.analysis.spearman_correlations_per_metric))
        self.assertTrue(callable(member_profitability.analysis.tier_analysis))
        self.assertTrue(callable(member_profitability.analysis.trade_count_reliability))
        self.assertTrue(callable(member_profitability.analysis.combined_metrics_analysis))
        self.assertTrue(callable(member_profitability.analysis.summarize_combined_metrics))


class TestSpearmanCorrelations(unittest.TestCase):

    def test_returns_dict_with_metric_keys(self):
        from member_profitability.analysis import spearman_correlations_per_metric
        all_wf = _make_wf_df()
        result = spearman_correlations_per_metric(all_wf)
        self.assertIsInstance(result, dict)
        # Each metric should have a dict value
        for k, v in result.items():
            self.assertIsInstance(v, dict)
            self.assertIn("n_windows", v)

    def test_empty_input_returns_zeroed_results(self):
        from member_profitability.analysis import spearman_correlations_per_metric
        all_wf = pd.DataFrame(columns=[
            "window", "shrunk_alpha", "test_alpha", "purchase_trades",
            "bayes_win_prob", "conviction_score", "sharpe_ratio",
            "prob_up_given_buy", "avg_spy_alpha_pct",
        ])
        result = spearman_correlations_per_metric(all_wf)
        # All metrics should report 0 windows
        for v in result.values():
            self.assertEqual(v["n_windows"], 0)


class TestTierAnalysis(unittest.TestCase):

    def test_returns_dict_with_metric_keys(self):
        from member_profitability.analysis import tier_analysis
        all_wf = _make_wf_df()
        result = tier_analysis(all_wf)
        self.assertIsInstance(result, dict)
        for k, v in result.items():
            self.assertIn("n_observations", v)


class TestTradeCountReliability(unittest.TestCase):

    def test_returns_dict_keyed_by_threshold(self):
        from member_profitability.analysis import trade_count_reliability
        all_wf = _make_wf_df()
        result = trade_count_reliability(all_wf)
        self.assertIsInstance(result, dict)
        # All keys should be ints (thresholds)
        for k, v in result.items():
            self.assertIsInstance(k, int)
            self.assertIsInstance(v, dict)


class TestCombinedMetrics(unittest.TestCase):

    def test_returns_dict_with_three_combinations(self):
        from member_profitability.analysis import combined_metrics_analysis
        all_wf = _make_wf_df()
        result = combined_metrics_analysis(all_wf)
        self.assertIn("combined_v1", result)
        self.assertIn("combined_v2", result)
        self.assertIn("trades_x_winprob", result)

    def test_summarize_handles_empty_combined(self):
        from member_profitability.analysis import summarize_combined_metrics
        result = summarize_combined_metrics({})
        self.assertEqual(result, {})


if __name__ == "__main__":
    unittest.main()
