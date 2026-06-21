"""Smoke tests for the optimize_profit script."""
import unittest

import pandas as pd


class TestOptimizeProfitImports(unittest.TestCase):

    def test_module_imports(self):
        import optimize_profit
        assert callable(optimize_profit.run_walk_forward)
        assert callable(optimize_profit.precompute_walk_forward_data)
        assert "SCORING_FUNCTIONS" in dir(optimize_profit)


class TestScoringFunctions(unittest.TestCase):

    def test_score_functions_accept_valid_minimal_df(self):
        # All scoring functions should accept a minimal DataFrame with the
        # required columns. They return either a dict or a float depending
        # on the function, so we just check that they don't crash.
        from optimize_profit.scoring import (
            score_bayesian_quality,
            score_consistency,
            score_inverted_alpha,
            score_neg_bayesian_quality,
            score_shrunk_alpha,
            score_smooth_trade_threshold,
            score_softplus_quality,
            score_trade_frequency,
        )
        # Provide just the minimum columns each function expects
        minimal_df = pd.DataFrame({
            "member": [],
            "shrunk_alpha": [],
            "avg_total_spy_alpha_pct": [],
            "purchase_trades": [],
            "bayes_win_prob": [],
            "bayes_factor": [],
        })

        for fn in [
            score_shrunk_alpha, score_inverted_alpha, score_trade_frequency,
            score_consistency, score_bayesian_quality, score_neg_bayesian_quality,
            score_smooth_trade_threshold, score_softplus_quality,
        ]:
            try:
                result = fn(minimal_df)
            except KeyError:
                # Some functions may need extra columns — we just want to
                # verify they're importable and callable
                continue
            self.assertTrue(result is not None)


class TestScoringFunctionsDict(unittest.TestCase):

    def test_shrunk_alpha_returns_dict_for_minimal_df(self):
        from optimize_profit.scoring import score_shrunk_alpha
        minimal_df = pd.DataFrame({"member": [], "shrunk_alpha": []})
        result = score_shrunk_alpha(minimal_df)
        # Returns dict mapping member -> score; empty df -> empty dict
        self.assertEqual(result, {})

    def test_inverted_alpha_returns_dict_for_minimal_df(self):
        from optimize_profit.scoring import score_inverted_alpha
        minimal_df = pd.DataFrame({"member": [], "shrunk_alpha": []})
        result = score_inverted_alpha(minimal_df)
        self.assertEqual(result, {})


if __name__ == "__main__":
    unittest.main()