"""Smoke tests for the sweep script."""
import unittest
from unittest.mock import MagicMock

import pandas as pd

import sweep
from sweep import SweepResult


class TestSweepImports(unittest.TestCase):

    def test_module_imports(self):
        # The module patches sys.argv to avoid typer parsing
        assert callable(sweep.run_single_backtest)
        assert callable(sweep.main)


class TestSweepResult(unittest.TestCase):

    def test_defaults(self):
        r = SweepResult(
            horizon=60, frequency_days=30, training_lookback_days=365,
            min_buyers=2, top_n=5, decay_lambda=0.005, bayes_prior_strength=20,
        )
        # All metric fields default to 0
        self.assertEqual(r.total_recs, 0)
        self.assertEqual(r.overall_alpha, 0.0)
        self.assertEqual(r.sharpe, 0.0)
        self.assertEqual(r.scoring_mode, "shrunk_alpha")

    def test_override_metrics(self):
        r = SweepResult(
            horizon=60, frequency_days=30, training_lookback_days=365,
            min_buyers=2, top_n=5, decay_lambda=0.005, bayes_prior_strength=20,
            total_recs=100, overall_alpha=5.2, win_rate=0.65,
        )
        self.assertEqual(r.total_recs, 100)
        self.assertEqual(r.overall_alpha, 5.2)
        self.assertEqual(r.win_rate, 0.65)


class TestEvalCombo(unittest.TestCase):

    def test_eval_combo_calls_run_single_backtest(self):
        from sweep import _eval_combo

        params_dict = {
            "horizon": 60, "frequency_days": 30, "training_lookback_days": 365,
            "min_buyers": 2, "top_n": 5, "decay_lambda": 0.005,
            "bayes_prior_strength": 20, "scoring_mode": "shrunk_alpha",
        }
        keys = list(params_dict.keys())
        signal_cache = {(60, 0.005): pd.DataFrame({"x": [1]})}
        all_tx = pd.DataFrame()
        prices = pd.DataFrame()

        with __import__("unittest.mock", fromlist=["patch"]).patch.object(
            sweep, "run_single_backtest", return_value=SweepResult(
                horizon=60, frequency_days=30, training_lookback_days=365,
                min_buyers=2, top_n=5, decay_lambda=0.005, bayes_prior_strength=20,
                overall_alpha=2.5,
            ),
        ):
            result = _eval_combo(params_dict, keys, signal_cache, all_tx, prices)
        self.assertEqual(result.overall_alpha, 2.5)


if __name__ == "__main__":
    unittest.main()