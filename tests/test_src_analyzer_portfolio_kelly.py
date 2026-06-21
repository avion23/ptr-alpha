"""Smoke tests for analyzer.portfolio.kelly submodule."""
import unittest

import pandas as pd


class TestKellyImports(unittest.TestCase):

    def test_module_imports(self):
        import analyzer.portfolio.kelly
        self.assertTrue(callable(analyzer.portfolio.kelly.kelly_fraction))
        self.assertTrue(callable(analyzer.portfolio.kelly.half_kelly))
        self.assertTrue(callable(analyzer.portfolio.kelly.compute_payout_ratio))
        self.assertTrue(callable(analyzer.portfolio.kelly.build_kelly_portfolio))

    def test_kelly_config_dataclass(self):
        from analyzer.portfolio.kelly import KellyConfig
        cfg = KellyConfig()
        self.assertEqual(cfg.capital, 100_000.0)
        self.assertTrue(cfg.use_half_kelly)
        self.assertTrue(cfg.crash_guard)


class TestKellyFractionMath(unittest.TestCase):

    def test_known_formula(self):
        # p=0.6, b=1.25 -> (0.75 - 0.4)/1.25 = 0.28
        from analyzer.portfolio.kelly import kelly_fraction
        self.assertAlmostEqual(kelly_fraction(0.6, 1.25), 0.28, places=4)

    def test_half_kelly_is_half(self):
        from analyzer.portfolio.kelly import half_kelly, kelly_fraction
        f_full = kelly_fraction(0.6, 1.25)
        f_half = half_kelly(0.6, 1.25)
        self.assertAlmostEqual(f_half, f_full / 2.0, places=6)

    def test_negative_edge_returns_zero(self):
        from analyzer.portfolio.kelly import kelly_fraction
        self.assertEqual(kelly_fraction(0.4, 1.0), 0.0)


class TestEmptyPortfolio(unittest.TestCase):

    def test_empty_recommendations_returns_empty(self):
        from analyzer.portfolio.kelly import build_kelly_portfolio
        result = build_kelly_portfolio(pd.DataFrame())
        self.assertTrue(result.empty)
        # Should still have the expected columns
        self.assertIn("ticker", result.columns)
        self.assertIn("weight", result.columns)
        self.assertIn("kelly_fraction", result.columns)


class TestSerializeReady(unittest.TestCase):

    def test_recs_without_member_fills_default(self):
        from analyzer.portfolio.kelly import _prepare_recommendations, KellyConfig
        recs = pd.DataFrame({
            "ticker": ["A"],
            "signal_score": [10.0],
            "crash_prob": [0.0],
        })
        cfg = KellyConfig()
        out = _prepare_recommendations(recs, cfg)
        self.assertIn("member", out.columns)
        self.assertEqual(out["member"].iloc[0], "unknown")
        self.assertIn("_win_rate", out.columns)


if __name__ == "__main__":
    unittest.main()
