"""Smoke tests for analyzer.portfolio.kelly submodule."""
import unittest





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






if __name__ == "__main__":
    unittest.main()
