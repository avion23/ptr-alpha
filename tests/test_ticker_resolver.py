import unittest
from datetime import date

from analyzer.ticker_resolver import TickerResolver


class TestTickerResolver(unittest.TestCase):

    def setUp(self):
        self.resolver = TickerResolver()

    def test_brk_b_to_brk_b(self):
        resolution = self.resolver.resolve("BRK.B")
        self.assertEqual(resolution.price_symbol, "BRK-B")
        self.assertEqual(resolution.status, "class_share")
        self.assertAlmostEqual(resolution.confidence, 1.0)

    def test_brk_a_to_brk_a(self):
        resolution = self.resolver.resolve("BRK.A")
        self.assertEqual(resolution.price_symbol, "BRK-A")
        self.assertEqual(resolution.status, "class_share")

    def test_bare_brk_defaults_to_brk_b(self):
        resolution = self.resolver.resolve("BRK")
        self.assertEqual(resolution.price_symbol, "BRK-B")
        self.assertEqual(resolution.status, "class_share")

    def test_bf_b_to_bf_b(self):
        resolution = self.resolver.resolve("BF.B")
        self.assertEqual(resolution.price_symbol, "BF-B")
        self.assertEqual(resolution.status, "class_share")

    def test_fb_to_meta_with_pre_rename_date(self):
        resolution = self.resolver.resolve("FB", trade_date=date(2021, 1, 1))
        self.assertEqual(resolution.price_symbol, "FB")
        self.assertEqual(resolution.status, "delisted")
        self.assertLess(resolution.confidence, 0.5)

    def test_fb_to_meta_with_post_rename_date(self):
        resolution = self.resolver.resolve("FB", trade_date=date(2022, 1, 1))
        self.assertEqual(resolution.price_symbol, "META")
        self.assertEqual(resolution.status, "renamed")
        self.assertAlmostEqual(resolution.confidence, 1.0)

    def test_fb_to_meta_no_date(self):
        resolution = self.resolver.resolve("FB")
        self.assertEqual(resolution.price_symbol, "META")
        self.assertEqual(resolution.status, "renamed")

    def test_aapl_already_valid(self):
        resolution = self.resolver.resolve("AAPL")
        self.assertEqual(resolution.price_symbol, "AAPL")
        self.assertEqual(resolution.status, "valid")
        self.assertAlmostEqual(resolution.confidence, 1.0)

    def test_unknown_ticker_unresolved(self):
        resolution = self.resolver.resolve("ZZZZZ")
        self.assertEqual(resolution.price_symbol, "ZZZZZ")
        self.assertEqual(resolution.status, "valid")
        self.assertAlmostEqual(resolution.confidence, 1.0)


    def test_case_insensitive(self):
        resolution = self.resolver.resolve("brk.b")
        self.assertEqual(resolution.price_symbol, "BRK-B")
        self.assertEqual(resolution.status, "class_share")

    def test_resolve_batch_mix(self):
        tickers = ["BRK.B", "FB", "AAPL", "ZZZZZ"]
        results = self.resolver.resolve_batch(tickers)

        self.assertEqual(len(results), 4)
        self.assertEqual(results["BRK.B"].price_symbol, "BRK-B")
        self.assertEqual(results["BRK.B"].status, "class_share")
        self.assertEqual(results["FB"].price_symbol, "META")
        self.assertEqual(results["FB"].status, "renamed")
        self.assertEqual(results["AAPL"].price_symbol, "AAPL")
        self.assertEqual(results["AAPL"].status, "valid")
        self.assertEqual(results["ZZZZZ"].price_symbol, "ZZZZZ")
        self.assertEqual(results["ZZZZZ"].status, "valid")


    def test_atvi_acquired_uses_original_symbol(self):
        resolution = self.resolver.resolve("ATVI")
        self.assertEqual(resolution.price_symbol, "ATVI")
        self.assertEqual(resolution.status, "acquired")
        self.assertIn("MSFT", resolution.notes)

    def test_atvi_acquired_ignores_trade_date(self):
        before = self.resolver.resolve("ATVI", trade_date=date(2020, 1, 1))
        after = self.resolver.resolve("ATVI", trade_date=date(2025, 1, 1))
        self.assertEqual(before.price_symbol, "ATVI")
        self.assertEqual(after.price_symbol, "ATVI")
        self.assertEqual(before.status, "acquired")
        self.assertEqual(after.status, "acquired")

    def test_celg_acquired_uses_original_symbol(self):
        resolution = self.resolver.resolve("CELG")
        self.assertEqual(resolution.price_symbol, "CELG")
        self.assertEqual(resolution.status, "acquired")
        self.assertIn("BMY", resolution.notes)

    def test_bll_to_ball_after_rename_date(self):
        resolution = self.resolver.resolve("BLL", trade_date=date(2022, 6, 13))
        self.assertEqual(resolution.price_symbol, "BALL")
        self.assertEqual(resolution.status, "renamed")

    def test_get_yfinance_tickers_deduplicates(self):
        tickers = ["BRK.B", "BRK.B", "AAPL", "AAPL"]
        yf_tickers = self.resolver.get_yfinance_tickers(tickers)
        self.assertEqual(yf_tickers, ["AAPL", "BRK-B"])

    def test_get_yfinance_tickers_resolves_renames(self):
        tickers = ["FB", "AAPL"]
        yf_tickers = self.resolver.get_yfinance_tickers(tickers)
        self.assertIn("META", yf_tickers)
        self.assertIn("AAPL", yf_tickers)
        self.assertNotIn("FB", yf_tickers)

    def test_resolution_notes_provide_context(self):
        resolution = self.resolver.resolve("BRK.B")
        self.assertIn("BRK.B", resolution.notes)
        self.assertIn("BRK-B", resolution.notes)

    def test_foxa_passthrough(self):
        resolution = self.resolver.resolve("FOXA")
        self.assertEqual(resolution.price_symbol, "FOXA")
        self.assertEqual(resolution.status, "class_share")


if __name__ == "__main__":
    unittest.main()
