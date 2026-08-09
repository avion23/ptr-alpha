import unittest
from datetime import date

from analyzer.ticker_resolver import TickerResolver


class TestTickerResolver(unittest.TestCase):
    def setUp(self):
        self.resolver = TickerResolver()

    def test_class_share_canaries(self):
        expected = {
            "BRK": "BRK-B",
            "BRKB": "BRK-B",
            "BRK.B": "BRK-B",
            "BRK.A": "BRK-A",
            "BF.B": "BF-B",
            "BF.A": "BF-A",
        }
        for raw, symbol in expected.items():
            with self.subTest(raw=raw):
                result = self.resolver.resolve(raw)
                self.assertEqual(result.price_symbol, symbol)
                self.assertEqual(result.status, "class_share")
                self.assertEqual(result.confidence, 1.0)

    def test_alias_requires_trade_date(self):
        result = self.resolver.resolve("FB")
        self.assertEqual(result.price_symbol, "FB")
        self.assertEqual(result.status, "date_required")
        self.assertFalse(self.resolver.is_strategy_eligible("FB"))

    def test_fb_uses_contemporaneous_symbol(self):
        before = self.resolver.resolve("FB", date(2022, 6, 8))
        after = self.resolver.resolve("FB", date(2022, 6, 9))
        self.assertEqual((before.price_symbol, before.status), ("FB", "pre_rename"))
        self.assertEqual((after.price_symbol, after.status), ("META", "renamed"))

    def test_sq_uses_ticker_change_date(self):
        before = self.resolver.resolve("SQ", date(2025, 1, 20))
        after = self.resolver.resolve("SQ", date(2025, 1, 21))
        self.assertEqual(before.price_symbol, "SQ")
        self.assertEqual(after.price_symbol, "XYZ")

    def test_bll_uses_ticker_change_date(self):
        before = self.resolver.resolve("BLL", date(2022, 10, 31))
        after = self.resolver.resolve("BLL", date(2022, 11, 1))
        self.assertEqual(before.price_symbol, "BLL")
        self.assertEqual(after.price_symbol, "BALL")

    def test_unknown_symbol_is_not_claimed_verified(self):
        for ticker in ("AAPL", "ZZZZZ"):
            result = self.resolver.resolve(ticker)
            self.assertEqual(result.price_symbol, ticker)
            self.assertEqual(result.status, "unverified")
            self.assertEqual(result.confidence, 0.0)

    def test_acquisition_is_date_aware_and_never_maps_acquirer(self):
        missing_date = self.resolver.resolve("ATVI")
        before = self.resolver.resolve("ATVI", date(2023, 10, 12))
        after = self.resolver.resolve("ATVI", date(2023, 10, 14))
        self.assertEqual(missing_date.status, "date_required")
        self.assertEqual(before.status, "pre_acquisition")
        self.assertEqual(after.status, "acquired")
        self.assertEqual(
            {missing_date.price_symbol, before.price_symbol, after.price_symbol},
            {"ATVI"},
        )
        self.assertFalse(self.resolver.is_strategy_eligible("ATVI", date(2023, 10, 14)))

    def test_real_pdf_pseudo_canaries_are_quarantined(self):
        # 20034095/20034670: ALLI was stale parse text for ARLP.
        # 20030630: MATT was a Matthews International mutual fund, not Mattel.
        for ticker in ("ALLI", "MATT", "SP", "THE", "NEW"):
            with self.subTest(ticker=ticker):
                result = self.resolver.resolve(ticker)
                self.assertEqual(result.price_symbol, ticker)
                self.assertEqual(result.status, "quarantined")
                self.assertFalse(
                    self.resolver.is_strategy_eligible(ticker, date(2025, 1, 1))
                )

    def test_verified_roblox_parser_artifact(self):
        result = self.resolver.resolve("ROBL")
        self.assertEqual(result.price_symbol, "RBLX")
        self.assertEqual(result.status, "pseudo_ticker")

    def test_get_yfinance_tickers_deduplicates_without_temporal_guess(self):
        tickers = self.resolver.get_yfinance_tickers(["BRK.B", "BRK.B", "AAPL", "FB"])
        self.assertEqual(tickers, ["AAPL", "BRK-B", "FB"])

    def test_case_insensitive(self):
        self.assertEqual(self.resolver.resolve("brk.b").price_symbol, "BRK-B")

    def test_resolution_notes_provide_context(self):
        resolution = self.resolver.resolve("BRK.B")
        self.assertIn("BRK.B", resolution.notes)
        self.assertIn("BRK-B", resolution.notes)


if __name__ == "__main__":
    unittest.main()
