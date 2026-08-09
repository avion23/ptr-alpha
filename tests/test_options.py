"""Option return safety tests."""

import unittest

from analyzer.options import UnsupportedOptionPricingError, estimate_options_leverage


class TestEstimateOptionsLeverage(unittest.TestCase):
    def test_stock_uses_observed_underlying_return(self):
        self.assertEqual(estimate_options_leverage("stock"), 1.0)

    def test_options_require_actual_contract_prices(self):
        for instrument in ("call", "put", "option", "Stock Option"):
            with self.subTest(instrument=instrument):
                with self.assertRaisesRegex(
                    UnsupportedOptionPricingError, "contract prices"
                ):
                    estimate_options_leverage(instrument, amount_midpoint=10_000_000)

    def test_unknown_instrument_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported instrument"):
            estimate_options_leverage("government securities")


if __name__ == "__main__":
    unittest.main()
