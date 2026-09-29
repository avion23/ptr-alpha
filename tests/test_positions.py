import unittest
from datetime import date

import pandas as pd

from analyzer.positions import build_positions, discount_alerts, PositionsError


def _prices(days, closes):
    idx = pd.to_datetime(days)
    return pd.DataFrame({"close": closes}, index=idx)


class TestPositions(unittest.TestCase):
    def test_multi_lot_fifo_and_discount(self):
        trades = pd.DataFrame(
            [
                {
                    "member": "M",
                    "ticker": "T",
                    "transaction_type": "Purchase",
                    "transaction_date": date(2025, 1, 14),
                    "amount_midpoint": 16911.0,
                },
                {
                    "member": "M",
                    "ticker": "T",
                    "transaction_type": "Purchase",
                    "transaction_date": date(2026, 1, 16),
                    "amount_midpoint": 16614.0,
                },
            ]
        )
        history = {
            "T": _prices(
                ["2025-01-14", "2025-01-15", "2026-01-16", "2026-09-21"],
                [169.11, 170.0, 166.14, 140.78],
            )
        }
        positions = build_positions(trades, history, as_of=date(2026, 9, 21))
        self.assertEqual(len(positions), 1)
        row = positions.iloc[0]
        self.assertAlmostEqual(row["shares"], 100.0 + 100.0, places=4)
        self.assertAlmostEqual(
            row["cost_basis"], (16911.0 + 16614.0) / 200.0, places=4
        )
        self.assertAlmostEqual(row["current_price"], 140.78, places=4)
        self.assertLess(row["discount_pct"], -10.0)

    def test_partial_sale_relieves_fifo(self):
        trades = pd.DataFrame(
            [
                {
                    "member": "M",
                    "ticker": "T",
                    "transaction_type": "Purchase",
                    "transaction_date": date(2025, 1, 10),
                    "amount_midpoint": 20000.0,
                },
                {
                    "member": "M",
                    "ticker": "T",
                    "transaction_type": "Sale Partial",
                    "transaction_date": date(2025, 6, 10),
                    "amount_midpoint": 10000.0,
                },
            ]
        )
        history = {
            "T": _prices(
                ["2025-01-10", "2025-06-10", "2026-09-21"],
                [100.0, 200.0, 200.0],
            )
        }
        positions = build_positions(trades, history, as_of=date(2026, 9, 21))
        self.assertEqual(len(positions), 1)
        row = positions.iloc[0]
        # Bought 200 @100, sold 50 @200 -> 150 shares, $15k cost.
        self.assertAlmostEqual(row["shares"], 150.0, places=4)
        self.assertAlmostEqual(row["cost_basis"], 100.0, places=4)

    def test_full_close_removes_position(self):
        trades = pd.DataFrame(
            [
                {
                    "member": "M",
                    "ticker": "T",
                    "transaction_type": "Purchase",
                    "transaction_date": date(2025, 1, 10),
                    "amount_midpoint": 10000.0,
                },
                {
                    "member": "M",
                    "ticker": "T",
                    "transaction_type": "Sale Full",
                    "transaction_date": date(2025, 6, 10),
                    "amount_midpoint": 20000.0,
                },
            ]
        )
        history = {
            "T": _prices(["2025-01-10", "2025-06-10"], [100.0, 200.0])
        }
        positions = build_positions(trades, history, as_of=date(2025, 6, 11))
        self.assertTrue(positions.empty)

    def test_missing_price_rows_skipped_not_crash(self):
        trades = pd.DataFrame(
            [
                {
                    "member": "M",
                    "ticker": "NOPE",
                    "transaction_type": "Purchase",
                    "transaction_date": date(2025, 1, 10),
                    "amount_midpoint": 10000.0,
                },
            ]
        )
        positions = build_positions(trades, {}, as_of=date(2025, 6, 11))
        self.assertTrue(positions.empty)

    def test_missing_columns_raise(self):
        with self.assertRaises(PositionsError):
            build_positions(
                pd.DataFrame([{"member": "M"}]), {}, as_of=date(2025, 1, 1)
            )

    def test_discount_threshold(self):
        frame = pd.DataFrame(
            [
                {"member": "M", "ticker": "A", "discount_pct": -17.0},
                {"member": "M", "ticker": "B", "discount_pct": -3.0},
            ]
        )
        alerts = discount_alerts(frame, threshold_pct=-10.0)
        self.assertEqual(list(alerts["ticker"]), ["A"])
        self.assertTrue(discount_alerts(frame.iloc[0:0]).empty)


if __name__ == "__main__":
    unittest.main()
