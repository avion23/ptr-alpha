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

    def test_open_band_falls_back_to_floor(self):
        trades = pd.DataFrame(
            [
                {
                    "member": "M",
                    "ticker": "T",
                    "transaction_type": "Purchase",
                    "transaction_date": date(2026, 1, 16),
                    "amount_midpoint": None,
                    "amount_raw": "$100,001 -",
                },
            ]
        )
        history = {"T": _prices(["2026-01-16", "2026-09-21"], [166.14, 140.78])}
        positions = build_positions(trades, history, as_of=date(2026, 9, 21))
        self.assertEqual(len(positions), 1)
        row = positions.iloc[0]
        self.assertAlmostEqual(row["total_cost"], 100001.0, places=2)
        self.assertEqual(row["floor_sized_lots"], 1)
        self.assertLess(row["discount_pct"], -10.0)

    def test_unparseable_raw_still_skipped(self):
        trades = pd.DataFrame(
            [
                {
                    "member": "M",
                    "ticker": "T",
                    "transaction_type": "Purchase",
                    "transaction_date": date(2026, 1, 16),
                    "amount_midpoint": None,
                    "amount_raw": "garbage",
                },
            ]
        )
        history = {"T": _prices(["2026-01-16"], [166.14])}
        self.assertTrue(
            build_positions(trades, history, as_of=date(2026, 1, 17)).empty
        )

    def test_option_rows_never_become_stock(self):
        trades = pd.DataFrame(
            [
                {
                    "member": "M",
                    "ticker": "T",
                    "transaction_type": "Purchase",
                    "transaction_date": date(2025, 1, 14),
                    "amount_midpoint": 750000.5,
                    "instrument_type": "option",
                },
            ]
        )
        history = {"T": _prices(["2025-01-14", "2026-09-21"], [169.11, 140.78])}
        self.assertTrue(
            build_positions(trades, history, as_of=date(2026, 9, 21)).empty
        )

    def test_undisclosed_rows_excluded_as_of(self):
        trades = pd.DataFrame(
            [
                {
                    "member": "M",
                    "ticker": "T",
                    "transaction_type": "Purchase",
                    "transaction_date": date(2026, 1, 16),
                    "disclosure_date": date(2026, 1, 23),
                    "amount_midpoint": 175000.5,
                },
            ]
        )
        history = {"T": _prices(["2026-01-16", "2026-09-21"], [166.14, 140.78])}
        # As of a date before disclosure, the position must not exist yet.
        self.assertTrue(
            build_positions(trades, history, as_of=date(2026, 1, 20)).empty
        )
        visible = build_positions(trades, history, as_of=date(2026, 9, 21))
        self.assertEqual(len(visible), 1)

    def test_unpromoted_loader_adds_deduped_raw(self):
        import duckdb
        from types import SimpleNamespace

        from analyzer.positions import load_member_trades

        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE canonical_transactions AS SELECT * FROM "
            "(VALUES ('M', 'T', 'Purchase', DATE '2025-01-14', DATE '2025-02-01', "
            "750000.5, '$500,001 - $1,000,000', 'house_pdf', 'g0', 'a0', 'w0', "
            "'option')) "
            "AS v(member, ticker, "
            "transaction_type, transaction_date, disclosure_date, amount_midpoint, "
            "amount_raw, source, ingestion_generation, source_record_id, "
            "source_row_id, instrument_type)"
        )
        conn.execute(
            "CREATE TABLE transactions AS SELECT * FROM "
            "(VALUES ('M', 'T', 'Purchase', DATE '2026-01-16', DATE '2026-01-23', "
            "175000.5, '$100,001 - $250,000', 'house_pdf', 'g9', 'a9', 'w9', "
            "'stock')) "
            "AS v(member, ticker, transaction_type, transaction_date, "
            "disclosure_date, amount_midpoint, amount_raw, source, "
            "ingestion_generation, source_record_id, source_row_id, "
            "instrument_type)"
        )
        db = SimpleNamespace(conn=conn)
        canon_only = load_member_trades(db, "M")
        self.assertEqual(len(canon_only), 1)
        widened = load_member_trades(db, "M", include_unpromoted=True)
        self.assertEqual(len(widened), 2)
        self.assertIn(
            date(2026, 1, 16), set(pd.to_datetime(widened["transaction_date"]).dt.date)
        )


if __name__ == "__main__":
    unittest.main()
