import unittest
from datetime import date

import pandas as pd

from analyzer.positions import (
    PositionsError,
    build_positions,
    discount_alerts,
    holdings_candidates,
)


def _prices(days, closes):
    idx = pd.to_datetime(days)
    return pd.DataFrame({"close": closes}, index=idx)


def _holdings_db(trades, prices):
    import duckdb
    from types import SimpleNamespace

    conn = duckdb.connect()
    trade_columns = [
        "member",
        "ticker",
        "transaction_type",
        "transaction_date",
        "disclosure_date",
        "amount_midpoint",
        "amount_raw",
        "source",
        "instrument_type",
        "source_record_id",
        "amends_source_record_id",
    ]
    trade_frame = pd.DataFrame(
        [
            {column: row.get(column) for column in trade_columns}
            for row in trades
        ],
        columns=trade_columns,
    )
    price_frame = pd.DataFrame(prices, columns=["ticker", "date", "close"])
    conn.register("trade_fixture", trade_frame)
    conn.register("price_fixture", price_frame)
    conn.execute("CREATE TABLE canonical_transactions AS SELECT * FROM trade_fixture")
    conn.execute("CREATE TABLE prices AS SELECT * FROM price_fixture")
    return SimpleNamespace(conn=conn), conn


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

    def test_bracket_op_description_never_becomes_stock(self):
        # House filings mark options inline as [OP] while instrument_type
        # stays NULL on some rows (Pelosi VST calls filed as such).
        trades = pd.DataFrame(
            [
                {
                    "member": "M",
                    "ticker": "T",
                    "transaction_type": "Purchase",
                    "transaction_date": date(2025, 1, 14),
                    "amount_midpoint": 750000.5,
                    "instrument_type": None,
                    "asset_description": "Vistra Corp. Common Stock (T) [OP]",
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

    def test_sale_full_clears_position(self):
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
                    # Filed amount exceeds implied cost: FIFO alone would
                    # leave phantom shares; Sale Full must clear regardless.
                    "member": "M",
                    "ticker": "T",
                    "transaction_type": "Sale Full",
                    "transaction_date": date(2025, 6, 10),
                    "amount_midpoint": 100.0,
                },
            ]
        )
        history = {"T": _prices(["2025-01-10", "2025-06-10"], [100.0, 200.0])}
        self.assertTrue(
            build_positions(trades, history, as_of=date(2025, 6, 11)).empty
        )

    def test_amended_row_supersedes_original(self):
        trades = pd.DataFrame(
            [
                {
                    "member": "M",
                    "ticker": "T",
                    "transaction_type": "Purchase",
                    "transaction_date": date(2025, 1, 10),
                    "amount_midpoint": 10000.0,
                    "source_record_id": "r1",
                    "amends_source_record_id": None,
                },
                {
                    "member": "M",
                    "ticker": "T",
                    "transaction_type": "Purchase",
                    "transaction_date": date(2025, 1, 10),
                    "amount_midpoint": 20000.0,
                    "source_record_id": "r2",
                    "amends_source_record_id": "r1",
                },
            ]
        )
        history = {"T": _prices(["2025-01-10", "2025-06-10"], [100.0, 100.0])}
        positions = build_positions(trades, history, as_of=date(2025, 6, 11))
        self.assertEqual(len(positions), 1)
        # Only the amendment's $20k counts, not $30k combined.
        self.assertAlmostEqual(
            positions.iloc[0]["total_cost"], 20000.0, places=2
        )

    def test_member_variants_unite_filed_names(self):
        import duckdb
        from types import SimpleNamespace

        from analyzer.positions import _member_variants

        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE transactions AS SELECT * FROM (VALUES "
            "('Charles J. Fleischmann'), ('Charles J. \"Chuck\" Fleischmann'), "
            "('Nancy Pelosi')) AS v(member)"
        )
        variants = _member_variants(
            SimpleNamespace(conn=conn), "Charles J. Fleischmann"
        )
        self.assertIn("Charles J. Fleischmann", variants)
        self.assertIn("Charles J. \"Chuck\" Fleischmann", variants)
        self.assertNotIn("Nancy Pelosi", variants)

    def test_member_variants_keep_conflicting_middle_initials_separate(self):
        import duckdb
        from types import SimpleNamespace

        from analyzer.positions import _member_variants

        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE transactions AS SELECT * FROM (VALUES "
            "('Mike A. Smith'), ('Michael B. Smith')) AS v(member)"
        )
        variants = _member_variants(SimpleNamespace(conn=conn), "Mike A. Smith")
        self.assertEqual(variants, ["Mike A. Smith"])

    def test_sale_full_closes_position_across_name_variants(self):
        trades = pd.DataFrame(
            [
                {
                    "member": "Chuck Fleischmann",
                    "ticker": "ZTS",
                    "transaction_type": "Purchase",
                    "transaction_date": date(2025, 1, 10),
                    "amount_midpoint": 10000.0,
                },
                {
                    "member": "Charles J Fleischmann",
                    "ticker": "ZTS",
                    "transaction_type": "Sale Full",
                    "transaction_date": date(2025, 6, 10),
                    "amount_midpoint": 100.0,
                },
            ]
        )
        history = {
            "ZTS": _prices(["2025-01-10", "2025-06-10"], [100.0, 200.0])
        }
        positions = build_positions(trades, history, as_of=date(2025, 6, 11))
        self.assertTrue(positions.empty)

    def test_nickname_resolves_to_filed_name(self):
        import duckdb
        from types import SimpleNamespace

        from analyzer.positions import _member_variants

        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE transactions AS SELECT * FROM (VALUES "
            "('Charles J Fleischmann')) AS v(member)"
        )
        variants = _member_variants(
            SimpleNamespace(conn=conn), "Chuck Fleischmann"
        )
        self.assertIn("Charles J Fleischmann", variants)

    def test_unpromoted_loader_adds_deduped_raw(self):
        import duckdb
        from types import SimpleNamespace

        from analyzer.positions import load_member_trades

        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE canonical_transactions AS SELECT * FROM "
            "(VALUES ('M', 'T', 'Purchase', DATE '2025-01-14', DATE '2025-02-01', "
            "750000.5, '$500,001 - $1,000,000', 'house_pdf', 'g0', 'a0', 'w0', "
            "'option', NULL)) "
            "AS v(member, ticker, "
            "transaction_type, transaction_date, disclosure_date, amount_midpoint, "
            "amount_raw, source, ingestion_generation, source_record_id, "
            "source_row_id, instrument_type, amends_source_record_id)"
        )
        conn.execute(
            "CREATE TABLE transactions AS SELECT * FROM "
            "(VALUES ('M', 'T', 'Purchase', DATE '2026-01-16', DATE '2026-01-23', "
            "175000.5, '$100,001 - $250,000', 'house_pdf', 'g9', 'a9', 'w9', "
            "'stock', NULL)) "
            "AS v(member, ticker, transaction_type, transaction_date, "
            "disclosure_date, amount_midpoint, amount_raw, source, "
            "ingestion_generation, source_record_id, source_row_id, "
            "instrument_type, amends_source_record_id)"
        )
        db = SimpleNamespace(conn=conn)
        canon_only = load_member_trades(db, "M")
        self.assertEqual(len(canon_only), 1)
        widened = load_member_trades(db, "M", include_unpromoted=True)
        self.assertEqual(len(widened), 2)
        self.assertIn(
            date(2026, 1, 16), set(pd.to_datetime(widened["transaction_date"]).dt.date)
        )

    def test_holdings_candidates_schema_and_option_filter(self):
        db, conn = _holdings_db(
            [
                {
                    "member": "M",
                    "ticker": "T",
                    "transaction_type": "Purchase",
                    "transaction_date": date(2025, 1, 14),
                    "disclosure_date": date(2025, 1, 20),
                    "amount_midpoint": 1000.0,
                    "source": "house_pdf",
                    "instrument_type": "stock",
                },
                {
                    "member": "M",
                    "ticker": "T",
                    "transaction_type": "Purchase",
                    "transaction_date": date(2025, 1, 15),
                    "disclosure_date": date(2025, 1, 20),
                    "amount_midpoint": 1000000.0,
                    "source": "house_pdf",
                    "instrument_type": "option",
                },
            ],
            [("T", date(2025, 1, 14), 100.0), ("T", date(2026, 9, 21), 120.0)],
        )
        try:
            candidates = holdings_candidates(db, "M", date(2026, 9, 21))
        finally:
            conn.close()

        self.assertEqual(
            list(candidates.columns),
            [
                "ticker",
                "actor_id",
                "kind",
                "source",
                "entry_ref",
                "event_date",
                "disclosure_date",
                "corroboration",
                "position_evidence",
                "as_of",
            ],
        )
        self.assertEqual(len(candidates), 1)
        row = candidates.iloc[0]
        self.assertEqual(row["actor_id"], "congress:M")
        self.assertEqual(row["kind"], "congress")
        self.assertEqual(row["entry_ref"], 100.0)
        self.assertFalse(row["corroboration"])

    def test_holdings_candidates_disclosure_cutoff(self):
        db, conn = _holdings_db(
            [
                {
                    "member": "M",
                    "ticker": "T",
                    "transaction_type": "Purchase",
                    "transaction_date": date(2026, 1, 16),
                    "disclosure_date": date(2026, 1, 23),
                    "amount_midpoint": 1000.0,
                    "source": "house_pdf",
                    "instrument_type": "stock",
                }
            ],
            [("T", date(2026, 1, 16), 100.0), ("T", date(2026, 1, 23), 110.0)],
        )
        try:
            before_disclosure = holdings_candidates(db, "M", date(2026, 1, 20))
            on_disclosure = holdings_candidates(db, "M", date(2026, 1, 23))
        finally:
            conn.close()

        self.assertTrue(before_disclosure.empty)
        self.assertEqual(len(on_disclosure), 1)
        self.assertEqual(on_disclosure.iloc[0]["disclosure_date"], date(2026, 1, 23))

    def test_holdings_candidates_13f_has_no_entry_reference(self):
        db, conn = _holdings_db(
            [
                {
                    "member": "Manager M",
                    "ticker": "T",
                    "transaction_type": "Purchase",
                    "transaction_date": date(2025, 1, 14),
                    "disclosure_date": date(2025, 2, 14),
                    "amount_midpoint": 1000.0,
                    "source": "13f",
                    "instrument_type": "stock",
                }
            ],
            [("T", date(2025, 1, 14), 100.0), ("T", date(2026, 9, 21), 120.0)],
        )
        try:
            candidates = holdings_candidates(db, "Manager M", date(2026, 9, 21))
        finally:
            conn.close()

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates.iloc[0]["kind"], "manager")
        self.assertIsNone(candidates.iloc[0]["entry_ref"])

    def test_holdings_candidates_entry_reference_is_blended_basis(self):
        db, conn = _holdings_db(
            [
                {
                    "member": "M",
                    "ticker": "T",
                    "transaction_type": "Purchase",
                    "transaction_date": date(2025, 1, 10),
                    "disclosure_date": date(2025, 1, 12),
                    "amount_midpoint": 2000.0,
                    "source": "house_pdf",
                    "instrument_type": "stock",
                },
                {
                    "member": "M",
                    "ticker": "T",
                    "transaction_type": "Purchase",
                    "transaction_date": date(2025, 6, 10),
                    "disclosure_date": date(2025, 6, 12),
                    "amount_midpoint": 4000.0,
                    "source": "gemini_ocr",
                    "instrument_type": "stock",
                },
            ],
            [
                ("T", date(2025, 1, 10), 100.0),
                ("T", date(2025, 6, 10), 200.0),
                ("T", date(2026, 9, 21), 180.0),
            ],
        )
        try:
            candidates = holdings_candidates(db, "M", date(2026, 9, 21))
        finally:
            conn.close()

        self.assertEqual(len(candidates), 1)
        row = candidates.iloc[0]
        self.assertAlmostEqual(row["entry_ref"], 150.0)
        self.assertEqual(row["source"], "gemini_ocr")
        self.assertEqual(row["event_date"], date(2025, 6, 10))
        self.assertEqual(row["disclosure_date"], date(2025, 6, 12))

    def test_dedupe_member_names_unites_variants(self):
        from analyzer.positions import dedupe_member_names

        self.assertEqual(
            dedupe_member_names(
                ["Michael McCaul", "Michael T. McCaul", "Nancy Pelosi"]
            ),
            ["Michael McCaul", "Nancy Pelosi"],
        )
        self.assertEqual(
            dedupe_member_names(
                ["Chuck Fleischmann", "Charles J. Fleischmann"]
            ),
            ["Chuck Fleischmann"],
        )
        self.assertEqual(
            dedupe_member_names(["Mike McCaul"], exclude=["Michael T. McCaul"]),
            [],
        )

    def test_gate_blocks_closed_officer_and_passes_open(self):
        from analyzer.positions import gate_closed_events

        events = pd.DataFrame(
            [
                {
                    "ticker": "T",
                    "actor_id": "officer:OPEN CEO",
                    "kind": "officer",
                    "entry_ref": 100.0,
                },
                {
                    "ticker": "T",
                    "actor_id": "officer:SHUT CEO",
                    "kind": "officer",
                    "entry_ref": 100.0,
                },
            ]
        )
        candidates = pd.DataFrame(
            [
                {
                    "ticker": "T",
                    "actor_id": "officer:OPEN CEO",
                    "position_evidence": True,
                },
            ]
        )
        gated = gate_closed_events(events, candidates)
        self.assertEqual(
            gated.loc[gated["actor_id"] == "officer:OPEN CEO", "entry_ref"].iloc[0],
            100.0,
        )
        shut = gated.loc[gated["actor_id"] == "officer:SHUT CEO"].iloc[0]
        self.assertTrue(pd.isna(shut["entry_ref"]))
        self.assertEqual(shut["blocked_reason"], "no open position at as_of")

    def test_filed_share_counts_close_exactly(self):
        from analyzer.positions import build_positions

        trades = pd.DataFrame(
            [
                {
                    "member": "M",
                    "ticker": "T",
                    "transaction_type": "Purchase",
                    "transaction_date": date(2026, 8, 31),
                    "amount_midpoint": 10000.0,
                    "instrument_type": "Common Stock",
                    "asset_description": "Common Stock; 100 shares @ $100.00; is_10b5_1=false",
                },
                {
                    "member": "M",
                    "ticker": "T",
                    "transaction_type": "Sale",
                    "transaction_date": date(2026, 9, 15),
                    "amount_midpoint": 9000.0,
                    "instrument_type": "Common Stock",
                    "asset_description": "Common Stock; 100 shares @ $90.00; is_10b5_1=false",
                },
            ]
        )
        history = {
            "T": _prices(
                ["2026-08-31", "2026-09-15", "2026-09-21"], [100.0, 100.0, 100.0]
            )
        }
        self.assertTrue(
            build_positions(trades, history, as_of=date(2026, 9, 21)).empty
        )


if __name__ == "__main__":
    unittest.main()
