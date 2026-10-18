"""End-product freshness checks: the surface the 2026-10-18 cron incident
showed was measured nowhere (writers exited 0 while canonical starved)."""

import duckdb
from datetime import date, datetime, timedelta, timezone
import json

import pandas as pd
from unittest import TestCase

from analyzer.freshness import (
    end_to_end_checks,
    dividends_populated,
    form4_freshness,
    price_freshness,
    receipts_freshness,
)
from analyzer.price_repository import previous_nyse_session


def _fresh_session() -> date:
    return previous_nyse_session(date.today() - timedelta(days=1)).date()


def _conn_with_prices(max_date) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE prices (ticker VARCHAR, date DATE, close DOUBLE)")
    if max_date is not None:
        con.execute(
            "INSERT INTO prices VALUES ('AAPL', ?, 1.0)", [max_date]
        )
    return con


class PriceFreshnessTest(TestCase):
    def test_current_prices_pass(self):
        con = _conn_with_prices(_fresh_session())
        check = price_freshness(con)
        self.assertTrue(check.ok, check.detail)

    def test_stale_prices_fail(self):
        con = _conn_with_prices(_fresh_session() - timedelta(days=30))
        check = price_freshness(con)
        self.assertFalse(check.ok)
        self.assertIn("sessions back", check.detail)

    def test_empty_prices_fail_closed(self):
        con = _conn_with_prices(None)
        self.assertFalse(price_freshness(con).ok)


class DividendsPopulatedTest(TestCase):
    def test_zero_rows_fail(self):
        con = duckdb.connect(":memory:")
        con.execute(
            "CREATE TABLE dividends (ticker VARCHAR, ex_date DATE, cash_amount DOUBLE)"
        )
        check = dividends_populated(con)
        self.assertFalse(check.ok)
        self.assertEqual("dividends rows: 0", check.detail)

    def test_populated_pass(self):
        con = duckdb.connect(":memory:")
        con.execute(
            "CREATE TABLE dividends (ticker VARCHAR, ex_date DATE, cash_amount DOUBLE)"
        )
        con.execute("INSERT INTO dividends VALUES ('AAPL', '2026-10-01', 0.5)")
        self.assertTrue(dividends_populated(con).ok)

    def test_absent_table_fails_closed(self):
        con = duckdb.connect(":memory:")
        check = dividends_populated(con)
        self.assertFalse(check.ok)
        self.assertIn("table absent", check.detail)


class Form4FreshnessTest(TestCase):
    def test_recent_disclosure_passes(self):
        con = duckdb.connect(":memory:")
        con.execute(
            "CREATE TABLE canonical_transactions (source VARCHAR, disclosure_date DATE)"
        )
        con.execute(
            "INSERT INTO canonical_transactions VALUES ('form4', ?)",
            [date.today() - timedelta(days=2)],
        )
        self.assertTrue(form4_freshness(con).ok)

    def test_silence_fails(self):
        con = duckdb.connect(":memory:")
        con.execute(
            "CREATE TABLE canonical_transactions (source VARCHAR, disclosure_date DATE)"
        )
        self.assertFalse(form4_freshness(con).ok)


class ReceiptsFreshnessTest(TestCase):
    def test_recent_receipt_passes(self):
        path = self._ledger(datetime.now(timezone.utc))
        self.assertTrue(receipts_freshness(path).ok)

    def test_old_receipt_fails(self):
        path = self._ledger(datetime.now(timezone.utc) - timedelta(days=30))
        self.assertFalse(receipts_freshness(path).ok)

    def test_missing_ledger_fails_closed(self):
        self.assertFalse(receipts_freshness(None).ok)

    @staticmethod
    def _ledger(published) -> str:
        import tempfile

        handle = tempfile.NamedTemporaryFile("w", suffix=".log", delete=False)
        handle.write(json.dumps({"published_at": published.isoformat()}) + "\n")
        handle.close()
        return handle.name


class EndToEndChecksTest(TestCase):
    def test_never_raises_on_unexpected_schema(self):
        con = duckdb.connect(":memory:")  # no tables at all
        checks = end_to_end_checks(con, receipts_path=None)
        self.assertEqual(
            ["prices.freshness", "dividends.populated", "form4.freshness",
             "receipts.freshness"],
            [c.name for c in checks],
        )
        self.assertTrue(all(not c.ok for c in checks))
        self.assertTrue(all("unavailable" in c.detail or c.detail for c in checks))

    def test_fed_pipeline_passes_every_check(self):
        con = duckdb.connect(":memory:")
        con.execute("CREATE TABLE prices (ticker VARCHAR, date DATE, close DOUBLE)")
        con.execute("INSERT INTO prices VALUES ('AAPL', ?, 1.0)", [_fresh_session()])
        con.execute(
            "CREATE TABLE dividends (ticker VARCHAR, ex_date DATE, cash_amount DOUBLE)"
        )
        con.execute("INSERT INTO dividends VALUES ('AAPL', '2026-10-01', 0.5)")
        con.execute(
            "CREATE TABLE canonical_transactions (source VARCHAR, disclosure_date DATE)"
        )
        con.execute(
            "INSERT INTO canonical_transactions VALUES ('form4', ?)",
            [date.today() - timedelta(days=1)],
        )
        checks = {
            c.name: c
            for c in end_to_end_checks(
                con, receipts_path=self._passing_ledger()
            )
        }
        self.assertTrue(all(c.ok for c in checks.values()), dict(checks))

    @staticmethod
    def _passing_ledger() -> str:
        import tempfile

        handle = tempfile.NamedTemporaryFile("w", suffix=".log", delete=False)
        handle.write(
            json.dumps(
                {"published_at": datetime.now(timezone.utc).isoformat()}
            )
            + "\n"
        )
        handle.close()
        return handle.name
