"""The merge's read-back tripwire: a writer may only exit 0 when canonical
verifiably contains what was refreshed (the 2026-10-18 gap was a merge step
that silently did not exist)."""

import duckdb
from datetime import date
from pathlib import Path
import sys
from unittest import TestCase

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from analyzer.database import Database  # noqa: E402
from scripts.refresh_prices import (  # noqa: E402
    merge_temp_into_canonical,
    verify_consumption,
)


def _make_temp(path: Path) -> None:
    con = duckdb.connect(str(path))
    con.execute("CREATE TABLE prices (ticker VARCHAR, date DATE, close DOUBLE)")
    con.execute("INSERT INTO prices VALUES ('AAPL', '2026-10-14', 1.0)")
    con.execute("INSERT INTO prices VALUES ('AAPL', '2026-10-15', 2.0)")
    con.execute("INSERT INTO prices VALUES ('MSFT', '2026-10-15', 3.0)")
    con.execute(
        "CREATE TABLE dividends (ticker VARCHAR, ex_date DATE, cash_amount DOUBLE)"
    )
    con.execute("INSERT INTO dividends VALUES ('AAPL', '2026-10-01', 0.25)")
    con.close()


class VerifyConsumptionTest(TestCase):
    def setUp(self):
        import tempfile

        self.tmp = Path(tempfile.mkdtemp(prefix="verify-consumed-"))
        self.temp_db = self.tmp / "prices.duckdb"
        _make_temp(self.temp_db)
        self.canonical_db = self.tmp / "congress.duckdb"
        db = Database(self.canonical_db)  # writable init creates full schema
        db.conn.close()

    def test_merge_satisfies_verification(self):
        stats = merge_temp_into_canonical(self.temp_db, self.canonical_db, ["AAPL"])
        self.assertEqual(2, stats["prices_added"])
        self.assertEqual(1, stats["dividends_added"])
        self.assertEqual(
            0, verify_consumption(self.temp_db, self.canonical_db, ["AAPL"])
        )

    def test_verification_detects_unmerged_keys(self):
        # AAPL: 2 missing price keys + 1 missing dividend key.
        self.assertEqual(
            3, verify_consumption(self.temp_db, self.canonical_db, ["AAPL"])
        )

    def test_verification_scopes_to_eligible_universe(self):
        merge_temp_into_canonical(self.temp_db, self.canonical_db, ["AAPL"])
        # MSFT was refreshed but not eligible: its absence is not a gap.
        self.assertEqual(
            0, verify_consumption(self.temp_db, self.canonical_db, ["AAPL"])
        )
        # ...until it is declared eligible (1 missing MSFT price key).
        self.assertEqual(
            1, verify_consumption(self.temp_db, self.canonical_db, ["MSFT", "AAPL"])
        )

    def test_canonical_row_loss_is_caught_after_merge(self):
        merge_temp_into_canonical(self.temp_db, self.canonical_db, ["AAPL"])
        con = duckdb.connect(str(self.canonical_db))
        con.execute("DELETE FROM prices WHERE date = '2026-10-15'")
        con.close()
        self.assertEqual(
            1, verify_consumption(self.temp_db, self.canonical_db, ["AAPL"])
        )
