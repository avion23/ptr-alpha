import shutil
import tempfile
import unittest
from datetime import date
from pathlib import Path

import pandas as pd

from analyzer.database import Database
from analyzer.price_snapshot import (
    PriceSnapshot,
    create_snapshot,
    save_snapshot,
    load_snapshot,
    compare_snapshots,
)


class TestPriceSnapshot(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.tmp_dir) / "test.duckdb"
        self.db = Database(self.db_path)

    def tearDown(self):
        self.db.close()
        shutil.rmtree(self.tmp_dir)

    def _seed_prices(self):
        dates = pd.bdate_range("2024-01-01", "2024-01-10")
        prices = pd.DataFrame(
            {"AAPL": [180.0 + i for i in range(len(dates))],
             "MSFT": [370.0 + i for i in range(len(dates))]},
            index=dates,
        )
        self.db.upsert_prices(prices)

    def test_create_snapshot_returns_valid_snapshot(self):
        self._seed_prices()
        tickers = ["AAPL", "MSFT"]
        snap = create_snapshot(self.db, tickers, date(2024, 1, 1), date(2024, 1, 15))

        self.assertIsInstance(snap, PriceSnapshot)
        self.assertTrue(snap.snapshot_id)
        self.assertTrue(snap.created_at)
        self.assertTrue(snap.git_sha)
        self.assertTrue(snap.python_version)
        self.assertEqual(snap.requested_tickers, 2)
        self.assertEqual(snap.resolved_tickers, 2)
        self.assertEqual(snap.unresolved_tickers, ())
        self.assertGreater(snap.price_rows, 0)
        self.assertEqual(snap.first_date, "2024-01-01")
        self.assertEqual(snap.last_date, "2024-01-10")
        self.assertIn("AAPL", snap.coverage_by_ticker)
        self.assertIn("MSFT", snap.coverage_by_ticker)

    def test_create_snapshot_counts_unresolved_tickers(self):
        self._seed_prices()
        tickers = ["AAPL", "MSFT", "GOOG"]
        snap = create_snapshot(self.db, tickers, date(2024, 1, 1), date(2024, 1, 15))

        self.assertEqual(snap.resolved_tickers, 2)
        self.assertEqual(snap.unresolved_tickers, ("GOOG",))
        self.assertEqual(snap.requested_tickers, 3)

    def test_create_snapshot_empty_db(self):
        snap = create_snapshot(self.db, ["AAPL"], date(2024, 1, 1), date(2024, 1, 15))

        self.assertEqual(snap.resolved_tickers, 0)
        self.assertEqual(snap.unresolved_tickers, ("AAPL",))
        self.assertEqual(snap.price_rows, 0)
        self.assertEqual(snap.first_date, "")
        self.assertEqual(snap.last_date, "")

    def test_save_load_roundtrip(self):
        self._seed_prices()
        snap = create_snapshot(self.db, ["AAPL", "MSFT"], date(2024, 1, 1), date(2024, 1, 15))
        path = Path(self.tmp_dir) / "snapshot.json"
        save_snapshot(snap, str(path))

        loaded = load_snapshot(str(path))
        self.assertEqual(loaded.snapshot_id, snap.snapshot_id)
        self.assertEqual(loaded.created_at, snap.created_at)
        self.assertEqual(loaded.git_sha, snap.git_sha)
        self.assertEqual(loaded.yfinance_version, snap.yfinance_version)
        self.assertEqual(loaded.python_version, snap.python_version)
        self.assertEqual(loaded.requested_tickers, snap.requested_tickers)
        self.assertEqual(loaded.resolved_tickers, snap.resolved_tickers)
        self.assertEqual(loaded.unresolved_tickers, snap.unresolved_tickers)
        self.assertEqual(loaded.price_rows, snap.price_rows)
        self.assertEqual(loaded.first_date, snap.first_date)
        self.assertEqual(loaded.last_date, snap.last_date)
        self.assertEqual(loaded.coverage_by_ticker, snap.coverage_by_ticker)

    def test_save_creates_parent_directory(self):
        snap = PriceSnapshot(
            snapshot_id="test-id",
            created_at="2024-01-01T00:00:00",
            git_sha="abc123",
            yfinance_version="0.2.0",
            python_version="3.11.0",
            requested_tickers=1,
            resolved_tickers=1,
            unresolved_tickers=(),
            price_rows=1,
            first_date="2024-01-01",
            last_date="2024-01-01",
            coverage_by_ticker={"A": {"first": "2024-01-01", "last": "2024-01-01", "days": 1, "gaps": 0}},
        )
        nested_path = Path(self.tmp_dir) / "sub" / "dir" / "snapshot.json"
        save_snapshot(snap, str(nested_path))
        self.assertTrue(nested_path.exists())

    def test_compare_snapshots_detects_added_tickers(self):
        old = PriceSnapshot(
            snapshot_id="old",
            created_at="2024-01-01",
            git_sha="aaa",
            yfinance_version="1.0",
            python_version="3.11.0",
            requested_tickers=2,
            resolved_tickers=2,
            unresolved_tickers=(),
            price_rows=10,
            first_date="2024-01-01",
            last_date="2024-01-10",
            coverage_by_ticker={
                "AAPL": {"first": "2024-01-01", "last": "2024-01-10", "days": 8, "gaps": 0},
                "MSFT": {"first": "2024-01-01", "last": "2024-01-10", "days": 8, "gaps": 0},
            },
        )
        new = PriceSnapshot(
            snapshot_id="new",
            created_at="2024-02-01",
            git_sha="bbb",
            yfinance_version="1.1",
            python_version="3.11.0",
            requested_tickers=3,
            resolved_tickers=3,
            unresolved_tickers=(),
            price_rows=15,
            first_date="2024-01-01",
            last_date="2024-01-15",
            coverage_by_ticker={
                "AAPL": {"first": "2024-01-01", "last": "2024-01-10", "days": 8, "gaps": 0},
                "MSFT": {"first": "2024-01-01", "last": "2024-01-10", "days": 8, "gaps": 0},
                "GOOG": {"first": "2024-01-05", "last": "2024-01-15", "days": 8, "gaps": 2},
            },
        )
        result = compare_snapshots(old, new)
        self.assertEqual(result["added_tickers"], ["GOOG"])
        self.assertEqual(result["removed_tickers"], [])
        self.assertEqual(result["requested_tickers_diff"], 1)
        self.assertEqual(result["resolved_tickers_diff"], 1)
        self.assertEqual(result["price_rows_diff"], 5)

    def test_compare_snapshots_detects_removed_tickers(self):
        old = PriceSnapshot(
            snapshot_id="old",
            created_at="2024-01-01",
            git_sha="aaa",
            yfinance_version="1.0",
            python_version="3.11.0",
            requested_tickers=2,
            resolved_tickers=2,
            unresolved_tickers=(),
            price_rows=10,
            first_date="2024-01-01",
            last_date="2024-01-10",
            coverage_by_ticker={
                "AAPL": {"first": "2024-01-01", "last": "2024-01-10", "days": 8, "gaps": 0},
                "MSFT": {"first": "2024-01-01", "last": "2024-01-10", "days": 8, "gaps": 0},
            },
        )
        new = PriceSnapshot(
            snapshot_id="new",
            created_at="2024-02-01",
            git_sha="bbb",
            yfinance_version="1.1",
            python_version="3.11.0",
            requested_tickers=1,
            resolved_tickers=1,
            unresolved_tickers=(),
            price_rows=5,
            first_date="2024-01-01",
            last_date="2024-01-10",
            coverage_by_ticker={
                "AAPL": {"first": "2024-01-01", "last": "2024-01-10", "days": 8, "gaps": 0},
            },
        )
        result = compare_snapshots(old, new)
        self.assertEqual(result["added_tickers"], [])
        self.assertEqual(result["removed_tickers"], ["MSFT"])

    def test_compare_snapshots_detects_coverage_changes(self):
        old = PriceSnapshot(
            snapshot_id="old",
            created_at="2024-01-01",
            git_sha="aaa",
            yfinance_version="1.0",
            python_version="3.11.0",
            requested_tickers=1,
            resolved_tickers=1,
            unresolved_tickers=(),
            price_rows=5,
            first_date="2024-01-01",
            last_date="2024-01-05",
            coverage_by_ticker={
                "AAPL": {"first": "2024-01-01", "last": "2024-01-05", "days": 5, "gaps": 0},
            },
        )
        new = PriceSnapshot(
            snapshot_id="new",
            created_at="2024-02-01",
            git_sha="bbb",
            yfinance_version="1.1",
            python_version="3.11.0",
            requested_tickers=1,
            resolved_tickers=1,
            unresolved_tickers=(),
            price_rows=10,
            first_date="2024-01-01",
            last_date="2024-01-10",
            coverage_by_ticker={
                "AAPL": {"first": "2024-01-01", "last": "2024-01-10", "days": 10, "gaps": 0},
            },
        )
        result = compare_snapshots(old, new)
        self.assertEqual(result["added_tickers"], [])
        self.assertEqual(result["removed_tickers"], [])
        self.assertEqual(len(result["changed_coverage"]), 1)
        self.assertEqual(result["changed_coverage"][0]["ticker"], "AAPL")

    def test_compare_snapshots_identical(self):
        snap = PriceSnapshot(
            snapshot_id="same",
            created_at="2024-01-01",
            git_sha="aaa",
            yfinance_version="1.0",
            python_version="3.11.0",
            requested_tickers=1,
            resolved_tickers=1,
            unresolved_tickers=(),
            price_rows=5,
            first_date="2024-01-01",
            last_date="2024-01-05",
            coverage_by_ticker={
                "AAPL": {"first": "2024-01-01", "last": "2024-01-05", "days": 5, "gaps": 0},
            },
        )
        result = compare_snapshots(snap, snap)
        self.assertEqual(result["added_tickers"], [])
        self.assertEqual(result["removed_tickers"], [])
        self.assertEqual(result["changed_coverage"], [])
        self.assertEqual(result["requested_tickers_diff"], 0)

    def test_snapshot_captures_git_sha(self):
        self._seed_prices()
        snap = create_snapshot(self.db, ["AAPL"], date(2024, 1, 1), date(2024, 1, 10))
        # git SHA should be a 40-char hex string or "unknown"
        self.assertTrue(
            len(snap.git_sha) == 40 or snap.git_sha == "unknown",
            f"Unexpected git SHA: {snap.git_sha}",
        )

    def test_snapshot_captures_versions(self):
        import sys
        self._seed_prices()
        snap = create_snapshot(self.db, ["AAPL"], date(2024, 1, 1), date(2024, 1, 10))
        expected_py = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        self.assertEqual(snap.python_version, expected_py)
        self.assertTrue(snap.yfinance_version)

    def test_coverage_detects_gaps(self):
        dates = pd.bdate_range("2024-01-01", "2024-01-05")
        # Only prices on Jan 1 and Jan 5, missing Jan 2-4
        prices = pd.DataFrame(
            {"AAPL": [100.0, None, None, None, 105.0]},
            index=dates,
        )
        self.db.upsert_prices(prices)
        # Upsert only inserts non-null, so only Jan 1 and Jan 5 exist
        snap = create_snapshot(self.db, ["AAPL"], date(2024, 1, 1), date(2024, 1, 5))
        self.assertIn("AAPL", snap.coverage_by_ticker)
        cov = snap.coverage_by_ticker["AAPL"]
        self.assertEqual(cov["first"], "2024-01-01")
        self.assertEqual(cov["last"], "2024-01-05")
        self.assertEqual(cov["days"], 2)
        self.assertGreater(cov["gaps"], 0)


if __name__ == "__main__":
    unittest.main()
