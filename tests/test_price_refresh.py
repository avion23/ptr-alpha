"""Focused tests for the price refresh + value-hashed snapshot pipeline.

Covers the enforced contracts: exact NYSE sessions, nonpositive quarantine,
ticker/asset eligibility, stale=unavailable, latest completed market session,
value-hashed manifest (data/code/config), and read-only operation against
temp databases (the canonical DB is never touched).
"""

import hashlib
import shutil
import tempfile
import unittest
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from analyzer.database import Database
from analyzer.price_snapshot import load_snapshot
from analyzer.price_source import YFinancePriceSource

from scripts import refresh_prices, snapshot_prices


def _nyse_dates(start: str, end: str):
    from analyzer.price_repository import nyse_sessions

    return nyse_sessions(date.fromisoformat(start), date.fromisoformat(end))


def _fake_download(frame: pd.DataFrame):
    """Return a _download_yfinance stand-in yielding the canned Close frame."""

    def fake(self, fetch_resolved, start, end):
        return frame

    return fake


class EligibilityTests(unittest.TestCase):
    def test_excludes_invalid_syntax_and_reserved_tokens(self):
        eligible, excluded = refresh_prices.select_eligible_assets(
            ["AAPL", "123", "ABC.DEF.GH", "BOND", "STOCK", "TICKER", "COUPON", "NOTES"]
        )
        self.assertNotIn("123", eligible)
        self.assertNotIn("ABC.DEF.GH", eligible)
        for reserved in ("BOND", "STOCK", "TICKER", "COUPON", "NOTES"):
            self.assertNotIn(reserved, eligible)
        self.assertIn("AAPL", eligible)
        self.assertIn("invalid_syntax", excluded)
        self.assertIn("reserved_non_equity", excluded)

    def test_excludes_quarantined_and_suspicious_tokens(self):
        eligible, excluded = refresh_prices.select_eligible_assets(
            ["SP", "ALLI", "MATT", "THE", "NEW", "MARY", "CITI", "MSFT"]
        )
        for bad in ("SP", "ALLI", "MATT", "THE", "NEW", "MARY", "CITI"):
            self.assertNotIn(bad, eligible)
        self.assertIn("MSFT", eligible)
        self.assertIn("quarantined_or_suspicious", excluded)

    def test_keeps_class_shares_and_normalizes_case(self):
        eligible, _ = refresh_prices.select_eligible_assets(
            ["brk.b", "BRK", "BRK.A", "googl", "aapl"]
        )
        self.assertIn("BRK.B", eligible)
        self.assertIn("BRK.A", eligible)
        self.assertIn("GOOGL", eligible)
        self.assertIn("AAPL", eligible)

    def test_deduplicates_and_adds_benchmark(self):
        eligible, _ = refresh_prices.select_eligible_assets(["AAPL", "aapl", "AAPL"])
        self.assertEqual(eligible, ["AAPL", "SPY"])

    def test_empty_input_still_requests_benchmark(self):
        eligible, _ = refresh_prices.select_eligible_assets([])
        self.assertEqual(eligible, ["SPY"])


class EndDateTests(unittest.TestCase):
    def test_weekend_rolls_back_to_friday(self):
        self.assertEqual(
            refresh_prices.refresh_end_date(date(2026, 8, 9)),
            date(2026, 8, 7),
        )
        self.assertEqual(
            refresh_prices.refresh_end_date(date(2026, 8, 8)),
            date(2026, 8, 7),
        )

    def test_holiday_rolls_back_to_previous_session(self):
        # 2026-01-01 is New Year's Day (holiday); 2025-12-31 is a session.
        self.assertEqual(
            refresh_prices.refresh_end_date(date(2026, 1, 1)),
            date(2025, 12, 31),
        )

    def test_session_day_is_kept(self):
        self.assertEqual(
            refresh_prices.refresh_end_date(date(2026, 8, 7)),
            date(2026, 8, 7),
        )


class RefreshPipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.source_db = self.tmp / "source.duckdb"
        self.temp_db = self.tmp / "refresh.duckdb"
        self.db = Database(self.source_db)

    def tearDown(self):
        self.db.close()
        shutil.rmtree(self.tmp)

    def _seed_transactions(self, tickers):
        for t in tickers:
            self.db.conn.execute("INSERT INTO transactions (ticker) VALUES (?)", [t])

    def _run(self, frame, start="2024-01-02", end="2024-01-05", **kwargs):
        self.db.close()
        original = YFinancePriceSource._download_yfinance
        YFinancePriceSource._download_yfinance = _fake_download(frame)
        try:
            return refresh_prices.refresh_prices(
                self.source_db,
                self.temp_db,
                start=date.fromisoformat(start),
                end=date.fromisoformat(end),
                **kwargs,
            )
        finally:
            YFinancePriceSource._download_yfinance = original

    def _frame(self, tickers, start="2024-01-02", end="2024-01-05"):
        dates = _nyse_dates(start, end)
        tickers = list(dict.fromkeys([*tickers, "SPY"]))
        return pd.DataFrame(
            {("Close", t): [100.0 + i for i in range(len(dates))] for t in tickers},
            index=dates,
        )

    def test_refresh_filters_eligibility_and_quarantines_nonpositive(self):
        self._seed_transactions(
            ["AAPL", "MSFT", "MARY", "BOND", "SP", "123", "zzz"]
        )
        dates = _nyse_dates("2024-01-02", "2024-01-05")
        frame = pd.DataFrame(
            {
                ("Close", "AAPL"): [100.0, 101.0, 102.0, 103.0],
                ("Close", "MSFT"): [200.0, 0.0, np.nan, 205.0],
                ("Close", "SPY"): [400.0, 401.0, 402.0, 403.0],
                ("Close", "ZZZ"): [10.0, 11.0, 12.0, 13.0],
            },
            index=dates,
        )
        report = self._run(frame)
        self.assertIn("AAPL", report.eligible_assets)
        self.assertIn("MSFT", report.eligible_assets)
        self.assertIn("SPY", report.eligible_assets)
        # ZZZ is syntactically valid, so it is requested (unresolved at fetch time).
        self.assertIn("ZZZ", report.eligible_assets)
        self.assertNotIn("ZZZ", report.unresolved_tickers)
        for bad in ("MARY", "BOND", "SP", "123"):
            self.assertNotIn(bad, report.eligible_assets)
        self.assertIn("invalid_syntax", report.excluded_assets)
        self.assertIn("reserved_non_equity", report.excluded_assets)
        self.assertIn("quarantined_or_suspicious", report.excluded_assets)
        self.assertEqual(report.rejected_observations, 0)

        # Nonpositive and NaN observations never persisted.
        check = Database(self.temp_db, read_only=True)
        try:
            rows = check.conn.execute(
                "SELECT COUNT(*) FROM prices WHERE close <= 0 OR NOT isfinite(close)"
            ).fetchone()[0]
            msft = check.conn.execute(
                "SELECT COUNT(*) FROM prices WHERE ticker = 'MSFT'"
            ).fetchone()[0]
        finally:
            check.close()
        self.assertEqual(rows, 0)
        self.assertEqual(msft, 2)  # 0.0 and NaN quarantined; 2 valid rows remain

    def test_refresh_never_touches_canonical_db(self):
        self._seed_transactions(["AAPL"])
        canonical = self.tmp / "data" / "congress.duckdb"
        report = self._run(self._frame(["AAPL"]))
        self.assertTrue(self.temp_db.exists())
        self.assertFalse(canonical.exists())
        self.assertNotEqual(str(report.temp_db), str(canonical))

    def test_refresh_rejects_non_session_end(self):
        self._seed_transactions(["AAPL"])
        with self.assertRaises(ValueError):
            self._run(
                self._frame(["AAPL"]), start="2024-01-02", end="2024-01-06"
            )  # Saturday

    def test_refresh_rejects_end_before_start(self):
        self._seed_transactions(["AAPL"])
        with self.assertRaises(ValueError):
            self._run(self._frame(["AAPL"]), start="2024-01-05", end="2024-01-02")

    def test_refresh_refuses_existing_temp_db(self):
        self._seed_transactions(["AAPL"])
        self.temp_db.write_bytes(b"")
        with self.assertRaises(FileExistsError):
            self._run(self._frame(["AAPL"]))
        self.temp_db.unlink()
        self._run(self._frame(["AAPL"]), force=True)

    def test_refresh_marks_stale_tickers_unavailable(self):
        self._seed_transactions(["AAPL", "MSFT"])
        end = "2024-02-02"
        dates = _nyse_dates("2024-01-02", end)
        frame = pd.DataFrame(
            {
                # MSFT stops trading 61 calendar days before end -> stale.
                ("Close", "AAPL"): [100.0 + i for i in range(len(dates))],
                ("Close", "MSFT"): [
                    200.0 if d < pd.Timestamp("2024-01-02") + pd.Timedelta(days=1) else np.nan
                    for d in dates
                ],
                ("Close", "SPY"): [400.0 + i for i in range(len(dates))],
            },
            index=dates,
        )
        report = self._run(frame, start="2024-01-02", end=end)
        self.assertIn("MSFT", report.stale_tickers)
        self.assertNotIn("AAPL", report.stale_tickers)


class SnapshotManifestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.db_path = self.tmp / "prices.duckdb"
        self.out = self.tmp / "staged"
        self.db = Database(self.db_path)

    def tearDown(self):
        self.db.close()
        shutil.rmtree(self.tmp)

    def _seed(self, values=None):
        self._ensure_open()
        dates = _nyse_dates("2024-01-02", "2024-01-05")
        prices = pd.DataFrame(
            {
                "AAPL": values if values is not None else [100.0, 101.0, 102.0, 103.0],
                "MSFT": [200.0, 201.0, 202.0, 203.0],
            },
            index=dates,
        )
        self.db.upsert_prices(prices)

    def _ensure_open(self):
        try:
            self.db.conn.execute("SELECT 1")
        except Exception:
            self.db = Database(self.db_path)

    def _close_db(self):
        try:
            self.db.conn.execute("SELECT 1")
            self.db.close()
        except Exception:
            pass

    def _manifest(self, **kwargs):
        self._close_db()
        defaults = dict(
            start=date(2024, 1, 2),
            end=date(2024, 1, 5),
            generation="gen-test-20240105",
            out_dir=self.out,
        )
        defaults.update(kwargs)
        return snapshot_prices.build_manifest(self.db_path, **defaults)

    def test_manifest_binds_data_code_and_config_hashes(self):
        self._seed()
        manifest = self._manifest()
        self.assertEqual(len(manifest["data_hash"]), 64)
        self.assertEqual(manifest["data_hash"], manifest["value_hash"])
        self.assertEqual(len(manifest["code_hash"]), 64)
        self.assertEqual(len(manifest["config_hash"]), 64)
        self.assertEqual(manifest["generation"], "gen-test-20240105")
        self.assertEqual(manifest["window"]["end"], "2024-01-05")
        self.assertEqual(manifest["tickers"]["resolved"], 2)
        self.assertEqual(manifest["price_rows"], 8)

    def test_manifest_artifacts_written(self):
        self._seed()
        manifest = self._manifest()
        self.assertTrue((self.out / "manifest.json").is_file())
        self.assertTrue((self.out / "snapshot.json").is_file())
        self.assertTrue((self.out / "prices.parquet").is_file())
        loaded = load_snapshot(str(self.out / "snapshot.json"))
        self.assertEqual(loaded.value_hash, manifest["data_hash"])
        parquet = pd.read_parquet(self.out / "prices.parquet")
        self.assertEqual(len(parquet), 8)
        self.assertEqual(set(parquet.columns), {"ticker", "date", "close"})

    def test_data_hash_changes_when_prices_change(self):
        first = self._manifest()
        self._seed(values=[100.0, 101.0, 999.0, 103.0])
        second = self._manifest()
        self.assertNotEqual(first["data_hash"], second["data_hash"])

    def test_data_hash_is_deterministic(self):
        first = self._manifest()
        second = self._manifest()
        self.assertEqual(first["data_hash"], second["data_hash"])

    def test_code_hash_tracks_pipeline_code(self, monkeypatch=None):
        import scripts.snapshot_prices as mod

        self._seed()
        self.db.close()
        code_dir = self.tmp / "code"
        code_dir.mkdir()
        f1 = code_dir / "price_source.py"
        f2 = code_dir / "price_repository.py"
        f1.write_text("V1")
        f2.write_text("V1")
        original_root = mod.REPO_ROOT
        mod.REPO_ROOT = code_dir
        try:
            manifest = snapshot_prices.build_manifest(
                self.db_path,
                start=date(2024, 1, 2),
                end=date(2024, 1, 5),
                generation="g",
                out_dir=self.out,
                code_files=["price_source.py", "price_repository.py"],
                config_files=["pyproject.toml"],
            )
            first_hash = manifest["code_hash"]
            f1.write_text("V2")
            manifest2 = snapshot_prices.build_manifest(
                self.db_path,
                start=date(2024, 1, 2),
                end=date(2024, 1, 5),
                generation="g",
                out_dir=self.out,
                code_files=["price_source.py", "price_repository.py"],
                config_files=["pyproject.toml"],
            )
            self.assertNotEqual(first_hash, manifest2["code_hash"])
            self.assertEqual(manifest["code_files"]["price_repository.py"], hashlib.sha256(b"V1").hexdigest())
        finally:
            mod.REPO_ROOT = original_root

    def test_snapshot_opens_db_read_only(self):
        self._seed()
        self._close_db()
        def _snapshot_rows():
            ro = Database(self.db_path, read_only=True)
            try:
                return ro.conn.execute(
                    "SELECT ticker, date, close FROM prices ORDER BY ticker, date"
                ).fetchdf()
            finally:
                ro.close()

        before = _snapshot_rows()
        manifest = self._manifest()
        after = _snapshot_rows()
        # The snapshot never changes price data.
        pd.testing.assert_frame_equal(before.reset_index(drop=True), after.reset_index(drop=True))
        # A read-only handle must reject writes outright.
        ro = Database(self.db_path, read_only=True)
        try:
            with self.assertRaises(Exception):
                ro.conn.execute("INSERT INTO prices VALUES ('X', DATE '2024-01-02', 1.0)")
        finally:
            ro.close()
        self.assertTrue(manifest)

    def test_snapshot_marks_stale_tickers(self):
        self._ensure_open()
        dates = _nyse_dates("2024-01-02", "2024-03-01")
        prices = pd.DataFrame(
            {
                "AAPL": [100.0 + i for i in range(len(dates))],
                "MSFT": [200.0] + [np.nan] * (len(dates) - 1),
            },
            index=dates,
        )
        self.db.upsert_prices(prices)
        manifest = self._manifest(start=date(2024, 1, 2), end=date(2024, 3, 1))
        self.assertIn("MSFT", manifest["tickers"]["stale"])
        self.assertNotIn("AAPL", manifest["tickers"]["stale"])

    def test_snapshot_rejects_non_session_end(self):
        self._seed()
        with self.assertRaises(ValueError):
            self._manifest(end=date(2024, 1, 6))  # Saturday

    def test_coverage_uses_exact_nyse_sessions(self):
        # 2024-01-15 (MLK Day) is not a session; it must not count as a gap.
        dates = _nyse_dates("2024-01-02", "2024-01-16")
        prices = pd.DataFrame(
            {"AAPL": [float(i + 1) for i in range(len(dates))]}, index=dates
        )
        self.db.upsert_prices(prices)
        self._manifest(start=date(2024, 1, 2), end=date(2024, 1, 16))
        snap = load_snapshot(str(self.out / "snapshot.json"))
        self.assertEqual(snap.coverage_by_ticker["AAPL"]["gaps"], 0)

    def test_cli_exit_codes(self):
        self._seed()
        self.db.close()
        rc = snapshot_prices.main(
            [
                "--db", str(self.db_path),
                "--start", "2024-01-02",
                "--end", "2024-01-05",
                "--generation", "gen-test-20240105",
                "--out", str(self.out),
            ]
        )
        self.assertEqual(rc, 0)
        rc_bad = snapshot_prices.main(
            [
                "--db", str(self.db_path),
                "--start", "2024-01-02",
                "--end", "2024-01-06",
                "--generation", "gen-test-20240105",
                "--out", str(self.out / "other"),
            ]
        )
        self.assertEqual(rc_bad, 1)


if __name__ == "__main__":
    unittest.main()
