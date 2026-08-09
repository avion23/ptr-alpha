"""Smoke tests for member_profitability.walk_forward module."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


class TestGenerateWindows(unittest.TestCase):
    def test_short_signals_returns_empty(self):
        from member_profitability.walk_forward import generate_windows

        # Single disclosure date doesn't allow a full 6mo train + 6mo test window
        sigs = pd.DataFrame(
            {
                "disclosure_date": pd.to_datetime(["2024-01-01"]),
            }
        )
        self.assertEqual(generate_windows(sigs), [])

    def test_long_signals_generates_windows(self):
        from member_profitability.walk_forward import generate_windows

        # 18 months of disclosures -> at least one window possible
        sigs = pd.DataFrame(
            {
                "disclosure_date": pd.date_range(
                    "2023-01-01", "2024-06-01", freq="30D"
                ),
            }
        )
        windows = generate_windows(sigs)
        self.assertGreater(len(windows), 0)
        first = windows[0]
        for key in ("train_start", "train_end", "test_start", "test_end"):
            self.assertIn(key, first)


def _signal_rows(rows):
    return pd.DataFrame(
        {
            "member": [row[0] for row in rows],
            "ticker": [row[1] for row in rows],
            "disclosure_date": pd.to_datetime([row[2] for row in rows]),
            "signal_type": ["Purchase"] * len(rows),
            "horizon_days": [60] * len(rows),
            "window_complete": [True] * len(rows),
            "entry_price": [100.0] * len(rows),
            "total_spy_alpha_pct": [row[3] for row in rows],
            # Deliberately unrelated: honest member statistics must not use it.
            "decayed_return_pct": [-row[3] for row in rows],
            "spy_alpha_pct": [-row[3] for row in rows],
            "total_return_pct": [row[3] for row in rows],
            "peak_potential_pct": [max(row[3], 0.0) for row in rows],
            "amount_midpoint": [1_000.0] * len(rows),
        }
    )


class TestPointInTimeCanaries(unittest.TestCase):
    def test_slice_requires_labels_mature_at_each_boundary(self):
        from member_profitability.walk_forward import _slice_window

        window = {
            "train_start": pd.Timestamp("2024-01-01"),
            "train_end": pd.Timestamp("2024-07-01"),
            "test_start": pd.Timestamp("2024-07-01"),
            "test_end": pd.Timestamp("2025-01-01"),
        }
        sigs = _signal_rows(
            [
                ("A", "A", "2024-05-01", 1.0),
                ("B", "B", "2024-06-01", 1.0),
                ("C", "C", "2024-10-01", 1.0),
                ("D", "D", "2024-12-01", 1.0),
            ]
        )

        train, test = _slice_window(sigs, window)

        self.assertEqual(train["member"].tolist(), ["A"])
        self.assertEqual(test["member"].tolist(), ["C"])

    def test_generated_test_periods_do_not_overlap(self):
        from member_profitability.walk_forward import generate_windows

        sigs = pd.DataFrame(
            {"disclosure_date": pd.date_range("2021-01-01", "2025-01-01", freq="30D")}
        )
        windows = generate_windows(sigs)

        self.assertGreater(len(windows), 2)
        for previous, current in zip(windows, windows[1:]):
            self.assertEqual(previous["test_end"], current["test_start"])

    def test_member_statistics_use_endpoint_excess_return_and_common_prior(self):
        from member_profitability.walk_forward import _rank_train

        sigs = _signal_rows(
            [
                ("Strong", "S1", "2024-01-01", 10.0),
                ("Strong", "S2", "2024-02-01", 8.0),
                ("Strong", "S3", "2024-03-01", 6.0),
                ("Weak", "W1", "2024-01-01", -4.0),
                ("Weak", "W2", "2024-02-01", -6.0),
            ]
        )
        ranked = _rank_train(sigs).set_index("member")

        self.assertGreater(
            ranked.loc["Strong", "bayes_positive_excess_prob"],
            ranked.loc["Weak", "bayes_positive_excess_prob"],
        )
        self.assertGreater(
            ranked.loc["Strong", "shrunk_excess_return_pct"],
            ranked.loc["Weak", "shrunk_excess_return_pct"],
        )
        self.assertGreater(ranked.loc["Strong", "avg_excess_return_pct"], 0)
        self.assertLess(ranked.loc["Weak", "avg_excess_return_pct"], 0)

    def test_timestamped_candidate_does_not_count_future_buyer(self):
        from member_profitability.position_sizing import _timestamped_recommendations

        test_sigs = _signal_rows(
            [
                ("A", "XYZ", "2024-07-01", 1.0),
                ("B", "XYZ", "2024-07-02", 2.0),
                ("C", "XYZ", "2024-07-03", 100.0),
            ]
        )
        rankings = pd.DataFrame(
            {
                "member": ["A", "B", "C"],
                "shrunk_excess_return_pct": [2.0, 2.0, 100.0],
            }
        )

        recs = _timestamped_recommendations(test_sigs, rankings, top_n=1, min_buyers=2)

        self.assertEqual(len(recs), 2)
        self.assertEqual(recs.iloc[0]["decision_date"], pd.Timestamp("2024-07-02"))
        self.assertEqual(recs.iloc[0]["rated_buyers"], 2)
        self.assertEqual(recs.iloc[0]["realized_excess_return_pct"], 2.0)
        self.assertEqual(recs.iloc[1]["decision_date"], pd.Timestamp("2024-07-03"))
        self.assertEqual(recs.iloc[1]["rated_buyers"], 3)

    def test_missing_outcome_changes_coverage_not_candidate_universe(self):
        from member_profitability.position_sizing import (
            _summarize_recommendations,
            _timestamped_recommendations,
        )

        disclosed = _signal_rows(
            [
                ("BUYER", "A", "2024-07-01", 4.0),
                ("BUYER", "B", "2024-07-01", 4.0),
            ]
        )
        rankings = pd.DataFrame(
            {"member": ["BUYER"], "shrunk_excess_return_pct": [2.0]}
        )
        with_outcome = _timestamped_recommendations(
            disclosed, rankings, top_n=1, min_buyers=1
        )
        missing = disclosed.copy()
        missing.loc[missing["ticker"] == "A", "total_spy_alpha_pct"] = float("nan")
        without_outcome = _timestamped_recommendations(
            missing, rankings, top_n=1, min_buyers=1
        )

        selection_columns = ["decision_date", "ticker", "rated_buyers", "score"]
        pd.testing.assert_frame_equal(
            with_outcome[selection_columns].reset_index(drop=True),
            without_outcome[selection_columns].reset_index(drop=True),
        )
        self.assertEqual(with_outcome.iloc[0]["ticker"], "A")
        with_summary = _summarize_recommendations(with_outcome)
        without_summary = _summarize_recommendations(without_outcome)
        self.assertEqual(with_summary["n_eligible_recommendations"], 1)
        self.assertEqual(without_summary["n_eligible_recommendations"], 1)
        self.assertEqual(with_summary["n_evaluable_recommendations"], 1)
        self.assertEqual(without_summary["n_evaluable_recommendations"], 0)
        self.assertEqual(without_summary["n_missing_outcome_recommendations"], 1)


class TestChronologicalRetrospectiveValidation(unittest.TestCase):
    def test_global_spacing_rejects_whole_boundary_dates_and_preserves_top_n(self):
        from member_profitability.position_sizing import _apply_global_decision_spacing

        rows = []
        for decision_date, tickers in (
            ("2024-06-15", ("A", "B")),
            ("2024-08-03", ("C", "D")),  # 49 days: reject whole date
            ("2024-08-14", ("E", "F")),  # 60 days: accept both rows
            ("2024-09-14", ("G", "H")),  # 31 days: reject whole date
            ("2024-10-13", ("I", "J")),  # 60 days: accept both rows
        ):
            for ticker in tickers:
                rows.append(
                    {
                        "decision_date": pd.Timestamp(decision_date),
                        "ticker": ticker,
                        "score": 1.0,
                    }
                )
        spaced = _apply_global_decision_spacing(pd.DataFrame(rows))

        dates = sorted(spaced["decision_date"].unique())
        self.assertEqual(
            dates,
            [
                pd.Timestamp("2024-06-15"),
                pd.Timestamp("2024-08-14"),
                pd.Timestamp("2024-10-13"),
            ],
        )
        self.assertEqual(spaced.groupby("decision_date").size().tolist(), [2, 2, 2])
        gaps = pd.Series(dates).diff().dropna().dt.days
        self.assertGreaterEqual(int(gaps.min()), 60)

    def test_end_to_end_windows_space_complete_candidate_stream_once(self):
        from member_profitability.position_sizing import (
            _recommendations_for_windows,
            _summarize_recommendations,
        )

        windows = [
            {
                "train_start": pd.Timestamp("2023-08-06"),
                "train_end": pd.Timestamp("2024-02-02"),
                "test_start": pd.Timestamp("2024-02-02"),
                "test_end": pd.Timestamp("2024-08-01"),
            },
            {
                "train_start": pd.Timestamp("2024-02-02"),
                "train_end": pd.Timestamp("2024-08-01"),
                "test_start": pd.Timestamp("2024-08-01"),
                "test_end": pd.Timestamp("2025-01-28"),
            },
        ]
        signals = _signal_rows(
            [
                ("BUYER", "A", "2024-06-15", 1.0),
                ("BUYER", "B", "2024-08-03", 2.0),
                ("BUYER", "C", "2024-08-14", 3.0),
            ]
        )
        rankings = pd.DataFrame(
            {"member": ["BUYER"], "shrunk_excess_return_pct": [2.0]}
        )
        with patch(
            "member_profitability.position_sizing._rank_train",
            return_value=rankings,
        ):
            recommendations = _recommendations_for_windows(
                signals, windows, top_n=1, min_buyers=1
            )

        self.assertEqual(
            recommendations["decision_date"].tolist(),
            [pd.Timestamp("2024-06-15"), pd.Timestamp("2024-08-14")],
        )
        self.assertEqual(recommendations["ticker"].tolist(), ["A", "C"])
        summary = _summarize_recommendations(recommendations)
        self.assertEqual(summary["n_eligible_recommendations"], 2)
        self.assertEqual(summary["n_evaluable_recommendations"], 1)
        self.assertEqual(summary["n_missing_outcome_recommendations"], 1)

    def test_retrospective_validation_outcomes_cannot_change_selected_parameters(self):
        from member_profitability.position_sizing import position_sizing_grid_search

        windows = [
            {
                "train_start": pd.Timestamp("2023-01-01"),
                "train_end": pd.Timestamp("2023-07-01"),
                "test_start": pd.Timestamp("2023-07-01"),
                "test_end": pd.Timestamp("2024-01-01"),
            },
            {
                "train_start": pd.Timestamp("2024-01-01"),
                "train_end": pd.Timestamp("2024-07-01"),
                "test_start": pd.Timestamp("2024-07-01"),
                "test_end": pd.Timestamp("2025-01-01"),
            },
        ]

        def run_with_retrospective_validation(retrospective_validation_return):
            def fake_recommendations(sigs, selected_windows, top_n, min_buyers):
                is_retrospective_validation = selected_windows[0]["test_start"] == pd.Timestamp("2024-07-01")
                realized = retrospective_validation_return if is_retrospective_validation else (
                    10.0 if (top_n, min_buyers) == (2, 3) else -1.0
                )
                return pd.DataFrame(
                    {
                        "decision_date": [pd.Timestamp("2024-01-01")],
                        "realized_excess_return_pct": [realized],
                    }
                )

            with patch(
                "member_profitability.position_sizing._recommendations_for_windows",
                side_effect=fake_recommendations,
            ):
                return position_sizing_grid_search(pd.DataFrame(), windows)

        positive = run_with_retrospective_validation(100.0)
        negative = run_with_retrospective_validation(-100.0)

        self.assertEqual(positive["selected_candidate"]["top_n"], 2)
        self.assertEqual(positive["selected_candidate"]["min_buyers"], 3)
        self.assertEqual(
            positive["selected_candidate"], negative["selected_candidate"]
        )
        self.assertNotEqual(positive["retrospective_validation"], negative["retrospective_validation"])


class TestReadOnlyLoading(unittest.TestCase):
    def test_transaction_query_stops_before_price_buffer_and_closes(self):
        from member_profitability.data import load_transactions_and_prices

        calls = {}

        class FakeDatabase:
            instance = None

            def __init__(self, path, read_only=False):
                self.closed = False
                self.read_only = read_only
                FakeDatabase.instance = self

            def get_transactions_by_date_range(self, start, end):
                calls["transactions"] = (start, end)
                return pd.DataFrame(
                    {
                        "member": ["Rep. Example"],
                        "ticker": ["ABC"],
                        "disclosure_date": [end],
                    }
                )

            def get_prices(self, tickers, start, end):
                calls["prices"] = (start, end)
                return pd.DataFrame({"ABC": [100.0]}, index=[start])

            def get_entry_prices(self, tickers, start, end):
                calls["entries"] = (start, end)
                return pd.DataFrame(
                    {
                        "member": ["Rep. Example"],
                        "ticker": ["ABC"],
                        "disclosure_date": [end],
                    }
                )

            def close(self):
                self.closed = True

        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "input.duckdb"
            db_path.touch()
            with patch("member_profitability.data.Database", FakeDatabase):
                load_transactions_and_prices(db_path, "2024-01-01", "2024-06-30")

        self.assertTrue(FakeDatabase.instance.read_only)
        self.assertTrue(FakeDatabase.instance.closed)
        self.assertEqual(calls["transactions"][1], pd.Timestamp("2024-06-30"))
        self.assertEqual(calls["entries"][1], pd.Timestamp("2024-06-30"))
        self.assertGreater(calls["prices"][1], calls["entries"][1])

    @unittest.skipUnless(os.environ.get("PTR_ALPHA_REAL_DB"), "real DB scenario not requested")
    def test_real_database_scenario_is_read_only_bounded_and_globally_spaced(self):
        from member_profitability.data import compute_signals, load_transactions_and_prices
        from member_profitability.position_sizing import position_sizing_grid_search
        from member_profitability.walk_forward import generate_windows

        db_path = Path(os.environ["PTR_ALPHA_REAL_DB"])
        before = db_path.stat().st_mtime_ns
        _, prices, entries, _ = load_transactions_and_prices(
            db_path, "2021-10-07", "2025-06-30"
        )
        signals = compute_signals(entries, prices)
        result = position_sizing_grid_search(signals, generate_windows(signals))
        after = db_path.stat().st_mtime_ns

        self.assertEqual(before, after)
        self.assertLessEqual(
            pd.to_datetime(entries["disclosure_date"]).max(),
            pd.Timestamp("2025-06-30"),
        )
        for key in ("selection_recommendations", "retrospective_validation_recommendations"):
            dates = pd.Series(
                sorted(
                    pd.to_datetime(row["decision_date"])
                    for row in result[key]
                )
            ).drop_duplicates()
            if len(dates) > 1:
                gaps = dates.diff().dropna().dt.days
                self.assertGreaterEqual(int(gaps.min()), 60, key)


if __name__ == "__main__":
    unittest.main()
