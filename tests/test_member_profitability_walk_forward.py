"""Smoke tests for member_profitability.walk_forward module."""

import unittest

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


if __name__ == "__main__":
    unittest.main()
