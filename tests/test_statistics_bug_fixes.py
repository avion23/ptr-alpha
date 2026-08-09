"""Unit tests for statistics bugs fixed in bugs #1–6.

Bug summary:
  #1 SPY double-division in assembly.py (high)
  #2 NaN-as-loss in dynamic prior in filters.py (medium)
  #3 NaN-as-miss in hit rates in ranking.py and sales.py (medium)
  #4 Kelly NaN fallback not triggered when avg_loss=NaN (high)
  #5 Sweep objective alpha_slope sign-inversion in sweep.py (critical)
  #6 Missing price windows default to 0.0 instead of NaN (medium)
"""

from __future__ import annotations

import math
import unittest

import pandas as pd


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_signal_df_with_nan(**override) -> pd.DataFrame:
    """Minimal purchase signal DataFrame; override any column with NaN."""
    base = {
        "member": ["Alice", "Alice"],
        "ticker": ["AAPL", "AAPL"],
        "signal_type": ["Purchase", "Purchase"],
        "horizon_days": [90, 90],
        "decayed_return_pct": [5.0, float("nan")],
        "peak_potential_pct": [8.0, float("nan")],
        "spy_alpha_pct": [4.0, float("nan")],
        "total_return_pct": [6.0, float("nan")],
        "total_spy_alpha_pct": [3.0, float("nan")],
        "entry_price": [100.0, 100.0],
        "disclosure_date": pd.to_datetime(["2024-01-01", "2024-02-01"]),
    }
    base.update(override)
    return pd.DataFrame(base)


# ── Bug #1: SPY double-division ───────────────────────────────────────────────


class TestSpyDoubleDivision(unittest.TestCase):
    """
    _populate_spy_arrays stores the decay-weighted mean (s_wr.sum() / s_ws) in
    r_spy_cum.  Prior to the fix, assembly._compute_derived_arrays divided by
    r_spy_wsum a second time, halving (or more) the reported SPY return.

    Numeric probe: SPY 100→110 over 1 trading day, decay_lambda=0.
      Window: prices=[100, 110], n=2
      spy_lr = [0, log(110/100)] ≈ [0, 0.09531]
      weights (decay_lambda=0) = [1, 1] → s_ws = 2
      weighted mean  = 0.09531 / 2 ≈ 0.04765
      decayed_spy_return_pct = 0.04765 * 100 ≈ 4.77%   ← correct
      double-division gave     0.04765 / 2 * 100 ≈ 2.38% ← buggy
    """

    def setUp(self):
        # Two prices: disclosure day 0 and day 1
        self.entry_prices = pd.DataFrame(
            {
                "member": ["Alice"],
                "ticker": ["TICK"],
                "disclosure_date": pd.to_datetime(["2024-01-01"]),
                "transaction_type": ["Purchase"],
                "entry_price": [100.0],
            }
        )
        self.prices = pd.DataFrame(
            {
                "TICK": [100.0, 100.0],  # flat ticker (decayed_return ≈ 0)
                "SPY": [100.0, 110.0],  # SPY rises 10%
            },
            index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
        )

    def test_spy_return_not_double_divided(self):
        from analyzer.signals.core import calculate_signal_potential

        signals = calculate_signal_potential(
            self.entry_prices, self.prices, [1], decay_lambda=0.0
        )
        self.assertEqual(len(signals), 1)
        dsr = signals.iloc[0]["decayed_spy_return_pct"]
        # Correct value ≈ 4.77%; buggy value was ≈ 2.38%.
        # log(110/100) ≈ 0.09531; two weights (1,1) → mean = 0.04765
        expected = math.log(110.0 / 100.0) / 2.0 * 100  # ≈ 4.77
        self.assertAlmostEqual(
            dsr,
            expected,
            places=3,
            msg=f"decayed_spy_return_pct={dsr:.4f} expected≈{expected:.4f}; "
            "double-division bug may not be fixed",
        )


# ── Bug #3: NaN-as-miss in hit rates ────────────────────────────────────────


class TestHitRateNaN(unittest.TestCase):
    """
    ranking._hit_rates_by_member used (total_return_pct > 0).mean() on the
    raw DataFrame including NaN rows.  NaN > 0 is False, so NaN observations
    were counted as misses in both numerator and denominator.

    Example: Alice with returns [5.0, NaN] → old: 1/2 = 50%, fix: 1/1 = 100%.
    """

    def test_peak_hit_rate_excludes_nan(self):
        from analyzer.member_ranking.ranking import _hit_rates_by_member

        purchases = pd.DataFrame(
            {
                "member": ["Alice", "Alice"],
                "peak_potential_pct": [10.0, float("nan")],
                "total_return_pct": [6.0, float("nan")],
                "entry_price": [100.0, 100.0],
                "disclosure_date": pd.to_datetime(["2024-01-01", "2024-02-01"]),
            }
        )
        idx = pd.Index(["Alice"])
        peak, _ = _hit_rates_by_member(purchases, idx, threshold=5.0)
        self.assertAlmostEqual(
            peak.loc["Alice"], 100.0, places=5, msg="peak_hit_rate counted NaN as miss"
        )


# ── Kelly outcome coverage ───────────────────────────────────────────────────


class TestKellyOutcomeCoverage(unittest.TestCase):
    """Kelly must abstain when historical loss magnitude is unavailable."""

    def test_all_positive_returns_without_loss_estimate_abstain(self):
        from analyzer.portfolio.kelly import build_kelly_portfolio, KellyConfig

        recs = pd.DataFrame(
            {
                "ticker": ["A", "B", "C"],
                "signal_score": [10.0, 8.0, 6.0],
                "member": ["m1", "m2", "m3"],
                "crash_prob": [0.0, 0.0, 0.0],
                "avg_return_pct": [5.0, 3.0, 1.0],
            }
        )

        portfolio = build_kelly_portfolio(recs, KellyConfig())

        self.assertTrue(portfolio.empty)


# ── Bug #6: Missing price windows should yield NaN (not 0.0) ─────────────────


class TestMissingWindowsNaN(unittest.TestCase):
    """
    When a ticker or SPY price window is absent, result arrays remain at their
    initial values.  Previously total_return and actual_spy_return defaulted to
    0.0 in the assembly step, making missing data indistinguishable from a flat
    trade.  After the fix they are NaN so downstream aggregations (dynamic
    prior, hit rates) can exclude them.

    A "missing ticker window" occurs when _price_arrays returns (None, None)
    because the column has all-NaN values — the kernel skips the ticker and
    r_disc_baseline stays NaN, so valid_disc = False → total_return = NaN.

    A "missing SPY" occurs when prices_df has no SPY column — the spy_has flag
    is False, r_spy_wsum stays 0, and spy_cum = NaN via np.where.
    """

    def test_total_return_pct_is_nan_for_missing_ticker_prices(self):
        """All-NaN ticker prices → r_disc_baseline stays NaN → total_return NaN."""
        from analyzer.signals.core import calculate_signal_potential

        entry_prices = pd.DataFrame(
            {
                "member": ["Alice"],
                "ticker": ["TICK"],
                "disclosure_date": pd.to_datetime(["2024-01-01"]),
                "transaction_type": ["Purchase"],
                "entry_price": [100.0],
            }
        )
        # TICK column is all-NaN → _price_arrays returns (None, None) → kernel skips
        prices = pd.DataFrame(
            {
                "TICK": [float("nan"), float("nan")],
                "SPY": [400.0, 401.0],
            },
            index=pd.to_datetime(["2024-01-01", "2024-02-01"]),
        )
        signals = calculate_signal_potential(entry_prices, prices, [30])
        self.assertEqual(len(signals), 1)
        row = signals.iloc[0]
        self.assertTrue(
            pd.isna(row["total_return_pct"]),
            f"total_return_pct={row['total_return_pct']}; should be NaN when ticker has no prices",
        )

    def test_decayed_spy_return_is_nan_when_no_spy_column(self):
        """prices_df without SPY column → spy_has=False → decayed_spy_return NaN."""
        from analyzer.signals.core import calculate_signal_potential

        entry_prices = pd.DataFrame(
            {
                "member": ["Alice"],
                "ticker": ["TICK"],
                "disclosure_date": pd.to_datetime(["2024-01-01"]),
                "transaction_type": ["Purchase"],
                "entry_price": [100.0],
            }
        )
        # No SPY column → spy arrays are None → r_spy_wsum stays 0 → spy_cum=NaN
        prices_no_spy = pd.DataFrame(
            {
                "TICK": [100.0, 101.0, 102.0],
            },
            index=pd.to_datetime(["2024-01-01", "2024-01-15", "2024-02-01"]),
        )
        signals = calculate_signal_potential(entry_prices, prices_no_spy, [30])
        self.assertEqual(len(signals), 1)
        row = signals.iloc[0]
        self.assertTrue(
            pd.isna(row["decayed_spy_return_pct"]),
            f"decayed_spy_return_pct={row['decayed_spy_return_pct']}; should be NaN without SPY",
        )

    def test_spy_alpha_pct_is_nan_without_spy_data(self):
        """When prices_df has no SPY column, spy_alpha_pct must be NaN."""
        from analyzer.signals.core import calculate_signal_potential

        entry_prices = pd.DataFrame(
            {
                "member": ["Alice"],
                "ticker": ["TICK"],
                "disclosure_date": pd.to_datetime(["2024-01-01"]),
                "transaction_type": ["Purchase"],
                "entry_price": [100.0],
            }
        )
        prices_no_spy = pd.DataFrame(
            {
                "TICK": [100.0, 101.0, 102.0],
            },
            index=pd.to_datetime(["2024-01-01", "2024-01-15", "2024-02-01"]),
        )
        signals = calculate_signal_potential(entry_prices, prices_no_spy, [30])
        row = signals.iloc[0]
        self.assertTrue(
            pd.isna(row["spy_alpha_pct"]),
            "spy_alpha_pct should be NaN when no SPY data is available",
        )


if __name__ == "__main__":
    unittest.main()
