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

import numpy as np
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
        self.entry_prices = pd.DataFrame({
            "member": ["Alice"],
            "ticker": ["TICK"],
            "disclosure_date": pd.to_datetime(["2024-01-01"]),
            "transaction_type": ["Purchase"],
            "entry_price": [100.0],
        })
        self.prices = pd.DataFrame({
            "TICK": [100.0, 100.0],   # flat ticker (decayed_return ≈ 0)
            "SPY":  [100.0, 110.0],   # SPY rises 10%
        }, index=pd.to_datetime(["2024-01-01", "2024-01-02"]))

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
        self.assertAlmostEqual(dsr, expected, places=3,
                               msg=f"decayed_spy_return_pct={dsr:.4f} expected≈{expected:.4f}; "
                                   "double-division bug may not be fixed")

    def test_spy_return_not_half_of_expected(self):
        """The buggy value was exactly half the correct value."""
        from analyzer.signals.core import calculate_signal_potential
        signals = calculate_signal_potential(
            self.entry_prices, self.prices, [1], decay_lambda=0.0
        )
        dsr = signals.iloc[0]["decayed_spy_return_pct"]
        # Ensure we are NOT returning half the expected value.
        expected = math.log(110.0 / 100.0) / 2.0 * 100
        self.assertFalse(
            abs(dsr - expected / 2.0) < 0.01,
            msg="decayed_spy_return_pct looks like the double-divided value"
        )


# ── Bug #2: NaN-as-loss in dynamic prior ────────────────────────────────────

class TestDynamicPriorNaN(unittest.TestCase):
    """
    _compute_dynamic_prior computed (decayed_return_pct > 0).mean().
    In pandas, NaN > 0 evaluates to False — so NaN observations were counted
    as losses, biasing the market-wide up-rate downward.

    Example: [5.0, NaN] → old: (True, False).mean() = 0.50
                          → fix: (True,).mean()      = 1.00 (clipped to 0.90)
    """

    def _make_signals(self, rets):
        return pd.DataFrame({
            "member": ["A"] * len(rets),
            "ticker": ["T"] * len(rets),
            "signal_type": ["Purchase"] * len(rets),
            "horizon_days": [90] * len(rets),
            "decayed_return_pct": rets,
            "peak_potential_pct": [8.0] * len(rets),
            "spy_alpha_pct": [1.0] * len(rets),
            "entry_price": [100.0] * len(rets),
            "disclosure_date": pd.to_datetime(["2024-01-01"] * len(rets)),
        })

    def test_nan_not_counted_as_loss(self):
        from analyzer.signals.filters import _compute_dynamic_prior
        # Only one valid positive return; NaN should be excluded entirely.
        signals = self._make_signals([5.0, float("nan")])
        prior = _compute_dynamic_prior(signals, 90)
        # All valid observations are positive → up_prob = 1.0, clipped to 0.90
        self.assertAlmostEqual(prior, 0.90, places=6,
                               msg=f"prior={prior}; NaN was counted as a loss")

    def test_all_nan_returns_default(self):
        from analyzer.signals.filters import _compute_dynamic_prior
        signals = self._make_signals([float("nan"), float("nan")])
        prior = _compute_dynamic_prior(signals, 90)
        self.assertAlmostEqual(prior, 0.50, places=6,
                               msg="all-NaN should return 0.50 default")

    def test_mixed_valid_nan(self):
        from analyzer.signals.filters import _compute_dynamic_prior
        # 2 wins, 1 loss, 1 NaN → up_prob = 2/3, not 2/4
        signals = self._make_signals([3.0, -1.0, 2.0, float("nan")])
        prior = _compute_dynamic_prior(signals, 90)
        self.assertAlmostEqual(prior, float(np.clip(2 / 3, 0.10, 0.90)), places=5)


# ── Bug #3: NaN-as-miss in hit rates ────────────────────────────────────────

class TestHitRateNaN(unittest.TestCase):
    """
    ranking._hit_rates_by_member used (total_return_pct > 0).mean() on the
    raw DataFrame including NaN rows.  NaN > 0 is False, so NaN observations
    were counted as misses in both numerator and denominator.

    Example: Alice with returns [5.0, NaN] → old: 1/2 = 50%, fix: 1/1 = 100%.
    """

    def _make_purchases(self, rets):
        n = len(rets)
        return pd.DataFrame({
            "member": ["Alice"] * n,
            "ticker": ["AAPL"] * n,
            "signal_type": ["Purchase"] * n,
            "horizon_days": [90] * n,
            "decayed_return_pct": rets,
            "peak_potential_pct": [8.0] * n,
            "spy_alpha_pct": [1.0] * n,
            "total_return_pct": [r + 1.0 if not math.isnan(r) else float("nan") for r in rets],
            "entry_price": [100.0] * n,
            "disclosure_date": pd.to_datetime(["2024-01-01"] * n),
        })

    def test_realized_hit_rate_excludes_nan(self):
        from analyzer.member_ranking.ranking import _hit_rates_by_member
        # [5.0, NaN] → only 1 valid observation, which is positive → 100 %
        purchases = self._make_purchases([5.0, float("nan")])
        idx = pd.Index(["Alice"])
        _, realized = _hit_rates_by_member(purchases, idx, threshold=3.0)
        self.assertIsNotNone(realized)
        rate = realized.loc["Alice"]
        self.assertAlmostEqual(rate, 100.0, places=5,
                               msg=f"realized_hit_rate={rate}; NaN counted as miss")

    def test_peak_hit_rate_excludes_nan(self):
        from analyzer.member_ranking.ranking import _hit_rates_by_member
        purchases = pd.DataFrame({
            "member": ["Alice", "Alice"],
            "peak_potential_pct": [10.0, float("nan")],
            "total_return_pct": [6.0, float("nan")],
            "entry_price": [100.0, 100.0],
            "disclosure_date": pd.to_datetime(["2024-01-01", "2024-02-01"]),
        })
        idx = pd.Index(["Alice"])
        peak, _ = _hit_rates_by_member(purchases, idx, threshold=5.0)
        self.assertAlmostEqual(peak.loc["Alice"], 100.0, places=5,
                               msg="peak_hit_rate counted NaN as miss")


class TestSalesHitRateNaN(unittest.TestCase):
    """sales._compute_member_stats had the same NaN-as-miss bug."""

    def test_realized_hit_rate_excludes_nan(self):
        from analyzer.member_ranking.sales import _compute_member_stats
        # Two rows: one positive total_return, one NaN total_return.
        grp = pd.DataFrame({
            "decayed_return_pct": [5.0, 3.0],
            "peak_potential_pct": [8.0, 6.0],
            "spy_alpha_pct": [1.0, 0.5],
            "total_return_pct": [5.0, float("nan")],  # 1 valid win, 1 NaN
        })
        # invert_returns=False so returns are taken as-is
        stats = _compute_member_stats("Alice", grp, market_prior=0.5,
                                      threshold=3.0, invert_returns=False)
        self.assertIsNotNone(stats)
        rate = stats["realized_hit_rate_pct"]
        # Only 1 valid observation (5.0 > 0) → 100%; old code gave 50%
        self.assertAlmostEqual(rate, 100.0, places=5,
                               msg=f"realized_hit_rate={rate}; NaN counted as miss")


# ── Bug #4: Kelly NaN fallback ───────────────────────────────────────────────

class TestKellyNaNFallback(unittest.TestCase):
    """
    When all returns in avg_return_pct are positive the losing slice is empty
    and .mean() returns NaN.  bool(NaN) is True in Python so `not avg_loss`
    was False, and `NaN <= 0` is also False — the fallback was never reached.
    NaN propagated through payout_ratio into kelly_fraction, eventually
    producing an empty portfolio.
    """

    def _make_recs_all_positive(self):
        return pd.DataFrame({
            "ticker": ["A", "B", "C"],
            "signal_score": [10.0, 8.0, 6.0],
            "member": ["m1", "m2", "m3"],
            "crash_prob": [0.0, 0.0, 0.0],
            "avg_return_pct": [5.0, 3.0, 1.0],  # all positive → avg_loss=NaN
        })

    def test_all_positive_returns_produces_non_empty_portfolio(self):
        from analyzer.portfolio.kelly import build_kelly_portfolio, KellyConfig
        recs = self._make_recs_all_positive()
        portfolio = build_kelly_portfolio(recs, KellyConfig())
        self.assertGreater(len(portfolio), 0,
                           "portfolio is empty; NaN avg_loss fallback not triggered")

    def test_all_positive_returns_uses_default_avg_loss(self):
        from analyzer.portfolio.kelly import _estimate_win_loss, KellyConfig
        recs = self._make_recs_all_positive()
        cfg = KellyConfig(default_avg_loss=0.012)
        avg_win, avg_loss = _estimate_win_loss(recs, cfg)
        self.assertFalse(math.isnan(avg_loss),
                         "avg_loss is NaN; fallback should have triggered")
        self.assertGreater(avg_loss, 0.0,
                           "avg_loss must be positive")

    def test_all_negative_returns_uses_default_avg_win(self):
        from analyzer.portfolio.kelly import _estimate_win_loss, KellyConfig
        recs = pd.DataFrame({
            "ticker": ["A"],
            "signal_score": [5.0],
            "member": ["m1"],
            "crash_prob": [0.0],
            "avg_return_pct": [-3.0],   # all negative → avg_win=NaN
        })
        cfg = KellyConfig(default_avg_win=0.015)
        avg_win, avg_loss = _estimate_win_loss(recs, cfg)
        self.assertFalse(math.isnan(avg_win),
                         "avg_win is NaN; fallback should have triggered")

    def test_estimate_win_loss_nan_does_not_propagate(self):
        from analyzer.portfolio.kelly import _estimate_win_loss, KellyConfig
        cfg = KellyConfig()
        recs = pd.DataFrame({
            "avg_return_pct": [5.0, 3.0],  # no losses
        })
        avg_win, avg_loss = _estimate_win_loss(recs, cfg)
        self.assertTrue(math.isfinite(avg_win))
        self.assertTrue(math.isfinite(avg_loss))


# ── Bug #5: Sweep alpha_slope sign inversion ─────────────────────────────────

class TestSweepAlphaSlopeSign(unittest.TestCase):
    """
    alpha_slope was computed as rank5_alpha - rank1_alpha.
    For a well-calibrated ranker rank1_alpha > rank5_alpha, so the old
    formula produced negative slopes for GOOD configs.
    nlargest(alpha_slope) then selected the worst configs (rank5 >> rank1).

    Fix: alpha_slope = rank1_alpha - rank5_alpha.
    A good config now has alpha_slope > 0.
    """

    def _make_result_with_rank_alphas(self, rank1, rank5):
        from sweep import SweepResult
        r = SweepResult(
            horizon=90, frequency_days=30, training_lookback_days=365,
            min_buyers=2, top_n=5, decay_lambda=0.005, bayes_prior_strength=20,
            rank1_alpha=rank1, rank5_alpha=rank5,
        )
        return r




    def test_sweep_module_uses_correct_sign(self):
        """Sign regression guard: executed alpha_slope is rank1 - rank5."""
        from datetime import date
        from unittest.mock import patch

        import analyzer.validation as validation
        from analyzer.pipeline import BacktestParams
        from analyzer.validation import _backtest_core

        recs = pd.DataFrame({
            "rank": [1, 2, 3, 4, 5],
            "ticker": ["A", "B", "C", "D", "E"],
        })
        evaluated = pd.DataFrame({
            "rank": [1, 2, 3, 4, 5],
            "ticker": ["A", "B", "C", "D", "E"],
            "bt_alpha_pct": [5.0, 2.0, 0.0, -2.0, -5.0],
            "bt_return_pct": [5.0, 2.0, 0.0, -2.0, -5.0],
        })
        params = BacktestParams(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 1),
            horizon=60,
            frequency_days=30,
            min_buyers=2,
            top_n=5,
        )

        with (
            patch.object(validation.analysis, "backtest_recommendations", return_value=recs),
            patch.object(validation.analysis, "evaluate_backtest", return_value=evaluated),
        ):
            result, _ = _backtest_core(
                all_transactions=pd.DataFrame(),
                prices=pd.DataFrame(),
                params=params,
                signals=pd.DataFrame(),
                bayes_prior_strength=20.0,
                decay_lambda=0.005,
                scoring_mode="shrunk_alpha",
            )

        self.assertEqual(result.rank1_alpha, 5.0)
        self.assertEqual(result.alpha_slope, 10.0)


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
        entry_prices = pd.DataFrame({
            "member": ["Alice"],
            "ticker": ["TICK"],
            "disclosure_date": pd.to_datetime(["2024-01-01"]),
            "transaction_type": ["Purchase"],
            "entry_price": [100.0],
        })
        # TICK column is all-NaN → _price_arrays returns (None, None) → kernel skips
        prices = pd.DataFrame({
            "TICK": [float("nan"), float("nan")],
            "SPY":  [400.0, 401.0],
        }, index=pd.to_datetime(["2024-01-01", "2024-02-01"]))
        signals = calculate_signal_potential(entry_prices, prices, [30])
        self.assertEqual(len(signals), 1)
        row = signals.iloc[0]
        self.assertTrue(
            pd.isna(row["total_return_pct"]),
            f"total_return_pct={row['total_return_pct']}; should be NaN when ticker has no prices"
        )

    def test_decayed_spy_return_is_nan_when_no_spy_column(self):
        """prices_df without SPY column → spy_has=False → decayed_spy_return NaN."""
        from analyzer.signals.core import calculate_signal_potential
        entry_prices = pd.DataFrame({
            "member": ["Alice"],
            "ticker": ["TICK"],
            "disclosure_date": pd.to_datetime(["2024-01-01"]),
            "transaction_type": ["Purchase"],
            "entry_price": [100.0],
        })
        # No SPY column → spy arrays are None → r_spy_wsum stays 0 → spy_cum=NaN
        prices_no_spy = pd.DataFrame({
            "TICK": [100.0, 101.0, 102.0],
        }, index=pd.to_datetime(["2024-01-01", "2024-01-15", "2024-02-01"]))
        signals = calculate_signal_potential(entry_prices, prices_no_spy, [30])
        self.assertEqual(len(signals), 1)
        row = signals.iloc[0]
        self.assertTrue(
            pd.isna(row["decayed_spy_return_pct"]),
            f"decayed_spy_return_pct={row['decayed_spy_return_pct']}; should be NaN without SPY"
        )

    def test_missing_window_not_counted_in_dynamic_prior(self):
        """NaN decayed_return_pct must not pollute the dynamic prior."""
        from analyzer.signals.filters import _compute_dynamic_prior
        # 2 rows: 1 with positive return, 1 with NaN
        signals = pd.DataFrame({
            "member": ["Alice", "Alice"],
            "ticker": ["A", "B"],
            "signal_type": ["Purchase", "Purchase"],
            "horizon_days": [90, 90],
            "decayed_return_pct": [5.0, float("nan")],
            "entry_price": [100.0, 100.0],
            "disclosure_date": pd.to_datetime(["2024-01-01", "2024-02-01"]),
        })
        prior = _compute_dynamic_prior(signals, 90)
        # Only 1 valid row, positive → up_prob = 1.0, clipped to 0.90
        self.assertAlmostEqual(prior, 0.90, places=5,
                               msg=f"prior={prior}; missing window counted as loss")

    def test_spy_alpha_pct_is_nan_without_spy_data(self):
        """When prices_df has no SPY column, spy_alpha_pct must be NaN."""
        from analyzer.signals.core import calculate_signal_potential
        entry_prices = pd.DataFrame({
            "member": ["Alice"],
            "ticker": ["TICK"],
            "disclosure_date": pd.to_datetime(["2024-01-01"]),
            "transaction_type": ["Purchase"],
            "entry_price": [100.0],
        })
        prices_no_spy = pd.DataFrame({
            "TICK": [100.0, 101.0, 102.0],
        }, index=pd.to_datetime(["2024-01-01", "2024-01-15", "2024-02-01"]))
        signals = calculate_signal_potential(entry_prices, prices_no_spy, [30])
        row = signals.iloc[0]
        self.assertTrue(
            pd.isna(row["spy_alpha_pct"]),
            "spy_alpha_pct should be NaN when no SPY data is available"
        )


if __name__ == "__main__":
    unittest.main()
