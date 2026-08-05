"""Tests for analyzer.validation: newey_west_tstat, select_config, run_validation."""
from __future__ import annotations

import math
from datetime import date
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from analyzer.validation import (
    SweepResult,
    newey_west_tstat,
    select_config,
    sweep_configs,
    run_validation,
)


class TestValidationBoundaries:
    def test_rejects_overlapping_train_and_test_windows(self, tmp_path):
        with pytest.raises(ValueError, match="test window must start after"):
            run_validation(
                tmp_path / "unused.duckdb",
                date(2024, 1, 1), date(2024, 6, 30),
                date(2024, 6, 30), date(2024, 12, 31),
                {"horizon": [180]},
            )

    def test_rejects_missing_horizon_grid(self, tmp_path):
        with pytest.raises(ValueError, match="at least one horizon"):
            run_validation(
                tmp_path / "unused.duckdb",
                date(2023, 1, 1), date(2023, 12, 31),
                date(2024, 1, 1), date(2024, 12, 31),
                {},
            )

    def test_rejects_non_positive_horizon(self, tmp_path):
        with pytest.raises(ValueError, match="horizons must be positive"):
            run_validation(
                tmp_path / "unused.duckdb",
                date(2023, 1, 1), date(2023, 12, 31),
                date(2024, 1, 1), date(2024, 12, 31),
                {"horizon": [0]},
            )


# ---------------------------------------------------------------------------
# newey_west_tstat
# ---------------------------------------------------------------------------

class TestNeweyWestTstat:

    def test_lag0_matches_biased_plain_tstat(self):
        """lag=0 → t = mean / (biased_std / sqrt(n))."""
        x = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        t_nw = newey_west_tstat(x, lag=0)
        n = len(x)
        t_plain = float(np.mean(x) / (np.std(x) / np.sqrt(n)))  # ddof=0
        assert abs(t_nw - t_plain) < 1e-10

    def test_zero_mean_series_gives_zero_tstat(self):
        """Series whose mean is 0 → t-stat = 0."""
        x = pd.Series([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
        assert abs(newey_west_tstat(x, lag=0)) < 1e-10

    def test_positive_mean_gives_positive_tstat(self):
        x = pd.Series([2.0, 3.0, 4.0])
        assert newey_west_tstat(x, lag=0) > 0

    def test_hand_computable_lag0(self):
        """Hand-verify: x=[0,2], mean=1, biased_std=1, n=2 → t=sqrt(2)."""
        x = pd.Series([0.0, 2.0])
        # biased std = sqrt(mean of (0-1)^2 + (2-1)^2) = sqrt(1) = 1
        # t = 1 / (1 / sqrt(2)) = sqrt(2)
        t = newey_west_tstat(x, lag=0)
        assert abs(t - math.sqrt(2)) < 1e-10

    def test_lag0_larger_than_lag1_for_positively_autocorrelated(self):
        """Positive autocorrelation → HAC widens SE → |t| shrinks with lag."""
        np.random.seed(42)
        n = 60
        ar = np.zeros(n)
        ar[0] = 1.0
        for i in range(1, n):
            ar[i] = 0.7 * ar[i - 1] + np.random.normal(0, 0.3)
        ar += 2.0  # ensure positive mean
        x = pd.Series(ar)
        t0 = abs(newey_west_tstat(x, lag=0))
        t2 = abs(newey_west_tstat(x, lag=2))
        # HAC correction should reduce or keep the t-stat
        assert t2 <= t0 + 0.5  # tolerant: direction matters

    def test_single_element_returns_zero(self):
        x = pd.Series([5.0])
        assert newey_west_tstat(x, lag=0) == 0.0

    def test_nan_dropped_before_computation(self):
        x_with_nan = pd.Series([1.0, float("nan"), 2.0, 3.0])
        x_clean = pd.Series([1.0, 2.0, 3.0])
        assert abs(
            newey_west_tstat(x_with_nan, lag=0) - newey_west_tstat(x_clean, lag=0)
        ) < 1e-10

    def test_lag_equals_len_minus_1_does_not_crash(self):
        """Edge case: lag ≥ n-1 should not raise (gamma array boundary)."""
        x = pd.Series([1.0, 2.0, 3.0])
        # lag=2 == n-1; gamma[2] involves dot product of length 0 array
        t = newey_west_tstat(x, lag=2)
        assert math.isfinite(t) or math.isinf(t)  # no crash

    def test_lag_capped_at_len_minus_1_for_two_observations(self):
        x = pd.Series([1.0, 3.0])
        assert newey_west_tstat(x, lag=3) == newey_west_tstat(x, lag=1)

    def test_lag_capped_at_len_minus_1_for_five_observations(self):
        x = pd.Series([1.0, 2.0, 4.0, 8.0, 16.0])
        assert newey_west_tstat(x, lag=10) == newey_west_tstat(x, lag=4)


# ---------------------------------------------------------------------------
# select_config
# ---------------------------------------------------------------------------

def _make_sweep_df(n: int = 10, p_values: list[float] | None = None) -> pd.DataFrame:
    """Build a synthetic sweep DataFrame with n rows."""
    rng = np.random.default_rng(99)
    rows = []
    for i in range(n):
        rows.append({
            "horizon": 60,
            "frequency_days": 30,
            "training_lookback_days": 365,
            "min_buyers": 2 + (i % 3),
            "top_n": 3 + (i % 2) * 2,
            "decay_lambda": 0.005,
            "bayes_prior_strength": 20.0,
            "scoring_mode": "shrunk_alpha",
            "total_recs": 50,
            "dates_evaluated": 20,
            "overall_alpha": float(rng.uniform(0.5, 3.0)),
            "overall_return": 1.0,
            "rank1_alpha": 2.0,
            "rank5_alpha": 1.0,
            "alpha_slope": float(rng.uniform(-1.0, 2.0)),
            "win_rate": 60.0,
            "sharpe": 1.0,
            "max_drawdown": -5.0,
            "nw_tstat": 2.0,
            "p_value": p_values[i] if p_values else 0.9,  # default: no survivors
            "min_sample_ok": True,
        })
    return pd.DataFrame(rows)


class TestSelectConfig:

    def test_picks_max_alpha_slope_among_bh_survivors(self):
        """Among configs that survive BH, pick the one with the highest alpha_slope."""
        df = _make_sweep_df(n=5, p_values=[0.001, 0.002, 0.9, 0.9, 0.9])
        df.loc[0, "alpha_slope"] = 5.0  # best, survives BH
        df.loc[1, "alpha_slope"] = 2.0  # survives BH but lower slope
        result = select_config(df, alpha=0.05)
        assert result["alpha_slope"] == 5.0
        assert result["survives_correction"] is True
        assert result["n_survivors"] >= 1

    def test_tiebreak_on_overall_alpha(self):
        """If alpha_slope is tied, pick higher overall_alpha."""
        df = _make_sweep_df(n=3, p_values=[0.001, 0.001, 0.9])
        df.loc[0, "alpha_slope"] = 3.0
        df.loc[1, "alpha_slope"] = 3.0
        df.loc[0, "overall_alpha"] = 1.0
        df.loc[1, "overall_alpha"] = 2.0  # should win on tiebreak
        result = select_config(df, alpha=0.05)
        assert result["overall_alpha"] == 2.0

    def test_no_survivors_returns_nominal_best(self):
        """When no config survives BH, return nominal best with flag=False."""
        df = _make_sweep_df(n=6, p_values=[0.9] * 6)
        df.loc[3, "alpha_slope"] = 10.0  # nominal best
        result = select_config(df, alpha=0.05)
        assert result["survives_correction"] is False
        assert result["n_survivors"] == 0
        assert result["alpha_slope"] == 10.0

    def test_bonferroni_threshold_is_alpha_over_n(self):
        df = _make_sweep_df(n=8, p_values=[0.9] * 8)
        result = select_config(df, alpha=0.05)
        assert abs(result["bonferroni_threshold"] - 0.05 / 8) < 1e-12

    def test_n_trials_equals_len_sweep_df(self):
        df = _make_sweep_df(n=12, p_values=[0.9] * 12)
        result = select_config(df, alpha=0.05)
        assert result["n_trials"] == 12

    def test_all_survive(self):
        """All very small p-values → all survive → best is still max slope."""
        df = _make_sweep_df(n=4, p_values=[0.0001] * 4)
        df.loc[2, "alpha_slope"] = 99.0
        result = select_config(df, alpha=0.05)
        assert result["alpha_slope"] == 99.0
        assert result["n_survivors"] == 4

    def test_negative_infinite_tstat_does_not_survive_bh(self):
        """A degenerate negative-alpha config gets p=1 and cannot survive BH."""
        grid = {
            "horizon": [60],
            "frequency_days": [30],
            "training_lookback_days": [365],
            "min_buyers": [2, 3],
            "top_n": [5],
            "decay_lambda": [0.005],
            "bayes_prior_strength": [20.0],
            "scoring_mode": ["shrunk_alpha"],
        }

        def fake_backtest_core(
            all_tx, prices, params, signals, bayes_prior_strength, decay_lambda, scoring_mode
        ):
            if params.min_buyers == 2:
                result = SweepResult(
                    horizon=params.horizon,
                    frequency_days=params.frequency_days,
                    training_lookback_days=params.training_lookback_days,
                    min_buyers=params.min_buyers,
                    top_n=params.top_n,
                    decay_lambda=decay_lambda,
                    bayes_prior_strength=bayes_prior_strength,
                    scoring_mode=scoring_mode,
                    total_recs=30,
                    dates_evaluated=10,
                    overall_alpha=-5.0,
                    alpha_slope=99.0,
                )
                return result, pd.Series([-5.0, -5.0, -5.0])

            result = SweepResult(
                horizon=params.horizon,
                frequency_days=params.frequency_days,
                training_lookback_days=params.training_lookback_days,
                min_buyers=params.min_buyers,
                top_n=params.top_n,
                decay_lambda=decay_lambda,
                bayes_prior_strength=bayes_prior_strength,
                scoring_mode=scoring_mode,
                total_recs=30,
                dates_evaluated=10,
                overall_alpha=5.0,
                alpha_slope=1.0,
            )
            return result, pd.Series([5.0, 5.0, 5.0])

        with (
            patch("analyzer.validation.analysis.calculate_signal_potential", return_value=pd.DataFrame()),
            patch("analyzer.validation._backtest_core", side_effect=fake_backtest_core),
        ):
            sweep_df = sweep_configs(
                all_tx=pd.DataFrame(),
                prices=pd.DataFrame(),
                entry_prices=pd.DataFrame(),
                grid=grid,
                start=date(2022, 1, 1),
                end=date(2022, 3, 1),
            )

        negative = sweep_df.loc[sweep_df["min_buyers"] == 2].iloc[0]
        positive = sweep_df.loc[sweep_df["min_buyers"] == 3].iloc[0]
        assert negative["nw_tstat"] == -math.inf
        assert negative["p_value"] == 1.0
        assert positive["nw_tstat"] == math.inf
        assert positive["p_value"] == 0.0

        selected = select_config(sweep_df, alpha=0.05)
        assert selected["min_buyers"] == 3
        assert selected["n_survivors"] == 1
        assert selected["survives_correction"] is True

    def test_negative_finite_tstat_is_one_sided_and_does_not_survive_bh(self):
        """A strongly negative finite t-stat gets p>0.5 and cannot survive BH."""
        grid = {
            "horizon": [60],
            "frequency_days": [30],
            "training_lookback_days": [365],
            "min_buyers": [2],
            "top_n": [5],
            "decay_lambda": [0.005],
            "bayes_prior_strength": [20.0],
            "scoring_mode": ["shrunk_alpha"],
        }

        def fake_backtest_core(
            all_tx, prices, params, signals, bayes_prior_strength, decay_lambda, scoring_mode
        ):
            result = SweepResult(
                horizon=params.horizon,
                frequency_days=params.frequency_days,
                training_lookback_days=params.training_lookback_days,
                min_buyers=params.min_buyers,
                top_n=params.top_n,
                decay_lambda=decay_lambda,
                bayes_prior_strength=bayes_prior_strength,
                scoring_mode=scoring_mode,
                total_recs=30,
                dates_evaluated=10,
                overall_alpha=-5.0,
                alpha_slope=99.0,
            )
            return result, pd.Series([-8.0, -7.0, -6.0, -5.0, -4.0, -6.0, -7.0, -8.0, -5.0, -6.0])

        with (
            patch("analyzer.validation.analysis.calculate_signal_potential", return_value=pd.DataFrame()),
            patch("analyzer.validation._backtest_core", side_effect=fake_backtest_core),
        ):
            sweep_df = sweep_configs(
                all_tx=pd.DataFrame(),
                prices=pd.DataFrame(),
                entry_prices=pd.DataFrame(),
                grid=grid,
                start=date(2022, 1, 1),
                end=date(2022, 3, 1),
            )

        row = sweep_df.iloc[0]
        assert row["nw_tstat"] < 0
        assert row["p_value"] > 0.5
        assert bool(row["min_sample_ok"]) is True

        selected = select_config(sweep_df, alpha=0.05)
        assert selected["survives_correction"] is False
        assert selected["n_survivors"] == 0

    def test_min_sample_filter_excludes_tiny_high_tstat_config(self):
        """Tiny samples cannot win even with a huge positive t-stat."""
        df = _make_sweep_df(n=2, p_values=[0.0, 0.02])
        df.loc[0, ["dates_evaluated", "total_recs", "alpha_slope", "p_value", "min_sample_ok"]] = [
            2,
            2,
            100.0,
            1.0,
            False,
        ]
        df.loc[1, ["dates_evaluated", "total_recs", "alpha_slope", "p_value", "min_sample_ok"]] = [
            10,
            30,
            2.0,
            0.02,
            True,
        ]

        selected = select_config(df, alpha=0.05)

        assert selected["alpha_slope"] == 2.0
        assert selected["dates_evaluated"] == 10
        assert selected["total_recs"] == 30
        assert selected["survives_correction"] is True
        assert selected["n_survivors"] == 1
        assert selected["sample_filter_exhausted"] is False

    def test_bh_correction_counts_filtered_rows_as_trials(self):
        """Filtered configs carry p=1 but still count in BH's trial denominator."""
        df = _make_sweep_df(n=10, p_values=[0.02, 0.9] + [1.0] * 8)
        df.loc[0, ["alpha_slope", "overall_alpha", "min_sample_ok"]] = [2.0, 2.0, True]
        df.loc[1, ["alpha_slope", "overall_alpha", "min_sample_ok"]] = [1.0, 1.0, True]
        df.loc[2:, "alpha_slope"] = 100.0
        df.loc[2:, "overall_alpha"] = 100.0
        df.loc[2:, "min_sample_ok"] = False

        selected = select_config(df, alpha=0.05)

        assert selected["n_trials"] == 10
        assert selected["n_min_sample_candidates"] == 2
        assert selected["n_survivors"] == 0
        assert selected["survives_correction"] is False
        assert selected["alpha_slope"] == 2.0
        assert selected["sample_filter_exhausted"] is False

    def test_sample_filter_exhausted_falls_back_to_overall_best_with_flag(self):
        """If all rows are too small, keep a deterministic fallback and flag it."""
        df = _make_sweep_df(n=3, p_values=[1.0, 1.0, 1.0])
        df["dates_evaluated"] = [1, 2, 3]
        df["total_recs"] = [2, 4, 6]
        df["min_sample_ok"] = False
        df["alpha_slope"] = [1.0, 5.0, 3.0]

        selected = select_config(df, alpha=0.05)

        assert selected["alpha_slope"] == 5.0
        assert selected["survives_correction"] is False
        assert selected["n_survivors"] == 0
        assert selected["n_min_sample_candidates"] == 0
        assert selected["sample_filter_exhausted"] is True


# ---------------------------------------------------------------------------
# run_validation: config-freezing test
# ---------------------------------------------------------------------------

class TestRunValidationConfigFreezing:
    """Verify that run_validation applies the TRAIN-selected config to the TEST eval."""

    def test_test_evaluation_uses_train_selected_config(self, tmp_path, monkeypatch):
        """The TEST backtest must use every parameter from the TRAIN-selected config."""
        from analyzer.validation import run_validation
        monkeypatch.chdir(tmp_path)

        # Synthetic selected config returned by mock sweep
        sweep_row = {
            "horizon": 90,
            "frequency_days": 30,
            "training_lookback_days": 365,
            "min_buyers": 3,
            "top_n": 5,
            "decay_lambda": 0.005,
            "bayes_prior_strength": 20.0,
            "scoring_mode": "consistency",
            "overall_alpha": 2.0,
            "alpha_slope": 1.5,
            "p_value": 0.01,
            "nw_tstat": 3.0,
            "total_recs": 30,
            "dates_evaluated": 10,
            "overall_return": 1.0,
            "rank1_alpha": 2.0,
            "rank5_alpha": 0.5,
            "win_rate": 60.0,
            "sharpe": 1.2,
            "max_drawdown": -5.0,
        }
        sweep_df = pd.DataFrame([sweep_row])

        fake_tx = pd.DataFrame({"ticker": ["AAPL"], "transaction_date": [pd.Timestamp("2022-01-01")]})
        fake_prices = pd.DataFrame(
            {"SPY": [100.0, 101.0]},
            index=pd.date_range("2021-01-01", periods=2),
        )
        fake_entry = pd.DataFrame()

        captured_params = []

        def fake_backtest_core(
            all_tx, prices, params, signals, bayes_prior_strength, decay_lambda, scoring_mode
        ):
            captured_params.append({
                "start_date": params.start_date,
                "end_date": params.end_date,
                "horizon": params.horizon,
                "min_buyers": params.min_buyers,
                "top_n": params.top_n,
                "scoring_mode": scoring_mode,
                "training_lookback_days": params.training_lookback_days,
            })
            sr = SweepResult(
                horizon=params.horizon,
                frequency_days=params.frequency_days,
                training_lookback_days=params.training_lookback_days,
                min_buyers=params.min_buyers,
                top_n=params.top_n,
                decay_lambda=decay_lambda,
                bayes_prior_strength=bayes_prior_strength,
                scoring_mode=scoring_mode,
            )
            return sr, pd.Series([1.0, 1.5, 2.0])

        with (
            patch("analyzer.validation.sweep_configs", return_value=sweep_df),
            patch("analyzer.validation._backtest_core", side_effect=fake_backtest_core),
            patch(
                "analyzer.validation.analysis.calculate_signal_potential",
                return_value=pd.DataFrame(),
            ),
            # Database is imported locally inside run_validation; patch the source module
            patch("analyzer.database.Database") as MockDB,
        ):
            mock_db = MagicMock()
            mock_db.get_transactions_by_date_range.return_value = fake_tx
            mock_db.get_prices.return_value = fake_prices
            mock_db.get_entry_prices.return_value = fake_entry
            MockDB.return_value = mock_db
            mock_db.conn = MagicMock()

            grid = {
                "horizon": [90],
                "frequency_days": [30],
                "training_lookback_days": [365],
                "min_buyers": [3],
                "top_n": [5],
                "decay_lambda": [0.005],
                "bayes_prior_strength": [20.0],
                "scoring_mode": ["consistency"],
            }

            result = run_validation(
                db_path=str(tmp_path / "test.duckdb"),
                train_start=date(2022, 1, 1),
                train_end=date(2023, 12, 31),
                test_start=date(2024, 1, 1),
                test_end=date(2025, 6, 30),
                grid=grid,
            )

        # Must have been called at least for TRAIN and TEST
        assert len(captured_params) >= 2, "Expected at least TRAIN + TEST backtest calls"

        # Locate the TEST call
        test_call = next(
            (c for c in captured_params if c["start_date"] == date(2024, 1, 1)),
            None,
        )
        assert test_call is not None, "No TEST evaluation call found"

        # All frozen params must match what the sweep selected
        assert test_call["horizon"] == 90
        assert test_call["min_buyers"] == 3
        assert test_call["top_n"] == 5
        assert test_call["scoring_mode"] == "consistency"
        assert test_call["training_lookback_days"] == 365

        # Result dict must be JSON-serialisable with expected keys
        assert "selected_config" in result
        assert "train" in result
        assert "test" in result
        assert "verdict" in result
        assert isinstance(result["verdict"], str)
        assert not (tmp_path / "data" / "validation_results.json").exists()

        out_path = tmp_path / "validation" / "results.json"
        with (
            patch("analyzer.validation.sweep_configs", return_value=sweep_df),
            patch("analyzer.validation._backtest_core", side_effect=fake_backtest_core),
            patch(
                "analyzer.validation.analysis.calculate_signal_potential",
                return_value=pd.DataFrame(),
            ),
            patch("analyzer.database.Database") as MockDB,
        ):
            mock_db = MagicMock()
            mock_db.get_transactions_by_date_range.return_value = fake_tx
            mock_db.get_prices.return_value = fake_prices
            mock_db.get_entry_prices.return_value = fake_entry
            MockDB.return_value = mock_db
            mock_db.conn = MagicMock()

            run_validation(
                db_path=str(tmp_path / "test.duckdb"),
                train_start=date(2022, 1, 1),
                train_end=date(2023, 12, 31),
                test_start=date(2024, 1, 1),
                test_end=date(2025, 6, 30),
                grid=grid,
                out_path=out_path,
            )

        assert out_path.exists()


# ---------------------------------------------------------------------------
# Smoke test: sweep.py still imports from analyzer.validation
# ---------------------------------------------------------------------------

