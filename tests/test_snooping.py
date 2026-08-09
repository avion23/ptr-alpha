"""Tests for dependence-safe snooping corrections."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analyzer.snooping import (
    analyze_snooping,
    benjamini_hochberg,
    bonferroni_correction,
    deflated_sharpe_ratio,
    max_stat_block_permutation,
)


def _series(values):
    return pd.Series(
        values, index=pd.date_range("2020-01-01", periods=len(values), freq="D")
    )


class TestBasicCorrections:
    def test_bonferroni_controls_arbitrary_dependence(self):
        assert bonferroni_correction(36, 0.05) == pytest.approx(0.05 / 36)

    def test_bonferroni_rejects_invalid_trial_count(self):
        with pytest.raises(ValueError):
            bonferroni_correction(0)

    def test_bh_step_up_preserves_input_order(self):
        values = np.array([0.9, 0.001, 0.01, 0.5])
        rejected = benjamini_hochberg(values, 0.05)
        assert rejected.tolist() == [False, True, True, False]

    def test_bh_empty_fails(self):
        with pytest.raises(ValueError):
            benjamini_hochberg([])


class TestMaxStatPermutation:
    def test_strong_effect_beats_synchronized_block_null(self):
        rng = np.random.default_rng(8)
        series = {
            0: _series(2.0 + rng.normal(0, 0.2, 180)),
            1: _series(1.0 + rng.normal(0, 0.2, 180)),
        }
        result = max_stat_block_permutation(
            series,
            {0: 0, 1: 0},
            n_permutations=999,
            block_days=1,
            seed=4,
        )
        assert result.adjusted_p_values.shape == (2,)
        assert result.adjusted_p_values[0] <= 0.05
        assert result.n_permutations == 999
        assert len(result.null_max_statistics) == 999

    def test_all_zero_known_value_has_adjusted_p_one(self):
        result = max_stat_block_permutation(
            {0: _series(np.zeros(50))},
            {0: 2},
            n_permutations=99,
            block_days=5,
            seed=1,
        )
        assert result.observed_statistics[0] == 0.0
        assert result.adjusted_p_values[0] == 1.0

    def test_calendar_block_permutation_is_reproducible(self):
        values = {0: _series(np.arange(40.0) - 20.0)}
        first = max_stat_block_permutation(
            values, {0: 2}, n_permutations=50, block_days=4, seed=12
        )
        second = max_stat_block_permutation(
            values, {0: 2}, n_permutations=50, block_days=4, seed=12
        )
        assert np.array_equal(first.null_max_statistics, second.null_max_statistics)

    @pytest.mark.parametrize(
        "kwargs, message",
        [
            (
                {
                    "series_by_trial": {},
                    "lags_by_trial": {},
                    "n_permutations": 1,
                    "block_days": 1,
                    "seed": 0,
                },
                "must not be empty",
            ),
            (
                {
                    "series_by_trial": {0: _series([1, 2])},
                    "lags_by_trial": {0: 0},
                    "n_permutations": 0,
                    "block_days": 1,
                    "seed": 0,
                },
                "positive",
            ),
            (
                {
                    "series_by_trial": {0: _series([1, 2])},
                    "lags_by_trial": {0: 0},
                    "n_permutations": 1,
                    "block_days": 0,
                    "seed": 0,
                },
                "positive",
            ),
        ],
    )
    def test_invalid_permutation_inputs_fail(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            max_stat_block_permutation(**kwargs)


class TestCoherentSnoopingReport:
    def test_return_series_is_required(self):
        frame = pd.DataFrame(
            [{"name": "a", "overall_alpha": 1.0, "alpha_slope": 9.0, "sharpe": 1.0}]
        )
        with pytest.raises(ValueError, match="per_date_returns is required"):
            analyze_snooping(frame)

    def test_requested_config_uses_its_own_series_and_status(self):
        rng = np.random.default_rng(3)
        requested = _series(2.0 + rng.normal(0, 0.2, 180))
        other = _series(-2.0 + rng.normal(0, 0.2, 180))
        frame = pd.DataFrame(
            [
                {
                    "name": "requested",
                    "overall_alpha": 2.0,
                    "alpha_slope": -100.0,
                    "sharpe": 1.5,
                },
                {
                    "name": "other",
                    "overall_alpha": 100.0,
                    "alpha_slope": 1000.0,
                    "sharpe": 9.0,
                },
            ],
            index=[10, 20],
        )
        report = analyze_snooping(
            frame,
            best_config={"name": "requested"},
            n_tests=2,
            per_date_returns={0: requested, 1: other},
            lags_by_trial={0: 0, 1: 0},
            n_permutations=999,
            block_days=1,
            seed=5,
        )
        assert report.overall_alpha == pytest.approx(requested.mean())
        assert report.alpha_slope == -100.0
        assert report.t_statistic > 0
        assert report.significant_bonferroni is True
        assert report.max_stat_p_value <= 0.05
        assert report.deployable is True
        assert report.bh_rejected is False  # no false BH claim

    def test_requested_config_missing_from_series_fails(self):
        frame = pd.DataFrame(
            [
                {"name": "a", "overall_alpha": 1.0, "alpha_slope": 0.0, "sharpe": 1.0},
                {"name": "b", "overall_alpha": 2.0, "alpha_slope": 0.0, "sharpe": 1.0},
            ]
        )
        with pytest.raises(ValueError, match="no series"):
            analyze_snooping(
                frame,
                best_config={"name": "b"},
                per_date_returns={0: _series(np.ones(20))},
                n_permutations=10,
            )


class TestDeflatedSharpeSafety:
    def test_single_trial_is_defined(self):
        value = deflated_sharpe_ratio(1.0, n_trials=1, n_observations=100)
        assert 0.0 <= value <= 1.0

    def test_more_trials_do_not_improve_same_sharpe(self):
        few = deflated_sharpe_ratio(1.0, n_trials=2, n_observations=250)
        many = deflated_sharpe_ratio(1.0, n_trials=100, n_observations=250)
        assert many < few
