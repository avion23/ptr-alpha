"""Regression tests for statistical assumptions that guard model selection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analyzer.member_ranking.bayes import normal_normal_posteriors
from analyzer.snooping import max_stat_moving_block_bootstrap


def _dated(values, start: str = "2022-01-01") -> pd.Series:
    return pd.Series(
        values,
        index=pd.date_range(start, periods=len(values), freq="D"),
        dtype=float,
    )


def test_normal_normal_fully_pools_when_between_spread_is_sampling_noise():
    # Group means differ by less than their estimated sampling variance. A
    # hierarchical model has no evidence for persistent member heterogeneity.
    fit = normal_normal_posteriors(
        [0.0, 2.0, 1.0, 3.0],
        ["A", "A", "B", "B"],
        prior_strength=1.0,
    )

    assert (fit["between_var"] < fit["within_var"]).all()
    assert (fit["shrinkage"] > 0.999999).all()
    np.testing.assert_allclose(fit["posterior_mean"], [1.5, 1.5], atol=1e-12)
    assert np.isfinite(fit.to_numpy(dtype=float)).all()
    assert (fit["posterior_std"] > 0).all()


def test_normal_normal_preserves_clear_between_member_separation():
    fit = normal_normal_posteriors(
        [0.0, 2.0, 10.0, 12.0],
        ["A", "A", "B", "B"],
        prior_strength=1.0,
    )

    assert (fit["between_var"] > fit["within_var"]).all()
    assert (fit["shrinkage"] < 0.1).all()
    assert fit.loc["A", "posterior_mean"] < fit.loc["B", "posterior_mean"]


def test_max_stat_synchronizes_trials_with_identical_calendar_support():
    rng = np.random.default_rng(17)
    values = rng.normal(0.3, 1.0, 80)
    result = max_stat_moving_block_bootstrap(
        {0: _dated(values), 1: _dated(values * 0.7 + 0.1)},
        {0: 2, 1: 2},
        {0: 5, 1: 5},
        n_bootstrap=199,
        seed=7,
    )

    assert result.support_group_count == 1
    assert any("exact calendar-support" in item for item in result.assumptions)
    assert result.null_statistics.shape == (199, 2)


def test_max_stat_does_not_fabricate_alignment_across_shifted_calendars():
    rng = np.random.default_rng(23)
    values = rng.normal(0.4, 1.0, 80)
    result = max_stat_moving_block_bootstrap(
        {
            0: _dated(values, "2022-01-01"),
            1: _dated(values, "2022-01-02"),
        },
        {0: 2, 1: 2},
        {0: 5, 1: 5},
        n_bootstrap=199,
        seed=11,
    )

    # Each trial is a singleton support group. Its groupwise max p-value is its
    # marginal p-value, followed by the explicit two-group Bonferroni bound.
    assert result.support_group_count == 2
    np.testing.assert_allclose(
        result.adjusted_p_values,
        np.minimum(result.marginal_p_values * 2.0, 1.0),
    )


def test_max_stat_rejects_duplicate_calendar_observations():
    series = _dated(np.arange(20.0))
    series.index = pd.DatetimeIndex([series.index[0], *series.index[:-1]])

    with pytest.raises(ValueError, match="calendar index must be unique"):
        max_stat_moving_block_bootstrap(
            {0: series},
            {0: 0},
            {0: 2},
            n_bootstrap=10,
            seed=0,
        )
