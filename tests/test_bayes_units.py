import warnings

import numpy as np
import pandas as pd

from analyzer.member_ranking.bayes import normal_normal_posteriors
from analyzer.signals.filters import _collapse_to_episodes


def _signals(rows: list[tuple[str, str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "member": [row[0] for row in rows],
            "ticker": [row[1] for row in rows],
            "disclosure_date": pd.to_datetime([row[2] for row in rows]),
            "total_spy_alpha_pct": [row[3] for row in rows],
            "signal_type": ["Purchase"] * len(rows),
            "horizon_days": [90] * len(rows),
        }
    )


def test_episode_deduplicates_only_same_public_event():
    signals = _signals(
        [
            ("Alice", "AAPL", "2024-01-01", 3.0),
            ("Alice", "AAPL", "2024-01-01", 3.0),
            ("Alice", "AAPL", "2024-01-13", 4.0),
        ]
    )

    collapsed = _collapse_to_episodes(signals)

    assert len(collapsed) == 2
    assert collapsed["episode_count"].tolist() == [2, 1]


def test_normal_normal_fit_is_scale_equivariant_at_one_millionth():
    outcomes = np.array([1.0, 2.0, -1.0, 0.0])
    groups = np.array(["A", "A", "B", "B"])
    base = normal_normal_posteriors(outcomes, groups)
    scaled = normal_normal_posteriors(outcomes * 1e-6, groups)

    np.testing.assert_allclose(
        scaled["posterior_mean"], base["posterior_mean"] * 1e-6, rtol=1e-12
    )
    np.testing.assert_allclose(
        scaled["posterior_std"], base["posterior_std"] * 1e-6, rtol=1e-12
    )
    np.testing.assert_allclose(scaled["shrinkage"], base["shrinkage"], rtol=1e-12)


def test_unresolved_between_variance_warns_and_fully_pools():
    # Thin noisy data (2026-like): sampling noise swamps the spread of group
    # means, so the method-of-moments population variance truncates and every
    # posterior collapses to the global mean.
    outcomes = np.array(
        [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 60.0, -60.0, 60.0, -60.0]
    )
    groups = np.array(
        ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "H1", "H1", "H2", "H2"]
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit = normal_normal_posteriors(outcomes, groups)

    assert any(
        issubclass(warning.category, UserWarning)
        and "collapses every member to the global mean" in str(warning.message)
        for warning in caught
    )
    np.testing.assert_allclose(fit["shrinkage"], 1.0, rtol=0, atol=0)
    assert fit["posterior_mean"].nunique() == 1


def test_resolved_between_variance_does_not_warn():
    outcomes = np.array([1.0, 2.0, -1.0, 0.0])
    groups = np.array(["A", "A", "B", "B"])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fit = normal_normal_posteriors(outcomes, groups)

    assert (fit["shrinkage"] < 1.0).all()
