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
            "decayed_return_pct": [row[3] for row in rows],
            "signal_type": ["Purchase"] * len(rows),
            "horizon_days": [90] * len(rows),
            "window_complete": [True] * len(rows),
            "peak_potential_pct": [10.0] * len(rows),
            "spy_alpha_pct": [1.0] * len(rows),
            "total_spy_alpha_pct": [1.0] * len(rows),
            "entry_price": [100.0] * len(rows),
            "amount_midpoint": [1.0] * len(rows),
        }
    )


def test_episode_span_is_bounded_by_episode_start():
    signals = _signals(
        [
            ("Alice", "AAPL", "2024-01-01", 3.0),
            ("Alice", "AAPL", "2024-01-13", 4.0),
            ("Alice", "AAPL", "2024-01-28", 5.0),
        ]
    )

    collapsed = _collapse_to_episodes(signals)

    assert len(collapsed) == 2
    assert collapsed["episode_count"].tolist() == [2, 1]


def test_episode_does_not_chain_past_fourteen_days():
    signals = _signals(
        [
            ("Alice", "AAPL", "2024-01-01", 3.0),
            ("Alice", "AAPL", "2024-01-14", 4.0),
            ("Alice", "AAPL", "2024-01-28", 5.0),
        ]
    )

    collapsed = _collapse_to_episodes(signals)

    assert len(collapsed) == 2
    assert collapsed["episode_count"].tolist() == [2, 1]


def test_normal_normal_fit_is_scale_equivariant_at_one_millionth():
    outcomes = np.array([1.0, 2.0, -1.0, 0.0])
    groups = np.array(["A", "A", "B", "B"])
    base = normal_normal_posteriors(outcomes, groups, prior_strength=2.0)
    scaled = normal_normal_posteriors(outcomes * 1e-6, groups, prior_strength=2.0)

    np.testing.assert_allclose(
        scaled["posterior_mean"], base["posterior_mean"] * 1e-6, rtol=1e-12
    )
    np.testing.assert_allclose(
        scaled["posterior_std"], base["posterior_std"] * 1e-6, rtol=1e-12
    )
    np.testing.assert_allclose(scaled["shrinkage"], base["shrinkage"], rtol=1e-12)
    np.testing.assert_allclose(
        scaled["effective_information"], base["effective_information"]
    )
