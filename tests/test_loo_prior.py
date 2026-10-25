import unittest
import warnings

import numpy as np
import pandas as pd

from analyzer.member_ranking.ranking import rank_members


def _signals(rows: list[tuple[str, str, str, float]]) -> pd.DataFrame:
    alpha = [row[3] for row in rows]
    return pd.DataFrame(
        {
            "member": [row[0] for row in rows],
            "ticker": [row[1] for row in rows],
            "disclosure_date": pd.to_datetime([row[2] for row in rows]),
            "signal_type": "Purchase",
            "horizon_days": 90,
            "window_complete": True,
            "total_return_pct": [value + 2.0 for value in alpha],
            "total_spy_alpha_pct": alpha,
        }
    )


class TestEmpiricalMemberPooling(unittest.TestCase):
    def test_purchase_rank_uses_endpoint_alpha_and_observed_hit_rates(self):
        signals = _signals(
            [
                ("Strong", "S1", "2024-01-01", 10.0),
                ("Strong", "S2", "2024-02-01", 8.0),
                ("Strong", "S3", "2024-03-01", 6.0),
                ("Weak", "W1", "2024-01-01", -4.0),
                ("Weak", "W2", "2024-02-01", -6.0),
            ]
        )

        ranked = rank_members(signals).set_index("member")

        self.assertGreater(ranked.loc["Strong", "shrunk_alpha_pct"], 0.0)
        self.assertLess(ranked.loc["Weak", "shrunk_alpha_pct"], 0.0)
        self.assertEqual(ranked.loc["Strong", "positive_alpha_rate"], 1.0)
        self.assertEqual(ranked.loc["Weak", "positive_alpha_rate"], 0.0)
        self.assertEqual(ranked.loc["Strong", "avg_spy_return_pct"], 2.0)

    def test_identical_evidence_gets_identical_pooled_estimate(self):
        ranked = rank_members(
            _signals(
                [
                    ("A", "A1", "2024-01-01", 4.0),
                    ("B", "B1", "2024-01-01", 4.0),
                    ("Peer", "P1", "2024-01-01", -4.0),
                ]
            )
        ).set_index("member")

        self.assertEqual(
            ranked.loc["A", "shrunk_alpha_pct"], ranked.loc["B", "shrunk_alpha_pct"]
        )
        self.assertEqual(
            ranked.loc["A", "alpha_shrinkage"], ranked.loc["B", "alpha_shrinkage"]
        )

    def test_more_episodes_reduce_shrinkage(self):
        rows = [
            (
                "Dense",
                f"D{i}",
                f"2024-{1 + i // 2:02d}-{1 + (i % 2) * 16:02d}",
                10.0 + (-1) ** i,
            )
            for i in range(8)
        ] + [
            ("Sparse", "S1", "2024-01-01", -8.0),
            ("Sparse", "S2", "2024-02-01", -10.0),
        ]
        ranked = rank_members(_signals(rows)).set_index("member")

        self.assertLess(
            ranked.loc["Dense", "alpha_shrinkage"],
            ranked.loc["Sparse", "alpha_shrinkage"],
        )
        self.assertGreater(
            ranked.loc["Dense", "purchase_episodes"],
            ranked.loc["Sparse", "purchase_episodes"],
        )

    def test_success_is_endpoint_excess_alpha(self):
        signals = _signals(
            [
                ("Negative Alpha", "A", "2024-01-01", -1.0),
                ("Positive Alpha", "B", "2024-01-01", 2.0),
            ]
        )
        ranked = rank_members(signals).set_index("member")

        self.assertEqual(ranked.loc["Negative Alpha", "positive_alpha_rate"], 0.0)
        self.assertEqual(ranked.loc["Positive Alpha", "positive_alpha_rate"], 1.0)

    def test_all_zero_endpoint_alpha_has_finite_posteriors(self):
        ranked = rank_members(
            _signals(
                [
                    ("A", "A1", "2024-01-01", 0.0),
                    ("A", "A2", "2024-02-01", 0.0),
                    ("B", "B1", "2024-01-01", 0.0),
                    ("B", "B2", "2024-02-01", 0.0),
                ]
            )
        )
        columns = ["shrunk_alpha_pct", "shrunk_alpha_std_pct", "alpha_shrinkage"]
        self.assertTrue(np.isfinite(ranked[columns].to_numpy()).all())
        self.assertTrue((ranked["shrunk_alpha_std_pct"] > 0).all())

    def test_same_public_event_is_one_episode(self):
        ranked = rank_members(
            _signals(
                [
                    ("Target", "T", "2024-01-01", 5.0),
                    ("Peer", "P", "2024-01-01", 8.0),
                    ("Peer", "P", "2024-01-01", 8.0),
                    ("Peer", "Q", "2024-02-01", -4.0),
                ]
            )
        ).set_index("member")

        self.assertEqual(ranked.loc["Peer", "purchase_episodes"], 2)
        self.assertEqual(ranked.loc["Target", "purchase_episodes"], 1)

    def test_resolved_between_variance_ranks_members_by_pooled_alpha(self):
        signals = _signals(
            [
                ("A", "A1", "2024-01-01", 20.0),
                ("A", "A2", "2024-02-01", 22.0),
                ("B", "B1", "2024-01-01", -20.0),
                ("B", "B2", "2024-02-01", -18.0),
            ]
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ranked = rank_members(signals)

        self.assertEqual(ranked["member"].tolist(), ["A", "B"])
        self.assertGreater(
            ranked.iloc[0]["shrunk_alpha_pct"], ranked.iloc[1]["shrunk_alpha_pct"]
        )

    def test_unresolved_ties_keep_alphabetical_order_and_count_member_dates(self):
        ranked_signals = _signals(
            [
                ("A", "A1", "2024-01-01", 0.0),
                ("A", "A2", "2024-01-01", 0.0),
                ("A", "A3", "2024-02-01", 0.0),
                ("B", "B1", "2024-01-01", 0.0),
                ("B", "B2", "2024-01-01", 0.0),
                ("B", "B3", "2024-02-01", 0.0),
            ]
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ranked = rank_members(ranked_signals)

        self.assertEqual(ranked["member"].tolist(), ["A", "B"])
        self.assertEqual(ranked["shrunk_alpha_pct"].nunique(), 1)
        warning = next(
            str(item.message)
            for item in caught
            if "between-variance is unresolved" in str(item.message)
        )
        self.assertIn("independent_date_member_groups=4", warning)

    def test_single_observation_members_warn_with_true_independent_group_count(self):
        signals = _signals(
            [
                ("C", "C1", "2024-01-01", 10.0),
                ("A", "A1", "2024-01-02", -10.0),
                ("B", "B1", "2024-01-03", 2.0),
            ]
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ranked = rank_members(signals)

        self.assertEqual(ranked["member"].tolist(), ["A", "B", "C"])
        warning = next(
            str(item.message) for item in caught if "UNIDENTIFIED" in str(item.message)
        )
        self.assertIn("independent_date_member_groups=3", warning)


if __name__ == "__main__":
    unittest.main()
