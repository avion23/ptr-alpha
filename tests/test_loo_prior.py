import unittest

import numpy as np
import pandas as pd

from analyzer.member_ranking.ranking import rank_members
from analyzer.member_ranking.sales import rank_sales


PRIOR_STRENGTH = 20.0


def _signals(rows: list[tuple[str, str, str, float]], signal_type: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "member": [row[0] for row in rows],
            "ticker": [row[1] for row in rows],
            "disclosure_date": pd.to_datetime([row[2] for row in rows]),
            "signal_type": signal_type,
            "horizon_days": 90,
            "window_complete": True,
            "decayed_return_pct": [row[3] for row in rows],
            "peak_potential_pct": [max(row[3], 0.0) for row in rows],
            "spy_alpha_pct": [row[3] for row in rows],
            "total_spy_alpha_pct": [row[3] for row in rows],
            "entry_price": 10.0,
            "amount_midpoint": 1_000.0,
        }
    )


class TestLeaveOneOutPrior(unittest.TestCase):
    def test_purchase_prior_is_common_and_does_not_reverse_perfect_member(self):
        signals = _signals(
            [
                ("Strong", "S1", "2024-01-01", 10.0),
                ("Strong", "S2", "2024-02-01", 8.0),
                ("Strong", "S3", "2024-03-01", 6.0),
                ("Weak", "W1", "2024-01-01", -4.0),
                ("Weak", "W2", "2024-02-01", -6.0),
            ],
            "Purchase",
        )

        ranked = rank_members(signals, _bayes_prior_strength=PRIOR_STRENGTH).set_index(
            "member"
        )

        common_prior = 3 / 5
        strong_expected = (common_prior * PRIOR_STRENGTH + 3) / (PRIOR_STRENGTH + 3)
        weak_expected = common_prior * PRIOR_STRENGTH / (PRIOR_STRENGTH + 2)
        self.assertAlmostEqual(
            ranked.loc["Strong", "bayes_win_prob"], strong_expected, places=3
        )
        self.assertAlmostEqual(
            ranked.loc["Weak", "bayes_win_prob"], weak_expected, places=3
        )
        self.assertGreater(
            ranked.loc["Strong", "bayes_win_prob"],
            ranked.loc["Weak", "bayes_win_prob"],
        )
        self.assertEqual(ranked["prior_win_prob"].nunique(), 1)
        self.assertNotIn("posterior_lift", ranked.columns)

    def test_single_member_uses_global_prior_fallback(self):
        signals = _signals(
            [
                ("Solo", "A", "2024-01-01", 5.0),
                ("Solo", "B", "2024-02-01", -2.0),
            ],
            "Purchase",
        )

        probability = rank_members(signals).iloc[0]["bayes_win_prob"]

        self.assertGreater(probability, 0.0)
        self.assertLess(probability, 1.0)

    def test_purchase_output_omits_bayes_factor(self):
        signals = _signals([("Solo", "A", "2024-01-01", 5.0)], "Purchase")

        self.assertNotIn("bayes_factor", rank_members(signals).columns)

    def test_identical_evidence_uses_identical_common_prior(self):
        signals = _signals(
            [
                ("A", "A1", "2024-01-01", 4.0),
                ("B", "B1", "2024-01-01", 4.0),
                ("Peer", "P1", "2024-01-01", -4.0),
            ],
            "Purchase",
        )

        ranked = rank_members(signals).set_index("member")

        self.assertEqual(
            ranked.loc["A", "prior_win_prob"], ranked.loc["B", "prior_win_prob"]
        )
        self.assertEqual(
            ranked.loc["A", "bayes_win_prob"], ranked.loc["B", "bayes_win_prob"]
        )

    def test_purchase_success_is_endpoint_excess_alpha_not_decayed_direction(self):
        signals = _signals(
            [
                ("Mismatch", "A", "2024-01-01", 5.0),
                ("Positive", "B", "2024-01-01", 2.0),
            ],
            "Purchase",
        )
        signals.loc[signals["member"] == "Mismatch", "total_spy_alpha_pct"] = -1.0

        ranked = rank_members(signals).set_index("member")

        self.assertEqual(ranked.loc["Mismatch", "prob_up_given_buy"], 0.0)
        self.assertEqual(ranked.loc["Positive", "prob_up_given_buy"], 1.0)

    def test_all_zero_endpoint_alpha_has_finite_regularized_posteriors(self):
        signals = _signals(
            [
                ("A", "A1", "2024-01-01", 0.0),
                ("A", "A2", "2024-02-01", 0.0),
                ("B", "B1", "2024-01-01", 0.0),
                ("B", "B2", "2024-02-01", 0.0),
            ],
            "Purchase",
        )

        ranked = rank_members(signals)

        columns = [
            "shrunk_alpha",
            "shrunk_alpha_std",
            "alpha_shrinkage",
            "alpha_effective_information",
        ]
        self.assertTrue(np.isfinite(ranked[columns].to_numpy()).all())
        self.assertTrue((ranked["shrunk_alpha_std"] > 0).all())

    def test_prior_strength_controls_continuous_shrinkage_across_range(self):
        signals = _signals(
            [
                ("Strong", "S1", "2024-01-01", 10.0),
                ("Strong", "S2", "2024-02-01", 12.0),
                ("Weak", "W1", "2024-01-01", -10.0),
                ("Weak", "W2", "2024-02-01", -12.0),
            ],
            "Purchase",
        )

        fits = {
            strength: rank_members(
                signals, _bayes_prior_strength=float(strength)
            ).set_index("member")
            for strength in (1, 20, 1000)
        }

        means = [abs(fits[s].loc["Strong", "shrunk_alpha"]) for s in (1, 20, 1000)]
        shrinkages = [fits[s].loc["Strong", "alpha_shrinkage"] for s in (1, 20, 1000)]
        self.assertGreater(means[0], means[1])
        self.assertGreater(means[1], means[2])
        self.assertLess(shrinkages[0], shrinkages[1])
        self.assertLess(shrinkages[1], shrinkages[2])

    def test_sales_prior_excludes_members_own_episodes_and_output_omits_factor(self):
        signals = _signals(
            [
                ("Early Seller", "A", "2024-01-01", -10.0),
                ("Early Seller", "B", "2024-02-01", -8.0),
                ("Early Seller", "C", "2024-03-01", -6.0),
                ("Late Seller", "D", "2024-01-01", 5.0),
            ],
            "Sale",
        )

        ranked = rank_sales(signals).set_index("member")

        early_seller_prior = 0.10  # The other member has zero loss-avoidance wins.
        expected = (early_seller_prior * PRIOR_STRENGTH + 3) / (PRIOR_STRENGTH + 3)
        self.assertAlmostEqual(
            ranked.loc["Early Seller", "bayes_win_prob"], expected, places=3
        )
        self.assertNotIn("bayes_factor", ranked.columns)

    def test_purchase_prior_counts_collapsed_episodes(self):
        signals = _signals(
            [
                ("Target", "T", "2024-01-01", 5.0),
                ("Peer", "P", "2024-01-01", 8.0),
                ("Peer", "P", "2024-01-05", 8.0),
                ("Peer", "Q", "2024-02-01", -4.0),
            ],
            "Purchase",
        )

        ranked = rank_members(signals, _bayes_prior_strength=PRIOR_STRENGTH).set_index(
            "member"
        )

        common_prior = 2 / 3  # Target win plus one Peer win in three episodes.
        expected = (common_prior * PRIOR_STRENGTH + 1) / (PRIOR_STRENGTH + 1)
        self.assertEqual(ranked.loc["Peer", "purchase_trades"], 2)
        self.assertAlmostEqual(
            ranked.loc["Target", "bayes_win_prob"], expected, places=3
        )


if __name__ == "__main__":
    unittest.main()
