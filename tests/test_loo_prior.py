import unittest

import numpy as np
import pandas as pd

from analyzer.member_ranking.ranking import rank_members
from analyzer.member_ranking.sales import rank_sales


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


class TestEmpiricalMemberPooling(unittest.TestCase):
    def test_purchase_rank_uses_endpoint_alpha_and_observed_hit_rate(self):
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

        ranked = rank_members(signals).set_index("member")

        self.assertGreater(ranked.loc["Strong", "shrunk_alpha"], 0.0)
        self.assertLess(ranked.loc["Weak", "shrunk_alpha"], 0.0)
        self.assertEqual(ranked.loc["Strong", "prob_up_given_buy"], 1.0)
        self.assertEqual(ranked.loc["Weak", "prob_up_given_buy"], 0.0)
        for obsolete in (
            "bayes_win_prob",
            "prior_win_prob",
            "posterior_lift",
            "conviction_score",
            "bayes_factor",
        ):
            self.assertNotIn(obsolete, ranked.columns)

    def test_identical_evidence_gets_identical_pooled_estimate(self):
        signals = _signals(
            [
                ("A", "A1", "2024-01-01", 4.0),
                ("B", "B1", "2024-01-01", 4.0),
                ("Peer", "P1", "2024-01-01", -4.0),
            ],
            "Purchase",
        )

        ranked = rank_members(signals).set_index("member")

        self.assertEqual(ranked.loc["A", "shrunk_alpha"], ranked.loc["B", "shrunk_alpha"])
        self.assertEqual(
            ranked.loc["A", "alpha_shrinkage"], ranked.loc["B", "alpha_shrinkage"]
        )

    def test_more_repeated_information_reduces_shrinkage(self):
        rows = [
            ("Dense", f"D{i}", f"2024-{1 + i // 2:02d}-{1 + (i % 2) * 16:02d}", 10.0 + (-1) ** i)
            for i in range(8)
        ] + [
            ("Sparse", "S1", "2024-01-01", -8.0),
            ("Sparse", "S2", "2024-02-01", -10.0),
        ]
        ranked = rank_members(_signals(rows, "Purchase")).set_index("member")

        self.assertLess(
            ranked.loc["Dense", "alpha_shrinkage"],
            ranked.loc["Sparse", "alpha_shrinkage"],
        )
        self.assertGreater(
            ranked.loc["Dense", "alpha_effective_information"],
            ranked.loc["Sparse", "alpha_effective_information"],
        )

    def test_success_is_endpoint_excess_alpha_not_decayed_direction(self):
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

    def test_episode_collapse_controls_effective_trade_count(self):
        signals = _signals(
            [
                ("Target", "T", "2024-01-01", 5.0),
                ("Peer", "P", "2024-01-01", 8.0),
                ("Peer", "P", "2024-01-05", 8.0),
                ("Peer", "Q", "2024-02-01", -4.0),
            ],
            "Purchase",
        )

        ranked = rank_members(signals).set_index("member")

        self.assertEqual(ranked.loc["Peer", "purchase_trades"], 2)
        self.assertEqual(ranked.loc["Target", "purchase_trades"], 1)

    def test_sales_rank_reports_observed_loss_avoidance_without_prior_knobs(self):
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

        self.assertEqual(ranked.loc["Early Seller", "prob_up_given_sell"], 1.0)
        self.assertEqual(ranked.loc["Late Seller", "prob_up_given_sell"], 0.0)
        self.assertGreater(
            ranked.loc["Early Seller", "avg_spy_alpha_pct"],
            ranked.loc["Late Seller", "avg_spy_alpha_pct"],
        )
        for obsolete in ("bayes_win_prob", "posterior_lift", "bayes_factor"):
            self.assertNotIn(obsolete, ranked.columns)


if __name__ == "__main__":
    unittest.main()
