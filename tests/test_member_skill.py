import unittest

import numpy as np
import pandas as pd

from analyzer.member_skill import (
    MemberSkillPosterior,
    _recency_weight,
    estimate_member_skills,
    score_members_for_ticker,
)


REF_DATE = pd.Timestamp("2025-06-01")
HORIZON = 90


def _make_signals(members_alpha: dict[str, list[float]], horizon: int = HORIZON) -> pd.DataFrame:
    """Build a signals DataFrame with known alphas per member.

    members_alpha: {member_name: [alpha1, alpha2, ...]}
    Each alpha becomes a separate trade with evenly spaced disclosure dates,
    all well before ref_date - horizon so they qualify as fully-elapsed.
    """
    rows = []
    for member, alphas in members_alpha.items():
        for i, alpha in enumerate(alphas):
            # Spread disclosure dates so they are all eligible (<= ref_date - horizon)
            disc_date = REF_DATE - pd.Timedelta(days=horizon + 30 * (len(alphas) - i))
            rows.append({
                "member": member,
                "ticker": f"T{i % 3}",
                "disclosure_date": disc_date,
                "signal_type": "Purchase",
                "horizon_days": horizon,
                "entry_price": 100.0,
                "spy_alpha_pct": alpha,
                "decayed_return_pct": alpha + 2.0,
                "total_return_pct": alpha + 5.0,
                "total_spy_alpha_pct": alpha + 1.0,
                "peak_potential_pct": alpha + 10.0,
            })
    return pd.DataFrame(rows)


class TestRecencyWeight(unittest.TestCase):
    def test_recent_trade_weighted_more_than_old_trade(self):
        recent = _recency_weight(pd.Timestamp("2025-05-01"), REF_DATE, HORIZON)
        old = _recency_weight(pd.Timestamp("2024-06-01"), REF_DATE, HORIZON)
        self.assertGreater(recent, old)

    def test_weight_equals_one_at_zero_days(self):
        w = _recency_weight(REF_DATE, REF_DATE, 365)
        self.assertAlmostEqual(w, 1.0)

    def test_weight_decreases_monotonically(self):
        half_life = 365
        weights = [_recency_weight(REF_DATE - pd.Timedelta(days=d), REF_DATE, half_life) for d in range(0, 1000, 50)]
        for i in range(1, len(weights)):
            self.assertGreater(weights[i - 1], weights[i])


class TestEstimateMemberSkills(unittest.TestCase):
    def test_shrinkage_member_with_fewer_trades_shrinks_more(self):
        """Member with 1 trade shrinks more than member with 50 trades."""
        alphas_a = [10.0] * 50
        alphas_b = [20.0]
        signals = _make_signals({"A": alphas_a, "B": alphas_b})

        skills = estimate_member_skills(
            signals, min_episodes=1, prior_strength=5.0,
            recency_half_life_days=365, horizon=HORIZON, ref_date=REF_DATE,
        )

        self.assertIn("A", skills)
        self.assertIn("B", skills)
        self.assertGreater(skills["B"].shrinkage, skills["A"].shrinkage)

    def test_posterior_mean_between_raw_and_global(self):
        """posterior_mean is between raw_alpha and global_mean."""
        signals = _make_signals({"A": [20.0, 22.0, 18.0], "B": [5.0, 3.0, 7.0]})

        skills = estimate_member_skills(
            signals, min_episodes=1, prior_strength=5.0,
            recency_half_life_days=365, horizon=HORIZON, ref_date=REF_DATE,
        )

        alphas = [s.alpha_mean for s in skills.values()]
        global_mean = np.mean(alphas)
        for member, skill in skills.items():
            if member == "A":
                raw_alpha = np.mean([20.0, 22.0, 18.0])
                low, high = min(global_mean, raw_alpha), max(global_mean, raw_alpha)
                self.assertGreaterEqual(skill.alpha_mean, low - 0.1)
                self.assertLessEqual(skill.alpha_mean, high + 0.1)

    def test_posterior_std_decreases_with_more_episodes(self):
        """posterior_std is smaller for members with more episodes."""
        signals = _make_signals({
            "FEW": [10.0, 12.0],
            "MANY": [10.0] * 20,
        })

        skills = estimate_member_skills(
            signals, min_episodes=1, prior_strength=5.0,
            recency_half_life_days=365, horizon=HORIZON, ref_date=REF_DATE,
        )

        self.assertLess(skills["MANY"].alpha_std, skills["FEW"].alpha_std)

    def test_shrinkage_formula(self):
        """Verify shrinkage formula: prior_strength / (n + prior_strength)."""
        signals = _make_signals({"A": [10.0] * 10})

        skills = estimate_member_skills(
            signals, min_episodes=1, prior_strength=5.0,
            recency_half_life_days=365, horizon=HORIZON, ref_date=REF_DATE,
        )

        skill = skills["A"]
        expected_shrinkage = 5.0 / (10 + 5.0)
        self.assertAlmostEqual(skill.shrinkage, expected_shrinkage)

    def test_empty_signals_returns_empty(self):
        skills = estimate_member_skills(pd.DataFrame())
        self.assertEqual(skills, {})


class TestScoreMembersForTicker(unittest.TestCase):
    def test_with_known_members(self):
        skills = {
            "Good": MemberSkillPosterior(
                member="Good", alpha_mean=15.0, alpha_std=2.0,
                n_episodes=20, shrinkage=0.2,
            ),
            "Bad": MemberSkillPosterior(
                member="Bad", alpha_mean=-5.0, alpha_std=5.0,
                n_episodes=3, shrinkage=0.6,
            ),
        }
        expected_alpha, uncertainty = score_members_for_ticker(
            "AAPL", ["Good", "Bad"], skills,
        )
        self.assertGreater(expected_alpha, 0.0)
        self.assertGreater(uncertainty, 0.0)

    def test_no_relevant_members_returns_zero(self):
        skills = {
            "Good": MemberSkillPosterior(
                member="Good", alpha_mean=15.0, alpha_std=2.0,
                n_episodes=20, shrinkage=0.2,
            ),
        }
        expected_alpha, uncertainty = score_members_for_ticker(
            "AAPL", ["Unknown"], skills,
        )
        self.assertEqual(expected_alpha, 0.0)

    def test_single_member_returns_posterior_mean(self):
        skills = {
            "Solo": MemberSkillPosterior(
                member="Solo", alpha_mean=10.0, alpha_std=1.0,
                n_episodes=15, shrinkage=0.25,
            ),
        }
        expected_alpha, _ = score_members_for_ticker(
            "AAPL", ["Solo"], skills,
        )
        self.assertAlmostEqual(expected_alpha, 10.0, places=1)

    def test_inverse_uncertainty_weighting(self):
        """Member with lower std gets more weight."""
        skills = {
            "Precise": MemberSkillPosterior(
                member="Precise", alpha_mean=10.0, alpha_std=1.0,
                n_episodes=30, shrinkage=0.14,
            ),
            "Noisy": MemberSkillPosterior(
                member="Noisy", alpha_mean=20.0, alpha_std=10.0,
                n_episodes=3, shrinkage=0.62,
            ),
        }
        expected_alpha, _ = score_members_for_ticker(
            "AAPL", ["Precise", "Noisy"], skills,
        )
        self.assertLess(expected_alpha, 15.0)


class TestIntegrationWithAnalysis(unittest.TestCase):
    def test_score_ticker_by_buyers_with_skills(self):
        """score_ticker_by_buyers with member_skills parameter."""
        from analyzer.analysis import score_ticker_by_buyers

        transactions = pd.DataFrame({
            "member": ["Alice", "Bob"],
            "ticker": ["AAPL", "AAPL"],
            "transaction_date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "disclosure_date": pd.to_datetime(["2024-01-03", "2024-01-04"]),
            "transaction_type": ["Purchase", "Purchase"],
        })
        signals = pd.DataFrame({
            "member": ["Alice", "Bob"],
            "ticker": ["AAPL", "AAPL"],
            "signal_type": ["Purchase", "Purchase"],
            "horizon_days": [90, 90],
            "decayed_return_pct": [10.0, 5.0],
            "peak_potential_pct": [12.0, 7.0],
            "spy_alpha_pct": [10.0, 5.0],
        })
        member_rankings = pd.DataFrame({
            "member": ["Alice", "Bob"],
            "avg_spy_alpha_pct": [10.0, 5.0],
            "purchase_trades": [10, 5],
            "bayes_win_prob": [0.65, 0.55],
        })
        skills = {
            "Alice": MemberSkillPosterior(
                member="Alice", alpha_mean=10.0, alpha_std=2.0,
                n_episodes=10, shrinkage=0.33,
            ),
            "Bob": MemberSkillPosterior(
                member="Bob", alpha_mean=5.0, alpha_std=4.0,
                n_episodes=5, shrinkage=0.5,
            ),
        }
        score = score_ticker_by_buyers(
            "AAPL", transactions, signals,
            member_rankings=member_rankings,
            member_skills=skills,
            uncertainty_penalty_lambda=0.5,
        )
        self.assertIn("uncertainty_lambda", score.columns)
        self.assertEqual(score.iloc[0]["uncertainty_lambda"], 0.5)

    def test_score_ticker_by_buyers_lambda_zero_no_penalty(self):
        """lambda=0 gives no uncertainty penalty: base_signal_score == quality_adjusted_avg."""
        from analyzer.analysis import score_ticker_by_buyers

        transactions = pd.DataFrame({
            "member": ["Alice", "Bob"],
            "ticker": ["AAPL", "AAPL"],
            "transaction_date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "disclosure_date": pd.to_datetime(["2024-01-03", "2024-01-04"]),
            "transaction_type": ["Purchase", "Purchase"],
        })
        signals = pd.DataFrame({
            "member": ["Alice", "Bob"],
            "ticker": ["AAPL", "AAPL"],
            "signal_type": ["Purchase", "Purchase"],
            "horizon_days": [90, 90],
            "decayed_return_pct": [10.0, 8.0],
            "peak_potential_pct": [12.0, 10.0],
            "spy_alpha_pct": [10.0, 8.0],
        })
        skills = {
            "Alice": MemberSkillPosterior(
                member="Alice", alpha_mean=10.0, alpha_std=2.0,
                n_episodes=10, shrinkage=0.33,
            ),
            "Bob": MemberSkillPosterior(
                member="Bob", alpha_mean=8.0, alpha_std=3.0,
                n_episodes=5, shrinkage=0.5,
            ),
        }
        score = score_ticker_by_buyers(
            "AAPL", transactions, signals,
            member_skills=skills,
            uncertainty_penalty_lambda=0.0,
        )
        # With lambda=0, base_signal_score should equal quality_adjusted_avg
        # (weighted mean of posterior means by inverse uncertainty)
        inv_stds = np.array([1.0 / 2.0, 1.0 / 3.0])
        weights = inv_stds / inv_stds.sum()
        expected_qa = float(np.dot(weights, np.array([10.0, 8.0])))
        self.assertAlmostEqual(score.iloc[0]["base_signal_score"], round(expected_qa, 2), places=1)

    def test_score_ticker_by_buyers_without_skills_unchanged(self):
        """Without member_skills, behavior is unchanged."""
        from analyzer.analysis import score_ticker_by_buyers

        transactions = pd.DataFrame({
            "member": ["Alice", "Charlie"],
            "ticker": ["AAPL", "AAPL"],
            "transaction_date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "disclosure_date": pd.to_datetime(["2024-01-03", "2024-01-04"]),
            "transaction_type": ["Purchase", "Purchase"],
            "owner_code": [None, "DC"],
            "amount_midpoint": [100000.0, 100000.0],
        })
        signals = pd.DataFrame({
            "member": ["Alice", "Charlie"],
            "ticker": ["AAPL", "AAPL"],
            "signal_type": ["Purchase", "Purchase"],
            "horizon_days": [90, 90],
            "decayed_return_pct": [10.0, 10.0],
            "peak_potential_pct": [12.0, 12.0],
            "spy_alpha_pct": [10.0, 10.0],
        })
        score = score_ticker_by_buyers("AAPL", transactions, signals)
        self.assertEqual(score.iloc[0]["base_signal_score"], 10.0)
        self.assertEqual(score.iloc[0]["uncertainty_lambda"], 0.0)


if __name__ == "__main__":
    unittest.main()
