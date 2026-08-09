import unittest

import numpy as np
import pandas as pd

from analyzer.member_skill import (
    MemberSkillPosterior,
    estimate_member_skills,
    score_member_posteriors,
)


REF_DATE = pd.Timestamp("2025-06-01")
HORIZON = 90


def _make_signals(
    members_alpha: dict[str, list[float]], horizon: int = HORIZON
) -> pd.DataFrame:
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
            rows.append(
                {
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
                }
            )
    return pd.DataFrame(rows)


class TestEstimateMemberSkills(unittest.TestCase):
    def test_shrinkage_member_with_fewer_trades_shrinks_more(self):
        """Member with 1 trade shrinks more than member with 50 trades."""
        alphas_a = [10.0] * 50
        alphas_b = [20.0]
        signals = _make_signals({"A": alphas_a, "B": alphas_b})

        skills = estimate_member_skills(
            signals,
            min_episodes=1,
            prior_strength=5.0,
            recency_half_life_days=365,
            horizon=HORIZON,
            ref_date=REF_DATE,
        )

        self.assertIn("A", skills)
        self.assertIn("B", skills)
        self.assertGreater(skills["B"].shrinkage, skills["A"].shrinkage)

    def test_posterior_mean_between_raw_and_global(self):
        """posterior_mean is between raw_alpha and global_mean."""
        signals = _make_signals({"A": [20.0, 22.0, 18.0], "B": [5.0, 3.0, 7.0]})

        skills = estimate_member_skills(
            signals,
            min_episodes=1,
            prior_strength=5.0,
            recency_half_life_days=365,
            horizon=HORIZON,
            ref_date=REF_DATE,
        )

        alphas = [s.alpha_mean for s in skills.values()]
        global_mean = np.mean(alphas)
        for member, skill in skills.items():
            if member == "A":
                raw_alpha = np.mean([21.0, 23.0, 19.0])
                low, high = min(global_mean, raw_alpha), max(global_mean, raw_alpha)
                self.assertGreaterEqual(skill.alpha_mean, low - 0.1)
                self.assertLessEqual(skill.alpha_mean, high + 0.1)

    def test_one_episode_has_higher_posterior_std_than_ten(self):
        """The same per-episode distribution is more certain with more data."""
        signals = _make_signals(
            {
                "ONE": [10.0],
                "TEN": [10.0] * 10,
            }
        )

        skills = estimate_member_skills(
            signals,
            min_episodes=1,
            prior_strength=5.0,
            recency_half_life_days=365,
            horizon=HORIZON,
            ref_date=REF_DATE,
        )

        self.assertGreater(skills["ONE"].alpha_std, skills["TEN"].alpha_std)

    def test_effective_n_increases_uncertainty_and_shrinkage_for_old_trades(self):
        rows = []
        recent_days_ago = [100 + 20 * i for i in range(10)]
        old_days_ago = [100] + [500 + 50 * i for i in range(9)]
        for member, days_ago_values, center in (
            ("RECENT", recent_days_ago, 0.0),
            ("OLD", old_days_ago, 20.0),
        ):
            for i, days_ago in enumerate(days_ago_values):
                alpha = center + (-1.0 if i % 2 == 0 else 1.0)
                rows.append(
                    {
                        "member": member,
                        "ticker": f"T{i % 3}",
                        "disclosure_date": REF_DATE - pd.Timedelta(days=days_ago),
                        "signal_type": "Purchase",
                        "horizon_days": HORIZON,
                        "spy_alpha_pct": alpha,
                        "total_spy_alpha_pct": alpha,
                    }
                )

        skills = estimate_member_skills(
            pd.DataFrame(rows),
            min_episodes=1,
            prior_strength=5.0,
            recency_half_life_days=100,
            horizon=HORIZON,
            ref_date=REF_DATE,
        )

        self.assertEqual(skills["OLD"].n_episodes, skills["RECENT"].n_episodes)
        self.assertGreater(skills["OLD"].alpha_std, skills["RECENT"].alpha_std)
        self.assertGreater(skills["OLD"].shrinkage, skills["RECENT"].shrinkage)

    def test_uniform_recency_scaling_reduces_effective_information(self):
        recent = _make_signals({"A": [8.0, 10.0, 12.0], "B": [-2.0, 0.0, 2.0]})
        old = recent.copy()
        old["disclosure_date"] = old["disclosure_date"] - pd.Timedelta(days=730)

        recent_skills = estimate_member_skills(
            recent,
            prior_strength=1.0,
            recency_half_life_days=100,
            horizon=HORIZON,
            ref_date=REF_DATE,
        )
        old_skills = estimate_member_skills(
            old,
            prior_strength=1.0,
            recency_half_life_days=100,
            horizon=HORIZON,
            ref_date=REF_DATE,
        )

        self.assertLess(
            old_skills["A"].effective_information,
            recent_skills["A"].effective_information,
        )
        self.assertGreater(old_skills["A"].shrinkage, recent_skills["A"].shrinkage)
        self.assertGreater(old_skills["A"].alpha_std, recent_skills["A"].alpha_std)

    def test_global_mean_member_has_nonzero_finite_posterior_std(self):
        signals = _make_signals(
            {
                "LOW": [0.0] * 5,
                "CENTER": [10.0] * 5,
                "HIGH": [20.0] * 5,
            }
        )

        skills = estimate_member_skills(
            signals,
            min_episodes=1,
            prior_strength=5.0,
            recency_half_life_days=365,
            horizon=HORIZON,
            ref_date=REF_DATE,
        )

        self.assertGreater(skills["CENTER"].alpha_std, 0.0)
        self.assertTrue(all(np.isfinite(skill.alpha_std) for skill in skills.values()))

    def test_all_zero_endpoint_alpha_has_finite_regularized_posteriors(self):
        signals = _make_signals({"A": [0.0, 0.0], "B": [0.0, 0.0]})

        skills = estimate_member_skills(signals, ref_date=REF_DATE)

        self.assertEqual(set(skills), {"A", "B"})
        for posterior in skills.values():
            values = [
                posterior.alpha_mean,
                posterior.alpha_std,
                posterior.shrinkage,
                posterior.effective_information,
            ]
            self.assertTrue(np.isfinite(values).all())
            self.assertGreater(posterior.alpha_std, 0.0)


class TestScoreMembersForTicker(unittest.TestCase):
    def test_with_known_members(self):
        skills = {
            "Good": MemberSkillPosterior(
                member="Good",
                alpha_mean=15.0,
                alpha_std=2.0,
                n_episodes=20,
                effective_information=1.0,
                shrinkage=0.2,
            ),
            "Bad": MemberSkillPosterior(
                member="Bad",
                alpha_mean=-5.0,
                alpha_std=5.0,
                n_episodes=3,
                effective_information=1.0,
                shrinkage=0.6,
            ),
        }
        expected_alpha, uncertainty = score_member_posteriors(
            ["Good", "Bad"],
            skills,
        )
        self.assertGreater(expected_alpha, 0.0)
        self.assertGreater(uncertainty, 0.0)

    def test_no_relevant_members_returns_zero(self):
        skills = {
            "Good": MemberSkillPosterior(
                member="Good",
                alpha_mean=15.0,
                alpha_std=2.0,
                n_episodes=20,
                effective_information=1.0,
                shrinkage=0.2,
            ),
        }
        expected_alpha, uncertainty = score_member_posteriors(
            ["Unknown"],
            skills,
        )
        self.assertEqual(expected_alpha, 0.0)

    def test_single_member_returns_posterior_mean(self):
        skills = {
            "Solo": MemberSkillPosterior(
                member="Solo",
                alpha_mean=10.0,
                alpha_std=1.0,
                n_episodes=15,
                effective_information=1.0,
                shrinkage=0.25,
            ),
        }
        expected_alpha, _ = score_member_posteriors(
            ["Solo"],
            skills,
        )
        self.assertAlmostEqual(expected_alpha, 10.0, places=1)

    def test_inverse_uncertainty_weighting(self):
        """Member with lower std gets more weight."""
        skills = {
            "Precise": MemberSkillPosterior(
                member="Precise",
                alpha_mean=10.0,
                alpha_std=1.0,
                n_episodes=30,
                effective_information=1.0,
                shrinkage=0.14,
            ),
            "Noisy": MemberSkillPosterior(
                member="Noisy",
                alpha_mean=20.0,
                alpha_std=10.0,
                n_episodes=3,
                effective_information=1.0,
                shrinkage=0.62,
            ),
        }
        expected_alpha, uncertainty = score_member_posteriors(
            ["Precise", "Noisy"],
            skills,
        )
        expected = (10.0 / 1.0**2 + 20.0 / 10.0**2) / (1.0 / 1.0**2 + 1.0 / 10.0**2)
        self.assertAlmostEqual(expected_alpha, expected)
        self.assertAlmostEqual(uncertainty, np.sqrt(1.0 / 1.01))


class TestMemberPosteriorIdentityValidation(unittest.TestCase):
    def test_duplicate_member_evidence_is_rejected(self):
        skills = {
            "Alice": MemberSkillPosterior(
                member="Alice",
                alpha_mean=2.0,
                alpha_std=1.0,
                n_episodes=3,
                effective_information=2.0,
                shrinkage=0.5,
            )
        }

        with self.assertRaisesRegex(ValueError, "duplicate member identities"):
            score_member_posteriors(["Alice", "ALICE"], skills)


if __name__ == "__main__":
    unittest.main()
