"""Smoke tests for analyzer.member_ranking module."""
import math
import unittest

from analyzer.member_ranking import (
    bayes_factor_against_market,
    bayesian_win_probability,
)


class TestBayesianWinProbability(unittest.TestCase):

    def test_returns_value_between_zero_and_one(self):
        for wins in range(0, 20):
            for losses in range(0, 20):
                p = bayesian_win_probability(wins, losses)
                self.assertGreaterEqual(p, 0.0)
                self.assertLessEqual(p, 1.0)

    def test_pulls_toward_prior_with_no_data(self):
        # With no observations, posterior = prior
        self.assertAlmostEqual(
            bayesian_win_probability(0, 0, market_prior=0.55, prior_strength=20.0),
            0.55,
            places=4,
        )

    def test_more_wins_increases_probability(self):
        low = bayesian_win_probability(1, 5, prior_strength=2.0)
        high = bayesian_win_probability(10, 5, prior_strength=2.0)
        self.assertLess(low, high)

    def test_custom_prior(self):
        # With strong prior and no data, posterior should match prior
        self.assertAlmostEqual(
            bayesian_win_probability(0, 0, market_prior=0.7, prior_strength=100.0),
            0.7,
            places=4,
        )


class TestBayesFactorAgainstMarket(unittest.TestCase):

    def test_no_observations_returns_one(self):
        # No data -> Bayes factor should be exactly 1 (no evidence either way)
        bf = bayes_factor_against_market(0, 0)
        self.assertEqual(bf, 1.0)

    def test_positive_observations_positive_bf(self):
        # Strongly outperforming market should give BF > 1
        bf = bayes_factor_against_market(20, 5, market_prior=0.55, prior_strength=5.0)
        self.assertGreater(bf, 1.0)

    def test_negative_observations_bf_positive(self):
        # Bayes factor should be strictly positive even under unfavorable data.
        bf = bayes_factor_against_market(2, 50, market_prior=0.55, prior_strength=1.0)
        self.assertGreater(bf, 0.0)

    def test_returns_finite_value(self):
        # Should not blow up to infinity under extreme conditions
        bf = bayes_factor_against_market(100, 1, market_prior=0.5, prior_strength=1.0)
        self.assertTrue(math.isfinite(bf))
        self.assertGreater(bf, 0.0)


if __name__ == "__main__":
    unittest.main()