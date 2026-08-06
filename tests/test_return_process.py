"""Tests for the OU return process model."""

import numpy as np

from analyzer.return_process import (
    OUParams,
    OUPosterior,
    ReturnProcessTracker,
    ar1_to_ou,
    build_prior,
    fit_ar1,
    fit_ou,
)


class TestFitAR1:
    def test_known_process(self):
        """AR(1) with known a=0.95, b=0.01 should recover."""
        rng = np.random.default_rng(42)
        a_true, b_true, s2_true = 0.95, 0.01, 0.001
        n = 500
        r = np.zeros(n)
        r[0] = b_true / (1 - a_true)
        for i in range(1, n):
            r[i] = b_true + a_true * r[i - 1] + rng.normal(0, np.sqrt(s2_true))

        a_est, b_est, s2_est = fit_ar1(r)
        assert abs(a_est - a_true) < 0.03, f"a={a_est:.3f} != {a_true}"
        assert abs(s2_est - s2_true) < 0.002, f"s2={s2_est:.4f} != {s2_true}"


class TestAR1ToOU:
    def test_roundtrip(self):
        """OU -> AR(1) -> OU should recover parameters."""
        ou = OUParams(theta=0.05, mu=0.10, sigma=0.20)
        a = ou.a
        b = ou.mu * ou.one_minus_a
        s2 = ou.sigma2_ou
        ou2 = ar1_to_ou(a, b, s2)
        assert abs(ou2.theta - ou.theta) < 1e-6
        assert abs(ou2.mu - ou.mu) < 1e-6

    def test_theta_clamp(self):
        """Negative a should clamp to positive."""
        ou = ar1_to_ou(-0.5, 0.01, 0.001)
        assert ou.theta > 0


class TestFitOU:
    def test_known_ou(self):
        """Fit OU to synthetic path, recover parameters."""
        rng = np.random.default_rng(42)
        theta, mu, sigma = 0.05, 0.10, 0.20
        dt = 1.0
        n = 500
        r = np.zeros(n)
        r[0] = mu
        for i in range(1, n):
            dr = theta * (mu - r[i - 1]) * dt + sigma * np.sqrt(dt) * rng.normal()
            r[i] = r[i - 1] + dr

        ou = fit_ou(r)
        assert abs(ou.theta - theta) < 0.03, f"theta={ou.theta:.3f}"
        assert abs(ou.mu - mu) < 0.05, f"mu={ou.mu:.3f}"

    def test_return_is_cumulative(self):
        """OU on cumulative returns should work."""
        rng = np.random.default_rng(123)
        daily_returns = rng.normal(0.001, 0.02, 200)
        cumulative = np.cumsum(daily_returns)
        ou = fit_ou(cumulative)
        assert ou.theta > 0


class TestBuildPrior:
    def test_with_curves(self):
        rng = np.random.default_rng(7)
        curves = []
        for _ in range(20):
            n = rng.integers(30, 100)
            daily = rng.normal(0.001, 0.02, n)
            curves.append(np.cumsum(daily))

        theta, beta, P, s2 = build_prior(curves)
        assert 0.001 < theta < 2.0
        assert P > 0
        assert s2 > 0


class TestOUPosterior:
    def test_v_positive_mu(self):
        """High mu should give positive V."""
        post = OUPosterior(
            mu_mean=0.10, mu_var=0.01, theta=0.05, sigma2_ou=0.01, n_obs=20
        )
        v = post.v(r_tau=0.05, rho=0.000137)
        assert v > 0

    def test_v_negative_mu(self):
        """Low mu and low current return should give negative V."""
        post = OUPosterior(
            mu_mean=-0.05, mu_var=0.01, theta=0.05, sigma2_ou=0.01, n_obs=20
        )
        v = post.v(r_tau=-0.02, rho=0.000137)
        assert v < 0


class TestReturnProcessTracker:
    def _make_curve(self, mu=0.10, theta=0.05, sigma=0.20, n=60, seed=42):
        rng = np.random.default_rng(seed)
        a_exact = np.exp(-theta)
        sigma_ou = sigma * np.sqrt((1 - a_exact**2) / (2 * theta))
        r = np.zeros(n)
        r[0] = 0.0
        for i in range(1, n):
            r[i] = mu + a_exact * (r[i - 1] - mu) + sigma_ou * rng.normal()
        return r

    def test_converges_to_true_mu(self):
        """KF posterior mu should converge to the realization's sample mean."""
        rng = np.random.default_rng(42)
        mu_true, theta_true, sigma_true = 0.10, 0.05, 0.20
        n = 2000
        a_exact = np.exp(-theta_true)
        sigma_ou = sigma_true * np.sqrt((1 - a_exact**2) / (2 * theta_true))
        r = np.zeros(n)
        r[0] = mu_true
        for i in range(1, n):
            r[i] = mu_true + a_exact * (r[i - 1] - mu_true) + sigma_ou * rng.normal()

        ou = fit_ou(r)
        tracker = ReturnProcessTracker(
            theta=ou.theta,
            beta_prior=mu_true * (1 - np.exp(-ou.theta)),
            P_prior=0.1,
            sigma2_ou=ou.sigma2_ou,
            min_observations=10,
        )
        for r_t in r:
            tracker.update(r_t)

        post = tracker.get_posterior()
        # With Q=0, KF averages all z-innovations → mu → sample mean of r
        mu_sample = float(np.mean(r))
        assert abs(post.mu_mean - mu_sample) < 0.05, (
            f"mu={post.mu_mean:.3f} != sample mean={mu_sample:.3f}"
        )
